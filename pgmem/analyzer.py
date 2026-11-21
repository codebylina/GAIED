"""
Independent analysis agent: reads Flowcept provenance and the memory graph, detects problems, writes its findings
(M2 diagnoses, M4 procedures, verdicts, edges) and sends guidance to the scientific agents. It reads and writes the
graph directly, rule by rule; it never rewrites an entry.

Separate process, never on the agents' critical path:
    python -m pgmem.analyzer                      online: memory events + Flowcept provenance, until stopped
    python -m pgmem.analyzer --post [--out DIR]   post-execution report of the campaign and of the memory

Rules, by event (deterministic first; the LLM only to interpret: put a statement in a checkable form, propose an
explanation, draft a procedure, advise a change of direction):
  result     redundant computation; data quality, and the impact analysis of an unreliable result (what rests on
             it, through IMPACT edges, up to PGMEM_ANALYSIS_DEPTH); verdicts of the statements in its scope
  failure    repeated failure of a molecule (cause confirmed by RDKit when it cannot read the SMILES); recurring
             failures of one class on molecules sharing a motif (graph: failure -> molecule -> motif -> other
             molecules -> their failures), with a candidate explanation, a procedure for that class and motif, and the
             procedure applied when it already exists
  knowledge  a statement resting on an unreliable result; its verdict, and what rests on it when it is contradicted
  proposals  proposals of molecules the team already measured or saw fail
  attempt    whether the guidance an agent received was applied and effective (APPLIED_IN edges); a procedure that
             worked before and no longer does; two iterations without improvement
  provenance redundant computations, LLM calls without tool calls, tasks in error (Flowcept records)

Processing: a memory event is acknowledged once processed (Redis consumer group) and recorded as done; an event
whose processing fails (e.g. LLM unavailable) stays pending and is retried. A Flowcept record is processed once per
task id; the records received online are counted, so a loss on Flowcept's channel is visible in the evaluation.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import time
from collections import Counter

import redis

from . import chem, verification
from .config import Config
from .domain import DOMAIN
from .llm import DRAFT_PROCEDURE, EXPLAIN_FAILURES, SUGGEST_DIRECTION, AnalysisLLM
from .provenance import ProvenanceIndex, as_dict, is_computation, make_source, smiles_of
from .retrieval import describe, procedure_history, unreliable
from .schema import Entry, Guidance, diagnosis, molecule_node, motif_node, procedure
from .store import MemoryStore

log = logging.getLogger("pgmem.analyzer")
ME = "analysis-agent"


def _as_text(v) -> str:
    """LLM answers sometimes give a list or an object where the schema expects text."""
    if isinstance(v, (list, tuple)):
        return "; ".join(_as_text(x).strip().rstrip(".") for x in v if x not in (None, ""))
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_as_text(x)}" for k, x in v.items())
    return "" if v is None else str(v)


class Analyzer:
    def __init__(self, store: MemoryStore, cfg: Config, llm: AnalysisLLM | None = None, source=None,
                 consumer: str | None = None) -> None:
        self.s, self.g, self.cfg = store, store.graph, cfg
        self.llm = llm if (llm is not None or not cfg.analyzer_llm) else AnalysisLLM()
        self.source = source
        self.index: ProvenanceIndex = source.index if source is not None else ProvenanceIndex()
        self.stream, self.group = store._k("events"), cfg.analyzer_group
        self.consumer = consumer or f"{socket.gethostname()}-{os.getpid()}"
        self.stats_key = store._k("analyzer", "stats")
        self.count_provenance = True           # records received during the run (provenance coverage)
        if source is not None:                 # 0 until records arrive: "none received" differs from "not measured"
            store.r.hsetnx(self.stats_key, "provenance_records", 0)
        try:
            store.r.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except redis.exceptions.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    # ================================================================== main loop
    def step(self, block_ms: int = 1000, retry_idle_ms: int = 30000) -> int:
        """Process what is available: pending (failed) events older than retry_idle_ms, new memory events, new
        provenance records. Returns the number of items processed."""
        n = 0
        claimed = self.s.r.xautoclaim(self.stream, self.group, self.consumer, min_idle_time=retry_idle_ms,
                                      start_id="0-0", count=50)
        msgs = list(claimed[1]) if claimed else []
        for _, batch in self.s.r.xreadgroup(self.group, self.consumer, {self.stream: ">"}, count=50,
                                            block=block_ms) or []:
            msgs.extend(batch)
        for msg_id, fields in msgs:
            n += self._handle_event(msg_id, fields)
        if self.source is not None:
            for rec in self.source.poll():
                self._on_provenance(rec)
                if self.count_provenance:
                    self.s.r.hincrby(self.stats_key, "provenance_records", 1)
                n += 1
        return n

    def _handle_event(self, msg_id: str, fields: dict) -> bool:
        done = self.s._k("analyzer", "done")
        if self.s.r.sismember(done, msg_id):
            self.s.r.xack(self.stream, self.group, msg_id)
            return False
        t0 = time.time()
        try:
            self._on_event(fields.get("event", ""), json.loads(fields.get("payload") or "{}"))
        except Exception as exc:  # noqa: BLE001 - kept pending, retried later: no analysis work is lost
            log.warning("event %s left pending (%s: %s)", msg_id, type(exc).__name__, exc)
            return False
        finally:   # overhead of the analysis agent, kept for the evaluation
            self.s.r.hincrbyfloat(self.stats_key, "event_seconds", time.time() - t0)
        self.s.r.sadd(done, msg_id)
        self.s.r.xack(self.stream, self.group, msg_id)
        return True

    def run_forever(self, idle_s: float = 0.5) -> None:
        log.info("analysis agent started (namespace %s, consumer %s)", self.s.ns, self.consumer)
        while True:
            if not self.step():
                time.sleep(idle_s)

    def _on_event(self, event: str, p: dict) -> None:
        """Rules on memory events, and the judgement of guidance (kept in every configuration: it is how the
        interventions are measured). Without the "memory" source (ablation) only the judgement runs."""
        rules = "memory" in self.cfg.analysis_sources
        handlers = {"result": [self._on_result], "failure": [self._on_failure], "knowledge": [self._on_knowledge],
                    "proposals": [self._on_proposals], "attempt": [self._on_attempt]}
        if event == "attempt":
            self._judge_guidance(p)
        for handler in handlers.get(event, []) if rules else []:
            handler(p)

    # ================================================================== findings, guidance, investigation
    def _fired(self, rule: str, key: str) -> str | None:
        return self.s.r.hget(self.s._k("analyzer", "fired"), f"{rule}|{key}")

    def _diagnose(self, rule: str, key: str, text: str, problem_type: str, status: str, based_on: list[str],
                  prov_refs: list[dict] | None = None, context: dict | None = None,
                  about: str | None = None) -> tuple[str, bool]:
        """The diagnosis of a problem, linked to what it is based on (DIAGNOSED_FROM). One diagnosis per rule and
        key: a later occurrence adds the new evidence to it. Returns (id, new)."""
        did = self._fired(rule, key)
        new = did is None
        if new:
            did, _ = self.s.add(diagnosis(rule, key, text, problem_type, status, ME, context, prov_refs,
                                          self.cfg.campaign_id))
            self.s.r.hset(self.s._k("analyzer", "fired"), f"{rule}|{key}", did)
            if about is None and DOMAIN.valid(key) and self.s.exists(molecule_node(key)):
                about = molecule_node(key)       # a diagnosis keyed by a molecule is about it
            if about:
                self.s.link(did, about, "ABOUT", ME)
        for node in dict.fromkeys(based_on):
            if self.s.exists(node):
                self.s.link(did, node, "DIAGNOSED_FROM", ME)
        return did, new

    def _guide(self, message: str, problem_type: str | None, status: str, applies: list[str], target: str = "*",
               check: dict | None = None) -> str | None:
        """Guidance to an agent (or "*"), linked to the diagnoses and procedures it applies (APPLIES)."""
        if not self.cfg.guidance:
            return None
        gid = self.s.add_guidance(Guidance(message, problem_type, status, target, check or {}))
        for node in applies:
            if node.startswith(("m2:", "m4:")):
                self.s.link(gid, node, "APPLIES", ME)
        return gid

    def _llm_json(self, system: str, user: str) -> dict:
        if self.llm is None:
            return {}
        self.s.r.hincrby(self.stats_key, "llm_calls", 1)
        return self.llm.ask_json(system, user)       # an exception leaves the event pending

    def investigate(self, node: str, depth: int = 2, limit: int = 20) -> str:
        """A bounded subgraph around a node, as text for the LLM: each node reached, the relation and direction of
        the last edge of its path, and its description."""
        lines = [f"{node}: {self._label(node)}"]
        for r in self.g.traverse(node, "both", None, depth, limit):
            e = r.path[-1]
            arrow = f"--{e['type']}-->" if e["target"] == r.node else f"<--{e['type']}--"
            lines.append(f"{'  ' * r.depth}{arrow} {r.node}: {self._label(r.node)}"
                         + (" (proposed by an LLM)" if e["origin"] == "llm" else ""))
        return "\n".join(lines)

    def _label(self, node: str) -> str:
        if node.startswith(("mol:", "motif:")):
            return node.split(":", 1)[1]
        if node.startswith("g:"):
            g = self.s.get_guidance(node)
            return f"guidance: {g.message[:160]}" if g else node
        e = self.s.get(node)
        return describe(self.s, e)[:240] if e else node

    def _impact(self, node: str) -> list[dict]:
        """What rests on a node (the impact analysis): each dependent entry or guidance, how far, through which
        relations."""
        return [{"node": r.node, "depth": r.depth, "via": [e["type"] for e in r.path]}
                for r in self.g.dependents(node, self.cfg.analysis_depth)]

    def _impact_text(self, impact: list[dict]) -> str:
        items = [d for d in impact if d["node"].startswith(("m3:", "m4:"))][:4]
        return "; ".join(f"{'directly' if d['depth'] == 1 else 'indirectly'}: {self._label(d['node'])[:140]}"
                         for d in items)

    # ================================================================== results
    def _on_result(self, p: dict) -> None:
        m1 = self.s.get(p["entry"])
        if m1 is None:
            return
        key = m1.key
        # redundant computations: the same molecule computed more than once (memory and provenance); one computation
        # appears in Flowcept as a tool_call and a parsl_task: count the parsl_task records, as the memory does
        computed = {r["task_id"] for e in self.s.by_key("M1", key) for r in e.prov_refs if r.get("task_id")}
        camp = {r.get("campaign_id") for r in m1.prov_refs}
        computed |= {r["task_id"] for r in self.index.computations() if r.get("subtype") == "parsl_task"
                     and r.get("campaign_id") in camp and chem.canonical(smiles_of(r) or "") == key}
        if len(computed) > 1:
            results = [e.id for e in self.s.by_key("M1", key)]
            did, new = self._diagnose("redundant_computation", key, f"{key} was computed {len(computed)} times",
                                      "workflow", "observation", results, [{"task_id": t} for t in sorted(computed)])
            if new:
                self._guide(f"{key} has already been computed ({m1.text}); do not compute it again.", "workflow",
                            "observation", [did], check={"avoid_keys": [key]})
        issues = chem.data_quality_issues(m1.data.get("value"), m1.data.get("conditions", {}))
        if issues:
            self._data_quality(m1, issues)
        if self.cfg.structured:
            self._reverify(m1)

    def _data_quality(self, m1: Entry, issues: list[str]) -> None:
        impact = self._impact(m1.id)
        downstream = [{"task_id": d["task_id"], "activity_id": d.get("activity_id")}
                      for r in m1.prov_refs if r.get("task_id") for d in self.index.downstream(r["task_id"])]
        rests = [d for d in impact if d["node"].startswith(("m3:", "m4:"))]
        text = f"Result {m1.text} is unreliable: {'; '.join(issues)}"
        if rests:
            text += f"; {len(rests)} statement(s) or procedure(s) rest on it"
        did, new = self._diagnose("invalid_artifact", m1.id, text, "data_quality", "observation", [m1.id],
                                  m1.prov_refs, {"subject": m1.id, "issues": issues, "affected": impact,
                                                 "downstream_tasks": downstream},
                                  molecule_node(m1.key) if self.s.exists(molecule_node(m1.key)) else None)
        if new:
            msg = f"The result {m1.text} is unreliable ({'; '.join(issues)}). Do not build conclusions on it"
            if rests:
                msg += "; reassess what rests on it: " + self._impact_text(impact)
            self._guide(msg + ".", "data_quality", "observation", [did])

    def _reverify(self, m1: Entry) -> None:
        """A new result: the verdicts of the statements in its scope may change."""
        good, bad = verification.results(self.s)
        for kid, rec in self.s.all_verifications().items():
            if not kid.startswith("m3:") or not verification.in_scope(rec.get("claim", {}), m1.key):
                continue
            res = verification.check(rec["claim"], good, bad)
            prev = verification.record(self.s, kid, rec["claim"], res, f"new result {m1.id}")
            k = self.s.get(kid)
            if res["verdict"] == "contradicted" and prev != "contradicted" and k is not None:
                self._report_contradiction(k, res, "claim_contradicted_by_new_result", f"{kid}|{m1.id}", "*",
                                           f"The new result {m1.text} contradicts the statement \"{k.text}\"",
                                           [m1.id])

    # ================================================================== failures
    def _on_failure(self, p: dict) -> None:
        f = self.s.get(p["entry"])
        if f is None:
            return
        key, fclass, smiles = f.key, f.data["failure_class"], f.data.get("spelling") or f.key
        same = [e for e in self.s.by_key("M2", key) if e.kind == "failure"]
        if len(same) > 1:                        # the same molecule failed again
            confirmed = chem.confirm_cause(fclass, smiles)
            did, new = self._diagnose("repeated_failure", key, f"{key} failed {len(same)} times ({fclass})"
                                      + ("; RDKit cannot read this SMILES" if confirmed else ""), "technical_failure",
                                      "explanation" if confirmed else "observation", [e.id for e in same],
                                      [r for e in same for r in e.prov_refs])
            if confirmed and new:                # a cause confirmed by a deterministic check
                verification.record(self.s, did, {"form": "rdkit_parse", "smiles": smiles},
                                    {"verdict": "supported", "reason": "RDKit cannot read the SMILES",
                                     "checkable": True, "evidence": [e.id for e in same]}, "RDKit check")
            if new:
                self._guide(f"{key} has failed {len(same)} times ({fclass})."
                            + (" RDKit cannot read this SMILES (confirmed)." if confirmed else "")
                            + " Do not propose it again.", "technical_failure",
                            "explanation" if confirmed else "observation", [did], check={"avoid_keys": [key]})
        motif, failures = self._pattern(f)
        if failures:
            self._recurring(fclass, motif, failures)
        proc = next(iter(self.s.by_key("M4", f"{fclass}|{motif or '*'}")), None)
        if proc is not None and proc.created_at < f.created_at and self.s.r.sadd(
                self.s._k("analyzer", "applied"), f"{proc.id}|{f.author}"):
            self._guide(f"Procedure for {fclass} failures" + (f" of molecules with a {motif}" if motif else "")
                        + f": {proc.data['actions']} (expected: {proc.data['expected_outcome']}; limits: "
                        f"{proc.data['limitations']})", "technical_failure", "observation", [proc.id], f.author,
                        {"failure_class": fclass, "motif": motif})

    def _pattern(self, f: Entry) -> tuple[str | None, list[Entry]]:
        """Recurring failures of f's class: on molecules sharing one of f's motifs (through the graph), or else on
        any molecules. The motif chosen is the one that most often fails among the molecules computed with it."""
        fclass = f.data["failure_class"]

        def failing(mol_node: str) -> list[Entry]:
            out = [self.s.get(e["source"]) for e in self.g.incoming(mol_node, ["ABOUT"])]
            return [e for e in out if e is not None and e.kind == "failure" and e.data["failure_class"] == fclass]
        best = None
        for m in sorted(DOMAIN.motifs(f.key) if DOMAIN.valid(f.key) else ()):
            mols = [e["source"] for e in self.g.incoming(motif_node(m), ["HAS_MOTIF"])]
            fails = [x for mol in mols for x in failing(mol)]
            n_fail = len({x.key for x in fails})
            n_ok = sum(1 for mol in mols if any(r.data.get("value") is not None and unreliable(self.s, r) is None
                                                for r in self.s.by_key("M1", mol[4:])))
            score = (n_fail / (n_fail + n_ok), n_fail)
            if n_fail >= 2 and (best is None or score > best[0]):
                best = (score, m, fails)
        if best:
            return best[1], best[2]
        fails = [e for e in self.s.all("M2") if e.kind == "failure" and e.data["failure_class"] == fclass]
        return None, fails if len({e.key for e in fails}) >= 2 else []

    def _recurring(self, fclass: str, motif: str | None, failures: list[Entry]) -> None:
        key, mols = f"{fclass}|{motif or '*'}", sorted({e.key for e in failures})
        what = f"molecules with a {motif.replace('_', ' ')} group" if motif else "different molecules"
        about = self.s.motif(motif) if motif else None
        if self._fired("recurring_failure", key):
            did, _ = self._diagnose("recurring_failure", key, "", "technical_failure", "observation",
                                    [e.id for e in failures])
            if self.cfg.structured and not self.s.by_key("M4", key):   # the earlier draft was unusable: again
                self._store_procedure(key, fclass, motif, failures, self._draft_procedure(fclass, motif, failures))
            return
        # LLM steps first: if the LLM is unavailable nothing is written and the event is retried later
        context = self.investigate(about, 2, 20) if about else ""
        expl = self._llm_json(EXPLAIN_FAILURES, f"Error class: {fclass}\nMotif: {motif or 'none'}\n" + "\n".join(
            f"{e.key}: {e.data['error'][:200]}" for e in failures[:10]) + (f"\nGraph context:\n{context}" if context
                                                                            else ""))
        draft = self._draft_procedure(fclass, motif, failures)
        did, _ = self._diagnose("recurring_failure", key, f"{len(mols)} {what} failed with {fclass}: "
                                + ", ".join(mols[:8]), "technical_failure", "observation", [e.id for e in failures],
                                [r for e in failures for r in e.prov_refs][:20], {"failure_class": fclass,
                                                                                  "motif": motif}, about)
        if expl.get("explanation"):
            eid, _ = self._diagnose("failure_explanation", key, f"Proposed explanation of the {fclass} failures of "
                                    f"{what}: {_as_text(expl['explanation'])}", "technical_failure", "explanation",
                                    [did], about=about)
            for e in failures:
                self.s.link(eid, e.id, "EXPLAINS", ME, "llm")
        self._guide(f"Computations of {what} fail repeatedly with {fclass} ({', '.join(mols[:5])}). "
                    + (_as_text(expl.get("advice")) or "Avoid similar molecules."), "technical_failure",
                    "hypothesis" if expl else "observation", [did],
                    check={"failure_class": fclass, "motif": motif, "avoid_keys": mols})
        self._store_procedure(key, fclass, motif, failures, draft)

    def _draft_procedure(self, fclass: str, motif: str | None, failures: list[Entry]) -> dict:
        """LLM draft of a procedure, given the failures and the team's knowledge about the motif (from the graph)."""
        if not self.cfg.structured:
            return {}
        know = [self.s.get(e["source"]) for e in self.g.incoming(motif_node(motif), ["ABOUT"])] if motif else []
        know = [k for k in know if k is not None and k.category == "M3"][:10]
        return self._llm_json(DRAFT_PROCEDURE, f"Failure class: {fclass}\nMotif: {motif or 'none'}\nFailures:\n"
                              + "\n".join(f"- [{e.id}] {e.text}: {e.data['error'][:200]}" for e in failures[:10])
                              + "\nKnowledge about the motif:\n"
                              + ("\n".join(f"- [{k.id}] {describe(self.s, k)}" for k in know) or "- none"))

    def _store_procedure(self, key: str, fclass: str, motif: str | None, failures: list[Entry], d: dict) -> None:
        if not d.get("actions") or self.s.by_key("M4", key):
            return
        f = {k: _as_text(d.get(k, "")) for k in ("text", "trigger", "preconditions", "actions", "expected_outcome",
                                                  "limitations")}
        pid, _ = self.s.add(procedure(f["text"] or f"Procedure for {key}", f["trigger"], f["preconditions"],
                                      f["actions"], f["expected_outcome"], f["limitations"], ME, key, fclass, motif,
                                      [r for e in failures for r in e.prov_refs][:20], self.cfg.campaign_id))
        if motif:
            self.s.link(pid, self.s.motif(motif), "ABOUT", ME)
        for e in failures:
            self.s.link(pid, e.id, "DERIVED_FROM", ME)
        justified = d.get("justified_by") if isinstance(d.get("justified_by"), list) else []
        for kid in justified:                    # proposed by the LLM, among the statements it was shown
            if isinstance(kid, str) and kid.startswith("m3:") and self.s.exists(kid):
                self.s.link(pid, kid, "JUSTIFIED_BY", ME, "llm")

    # ================================================================== knowledge
    def _on_knowledge(self, p: dict) -> None:
        good, bad = verification.results(self.s)
        measured = ", ".join(sorted({e.key for e in self.s.all("M1")}))[:3000]
        for kid in p.get("entries", []):
            k = self.s.get(kid)
            if k is None:
                continue
            for edge in self.g.outgoing(kid, ["SUPPORTED_BY"]):     # a statement resting on an unreliable result
                m1 = self.s.get(edge["target"])
                why = unreliable(self.s, m1) if m1 is not None and m1.category == "M1" else None
                if why:
                    did, new = self._diagnose("knowledge_on_unreliable_result", f"{kid}|{m1.id}",
                                              f"The statement \"{k.text}\" rests on an unreliable result ({m1.text}: "
                                              f"{why})", "data_quality", "observation", [kid, m1.id], m1.prov_refs)
                    if new:
                        self._guide(f"Your statement \"{k.text}\" rests on an unreliable result ({m1.text}); "
                                    "reassess it.", "data_quality", "observation", [did], k.author)
            if self.cfg.structured and not self.s.get_verification(kid):
                claim = verification.normalize(self._llm_json(verification.extraction_prompt(),
                                                              f"Statement: {k.text}\nMeasured molecules: {measured}"))
                res = verification.check(claim, good, bad)
                verification.record(self.s, kid, claim, res, "new statement")
                if res["verdict"] == "contradicted":
                    self._report_contradiction(k, res, "claim_contradicted", kid, k.author,
                                               f"Your statement \"{k.text}\" is contradicted by the team's results")

    def _report_contradiction(self, k: Entry, res: dict, rule: str, key: str, target: str, lead: str,
                              extra: list[str] | None = None) -> None:
        impact = self._impact(k.id)
        did, new = self._diagnose(rule, key, f"{lead}: {res['reason']}", "scientific_contradiction", "observation",
                                  [k.id] + (extra or []) + res["evidence"], context={"subject": k.id,
                                                                                    "affected": impact})
        if not new:
            return
        self._guide(f"{lead} ({res['reason']}). Treat it as not established; revise or test it.",
                    "scientific_contradiction", "observation", [did], target)
        procs = [d["node"] for d in impact if d["node"].startswith("m4:")]
        if procs:
            self._guide(f"The statement \"{k.text}\" is contradicted, and procedures rest on it: "
                        f"{self._impact_text(impact)}. Apply them with caution.", "scientific_contradiction",
                        "observation", [did] + procs)

    # ================================================================== proposals and attempts
    def _on_proposals(self, p: dict) -> None:
        """Proposals of what the team already measured or saw fail (as known when the proposals were made)."""
        agent, known, failed = p.get("agent", ""), list(p.get("known", [])), list(p.get("failed", []))
        if not (known or failed):
            return
        based = [e.id for k in known + failed for e in self.s.by_key("M1", k) + self.s.by_key("M2", k)
                 if e.category == "M1" or e.kind == "failure"][:20]
        did, new = self._diagnose("repeated_proposals", f"{agent}|{p.get('iteration')}",
                                  f"{agent} proposed {len(known)} already measured and {len(failed)} already failed "
                                  f"molecule(s): {', '.join((known + failed)[:8])}", "workflow", "observation", based)
        if new:
            self._guide(f"You proposed molecules the team has already measured ({', '.join(known[:5]) or '-'}) "
                        f"or seen fail ({', '.join(failed[:5]) or '-'}). Check the team's results before proposing.",
                        "workflow", "observation", [did], agent, {"avoid_keys": known + failed})

    def _judge_guidance(self, p: dict) -> None:
        """Was the guidance delivered to this agent applied in its next proposals (none of the molecules or motif it
        warns against), and effective in their outcome (no recomputation of those molecules, no new failure of the
        class it is about)? Judged once per guidance and agent, on the first attempt whose proposals came after the
        delivery; recorded as an APPLIED_IN edge."""
        a = self.s.get(p.get("entry", ""))
        if a is None or not a.data.get("proposed") or a.data.get("proposed_at") is None:
            return                               # no proposals in this iteration: nothing to judge yet
        agent, keys = a.author, set(a.data["proposed"])
        part = [self.s.get(e["source"]) for e in self.g.incoming(a.id, ["PART_OF"])]
        part = [e for e in part if e is not None and e.created_at >= a.data["proposed_at"]]   # outcome of the proposals
        for g in self.s.all_guidance():
            first = min((d["t"] for d in g.delivered if d["agent"] == agent), default=None)
            if first is None or first > a.data["proposed_at"] or any(
                    e["props"].get("agent") == agent for e in self.g.outgoing(g.id, ["APPLIED_IN"])):
                continue
            avoid, fclass, motif = set(g.check.get("avoid_keys", [])), g.check.get("failure_class"), g.check.get("motif")
            if fclass == "invalid_smiles":       # the asked behaviour (valid SMILES) is itself the outcome
                applied, effective = all(DOMAIN.valid(k) for k in keys), None
            else:
                carries = motif is not None and any(motif in DOMAIN.motifs(k) for k in keys)
                applied = (not avoid & keys and not carries) if (avoid or motif) else None
                effective = None
                if applied and (avoid or fclass):
                    recomputed = [e for e in part if e.category == "M1" and e.key in avoid]
                    # the expected outcome is no new failure of the class, whatever the molecule: a procedure for
                    # one motif that no longer prevents the class (the failing motif changed) is not effective
                    refailed = [e for e in part if e.kind == "failure" and e.data["failure_class"] == fclass]
                    effective = not recomputed and not refailed
            self.s.link(g.id, a.id, "APPLIED_IN", ME, agent=agent, applied=applied, effective=effective)
            if effective is False:
                for edge in self.g.outgoing(g.id, ["APPLIES"]):
                    if edge["target"].startswith("m4:"):
                        self._procedure_outcome(edge["target"], a, g.id)

    def _procedure_outcome(self, pid: str, a: Entry, gid: str) -> None:
        """A procedure that worked before and did not this time: a diagnosis, guidance to every agent, and the
        statements that justify it named for reassessment."""
        apps = [x for e in self.g.incoming(pid, ["APPLIES"]) for x in self.g.outgoing(e["source"], ["APPLIED_IN"])]
        if not any(x["props"].get("effective") is True for x in apps):
            return
        proc = self.s.get(pid)
        did, new = self._diagnose("procedure_no_longer_effective", f"{pid}|{a.id}",
                                  f"The procedure \"{proc.text}\" worked before but not in its last application "
                                  f"({procedure_history(self.s, pid)})", "workflow", "observation", [pid, a.id, gid])
        if new:
            just = [self.s.get(e["target"]) for e in self.g.outgoing(pid, ["JUSTIFIED_BY"])]
            self._guide(f"The procedure \"{proc.text}\" did not have its expected effect in its last application "
                        "although it worked before: it may be outdated."
                        + "".join(f" The statement it rests on may need reassessment: \"{k.text}\"." for k in just
                                  if k is not None), "workflow", "observation", [did, pid])

    def _on_attempt(self, p: dict) -> None:
        """Two consecutive iterations of an agent without improving its best result: advice from the LLM."""
        agent = p.get("agent", "")
        mine = sorted((e for e in self.s.by_key("M2", agent) if e.kind == "attempt"), key=lambda e: e.created_at)
        last = mine[-2:]
        if len(last) < 2 or any(e.data.get("improved") is not False for e in last) or self._fired(
                "no_improvement", f"{agent}|{last[-1].id}"):
            return
        best = sorted((e for e in self.s.all("M1") if e.data.get("value") is not None),
                      key=lambda e: -e.data["value"])[:5]
        fails = [e.text for e in self.s.all("M2") if e.kind == "failure"][-5:]
        know = [describe(self.s, e) for e in self.s.recent("M3", self.cfg.retrieval_limit)]
        adv = self._llm_json(SUGGEST_DIRECTION, "Recent iterations:\n" + "\n".join(e.text for e in last)
                             + "\nTeam best results:\n" + "\n".join(e.text for e in best) + "\nFailures:\n"
                             + "\n".join(fails) + "\nKnowledge:\n" + "\n".join(know))
        did, new = self._diagnose("no_improvement", f"{agent}|{last[-1].id}",
                                  f"{agent}: two consecutive iterations without improvement", "workflow",
                                  "observation", [e.id for e in last])
        if new and adv.get("advice"):
            self._guide(_as_text(adv["advice"]), "workflow", "hypothesis", [did], agent)

    # ================================================================== provenance records
    def _on_provenance(self, rec: dict) -> None:
        sub = rec.get("subtype")
        if is_computation(rec) and sub == "parsl_task":
            key = chem.canonical(smiles_of(rec) or "")
            ids = sorted({r["task_id"] for r in self.index.computations() if r.get("subtype") == "parsl_task"
                          and r.get("campaign_id") == rec.get("campaign_id")
                          and chem.canonical(smiles_of(r) or "") == key})
            if len(ids) > 1:
                self._diagnose("redundant_computation", key, f"{key} was computed {len(ids)} times", "workflow",
                               "observation", [e.id for e in self.s.by_key("M1", key)], [{"task_id": t} for t in ids])
        elif sub == "llm_call":
            if as_dict(rec.get("used")).get("has_tools") and not as_dict(rec.get("generated")).get("tool_calls"):
                wf = rec.get("workflow_id", "")
                self._diagnose("llm_without_tool_calls", wf, "An LLM call asked for tool calls returned none (the "
                               "agent retries)", "workflow", "observation", [],
                               [{"task_id": rec.get("task_id"), "subtype": "llm_call", "workflow_id": wf}])
        elif str(rec.get("status", "")) == "ERROR":
            self._diagnose("task_error", str(rec.get("activity_id")), f"Tasks {rec.get('activity_id')} ended in "
                           f"ERROR (first error: {str(rec.get('stderr'))[:200]})", "technical_failure",
                           "observation", [], [{"task_id": rec.get("task_id"), "subtype": sub}])

    # ================================================================== post-execution report
    def post_execution(self) -> dict:
        """After a run: provenance summary (Flowcept's own analysis functions), findings, procedures with their
        history, guidance outcomes, strategy, coordination and the size of the memory graph."""
        if self.source is not None:
            for rec in self.source.poll():
                self._on_provenance(rec)
        records = self.index.records()
        report: dict = {"namespace": self.s.ns, "campaign_id": self.cfg.campaign_id}
        try:
            from flowcept.agents.prov_analysis import core as pa
            report["provenance_summary"] = {k: v for k, v in pa.summarize_execution(records).items()
                                            if k != "activities"}
            report["provenance_errors"] = pa.analyze_errors(records)
        except Exception as exc:  # noqa: BLE001
            report["provenance_summary_error"] = str(exc)
        # GAIED records each computation as a parsl_task; files written before that fix only have the tool_call
        comps = [r for r in records if is_computation(r) and r.get("subtype") == "parsl_task"] or \
                [r for r in records if is_computation(r)]
        per_mol = Counter(chem.canonical(smiles_of(r) or "") for r in comps)
        report["computations"] = {"total": len(comps), "distinct_molecules": len(per_mol),
                                  "redundant": sum(n - 1 for n in per_mol.values() if n > 1)}
        report["findings"] = [{"id": e.id, "rule": e.data["rule"], "text": e.text,
                               "problem_type": e.data["problem_type"], "epistemic_status": e.data["epistemic_status"],
                               "based_on": len(self.g.outgoing(e.id, ["DIAGNOSED_FROM"]))}
                              for e in self.s.all("M2") if e.kind == "diagnosis"]
        report["procedures"] = [{"id": e.id, "text": e.text, "history": procedure_history(self.s, e.id)}
                                for e in self.s.all("M4")]
        judged = [x for g in self.s.all_guidance() for x in self.g.outgoing(g.id, ["APPLIED_IN"])]
        gs = self.s.all_guidance()
        report["guidance"] = {"created": len(gs), "delivered": sum(1 for g in gs if g.delivered),
                              "judged": len(judged),
                              **{f"{k}_{v}": sum(1 for x in judged if x["props"].get(k) is b)
                                 for k in ("applied", "effective") for v, b in (("yes", True), ("no", False))}}
        attempts = [e for e in self.s.all("M2") if e.kind == "attempt"]
        report["strategy"] = {"attempts": len(attempts), "improving": sum(1 for e in attempts if e.data["improved"])}
        waits = [json.loads(f.get("payload") or "{}") for _, f in self.s.r.xrange(self.stream)
                 if f.get("event") == "waited"]
        report["coordination"] = {"waits": len(waits), **{o: sum(1 for w in waits if w.get("outcome") == o)
                                                          for o in ("result", "failure", "computes")}}
        report["memory"] = {c: self.s.count(c) for c in ("M1", "M2", "M3", "M4")}
        report["edges"] = self.g.count_by_type()
        return report


def write_report(report: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"analysis_{int(time.time())}.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    lines = [f"# Post-execution analysis ({report.get('namespace')})", "",
             f"- memory: {report.get('memory')}", f"- edges: {report.get('edges')}",
             f"- computations: {report.get('computations')}", f"- coordination: {report.get('coordination')}",
             f"- guidance: {report.get('guidance')}", f"- strategy: {report.get('strategy')}", "", "## Findings", ""]
    lines += [f"- [{f['problem_type']} / {f['epistemic_status']}] {f['text']} (based on {f['based_on']})"
              for f in report.get("findings", [])]
    lines += ["", "## Procedures", ""] + [f"- {p['text']}: {p['history']}" for p in report.get("procedures", [])]
    with open(path[:-5] + ".md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def latest_campaign(pattern: str) -> str | None:
    """Campaign of the most recently written Flowcept file."""
    import glob as _glob
    for path in sorted(_glob.glob(pattern), key=os.path.getmtime, reverse=True):
        for line in open(path, encoding="utf-8"):
            try:
                c = json.loads(line).get("campaign_id")
            except ValueError:
                continue
            if c:
                return c
    return None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Provenance and memory analysis agent")
    ap.add_argument("--post", action="store_true", help="post-execution report, then exit")
    ap.add_argument("--out", default="results/analysis", help="where the post-execution report is written")
    ap.add_argument("--jsonl", default=None, help="Flowcept JSONL files to read (glob)")
    ap.add_argument("--campaign", default=os.environ.get("PGMEM_CAMPAIGN"),
                    help="campaign to analyze after execution (default: the latest one in the JSONL files)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = Config.from_env()
    if cfg.mode == "off":
        cfg.mode = "full"                        # the analysis agent is only started when it is wanted
    store = MemoryStore.connect(cfg.redis_host, cfg.redis_port, cfg.namespace)
    campaign = args.campaign or (latest_campaign(args.jsonl or cfg.jsonl_glob) if args.post else None)
    cfg.campaign_id = campaign
    kind = "none" if "provenance" not in cfg.analysis_sources else ("jsonl" if args.post else cfg.provenance_source)
    source = make_source(kind, cfg.redis_host, cfg.redis_port, args.jsonl or cfg.jsonl_glob, campaign,
                         online=not args.post)
    an = Analyzer(store, cfg, source=source)
    if args.post:
        an.count_provenance = False              # reading the whole provenance again is not what was received online
        an.step(block_ms=10, retry_idle_ms=0)    # finish pending memory events (and read the provenance) first
        print("report:", write_report(an.post_execution(), args.out))
        return
    an.run_forever()


if __name__ == "__main__":
    main()
