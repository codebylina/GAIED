"""
Independent analysis agent: reads Flowcept provenance and the shared memory (M1-M4), detects problems, writes its
findings to memory (M2 diagnoses, notes, evidence, M4 procedures) and sends guidance to the scientific agents.

Separate process, never on the agents' critical path:
    python -m pgmem.analyzer                      online: memory events + Flowcept provenance, until stopped
    python -m pgmem.analyzer --post [--out DIR]   post-execution analysis of the campaign and of the memory

Event processing
  selection     memory events from the agents (results, failures, proposals, attempts, knowledge, deliveries) and
                the Flowcept records listed in provenance.RELEVANT_SUBTYPES; everything else is ignored
  processing    deterministic rules first; the LLM only for interpretation (candidate explanations, comparing a result
                with knowledge statements, advice, drafting procedures)
  deduplication a memory event is acknowledged only once processed (Redis consumer group) and recorded as done;
                a Flowcept record is processed once per task_id
  linking       every finding carries the memory entry ids and Flowcept task ids it is based on
  durability    an event whose processing fails (e.g. LLM unavailable) stays pending and is retried; a stopped
                analysis agent resumes from the pending events and re-reads the provenance files

Every finding states its problem type (technical_failure, data_quality, workflow, scientific_contradiction) and its
basis (observed_fact, candidate_explanation, confirmed_cause). The analysis agent never establishes scientific truth
and never overwrites an entry: it adds diagnoses, notes and evidence, and proposes.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import time
from collections import Counter, defaultdict

import redis

from . import chem
from .config import Config
from .llm import DRAFT_PROCEDURE, EXPLAIN_FAILURES, JUDGE_RESULT, SUGGEST_DIRECTION, AnalysisLLM
from .provenance import ProvenanceIndex, as_dict, as_float, is_computation, make_source, smiles_of
from .schema import Entry, Guidance, experience, procedure, validation_status
from .store import MemoryStore

log = logging.getLogger("pgmem.analyzer")
ME = "analysis-agent"


class Analyzer:
    def __init__(self, store: MemoryStore, cfg: Config, llm: AnalysisLLM | None = None, source=None,
                 consumer: str | None = None) -> None:
        self.s, self.cfg = store, cfg
        self.llm = llm if (llm is not None or not cfg.analyzer_llm) else AnalysisLLM()
        self.source = source
        self.index: ProvenanceIndex = source.index if source is not None else ProvenanceIndex()
        self.stream = store._k("events")
        self.group = cfg.analyzer_group
        self.consumer = consumer or f"{socket.gethostname()}-{os.getpid()}"
        self.stats = Counter()
        try:
            self.s.r.xgroup_create(self.stream, self.group, id="0", mkstream=True)
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
        new = self.s.r.xreadgroup(self.group, self.consumer, {self.stream: ">"}, count=50, block=block_ms)
        for _, batch in new or []:
            msgs.extend(batch)
        for msg_id, fields in msgs:
            if self._handle_event(msg_id, fields):
                n += 1
        if self.source is not None:
            for rec in self.source.poll():
                self._on_provenance(rec)
                n += 1
        return n

    def _handle_event(self, msg_id: str, fields: dict) -> bool:
        done = self.s._k("analyzer", "done")
        if self.s.r.sismember(done, msg_id):
            self.s.r.xack(self.stream, self.group, msg_id)
            return False
        try:
            self._on_event(fields.get("event", ""), json.loads(fields.get("payload") or "{}"))
        except Exception as exc:  # noqa: BLE001 - kept pending, retried later: no analysis work is lost
            self.stats["failed_attempts"] += 1
            log.warning("event %s left pending (%s: %s)", msg_id, type(exc).__name__, exc)
            return False
        self.s.r.sadd(done, msg_id)
        self.s.r.xack(self.stream, self.group, msg_id)
        self.stats["events"] += 1
        return True

    def run_forever(self, idle_s: float = 0.5) -> None:
        log.info("analysis agent started (namespace %s, consumer %s)", self.s.ns, self.consumer)
        while True:
            if not self.step():
                time.sleep(idle_s)

    # ================================================================== dispatch
    def _on_event(self, event: str, p: dict) -> None:
        handler = {"result": self._on_result, "failure": self._on_failure, "proposals": self._on_proposals,
                   "attempt": self._on_attempt, "knowledge": self._on_knowledge}.get(event)
        if handler:
            handler(p)

    # ================================================================== helpers
    def _fired(self, rule: str, key: str) -> str | None:
        """The M2 diagnosis already written for this rule and key, if any (one diagnosis per problem; later
        occurrences are added to it)."""
        return self.s.r.hget(self.s._k("analyzer", "fired"), f"{rule}|{key}")

    def _diagnose(self, rule: str, key: str, text: str, problem_type: str, basis: str, refs: list[str],
                  prov_refs: list[dict], context: dict | None = None) -> tuple[str, bool]:
        """Write (or extend) the M2 diagnosis of a problem. Returns (id, new)."""
        prev = self._fired(rule, key)
        if prev:
            old = self.s.get(prev)
            new_refs = [r for r in refs if r not in old.refs]
            new_prov = [p for p in prov_refs if p not in old.prov_refs]
            if new_refs or new_prov:            # a new occurrence only when new evidence is involved

                def fn(e: Entry) -> None:
                    e.refs.extend(new_refs)
                    e.prov_refs.extend(new_prov)
                    e.data["outcome"]["occurrences"] = e.data["outcome"].get("occurrences", 1) + 1
                self.s.update(prev, ME, "new occurrence", fn)
            return prev, False
        e = experience("diagnosis", text, ME, key=key, context={"rule": rule, **(context or {})},
                       outcome={"occurrences": 1}, prov_refs=prov_refs, refs=refs, problem_type=problem_type,
                       basis=basis, campaign_id=self.cfg.campaign_id)
        eid, _ = self.s.add(e, dedup=False)
        self.s.r.hset(self.s._k("analyzer", "fired"), f"{rule}|{key}", eid)
        self.stats[f"finding:{rule}"] += 1
        return eid, True

    def _guide(self, message: str, problem_type: str | None, basis: str, evidence: list[str], prov_refs: list[dict],
               target: str = "*", check: dict | None = None, procedure_id: str | None = None) -> str | None:
        if not self.cfg.guidance:
            return None
        g = Guidance(message=message, problem_type=problem_type, basis=basis, target=target, evidence=evidence,
                     prov_refs=prov_refs, check=check or {}, procedure_id=procedure_id)
        self.stats["guidance"] += 1
        return self.s.add_guidance(g)

    def _llm_json(self, system: str, user: str) -> dict:
        if self.llm is None:
            return {}
        return self.llm.ask_json(system, user)       # an exception leaves the event pending

    # ================================================================== rules on memory events
    def _on_result(self, p: dict) -> None:
        m1 = self.s.get(p["entry"])
        if m1 is None:
            return
        key = m1.key
        # 1. redundant computations: the same molecule computed more than once (memory and provenance)
        computed = [r for e in self.s.by_key("M1", key) for r in e.prov_refs if r.get("task_id")]
        camp = {r.get("campaign_id") for r in m1.prov_refs}
        # one computation appears in Flowcept as a tool_call and as a parsl_task: count the parsl_task records,
        # whose task ids are the ones memory entries reference
        prov_runs = [r for r in self.index.computations() if chem.canonical(smiles_of(r) or "") == key
                     and r.get("campaign_id") in camp and r.get("subtype") == "parsl_task"]
        task_ids = {r["task_id"] for r in computed} | {r["task_id"] for r in prov_runs}
        if len(task_ids) > 1:
            refs = [e.id for e in self.s.by_key("M1", key)]
            _, new = self._diagnose("redundant_computation", key,
                                    f"{key} was computed {len(task_ids)} times", "workflow", "observed_fact",
                                    refs, [{"task_id": t} for t in sorted(task_ids)])
            if new:
                self._guide(f"{key} has already been computed ({m1.text}); do not compute it again.", "workflow",
                            "observed_fact", refs, m1.prov_refs, check={"avoid_keys": [key]})
        # 2. data quality of the result: invalid or incomplete artifact, and what depends on it
        issues = chem.data_quality_issues(m1.data.get("value"), m1.data.get("conditions", {}))
        if issues:
            self._data_quality(m1, issues)
        # 3. compare the new result with existing knowledge
        if self.cfg.structured and not issues:
            self._compare_with_knowledge(m1)

    def _data_quality(self, m1: Entry, issues: list[str]) -> None:
        affected = self.s.citing(m1.id)
        tids = [r["task_id"] for r in m1.prov_refs if r.get("task_id")]
        downstream = [d for t in tids for d in self.index.downstream(t)]
        text = f"Result {m1.text} is unreliable: {'; '.join(issues)}"
        if affected:
            text += f"; {len(affected)} knowledge/procedure entr(ies) rely on it and should be reassessed"
        did, new = self._diagnose("invalid_artifact", m1.id, text, "data_quality", "observed_fact",
                                  [m1.id] + [a.id for a in affected], m1.prov_refs,
                                  context={"issues": issues, "affected": [a.id for a in affected],
                                           "downstream_tasks": [{"task_id": d["task_id"],
                                                                 "activity_id": d.get("activity_id")}
                                                                for d in downstream]})
        if new:
            self.s.add_note(m1.id, ME, f"data-quality problem: {'; '.join(issues)}", refs=[did])
            msg = f"The result {m1.text} is unreliable ({'; '.join(issues)}). Do not build conclusions on it"
            if affected:
                msg += "; reassess: " + " | ".join(a.text for a in affected[:3])
            self._guide(msg + ".", "data_quality", "observed_fact", [did, m1.id], m1.prov_refs)

    def _compare_with_knowledge(self, m1: Entry) -> None:
        cands = {e.id: e for e in self.s.search("M3", f"{m1.key} {m1.data.get('quantity', '')}",
                                                limit=self.cfg.retrieval_limit)}
        for e in self.s.all("M3"):
            if any(self.s.get(r) and self.s.get(r).key == m1.key for r in e.refs if r.startswith("m1:")):
                cands[e.id] = e
        cands.pop(None, None)
        if not cands or self.llm is None:
            return
        user = (f"New result: {m1.text}\nmolecule: {m1.key}\nvalue: {m1.data.get('value')}\n"
                f"conditions: {json.dumps(m1.data.get('conditions', {}))}\n\nStatements:\n" +
                "\n".join(f"[{e.id}] ({e.kind}, scope: {e.data.get('scope') or 'not stated'}) {e.text}"
                          for e in cands.values()))
        verdicts = self._llm_json(JUDGE_RESULT, user).get("verdicts", [])
        for v in verdicts if isinstance(verdicts, list) else []:
            e = cands.get(v.get("id"))
            rel = v.get("relation")
            if e is None or rel not in ("supports", "contradicts"):
                continue
            side = "supporting_evidence" if rel == "supports" else "contradicting_evidence"
            if any(ev.get("entry") == m1.id for ev in e.data.get(side, [])):
                continue
            self.s.add_evidence(e.id, side, {"entry": m1.id, "prov_refs": m1.prov_refs}, ME,
                                note=f"candidate, judged by the analysis agent: {v.get('reason', '')}")
            if rel == "contradicts":
                did, new = self._diagnose("contradiction", f"{e.id}|{m1.id}",
                                          f"New result {m1.text} appears to contradict: {e.text}",
                                          "scientific_contradiction", "candidate_explanation", [e.id, m1.id],
                                          m1.prov_refs, context={"reason": v.get("reason")})
                if new:
                    self._guide(f"The new result {m1.text} appears to contradict the statement \"{e.text}\". "
                                "Take both into account; the statement may need revision.",
                                "scientific_contradiction", "candidate_explanation", [did, e.id, m1.id],
                                m1.prov_refs)

    def _on_failure(self, p: dict) -> None:
        f = self.s.get(p["entry"])
        if f is None:
            return
        key, fclass = f.key, f.data["context"].get("failure_class", "other")
        smiles = f.data["context"].get("smiles", key)
        same_mol = [e for e in self.s.by_key("M2", key) if e.kind == "failure"]
        if len(same_mol) > 1:                       # repeated: the same molecule failed again
            confirmed = chem.confirm_cause(fclass, smiles)
            did, new = self._diagnose("repeated_failure", key, f"{key} failed {len(same_mol)} times ({fclass})",
                                      "technical_failure", "confirmed_cause" if confirmed else "observed_fact",
                                      [e.id for e in same_mol], [r for e in same_mol for r in e.prov_refs])
            if new:
                why = " RDKit cannot parse this SMILES (confirmed)." if confirmed else ""
                self._guide(f"{key} has failed {len(same_mol)} times ({fclass}).{why} Do not propose it again.",
                            "technical_failure", "confirmed_cause" if confirmed else "observed_fact",
                            [did], f.prov_refs, check={"avoid_keys": [key]})
        # recurring failure class across different molecules
        same_class = [e for e in self.s.all("M2") if e.kind == "failure"
                      and e.data["context"].get("failure_class") == fclass]
        mols = sorted({e.key for e in same_class})
        if len(mols) > 1:
            if not self._fired("recurring_failure_class", fclass):
                # LLM steps first: if the LLM is unavailable nothing is written and the event is retried later
                expl = self._llm_json(EXPLAIN_FAILURES, "Error class: " + fclass + "\n" + "\n".join(
                    f"{e.key}: {e.data['outcome'].get('error', '')[:200]}" for e in same_class[:10]))
                draft = self._draft_procedure(fclass, same_class)
                did, _ = self._diagnose("recurring_failure_class", fclass,
                                        f"{len(mols)} different molecules failed with {fclass}: {', '.join(mols[:8])}",
                                        "technical_failure", "observed_fact", [e.id for e in same_class],
                                        [r for e in same_class for r in e.prov_refs])
                if expl.get("explanation"):
                    self._diagnose("failure_explanation", fclass, f"Candidate explanation for {fclass} failures: "
                                   f"{expl['explanation']}", "technical_failure", "candidate_explanation", [did], [])
                self._guide(f"Computations fail repeatedly with {fclass} ({', '.join(mols[:5])}). "
                            + (expl.get("advice") or "Avoid similar molecules."), "technical_failure",
                            "candidate_explanation" if expl else "observed_fact", [did],
                            [r for e in same_class for r in e.prov_refs][:10],
                            check={"failure_class": fclass, "avoid_keys": mols})
                self._store_procedure(fclass, same_class, draft)
            else:
                did = self._fired("recurring_failure_class", fclass)
                self.s.update(did, ME, "new occurrence",
                              lambda e: e.refs.append(f.id) if f.id not in e.refs else None)
        # a procedure already exists for this situation: apply it as guidance to this agent
        for proc in self.s.by_key("M4", fclass):
            if self.s.r.sadd(self.s._k("analyzer", "applied"), f"{proc.id}|{f.author}|{fclass}"):
                self._guide(f"Procedure for {fclass}: {proc.data['actions']} (expected: "
                            f"{proc.data['expected_outcome']}; limits: {proc.data['limitations']})",
                            "technical_failure", "observed_fact", [proc.id, f.id], f.prov_refs, target=f.author,
                            check={"failure_class": fclass}, procedure_id=proc.id)

    def _draft_procedure(self, key: str, episodes: list[Entry]) -> dict:
        """LLM draft of a procedure from experiences showing the same pattern (empty if one exists already)."""
        if not self.cfg.structured or self.s.by_key("M4", key):
            return {}
        return self._llm_json(DRAFT_PROCEDURE, "\n".join(
            f"- [{e.id}] {e.text}: {json.dumps(e.data['outcome'])[:200]}" for e in episodes[:10]))

    def _store_procedure(self, key: str, episodes: list[Entry], d: dict) -> str | None:
        """M4 procedure linked to the experiences it was derived from."""
        if not d.get("actions") or self.s.by_key("M4", key):
            return None
        kind = d.get("kind") if d.get("kind") in ("procedure", "strategy", "recommendation") else "procedure"
        m4 = procedure(kind, d.get("text", ""), d.get("trigger", ""), d.get("preconditions", ""), d["actions"],
                       d.get("expected_outcome", ""), d.get("limitations", ""), [e.id for e in episodes], ME,
                       key=key, prov_refs=[r for e in episodes for r in e.prov_refs][:20],
                       campaign_id=self.cfg.campaign_id)
        pid, _ = self.s.add(m4)
        self.stats["procedures"] += 1
        return pid

    def _propose_procedure(self, key: str, episodes: list[Entry]) -> str | None:
        return self._store_procedure(key, episodes, self._draft_procedure(key, episodes))

    def _on_proposals(self, p: dict) -> None:
        agent, keys = p.get("agent", ""), list(p.get("keys", []))
        # inefficient exploration: proposing what the team already measured or saw fail
        known = [k for k in keys if self.s.by_key("M1", k)]
        failed = [k for k in keys if any(e.kind == "failure" for e in self.s.by_key("M2", k))]
        if known or failed:
            refs = [e.id for k in known + failed for e in self.s.by_key("M1", k) + self.s.by_key("M2", k)][:20]
            did, new = self._diagnose("repeated_proposals", f"{agent}|{p.get('iteration')}",
                                      f"{agent} proposed {len(known)} already measured and {len(failed)} already "
                                      f"failed molecule(s): {', '.join((known + failed)[:8])}", "workflow",
                                      "observed_fact", refs, [])
            if new:
                self._guide(f"You proposed molecules the team has already measured ({', '.join(known[:5]) or '-'}) "
                            f"or seen fail ({', '.join(failed[:5]) or '-'}). Check the team's results before "
                            "proposing.", "workflow", "observed_fact", [did], [], target=agent,
                            check={"avoid_keys": known + failed})
        # guidance tracking: was guidance delivered to this agent applied in these proposals?
        for g in self.s.all_guidance():
            if not any(d["agent"] == agent for d in g.delivered) or agent in g.applied:
                continue
            avoid = set(g.check.get("avoid_keys", []))
            applied = (not (avoid & set(keys))) if avoid else None
            self.s.update_guidance(g.id, lambda x: x.applied.__setitem__(agent, applied))

    def _on_attempt(self, p: dict) -> None:
        agent = p.get("agent", "")
        att = self.s.get(p["entry"])
        if att is None:
            return
        # guidance effectiveness, judged on what this agent did since the guidance was delivered
        for g in self.s.all_guidance():
            deliveries = [d for d in g.delivered if d["agent"] == agent]
            if not deliveries or agent in g.effective or agent not in g.applied:
                continue
            since = deliveries[0]["t"]
            mine = [e for c in ("M1", "M2") for e in self.s.all(c) if e.author == agent and e.created_at >= since]
            avoid, fclass = set(g.check.get("avoid_keys", [])), g.check.get("failure_class")
            if (not avoid and not fclass) or g.applied.get(agent) is False:
                effective = None        # nothing checkable, or not applied: the effect is not attributable
            else:
                recomputed = [e for e in mine if e.key in avoid and e.category == "M1"
                              and any(r.get("task_id") for r in e.prov_refs)]
                refailed = [e for e in mine if e.kind == "failure" and fclass
                            and e.data["context"].get("failure_class") == fclass]
                effective = not recomputed and not refailed
            self.s.update_guidance(g.id, lambda x: x.effective.__setitem__(agent, effective))
            out = experience("outcome", f"Guidance to {agent} was "
                             f"{'applied' if g.applied.get(agent) else 'not applied' if g.applied.get(agent) is False else 'not checkable'}"
                             f" and {'effective' if effective else 'not effective' if effective is False else 'of unknown effect'}: "
                             f"{g.message[:160]}", ME, key=agent, context={"guidance": g.id},
                             outcome={"applied": g.applied.get(agent), "effective": effective},
                             refs=[e for e in [g.procedure_id] if e] + g.evidence[:5], campaign_id=self.cfg.campaign_id)
            oid, _ = self.s.add(out, dedup=False)
            if g.procedure_id:
                self._record_application(g.procedure_id, agent, g.id, oid, effective)
        # repeated unsuccessful strategy: this attempt and the previous one did not improve the agent's best
        mine = sorted([e for e in self.s.by_key("M2", agent) if e.kind == "attempt"], key=lambda e: e.created_at)
        last = mine[-2:]
        if (len(last) == 2 and all(e.data["outcome"].get("improved") is False for e in last)
                and not self._fired("no_improvement", f"{agent}|{last[-1].id}")):
            best = sorted(self.s.all("M1"), key=lambda e: -(e.data.get("value") or float("-inf")))[:5]
            fails = [e.text for e in self.s.all("M2") if e.kind == "failure"][-5:]
            know = [e.text for e in self.s.recent("M3", self.cfg.retrieval_limit)]
            adv = self._llm_json(SUGGEST_DIRECTION, "Recent iterations:\n" + "\n".join(
                e.text for e in last) + "\nTeam best results:\n" + "\n".join(e.text for e in best) +
                "\nFailures:\n" + "\n".join(fails) + "\nKnowledge:\n" + "\n".join(know))
            did, new = self._diagnose("no_improvement", f"{agent}|{last[-1].id}",
                                      f"{agent}: two consecutive iterations without improvement", "workflow",
                                      "observed_fact", [e.id for e in last], [r for e in last for r in e.prov_refs])
            if new:
                if adv.get("advice"):
                    self._guide(adv["advice"], "workflow", "candidate_explanation", [did] + [e.id for e in best],
                                [], target=agent)

    def _record_application(self, pid: str, agent: str, gid: str, outcome_id: str, effective) -> None:
        proc = self.s.get(pid)
        if proc is None:
            return
        was_effective = any(a.get("effective") is True for a in proc.data.get("applications", []))
        self.s.update(pid, ME, "application recorded", lambda e: e.data["applications"].append(
            {"agent": agent, "guidance": gid, "outcome": outcome_id, "effective": effective, "t": time.time()}))
        if effective is False and was_effective:   # it worked before and did not work this time
            self._diagnose("procedure_no_longer_effective", f"{pid}|{outcome_id}",
                           f"Procedure {pid} was effective before but not in its last application "
                           f"({validation_status(self.s.get(pid))})", "workflow", "observed_fact",
                           [pid, outcome_id], [])
            self._guide(f"The procedure \"{proc.text}\" did not have its expected effect in its last application "
                        "although it worked before: it may be outdated.", "workflow", "observed_fact",
                        [pid, outcome_id], [])

    def _on_knowledge(self, p: dict) -> None:
        """New knowledge resting on a result already known to be unreliable."""
        for kid in p.get("entries", []):
            k = self.s.get(kid)
            if k is None:
                continue
            for ev in k.data.get("supporting_evidence", []) + k.data.get("contradicting_evidence", []):
                m1 = self.s.get(ev.get("entry", "")) if ev.get("entry") else None
                if m1 and m1.data.get("notes"):
                    self._diagnose("knowledge_on_unreliable_result", f"{k.id}|{m1.id}",
                                   f"Knowledge \"{k.text}\" relies on a result with a data-quality problem "
                                   f"({m1.text}); it should be reassessed", "data_quality", "observed_fact",
                                   [k.id, m1.id], m1.prov_refs)

    # ================================================================== rules on provenance records
    def _on_provenance(self, rec: dict) -> None:
        sub, status = rec.get("subtype"), str(rec.get("status", ""))
        if is_computation(rec) and rec.get("subtype") == "parsl_task":
            key = chem.canonical(smiles_of(rec) or "")
            runs = [r for r in self.index.computations() if chem.canonical(smiles_of(r) or "") == key
                    and r.get("campaign_id") == rec.get("campaign_id")]
            ids = sorted({r.get("task_id") for r in runs if r.get("subtype") == "parsl_task"})
            if len(ids) > 1:
                self._diagnose("redundant_computation", key, f"{key} was computed {len(ids)} times",
                               "workflow", "observed_fact", [e.id for e in self.s.by_key("M1", key)],
                               [{"task_id": t} for t in ids])
        elif sub == "llm_call":
            gen = as_dict(rec.get("generated"))
            used = as_dict(rec.get("used"))
            if used.get("has_tools") and not gen.get("tool_calls"):
                wf = rec.get("workflow_id", "")
                self._diagnose("llm_without_tool_calls", wf, "An LLM call asked for tool calls returned none "
                               "(the agent retries)", "workflow", "observed_fact", [],
                               [{"task_id": rec.get("task_id"), "subtype": "llm_call", "workflow_id": wf}])
        elif status == "ERROR":
            self._diagnose("task_error", str(rec.get("activity_id")), f"Task {rec.get('activity_id')} ended in "
                           f"ERROR: {str(rec.get('stderr'))[:200]}", "technical_failure", "observed_fact", [],
                           [{"task_id": rec.get("task_id"), "subtype": sub}])

    # ================================================================== post-execution analysis
    def post_execution(self, records: list[dict] | None = None) -> dict:
        """Broader analysis after a run: provenance summary (Flowcept's own analysis functions), consolidation of
        memory (procedures from recurring failure experiences), strategy and guidance effectiveness."""
        if self.source is not None:
            for rec in self.source.poll():
                self._on_provenance(rec)
        records = records if records is not None else self.index.records()
        report: dict = {"namespace": self.s.ns, "campaign_id": self.cfg.campaign_id}
        try:
            from flowcept.agents.prov_analysis import core as pa
            report["provenance_summary"] = {k: v for k, v in pa.summarize_execution(records).items()
                                            if k != "activities"}
            report["provenance_errors"] = pa.analyze_errors(records)
        except Exception as exc:  # noqa: BLE001
            report["provenance_summary_error"] = str(exc)
        comps = [r for r in records if is_computation(r) and r.get("subtype") == "parsl_task"] or \
                [r for r in records if is_computation(r)]
        per_mol = Counter(chem.canonical(smiles_of(r) or "") for r in comps)
        report["computations"] = {"total": len(comps), "distinct_molecules": len(per_mol),
                                  "redundant": sum(n - 1 for n in per_mol.values() if n > 1)}
        by_class = defaultdict(list)
        for e in self.s.all("M2"):
            if e.kind == "failure":
                by_class[e.data["context"].get("failure_class", "other")].append(e)
        report["procedures_proposed"] = [pid for c, eps in by_class.items()
                                         if len({e.key for e in eps}) > 1 and (pid := self._propose_procedure(c, eps))]
        report["procedures"] = [{"id": e.id, "text": e.text, "validation_status": validation_status(e)}
                                for e in self.s.all("M4")]
        attempts = [e for e in self.s.all("M2") if e.kind == "attempt"]
        report["strategy"] = {"attempts": len(attempts),
                              "improving": sum(1 for e in attempts if e.data["outcome"].get("improved")),
                              "by_agent": dict(Counter(e.key for e in attempts))}
        gs = self.s.all_guidance()
        report["guidance"] = {
            "created": len(gs), "delivered": sum(1 for g in gs if g.delivered),
            "applied": sum(1 for g in gs for v in g.applied.values() if v is True),
            "not_applied": sum(1 for g in gs for v in g.applied.values() if v is False),
            "effective": sum(1 for g in gs for v in g.effective.values() if v is True),
            "not_effective": sum(1 for g in gs for v in g.effective.values() if v is False)}
        diags = [e for e in self.s.all("M2") if e.kind == "diagnosis"]
        report["findings"] = [{"id": e.id, "rule": e.data["context"].get("rule"), "text": e.text,
                               "problem_type": e.data.get("problem_type"), "basis": e.data.get("basis"),
                               "occurrences": e.data["outcome"].get("occurrences")} for e in diags]
        report["memory"] = {c: self.s.count(c) for c in ("M1", "M2", "M3", "M4")}
        report["llm_calls"] = getattr(self.llm, "calls", 0)
        return report


def write_report(report: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"analysis_{int(time.time())}.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    lines = [f"# Post-execution analysis ({report.get('namespace')})", "",
             f"- memory: {report.get('memory')}", f"- computations: {report.get('computations')}",
             f"- guidance: {report.get('guidance')}", f"- strategy: {report.get('strategy')}", "", "## Findings", ""]
    lines += [f"- [{f['problem_type']} / {f['basis']}] {f['text']} (x{f['occurrences']})"
              for f in report.get("findings", [])]
    lines += ["", "## Procedures", ""] + [f"- {p['text']} — {p['validation_status']}"
                                         for p in report.get("procedures", [])]
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
    ap.add_argument("--post", action="store_true", help="post-execution analysis, then exit")
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
    campaign = args.campaign
    if args.post and not campaign:
        campaign = latest_campaign(args.jsonl or cfg.jsonl_glob)
    cfg.campaign_id = campaign
    source = make_source("jsonl" if args.post else cfg.provenance_source, cfg.redis_host, cfg.redis_port,
                         args.jsonl or cfg.jsonl_glob, campaign, online=not args.post)
    an = Analyzer(store, cfg, source=source)
    if args.post:
        an.step(block_ms=10, retry_idle_ms=0)   # finish pending memory events first
        print("report:", write_report(an.post_execution(), args.out))
        return
    an.run_forever()


if __name__ == "__main__":
    main()
