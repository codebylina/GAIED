"""
What a scientific agent does with the shared memory, at the integration points of its workflow:

  planning              context(): the memory relevant to the current task (pgmem/retrieval.py): results on similar
                        molecules and the team's best (M1), experiences (M2), knowledge with its verification (M3),
                        applicable procedures (M4), and the analysis agent's guidance
  experiment selection  explored(): what the team already measured or saw fail; query_memory, a tool the agent may
                        call itself before deciding (answer()); claim_or_wait(): an existing result, or the lock
  execution             record_result() -> M1 (with the Flowcept task id); record_failure() -> M2
  conclusion            conclude_instruction() + record_conclusion() -> proposed knowledge (M3) and the attempt (M2)
  critique / revision   context() again, with the guidance received since

Modes (Config.mode): basic = results shared (M1 only, read back at experiment selection); structured = M1-M4;
full = structured + guidance from the analysis agent. Every call is a few Redis operations; none waits for the
analysis agent. Retrieval is task-aware (PGMEM_RETRIEVAL=contextual) or, for comparison, the earlier one: best
results and keyword search (PGMEM_RETRIEVAL=recent).
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
import time
import uuid

from . import chem
from .config import Config
from . import retrieval
from .schema import EPISTEMIC_STATUS, M3_KINDS, ProvRef, experience, knowledge, observation, validation_status
from .store import MemoryStore

log = logging.getLogger("pgmem.agent")

# agent and LangGraph node being executed (written into the provenance of each LLM call)
CURRENT_NODE: contextvars.ContextVar[str | None] = contextvars.ContextVar("pgmem_node", default=None)
CURRENT_AGENT: contextvars.ContextVar[str | None] = contextvars.ContextVar("pgmem_agent", default=None)

QUANTITY, UNIT = "ionization_energy", "eV"    # ASE returns energies in eV

KNOWLEDGE_INSTRUCTION = """

After your conclusions, add a section that starts with the line KNOWLEDGE and contains ONLY a JSON list. One item per
scientific statement you can make from the results (at most 5):
{"statement": "...", "kind": "hypothesis|claim|finding|rule|constraint|negative_finding|uncertainty",
 "epistemic_status": "observation|hypothesis|explanation", "scope": "where the statement applies",
 "supporting_molecules": ["SMILES", ...], "contradicting_molecules": ["SMILES", ...],
 "uncertainty": "what is uncertain"}
Use "observation" only for what the measured values directly show, "hypothesis" for a conjecture to be tested, and
"explanation" for a proposed reason. Cite only molecules from the simulation results."""

_BLOCK = re.compile(r"\n?\s*\**KNOWLEDGE\**\s*:?\s*\n?", re.I)


def from_env(agent_name: str) -> "MemoryClient | None":
    cfg = Config.from_env()
    if not cfg.enabled:
        return None
    return MemoryClient(MemoryStore.connect(cfg.redis_host, cfg.redis_port, cfg.namespace), cfg, agent_name)


class MemoryClient:
    def __init__(self, store: MemoryStore, cfg: Config, agent: str) -> None:
        self.s, self.cfg, self.agent = store, cfg, agent
        self.best: float | None = None          # this agent's best value, to describe each attempt's outcome
        self.iteration = 0
        self._iter_results: list[str] = []
        self._iter_failures: list[str] = []
        self._iter_proposals: list[str] = []
        self._best_start: float | None = None
        self.seed = agent.split("#")[0]          # GAIED names an agent after its seed molecule
        self._recent_proposals: list[str] = []
        self._recent_failure_classes: set[str] = set()

    # ------------------------------------------------------------------ experiment selection
    def explored(self) -> str:
        """Team-wide list of measured and failed molecules, for the tool-calling prompt."""
        measured = [f"{e.key} ({e.data.get('value'):.4g} {e.data.get('unit', '')})" for e in self.s.all("M1")
                    if e.data.get("value") is not None]
        out = ""
        if measured:
            out += "\nMolecules already simulated by the team (do not repeat): " + ", ".join(measured)
        if self.cfg.structured:
            failed = sorted({e.key for e in self.s.all("M2") if e.kind == "failure"})
            if failed:
                out += "\nMolecules whose simulation failed (do not repeat): " + ", ".join(failed)
        return out

    def lookup(self, smiles: str, conditions: dict) -> float | None:
        """An existing result for the same molecule under the same conditions, if any."""
        if not self.cfg.reuse_results:
            return None
        key = chem.canonical(smiles)
        for e in self.s.by_key("M1", key):
            c = e.data.get("conditions", {})
            if all(c.get(k) == v for k, v in conditions.items()) and e.data.get("value") is not None:
                self.s.emit("reused", {"entry": e.id, "agent": self.agent, "key": key})
                self._seen(e.data["value"])
                self._iter_results.append(e.id)
                return float(e.data["value"])
        return None

    # ------------------------------------------------------------------ coordination (lock per molecule)
    _RELEASE = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"

    def _claim_key(self, key: str) -> str:
        return self.s._k("claim", key)

    async def claim_or_wait(self, smiles: str, conditions: dict, poll_s: float = 1.0) -> tuple[float | None, str | None]:
        """Before computing: (value, None) when the result exists or arrives while another computation of the same
        molecule runs; (None, token) when this call holds the lock and must compute, then call release(); a failure
        of the computation it waited for is raised as is. Without locks: (lookup result, None)."""
        self._known_failure(smiles)
        value = self.lookup(smiles, conditions)
        if value is not None or not (self.cfg.claims and self.cfg.reuse_results):
            return value, None
        key, token, t0, waited = chem.canonical(smiles), uuid.uuid4().hex, time.time(), False
        while True:
            if waited:            # another call holds the lock: did its result or failure arrive?
                await asyncio.sleep(poll_s)
                value = self.lookup(smiles, conditions)
                if value is not None:
                    self._waited(key, t0, "result")
                    return value, None
                failed = [e for e in self.s.by_key("M2", key) if e.kind == "failure" and e.created_at >= t0]
                if failed:
                    self._iter_failures.append(failed[0].id)
                    self._waited(key, t0, "failure")
                    raise RuntimeError(f"{failed[0].data['outcome'].get('error', 'computation failed')} "
                                       "(computed by another call)")
            if self.s.r.set(self._claim_key(key), f"{self.agent}|{token}", nx=True, ex=self.cfg.claim_ttl_s):
                value = None if waited else self.lookup(smiles, conditions)   # finished just before the lock
                if value is not None:
                    self.release(smiles, token)
                    return value, None
                if waited:        # the lock expired or was released without a stored result
                    self._waited(key, t0, "computes")
                return None, token
            waited = True

    def _known_failure(self, smiles: str) -> None:
        """Ablation PGMEM_REUSE_FAILURES: a molecule whose computation already failed is not computed again; the
        recorded failure is returned instead (a reuse of negative results, like a cache)."""
        if not (self.cfg.reuse_failures and self.cfg.structured):
            return
        key = chem.canonical(smiles)
        failed = [e for e in self.s.by_key("M2", key) if e.kind == "failure"]
        if failed:
            self._iter_failures.append(failed[0].id)
            self.s.emit("reused_failure", {"entry": failed[0].id, "agent": self.agent, "key": key})
            raise RuntimeError(f"{failed[0].data['outcome'].get('error', 'computation failed')} "
                               "(failed before; not computed again)")

    def release(self, smiles: str, token: str | None) -> None:
        if token:
            self.s.r.eval(self._RELEASE, 1, self._claim_key(chem.canonical(smiles)), f"{self.agent}|{token}")

    def _waited(self, key: str, t0: float, outcome: str) -> None:
        self.s.emit("waited", {"agent": self.agent, "key": key, "seconds": round(time.time() - t0, 1),
                               "outcome": outcome})

    def record_proposals(self, smiles_list: list[str]) -> None:
        keys = [chem.canonical(s) for s in smiles_list]
        self._iter_proposals = keys
        if keys:
            self._recent_proposals = (keys + self._recent_proposals)[:20]
        self._best_start = self.best
        # what the memory already holds when the proposals are made; the analysis agent reads the event later,
        # after this agent's own new results are stored, so it must not recompute this itself
        known = [k for k in keys if self.s.by_key("M1", k)]
        failed = [k for k in keys if any(e.kind == "failure" for e in self.s.by_key("M2", k))]
        self.s.emit("proposals", {"agent": self.agent, "iteration": self.iteration, "keys": keys,
                                  "known": known, "failed": failed})

    # ------------------------------------------------------------------ execution
    def record_result(self, smiles: str, value: float, conditions: dict, task_id: str | None,
                      workflow_id: str | None, campaign_id: str | None) -> str:
        key = chem.canonical(smiles)
        prov = ProvRef(task_id=task_id, workflow_id=workflow_id, campaign_id=campaign_id, subtype="parsl_task",
                       activity_id="compute_ionization_energy")
        obs = observation(key, QUANTITY, value, UNIT, conditions, self.agent, prov, campaign_id)
        obs.data["smiles_given"] = smiles      # the spelling used, kept out of the conditions (deduplication)
        eid, merged = self.s.add(obs)
        self._seen(value)
        self._iter_results.append(eid)
        self.s.emit("result", {"entry": eid, "agent": self.agent, "key": key, "task_id": task_id,
                               "merged": merged})
        return eid

    def record_failure(self, smiles: str, error: str, task_id: str | None, workflow_id: str | None,
                       campaign_id: str | None) -> str | None:
        if not self.cfg.structured:
            return None
        key = chem.canonical(smiles)
        fclass = chem.failure_class(error)
        e = experience("failure", f"Simulation of {key} failed ({fclass})", self.agent, key=key,
                       context={"smiles": smiles, "failure_class": fclass, "iteration": self.iteration},
                       outcome={"error": str(error)[:500]},
                       prov_refs=[ProvRef(task_id=task_id, workflow_id=workflow_id, campaign_id=campaign_id,
                                          subtype="parsl_task").to_dict()], campaign_id=campaign_id)
        eid, _ = self.s.add(e)
        self._iter_failures.append(eid)
        self._recent_failure_classes.add(fclass)
        self.s.emit("failure", {"entry": eid, "agent": self.agent, "key": key, "failure_class": fclass,
                                "task_id": task_id})
        return eid

    def _seen(self, value) -> None:
        if isinstance(value, (int, float)) and (self.best is None or value > self.best):
            self.best = float(value)

    # ------------------------------------------------------------------ conclusion
    def conclude_instruction(self) -> str:
        return KNOWLEDGE_INSTRUCTION if self.cfg.structured else ""

    def record_conclusion(self, text: str, plan: str, prov: dict | None) -> str:
        """Store the proposed knowledge (M3) and this iteration's attempt (M2); return the conclusions without the
        KNOWLEDGE block, as the workflow continues with them."""
        if not self.cfg.structured:
            return text
        best_before = self._best_start
        body, items = split_knowledge(text)
        prov_refs = [prov] if prov else []
        ids = []
        for it in items:
            sup, con = self._evidence(it.get("supporting_molecules")), self._evidence(it.get("contradicting_molecules"))
            kind = it.get("kind") if it.get("kind") in M3_KINDS else "claim"
            status = it.get("epistemic_status") if it.get("epistemic_status") in EPISTEMIC_STATUS else "hypothesis"
            stmt = str(it.get("statement", "")).strip()
            if not stmt:
                continue
            kid, _ = self.s.add(knowledge(kind, stmt, status, self.agent, scope=str(it.get("scope", "")),
                                          supporting=sup, contradicting=con,
                                          uncertainty=str(it.get("uncertainty", "")), prov_refs=prov_refs,
                                          campaign_id=self.cfg.campaign_id))
            ids.append(kid)
        if ids:
            self.s.emit("knowledge", {"entries": ids, "agent": self.agent})
        improved = None if best_before is None and self.best is None else (
            self.best is not None and (best_before is None or self.best > best_before))
        att = experience("attempt", f"{self.agent}, iteration {self.iteration}: {len(self._iter_proposals)} proposed, "
                         f"{len(self._iter_results)} results, {len(self._iter_failures)} failures"
                         + (f", best {self.best:.4g}" if self.best is not None else ""), self.agent, key=self.agent,
                         context={"iteration": self.iteration, "plan_excerpt": plan[:400],
                                  "proposed": self._iter_proposals},
                         outcome={"best_before": best_before, "best_after": self.best, "improved": improved,
                                  "results": self._iter_results, "failures": self._iter_failures,
                                  "knowledge": ids},
                         prov_refs=prov_refs, refs=self._iter_results + self._iter_failures + ids,
                         campaign_id=self.cfg.campaign_id)
        aid, _ = self.s.add(att)
        self.s.emit("attempt", {"entry": aid, "agent": self.agent, "iteration": self.iteration})
        self.iteration += 1
        if not self._iter_failures:              # failure classes met in the last iteration stay "recent"
            self._recent_failure_classes = set()
        self._iter_results, self._iter_failures, self._iter_proposals = [], [], []
        return body

    def _evidence(self, mols) -> list[dict]:
        out = []
        for m in mols if isinstance(mols, list) else []:
            key = chem.canonical(str(m))
            # a measured result, or else a recorded failure (knowledge about a molecule that failed cites it)
            found = self.s.by_key("M1", key)[:1] or [e for e in self.s.by_key("M2", key) if e.kind == "failure"][:1]
            for e in found:
                out.append({"entry": e.id, "prov_refs": e.prov_refs})
        return out

    # ------------------------------------------------------------------ planning, critique, revision
    def task_query(self, text: str, molecules: list[str] | None = None) -> retrieval.Query:
        """The current task: the text given (plan, critique, question), the molecules it names, the agent's seed
        and recent proposals, and the failure classes it met recently."""
        mols = list(molecules or []) + [self.seed] + self._recent_proposals
        return retrieval.build_query(text, mols, self._recent_failure_classes)

    def context(self, node: str, query: str) -> str:
        """The memory section added to a node's prompt; guidance received here is marked delivered."""
        if not self.cfg.structured:
            return ""
        q = self.task_query(query)               # also logged in the ablation, to judge what it retrieves
        sections = self._recent_sections(query) if self.cfg.retrieval == "recent" else self._task_sections(q)
        parts = [f"{title}:\n" + "\n".join(f"- {line}" for line in lines) for title, lines, _ in sections if lines]
        entries: dict[str, list[str]] = {"M1": [], "M2": [], "M3": [], "M4": []}
        for _, _, ids in sections:
            for c, i in ids:
                entries[c].append(i)
        self.s.emit("context", {"agent": self.agent, "node": node, "query": query[:200], "iteration": self.iteration,
                                "molecules": q.molecules, "entries": entries})
        if self.cfg.guidance:
            gs = self.s.pending_guidance(self.agent)
            if gs:
                parts.append("Guidance from the analysis agent:\n" + "\n".join(
                    f"- {g.message} [{g.basis.replace('_', ' ')}]" for g in gs))
                for g in gs:
                    self.s.mark_delivered(g.id, self.agent, node)
        return ("\n\nSHARED MEMORY\n" + "\n\n".join(parts)) if parts else ""

    def _task_sections(self, q: retrieval.Query) -> list:
        n, cats = self.cfg.retrieval_limit, self.cfg.context_categories
        sections = []
        if "M1" in cats:
            similar = retrieval.retrieve(self.s, "M1", q, n, keep=lambda e: e.data.get("value") is not None)
            sections.append(("Results on molecules related to the current task (M1)",
                             [f"{_with_notes(e)} [similarity {p['structure']:.2f}]" for e, _, p in similar],
                             [("M1", e.id) for e, _, _ in similar]))
            shown = {e.id for e, _, _ in similar}
            best = [e for e in sorted([e for e in self.s.all("M1") if e.data.get("value") is not None],
                                      key=lambda e: -e.data["value"]) if e.id not in shown][:3]
            sections.append(("Best results of the team so far, for reference (M1)", [_with_notes(e) for e in best],
                             [("M1", e.id) for e in best]))
        if "M2" in cats:
            exps = retrieval.retrieve(self.s, "M2", q, n, kinds=retrieval.M2_KINDS,
                                      keep=lambda e: e.kind != "attempt" or e.data["outcome"].get("improved"))
            sections.append(("Relevant experiences of the team (M2)",
                             [f"[{'improving attempt' if e.kind == 'attempt' else e.kind}] {e.text}"
                              for e, _, _ in exps], [("M2", e.id) for e, _, _ in exps]))
        if "M3" in cats:
            know = retrieval.retrieve(self.s, "M3", q, n)
            sections.append(("Team knowledge related to the task (M3), with its evidence and verification",
                             [_knowledge_line(e, self.s) for e, _, _ in know], [("M3", e.id) for e, _, _ in know]))
        if "M4" in cats:
            procs = [(e, why) for e in self.s.all("M4") if (why := retrieval.applicable(e, q, self.s))][:n]
            sections.append(("Procedures that apply now (M4)", [
                f"{e.text} Applies because {why}. When: {e.data['trigger']}. Do: {e.data['actions']}. "
                f"Limits: {e.data['limitations']}. ({validation_status(e)})" for e, why in procs],
                [("M4", e.id) for e, _ in procs]))
        return sections

    def _recent_sections(self, query: str) -> list:
        """The earlier retrieval (ablation): the team's best results and a keyword search per category."""
        n, cats = self.cfg.retrieval_limit, self.cfg.context_categories
        best = sorted([e for e in self.s.all("M1") if e.data.get("value") is not None],
                      key=lambda e: -e.data["value"])[:n] if "M1" in cats else []
        know = self.s.search("M3", query, limit=n) if "M3" in cats else []
        exps = self.s.search("M2", query, limit=n, kinds=("failure", "diagnosis", "outcome")) if "M2" in cats else []
        procs = self.s.search("M4", query, limit=n) if "M4" in cats else []
        return [("Best results of the team (M1)", [_with_notes(e) for e in best], [("M1", e.id) for e in best]),
                ("Team knowledge (M3), with its evidence", [_knowledge_line(e, self.s) for e in know],
                 [("M3", e.id) for e in know]),
                ("Relevant experiences (M2)", [f"[{e.kind}] {e.text}" for e in exps], [("M2", e.id) for e in exps]),
                ("Procedures (M4)", [f"{e.text} When: {e.data['trigger']}. Do: {e.data['actions']}. "
                                     f"Limits: {e.data['limitations']}. ({validation_status(e)})" for e in procs],
                 [("M4", e.id) for e in procs])]

    # ------------------------------------------------------------------ on demand
    def answer(self, question: str, smiles: str | None = None) -> str:
        """Answer an agent's own question to the memory (the query_memory tool): results on the molecule and on
        similar ones, related failures and diagnoses, knowledge and its verification, applicable procedures, each
        with its reference to the Flowcept task when there is one."""
        t0 = time.time()
        q = self.task_query(question, [smiles] if smiles else None)
        if smiles and chem.parses(smiles):       # the molecule asked about first, not the agent's other ones
            key = chem.canonical(smiles)
            q = retrieval.build_query(question, [key], self._recent_failure_classes)
        k, cats = self.cfg.retrieval_limit, self.cfg.context_categories
        found = {
            "M1": retrieval.retrieve(self.s, "M1", q, k, keep=lambda e: e.data.get("value") is not None)
            if "M1" in cats else [],
            "M2": retrieval.retrieve(self.s, "M2", q, k, kinds=("failure", "diagnosis", "outcome"))
            if "M2" in cats else [],
            "M3": retrieval.retrieve(self.s, "M3", q, k) if "M3" in cats else [],
        }
        procs = [(e, why) for e in self.s.all("M4") if (why := retrieval.applicable(e, q, self.s))][:k] \
            if "M4" in cats else []
        lines = []
        for e, _, p in found["M1"]:
            lines.append(f"[result {e.id}] {_with_notes(e)} (similarity {p['structure']:.2f}; "
                         f"Flowcept task {_task(e)})")
        for e, _, _ in found["M2"]:
            lines.append(f"[{e.kind} {e.id}] {e.text}" + (f" (Flowcept task {_task(e)})" if _task(e) else ""))
        for e, _, _ in found["M3"]:
            lines.append(f"[knowledge {e.id}] {_knowledge_line(e, self.s)}")
        for e, why in procs:
            lines.append(f"[procedure {e.id}] {e.text} Applies because {why}. Do: {e.data['actions']}")
        ids = {c: [e.id for e, _, _ in v] for c, v in found.items()} | {"M4": [e.id for e, _ in procs]}
        self.s.emit("memory_query", {"agent": self.agent, "question": question[:300], "smiles": smiles,
                                     "molecules": q.molecules, "iteration": self.iteration, "entries": ids,
                                     "seconds": round(time.time() - t0, 3)})
        return "\n".join(lines) if lines else "The shared memory has nothing related to this question yet."

    def tool(self):
        """The query_memory tool offered to the agent at experiment selection (LangChain StructuredTool)."""
        from langchain_core.tools import StructuredTool

        def query_memory(question: str, smiles: str = "") -> str:
            return self.answer(question, smiles or None)
        return StructuredTool.from_function(
            query_memory, name="query_memory",
            description="Ask the team's shared memory before deciding: results on a molecule and similar ones, "
                        "failures, knowledge with its verification, procedures. Args: question (text), "
                        "smiles (optional, the molecule you are considering).")


def _with_notes(e) -> str:
    notes = e.data.get("notes", [])
    return e.text + (f" (WARNING: {notes[-1]['note']})" if notes else "")


def _task(e) -> str | None:
    return next((r.get("task_id") for r in e.prov_refs if r.get("task_id")), None)


def _knowledge_line(e, store: MemoryStore | None = None) -> str:
    d = e.data
    ver = store.get_verification(e.id) if store is not None else None
    return (f"{e.text} [{e.kind}, {d['epistemic_status']}; {len(d['supporting_evidence'])} result(s) cited for, "
            f"{len(d['contradicting_evidence'])} against"
            + (f"; uncertain: {d['uncertainty']}" if d.get('uncertainty') else "")
            + (f"; checked against the results: {ver['verdict']} ({ver['reason']})"
               if ver and ver.get("verdict") else "") + "]")


def split_knowledge(text: str) -> tuple[str, list[dict]]:
    """Split a conclusion into its prose and the items of its KNOWLEDGE block (empty when absent or unreadable)."""
    m = list(_BLOCK.finditer(text or ""))
    if not m:
        return text, []
    start = m[-1].start()
    body, block = text[:start].rstrip(), text[m[-1].end():]
    i, j = block.find("["), block.rfind("]")
    if i < 0 or j <= i:
        return body, []
    try:
        items = json.loads(block[i:j + 1])
    except ValueError:
        log.warning("KNOWLEDGE block not readable as JSON; no knowledge stored for this conclusion")
        return body, []
    return body, [x for x in items if isinstance(x, dict)]
