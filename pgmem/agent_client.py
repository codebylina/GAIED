"""
What a scientific agent does with the shared memory, at the integration points of its workflow. The agent never
touches the graph: it calls these methods, and the code creates the nodes and edges.

  planning, critique,   context(): the memory relevant to the current task, retrieved by pgmem/retrieval.py and
  revision              completed with each entry's neighbors, and the analysis agent's guidance
  experiment selection  explored(): what the team already measured or saw fail; the query_memory tool the agent may
                        call itself (answer()); claim_or_wait(): an existing result, or the per-molecule lock
  execution             record_result() -> M1 ABOUT its molecule; record_failure() -> M2 failure ABOUT its molecule
  conclusion            record_conclusion(): the agent's statements (M3, ABOUT the molecules and motifs they name,
                        SUPPORTED_BY / CONTRADICTED_BY / EXPLAINS what they cite, origin "llm") and the iteration's
                        attempt (M2), which its results and failures are PART_OF

Modes (Config.mode): basic = results shared and the lock; structured = the whole memory; full = structured + the
analysis agent's guidance. Every call is a few Redis operations; none waits for the analysis agent.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
import time
import uuid

from . import chem, retrieval
from .config import Config
from .domain import DOMAIN
from .embeddings import Embedder
from .schema import EPISTEMIC_STATUS, ProvRef, attempt, failure, knowledge, result
from .store import MemoryStore

log = logging.getLogger("pgmem.agent")

# agent and LangGraph node being executed (written into the provenance of each LLM call)
CURRENT_NODE: contextvars.ContextVar[str | None] = contextvars.ContextVar("pgmem_node", default=None)
CURRENT_AGENT: contextvars.ContextVar[str | None] = contextvars.ContextVar("pgmem_agent", default=None)

QUANTITY, UNIT = "ionization_energy", "eV"    # ASE returns energies in eV

KNOWLEDGE_INSTRUCTION = """

After your conclusions, add a section that starts with the line KNOWLEDGE and contains ONLY a JSON list. One item per
scientific statement you can make from the results (at most 5):
{"statement": "...", "epistemic_status": "observation|hypothesis|explanation", "scope": "where it applies",
 "supporting_molecules": ["SMILES", ...], "contradicting_molecules": ["SMILES", ...],
 "explains_molecules": ["SMILES", ...], "uncertainty": "what is uncertain"}
"observation": what the measured values directly show; "hypothesis": a conjecture to be tested; "explanation": a
proposed reason for results already seen (list them in explains_molecules). Cite only molecules from the simulation
results."""

_BLOCK = re.compile(r"\n?\s*\**KNOWLEDGE\**\s*:?\s*\n?", re.I)


def from_env(agent_name: str) -> "MemoryClient | None":
    cfg = Config.from_env()
    if not cfg.enabled:
        return None
    return MemoryClient(MemoryStore.connect(cfg.redis_host, cfg.redis_port, cfg.namespace), cfg, agent_name)


class MemoryClient:
    def __init__(self, store: MemoryStore, cfg: Config, agent: str) -> None:
        self.s, self.cfg, self.agent = store, cfg, agent
        self.emb = Embedder(store, cfg.embed_model, cfg.embed_url) if "semantic" in cfg.channels else None
        self.seed = agent.split("#")[0]          # GAIED names an agent after its seed molecule
        self.best: float | None = None          # this agent's best value, to describe each attempt's outcome
        self.iteration = 0
        self._results: list[str] = []           # this iteration: results computed, failures, results reused
        self._failures: list[str] = []
        self._reused: list[str] = []
        self._proposals: list[str] = []
        self._proposed_at: float | None = None
        self._best_start: float | None = None
        self._recent_proposals: list[str] = []
        self._recent_failure_classes: set[str] = set()

    # ------------------------------------------------------------------ experiment selection
    def explored(self) -> str:
        """Team-wide list of measured (and, with the structured memory, failed) molecules, for tool calling."""
        measured = [f"{e.key} ({e.data['value']:.4g} {e.data['unit']})" for e in self.s.all("M1")
                    if e.data.get("value") is not None]
        out = ("\nMolecules already simulated by the team (do not repeat): " + ", ".join(measured)) if measured else ""
        if self.cfg.structured:
            failed = sorted({e.key for e in self.s.all("M2") if e.kind == "failure"})
            if failed:
                out += "\nMolecules whose simulation failed (do not repeat): " + ", ".join(failed)
        return out

    def lookup(self, smiles: str, conditions: dict) -> float | None:
        """An existing result for the same molecule under the same conditions, if any."""
        key = chem.canonical(smiles)
        for e in self.s.by_key("M1", key):
            c = e.data.get("conditions", {})
            if all(c.get(k) == v for k, v in conditions.items()) and e.data.get("value") is not None:
                self.s.emit("reused", {"entry": e.id, "agent": self.agent, "key": key})
                self._seen(e.data["value"])
                self._reused.append(key)
                return float(e.data["value"])
        return None

    # ------------------------------------------------------------------ coordination (lock per molecule)
    _RELEASE = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"

    def _claim_key(self, key: str) -> str:
        return self.s._k("claim", key)

    async def claim_or_wait(self, smiles: str, conditions: dict, poll_s: float = 1.0) -> tuple[float | None, str | None]:
        """Before computing: (value, None) when the result exists or arrives while another computation of the same
        molecule runs; (None, token) when this call holds the lock and must compute, then call release(); a failure
        of the computation it waited for is raised as is."""
        value = self.lookup(smiles, conditions)
        if value is not None:
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
                    self._waited(key, t0, "failure")
                    raise RuntimeError(f"{failed[0].data['error'] or 'computation failed'} (computed by another call)")
            if self.s.r.set(self._claim_key(key), f"{self.agent}|{token}", nx=True, ex=self.cfg.claim_ttl_s):
                value = None if waited else self.lookup(smiles, conditions)   # finished just before the lock
                if value is not None:
                    self.release(smiles, token)
                    return value, None
                if waited:        # the lock expired or was released without a stored result
                    self._waited(key, t0, "computes")
                return None, token
            waited = True

    def release(self, smiles: str, token: str | None) -> None:
        if token:
            self.s.r.eval(self._RELEASE, 1, self._claim_key(chem.canonical(smiles)), f"{self.agent}|{token}")

    def _waited(self, key: str, t0: float, outcome: str) -> None:
        self.s.emit("waited", {"agent": self.agent, "key": key, "seconds": round(time.time() - t0, 1),
                               "outcome": outcome})

    def record_proposals(self, smiles_list: list[str]) -> None:
        keys = [chem.canonical(s) for s in smiles_list]
        self._proposals, self._proposed_at, self._best_start = keys, time.time(), self.best
        if keys:
            self._recent_proposals = (keys + self._recent_proposals)[:20]
        # what the memory holds when the proposals are made (the analysis agent reads the event later, after this
        # agent's own new results are stored, so it must not recompute this itself)
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
        eid, merged = self.s.add(result(key, QUANTITY, value, UNIT, conditions, self.agent, prov, campaign_id,
                                        spelling=smiles))
        if DOMAIN.valid(key):
            self.s.link(eid, self.s.molecule(key), "ABOUT", self.agent)
        self._seen(value)
        self._results.append(eid)
        self.s.emit("result", {"entry": eid, "agent": self.agent, "key": key, "task_id": task_id, "merged": merged})
        return eid

    def record_failure(self, smiles: str, error: str, task_id: str | None, workflow_id: str | None,
                       campaign_id: str | None) -> str | None:
        if not self.cfg.structured:
            return None
        key, fclass = chem.canonical(smiles), chem.failure_class(error)
        prov = ProvRef(task_id=task_id, workflow_id=workflow_id, campaign_id=campaign_id, subtype="parsl_task")
        eid, _ = self.s.add(failure(key, fclass, str(error), self.agent, self.iteration, smiles, [prov.to_dict()],
                                    campaign_id))
        if DOMAIN.valid(key):
            self.s.link(eid, self.s.molecule(key), "ABOUT", self.agent)
        self._failures.append(eid)
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
        """Store the agent's statements (M3) and this iteration's attempt (M2); return the conclusions without the
        KNOWLEDGE block, as the workflow continues with them."""
        if not self.cfg.structured:
            return text
        body, items = split_knowledge(text)
        prov_refs = [prov] if prov else []
        ids = [kid for it in items if (kid := self._statement(it, prov_refs))]
        if ids:
            self.s.emit("knowledge", {"entries": ids, "agent": self.agent})
        before = self._best_start
        improved = None if before is None and self.best is None else (
            self.best is not None and (before is None or self.best > before))
        aid, _ = self.s.add(attempt(
            f"{self.agent}, iteration {self.iteration}: {len(self._proposals)} proposed, {len(self._results)} "
            f"computed, {len(self._reused)} reused, {len(self._failures)} failed"
            + (f", best {self.best:.4g}" if self.best is not None else ""), self.agent, self.iteration,
            self._proposals, self._proposed_at, self._reused, improved, before, self.best, plan[:400], prov_refs,
            self.cfg.campaign_id))
        for eid in self._results + self._failures:
            self.s.link(eid, aid, "PART_OF", self.agent)
        self.s.emit("attempt", {"entry": aid, "agent": self.agent, "iteration": self.iteration})
        self.iteration += 1
        if not self._failures:                  # failure classes met in the last iteration stay "recent"
            self._recent_failure_classes = set()
        self._results, self._failures, self._reused, self._proposals = [], [], [], []
        return body

    def _statement(self, it: dict, prov_refs: list[dict]) -> str | None:
        stmt = str(it.get("statement", "")).strip()
        if not stmt:
            return None
        status = it.get("epistemic_status") if it.get("epistemic_status") in EPISTEMIC_STATUS else "hypothesis"
        kid, _ = self.s.add(knowledge(stmt, status, self.agent, scope=str(it.get("scope", "")),
                                      uncertainty=str(it.get("uncertainty", "")), prov_refs=prov_refs,
                                      campaign_id=self.cfg.campaign_id))
        cited = {"SUPPORTED_BY": it.get("supporting_molecules"), "CONTRADICTED_BY": it.get("contradicting_molecules"),
                 "EXPLAINS": it.get("explains_molecules")}
        about = set(DOMAIN.entities_in(stmt))
        for rel, mols in cited.items():
            for m in mols if isinstance(mols, list) else []:
                key = chem.canonical(str(m))
                found = self.s.by_key("M1", key)[:1] or [e for e in self.s.by_key("M2", key) if e.kind == "failure"][:1]
                for e in found:             # the agent's citation of its evidence: proposed, not checked
                    self.s.link(kid, e.id, rel, self.agent, "llm")
                    about.add(key)
        for key in about:
            if DOMAIN.valid(key):
                self.s.link(kid, self.s.molecule(key), "ABOUT", self.agent)
        for motif in DOMAIN.motifs_named_in(stmt):
            self.s.link(kid, self.s.motif(motif), "ABOUT", self.agent)
        return kid

    # ------------------------------------------------------------------ planning, critique, revision
    def task_query(self, text: str, molecules: list[str] | None = None) -> retrieval.Query:
        """The current task: the text given, the molecules it names, the agent's seed and recent proposals, and the
        failure classes it met recently."""
        mols = list(molecules or []) + [self.seed] + self._recent_proposals
        return retrieval.build_query(text, mols, self._recent_failure_classes)

    def _retrieve(self, category: str, q: retrieval.Query, **kw) -> list:
        return retrieval.retrieve(self.s, category, q, self.cfg.retrieval_limit, self.cfg.channels, self.emb, **kw)

    def _procedures(self, q: retrieval.Query) -> list[tuple]:
        return [(e, why) for e in self.s.all("M4") if (why := retrieval.applicable(e, q, self.s))][
            :self.cfg.retrieval_limit]

    def _found(self, q: retrieval.Query) -> dict:
        return {"M1": self._retrieve("M1", q, keep=lambda e: e.data.get("value") is not None),
                "M2": self._retrieve("M2", q, kinds=("failure", "diagnosis", "attempt"), keep=_worth_showing),
                "M3": self._retrieve("M3", q)}

    def context(self, node: str, query: str) -> str:
        """The memory section added to a node's prompt; guidance shown here is marked delivered."""
        if not self.cfg.structured:
            return ""
        q = self.task_query(query)
        found, procs = self._found(q), self._procedures(q)
        shown = {e.id for e, _, _ in found["M1"]}
        best = [e for e in sorted((e for e in self.s.all("M1") if e.data.get("value") is not None),
                                  key=lambda e: -e.data["value"]) if e.id not in shown][:3]
        d = lambda e: retrieval.describe(self.s, e)  # noqa: E731
        sections = [
            ("Results on molecules related to the current task (M1)", [d(e) for e, _, _ in found["M1"]]),
            ("Best results of the team so far, for reference (M1)", [d(e) for e in best]),
            ("Relevant experiences of the team (M2)",
             [f"[{'improving attempt' if e.kind == 'attempt' else e.kind}] {d(e)}" for e, _, _ in found["M2"]]),
            ("Team knowledge related to the task (M3), with its evidence and verdict", [d(e) for e, _, _ in found["M3"]]),
            ("Procedures that apply now (M4)", [f"{d(e)} Applies because {why}." for e, why in procs])]
        parts = [f"{title}:\n" + "\n".join(f"- {line}" for line in lines) for title, lines in sections if lines]
        self._log("context", q, found, procs, node=node, query=query[:200],
                  extra_m1=[e.id for e in best])
        if self.cfg.guidance:
            gs = self.s.pending_guidance(self.agent)
            if gs:
                parts.append("Guidance from the analysis agent:\n" + "\n".join(
                    f"- {g.message} [{g.epistemic_status}]" for g in gs))
                for g in gs:
                    self.s.mark_delivered(g.id, self.agent, node)
        return ("\n\nSHARED MEMORY\n" + "\n\n".join(parts)) if parts else ""

    def _log(self, event: str, q: retrieval.Query, found: dict, procs: list, extra_m1: list | None = None,
             **payload) -> None:
        """What was retrieved, with the rank each channel gave (the evaluation judges relevance from this)."""
        entries = {c: [e.id for e, _, _ in v] for c, v in found.items()}
        entries["M1"] += extra_m1 or []
        entries["M4"] = [e.id for e, _ in procs]
        ranks = {e.id: r for v in found.values() for e, _, r in v}
        self.s.emit(event, {"agent": self.agent, "iteration": self.iteration, "molecules": q.molecules,
                            "entries": entries, "ranks": ranks, **payload})

    # ------------------------------------------------------------------ on demand
    def answer(self, question: str, smiles: str | None = None) -> str:
        """The query_memory tool: what the memory holds about the question and the molecule asked about, each entry
        with its neighbors and its Flowcept task."""
        t0 = time.time()
        q = self.task_query(question, [smiles] if smiles else None)
        if smiles and DOMAIN.valid(smiles):     # the molecule asked about, not the agent's other ones
            q = retrieval.build_query(question, [smiles], self._recent_failure_classes)
        found, procs = self._found(q), self._procedures(q)
        lines = []
        for c, hits in found.items():
            for e, _, _ in hits:
                task = retrieval.task_of(e)
                lines.append(f"[{e.kind} {e.id}] {retrieval.describe(self.s, e)}"
                             + (f" (Flowcept task {task})" if task else ""))
        lines += [f"[procedure {e.id}] {retrieval.describe(self.s, e)} Applies because {why}." for e, why in procs]
        self._log("memory_query", q, found, procs, question=question[:300], smiles=smiles,
                  seconds=round(time.time() - t0, 3))
        return "\n".join(lines) if lines else "The shared memory has nothing related to this question yet."

    def tool(self):
        """The query_memory tool offered to the agent at experiment selection (LangChain StructuredTool)."""
        from langchain_core.tools import StructuredTool

        def query_memory(question: str, smiles: str = "") -> str:
            return self.answer(question, smiles or None)
        return StructuredTool.from_function(
            query_memory, name="query_memory",
            description="Ask the team's shared memory before deciding: results on a molecule and similar ones, "
                        "failures, knowledge with its verdict, procedures. Args: question (text), "
                        "smiles (optional, the molecule you are considering).")


def _worth_showing(e) -> bool:
    """Experiences shown to agents: failures, diagnoses (explanations come with the failures they explain), and
    attempts that improved."""
    if e.kind == "attempt":
        return e.data.get("improved") is True
    return not (e.kind == "diagnosis" and e.data["rule"] == "failure_explanation")


def split_knowledge(text: str) -> tuple[str, list[dict]]:
    """Split a conclusion into its prose and the items of its KNOWLEDGE block (empty when absent or unreadable)."""
    m = list(_BLOCK.finditer(text or ""))
    if not m:
        return text, []
    body, block = text[:m[-1].start()].rstrip(), text[m[-1].end():]
    i, j = block.find("["), block.rfind("]")
    if i < 0 or j <= i:
        return body, []
    try:
        items = json.loads(block[i:j + 1])
    except ValueError:
        log.warning("KNOWLEDGE block not readable as JSON; no knowledge stored for this conclusion")
        return body, []
    return body, [x for x in items if isinstance(x, dict)]
