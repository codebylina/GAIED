"""
What a scientific agent does with the shared memory, at the integration points of its workflow:

  planning              context(): relevant knowledge (M3), experiences (M2), procedures (M4), team results (M1),
                        and the analysis agent's guidance
  experiment selection  explored(): what the team already measured or saw fail; lookup(): an existing result
  execution             record_result() -> M1 (with the Flowcept task id); record_failure() -> M2
  conclusion            conclude_instruction() + record_conclusion() -> proposed knowledge (M3) and the attempt (M2)
  critique / revision   context() again, with the guidance received since

Modes (Config.mode): basic = results shared (M1 only, read back at experiment selection); structured = M1-M4;
full = structured + guidance from the analysis agent. Every call is a few Redis operations; none waits for the
analysis agent.
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
from .schema import EPISTEMIC_STATUS, M3_KINDS, ProvRef, experience, knowledge, observation, validation_status
from .store import MemoryStore

log = logging.getLogger("pgmem.agent")

# LangGraph node being executed by this agent (written into the provenance of its LLM calls, see G7)
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
        self._renewers: dict = {}               # lock token -> task renewing the lock while computing
        self.best: float | None = None          # this agent's best value, to describe each attempt's outcome
        self.iteration = 0
        self._iter_results: list[str] = []
        self._iter_failures: list[str] = []
        self._iter_proposals: list[str] = []
        self._best_start: float | None = None

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
    _RENEW = ("if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('expire', KEYS[1], ARGV[2]) "
              "else return 0 end")

    def _claim_key(self, key: str) -> str:
        return self.s._k("claim", key)

    async def claim_or_wait(self, smiles: str, conditions: dict, poll_s: float = 1.0) -> tuple[float | None, str | None]:
        """Before computing: (value, None) when the result exists or arrives while another computation of the same
        molecule runs; (None, token) when this call holds the lock and must compute, then call release(); a failure
        of the computation it waited for is raised as is. Without locks: (lookup result, None)."""
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
                self._renewers[token] = asyncio.ensure_future(self._renew(key, token))
                return None, token
            waited = True

    async def _renew(self, key: str, token: str) -> None:
        """Keep the lock alive while this computation runs (it expires soon after the agent stops)."""
        while True:
            await asyncio.sleep(max(1.0, self.cfg.claim_ttl_s / 3))
            try:
                if not self.s.r.eval(self._RENEW, 1, self._claim_key(key), f"{self.agent}|{token}",
                                     self.cfg.claim_ttl_s):
                    return
            except Exception as exc:  # noqa: BLE001 - the lock then simply expires
                log.warning("lock renewal failed for %s: %s", key, exc)

    def release(self, smiles: str, token: str | None) -> None:
        if token:
            task = self._renewers.pop(token, None)
            if task is not None:
                task.cancel()
            self.s.r.eval(self._RELEASE, 1, self._claim_key(chem.canonical(smiles)), f"{self.agent}|{token}")

    def _waited(self, key: str, t0: float, outcome: str) -> None:
        self.s.emit("waited", {"agent": self.agent, "key": key, "seconds": round(time.time() - t0, 1),
                               "outcome": outcome})

    def record_proposals(self, smiles_list: list[str]) -> None:
        keys = [chem.canonical(s) for s in smiles_list]
        self._iter_proposals = keys
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
        eid, merged = self.s.add(observation(key, QUANTITY, value, UNIT, {**conditions, "smiles_given": smiles},
                                             self.agent, prov, campaign_id))
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
    def context(self, node: str, query: str) -> str:
        """The memory section added to a node's prompt; guidance received here is marked delivered."""
        if not self.cfg.structured:
            return ""
        n = self.cfg.retrieval_limit
        parts = []
        best = sorted([e for e in self.s.all("M1") if e.data.get("value") is not None],
                      key=lambda e: -e.data["value"])[:n]
        if best:
            parts.append("Best results of the team (M1):\n" + "\n".join(f"- {_with_notes(e)}" for e in best))
        know = self.s.search("M3", query, limit=n)
        if know:
            parts.append("Team knowledge (M3), with its evidence:\n" + "\n".join(f"- {_knowledge_line(e)}"
                                                                               for e in know))
        exps = [e for e in self.s.search("M2", query, limit=3 * n) if e.kind in ("failure", "diagnosis", "outcome")][:n]
        if exps:
            parts.append("Relevant experiences (M2):\n" + "\n".join(f"- [{e.kind}] {e.text}" for e in exps))
        procs = self.s.search("M4", query, limit=n)
        if procs:
            parts.append("Procedures (M4):\n" + "\n".join(
                f"- {e.text} When: {e.data['trigger']}. Do: {e.data['actions']}. Limits: {e.data['limitations']}. "
                f"({validation_status(e)})" for e in procs))
        if self.cfg.guidance:
            gs = self.s.pending_guidance(self.agent)
            if gs:
                parts.append("Guidance from the analysis agent:\n" + "\n".join(
                    f"- {g.message} [{g.basis.replace('_', ' ')}]" for g in gs))
                for g in gs:
                    self.s.mark_delivered(g.id, self.agent, node)
        return ("\n\nSHARED MEMORY\n" + "\n\n".join(parts)) if parts else ""


def _with_notes(e) -> str:
    notes = e.data.get("notes", [])
    return e.text + (f" (WARNING: {notes[-1]['note']})" if notes else "")


def _knowledge_line(e) -> str:
    d = e.data
    return (f"{e.text} [{e.kind}, {d['epistemic_status']}; {len(d['supporting_evidence'])} result(s) for, "
            f"{len(d['contradicting_evidence'])} against" + (f"; uncertain: {d['uncertainty']}" if d.get('uncertainty')
                                                           else "") + "]")


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
