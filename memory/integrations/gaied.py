"""
GAIED integration: one interface, three implementations, selected by MEMORY_MODE.

  none                  original GAIED (configuration 1)
  basic                 the first docker-local shared memory (memory/baseline/naive.py) (configuration 2)
  structured            the four-layer memory (configuration 3)
  structured+analysis   same agents as "structured"; the analysis service runs next to them
                        (`python -m analysis run`) (configuration 4)

The hooks in mol_design_agents.py are marked [memory] and only call this interface:
  compute(smiles, run_xtb)        simulation: reuse / wait / claim -> compute -> record
  context(node, query)            planning, tool selection, critique, plan update: memory text for the prompt
  known()                         molecules not worth proposing again
  after_proposals(smiles)         after tool selection: compliance with the guidance received
  acknowledge()                   the LLM call that received the last context completed
  learn(conclusions, llm)         conclusion: propose knowledge where justified (structured: M3 proposals
                                  evaluated by rules; basic: raw lessons)
The memory never blocks the agent: if Redis is unreachable or a memory call fails, the agent continues
without it (logged), and a simulation is never run twice because of a memory error.

Environment: MEMORY_MODE, MEMORY_NAMESPACE (default "gaied": shared across campaigns, so agents learn from
earlier executions; set one per experiment run for isolation), REDIS_HOST, MEMORY_BUDGET_TOKENS (700),
MEMORY_LESSONS (true), MEMORY_LESSON_MIN_NEW (3).
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid

log = logging.getLogger("memory.gaied")
METHOD = "xtb"
MODES = ("none", "basic", "structured", "structured+analysis")


def mode() -> str:
    m = os.environ.get("MEMORY_MODE", "none").strip().lower()
    if m not in MODES:
        log.warning("unknown MEMORY_MODE=%s, using none", m)
        return "none"
    return m


class NoMemory:
    name = "none"

    async def compute(self, smiles, run_xtb):
        return await run_xtb()

    def context(self, node, query=""):
        return ""

    def known(self):
        return []

    def after_proposals(self, smiles):
        pass

    def acknowledge(self):
        pass

    async def learn(self, conclusions, llm, results=None):
        pass


# ---------------------------------------------------------------------- configuration 2
class BasicMemory(NoMemory):
    """The docker-local prototype, wired exactly as it was (commit 0fd589b)."""
    name = "basic"

    def __init__(self, agent: str, campaign: str) -> None:
        import redis

        from memory.baseline.naive import SharedMemory as Naive
        client = redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"), decode_responses=True)
        client.ping()
        self.mem = Naive(client, campaign_id=campaign, agent_id=agent)
        client.zadd("naive:campaigns", {campaign: __import__("time").time()})
        if self.mem.get_goal() is None:
            self.mem.set_goal({"objective": "find molecules with the highest ionization energy (xTB)",
                               "constraints": ["valid SMILES", "synthesizable", "low cost"]})

    async def compute(self, smiles, run_xtb):
        return await self.mem.get_or_compute(smiles, run_xtb, new_task_id=lambda: str(uuid.uuid4()))

    def context(self, node, query=""):
        try:
            return "\n\n" + self.mem.render_for_prompt()
        except Exception as exc:
            log.warning("basic memory read failed: %s", exc)
            return ""

    def known(self):
        try:
            return self.mem.known_molecules()
        except Exception:
            return []

    async def learn(self, conclusions, llm, results=None):
        try:
            evidence = []
            for smiles, value in (results or {}).items():
                known = self.mem.get_result(smiles)
                evidence.append({"smiles": smiles, "result": value, "task_id": known.get("task_id") if known else None})
            self.mem.push_lesson(conclusions, kind="conclusion", evidence=evidence)
        except Exception as exc:
            log.warning("basic memory write failed: %s", exc)


# ---------------------------------------------------------------------- configurations 3 and 4
KNOWLEDGE_SYSTEM = """You turn a research agent's conclusions into checkable scientific claims for a shared memory.
A claim will be checked against the measurements by rules; it is only reused if the data support it.

Answer with ONLY a JSON object: {"claims": [{"statement": "...", "kind": "claim|finding|negative_finding|hypothesis|constraint",
  "claim": {...}, "evidence": ["o1", "o2"]}]}
"claim" is one of:
  {"type": "feature_effect", "feature": "<feature>", "op": ">", "value": 0, "effect": "higher"}   (or "lower")
  {"type": "comparison", "better": "<molecule>", "worse": "<molecule>"}
  {"type": "threshold", "region": {"feature": "<feature>", "op": ">", "value": 0}, "op": ">", "value": <number>}
  {"type": "negative_region", "feature": "<feature>", "op": ">", "value": 0}    (experiments there are impossible)
  {"type": "none"}                                                               (not checkable)
"evidence" lists the references (o1, o2...) of the measurements the claim is based on.
Give at most 3 claims, only what the measurements show. If nothing new can be concluded, answer {"claims": []}."""


def _first_json(text: str):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    return None


class StructuredMemory(NoMemory):
    name = "structured"

    def __init__(self, memory, xtb_params: dict, label: str = "structured") -> None:
        self.m, self.params, self.name = memory, xtb_params, label
        self.budget = int(os.environ.get("MEMORY_BUDGET_TOKENS", "700"))
        self._shown: dict = {}
        self._history: list[str] = []

    async def compute(self, smiles, run_xtb):
        import redis
        started = {"flag": False}

        async def run(task_id):
            started["flag"] = True
            return await run_xtb(task_id)
        try:
            return await self.m.get_or_compute(smiles, run, method=METHOD, params=self.params)
        except (redis.ConnectionError, redis.TimeoutError) as exc:
            if started["flag"]:
                raise
            log.warning("memory unavailable (%s): computing without it", exc)
            return await run_xtb()

    def context(self, node, query=""):
        try:
            text, self._shown = self.m.context(node, METHOD, self.params, query=query, budget_tokens=self.budget)
            return f"\n\n{text}" if text else ""
        except Exception as exc:
            log.warning("memory context failed: %s", exc)
            return ""

    def acknowledge(self):
        try:
            self.m.acknowledge(self._shown)
        except Exception:
            pass

    def known(self):
        try:
            return self.m.known_entities(METHOD, self.params, limit=150)
        except Exception:
            return []

    def after_proposals(self, smiles):
        try:
            self.m.check_guidance(smiles, METHOD, self.params, history=list(self._history))
            self._history += [self.m.results.key_of(s) for s in smiles]
        except Exception as exc:
            log.warning("guidance check failed: %s", exc)

    async def learn(self, conclusions, llm, results=None):
        """Conclusions -> checkable claims (one LLM call), only when at least MEMORY_LESSON_MIN_NEW results
        arrived since the last extraction and no other agent is extracting from the same data."""
        if os.environ.get("MEMORY_LESSONS", "true").lower() not in ("1", "true", "yes"):
            return
        from langchain.messages import HumanMessage, SystemMessage

        from memory.domain import condition_id
        m = self.m
        cond = condition_id(METHOD, self.params)
        try:
            n = m.results.count(cond)
            done = int(m.store.r.get(m.store.k("extract", cond)) or 0)
            if n - done < int(os.environ.get("MEMORY_LESSON_MIN_NEW", "3")):
                m.store.metric("extractions.skipped_no_new_data")
                return
            token = m.coord.claim(f"_extract/{cond}", m.agent, ttl_s=900)
            if token is None:
                m.store.metric("extractions.skipped_busy")
                return
        except Exception as exc:
            log.warning("knowledge extraction skipped: %s", exc)
            return
        ok = False
        try:
            recs = m.results.in_condition(cond, limit=40)
            refs, lines = {}, []
            for i, r in enumerate(recs, 1):
                refs[f"o{i}"] = r.id
                lines.append(f"o{i} | {r.entity} | " + (f"{r.value:.6g}" if r.outcome == "success" else "NEGATIVE"))
            user = (f"OBJECTIVE: {m.domain.objective} (higher is better)\nMEASUREMENTS (ref | molecule | value):\n"
                    + "\n".join(lines) + f"\n\nFEATURES: {m.domain.feature_doc()}\n\nCONCLUSIONS:\n{conclusions}")
            answer = await llm.ainvoke([SystemMessage(content=KNOWLEDGE_SYSTEM), HumanMessage(content=user)])
            data = _first_json(answer.content) or {}
            for c in (data.get("claims") or [])[:3] if isinstance(data, dict) else []:
                if not isinstance(c, dict) or not str(c.get("statement", "")).strip():
                    continue
                evidence = [refs.get(str(e).strip(), str(e)) for e in c.get("evidence") or []] or list(refs.values())
                kind = c.get("kind") if c.get("kind") in ("claim", "finding", "negative_finding", "hypothesis",
                                                          "constraint") else "claim"
                k, created, _ = m.propose_knowledge(str(c["statement"]), kind=kind, claim=c.get("claim") or None,
                                                    method=METHOD, params=self.params, supporting=evidence)
                log.warning("memory knowledge %s (%s): %s", k.id, k.status, k.statement[:80])
            ok = True
            m.store.metric("extractions.done")
        except Exception as exc:
            log.warning("knowledge extraction failed: %s", exc)
        finally:
            if ok:
                m.store.r.set(m.store.k("extract", cond), n)
            m.coord.release(f"_extract/{cond}", token)


# ---------------------------------------------------------------------- factory
def for_agent(agent: str, xtb_params: dict, campaign: str | None, sink=None, prov_context=None,
              workflows: dict | None = None):
    """The memory of one GAIED agent for the configured MEMORY_MODE (NoMemory on any setup error)."""
    m = mode()
    try:
        if m == "none":
            return NoMemory()
        if m == "basic":
            return BasicMemory(agent, os.environ.get("MEMORY_NAMESPACE") or campaign or "default")
        from memory import SharedMemory
        from memory.domains.chemistry import ChemistryDomain
        mem = SharedMemory.connect(ChemistryDomain(default_params=xtb_params),
                                   os.environ.get("MEMORY_NAMESPACE", "gaied"), agent,
                                   host=os.environ.get("REDIS_HOST", "localhost"),
                                   provenance_sink=sink, prov_context=prov_context)
        mem.register_agent(METHOD, xtb_params, workflows, campaign)
        return StructuredMemory(mem, xtb_params, label=m)
    except Exception as exc:
        log.warning("MEMORY_MODE=%s unavailable (%s): running without shared memory", m, exc)
        return NoMemory()


def report(xtb_params: dict, campaign: str | None) -> None:
    """End-of-campaign summary printed by run-07.py."""
    m = mode()
    if m in ("none", "basic"):
        return
    try:
        from memory import SharedMemory
        from memory.domains.chemistry import ChemistryDomain
        mem = SharedMemory.connect(ChemistryDomain(default_params=xtb_params),
                                   os.environ.get("MEMORY_NAMESPACE", "gaied"), "orchestrator",
                                   host=os.environ.get("REDIS_HOST", "localhost"))
        q = mem.report()
        print("=" * 80)
        print(f"Shared memory '{mem.namespace}' ({m}):")
        for L in ("M1", "M2", "M3", "M4"):
            print(f"  {L}: {q[L]['active']} active {q[L]['by_status']}")
        dec = {k: v for k, v in q["metrics"].items() if k.startswith("decisions.")}
        print(f"  decisions: {dec}")
    except Exception as exc:
        print(f"[memory] report failed: {exc}")
