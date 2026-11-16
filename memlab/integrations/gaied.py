"""
Glue between GAIED (Academy + LangGraph agents, Flowcept provenance) and the memory conditions.

Environment:
  MEMORY_CONDITION   a condition of MEMORY_CONFIG (e.g. none, mem0, mem0_verified, registry_only,
                     evidence_unverified, evidence). Unset: original GAIED, nothing recorded.
  MEMORY_CONFIG      experiment configuration with memory_defaults (default bench/configs/evidence_main.json)
  MEMORY_CAMPAIGN_ID namespace of the measurement logs in Redis (default FLOWCEPT_CAMPAIGN_ID)
  MEM0_URL           the Mem0 service started by run-07.py (default http://127.0.0.1:8765)
  MEMORY_WAIT_S      longest wait for a molecule another agent is computing (default 900; the wait ends
                     earlier when the result arrives or when the other agent's reservation lapses)
  GOAL_TEXT, GOAL_MAX_HEAVY_ATOMS, GOAL_ALLOWED_ELEMENTS   the Goal (optional constraints)
"""
from __future__ import annotations

import json
import logging
import os

from memlab import MemoryConfig, RedisStore, make_memory
from memlab.adapters.chemistry import ChemistryAdapter

logger = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_GOAL = "Find molecules with high ionization energy, computed with xTB."
DEFAULT_MEM0_PORT = 8765


def condition() -> str | None:
    return os.environ.get("MEMORY_CONDITION") or None


def condition_spec() -> dict:
    """The condition's entry in MEMORY_CONFIG (memory kind and verification settings)."""
    name = condition()
    path = os.environ.get("MEMORY_CONFIG", os.path.join(ROOT, "bench", "configs", "evidence_main.json"))
    for c in json.load(open(path))["conditions"]:
        if c["name"] == name:
            return c
    raise ValueError(f"MEMORY_CONDITION={name} is not a condition of {path}")


def verifier_from_env(store, llm=None):
    from memlab.verifier import VerificationConfig, Verifier
    return Verifier(store, adapter_from_env(), llm, VerificationConfig(**(condition_spec().get("verification") or {})))


def defaults() -> dict:
    path = os.environ.get("MEMORY_CONFIG", os.path.join(ROOT, "bench", "configs", "evidence_main.json"))
    return json.load(open(path)).get("memory_defaults", {})


def campaign() -> str:
    return os.environ.get("MEMORY_CAMPAIGN_ID") or os.environ.get("FLOWCEPT_CAMPAIGN_ID") or "default"


def adapter_from_env() -> ChemistryAdapter:
    heavy = os.environ.get("GOAL_MAX_HEAVY_ATOMS")
    elements = os.environ.get("GOAL_ALLOWED_ELEMENTS")
    return ChemistryAdapter(os.environ.get("GOAL_TEXT", DEFAULT_GOAL),
                            max_heavy_atoms=int(heavy) if heavy else None,
                            allowed_elements=[e.strip() for e in elements.split(",")] if elements else None)


def memory_from_env(backend=None, verifier=None, llm=None):
    """The memory an agent uses (backend=None: the Mem0 service over HTTP). With llm, the evidence memory
    also distills and verifies (run-07.py does this in one process; agents only read)."""
    cond = condition()
    if not cond:
        return None
    try:
        store = RedisStore.connect(campaign())
    except Exception as exc:
        logger.warning("memory disabled (Redis unreachable): %s", exc)
        return None
    cfg = MemoryConfig(**defaults())
    if cond == "none":
        cfg.briefing_budget_chars = 0
        return make_memory("none", store, adapter_from_env(), cfg)
    kind = condition_spec()["memory"]
    if kind in ("mem0", "mem0_verified"):
        if backend is None:
            from memlab.mem0_memory import Mem0Client
            backend = Mem0Client(os.environ.get("MEM0_URL", f"http://127.0.0.1:{DEFAULT_MEM0_PORT}"))
        kw = {"backend": backend}
        if kind == "mem0_verified":   # agents only read statuses; the verifier runs in run-07.py
            kw["verifier"] = verifier
            if verifier is None:
                kw["verifier"] = verifier_from_env(store)
        return make_memory(kind, store, adapter_from_env(), cfg, **kw)
    if kind == "evidence":
        spec = condition_spec()
        return make_memory("evidence", store, adapter_from_env(), cfg, llm=llm, verifier=verifier,
                           evidence=spec.get("evidence"), verification=spec.get("verification"))
    raise ValueError(f"MEMORY_CONDITION={cond}: unknown memory kind {kind}")


async def gated_compute(memory, agent: str, entity: str, compute):
    """Run compute() unless the experiment registry says not to (evidence memory only).

    already measured   -> the measured value, no computation
    failed before      -> RuntimeError with the recorded cause (the agent sees it as the result)
    in progress        -> wait for the other agent's result (or for its reservation to lapse)
    unknown            -> compute, holding a reservation renewed while it runs
    """
    import asyncio
    if memory is None or not hasattr(memory, "claim"):
        return await compute()
    reg = memory.registry
    for _ in range(2):
        d = memory.claim(agent, entity)
        memory.log_decisions(agent, -1, [d])      # -1: GAIED does not pass the iteration here
        if d.action == "known":
            return d.value
        if d.action == "failed_before":
            v = reg.view(d.entity)
            raise RuntimeError(f"not run: {entity} already failed ({v.get('state')}: {v.get('error') or 'no message'}, "
                               f"by {d.owner})")
        if d.action != "in_progress":
            break
        got = await asyncio.to_thread(memory.await_results, agent, [d.entity],
                                      float(os.environ.get("MEMORY_WAIT_S", "900")))
        r = got.get(d.entity)
        if r and r["status"] == "ok":
            return r["value"]
        if r:
            raise RuntimeError(f"not run: {entity} failed for {d.owner} ({r.get('cause')})")
    stop = asyncio.Event()

    async def renew():
        while not stop.is_set():
            await asyncio.sleep(reg.cfg.lease_ttl_s / 3)
            reg.renew(agent, d.entity)

    keeper = asyncio.create_task(renew())
    try:
        return await compute()
    finally:
        stop.set()
        keeper.cancel()
        memory.release(agent, d.entity)


def block(memory, agent: str, node: str, query: str, iteration: int) -> str:
    """Memory text appended to a prompt ('' without memory)."""
    if memory is None:
        return ""
    try:
        text = memory.briefing(agent, node, query, iteration)
    except Exception as exc:
        logger.warning("memory briefing failed: %s", exc)
        return ""
    return f"\n\nSHARED MEMORY OF THE TEAM:\n{text}" if text else ""
