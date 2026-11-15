"""
Glue between GAIED (Academy + LangGraph agents, Flowcept provenance) and the memory conditions.

Environment:
  MEMORY_CONDITION   a condition of MEMORY_CONFIG: none, mem0, mem0_verified, mem0_textcheck,
                     mem0_verified_nolayers, mem0_verified_posthoc, mem0_verified_agentindep. Unset: original GAIED, nothing recorded.
  MEMORY_CONFIG      experiment configuration with memory_defaults (default bench/configs/main.json)
  MEMORY_CAMPAIGN_ID namespace of the measurement logs in Redis (default FLOWCEPT_CAMPAIGN_ID)
  MEM0_URL           the Mem0 service started by run-07.py (default http://127.0.0.1:8765)
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
    path = os.environ.get("MEMORY_CONFIG", os.path.join(ROOT, "bench", "configs", "main.json"))
    for c in json.load(open(path))["conditions"]:
        if c["name"] == name:
            return c
    raise ValueError(f"MEMORY_CONDITION={name} is not a condition of {path}")


def verifier_from_env(store, llm=None):
    from memlab.verifier import VerificationConfig, Verifier
    return Verifier(store, adapter_from_env(), llm, VerificationConfig(**(condition_spec().get("verification") or {})))


def defaults() -> dict:
    path = os.environ.get("MEMORY_CONFIG", os.path.join(ROOT, "bench", "configs", "main.json"))
    return json.load(open(path)).get("memory_defaults", {})


def campaign() -> str:
    return os.environ.get("MEMORY_CAMPAIGN_ID") or os.environ.get("FLOWCEPT_CAMPAIGN_ID") or "default"


def adapter_from_env() -> ChemistryAdapter:
    heavy = os.environ.get("GOAL_MAX_HEAVY_ATOMS")
    elements = os.environ.get("GOAL_ALLOWED_ELEMENTS")
    return ChemistryAdapter(os.environ.get("GOAL_TEXT", DEFAULT_GOAL),
                            max_heavy_atoms=int(heavy) if heavy else None,
                            allowed_elements=[e.strip() for e in elements.split(",")] if elements else None)


def memory_from_env(backend=None, verifier=None):
    """The memory an agent uses (backend=None: the Mem0 service over HTTP)."""
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
    raise ValueError(f"MEMORY_CONDITION={cond}: unknown memory kind {kind}")


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
