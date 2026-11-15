"""
Glue between GAIED (Academy + LangGraph agents, Flowcept provenance) and the memory.

Environment:
  MEMORY_CONDITION   name of a condition of the experiment configuration (none, shared_log, rag, mem0,
                     provmem, provmem_no_layers, ...). Unset: no memory at all (original GAIED).
  MEMORY_CONFIG      experiment configuration holding the conditions and memory_defaults
                     (default bench/configs/main.json, so GAIED and the benchmark use the same settings)
  MEMORY_CAMPAIGN_ID namespace of the memory in Redis (default FLOWCEPT_CAMPAIGN_ID)
  GOAL_TEXT, GOAL_MAX_HEAVY_ATOMS, GOAL_ALLOWED_ELEMENTS   the Goal (optional constraints)
"""
from __future__ import annotations

import json
import logging
import os

from provmem import LLM, MemoryConfig, RedisStore, make_memory
from provmem.adapters.chemistry import ChemistryAdapter

logger = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_GOAL = "Find molecules with high ionization energy, computed with xTB."


def condition() -> tuple[dict, dict] | None:
    name = os.environ.get("MEMORY_CONDITION")
    if not name:
        return None
    path = os.environ.get("MEMORY_CONFIG", os.path.join(ROOT, "bench", "configs", "main.json"))
    cfg = json.load(open(path))
    for c in cfg["conditions"]:
        if c["name"] == name:
            return c, cfg.get("memory_defaults", {})
    raise ValueError(f"MEMORY_CONDITION={name} is not a condition of {path}")


def adapter_from_env() -> ChemistryAdapter:
    heavy = os.environ.get("GOAL_MAX_HEAVY_ATOMS")
    elements = os.environ.get("GOAL_ALLOWED_ELEMENTS")
    return ChemistryAdapter(os.environ.get("GOAL_TEXT", DEFAULT_GOAL),
                            max_heavy_atoms=int(heavy) if heavy else None,
                            allowed_elements=[e.strip() for e in elements.split(",")] if elements else None)


def memory_from_env():
    found = condition()
    if not found:
        return None
    cond, defaults = found
    namespace = os.environ.get("MEMORY_CAMPAIGN_ID") or os.environ.get("FLOWCEPT_CAMPAIGN_ID") or "default"
    try:
        store = RedisStore.connect(namespace)
    except Exception as exc:
        logger.warning("memory disabled (Redis unreachable): %s", exc)
        return None
    cfg = MemoryConfig(**{**defaults, **cond.get("cfg", {})})
    llm = LLM() if (os.environ.get("LLM_MODEL") or os.environ.get("OLLAMA_MODEL")) else None
    return make_memory(cond["memory"], store, adapter_from_env(), cfg, llm=llm)


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
