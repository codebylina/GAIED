"""
Glue between GAIED (Academy + LangGraph agents, xTB, Flowcept) and the shared memory. Everything GAIED-specific
about the memory is here; the hooks in mol_design_agents.py are marked [memory].

Environment:
  MEMORY           "true" to enable the shared memory (unset / "false": original GAIED, the baseline)
  MEMORY_NAMESPACE    memory namespace in Redis. Default: the Flowcept campaign id (one memory per run).
                   Set a fixed name (e.g. "gaied") to accumulate knowledge across campaigns.
  MEMORY_LESSONS      "true" (default): conclusions are turned into lessons (one extra LLM call per iteration)
  MEMORY_CLAIM_TTL    seconds before the claim of a silent (crashed) agent expires (default 300)
  MEMORY_BRIEFING_CHARS  size limit of the memory text added to a prompt (default 2500)
  REDIS_HOST       Redis host (same instance as Academy)
"""
from __future__ import annotations

import logging
import os
from typing import Callable

from memory import Memory
from memory.domains.chemistry import ChemistryDomain

logger = logging.getLogger(__name__)
METHOD = "xtb"


def enabled() -> bool:
    return os.environ.get("MEMORY", "false").lower() in ("1", "true", "yes")


def from_env(agent: str, xtb_params: dict, sink: Callable[[dict], None] | None = None,
             context: Callable[[], dict] | None = None) -> Memory | None:
    """The agent's memory, or None when disabled or when Redis is unreachable (GAIED then runs as the
    original, without shared memory)."""
    if not enabled():
        return None
    namespace = os.environ.get("MEMORY_NAMESPACE") or os.environ.get("FLOWCEPT_CAMPAIGN_ID") or "default"
    try:
        return Memory.connect(ChemistryDomain(default_params=xtb_params), namespace,
                           host=os.environ.get("REDIS_HOST", "localhost"), agent=agent,
                           claim_ttl_s=float(os.environ.get("MEMORY_CLAIM_TTL", "300")),
                           provenance_sink=sink, context=context)
    except Exception as exc:
        logger.warning("Memory disabled (%s)", exc)
        return None


def briefing(memory: Memory | None, xtb_params: dict, provenance: dict | None = None, node: str | None = None) -> str:
    """Text appended to a prompt ('' without memory or on error: the memory never breaks the agent)."""
    if memory is None:
        return ""
    try:
        text = memory.briefing(METHOD, xtb_params, max_chars=int(os.environ.get("MEMORY_BRIEFING_CHARS", "2500")),
                               provenance=provenance, label=node)
        return f"\n\n{text}" if text else ""
    except Exception as exc:
        logger.warning("Memory briefing failed: %s", exc)
        return ""


def known_molecules(memory: Memory | None, xtb_params: dict, limit: int = 100) -> list[str]:
    if memory is None:
        return []
    try:
        return memory.known_entities(METHOD, xtb_params, limit)
    except Exception as exc:
        logger.warning("Memory read failed: %s", exc)
        return []


async def learn(memory: Memory | None, llm, conclusions: str, xtb_params: dict,
                provenance: dict | None = None) -> list[dict]:
    """Turn the conclusions of an iteration into lessons (LLM), then verify them (rules).
    `llm` is the agent's own client (anything with `await llm.ainvoke(messages)`)."""
    if memory is None or os.environ.get("MEMORY_LESSONS", "true").lower() not in ("1", "true", "yes"):
        return []
    from langchain.messages import HumanMessage, SystemMessage
    try:
        system, user, refs = memory.lesson_prompt(conclusions, METHOD, xtb_params)
        if not refs:
            return []
        answer = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
        lessons = memory.propose_from_llm(answer.content, refs, provenance=provenance)
        verdicts = dict(memory.verify_pending())
        for l in lessons:
            logger.warning("Memory lesson %s: %s -> %s", l["id"], l["statement"][:80], verdicts.get(l["id"], "queued"))
        return lessons
    except Exception as exc:
        logger.warning("Memory lesson step failed: %s", exc)
        return []


def print_report(xtb_params: dict) -> None:
    """End-of-campaign summary (called by run-07.py)."""
    memory = from_env("orchestrator", xtb_params)
    if memory is None:
        return
    try:
        memory.verify_pending()
        rep = memory.report()
        print("=" * 80)
        print(f"Shared memory '{rep['namespace']}': {rep['observations']} observations, lessons {rep['lessons']}")
        print("counters: " + (", ".join(f"{k}={v}" for k, v in sorted(rep["stats"].items())) or "none"))
        print(f"details: python -m memory show {rep['namespace']}   |   python -m memory lessons {rep['namespace']}")
    except Exception as exc:
        print(f"[Memory] report failed: {exc}")
