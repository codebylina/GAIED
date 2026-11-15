"""Harness for shared-memory experiments with multi-agent campaigns (GAIED and a synthetic benchmark).

Conditions on this branch: "none" (no shared memory) and "mem0" (Mem0, used as designed).
"""
from __future__ import annotations

from .adapter import DomainAdapter, Goal, Objective, ProvenanceTaskAdapter
from .base import Decision, Memory, MemoryConfig
from .llm import LLM, LLMUsage
from .store import InMemoryStore, RedisStore, Store


class NoMemory(Memory):
    """No shared memory: each agent sees only its own history (kept by the agent itself)."""
    name = "none"


def make_memory(kind: str, store: Store, adapter: DomainAdapter, cfg: MemoryConfig, **kw) -> Memory:
    if kind == "none":
        return NoMemory(store, adapter, cfg, llm=kw.get("llm"))
    if kind == "mem0":
        from .mem0_memory import Mem0Memory
        return Mem0Memory(store, adapter, cfg, llm=kw.get("llm"), backend=kw["backend"], top_k=kw.get("top_k"))
    raise ValueError(f"unknown memory kind: {kind}")


__all__ = ["DomainAdapter", "Goal", "Objective", "ProvenanceTaskAdapter", "Decision", "Memory", "MemoryConfig",
           "LLM", "LLMUsage", "InMemoryStore", "RedisStore", "Store", "NoMemory", "make_memory"]
