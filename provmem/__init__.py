"""ProvMem: shared memory for multi-agent experimental campaigns, grounded in workflow provenance."""
from __future__ import annotations

from .adapter import DomainAdapter, Goal, Objective, ProvenanceTaskAdapter
from .base import Decision, Memory, MemoryConfig
from .baselines import REGISTRY as BASELINES
from .llm import LLM, LLMUsage
from .memory import ProvMem
from .retrieval import Retriever
from .store import InMemoryStore, RedisStore, Store


def make_memory(kind: str, store: Store, adapter: DomainAdapter, cfg: MemoryConfig,
                llm: LLM | None = None, retriever: Retriever | None = None, **kw) -> Memory:
    if kind == "provmem":
        return ProvMem(store, adapter, cfg, llm=llm, retriever=retriever, **kw)
    if kind in BASELINES:
        return BASELINES[kind](store, adapter, cfg, llm=llm, retriever=retriever, **kw)
    raise ValueError(f"unknown memory kind: {kind}")


__all__ = ["DomainAdapter", "Goal", "Objective", "ProvenanceTaskAdapter", "Decision", "Memory", "MemoryConfig",
           "LLM", "LLMUsage", "ProvMem", "Retriever", "InMemoryStore", "RedisStore", "Store", "make_memory"]
