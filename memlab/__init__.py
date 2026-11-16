"""Harness for shared-memory experiments with multi-agent campaigns (GAIED and a synthetic benchmark).

Memory kinds: "none" (no shared memory), "mem0" (Mem0, used as designed), "mem0_verified" (Mem0 with the
knowledge verification layer) and "evidence" (the evidence-grounded layered memory, memlab/evidence_memory.py).
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
    if kind == "mem0_verified":
        from .verified_memory import VerifiedMem0Memory
        from .verifier import VerificationConfig, Verifier
        vcfg = VerificationConfig(**(kw.get("verification") or {}))
        verifier = kw.get("verifier") or Verifier(store, adapter, kw.get("llm"), vcfg)
        return VerifiedMem0Memory(store, adapter, cfg, llm=kw.get("llm"), backend=kw["backend"],
                                  top_k=kw.get("top_k"), verifier=verifier)
    if kind == "evidence":
        from .evidence_memory import EvidenceMemory
        from .registry import RegistryConfig
        from .verifier import VerificationConfig, Verifier
        ev = dict(kw.get("evidence") or {})
        reg = RegistryConfig(**ev.pop("registry", {}))
        verifier = kw.get("verifier")
        if verifier is None and ev.get("verify", True) and ev.get("knowledge", True):
            verifier = Verifier(store, adapter, kw.get("llm"), VerificationConfig(**(kw.get("verification") or {})))
        return EvidenceMemory(store, adapter, cfg, llm=kw.get("llm"), verifier=verifier, registry_cfg=reg, **ev)
    raise ValueError(f"unknown memory kind: {kind}")


__all__ = ["DomainAdapter", "Goal", "Objective", "ProvenanceTaskAdapter", "Decision", "Memory", "MemoryConfig",
           "LLM", "LLMUsage", "InMemoryStore", "RedisStore", "Store", "NoMemory", "make_memory"]
