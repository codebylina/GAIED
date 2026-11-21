"""Settings of the shared memory and of the analysis agent, read from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass

MODES = ("off", "basic", "structured", "full")
CHANNELS = ("bm25", "semantic", "chem", "graph")
SOURCES = ("memory", "provenance")


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _list(name: str, default: tuple, allowed: tuple) -> tuple:
    v = os.environ.get(name)
    out = default if v is None else tuple(x.strip() for x in v.split(",") if x.strip())
    bad = [x for x in out if x not in allowed]
    if bad:
        raise ValueError(f"{name}: unknown value(s) {bad}, allowed {allowed}")
    return out


@dataclass
class Config:
    # off: no shared memory (original GAIED); basic: results shared between agents and the per-molecule lock;
    # structured: the graph memory written and read by the agents; full: structured + the analysis agent's guidance
    mode: str = "off"
    namespace: str = "default"
    redis_host: str = "localhost"
    redis_port: int = 6379
    campaign_id: str | None = None
    retrieval_limit: int = 5                # entries per section of a prompt (prompt size, not a scientific rule)
    channels: tuple = CHANNELS              # retrieval channels fused by RRF (pgmem/retrieval.py)
    embed_model: str = "nomic-embed-text"   # Ollama embedding model of the semantic channel
    embed_url: str = "http://localhost:11434"
    on_demand: bool = True                  # agents may call the query_memory tool at experiment selection
    memory_queries: int = 2                 # rounds of memory questions allowed before deciding
    claim_ttl_s: int = 600                  # a per-molecule lock left by a crashed agent expires after this
    analyzer_group: str = "analyzer"
    analyzer_llm: bool = True               # False: the analysis agent runs its deterministic rules only
    analysis_depth: int = 3                 # how far the analysis agent follows what rests on a node
    analysis_sources: tuple = SOURCES       # memory events and Flowcept provenance
    provenance_source: str = "auto"         # auto | stream | jsonl | none
    jsonl_glob: str = "output/flowcept_buffer_*.jsonl"

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def structured(self) -> bool:
        return self.mode in ("structured", "full")

    @property
    def guidance(self) -> bool:
        return self.mode == "full"

    @classmethod
    def from_env(cls) -> "Config":
        mode = os.environ.get("PGMEM_MODE", "off").strip().lower()
        if mode not in MODES:
            raise ValueError(f"PGMEM_MODE must be one of {MODES}, got {mode!r}")
        return cls(
            mode=mode,
            namespace=os.environ.get("PGMEM_NAMESPACE", "default"),
            redis_host=os.environ.get("REDIS_HOST", "localhost"),
            redis_port=int(os.environ.get("REDIS_PORT", "6379")),
            campaign_id=os.environ.get("FLOWCEPT_CAMPAIGN_ID"),
            retrieval_limit=int(os.environ.get("PGMEM_RETRIEVAL_LIMIT", "5")),
            channels=_list("PGMEM_CHANNELS", CHANNELS, CHANNELS),
            embed_model=os.environ.get("PGMEM_EMBED_MODEL", "nomic-embed-text"),
            embed_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            on_demand=_bool("PGMEM_ON_DEMAND", True),
            memory_queries=int(os.environ.get("PGMEM_MEMORY_QUERIES", "2")),
            claim_ttl_s=int(os.environ.get("PGMEM_CLAIM_TTL_S", "600")),
            analyzer_llm=_bool("PGMEM_ANALYZER_LLM", True),
            analysis_depth=int(os.environ.get("PGMEM_ANALYSIS_DEPTH", "3")),
            analysis_sources=_list("PGMEM_ANALYSIS_SOURCES", SOURCES, SOURCES),
            provenance_source=os.environ.get("PGMEM_PROVENANCE", "auto"),
            jsonl_glob=os.environ.get("PGMEM_JSONL_GLOB", "output/flowcept_buffer_*.jsonl"),
        )
