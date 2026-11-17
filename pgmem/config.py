"""Settings of the shared memory and of the analysis agent, read from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

MODES = ("off", "basic", "structured", "full")


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    # off: no shared memory (original GAIED); basic: results shared between agents, nothing else;
    # structured: M1-M4 written and read by the agents; full: structured + guidance from the analysis agent.
    mode: str = "off"
    namespace: str = "default"              # one persistent memory per namespace, kept across campaigns
    redis_host: str = "localhost"
    redis_port: int = 6379
    campaign_id: str | None = None
    retrieval_limit: int = 5                # entries per category put in a prompt (prompt size, not a scientific rule)
    reuse_results: bool = True              # experiment selection: reuse a result already in M1 instead of recomputing
    analyzer_group: str = "analyzer"
    analyzer_llm: bool = True               # False: the analysis agent runs its deterministic rules only
    provenance_source: str = "auto"         # auto | stream | jsonl | none
    jsonl_glob: str = "output/flowcept_buffer_*.jsonl"
    extra: dict = field(default_factory=dict)

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
            reuse_results=_bool("PGMEM_REUSE_RESULTS", True),
            analyzer_llm=_bool("PGMEM_ANALYZER_LLM", True),
            provenance_source=os.environ.get("PGMEM_PROVENANCE", "auto"),
            jsonl_glob=os.environ.get("PGMEM_JSONL_GLOB", "output/flowcept_buffer_*.jsonl"),
        )
