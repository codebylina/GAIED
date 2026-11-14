"""
FALDA as the shared memory of the GAIED agents (baseline).

FALDA (https://github.com/rick-stevens-ai/falda) is used the way its harness guide
describes a "live memory provider" (docs/HARNESS_INTEGRATION.md, section 1b):
  - write each turn of the agent to the T0 Stream            (/stream/add)
  - recall with hybrid dense + BM25 search on Stream and Atoms (/stream/search, /atoms/search)
  - read the Core document                                    (/core/read)
  - distillation T0 -> T1 -> T2 -> T3 is done by FALDA's own sidecar (falda_distiller.py)

Multi-agent sharing follows docs/POOLS.md: every agent is a tenant, and all the agents of
a campaign read and write one declared shared pool (one pool per campaign, so that every
campaign starts from an empty memory, like the other memory conditions).

FALDA has no coordination between agents (no locks, no notion of a computation in
progress), no link to the measured data and no verification: this module adds none of
them, on purpose. It only translates GAIED events into FALDA calls.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_NAME_OK = re.compile(r"[^a-z0-9_-]+")


def falda_name(text: str, prefix: str) -> str:
    """A valid FALDA tenant / pool name (^[a-z0-9][a-z0-9_-]{0,63}$)."""
    name = _NAME_OK.sub("-", (text or "").lower()).strip("-")
    return f"{prefix}-{name}"[:64].rstrip("-")


def campaign_pool() -> str:
    campaign = os.environ.get("MEMORY_CAMPAIGN_ID") or os.environ.get("FLOWCEPT_CAMPAIGN_ID") or "default"
    return falda_name(campaign, "gaied")


class FaldaMemory:
    def __init__(self, url: str, tenant: str, pool: str, k_atoms: int = 8, k_stream: int = 5,
                 max_chars: int = 3000, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.tenant = tenant
        self.pool = pool
        self.k_atoms = k_atoms
        self.k_stream = k_stream
        self.max_chars = max_chars
        self.timeout = timeout
        self.last_recall_ms: float | None = None

    @classmethod
    def from_env(cls, agent_id: str) -> "FaldaMemory | None":
        """The agent's FALDA memory if MEMORY_MODE=falda and the gateway answers."""
        if os.environ.get("MEMORY_MODE", "none").strip().lower() != "falda":
            return None
        mem = cls(os.environ.get("FALDA_URL", "http://falda:8077"),
                  falda_name(agent_id.split("#")[-1], "agent"), campaign_pool(),
                  k_atoms=int(os.environ.get("FALDA_K_ATOMS", "8")),
                  k_stream=int(os.environ.get("FALDA_K_STREAM", "5")),
                  max_chars=int(os.environ.get("FALDA_MAX_CHARS", "3000")))
        try:
            mem.call("/pools/grant", {"name": mem.pool, "tenant": mem.tenant, "access": "readwrite"},
                     scoped=False)
        except Exception as exc:
            logger.warning("FALDA memory unavailable (%s): agent runs without memory", exc)
            return None
        return mem

    # ------------------------------------------------------------------ HTTP
    def call(self, route: str, payload: dict, scoped: bool = True) -> dict:
        body = dict(payload)
        if scoped:
            body.setdefault("tenant", self.tenant)
            body.setdefault("pool", self.pool)
        req = urllib.request.Request(f"{self.url}{route}", data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"FALDA {route}: HTTP {exc.code} {exc.read().decode()[:200]}") from exc

    # ------------------------------------------------------------------ campaign setup (orchestrator)
    @staticmethod
    def declare_campaign_pool(url: str, goal_text: str) -> str:
        """Declare the shared pool of the campaign and write the Goal as its Core (T3)."""
        pool = campaign_pool()
        admin = FaldaMemory(url, "orchestrator", pool)
        try:
            admin.call("/pools/declare", {"name": pool, "members": {"orchestrator": "readwrite",
                                                                    "distiller": "readwrite"},
                                          "description": "GAIED campaign shared memory"},
                       scoped=False)
        except RuntimeError as exc:
            if "exists" not in str(exc):
                raise
        admin.call("/core/write", {"content": goal_text})
        return pool

    # ------------------------------------------------------------------ write (T0 Stream)
    def write(self, role: str, content: str, node: str) -> None:
        """One turn of the agent, as FALDA's harnesses record it."""
        text = (content or "").strip()
        if not text:
            return
        try:
            self.call("/stream/add", {"session_id": self.tenant,
                                      "messages": [{"role": role,
                                                    "content": f"[{self.tenant} | {node}] {text}"}]})
        except Exception as exc:
            logger.warning("FALDA write failed: %s", exc)

    # ------------------------------------------------------------------ recall
    def recall(self, query: str) -> str:
        """Bounded memory block for a prompt: Core + top atoms + top past turns."""
        t0 = time.perf_counter()
        parts = []
        try:
            core = self.call("/core/read", {}).get("content") or ""
            if core.strip():
                parts.append("PROJECT CORE (shared memory):\n" + core.strip()[:1500])
            atoms = self.call("/atoms/search", {"query": query, "limit": self.k_atoms}).get("items", [])
            if atoms:
                parts.append("RELEVANT MEMORIES (shared memory):\n" + "\n".join(
                    f"- ({a.get('type', 'fact')}) {a.get('content', '')[:400]}" for a in atoms))
            turns = self.call("/stream/search", {"query": query, "limit": self.k_stream}).get("messages", [])
            if turns:
                parts.append("RELEVANT PAST MESSAGES OF THE TEAM (shared memory):\n" + "\n".join(
                    f"- {m.get('content', '')[:400]}" for m in turns))
        except Exception as exc:
            logger.warning("FALDA recall failed: %s", exc)
        self.last_recall_ms = (time.perf_counter() - t0) * 1000
        text = "\n\n".join(parts)
        return text[:self.max_chars]
