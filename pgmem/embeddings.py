"""
Embeddings for the semantic retrieval channel: an Ollama embedding model (PGMEM_EMBED_MODEL, at OLLAMA_URL), with the
vector of each entry cached in Redis (pgmem:{ns}:emb). With LLM_PROVIDER=scripted (tests without any model) a
deterministic bag-of-words vector stands in. When the model cannot be reached, the channel is switched off for the
process (one warning) and retrieval goes on with the other channels.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re

import requests

log = logging.getLogger("pgmem.embeddings")
_WORD = re.compile(r"[a-z0-9]+")


def _scripted(text: str, dim: int = 256) -> list[float]:
    v = [0.0] * dim
    for w in _WORD.findall((text or "").lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class Embedder:
    def __init__(self, store, model: str, url: str) -> None:
        self.s, self.model, self.url = store, model, url.rstrip("/")
        self.scripted = os.environ.get("LLM_PROVIDER", "").lower() == "scripted"
        self.available = bool(model)

    def _embed(self, texts: list[str]) -> list[list[float]] | None:
        if self.scripted:
            return [_scripted(t) for t in texts]
        if not self.available:
            return None
        try:
            resp = requests.post(f"{self.url}/api/embed", json={"model": self.model, "input": texts}, timeout=120)
            resp.raise_for_status()
            vecs = resp.json().get("embeddings")
            if not isinstance(vecs, list) or len(vecs) != len(texts):
                raise ValueError("unexpected answer")
            return vecs
        except (requests.RequestException, ValueError) as exc:
            self.available = False
            log.warning("embedding model %s unavailable (%s): semantic retrieval is off for this process",
                        self.model, exc)
            return None

    def query(self, text: str) -> list[float] | None:
        got = self._embed([text]) if text.strip() else None
        return got[0] if got else None

    def entries(self, entries: list) -> dict[str, list[float]]:
        """Vectors of entries (id -> vector), computed once per entry and cached."""
        key = self.s._k("emb")
        cached = dict(zip([e.id for e in entries], self.s.r.hmget(key, [e.id for e in entries]))) if entries else {}
        out = {i: json.loads(v) for i, v in cached.items() if v}
        todo = [e for e in entries if e.id not in out]
        if todo:
            vecs = self._embed([e.text for e in todo])
            if vecs is None:
                return out
            for e, v in zip(todo, vecs):
                out[e.id] = v
            self.s.r.hset(key, mapping={e.id: json.dumps(v) for e, v in zip(todo, vecs)})
        return out
