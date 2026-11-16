"""
Hybrid retrieval, done the way FALDA does it (src/falda.ts, `hybrid`), so that a comparison with FALDA differs
only in how information moves between layers, not in how it is found:

  - dense recall: cosine similarity of embeddings of the raw text (no prefix), model nomic-embed-text by default
    (FALDA_EMBED_MODEL's default), served by Ollama, an OpenAI-compatible endpoint, or hashed bag-of-words
    vectors for scripted runs;
  - lexical recall: BM25 over the same texts, only documents that match a query term (FTS5 MATCH semantics);
  - each list keeps 2 x limit documents; reciprocal-rank fusion with k = 60; the top `limit` are returned.
    limit = 10 is FALDA's default per searched collection.

Embeddings are cached in the store (shared by every process of a campaign). If the embedding service fails, the
search falls back to its lexical half and says so in the log.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os

from .textrank import scores as bm25_scores
from .textrank import tokens

logger = logging.getLogger(__name__)
RRF_K = 60          # FALDA's constant
DEFAULT_LIMIT = 10  # FALDA's default number of results per collection


class Embedder:
    def __init__(self, store=None, provider: str | None = None, model: str | None = None) -> None:
        self.store = store
        self.provider = provider or os.environ.get("EMBED_PROVIDER") or os.environ.get("LLM_PROVIDER", "ollama")
        self.model = model or os.environ.get("EMBED_MODEL", "nomic-embed-text")
        self._cache: dict[str, list[float]] = {}
        self.calls = 0

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.provider}|{self.model}|{text}".encode()).hexdigest()

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        keys = [self._key(t) for t in texts]
        missing = []
        for k, t in zip(keys, texts):
            if k in self._cache:
                continue
            hit = self.store.get("emb", k) if self.store is not None else None
            if hit:
                self._cache[k] = hit["v"]
            else:
                missing.append((k, t))
        if missing:
            try:
                vecs = self._backend([t for _, t in missing])
            except Exception as exc:  # noqa: BLE001
                logger.warning("embedding failed (%s): lexical search only", exc)
                return None
            self.calls += 1
            for (k, _), v in zip(missing, vecs):
                self._cache[k] = v
                if self.store is not None:
                    self.store.put("emb", k, {"v": v})
        return [self._cache[k] for k in keys]

    def _backend(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "scripted":
            return [hashed(t) for t in texts]
        import requests
        if self.provider == "openai":
            base = (os.environ.get("LLM_BASE_URL") or "").rstrip("/")
            r = requests.post(f"{base}/embeddings", json={"model": self.model, "input": texts},
                              headers={"Authorization": f"Bearer {os.environ.get('LLM_API_KEY', '')}"}, timeout=120)
            r.raise_for_status()
            return [d["embedding"] for d in r.json()["data"]]
        url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        r = requests.post(f"{url}/api/embed", json={"model": self.model, "input": texts}, timeout=120)
        r.raise_for_status()
        return r.json()["embeddings"]


def hashed(text: str, dims: int = 256) -> list[float]:
    """Deterministic bag-of-words vector (scripted runs and tests only)."""
    v = [0.0] * dims
    for w in tokens(text):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dims] += 1.0
    return v


def cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def hybrid(embedder: Embedder | None, query: str, docs: list[tuple[str, str]], limit: int = DEFAULT_LIMIT) -> list[str]:
    """docs: [(id, text)] -> ids of the top `limit` documents, best first."""
    if not docs:
        return []
    score: dict[str, float] = {}
    lexical = bm25_scores(query, docs)[:2 * limit]
    for i, (doc_id, _) in enumerate(lexical):
        score[doc_id] = score.get(doc_id, 0.0) + 1.0 / (RRF_K + i)
    vecs = embedder.embed([query] + [t for _, t in docs]) if embedder is not None else None
    if vecs is not None:
        q = vecs[0]
        dense = sorted(range(len(docs)), key=lambda j: -cosine(q, vecs[j + 1]))[:2 * limit]
        for i, j in enumerate(dense):
            score[docs[j][0]] = score.get(docs[j][0], 0.0) + 1.0 / (RRF_K + i)
    order = {d: n for n, (d, _) in enumerate(docs)}
    return [d for d, _ in sorted(score.items(), key=lambda kv: (-kv[1], order[kv[0]]))][:limit]


__all__ = ["Embedder", "hybrid", "RRF_K", "DEFAULT_LIMIT"]
