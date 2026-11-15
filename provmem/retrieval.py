"""
Text retrieval over a small corpus (claims, episodes, baseline memories).

Lexical ranking (BM25) is always available. Dense ranking is added when an embedding endpoint is
configured (EMBED_MODEL with Ollama's /api/embed), and the two rankings are fused with reciprocal
rank fusion. The same retriever is used by every memory condition, so retrieval quality is not a
confound between conditions.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import threading
from collections import Counter

_TOKEN = re.compile(r"[A-Za-z0-9#=()\[\]@+\-]+")
RRF_K = 60  # constant of reciprocal rank fusion (Cormack et al., 2009)


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def bm25_rank(query: str, docs: list[tuple[str, str]], k1: float = 1.2, b: float = 0.75) -> list[str]:
    """Standard Okapi BM25 parameters (Robertson and Zaragoza, 2009)."""
    q = tokens(query)
    toks = {i: tokens(t) for i, t in docs}
    n = len(docs) or 1
    avg = sum(len(t) for t in toks.values()) / n
    df = Counter(w for t in toks.values() for w in set(t))
    scores = {}
    for i, t in toks.items():
        tf = Counter(t)
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / (avg or 1)))
        scores[i] = s
    return [i for i, s in sorted(scores.items(), key=lambda x: -x[1]) if s > 0]


class Embedder:
    def __init__(self) -> None:
        self.model = os.environ.get("EMBED_MODEL", "")
        self.url = (os.environ.get("EMBED_URL") or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self._cache: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self.enabled = bool(self.model)

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        if not self.enabled:
            return None
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        missing = [t for t, k in zip(texts, keys) if k not in self._cache]
        if missing:
            try:
                import requests
                r = requests.post(f"{self.url}/api/embed", json={"model": self.model, "input": missing}, timeout=300)
                r.raise_for_status()
                vecs = r.json()["embeddings"]
            except Exception:
                return None
            with self._lock:
                for t, v in zip(missing, vecs):
                    self._cache[hashlib.sha1(t.encode()).hexdigest()] = v
        return [self._cache[k] for k in keys]


def _cos(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class Retriever:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or Embedder()

    def search(self, query: str, docs: list[tuple[str, str]], k: int) -> list[str]:
        if not docs or k <= 0:
            return []
        rankings = [bm25_rank(query, docs)]
        vecs = self.embedder.embed([query] + [t for _, t in docs]) if self.embedder.enabled else None
        if vecs:
            qv, dv = vecs[0], vecs[1:]
            sims = sorted(((docs[i][0], _cos(qv, v)) for i, v in enumerate(dv)), key=lambda x: -x[1])
            rankings.append([i for i, _ in sims])
        fused: dict[str, float] = {}
        for ranking in rankings:
            for r, doc_id in enumerate(ranking):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (RRF_K + r + 1)
        return [i for i, _ in sorted(fused.items(), key=lambda x: -x[1])][:k]
