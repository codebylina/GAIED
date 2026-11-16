"""
Lexical relevance (Okapi BM25 with its usual constants) to order memory items for a query.
Retrieval is not what the evidence memory studies; this keeps it simple, local and deterministic.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[A-Za-z0-9_#=()\[\]@+\-]+")
_K1, _B = 1.2, 0.75


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def rank(query: str, docs: list[tuple[str, str]]) -> list[str]:
    """docs: [(id, text)] -> ids, most relevant first (ties keep the input order)."""
    return [i for i, _ in _scored(query, docs)]


def scores(query: str, docs: list[tuple[str, str]]) -> list[tuple[str, float]]:
    """Only the documents that contain a query term, best first (as an FTS5 MATCH would return them)."""
    return [(i, s) for i, s in _scored(query, docs) if s > 0]


def _scored(query: str, docs: list[tuple[str, str]]) -> list[tuple[str, float]]:
    if not docs:
        return []
    toks = [tokens(t) for _, t in docs]
    n = len(docs)
    avg = sum(len(t) for t in toks) / n or 1.0
    df = Counter(w for t in toks for w in set(t))
    q = set(tokens(query))
    out = []
    for i, t in enumerate(toks):
        tf = Counter(t)
        s = 0.0
        for w in q:
            if w in tf:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (_K1 + 1) / (tf[w] + _K1 * (1 - _B + _B * len(t) / avg))
        out.append((-s, i))
    return [(docs[i][0], -s) for s, i in sorted(out)]
