"""
Hybrid retrieval from the memory graph, for the sections added to the agents' prompts (scheduled) and for the
questions they ask themselves (query_memory, on demand).

A query is the current task: its text, the molecules it is about (named in the text, the agent's seed and recent
proposals) and the failure classes the agent met recently. Up to four channels rank the candidate entries of a
category, each on its own (PGMEM_CHANNELS):
  bm25      keyword search (BM25) over the entries' words: exact names, SMILES, error classes
  semantic  embeddings (pgmem/embeddings.py), on the natural-language entries (M2, M3, M4)
  chem      structural similarity (domain similarity: Tanimoto) between the task's molecules and the entry's
  graph     the graph around the task's molecules: entries ABOUT one of them (distance 1), ABOUT one of their motifs
            (2), ABOUT another molecule with one of their motifs (3)
The rankings are fused by Reciprocal Rank Fusion (score = sum over channels of 1 / (RRF_K + rank)): no weights to
tune. Only the domain channel (chem) and the motif nodes of the graph are specific to chemistry.

Retrieval is followed by completion: each entry is shown with its direct neighbors in the graph (describe()): the
evidence and verdict of a statement, the warning on an unreliable result, the history of a procedure.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .domain import DOMAIN
from .embeddings import Embedder, cosine
from .graph import EVIDENCE
from .schema import Entry, molecule_node, motif_node
from .store import MemoryStore

RRF_K = 60                      # the usual constant of Reciprocal Rank Fusion
CHANNEL_DEPTH = 20              # entries each channel contributes to the fusion
SIMILAR = 0.4                   # Tanimoto similarity from which two molecules are "similar" (applicability)
_WORD = re.compile(r"[A-Za-z0-9_\-\[\]()=#@+\\/.]+")


def words(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text or "") if len(t) > 1]


@dataclass
class Query:
    text: str = ""
    molecules: list[str] = field(default_factory=list)
    failure_classes: set[str] = field(default_factory=set)

    @property
    def motifs(self) -> set[str]:
        return set().union(*(DOMAIN.motifs(m) for m in self.molecules)) if self.molecules else set()


def build_query(text: str = "", molecules: list[str] | None = None, failure_classes: set[str] | None = None) -> Query:
    mols: list[str] = []
    for m in list(molecules or []) + DOMAIN.entities_in(text):
        k = DOMAIN.canonical(m)
        if k and DOMAIN.valid(k) and k not in mols:
            mols.append(k)
    return Query(text=text, molecules=mols[:20], failure_classes=set(failure_classes or ()))


def _document(e: Entry) -> str:
    d = e.data
    extra = [d.get(k, "") for k in ("scope", "trigger", "actions", "failure_class", "rule", "uncertainty")]
    return " ".join([e.text, e.key] + [str(x) for x in extra if x])


# ---------------------------------------------------------------------------------------------- channels
def bm25(q: Query, cands: list[Entry], k1: float = 1.5, b: float = 0.75) -> list[str]:
    terms = set(words(q.text)) | {m.lower() for m in q.molecules}
    if not terms or not cands:
        return []
    docs = {e.id: words(_document(e)) for e in cands}
    avg = sum(len(d) for d in docs.values()) / len(docs) or 1.0
    df = Counter(t for d in docs.values() for t in set(d))
    n, scores = len(docs), {}
    for i, d in docs.items():
        tf, s = Counter(d), 0.0
        for t in terms & set(d):
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg))
        if s > 0:
            scores[i] = s
    return sorted(scores, key=lambda i: -scores[i])


def semantic(q: Query, cands: list[Entry], emb: Embedder) -> list[str]:
    texts = [e for e in cands if e.category in ("M2", "M3", "M4")]
    qv = emb.query(q.text) if texts else None
    if qv is None:
        return []
    vecs = emb.entries(texts)
    scores = {i: cosine(qv, v) for i, v in vecs.items()}
    return sorted((i for i in scores if scores[i] > 0), key=lambda i: -scores[i])


def chem_similarity(q: Query, cands: list[Entry], store: MemoryStore) -> list[str]:
    if not q.molecules:
        return []
    scores = {}
    for e in cands:
        mols = store.molecules_of(e.id)
        s = max((DOMAIN.similarity(a, m) for a in q.molecules for m in mols), default=0.0)
        if s > 0:
            scores[e.id] = s
    return sorted(scores, key=lambda i: -scores[i])


def graph_neighborhood(q: Query, cands: list[Entry], store: MemoryStore) -> list[str]:
    g, ids, dist = store.graph, {e.id for e in cands}, {}

    def reach(node: str, d: int) -> None:
        for edge in g.incoming(node, ["ABOUT"]):
            if edge["source"] in ids and edge["source"] not in dist:
                dist[edge["source"]] = d
    for m in q.molecules:
        reach(molecule_node(m), 1)
    for motif in q.motifs:
        reach(motif_node(motif), 2)
    own = {molecule_node(m) for m in q.molecules}
    for motif in q.motifs:
        for edge in g.incoming(motif_node(motif), ["HAS_MOTIF"])[:50]:
            if edge["source"] not in own:
                reach(edge["source"], 3)
    created = {e.id: e.created_at for e in cands}
    return sorted(dist, key=lambda i: (dist[i], -created[i]))


def retrieve(store: MemoryStore, category: str, q: Query, limit: int, channels: tuple, emb: Embedder | None = None,
             kinds: tuple | None = None, keep=None, scan: int = 500) -> list[tuple[Entry, float, dict]]:
    """The `limit` best entries of a category for the query: (entry, fused score, rank in each channel)."""
    cands = [e for e in store.recent(category, scan)
             if (kinds is None or e.kind in kinds) and (keep is None or keep(e))]
    if not cands:
        return []
    rankings = {}
    if "bm25" in channels:
        rankings["bm25"] = bm25(q, cands)
    if "semantic" in channels and emb is not None:
        rankings["semantic"] = semantic(q, cands, emb)
    if "chem" in channels:
        rankings["chem"] = chem_similarity(q, cands, store)
    if "graph" in channels:
        rankings["graph"] = graph_neighborhood(q, cands, store)
    fused: dict[str, float] = {}
    ranks: dict[str, dict] = {}
    for ch, ranking in rankings.items():
        for r, i in enumerate(ranking[:CHANNEL_DEPTH], 1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r)
            ranks.setdefault(i, {})[ch] = r
    by_id = {e.id: e for e in cands}
    best = sorted(fused, key=lambda i: (-fused[i], -by_id[i].created_at))[:limit]
    return [(by_id[i], fused[i], ranks[i]) for i in best]


def applicable(proc: Entry, q: Query, store: MemoryStore) -> str | None:
    """Why a procedure applies to the current task, or None. A procedure for a motif applies when the task's
    molecules carry that motif or resemble the molecules it was derived from; a procedure for a failure class on
    any molecule applies when the agent met that class recently."""
    motif = proc.data.get("motif")
    if motif is None:
        fc = proc.data.get("failure_class")
        return f"{fc} failures occurred recently" if fc in q.failure_classes else None
    if motif in q.motifs:
        return f"the molecules at hand have a {motif.replace('_', ' ')} group"
    failed = [m for e in store.graph.outgoing(proc.id, ["DERIVED_FROM"]) for m in store.molecules_of(e["target"])]
    sim = max((DOMAIN.similarity(a, b) for a in q.molecules for b in failed), default=0.0)
    if sim >= SIMILAR:
        return f"the molecules at hand resemble those that failed ({sim:.2f} similarity)"
    return None


# ---------------------------------------------------------------------------------------------- completion
def task_of(e: Entry) -> str | None:
    return next((r.get("task_id") for r in e.prov_refs if r.get("task_id")), None)


def unreliable(store: MemoryStore, e: Entry) -> str | None:
    """The data-quality problem of a result, if any: its own conditions, or the analysis agent's diagnosis."""
    if e.data.get("conditions", {}).get("converged") is False:
        return "geometry optimization did not converge"
    for edge in store.graph.incoming(e.id, ["DIAGNOSED_FROM"]):
        d = store.get(edge["source"])
        if d is not None and d.data.get("problem_type") == "data_quality" and d.data.get("context", {}).get(
                "subject") == e.id:
            return d.text
    return None


def evidence_summary(store: MemoryStore, statement_id: str) -> str:
    edges = store.graph.outgoing(statement_id, EVIDENCE)
    det = Counter(e["type"] for e in edges if e["origin"] == "deterministic")
    cited = Counter(e["type"] for e in edges if e["origin"] == "llm")
    parts = []
    if cited:
        parts.append(f"cited by its author: {cited['SUPPORTED_BY']} for, {cited['CONTRADICTED_BY']} against")
    if det:
        parts.append(f"checked: {det['SUPPORTED_BY']} result(s) for, {det['CONTRADICTED_BY']} against")
    return "; ".join(parts)


def procedure_history(store: MemoryStore, proc_id: str) -> str:
    """Where a procedure comes from and what its applications gave."""
    g = store.graph
    derived = len(g.outgoing(proc_id, ["DERIVED_FROM"]))
    apps = [a for e in g.incoming(proc_id, ["APPLIES"]) for a in g.outgoing(e["source"], ["APPLIED_IN"])]
    eff = sum(1 for a in apps if a["props"].get("effective") is True)
    inef = sum(1 for a in apps if a["props"].get("effective") is False)
    applied = sum(1 for a in apps if a["props"].get("applied") is True)
    just = [store.get(e["target"]) for e in g.outgoing(proc_id, ["JUSTIFIED_BY"])]
    text = f"derived from {derived} failure(s); judged {len(apps)} time(s): applied {applied}, effective {eff}, " \
           f"not effective {inef}"
    return text + "".join(f"; justified by \"{k.text}\"" for k in just if k is not None)


def describe(store: MemoryStore, e: Entry) -> str:
    """One line for an entry and its direct neighbors (the completion step)."""
    if e.category == "M1":
        why = unreliable(store, e)
        return e.text + (f" (WARNING, unreliable: {why})" if why else "")
    if e.kind == "failure":
        expl = [store.get(x["source"]) for x in store.graph.incoming(e.id, ["EXPLAINS"])]
        expl = [x for x in expl if x is not None]
        return e.text + (f" (proposed explanation: {expl[-1].text})" if expl else "")
    if e.kind == "diagnosis":
        return f"{e.text} [{e.data['problem_type'].replace('_', ' ')}, {e.data['epistemic_status']}]"
    if e.category == "M3":
        d, ver = e.data, store.get_verification(e.id)
        verdict = f"verdict: {ver['verdict']} ({ver['reason']})" if ver else "not verified yet"
        ev = evidence_summary(store, e.id)
        return (f"{e.text} [{d['epistemic_status']}; {verdict}" + (f"; {ev}" if ev else "")
                + (f"; uncertain: {d['uncertainty']}" if d.get("uncertainty") else "") + "]")
    if e.category == "M4":
        d = e.data
        return (f"{e.text} When: {d['trigger']}. Do: {d['actions']}. Limits: {d['limitations']}. "
                f"({procedure_history(store, e.id)})")
    return e.text
