"""
Task-aware retrieval from the shared memory, used both for the memory sections the workflow adds to a node's prompt
(scheduled) and for the questions an agent asks itself (on demand, `MemoryClient.answer`).

A query describes the current task: its text (plan, critique, question), the molecules it is about (written in the
text, the agent's seed, its recent proposals) and the failure classes recently met. Each candidate entry is scored by
a hybrid of
  - structural similarity: Tanimoto similarity of Morgan fingerprints between the query's molecules and the entry's
    molecules (the most informative signal for chemistry);
  - motif overlap: Jaccard overlap of the functional groups of chem.FEATURES;
  - lexical overlap: shared words between the query text and the entry (cosine of word sets);
  - failure-class match, for failures, diagnoses and procedures.
Structured filters (category, kind) apply before ranking. Entries with no signal at all are not returned.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import chem
from .schema import Entry
from .store import MemoryStore, tokens

# weights of the hybrid score (design parameters of the retrieval, reported with the evaluation)
WEIGHTS = {"structure": 0.4, "motif": 0.25, "text": 0.2, "failure_class": 0.15}
# experiences worth showing an agent: problems, analyses, outcomes of guidance, and attempts that improved
M2_KINDS = ("failure", "diagnosis", "outcome", "attempt")


@dataclass
class Query:
    text: str = ""
    molecules: list[str] = field(default_factory=list)
    failure_classes: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.words = tokens(self.text)
        self.features = set().union(*(chem.features(m) for m in self.molecules)) if self.molecules else set()


def build_query(text: str = "", molecules: list[str] | None = None, failure_classes: set[str] | None = None) -> Query:
    mols = []
    for m in list(molecules or []) + chem.molecules_in(text):
        k = chem.canonical(m)
        if k and chem.parses(k) and k not in mols:
            mols.append(k)
    return Query(text=text, molecules=mols[:20], failure_classes=set(failure_classes or ()))


def entry_molecules(e: Entry, store: MemoryStore) -> list[str]:
    """Molecules an entry is about."""
    if e.category == "M1" or (e.category == "M2" and e.kind == "failure"):
        return [e.key] if e.key else []
    if e.category == "M2" and e.kind == "attempt":
        return list(e.data.get("context", {}).get("proposed", []))[:20]
    if e.category == "M3":
        ids = [ev["entry"] for ev in e.data["supporting_evidence"] + e.data["contradicting_evidence"]]
        keys = [x.key for x in (store.get(i) for i in ids) if x is not None]
        return list(dict.fromkeys(keys + chem.molecules_in(e.text)))
    if e.category == "M4":
        keys = [x.key for x in (store.get(i) for i in e.data.get("supporting_episodes", [])) if x is not None]
        return [k for k in dict.fromkeys(keys) if chem.parses(k)]
    return chem.molecules_in(e.text)


FAILURE_CLASSES = ("invalid_smiles", "embedding_failure", "scf_not_converged", "timeout", "other")


def entry_failure_class(e: Entry) -> str | None:
    """The failure class an entry is about: a failure's class, the class a diagnosis is keyed by, a procedure's
    trigger class."""
    if e.category == "M2" and e.kind == "failure":
        return e.data.get("context", {}).get("failure_class")
    if (e.category == "M2" and e.kind == "diagnosis") or e.category == "M4":
        return e.key if e.key in FAILURE_CLASSES else None
    return None


def score(e: Entry, q: Query, store: MemoryStore) -> tuple[float, dict]:
    mols = entry_molecules(e, store)
    structure = max((chem.similarity(a, b) for a in q.molecules for b in mols), default=0.0)
    feats = set().union(*(chem.features(m) for m in mols)) if mols else set()
    motif = len(q.features & feats) / len(q.features | feats) if (q.features or feats) and q.features else 0.0
    words = tokens(f"{e.text} {e.key}")
    text = len(q.words & words) / math.sqrt(len(q.words) * len(words)) if q.words and words else 0.0
    fc = entry_failure_class(e)
    fclass = 1.0 if fc and fc in q.failure_classes else 0.0
    parts = {"structure": structure, "motif": motif, "text": text, "failure_class": fclass}
    return sum(WEIGHTS[k] * v for k, v in parts.items()), parts


def retrieve(store: MemoryStore, category: str, q: Query, limit: int, kinds=None, scan: int = 500,
             keep=None) -> list[tuple[Entry, float, dict]]:
    """The `limit` best entries of a category for the query (kinds filtered before ranking)."""
    out = []
    for e in store.recent(category, scan):
        if kinds is not None and e.kind not in kinds:
            continue
        if keep is not None and not keep(e):
            continue
        s, parts = score(e, q, store)
        if s > 0:
            out.append((e, s, parts))
    out.sort(key=lambda x: (-x[1], -x[0].created_at))
    return out[:limit]


def applicable(procedure: Entry, q: Query, store: MemoryStore) -> str | None:
    """Why a procedure applies to the current task, or None: its trigger (a failure class) was met recently, or the
    task's molecules resemble those of the experiences it was derived from."""
    if procedure.key in q.failure_classes:
        return f"{procedure.key} failures occurred recently"
    mols = entry_molecules(procedure, store)
    sim = max((chem.similarity(a, b) for a in q.molecules for b in mols), default=0.0)
    if sim >= 0.4:
        return f"the molecules at hand resemble those that failed ({sim:.2f} similarity)"
    return None
