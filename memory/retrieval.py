"""
Retrieval and context construction for LLM prompts.

  exact       results of one experiment (ResultsMemory.lookup)
  contextual  records of a layer filtered by metadata (status, kind, condition, targets...) and ranked by
              lexical relevance to the current query (BM25-style weighting), status, priority and recency

No embeddings: the vocabulary of this domain (SMILES, feature names, short statements) is matched well
lexically, and a vector store would add a model and a dependency without a demonstrated need.

`context()` builds the memory part of a prompt for one LangGraph node, within a token budget split into
per-layer quotas that depend on the node, and returns the ids it showed (usage + delivery tracking).
Knowledge is always labelled with its epistemic status: only validated / supported items are presented
as knowledge; contested, refuted and proposed items are presented as such.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter

TOKEN_CHARS = 4          # rough size of a token, for budgeting
_WORD = re.compile(r"[A-Za-z0-9_#=()\[\]@+\-]+")

QUOTAS = {   # share of the budget per section, by node
    "plan":         {"guidance": .25, "knowledge": .40, "results": .25, "experience": .10},
    "tool_calling": {"guidance": .30, "results": .45, "knowledge": .20, "experience": .05},
    "critique":     {"experience": .35, "knowledge": .30, "results": .20, "guidance": .15},
    "update":       {"guidance": .35, "knowledge": .30, "results": .20, "experience": .15},
}
STATUS_WEIGHT = {"validated": 1.0, "supported": 0.8, "contested": 0.6, "refuted": 0.5, "proposed": 0.4,
                 "active": 1.0, "open": 1.0, "investigating": 0.9, "confirmed": 1.0}


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text or "")]


def relevance(query: str, docs: list[str]) -> list[float]:
    """BM25-style scores of documents for a query (k1=1.2, b=0.75)."""
    q = set(tokens(query))
    toks = [tokens(d) for d in docs]
    if not q or not docs:
        return [0.0] * len(docs)
    avg = sum(map(len, toks)) / len(toks) or 1.0
    df = Counter(t for ts in toks for t in set(ts))
    out = []
    for ts in toks:
        tf = Counter(ts)
        s = 0.0
        for t in q & set(ts):
            idf = math.log(1 + (len(docs) - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(ts) / avg))
        out.append(s)
    return out


def rank(records: list, query: str, text_of, now: float | None = None) -> list:
    """Order records by relevance, status weight, priority and recency (half-life one hour)."""
    now = now or time.time()
    rel = relevance(query, [text_of(r) for r in records])
    top = max(rel) if rel and max(rel) > 0 else 1.0

    def score(i_r):
        i, r = i_r
        recency = 0.5 ** ((now - r.updated_at) / 3600.0)
        return (0.5 * rel[i] / top + 0.3 * STATUS_WEIGHT.get(r.status, 0.5) + 0.2 * recency
                + 0.1 * getattr(r, "priority", 0))
    return [r for _, r in sorted(enumerate(records), key=score, reverse=True)]


class ContextBuilder:
    def __init__(self, memory) -> None:
        self.m = memory

    def _results(self, agent: str, cond: str, n: int = 12) -> list[tuple[str | None, str]]:
        res = self.m.results
        lines: list[tuple[str | None, str]] = []
        best = res.best(cond, n)
        if best:
            better = "higher" if self.m.domain.direction == "max" else "lower"
            lines.append((None, f"Measured by the team ({self.m.domain.objective}, {better} is better; "
                                f"{res.count(cond)} results):"))
            lines += [(None, f"  {e}: {v:.6g}") for e, v in best]
        running = [t.split("/", 1)[1] for t, a in self.m.coord.running().items()
                   if t.startswith(cond + "/") and a != agent]
        if running:
            lines.append((None, "Being computed now by other agents (do not propose): " + " ".join(running[:20])))
        neg = list(res.negatives(cond))
        if neg:
            lines.append((None, "Negative scientific results (do not retry): "
                                + " ".join(e.removeprefix("invalid:") for e in neg[:20])))
        return lines

    def _knowledge(self, cond: str, query: str) -> list[tuple[str | None, str]]:
        ks = [k for k in self.m.knowledge.active(statuses=("validated", "supported", "contested", "refuted", "proposed"))
              if k.condition in (cond, "")]
        ks = rank(ks, query, lambda k: k.statement)
        label = {"validated": "VALIDATED", "supported": "SUPPORTED (not yet independently confirmed)",
                 "contested": "CONTESTED (new results disagree, being re-evaluated)",
                 "refuted": "REFUTED (do NOT rely on it)", "proposed": "HYPOTHESIS (unverified, worth testing)"}
        order = ["validated", "supported", "contested", "refuted", "proposed"]
        ks.sort(key=lambda k: order.index(k.status))
        lines = [(None, "Scientific knowledge of the team:")] if ks else []
        lines += [(f"M3:{k.id}", f"  - {label[k.status]}: {k.statement} "
                                 f"[{len(k.supporting)} for / {len(k.contradicting)} against]") for k in ks]
        return lines

    def _experience(self, agent: str, query: str) -> list[tuple[str | None, str]]:
        eps = [e for e in self.m.experience.open() if not e.agents or agent in e.agents or e.severity != "low"]
        eps = rank(eps, query, lambda e: e.summary)[:6]
        lines = [(None, "Problems observed in this campaign (facts; explanations are hypotheses):")] if eps else []
        for e in eps:
            cause = f" Cause (confirmed): {e.confirmed_cause}." if e.confirmed_cause else (
                f" Possible explanation: {e.candidate_explanations[0]}" if e.candidate_explanations else "")
            lines.append((f"M2:{e.id}", f"  - [{e.kind}, x{e.occurrences}] {e.summary}.{cause}"))
        return lines

    def _guidance(self, agent: str, node: str, cond: str, new_ids: set[str]) -> list[tuple[str | None, str]]:
        gs = self.m.guidance.for_agent(agent, node, cond)
        lines = [(None, "Instructions for you (follow them unless the measurements show otherwise):")] if gs else []
        for g in gs:
            mark = "NEW " if g.id in new_ids else ""
            lines.append((f"M4:{g.id}", f"  - {mark}[{g.kind}, {g.status}] {g.text}"))
        return lines

    def build(self, agent: str, node: str, cond: str, query: str = "", budget_tokens: int = 700,
              new_guidance: set[str] | None = None) -> tuple[str, dict[str, list[str]]]:
        quotas = QUOTAS.get(node, QUOTAS["plan"])
        budget = budget_tokens * TOKEN_CHARS
        sections = {"guidance": lambda: self._guidance(agent, node, cond, new_guidance or set()),
                    "knowledge": lambda: self._knowledge(cond, query),
                    "results": lambda: self._results(agent, cond),
                    "experience": lambda: self._experience(agent, query)}
        out, shown, spare = ["SHARED MEMORY OF THE TEAM"], {"M2": [], "M3": [], "M4": []}, 0
        used = len(out[0])
        for name, share in sorted(quotas.items(), key=lambda x: -x[1]):
            cap = int(budget * share) + spare
            spent = 0
            lines = sections[name]()
            for ref, text in lines:
                if spent + len(text) + 1 > cap or used + len(text) + 1 > budget:
                    break
                out.append(text)
                spent += len(text) + 1
                used += len(text) + 1
                if ref:
                    layer, rid = ref.split(":", 1)
                    shown[layer].append(rid)
            spare = max(0, cap - spent)
        text = "\n".join(out) if len(out) > 1 else ""
        return text, shown
