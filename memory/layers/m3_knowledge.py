"""
M3 — Scientific Knowledge Memory: what the team believes about the science, and how strongly.

Every item has a statement, an optional machine-checkable claim, a scope (condition + region),
applicability, supporting and contradicting evidence (M1 ids, i.e. Flowcept task ids) and an epistemic
status:

  proposed    written by an agent, the analysis service or a human; not checked, or not checkable
  supported   the measurements agree with it (on the cited evidence AND on all comparable results)
  validated   supported, and confirmed by measurements made AFTER it was proposed (pre-registration
              rule: data that suggested a claim cannot confirm it), or validated by a human
  contested   new measurements disagree with it: it is re-evaluated, never deleted
  refuted     its own evidence contradicts it
  stale       it has not been re-evaluated although its scope received new data (set by the lifecycle)

The status is decided by deterministic rules (Evaluator); an LLM or the analysis service may only
propose. A human decision locks the status against automatic changes (the evaluation is still recorded).

Checkable claim types:
  comparison      {"type": "comparison", "better": A, "worse": B}
  feature_effect  {"type": "feature_effect", "feature": F, "op": ">", "value": 0, "effect": "higher"}
  threshold       {"type": "threshold", "region": {"feature": F, "op": ">", "value": 0} | null,
                   "op": ">", "value": X}           results in the region have an objective op X
  negative_region {"type": "negative_region", "feature": F, "op": ">", "value": 0}
                   experiments in the region give negative scientific results
"""
from __future__ import annotations

import hashlib
import json
import operator
import time
from statistics import mean

from ..domain import Domain
from ..schemas import KnowledgeRecord
from ..store import Store
from .m1_results import ResultsMemory

OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq,
       "!=": operator.ne}
ACTIVE = ("proposed", "supported", "validated", "contested")


class Evaluator:
    """Deterministic evaluation of a checkable claim against M1 results."""

    def __init__(self, results: ResultsMemory, domain: Domain, min_per_group: int = 2, min_confirm: int = 2,
                 threshold_fraction: float = 0.8) -> None:
        self.results, self.domain = results, domain
        self.min_per_group, self.min_confirm, self.fraction = min_per_group, min_confirm, threshold_fraction

    def _in_region(self, entity: str, region: dict | None) -> bool | None:
        if not region:
            return True
        f = self.domain.features(entity)
        if region["feature"] not in f:
            return None
        return OPS[region.get("op", ">")](f[region["feature"]], float(region.get("value", 0)))

    def _check(self, claim: dict, per: dict[str, dict]) -> dict:
        """Does the claim hold on these per-entity summaries? -> holds (True/False/None), agree, disagree ids."""
        t = claim.get("type")
        vals = {e: d["mean"] for e, d in per.items() if d["mean"] is not None}
        if t == "comparison":
            b, w = (self.domain.canonical(claim["better"]) or claim["better"],
                    self.domain.canonical(claim["worse"]) or claim["worse"])
            if b not in vals or w not in vals:
                return {"holds": None, "reason": f"no measurement of {[e for e in (b, w) if e not in vals]}"}
            ok = self.domain.better(vals[b], vals[w])
            ids = per[b]["ids"] + per[w]["ids"]
            return {"holds": ok, "agree": ids if ok else [], "disagree": [] if ok else ids,
                    "reason": f"{b}={vals[b]:.4g} vs {w}={vals[w]:.4g}", "n": 2}
        if t == "feature_effect":
            region = {"feature": claim["feature"], "op": claim.get("op", ">"), "value": claim.get("value", 0)}
            inside = {e: v for e, v in vals.items() if self._in_region(e, region) is True}
            outside = {e: v for e, v in vals.items() if self._in_region(e, region) is False}
            if min(len(inside), len(outside)) < self.min_per_group:
                return {"holds": None, "reason": f"too few entities: {len(inside)} in region, {len(outside)} "
                                                 f"outside (at least {self.min_per_group} each)"}
            m_in, m_out = mean(inside.values()), mean(outside.values())
            higher = claim.get("effect", "higher") == "higher"
            ok = (m_in > m_out) if higher else (m_in < m_out)
            agree = [i for e, v in inside.items() if ((v > m_out) if higher else (v < m_out)) for i in per[e]["ids"]]
            disagree = [i for e, v in inside.items() if ((v <= m_out) if higher else (v >= m_out)) for i in per[e]["ids"]]
            return {"holds": ok, "agree": agree, "disagree": disagree, "n": len(inside) + len(outside),
                    "reason": f"mean {m_in:.4g} in region vs {m_out:.4g} outside ({len(inside)} vs {len(outside)})"}
        if t == "threshold":
            region, test, x = claim.get("region"), OPS[claim.get("op", ">")], float(claim["value"])
            inside = {e: v for e, v in vals.items() if self._in_region(e, region) is True}
            if len(inside) < self.min_per_group:
                return {"holds": None, "reason": f"too few entities in region ({len(inside)})"}
            good = [e for e, v in inside.items() if test(v, x)]
            ok = len(good) / len(inside) >= self.fraction
            return {"holds": ok, "agree": [i for e in good for i in per[e]["ids"]],
                    "disagree": [i for e in inside if e not in good for i in per[e]["ids"]], "n": len(inside),
                    "reason": f"{len(good)}/{len(inside)} results {claim.get('op', '>')} {x:g}"}
        if t == "negative_region":
            region = {"feature": claim["feature"], "op": claim.get("op", ">"), "value": claim.get("value", 0)}
            inside = {e: d for e, d in per.items() if self._in_region(e, region) is True}
            if len(inside) < self.min_per_group:
                return {"holds": None, "reason": f"too few entities in region ({len(inside)})"}
            neg = [e for e, d in inside.items() if d["negative"]]
            ok = len(neg) / len(inside) >= self.fraction
            return {"holds": ok, "agree": [i for e in neg for i in per[e]["ids"]],
                    "disagree": [i for e in inside if e not in neg for i in per[e]["ids"]], "n": len(inside),
                    "reason": f"{len(neg)}/{len(inside)} experiments in region were negative"}
        return {"holds": None, "reason": "no machine-checkable claim"}

    def evaluate(self, k: KnowledgeRecord) -> dict:
        if not k.claim or k.claim.get("type") in (None, "none"):
            return {"checkable": False, "reason": "no machine-checkable claim: needs a human or a dedicated experiment"}
        if not k.condition:
            return {"checkable": False, "reason": "no experimental condition"}
        try:
            everything = self.results.per_entity(k.condition)
            cited_ids = set(k.supporting)
            cited = {}
            for e, d in everything.items():
                ids = [i for i in d["ids"] if i in cited_ids]
                if ids:
                    sub = [r for r in self.results.s.get_many("M1", ids)]
                    vals = [r.value for r in sub if r.outcome == "success"]
                    cited[e] = {"ids": ids, "values": vals, "mean": mean(vals) if vals else None,
                                "negative": any(r.outcome == "negative" for r in sub)}
            later = {e: d for e, d in everything.items() if d["first"] > k.created_at}
            on_cited = self._check(k.claim, cited) if cited else {"holds": None, "reason": "no cited evidence"}
            on_all = self._check(k.claim, everything)
            on_later = self._check(k.claim, later) if later else {"holds": None, "reason": "no later measurement"}
        except (KeyError, TypeError, ValueError) as exc:
            return {"checkable": False, "malformed": True, "reason": f"malformed claim: {exc}"}
        return {"checkable": True, "cited": on_cited, "all": on_all, "later": on_later,
                "n_results": self.results.count(k.condition), "t": time.time()}


class KnowledgeMemory:
    def __init__(self, store: Store, results: ResultsMemory, domain: Domain, evaluator: Evaluator) -> None:
        self.s, self.results, self.domain, self.ev = store, results, domain, evaluator

    def _key(self, kind: str, statement: str, claim: dict | None, condition: str) -> str:
        if claim and claim.get("type") not in (None, "none"):
            c = dict(claim)
            for f in ("better", "worse"):
                if f in c:
                    c[f] = self.domain.canonical(str(c[f])) or c[f]
            blob = {"claim": c, "condition": condition}
        else:
            blob = {"kind": kind, "text": " ".join(statement.lower().split()), "condition": condition}
        return hashlib.sha1(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def propose(self, statement: str, *, kind: str = "claim", claim: dict | None = None, condition: str = "",
                scope: dict | None = None, applicability: str = "", supporting: list[str] | None = None,
                source: str = "agent", by: str, provenance: list[dict] | None = None,
                memory_refs: list[str] | None = None) -> tuple[KnowledgeRecord, bool, list[dict]]:
        """Propose knowledge (status proposed), then evaluate it. If the same claim already exists, the new
        evidence and the endorsement are merged into it (recorded in its history) and it is re-evaluated.
        Returns (record, created, transitions)."""
        rec = KnowledgeRecord(kind=kind, statement=statement, claim=claim, condition=condition,
                              scope=scope or {}, applicability=applicability, supporting=list(supporting or []),
                              source=source, created_by=by, provenance=provenance or [],
                              memory_refs=memory_refs or [], key=self._key(kind, statement, claim, condition))
        stored, created = self.s.add(rec, actor=by, reason=f"proposed by {source}")
        if not created:
            def endorse(r: KnowledgeRecord):
                r.supporting = list(dict.fromkeys(r.supporting + list(supporting or [])))
                r.evaluation = {**r.evaluation, "endorsed_by": sorted(set(r.evaluation.get("endorsed_by", [])) | {by})}
            stored = self.s.update("M3", stored.id, endorse, by, "proposed again (evidence merged)")
        self.s.emit("knowledge", id=stored.id, created=created, status=stored.status, source=source)
        transitions = self.reevaluate(stored.id, by="knowledge-evaluator")
        return self.s.get("M3", stored.id), created, transitions

    def decide(self, k: KnowledgeRecord, ev: dict) -> tuple[str, str]:
        """Next epistemic status from an evaluation (pure function of the evidence)."""
        cur = k.status
        if not ev.get("checkable"):
            return cur, ev["reason"]
        cited, alld, later = ev["cited"], ev["all"], ev["later"]
        if cited["holds"] is False and cur in ("proposed", "refuted"):
            return "refuted", f"its own evidence contradicts it: {cited['reason']}"
        if alld["holds"] is False:
            if cur in ("supported", "validated", "contested"):
                return "contested", f"contradicted by the measurements: {alld['reason']}"
            return ("contested" if cited["holds"] else cur), f"all measurements disagree: {alld['reason']}"
        if alld["holds"] is None and cited["holds"] is None:
            return cur, f"insufficient evidence: {alld['reason']}"
        if alld["holds"] is True or cited["holds"] is True:
            if later.get("holds") is True and later.get("n", 0) >= self.ev.min_confirm:
                return "validated", f"confirmed by measurements made after it was proposed: {later['reason']}"
            if cur == "validated":
                return cur, f"still holds: {alld['reason']}"
            return "supported", f"the measurements agree: {(alld if alld['holds'] else cited)['reason']}"
        return cur, "no change"

    def reevaluate(self, rec_id: str, by: str) -> list[dict]:
        k = self.s.get("M3", rec_id)
        if k is None or k.status == "archived":
            return []
        ev = self.ev.evaluate(k)
        status, reason = self.decide(k, ev)
        if k.evaluation.get("locked_by_human"):
            status, reason = k.status, f"locked by human review (rule says: {reason})"
        transitions = []

        def change(r: KnowledgeRecord):
            if ev.get("checkable"):
                alld = ev["all"]
                r.supporting = list(dict.fromkeys(r.supporting + alld.get("agree", [])))[-200:]
                r.contradicting = list(dict.fromkeys(r.contradicting + alld.get("disagree", [])))[-200:]
                n_s, n_c = len(r.supporting), len(r.contradicting)
                r.confidence = round(n_s / (n_s + n_c) * min(1.0, (n_s + n_c) / 10), 3) if n_s + n_c else 0.0
            r.evaluation = {**r.evaluation, "last": {k_: v for k_, v in ev.items() if k_ != "t"},
                            "reason": reason, "evaluated_at": time.time(),
                            "n_results_at_eval": ev.get("n_results", 0)}
            if status != r.status:
                transitions.append({"id": r.id, "from": r.status, "to": status, "reason": reason})
                r.status = status
        self.s.update("M3", rec_id, change, by, f"evaluated: {reason}")
        for t in transitions:
            self.s.emit("knowledge_status", **t)
        return transitions

    def on_result(self, entity: str, condition: str, by: str) -> list[dict]:
        """A new result arrived: re-evaluate the checkable knowledge whose scope it touches
        (contradiction detection). Returns the status transitions."""
        out = []
        for k in self.s.find("M3", condition=condition, status=list(ACTIVE)):
            if not k.claim or k.claim.get("type") in (None, "none"):
                continue
            c = k.claim
            if c.get("type") == "comparison" and entity not in {self.domain.canonical(str(c.get(x))) for x in ("better", "worse")}:
                continue
            out += self.reevaluate(k.id, by=by)
        return out

    def review(self, rec_id: str, status: str, reason: str, reviewer: str) -> KnowledgeRecord | None:
        """Human decision: sets the status and locks it against automatic changes."""
        def change(r: KnowledgeRecord):
            r.status = status
            r.evaluation = {**r.evaluation, "locked_by_human": reviewer, "human_reason": reason}
        rec = self.s.update("M3", rec_id, change, reviewer, f"human review: {reason}")
        self.s.emit("knowledge_status", id=rec_id, to=status, reason=reason, by=reviewer)
        return rec

    def active(self, condition: str | None = None, statuses=ACTIVE) -> list[KnowledgeRecord]:
        f = {"status": list(statuses)}
        if condition:
            f["condition"] = condition
        return self.s.find("M3", **f)
