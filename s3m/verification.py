"""
Verification: may a lesson be reused as knowledge? Decided by explicit, deterministic rules
computed on the observations, never by an LLM.

Common rules, for every lesson:
  1. it cites observations, and they all exist                      otherwise rejected
  2. it has a machine-checkable claim                                otherwise needs_review
  3. the cited observations come from one comparable condition       otherwise needs_review
     (same method, same parameters)
Then the rule of the claim's kind, applied twice:
  - on the cited evidence: does it really show what the lesson says?  no -> rejected
  - on every comparable observation in memory: is the lesson still    no -> needs_review
    true when the observations it did not cite are included?          (contradicting data)

Claim kinds provided here (a workflow can register its own with `Verifier.register`):
  comparison      {"kind": "comparison", "better": A, "worse": B}
                  A has a better objective value than B
  feature_effect  {"kind": "feature_effect", "feature": F, "op": ">", "value": 0, "effect": "higher"}
                  entities whose feature F satisfies (op, value) have a higher (or lower) mean value
                  than the other entities; needs at least `min_per_group` entities in each group

The rule is simple on purpose (mean difference, minimum group size): it is a sanity check of what the
data show, not a statistical test. Values are averaged per entity first, so an entity measured
several times counts once.
"""
from __future__ import annotations

import operator
from statistics import mean
from typing import Callable

from .domain import Domain
from .observations import ExperimentalMemory

OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq,
       "!=": operator.ne}

Verdict = tuple[str, str, dict]   # (status, reason, details)


class Verifier:
    method = "deterministic-rules-v1"

    def __init__(self, observations: ExperimentalMemory, domain: Domain, min_per_group: int = 2) -> None:
        self.obs, self.domain, self.min_per_group = observations, domain, min_per_group
        self.checks: dict[str, Callable[[dict, list[dict], str], Verdict]] = {
            "comparison": self._comparison,
            "feature_effect": self._feature_effect,
        }

    def register(self, kind: str, check: Callable[[dict, list[dict], str], Verdict]) -> None:
        """check(claim, cited_observations, condition_id) -> (status, reason, details)."""
        self.checks[kind] = check

    # ------------------------------------------------------------------ entry point
    def verify(self, lesson: dict) -> Verdict:
        ids = lesson.get("evidence_ids") or []
        if not ids:
            return "rejected", "the lesson cites no observation", {}
        found = self.obs.get_many(ids)
        missing = [i for i, o in zip(ids, found) if o is None]
        if missing:
            return "rejected", f"cited observations do not exist: {missing[:5]}", {"missing": missing}
        claim = lesson.get("claim") or {}
        kind = claim.get("kind", "none")
        if kind not in self.checks:
            return ("needs_review", "no machine-checkable claim: a dedicated experiment or a human "
                    "must decide", {"kind": kind})
        conds = {o["condition"] for o in found}
        if len(conds) > 1:
            return ("needs_review", "cited observations come from different conditions "
                    "(method or parameters differ)", {"conditions": sorted(conds)})
        try:
            return self.checks[kind](claim, found, conds.pop())
        except (KeyError, TypeError, ValueError) as exc:
            return "rejected", f"malformed {kind} claim: {exc}", {"claim": claim}

    # ------------------------------------------------------------------ helpers
    def _per_entity(self, observations: list[dict]) -> dict[str, float]:
        """Mean value per entity, successful observations only."""
        vals: dict[str, list[float]] = {}
        for o in observations:
            if o["status"] == "success":
                vals.setdefault(o["entity"], []).append(o["value"])
        return {e: mean(v) for e, v in vals.items()}

    def _canonical(self, raw) -> str:
        return self.domain.canonical(str(raw)) or str(raw)

    # ------------------------------------------------------------------ comparison
    def _comparison(self, claim: dict, cited: list[dict], cond: str) -> Verdict:
        better, worse = self._canonical(claim["better"]), self._canonical(claim["worse"])
        vals = self._per_entity(cited)
        absent = [e for e in (better, worse) if e not in vals]
        if absent:
            return "rejected", f"no successful cited measurement of {absent}", {"absent": absent}
        details = {"better": better, "worse": worse, "cited": {better: vals[better], worse: vals[worse]}}
        if not self.domain.better(vals[better], vals[worse]):
            return ("rejected", f"cited values contradict the lesson: {better}={vals[better]:.4g}, "
                    f"{worse}={vals[worse]:.4g}", details)
        everything = self._per_entity(self.obs.of_entity(better, cond) + self.obs.of_entity(worse, cond))
        details["all"] = {better: everything[better], worse: everything[worse]}
        if not self.domain.better(everything[better], everything[worse]):
            return "needs_review", "other measurements of the same entities contradict the lesson", details
        return "verified", f"{better} ({vals[better]:.4g}) is better than {worse} ({vals[worse]:.4g})", details

    # ------------------------------------------------------------------ feature effect
    def _groups(self, per_entity: dict[str, float], feature: str, op: str, value: float):
        test = OPS[op]
        inside, outside = {}, {}
        for e, v in per_entity.items():
            feats = self.domain.features(e)
            if feature not in feats:
                continue
            (inside if test(feats[feature], value) else outside)[e] = v
        return inside, outside

    def _feature_effect(self, claim: dict, cited: list[dict], cond: str) -> Verdict:
        feature, op, value = claim["feature"], claim.get("op", ">"), float(claim.get("value", 0))
        effect = claim.get("effect", "higher")
        if op not in OPS or effect not in ("higher", "lower"):
            raise ValueError(f"op must be one of {list(OPS)} and effect 'higher' or 'lower'")
        sample = next((o["entity"] for o in cited if self.domain.features(o["entity"])), None)
        if sample is None or feature not in self.domain.features(sample):
            return "rejected", f"unknown feature '{feature}'", {"feature": feature}

        def summary(per_entity):
            inside, outside = self._groups(per_entity, feature, op, value)
            s = {"n_with": len(inside), "n_without": len(outside)}
            if inside and outside:
                s.update(mean_with=mean(inside.values()), mean_without=mean(outside.values()))
                s["difference"] = s["mean_with"] - s["mean_without"]
            return s

        def agrees(s):
            return (s["difference"] > 0) if effect == "higher" else (s["difference"] < 0)

        scope = f"{feature} {op} {value:g}"
        cited_s = summary(self._per_entity(cited))
        details = {"scope": scope, "effect": effect, "cited": cited_s}
        if min(cited_s["n_with"], cited_s["n_without"]) < self.min_per_group:
            return ("needs_review", f"too little evidence: {cited_s['n_with']} entities with {scope}, "
                    f"{cited_s['n_without']} without (at least {self.min_per_group} each)", details)
        if not agrees(cited_s):
            return ("rejected", f"cited values contradict the lesson: mean {cited_s['mean_with']:.4g} with "
                    f"{scope} vs {cited_s['mean_without']:.4g} without", details)
        all_s = summary(self._per_entity(self.obs.in_condition(cond)))
        details["all"] = all_s
        if min(all_s["n_with"], all_s["n_without"]) >= self.min_per_group and not agrees(all_s):
            return ("needs_review", f"contradicted by all comparable observations: mean "
                    f"{all_s['mean_with']:.4g} with {scope} vs {all_s['mean_without']:.4g} without", details)
        return ("verified", f"mean {cited_s['mean_with']:.4g} with {scope} vs {cited_s['mean_without']:.4g} "
                f"without ({cited_s['n_with']} vs {cited_s['n_without']} entities)", details)
