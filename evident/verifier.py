"""
Evidence check of claims against the measured facts (pure functions, no Redis).

For a claim "feature F increases (or decreases) the ionization energy", the verifier
compares all successfully computed molecules WITH F against those WITHOUT F:
  effect = mean(with) - mean(without)    (energy units of xTB/ASE, eV), Welch t
  adjusted effect = regression coefficient of the feature when all other features and
           the molecule size are in the model (separates co-occurring features);
           the verdict uses it whenever it can be estimated
  replications = number of agents whose OWN molecules show the same effect
                 (independent replication: one agent cannot establish a claim alone)

Lifecycle (never deleted, every change is versioned in the claims log):
  proposed    asserted by an agent, not enough evidence yet       (not shown to agents)
  supported   effect in the claimed direction, |t| >= T_MIN        (shown, with numbers)
  established supported + n_with >= N_ESTABLISHED + >= 2 agents   (shown, with numbers)
  contested   was supported/established, evidence no longer holds  (shown as a warning)
  rejected    significant effect in the OPPOSITE direction         (shown as a warning)
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

STATUSES = ("proposed", "supported", "established", "contested", "rejected")
SHOWN_AS_KNOWLEDGE = ("supported", "established")
SHOWN_AS_WARNING = ("contested", "rejected")


@dataclass
class VerifierParams:
    min_n: int = 3              # molecules needed on each side (with / without the feature)
    t_min: float = 2.0          # |Welch t| needed to call an effect significant
    min_effect: float = 0.10    # smallest effect size worth stating (eV)
    n_established: int = 6      # molecules with the feature needed to establish a claim
    min_replications: int = 2   # agents that must observe the effect independently

    @classmethod
    def from_env(cls) -> "VerifierParams":
        e = os.environ.get
        return cls(
            min_n=int(e("EVIDENT_MIN_N", cls.min_n)),
            t_min=float(e("EVIDENT_T_MIN", cls.t_min)),
            min_effect=float(e("EVIDENT_MIN_EFFECT", cls.min_effect)),
            n_established=int(e("EVIDENT_N_ESTABLISHED", cls.n_established)),
            min_replications=int(e("EVIDENT_MIN_REPLICATIONS", cls.min_replications)),
        )


def _mean_var(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else 0.0
    return m, v


def welch(with_f: list[float], without_f: list[float]) -> tuple[float, float]:
    """(effect, t) of mean(with) - mean(without)."""
    m1, v1 = _mean_var(with_f)
    m2, v2 = _mean_var(without_f)
    effect = m1 - m2
    se = math.sqrt(v1 / len(with_f) + v2 / len(without_f))
    if se == 0:
        t = 0.0 if effect == 0 else math.copysign(1e6, effect)
    else:
        t = effect / se
    return effect, t


def ok_facts(facts) -> list[dict]:
    """One successful fact per molecule (facts may contain duplicates or errors)."""
    seen, out = set(), []
    for f in facts:
        if f.get("status") != "ok" or f.get("energy") is None:
            continue
        if f["smiles"] in seen:
            continue
        seen.add(f["smiles"])
        out.append(f)
    return out


def measure(feature: str, facts: list[dict], params: VerifierParams) -> dict:
    """Effect of one feature over the given (successful, unique) facts."""
    with_f = [f["energy"] for f in facts if feature in f.get("features", ())]
    without = [f["energy"] for f in facts if feature not in f.get("features", ())]
    ev = {"n_with": len(with_f), "n_without": len(without), "effect": None, "t": None,
          "mean_with": None, "mean_without": None}
    if len(with_f) >= 2 and len(without) >= 2:
        effect, t = welch(with_f, without)
        ev.update(effect=round(effect, 4), t=round(t, 3),
                  mean_with=round(sum(with_f) / len(with_f), 4),
                  mean_without=round(sum(without) / len(without), 4))
    return ev


def adjusted_effects(facts: list[dict], min_count: int = 2) -> dict[str, dict]:
    """Effect of each feature ADJUSTED for all the others (ordinary least squares on
    feature indicators + molecule size). A marginal comparison is confounded when
    features co-occur (every urea is also a carbonyl); the regression separates them.
    Features whose indicator column duplicates another one are not identifiable."""
    import numpy as np
    facts = ok_facts(facts)
    n = len(facts)
    cols, seen = [], {}
    for name in sorted({f for x in facts for f in x.get("features", ())}):
        col = tuple(1.0 if name in x.get("features", ()) else 0.0 for x in facts)
        if min_count <= sum(col) <= n - min_count:
            if col in seen:
                seen[col].append(name)
            else:
                seen[col] = [name]
                cols.append(col)
    groups = [seen[c] for c in cols]
    out = {name: {"identifiable": False, "aliases": g} for g in groups for name in g if len(g) > 1}
    p = len(cols) + 2                                   # + intercept + heavy atoms
    if n < p + 3:
        return out
    X = np.column_stack([np.ones(n), [x.get("heavy_atoms") or 0 for x in facts]]
                        + [np.array(c) for c in cols])
    y = np.array([x["energy"] for x in facts])
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    sigma2 = float(resid @ resid) / max(n - np.linalg.matrix_rank(X), 1)
    se = np.sqrt(np.clip(np.diag(xtx_inv) * sigma2, 1e-12, None))
    for j, g in enumerate(groups):
        if len(g) > 1:
            continue
        out[g[0]] = {"identifiable": True, "coef": round(float(beta[j + 2]), 4),
                     "se": round(float(se[j + 2]), 4), "t": round(float(beta[j + 2] / se[j + 2]), 3)}
    return out


def evaluate(feature: str, direction: str, facts: list[dict], params: VerifierParams,
             adjusted: dict | None = None) -> dict:
    """Evidence for (feature, direction) and the verdict it implies. The verdict uses the
    adjusted effect when the regression can estimate it, the marginal one otherwise."""
    facts = ok_facts(facts)
    ev = measure(feature, facts, params)
    sign = 1 if direction == "up" else -1
    adj = (adjusted if adjusted is not None else adjusted_effects(facts)).get(feature)
    ev["adjusted"] = adj
    effect, t = ev["effect"], ev["t"]
    if adj and adj.get("identifiable"):
        effect, t = adj["coef"], adj["t"]
    if ev["n_with"] < params.min_n or ev["n_without"] < params.min_n or effect is None:
        verdict = "insufficient"
    elif adj is not None and not adj.get("identifiable"):
        verdict = "inconclusive"                       # confounded: cannot be separated
    elif abs(t) >= params.t_min and abs(effect) >= params.min_effect:
        verdict = "supports" if effect * sign > 0 else "contradicts"
    else:
        verdict = "inconclusive"

    # independent replication: the effect inside each agent's own molecules
    agents: dict[str, list[dict]] = {}
    for f in facts:
        agents.setdefault(f.get("agent") or "?", []).append(f)
    tested, replicated = 0, 0
    for own in agents.values():
        sub = measure(feature, own, params)
        if sub["n_with"] >= 2 and sub["n_without"] >= 2 and sub["effect"] is not None:
            tested += 1
            if sub["effect"] * sign >= params.min_effect:
                replicated += 1
    ev.update(verdict=verdict, agents_tested=tested, replications=replicated)
    return ev


def next_status(previous: str | None, ev: dict, params: VerifierParams) -> str:
    verdict = ev["verdict"]
    if verdict == "supports":
        if ev["n_with"] >= params.n_established and ev["replications"] >= params.min_replications:
            return "established"
        return "supported"
    if verdict == "contradicts":
        return "rejected"
    # insufficient or inconclusive evidence
    if previous in ("supported", "established", "contested"):
        return "contested"
    if previous == "rejected":
        return "rejected"
    return "proposed"


def reference_effects(facts: list[dict], features: list[str], params: VerifierParams) -> dict:
    """Ground truth used by the analysis: verdict of BOTH directions of every feature
    over a large pool of facts (e.g. all the runs of an experiment)."""
    out = {}
    adjusted = adjusted_effects(facts)
    for feat in features:
        up = evaluate(feat, "up", facts, params, adjusted)
        if up["verdict"] == "supports":
            truth = "up"
        elif up["verdict"] == "contradicts":
            truth = "down"
        elif up["verdict"] == "insufficient":
            truth = "unknown"
        else:
            truth = "none"
        out[feat] = {"truth": truth, **{k: up[k] for k in ("effect", "t", "n_with", "n_without")}}
    return out
