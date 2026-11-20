"""
Claim-evidence verification: is a statement of the team's knowledge (M3) supported by the measured results (M1)?

The LLM only translates a statement into one of a few structured forms; the check itself is deterministic, on the
results that have no data-quality problem (finite value, converged optimization, no analysis-agent note):

  value         "the IE of X is about v"            |IE(X) - v| <= VALUE_TOLERANCE_EV
  bound         "the IE of X is above / below v"    IE(X) > v or < v
  comparison    "X has a higher / lower IE than Y"  IE(X) - IE(Y), undecided within COMPARISON_MARGIN_EV
  motif_effect  "feature F increases / decreases IE" mean IE with F vs without F (Welch's t, |t| >= T_THRESHOLD
                                                     with at least MIN_GROUP results in each group)

Each check gives a verdict: supported, contradicted, or inconclusive (missing or unreliable results, or no clear
difference) with its reason and the results used. A verdict is evidence about a statement, not the truth of it: it
only says whether the team's current results agree. Records are kept apart from the M3 entries (linked by id), with
their history, and are recomputed when a new result falls in a statement's scope.
"""
from __future__ import annotations

import math
import statistics
import time

from . import chem
from .llm import EXTRACT_CLAIM
from .schema import Entry
from .store import MemoryStore

# policy of the checks (stated here, reported with the evaluation)
VALUE_TOLERANCE_EV = 0.05        # xTB is deterministic for a given geometry: a stated value matches within this
COMPARISON_MARGIN_EV = 0.01
MIN_GROUP = 2                    # "at least two" results with and without the feature
T_THRESHOLD = 2.0                # roughly a 95 % two-sided level

FORMS = ("value", "bound", "comparison", "motif_effect")


def extraction_prompt() -> str:
    return EXTRACT_CLAIM.replace("FEATURE_NAMES", ", ".join(chem.FEATURES))


def normalize(claim: dict) -> dict:
    """Check the LLM's structured claim; anything not checkable becomes {"form": "none"}."""
    if not isinstance(claim, dict) or claim.get("form") not in FORMS:
        return {"form": "none"}
    c = {"form": claim["form"]}
    try:
        if c["form"] in ("value", "bound"):
            c["molecule"] = chem.canonical(claim["molecule"])
            c["value"] = float(claim["value"])
            if c["form"] == "bound":
                if claim.get("relation") not in ("above", "below"):
                    return {"form": "none"}
                c["relation"] = claim["relation"]
            ok = chem.parses(c["molecule"])
        elif c["form"] == "comparison":
            c["a"], c["b"] = chem.canonical(claim["a"]), chem.canonical(claim["b"])
            c["relation"] = claim.get("relation")
            ok = chem.parses(c["a"]) and chem.parses(c["b"]) and c["relation"] in ("higher", "lower") \
                and c["a"] != c["b"]
        else:
            c["feature"], c["direction"] = claim.get("feature"), claim.get("direction")
            ok = c["feature"] in chem.FEATURES and c["direction"] in ("increases", "decreases")
    except (KeyError, TypeError, ValueError):
        return {"form": "none"}
    return c if ok else {"form": "none"}


def reliable(e: Entry) -> bool:
    v = e.data.get("value")
    return (isinstance(v, (int, float)) and math.isfinite(v) and not e.data.get("notes")
            and e.data.get("conditions", {}).get("converged") is not False)


def results(store: MemoryStore, values: dict[str, float] | None = None) -> tuple[dict, set]:
    """Reliable results by molecule (the first entry of each), and the molecules that only have unreliable ones.
    `values` replaces the stored values (the evaluation uses it to check with the true values)."""
    good, bad = {}, set()
    for e in store.all("M1"):
        if reliable(e):
            good.setdefault(e.key, (values.get(e.key, e.data["value"]) if values else e.data["value"], e.id))
        else:
            bad.add(e.key)
    return good, bad - set(good)


def check(claim: dict, store: MemoryStore, values: dict[str, float] | None = None) -> dict:
    good, bad = results(store, values)

    def missing(*mols) -> dict | None:
        for m in mols:
            if m not in good:
                why = "rests on an unreliable result" if m in bad else "no result yet"
                return {"verdict": "inconclusive", "reason": f"{m}: {why}", "evidence": []}
        return None

    form = claim.get("form")
    if form in ("value", "bound"):
        m = claim["molecule"]
        if (r := missing(m)) is not None:
            return r
        v, eid = good[m]
        if form == "value":
            ok = abs(v - claim["value"]) <= VALUE_TOLERANCE_EV
            reason = f"measured {v:.3f} eV vs stated {claim['value']:.3f} eV"
        else:
            ok = v > claim["value"] if claim["relation"] == "above" else v < claim["value"]
            reason = f"measured {v:.3f} eV, stated {claim['relation']} {claim['value']:.3f} eV"
        return {"verdict": "supported" if ok else "contradicted", "reason": reason, "evidence": [eid]}
    if form == "comparison":
        a, b = claim["a"], claim["b"]
        if (r := missing(a, b)) is not None:
            return r
        (va, ea), (vb, eb) = good[a], good[b]
        diff = va - vb
        reason = f"{a} {va:.3f} eV vs {b} {vb:.3f} eV"
        if abs(diff) <= COMPARISON_MARGIN_EV:
            return {"verdict": "inconclusive", "reason": reason + " (equal within the margin)", "evidence": [ea, eb]}
        ok = diff > 0 if claim["relation"] == "higher" else diff < 0
        return {"verdict": "supported" if ok else "contradicted", "reason": reason, "evidence": [ea, eb]}
    if form == "motif_effect":
        f = claim["feature"]
        with_f = [(v, i) for k, (v, i) in good.items() if f in chem.features(k)]
        without = [(v, i) for k, (v, i) in good.items() if f not in chem.features(k)]
        if len(with_f) < MIN_GROUP or len(without) < MIN_GROUP:
            return {"verdict": "inconclusive", "evidence": [],
                    "reason": f"{len(with_f)} result(s) with {f}, {len(without)} without (need {MIN_GROUP} each)"}
        a, b = [v for v, _ in with_f], [v for v, _ in without]
        diff = statistics.mean(a) - statistics.mean(b)
        se = math.sqrt(statistics.variance(a) / len(a) + statistics.variance(b) / len(b))
        t = diff / se if se > 0 else math.copysign(math.inf, diff) if diff else 0.0
        reason = (f"mean {statistics.mean(a):.2f} eV with {f} (n={len(a)}) vs {statistics.mean(b):.2f} eV "
                  f"without (n={len(b)}), t={t:.1f}")
        evidence = [i for _, i in with_f + without]
        if abs(t) < T_THRESHOLD:
            return {"verdict": "inconclusive", "reason": reason + " (no clear difference)", "evidence": evidence}
        ok = diff > 0 if claim["direction"] == "increases" else diff < 0
        return {"verdict": "supported" if ok else "contradicted", "reason": reason, "evidence": evidence}
    return {"verdict": "not_checkable", "reason": "not one of the checkable forms", "evidence": []}


def in_scope(claim: dict, key: str) -> bool:
    """Whether a new result for this molecule can change the claim's verdict."""
    form = claim.get("form")
    if form in ("value", "bound"):
        return claim["molecule"] == key
    if form == "comparison":
        return key in (claim["a"], claim["b"])
    if form == "motif_effect":
        return True               # every result enters one of the two groups
    return False


def record(store: MemoryStore, entry_id: str, claim: dict, result: dict, trigger: str) -> tuple[dict, str | None]:
    """Store a verification; returns the record and the previous verdict."""
    old = store.get_verification(entry_id) or {"claim": claim, "history": []}
    prev = old.get("verdict")
    rec = {**old, "claim": claim, **result, "t": time.time()}
    if prev != result["verdict"]:
        rec["history"] = old.get("history", []) + [{"t": rec["t"], "verdict": result["verdict"],
                                                    "reason": result["reason"], "trigger": trigger}]
    store.set_verification(entry_id, rec)
    return rec, prev
