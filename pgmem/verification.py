"""
Verification of statements against the measured results: the verdict axis of reliability (the epistemic status is
the author's own declaration, pgmem/schema.py).

The LLM only translates a statement into one of a few structured forms; the check is deterministic, on the results
without a data-quality problem (finite value, converged optimization, no data-quality diagnosis):

  value         "the IE of X is about v"            |IE(X) - v| <= VALUE_TOLERANCE_EV
  bound         "the IE of X is above / below v"    IE(X) > v or < v
  comparison    "X has a higher / lower IE than Y"  IE(X) - IE(Y), undecided within COMPARISON_MARGIN_EV
  motif_effect  "motif M increases / decreases IE"  mean IE with M vs without M (Welch's t, |t| >= T_THRESHOLD with
                                                    at least MIN_GROUP results in each group)

Three verdicts: supported, contradicted, inconclusive (a statement that cannot be put in one of the forms, missing
or unreliable results, no clear difference). Missing evidence never gives "contradicted". A verdict says whether the
team's current results agree with a statement, not whether it is true.

A check is recorded in two places: the verdict record (pgmem:{ns}:verification, with its history), and the results
it used as deterministic evidence edges of the statement (SUPPORTED_BY / CONTRADICTED_BY, created_by
"verification"). A new check replaces the previous check's edges; edges proposed by the statement's author stay.
"""
from __future__ import annotations

import math
import statistics
import time

from . import chem
from .domain import DOMAIN
from .llm import EXTRACT_CLAIM
from .retrieval import unreliable
from .store import MemoryStore

VALUE_TOLERANCE_EV = 0.05        # xTB is deterministic for a given geometry: a stated value matches within this
COMPARISON_MARGIN_EV = 0.01
MIN_GROUP = 2
T_THRESHOLD = 2.0                # roughly a 95 % two-sided level
FORMS = ("value", "bound", "comparison", "motif_effect")
BY = "verification"


def extraction_prompt() -> str:
    return EXTRACT_CLAIM.replace("FEATURE_NAMES", ", ".join(m for m in chem.FEATURES if m not in chem.GENERIC))


def normalize(claim: dict) -> dict:
    """Check the LLM's structured claim; anything not checkable becomes {"form": "none"}."""
    if not isinstance(claim, dict) or claim.get("form") not in FORMS:
        return {"form": "none"}
    c = {"form": claim["form"]}
    try:
        if c["form"] in ("value", "bound"):
            c["molecule"], c["value"] = DOMAIN.canonical(claim["molecule"]), float(claim["value"])
            ok = DOMAIN.valid(c["molecule"])
            if c["form"] == "bound":
                c["relation"] = claim.get("relation")
                ok = ok and c["relation"] in ("above", "below")
        elif c["form"] == "comparison":
            c["a"], c["b"] = DOMAIN.canonical(claim["a"]), DOMAIN.canonical(claim["b"])
            c["relation"] = claim.get("relation")
            ok = DOMAIN.valid(c["a"]) and DOMAIN.valid(c["b"]) and c["relation"] in ("higher", "lower") \
                and c["a"] != c["b"]
        else:
            c["feature"], c["direction"] = claim.get("feature"), claim.get("direction")
            ok = c["feature"] in chem.FEATURES and c["direction"] in ("increases", "decreases")
    except (KeyError, TypeError, ValueError):
        return {"form": "none"}
    return c if ok else {"form": "none"}


def results(store: MemoryStore, values: dict[str, float] | None = None) -> tuple[dict, set]:
    """Reliable results by molecule (value, entry id of its first result), and the molecules that only have
    unreliable ones. `values` replaces the stored values (the evaluation checks with the true values)."""
    good, bad = {}, set()
    for e in store.all("M1"):
        v = e.data.get("value")
        if isinstance(v, (int, float)) and math.isfinite(v) and unreliable(store, e) is None:
            good.setdefault(e.key, ((values or {}).get(e.key, v), e.id))
        else:
            bad.add(e.key)
    return good, bad - set(good)


def check(claim: dict, good: dict, bad: set) -> dict:
    """Verdict of a structured claim on the reliable results: {verdict, reason, checkable, evidence}."""
    def missing(*mols) -> dict | None:
        for m in mols:
            if m not in good:
                why = "only unreliable results" if m in bad else "no result yet"
                return {"verdict": "inconclusive", "reason": f"{m}: {why}", "checkable": True, "evidence": []}
        return None

    def verdict(ok: bool, reason: str, evidence: list) -> dict:
        return {"verdict": "supported" if ok else "contradicted", "reason": reason, "checkable": True,
                "evidence": evidence}

    form = claim.get("form")
    if form in ("value", "bound"):
        m = claim["molecule"]
        if (r := missing(m)) is not None:
            return r
        v, eid = good[m]
        if form == "value":
            return verdict(abs(v - claim["value"]) <= VALUE_TOLERANCE_EV,
                           f"measured {v:.3f} eV vs stated {claim['value']:.3f} eV", [eid])
        return verdict(v > claim["value"] if claim["relation"] == "above" else v < claim["value"],
                       f"measured {v:.3f} eV, stated {claim['relation']} {claim['value']:.3f} eV", [eid])
    if form == "comparison":
        a, b = claim["a"], claim["b"]
        if (r := missing(a, b)) is not None:
            return r
        (va, ea), (vb, eb) = good[a], good[b]
        reason = f"{a} {va:.3f} eV vs {b} {vb:.3f} eV"
        if abs(va - vb) <= COMPARISON_MARGIN_EV:
            return {"verdict": "inconclusive", "reason": reason + " (equal within the margin)", "checkable": True,
                    "evidence": []}
        return verdict(va > vb if claim["relation"] == "higher" else va < vb, reason, [ea, eb])
    if form == "motif_effect":
        f = claim["feature"]
        with_f = [(v, i) for k, (v, i) in good.items() if f in chem.features(k)]
        without = [(v, i) for k, (v, i) in good.items() if f not in chem.features(k)]
        if len(with_f) < MIN_GROUP or len(without) < MIN_GROUP:
            return {"verdict": "inconclusive", "checkable": True, "evidence": [],
                    "reason": f"{len(with_f)} result(s) with {f}, {len(without)} without (need {MIN_GROUP} each)"}
        a, b = [v for v, _ in with_f], [v for v, _ in without]
        diff = statistics.mean(a) - statistics.mean(b)
        se = math.sqrt(statistics.variance(a) / len(a) + statistics.variance(b) / len(b))
        t = diff / se if se > 0 else (math.copysign(math.inf, diff) if diff else 0.0)
        reason = (f"mean {statistics.mean(a):.2f} eV with {f} (n={len(a)}) vs {statistics.mean(b):.2f} eV "
                  f"without (n={len(b)}), t={t:.1f}")
        if abs(t) < T_THRESHOLD:
            return {"verdict": "inconclusive", "reason": reason + " (no clear difference)", "checkable": True,
                    "evidence": []}
        return verdict(diff > 0 if claim["direction"] == "increases" else diff < 0, reason,
                       [i for _, i in with_f + without])
    return {"verdict": "inconclusive", "reason": "cannot be checked against the results", "checkable": False,
            "evidence": []}


def in_scope(claim: dict, key: str) -> bool:
    """Whether a new result for this molecule can change the claim's verdict."""
    form = claim.get("form")
    if form in ("value", "bound"):
        return claim["molecule"] == key
    if form == "comparison":
        return key in (claim["a"], claim["b"])
    return form == "motif_effect"            # every result enters one of the two groups


def record(store: MemoryStore, entry_id: str, claim: dict, result: dict, trigger: str) -> str | None:
    """Store a verdict and its evidence edges; returns the previous verdict."""
    old = store.get_verification(entry_id) or {"history": []}
    prev, t = old.get("verdict"), time.time()
    rec = {"claim": claim, "verdict": result["verdict"], "reason": result["reason"],
           "checkable": result["checkable"], "t": t, "history": old.get("history", [])}
    if prev != result["verdict"]:
        rec["history"] = rec["history"] + [{"t": t, "verdict": result["verdict"], "reason": result["reason"],
                                            "trigger": trigger}]
    store.set_verification(entry_id, rec)
    for e in store.graph.outgoing(entry_id, ["SUPPORTED_BY", "CONTRADICTED_BY"]):
        if e["created_by"] == BY:
            store.graph.remove(e["id"])
    rel = {"supported": "SUPPORTED_BY", "contradicted": "CONTRADICTED_BY"}.get(result["verdict"])
    for eid in result["evidence"] if rel else []:
        store.link(entry_id, eid, rel, BY, check=claim.get("form"))
    return prev
