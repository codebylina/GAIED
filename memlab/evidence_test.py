"""
The decision rule of the verifier, written once in code and the same for every memory.

The LLM's test (`def test(df)`) only says WHICH measurements a memory makes a prediction about and WHAT it predicts,
in one of four forms. This module checks that answer, compares the prediction with the measured outcomes and
decides, so that no decision rule is written by the LLM for the claim it is judging:

  pairs      [[a, b], ...]                      value(a) > value(b) for each pair (matched designs)
  groups     {"higher": [...], "lower": [...]}  values in "higher" tend to exceed those in "lower"
  fail_rate  {"more": [...], "less": [...]}     designs in "more" fail more often than those in "less"
  facts      [{"id", "status", "value"}, ...]   the outcome of these measurements (reports, predictions)

Statistics (exact where small, one-sided, against "no effect"):
  pairs      sign test on disjoint pairs (each measurement used once, in the order returned; ties and failures
             left out), so dependent pairs do not inflate the evidence
  groups     Mann-Whitney U test on the values of the measurements that ran
  fail_rate  Fisher's exact test on the failure counts
  facts      each fact agrees or not: same status, and the value written is the measured one rounded or
             truncated at the precision written

The only constant is the significance level `alpha` (0.05 by convention, a setting of the verifier). With it a
sign test needs at least 5 agreeing pairs and no disagreement, so one comparison never establishes anything.
"""
from __future__ import annotations

import math

FORMS = ("pairs", "groups", "fail_rate", "facts")


class TestError(Exception):
    pass


# ---------------------------------------------------------------------------------------------------- parsing
def _id_error(key: str, v) -> TestError:
    what = ("a number: list measurement ids (ids(frame)), not values" if _is_number(v)
            else "not a value of df['id']: list the ids of the measurement rows (ids(frame)), never the memory's id")
    return TestError(f"'{key}' contains {str(v)[:40]!r}, which is {what}")


def _is_number(v) -> bool:
    try:
        float(str(v))
        return True
    except ValueError:
        return False


def _ids(key: str, vals, known: set[str], by_entity: dict[str, list[str]], one: bool) -> list[str]:
    if vals is None:
        return []
    if isinstance(vals, (str, int, float)):
        vals = [vals]
    if not isinstance(vals, (list, tuple)):
        raise TestError(f"'{key}' must be a list of values of df['id']")
    out = []
    for v in vals:
        s = str(v)
        if s in known:
            out.append(s)
        elif s in by_entity:                    # a design key stands for its measurements (the latest, for one)
            out.extend(by_entity[s][-1:] if one else by_entity[s])
        else:
            raise _id_error(key, v)
    return out


def parse(out, known: set[str], by_entity: dict[str, list[str]] | None = None) -> dict:
    """The comparison a test returned, normalized: {"form": ..., ...} or {"form": "untestable"}."""
    by_entity = by_entity or {}
    if not isinstance(out, dict):
        raise TestError("test(df) must return a dict")
    note = str(out.get("note") or "")[:500]
    if out.get("untestable"):
        return {"form": "untestable", "note": note or str(out["untestable"])[:300]}
    given = [f for f in FORMS if out.get(f) is not None]
    if len(given) != 1:
        raise TestError(f"return exactly one of the keys {', '.join(FORMS)} (got {', '.join(given) or 'none'})")
    form = given[0]
    val = out[form]
    res = {"form": form, "note": note}
    if form == "pairs":
        if not isinstance(val, (list, tuple)):
            raise TestError("'pairs' must be a list of [id_a, id_b]")
        pairs = []
        for p in val:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                raise TestError("each element of 'pairs' must be [id_a, id_b]")
            a = _ids("pairs", [p[0]], known, by_entity, one=True)
            b = _ids("pairs", [p[1]], known, by_entity, one=True)
            pairs.append((a[0], b[0]))
        res["pairs"] = pairs
    elif form in ("groups", "fail_rate"):
        hi, lo = ("higher", "lower") if form == "groups" else ("more", "less")
        if not isinstance(val, dict) or hi not in val or lo not in val:
            raise TestError(f"'{form}' must be {{\"{hi}\": [ids], \"{lo}\": [ids]}}")
        a = sorted(set(_ids(f"{form}.{hi}", val[hi], known, by_entity, one=False)))
        b = sorted(set(_ids(f"{form}.{lo}", val[lo], known, by_entity, one=False)))
        if set(a) & set(b):
            raise TestError(f"the same measurements are in '{hi}' and '{lo}'")
        res[hi], res[lo] = a, b
    else:
        if not isinstance(val, (list, tuple)):
            raise TestError("'facts' must be a list of {\"id\", \"status\", \"value\"}")
        facts = []
        for f in val:
            if not isinstance(f, dict) or "id" not in f:
                raise TestError("each fact must be a dict with an 'id'")
            for i in _ids("facts", [f["id"]], known, by_entity, one=True):
                st = f.get("status")
                st = None if st is None else str(st).lower()
                if st not in (None, "ok", "failed"):
                    raise TestError("a fact's status must be 'ok' or 'failed'")
                v = f.get("value")
                if v is not None and not _is_number(v):
                    raise TestError("a fact's value must be a number (as written in the memory) or null")
                facts.append({"id": i, "status": st, "value": None if v is None else str(v)})
        res["facts"] = facts
    return res


def involved(comp: dict) -> list[str]:
    f = comp["form"]
    if f == "pairs":
        return sorted({i for p in comp["pairs"] for i in p})
    if f == "groups":
        return comp["higher"] + comp["lower"]
    if f == "fail_rate":
        return comp["more"] + comp["less"]
    if f == "facts":
        return [x["id"] for x in comp["facts"]]
    return []


# ---------------------------------------------------------------------------------------------------- statistics
def binom_tail(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 1/2)."""
    if n <= 0:
        return 1.0
    return sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n


def fisher_greater(a: int, n1: int, c: int, n2: int) -> float:
    """One-sided Fisher exact test: P(failures in group 1 >= a | margins), a of n1 vs c of n2 failed."""
    k, n = a + c, n1 + n2
    if n1 == 0 or n2 == 0 or k == 0:
        return 1.0
    total = math.comb(n, k)
    return sum(math.comb(n1, x) * math.comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / total


def mann_whitney_greater(x: list[float], y: list[float]) -> float:
    """One-sided Mann-Whitney U: P(U >= u) under 'no difference' (values in x tend to exceed those in y)."""
    if not x or not y:
        return 1.0
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(x, y, alternative="greater").pvalue)
    except ImportError:                          # normal approximation with tie correction
        n1, n2 = len(x), len(y)
        u = sum((xi > yj) + 0.5 * (xi == yj) for xi in x for yj in y)
        allv = sorted(x + y)
        ties = sum(t ** 3 - t for t in (allv.count(v) for v in set(allv)))
        n = n1 + n2
        var = n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1)))
        if var <= 0:
            return 1.0
        z = (u - n1 * n2 / 2 - 0.5) / math.sqrt(var)
        return 0.5 * math.erfc(z / math.sqrt(2))


def _value(r: dict | None):
    if not r or r.get("status") != "ok" or r.get("value") is None:
        return None
    try:
        v = float(r["value"])
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _unit(written: str) -> float:
    s = written.strip().lower()
    mant = s.split("e")[0]
    dec = len(mant.split(".")[1]) if "." in mant else 0
    exp = int(s.split("e")[1]) if "e" in s else 0
    return 10.0 ** (exp - dec)


def same_number(v: float, written: str) -> bool:
    """The written number is v rounded, or truncated, at the precision it is written with."""
    w, u = float(written), _unit(written)
    eps = 1e-9 * max(1.0, abs(v))
    rounded = abs(v - w) <= 0.5 * u + eps
    truncated = -eps <= (v - w) * (1 if v >= 0 else -1) < u + eps
    return rounded or truncated


def stat(comp: dict, by_id: dict) -> dict:
    """The test of a comparison: p_for (the prediction holds beyond chance), p_against (its opposite holds),
    n (independent units used), and the measurements that agree / disagree with the prediction."""
    f = comp["form"]
    agree, disagree = [], []
    if f == "pairs":
        used = set()
        k_for = k_against = 0
        for a, b in comp["pairs"]:
            va, vb = _value(by_id.get(a)), _value(by_id.get(b))
            if a == b or a in used or b in used or va is None or vb is None or va == vb:
                continue
            used |= {a, b}
            if va > vb:
                k_for += 1
                agree += [a, b]
            else:
                k_against += 1
                disagree += [a, b]
        n = k_for + k_against
        return {"test": "sign", "n": n, "k_for": k_for, "k_against": k_against,
                "p_for": binom_tail(k_for, n), "p_against": binom_tail(k_against, n),
                "agree": agree, "disagree": disagree}
    if f == "groups":
        x = [(i, v) for i in comp["higher"] if (v := _value(by_id.get(i))) is not None]
        y = [(i, v) for i in comp["lower"] if (v := _value(by_id.get(i))) is not None]
        xs, ys = [v for _, v in x], [v for _, v in y]
        if xs and ys:
            mx, my = _median(xs), _median(ys)
            agree = [i for i, v in x if v > my] + [i for i, v in y if v < mx]
            disagree = [i for i, v in x if v <= my] + [i for i, v in y if v >= mx]
        return {"test": "mann-whitney", "n": len(xs) + len(ys) if xs and ys else 0, "n_higher": len(xs), "n_lower": len(ys),
                "mean_higher": _mean(xs), "mean_lower": _mean(ys),
                "p_for": mann_whitney_greater(xs, ys), "p_against": mann_whitney_greater(ys, xs),
                "agree": agree, "disagree": disagree}
    if f == "fail_rate":
        more = [i for i in comp["more"] if i in by_id]
        less = [i for i in comp["less"] if i in by_id]
        fm = [i for i in more if by_id[i]["status"] == "failed"]
        fl = [i for i in less if by_id[i]["status"] == "failed"]
        agree = fm + [i for i in less if i not in fl]
        disagree = [i for i in more if i not in fm] + fl
        n1, n2 = len(more), len(less)
        return {"test": "fisher", "n": n1 + n2 if n1 and n2 else 0, "failed_more": len(fm), "n_more": n1, "failed_less": len(fl),
                "n_less": n2, "p_for": fisher_greater(len(fm), n1, len(fl), n2),
                "p_against": fisher_greater(n1 - len(fm), n1, n2 - len(fl), n2),
                "agree": agree, "disagree": disagree}
    if f == "facts":
        for x in comp["facts"]:
            r = by_id.get(x["id"])
            if r is None:
                continue
            ok = True
            if x["status"] is not None and r["status"] != x["status"]:
                ok = False
            if ok and x["value"] is not None:
                v = _value(r)
                ok = v is not None and same_number(v, x["value"])
            (agree if ok else disagree).append(x["id"])
        return {"test": "facts", "n": len(agree) + len(disagree), "agree": agree, "disagree": disagree,
                "p_for": 1.0, "p_against": 1.0}
    raise TestError(f"unknown form {f}")


def restrict(comp: dict, new: set[str]) -> dict | None:
    """The same comparison on the measurements made after the claim (new data only); a side of a group comparison
    with no new measurement keeps all its measurements as the reference. None when nothing is new."""
    f = comp["form"]
    if f == "pairs":
        pairs = [p for p in comp["pairs"] if p[0] in new or p[1] in new]
        return {**comp, "pairs": pairs} if pairs else None
    if f in ("groups", "fail_rate"):
        hi, lo = ("higher", "lower") if f == "groups" else ("more", "less")
        a, b = [i for i in comp[hi] if i in new], [i for i in comp[lo] if i in new]
        if not a and not b:
            return None
        return {**comp, hi: a or comp[hi], lo: b or comp[lo]}
    return None


def _median(v: list[float]) -> float:
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _mean(v: list[float]):
    return sum(v) / len(v) if v else None
