"""
Manual audit of the tests the verifier's LLM wrote: a test that runs is not necessarily a test of the claim.

    python -m bench.audit_tests results/<config name> [--condition mem0_verified] [--n 60] [--seed 0]
    python -m bench.audit_tests --score results/<config name>/audit_tests.csv

The sample is stratified by final status (proportional allocation), over the runs of one condition. A reviewer fills three columns with yes / no (or leaves them empty):
  human_scope_faithful     the scoped statement says what the memory says (not narrower, not inverted)
  human_test_matches       the test and its decision rule check the scoped statement (right rows, right
                           comparison, a reference group for a generalization)
  human_verdict_correct    the final status is right given the measurements of the run
--score reports each share with a 95% Wilson interval, and the agreement of the human verdict with the
verifier and with the evaluator (Cohen's kappa) when the evaluator's verdict is in the cache.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
from collections import defaultdict

ASSERTED = ("verified", "supported", "established", "accepted")
HUMAN = ("human_scope_faithful", "human_test_matches", "human_verdict_correct")


def oracle_truth(cache: dict, d: dict, text: str):
    env_id = json.dumps(d["env"], sort_keys=True, default=str) + "|" + \
        f"{d['scenario']['name']}|{d['condition']['name']}|{d['seed']}"
    hit = cache.get(hashlib.sha1(f"{env_id}|{text.strip()}".encode()).hexdigest())
    return None if not hit else hit.get("truth")


def build(directory: str, n: int, seed: int, condition: str) -> None:
    cache_p = os.path.join(directory, "oracle_cache.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    pool = []
    for p in sorted(glob.glob(os.path.join(directory, "runs", "*.json"))):
        d = json.load(open(p))
        ver = (d.get("memory", {}).get("extra") or {}).get("verification")
        if d.get("status") != "ok" or not ver or d["condition"]["name"] != condition:
            continue
        for x in ver["verified_memories"]:
            if not x.get("code"):
                continue
            ev = x.get("eval") or {}
            truth = oracle_truth(cache, d, x["scoped"])
            pool.append({"run": os.path.basename(p), "memory_id": x["id"], "status": x["status"], "kind": x["kind"],
                         "author": x.get("author"), "original": x["text"], "scoped": x["scoped"],
                         "verdict": ev.get("verdict"), "n_support": len(ev.get("support", [])),
                         "n_contradict": len(ev.get("contradict", [])),
                         "classes": (ev.get("classes") or {}).get("support", ""),
                         "oracle_scoped": "" if truth is None else str(truth).lower(), "code": x["code"],
                         **{h: "" for h in HUMAN}, "notes": ""})
    if not pool:
        raise SystemExit("no formalized memory in these runs")
    rng = random.Random(seed)
    strata = defaultdict(list)
    for r in pool:
        strata[r["status"]].append(r)
    n = min(n or len(pool), len(pool))
    sample = []
    for k, rows in sorted(strata.items()):      # proportional allocation, at least one per status
        rng.shuffle(rows)
        sample += rows[:max(1, round(n * len(rows) / len(pool)))]
    out = os.path.join(directory, "audit_tests.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sample[0]))
        w.writeheader()
        w.writerows(sample)
    print(f"{len(sample)} tests of {len(pool)} written to {out}")


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return math.nan, math.nan
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def kappa(pairs: list[tuple[bool, bool]]) -> float:
    n = len(pairs)
    if not n:
        return math.nan
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else math.nan


def score(path: str) -> None:
    rows = list(csv.DictReader(open(path)))
    yes = lambda v: v.strip().lower() in ("yes", "y", "true", "1")
    filled = lambda v: v.strip().lower() in ("yes", "y", "true", "1", "no", "n", "false", "0")
    for h in HUMAN:
        vals = [yes(r[h]) for r in rows if filled(r[h])]
        lo, hi = wilson(sum(vals), len(vals))
        print(f"{h}: {sum(vals)}/{len(vals)} = {sum(vals) / len(vals) if vals else math.nan:.2f} "
              f"[{lo:.2f}, {hi:.2f}]")
    # human truth of the shown statement: verdict correct and asserted <=> true
    h_true, v_pairs, o_pairs = {}, [], []
    for r in rows:
        if not filled(r["human_verdict_correct"]) or r["status"] not in ASSERTED + ("refuted",):
            continue
        asserted = r["status"] in ASSERTED
        truth = asserted if yes(r["human_verdict_correct"]) else not asserted
        h_true[r["memory_id"]] = truth
        v_pairs.append((truth, asserted))
        if r["oracle_scoped"] in ("true", "false"):
            o_pairs.append((truth, r["oracle_scoped"] == "true"))
    print(f"human vs verifier: kappa {kappa(v_pairs):.2f} on {len(v_pairs)} decided memories")
    print(f"human vs evaluator: kappa {kappa(o_pairs):.2f} on {len(o_pairs)} memories")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?")
    ap.add_argument("--n", type=int, default=0, help="sample size (0: every formalized memory)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", default="mem0_verified")
    ap.add_argument("--score")
    a = ap.parse_args()
    if a.score:
        score(a.score)
    elif a.directory:
        build(a.directory, a.n, a.seed, a.condition)
    else:
        ap.error("give a results directory or --score <csv>")


if __name__ == "__main__":
    main()
