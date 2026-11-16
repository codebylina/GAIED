"""
Metrics of GAIED campaigns run with a memory condition (no ground truth: no oracle metrics).

    python -m bench.analyze_gaied results/gaied          (reads every *.json written by MEMORY_EXPORT)
"""
from __future__ import annotations

import csv
import glob
import math
import os
import statistics as st
import sys
from collections import defaultdict

from bench.analyze import bootstrap_ci


def metrics(d: dict) -> dict:
    obs = sorted(d["observations"], key=lambda o: o["t"])
    m = {"condition": (d.get("campaign") or {}).get("condition") or d["memory"], "experiments": len(obs)}
    seen, failed, dup, frep = set(), set(), 0, 0
    for o in obs:
        dup += o["entity"] in seen
        frep += o["status"] == "failed" and o["entity"] in failed
        if o["status"] == "failed":
            failed.add(o["entity"])
        seen.add(o["entity"])
    m["distinct"], m["repeat_rate"], m["repeated_failures"] = len(seen), dup / len(obs) if obs else 0.0, frep
    vals = sorted({o["entity"]: o["value"] for o in obs if o["status"] == "ok"}.values(), reverse=True)
    m["best_value"] = vals[0] if vals else math.nan
    m["top5_mean"] = st.mean(vals[:5]) if vals else math.nan
    m["goal_violations"] = sum(1 for o in obs if o.get("violations"))
    dec = [x for b in d.get("decisions", []) for x in b["decisions"]]
    m["avoided_by_memory"] = sum(x["action"] in ("known", "failed_before", "in_progress") for x in dec)
    for purpose, u in (d.get("llm_usage") or {}).items():
        m[f"llm_calls_{purpose}"] = u.get("calls", 0)
        m[f"llm_seconds_{purpose}"] = u.get("seconds", 0.0)
    br = d.get("briefings", [])
    m["briefing_chars"] = st.mean(b["chars"] for b in br) if br else 0.0
    layers = defaultdict(int)
    for c in (d.get("extra") or {}).get("claims", []):
        layers[c["layer"]] += 1
    m.update({f"claims_{k}": v for k, v in layers.items()})
    m.update(knowledge_metrics(d))
    return m


BAD = ("refuted", "unsupported", "contested")
ASSERTED = ("verified", "supported", "established", "accepted")


def knowledge_metrics(d: dict) -> dict:
    """Evidence memory: atoms, their statuses and what the agents were shown. GAIED has no truth evaluator, so a
    claim counts as wrong when the campaign's own measurements contradict it or do not contain the result it
    reports, according to the replayed verification (bench/replay_verify.py) when present, else the run's."""
    ex = d.get("extra") or {}
    atoms = ex.get("atoms")
    if atoms is None:
        return {}
    m = {"journal_messages": len(ex.get("journal", [])), "atoms": len(atoms),
         "restatements": sum(len(a.get("restatements", [])) for a in atoms)}
    run = {x["id"]: x for x in (ex.get("verification") or {}).get("verified_memories", [])}
    for st_ in ("established", "supported", "hypothesis", "open", "contested", "refuted", "unsupported",
                "verified", "unverifiable", "pending"):
        if run:
            m[f"run_{st_}"] = sum(x["status"] == st_ for x in run.values())
    if run:
        lags = [x["history"][0]["t"] - x["registered_at"] for x in run.values() if x.get("history")]
        m["verif_lag_s"] = st.mean(lags) if lags else math.nan
    final = {k: v["status"] for k, v in ((d.get("replay") or {}).get("statuses") or {}).items()}
    source = "replay" if final else "run"
    if not final:
        final = {k: x["status"] for k, x in run.items()}
    if not final:
        return m
    imp = [i for b in d.get("briefings", []) for i in b["items"] if i.get("kind") == "claim" and i["id"] in final]
    shown_true = [i for i in imp if i.get("asserted")]
    m["impressions_shown_true"] = len(shown_true)
    m["impressions_shown_true_bad"] = sum(final[i["id"]] in BAD for i in shown_true)
    m["share_shown_true_bad"] = m["impressions_shown_true_bad"] / len(shown_true) if shown_true else math.nan
    m["atoms_bad_final"] = sum(s in BAD for s in final.values())
    m["atoms_asserted_final"] = sum(s in ASSERTED for s in final.values())
    m["status_source_replay"] = 1 if source == "replay" else 0
    return m


def main(directory: str) -> None:
    import json
    rows = [metrics(json.load(open(p))) for p in sorted(glob.glob(os.path.join(directory, "*.json")))]
    if not rows:
        raise SystemExit("no export found")
    cols = sorted({k for r in rows for k in r})
    with open(os.path.join(directory, "metrics_gaied.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    by = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    conds = sorted(by)
    lines = ["| metric | " + " | ".join(f"{c} (n={len(by[c])})" for c in conds) + " |", "|---|" + "---|" * len(conds)]
    for k in cols:
        if k == "condition":
            continue
        cells = []
        for c in conds:
            xs = [r[k] for r in by[c] if isinstance(r.get(k), (int, float)) and not (isinstance(r[k], float) and math.isnan(r[k]))]
            if not xs:
                cells.append("–")
                continue
            lo, hi = bootstrap_ci(xs)
            cells.append(f"{st.mean(xs):.3g}" + (f" [{lo:.3g}, {hi:.3g}]" if not math.isnan(lo) else ""))
        lines.append(f"| `{k}` | " + " | ".join(cells) + " |")
    with open(os.path.join(directory, "report_gaied.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
