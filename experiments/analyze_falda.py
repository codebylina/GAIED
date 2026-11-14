"""
Analysis of the campaigns of this branch (FALDA baseline, and "none" if it was run).

    python experiments/analyze_falda.py experiments/results/falda

Input : <dir>/runs/*.json (written by experiments/run_falda.py)
Output: <dir>/metrics_runs.csv, <dir>/report.md
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

LLM_NODES = ("plan", "tool_calling", "conclude", "critique", "update")
METRICS = [
    ("best_energy", "best ionization energy (any molecule)"),
    ("best_compliant", "best ionization energy among molecules respecting the Goal"),
    ("top5_mean", "mean of the 5 best"),
    ("xtb_runs", "xTB computations"),
    ("duplicate_rate", "duplicate computations / runs"),
    ("invalid_repeat", "computations of already failed molecules"),
    ("unique_valid", "distinct valid molecules measured"),
    ("scaffolds", "distinct Murcko scaffolds"),
    ("diversity", "mean pairwise Tanimoto distance"),
    ("redundant_proposal_rate", "proposals of already known molecules"),
    ("violation_rate", "proposals breaking a Goal constraint"),
    ("tool_call_failures", "iterations without a parsable tool call (after retries)"),
    ("tool_call_retries", "extra LLM requests needed to get a tool call"),
    ("llm_call_s", "mean duration of an LLM step (s)"),
    ("recall_chars", "memory text added to a prompt (characters)"),
    ("recall_ms", "memory recall time (ms)"),
    ("falda_stream", "FALDA T0 stream messages at the end"),
    ("falda_atoms", "FALDA T1 atoms at the end"),
    ("falda_scenes", "FALDA T2 scenes at the end"),
    ("falda_core_chars", "FALDA T3 core size at the end (characters)"),
    ("wall_time_s", "campaign wall time (s)"),
]


def diversity(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps, scaffolds = [], set()
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fps.append(gen.GetFingerprint(m))
        scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=m) or "acyclic")
    dists = [1 - x for i in range(len(fps)) for x in DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])]
    return len(scaffolds), (st.mean(dists) if dists else None)


def compliant(smiles, constraints):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return False
    if constraints.get("max_heavy_atoms") and m.GetNumHeavyAtoms() > int(constraints["max_heavy_atoms"]):
        return False
    allowed = constraints.get("allowed_elements")
    return not allowed or all(a.GetSymbol() in allowed for a in m.GetAtoms())


def run_metrics(d):
    tel, facts = d["telemetry"], sorted(d["facts_log"], key=lambda f: f["t"])
    constraints = (d.get("goal") or {}).get("constraints", {})
    m = {"condition": d.get("meta", {}).get("condition") or d["mode"], "file": d["_file"],
         "agents": len(d.get("agents", [])), "xtb_runs": len(facts)}
    seen, failed, dup, inv = set(), set(), 0, 0
    for f in facts:
        dup += f["smiles"] in seen
        inv += f["status"] == "error" and f["smiles"] in failed
        if f["status"] == "error":
            failed.add(f["smiles"])
        seen.add(f["smiles"])
    m["duplicate_rate"] = dup / len(facts) if facts else 0
    m["invalid_repeat"] = inv
    ok = {}
    for f in facts:
        if f["status"] == "ok" and f.get("energy") is not None:
            ok.setdefault(f["smiles"], f["energy"])
    energies = sorted(ok.values(), reverse=True)
    m["best_energy"] = energies[0] if energies else None
    m["top5_mean"] = st.mean(energies[:5]) if energies else None
    good = [e for s, e in ok.items() if compliant(s, constraints)]
    m["best_compliant"] = max(good) if good else None
    m["unique_valid"] = len(ok)
    m["scaffolds"], m["diversity"] = diversity(list(ok))
    props = [p for e in tel if e["e"] == "proposals" for p in e["items"]]
    n = len(props) or 1
    m["redundant_proposal_rate"] = sum(p["team_known"] != "new" or p["own_repeat"] for p in props) / n
    m["violation_rate"] = sum(bool(p["violations"]) for p in props) / n
    m["tool_call_failures"] = sum(e["e"] == "tool_call_failed" for e in tel)
    m["tool_call_retries"] = sum(e.get("retries", 0) for e in tel if e["e"] == "proposals")
    llm = [e["duration"] for e in tel if e["e"] == "node" and e["node"] in LLM_NODES]
    m["llm_call_s"] = st.mean(llm) if llm else None
    br = [e for e in tel if e["e"] == "briefing"]
    m["recall_chars"] = st.mean(e["chars"] for e in br) if br else 0
    m["recall_ms"] = st.mean(e.get("ms", 0) for e in br) if br else 0
    for k, v in (d.get("meta", {}).get("falda") or {}).items():
        m[f"falda_{k}"] = v
    m["wall_time_s"] = (max(e["t"] for e in tel) - min(e["t"] for e in tel)) if tel else 0
    return m


def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def main(directory):
    runs = []
    for path in sorted(p for p in glob.glob(os.path.join(directory, "runs", "*.json"))
                       if not p.endswith(".meta.json")):
        with open(path) as fh:
            d = json.load(fh)
        d["_file"] = os.path.basename(path)
        runs.append(d)
    if not runs:
        sys.exit(f"No runs in {directory}/runs")
    rows = [run_metrics(d) for d in runs]
    cols = sorted({k for r in rows for k in r})
    with open(os.path.join(directory, "metrics_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    by = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    conds = sorted(by)
    lines = [f"# Campaign report: `{os.path.basename(os.path.abspath(directory))}`", "",
             "- runs: " + ", ".join(f"{c}: {len(by[c])}" for c in conds), "",
             "| metric | " + " | ".join(f"{c} (mean ± sd)" for c in conds) + " |",
             "|---|" + "---|" * len(conds)]
    for key, label in METRICS:
        cells = []
        for c in conds:
            xs = [r[key] for r in by[c] if r.get(key) is not None]
            cells.append("–" if not xs else
                         f"{fmt(float(st.mean(xs)))} ± {fmt(float(st.stdev(xs)))}" if len(xs) > 1
                         else fmt(float(xs[0])))
        if any(x != "–" for x in cells):
            lines.append(f"| {label} (`{key}`) | " + " | ".join(cells) + " |")
    with open(os.path.join(directory, "report.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
