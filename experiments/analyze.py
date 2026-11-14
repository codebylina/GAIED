"""
Analysis of EVIDENT experiments: metrics per run, aggregation per condition, statistical
comparisons, figures and a Markdown report.

    python experiments/analyze.py experiments/results/<experiment>

Input : <dir>/runs/*.json   (exports written by run_matrix.py or synthetic.py)
Output: <dir>/metrics_runs.csv, <dir>/metrics_conditions.csv, <dir>/report.md, <dir>/fig_*.png

Ground truth for claims
  synthetic runs : the known rules of the fake xTB (meta.true_effects)
  real runs      : the effect of each feature measured on the POOL of all facts of all
                   runs of the experiment (much more data than any single run)
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evident.features import FEATURE_NAMES, canonical_smiles, mol_info  # noqa: E402
from evident.verifier import SHOWN_AS_KNOWLEDGE, VerifierParams, ok_facts, reference_effects  # noqa: E402
from experiments.conditions import COLORS, CONDITIONS, EXTERNAL_CONDITIONS, QUESTIONS  # noqa: E402

KEY_METRICS = [
    ("best_at_budget", "best IE within the common budget of xTB runs", "higher"),
    ("best_energy", "best IE of the campaign", "higher"),
    ("top5_mean", "mean of the 5 best", "higher"),
    ("xtb_runs", "xTB computations", "lower"),
    ("duplicate_rate", "duplicate computations / runs", "lower"),
    ("redundant_proposal_rate", "proposals of already known molecules", "lower"),
    ("invalid_repeat", "computations of already failed molecules", "lower"),
    ("taken_at_proposal_rate", "proposals of molecules already reserved by another agent", "lower"),
    ("wait_s", "time an agent waited for a molecule computed by another (s)", "lower"),
    ("notify_ms", "fact published -> received by the other agents (ms)", "lower"),
    ("violation_rate", "proposals breaking a Goal constraint", "lower"),
    ("unique_valid", "distinct valid molecules measured", "higher"),
    ("scaffolds", "distinct Murcko scaffolds", "higher"),
    ("diversity", "mean pairwise Tanimoto distance", "higher"),
    ("anchoring", "similarity of each proposal to the team's 10 best molecules known at that time", "lower"),
    ("near_top_rate", "proposals very similar (Tanimoto >= 0.7) to one of the team's 10 best", "lower"),
    ("assertion_accuracy", "claims asserted by agents that are true", "higher"),
    ("exposure_precision", "knowledge SHOWN to agents that is true", "higher"),
    ("false_exposures", "false knowledge items shown to agents", "lower"),
    ("injection_adoption_delta", "share of proposals with the injected feature (after - before)", "lower"),
    ("injected_exposures", "times the injected claim was shown as knowledge", "lower"),
    ("stops", "STOP directives received by agents", ""),
    ("cancelled", "computations cancelled by a STOP (before or during)", ""),
    ("cancelled_valid", "cancelled computations of molecules that respected the Goal", "lower"),
    ("stop_delivery_s", "STOP sent -> received by the agent (s)", "lower"),
    ("stop_reaction_s", "STOP sent -> running activity cancelled (s)", "lower"),
    ("detection_s", "event -> STOP sent (s)", "lower"),
    ("violation_rate_after_stop", "violation rate after the first STOP", "lower"),
    ("tool_call_failures", "iterations without a parsable tool call (after retries)", "lower"),
    ("tool_call_retries", "extra LLM requests needed to get a tool call", "lower"),
    ("briefing_ms", "briefing build time (ms)", "lower"),
    ("briefing_chars", "briefing size (characters)", ""),
    ("memory_overhead", "memory time / wall time", "lower"),
    ("wall_time_s", "campaign wall time (s)", ""),
]


# ------------------------------------------------------------------ helpers
def _fp(smiles, _cache={}):
    """Morgan fingerprint (radius 2, 2048 bits), None for an invalid SMILES (cached)."""
    if smiles not in _cache:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        m = Chem.MolFromSmiles(smiles or "")
        _cache[smiles] = gen.GetFingerprint(m) if m is not None else None
    return _cache[smiles]


def anchoring(props, facts, top=10, near=0.7):
    """How much proposals stick to the best molecules the team knew when they were made.

    For each valid proposal, the team's `top` best measured molecules at that moment are
    rebuilt from the facts (in every mode, so that isolated agents give the reference:
    what they WOULD have been shown); the score is the highest Tanimoto similarity to one
    of them. Exact repeats of measured molecules are left out (they are counted by the
    redundant-proposal rate). Returns (mean score, share of proposals with a score >= near)."""
    from rdkit import DataStructs
    ok = sorted((f for f in facts if f["status"] == "ok"), key=lambda f: f["t"])
    scores = []
    for _, t, p in props:
        if not p.get("valid"):
            continue
        known = {}
        for f in ok:
            if f["t"] >= t:
                break
            known.setdefault(f["smiles"], f["energy"])
        if p["smiles"] in known:         # an exact repeat is counted as redundant, not here
            continue
        best = sorted(known, key=known.get, reverse=True)[:top]
        fp = _fp(p["smiles"])
        fps = [x for x in (_fp(s) for s in best) if x is not None]
        if fp is None or not fps:
            continue
        scores.append(max(DataStructs.BulkTanimotoSimilarity(fp, fps)))
    if not scores:
        return None, None
    return float(np.mean(scores)), float(np.mean([s >= near for s in scores]))


def _fingerprints(smiles):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps, scaffolds = [], set()
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fps.append(gen.GetFingerprint(m))
        scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=m) or "acyclic")
    if len(fps) < 2:
        return len(scaffolds), None
    dists = []
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        dists += [1 - x for x in sims]
    return len(scaffolds), float(np.mean(dists))


def load_runs(directory):
    runs = []
    for path in sorted(p for p in glob.glob(os.path.join(directory, "runs", "*.json"))
                       if not p.endswith(".meta.json")):
        with open(path) as fh:
            d = json.load(fh)
        d["_file"] = os.path.basename(path)
        _fill_features(d)
        d["_condition"] = d.get("meta", {}).get("condition") or d["_file"].rsplit("_r", 1)[0]
        runs.append(d)
    return runs


def _fill_features(d):
    """Runs recorded outside EVIDENT (e.g. the FALDA baseline) store no chemical features:
    compute them here with the same vocabulary, so every metric is computed identically."""
    for f in d.get("facts_log", []):
        if not f.get("features"):
            info = mol_info(f["smiles"])
            f["features"], f["heavy_atoms"] = info["features"], info["heavy_atoms"]
    for e in d.get("telemetry", []):
        if e.get("e") == "proposals":
            for p in e["items"]:
                if not p.get("features"):
                    p["features"] = mol_info(p["smiles"])["features"]
        if e.get("e") == "briefing" and e.get("text") and not e.get("lesson_claims"):
            # free-text memory (FALDA): claims found by the conservative keyword parser
            from evident.claims import parse_claims_keywords
            e["lesson_claims"] = parse_claims_keywords(e["text"])


def ground_truth(runs):
    meta = runs[0].get("meta", {})
    if meta.get("synthetic") and meta.get("true_effects"):
        te = meta["true_effects"]
        return {f: ("up" if te.get(f, 0) > 0 else "down" if te.get(f, 0) < 0 else "none")
                for f in FEATURE_NAMES}, "known rules of the synthetic xTB"
    pool = []
    for d in runs:
        pool += d["facts_log"]
    ref = reference_effects(pool, list(FEATURE_NAMES), VerifierParams())
    return {f: v["truth"] for f, v in ref.items()}, f"pooled facts of {len(runs)} runs"


def _correct(claim, truth):
    t = truth.get(claim["feature"], "unknown")
    if t == "unknown":
        return None
    return claim["direction"] == t


# ------------------------------------------------------------------ metrics of one run
def run_metrics(d, truth, budget):
    tel = d["telemetry"]
    t0 = min(e["t"] for e in tel) if tel else 0
    m = {"condition": d["_condition"], "file": d["_file"], "mode": d["mode"],
         "agents": len(d.get("agents", []))}
    facts = sorted(d["facts_log"], key=lambda f: f["t"])
    m["xtb_runs"] = len(facts)
    seen, dup, invalid_repeat, failed_before = set(), 0, 0, set()
    curve, best = [], -math.inf
    for f in facts:
        if f["smiles"] in seen:
            dup += 1
        if f["status"] == "error" and f["smiles"] in failed_before:
            invalid_repeat += 1
        if f["status"] == "error":
            failed_before.add(f["smiles"])
        seen.add(f["smiles"])
        if f["status"] == "ok":
            best = max(best, f["energy"])
        curve.append(best if best > -math.inf else None)
    m["duplicates"] = dup
    m["duplicate_rate"] = dup / len(facts) if facts else 0
    m["invalid_repeat"] = invalid_repeat
    ok = ok_facts(facts)
    energies = sorted((f["energy"] for f in ok), reverse=True)
    m["best_energy"] = energies[0] if energies else None
    m["top5_mean"] = float(np.mean(energies[:5])) if energies else None
    m["best_at_budget"] = next((c for c in reversed(curve[:budget]) if c is not None), None)
    m["unique_valid"] = len(ok)
    m["scaffolds"], m["diversity"] = _fingerprints([f["smiles"] for f in ok])
    m["_curve"] = curve
    m["wall_time_s"] = (max(e["t"] for e in tel) - t0) if tel else 0

    # proposals (decisions logged BEFORE computing)
    props = [(e["agent"], e["t"], p) for e in tel if e["e"] == "proposals" for p in e["items"]]
    n = len(props) or 1
    m["proposals"] = len(props)
    m["redundant_proposal_rate"] = sum(p["team_known"] in ("computed", "in_progress", "failed")
                                       or p["own_repeat"] for _, _, p in props) / n
    m["violation_rate"] = sum(bool(p["violations"]) for _, _, p in props) / n
    m["anchoring"], m["near_top_rate"] = anchoring(props, facts)

    # claims asserted, and knowledge shown
    asserted = [c for e in tel if e["e"] == "conclusion" and not e.get("liar") for c in e["claims"]]
    judged = [x for x in (_correct(c, truth) for c in asserted) if x is not None]
    m["claims_asserted"] = len(asserted)
    m["assertion_accuracy"] = float(np.mean(judged)) if judged else None
    shown = []
    for e in tel:
        if e["e"] != "briefing":
            continue
        for c in e.get("shown_claims", []):
            if c["status"] in SHOWN_AS_KNOWLEDGE:
                f, dr = c["id"].rsplit(":", 1)
                shown.append({"feature": f, "direction": dr})
        shown += e.get("lesson_claims", [])
    sj = [x for x in (_correct(c, truth) for c in shown) if x is not None]
    m["knowledge_exposures"] = len(shown)
    m["exposure_precision"] = float(np.mean(sj)) if sj else None
    m["false_exposures"] = sum(1 for x in sj if not x)

    # fault injection
    inj = [e for e in tel if e["e"] == "injection"]
    if inj:
        feat = inj[0]["claims"][0]["feature"]
        t_inj = inj[0]["t"]
        liar = d.get("liar")
        before = [feat in p["features"] for a, t, p in props if a != liar and t < t_inj]
        after = [feat in p["features"] for a, t, p in props if a != liar and t >= t_inj]
        m["injected_feature"] = feat
        m["injection_adoption_before"] = float(np.mean(before)) if before else None
        m["injection_adoption_after"] = float(np.mean(after)) if after else None
        if before and after:
            m["injection_adoption_delta"] = m["injection_adoption_after"] - m["injection_adoption_before"]
        m["injected_exposures"] = sum(1 for c in shown if c["feature"] == feat and c["direction"] == "up")

    # real-time control
    sent = [e for e in tel if e["e"] == "directive_sent"]
    applied = [e for e in tel if e["e"] == "stop_applied"]
    received = [e for e in tel if e["e"] == "directive_received" and e["kind"] == "stop"]
    m["directives"] = len(sent)
    m["stops"] = len(received)
    cancelled = [e for e in tel if e["e"] == "compute"
                 and e["outcome"] in ("cancelled", "cancelled_before_start")]
    m["cancelled"] = len(cancelled)
    violating = {p["smiles"] for _, _, p in props if p["violations"]}
    m["cancelled_valid"] = sum(1 for e in cancelled if e["smiles"] not in violating)
    if received:
        m["stop_delivery_s"] = float(np.mean([e["delivery"] for e in received]))
    if applied:
        m["stop_reaction_s"] = float(np.mean([e["reaction"] for e in applied]))
    det = [e["sent_at"] - e["trigger_t"] for e in sent if e.get("trigger_t")]
    if det:
        m["detection_s"] = float(np.mean(det))
    stops_sent = {e["target"]: e["sent_at"] for e in reversed(sent) if e["kind"] == "stop"}
    if stops_sent:
        after = [bool(p["violations"]) for a, t, p in props if a in stops_sent and t > stops_sent[a]]
        m["violation_rate_after_stop"] = float(np.mean(after)) if after else None

    # coordination: reservations at proposal time, waits, notification latency
    reserved = [p for _, _, p in props if "reservation" in p]
    if reserved:
        m["taken_at_proposal_rate"] = sum(p["reservation"] == "taken" for p in reserved) / len(reserved)
    waits = [e["wait"] for e in tel if e["e"] == "compute" and e["outcome"] == "waited"]
    if waits:
        m["wait_s"] = float(np.mean(waits))
    lat = [(e["notify_ms_mean"], e["notify_n"]) for e in tel
           if e["e"] == "iteration_end" and e.get("notify_ms_mean") is not None]
    if lat:
        m["notify_ms"] = sum(x * n for x, n in lat) / sum(n for _, n in lat)

    # LLM output format
    m["tool_call_failures"] = sum(1 for e in tel if e["e"] == "tool_call_failed")
    m["tool_call_retries"] = sum(e.get("retries", 0) for e in tel if e["e"] == "proposals")

    # overhead
    br = [e for e in tel if e["e"] == "briefing"]
    m["briefing_ms"] = float(np.mean([e.get("ms", 0) for e in br])) if br else 0
    m["briefing_chars"] = float(np.mean([e["chars"] for e in br])) if br else 0
    mem_s = sum(e.get("ms", 0) for e in br) / 1000 + sum(e.get("duration", 0) for e in tel if e["e"] == "verify")
    m["memory_overhead"] = mem_s / m["wall_time_s"] if m["wall_time_s"] else 0
    return m


# ------------------------------------------------------------------ statistics
def bootstrap_ci(xs, n=2000, seed=0):
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return (float(xs[0]), float(xs[0])) if len(xs) else (None, None)
    rng = np.random.default_rng(seed)
    means = rng.choice(xs, (n, len(xs))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cliffs_delta(a, b):
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def compare(a, b):
    from scipy.stats import mannwhitneyu
    if len(a) < 2 or len(b) < 2:
        return None, cliffs_delta(a, b) if a and b else None
    try:
        p = mannwhitneyu(a, b, alternative="two-sided").pvalue
    except ValueError:
        p = None
    return p, cliffs_delta(a, b)


# ------------------------------------------------------------------ figures
def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#b5b4ae")
    ax.spines["bottom"].set_color("#b5b4ae")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)


def figures(by_cond, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelcolor": "#52514e",
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    conds = [c for c in list(CONDITIONS) + list(EXTERNAL_CONDITIONS) if c in by_cond]
    files = []

    # 1. best value found vs cost
    main = [c for c in conds if "inject" not in c]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for c in main:
        curves = [r["_curve"] for r in by_cond[c]]
        L = min(len(x) for x in curves)
        if L == 0:
            continue
        arr = np.array([[v if v is not None else np.nan for v in x[:L]] for x in curves], float)
        med = np.nanmedian(arr, axis=0)
        x = np.arange(1, L + 1)
        ax.plot(x, med, color=COLORS[c], linewidth=2, label=c)
        if len(curves) > 1:
            ax.fill_between(x, np.nanpercentile(arr, 25, axis=0), np.nanpercentile(arr, 75, axis=0),
                            color=COLORS[c], alpha=0.15, linewidth=0)
    _style(ax)
    ax.set_xlabel("xTB computations (cost)")
    ax.set_ylabel("best ionization energy so far")
    ax.set_title("Best molecule found vs. computational cost (median, IQR band)", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    files.append("fig_best_vs_cost.png")
    fig.savefig(os.path.join(out, files[-1]), dpi=200)
    plt.close(fig)

    # 2. wasted work, small multiples
    panels = [("duplicate_rate", "duplicate computations"),
              ("redundant_proposal_rate", "redundant proposals"),
              ("violation_rate", "off-goal proposals")]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), sharey=False)
    for ax, (key, title) in zip(axes, panels):
        vals = [np.mean([r[key] for r in by_cond[c]]) for c in conds]
        errs = [np.std([r[key] for r in by_cond[c]]) for c in conds]
        ax.bar(range(len(conds)), vals, yerr=errs, color=[COLORS[c] for c in conds], width=0.7,
               edgecolor="#fcfcfb", linewidth=2, error_kw={"elinewidth": 0.8, "ecolor": "#52514e"})
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=7)
        ax.set_title(title, loc="left")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        _style(ax)
    fig.suptitle("Wasted or off-goal work (mean ± sd over runs)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    files.append("fig_waste.png")
    fig.savefig(os.path.join(out, files[-1]), dpi=200)
    plt.close(fig)

    # 3. quality of shared knowledge, and robustness to an injected false claim
    kc = [c for c in conds if c.startswith(("naive", "evident", "falda"))]
    inj = [c for c in conds if "inject" in c]
    fig, axes = plt.subplots(1, 2 if inj else 1, figsize=(8 if inj else 4.5, 3.2), squeeze=False)
    ax = axes[0][0]
    vals = []
    for c in kc:                         # nan when nothing was shown to the agents
        xs = [r["exposure_precision"] for r in by_cond[c] if r.get("exposure_precision") is not None]
        vals.append(float(np.mean(xs)) if xs else np.nan)
    ax.bar(range(len(kc)), vals, color=[COLORS[c] for c in kc], width=0.7, edgecolor="#fcfcfb", linewidth=2)
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=8, color="#0b0b0b")
    ax.set_xticks(range(len(kc)))
    ax.set_xticklabels(kc, rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Knowledge shown to agents that is true", loc="left")
    _style(ax)
    if inj:
        ax = axes[0][1]
        vals = [np.nanmean([r.get("injected_exposures", np.nan) for r in by_cond[c]]) for c in inj]
        ax.bar(range(len(inj)), vals, color=[COLORS[c] for c in inj], width=0.6, edgecolor="#fcfcfb", linewidth=2)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=8, color="#0b0b0b")
        ax.set_xticks(range(len(inj)))
        ax.set_xticklabels(inj, fontsize=7)
        ax.set_title("Times the false claim was shown as knowledge", loc="left")
        _style(ax)
    fig.tight_layout()
    files.append("fig_knowledge.png")
    fig.savefig(os.path.join(out, files[-1]), dpi=200)
    plt.close(fig)
    return files


# ------------------------------------------------------------------ report
def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def main(directory):
    runs = load_runs(directory)
    if not runs:
        sys.exit(f"No runs in {directory}/runs")
    truth, truth_src = ground_truth(runs)
    budget = min(len(d["facts_log"]) for d in runs) or 1
    rows = [run_metrics(d, truth, budget) for d in runs]
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    order = list(CONDITIONS) + list(EXTERNAL_CONDITIONS)
    conds = [c for c in order if c in by_cond] + [c for c in by_cond if c not in order]

    cols = sorted({k for r in rows for k in r if not k.startswith("_")})
    with open(os.path.join(directory, "metrics_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})

    agg = {}
    for c in conds:
        agg[c] = {}
        for key, _, _ in KEY_METRICS:
            xs = [r[key] for r in by_cond[c] if r.get(key) is not None]
            if xs:
                lo, hi = bootstrap_ci(xs)
                agg[c][key] = {"mean": float(np.mean(xs)), "sd": float(np.std(xs)), "n": len(xs),
                               "lo": lo, "hi": hi, "values": xs}
    with open(os.path.join(directory, "metrics_conditions.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "metric", "mean", "sd", "ci95_low", "ci95_high", "n_runs"])
        for c in conds:
            for key, v in agg[c].items():
                w.writerow([c, key, v["mean"], v["sd"], v["lo"], v["hi"], v["n"]])

    figs = figures(by_cond, directory)
    L = [f"# EVIDENT experiment report: `{os.path.basename(os.path.abspath(directory))}`", "",
         f"- runs: {len(rows)} ({', '.join(f'{c}: {len(by_cond[c])}' for c in conds)})",
         f"- common budget for `best_at_budget`: first {budget} xTB computations of each run",
         f"- ground truth for claims: {truth_src}",
         "- ground-truth directions: " + ", ".join(f"{f} {t}" for f, t in truth.items() if t in ("up", "down")),
         "", "## Metrics per condition (mean, 95% bootstrap CI)", ""]
    L.append("| metric | " + " | ".join(conds) + " |")
    L.append("|---|" + "---|" * len(conds))
    for key, label, _ in KEY_METRICS:
        cells = []
        for c in conds:
            v = agg[c].get(key)
            cells.append("–" if v is None else f"{fmt(v['mean'])} [{fmt(v['lo'])}, {fmt(v['hi'])}]")
        if any(x != "–" for x in cells):
            L.append(f"| {label} (`{key}`) | " + " | ".join(cells) + " |")
    L += ["", "## Research questions (Mann-Whitney U, Cliff's delta: B vs A)", "",
          "| question | metric | A | B | mean A | mean B | p | delta | better? |",
          "|---|---|---|---|---|---|---|---|---|"]
    for q, (a, b) in QUESTIONS.items():
        if a not in agg or b not in agg:
            continue
        for key, label, better in KEY_METRICS:
            if key not in agg[a] or key not in agg[b] or not better:
                continue
            va, vb = agg[a][key]["values"], agg[b][key]["values"]
            p, delta = compare(vb, va)
            ma, mb = np.mean(va), np.mean(vb)
            improved = (mb > ma) if better == "higher" else (mb < ma)
            if ma == mb:
                continue
            L.append(f"| {q} | {key} | {a} | {b} | {fmt(float(ma))} | {fmt(float(mb))} | "
                     f"{fmt(p)} | {fmt(delta)} | {'yes' if improved else 'no'} |")
    L += ["", "With few repetitions per condition, p-values are indicative only; report effect sizes "
          "and confidence intervals, and increase the repetitions for the paper.", "",
          "## Figures", ""] + [f"![{f}]({f})" for f in figs]
    with open(os.path.join(directory, "report.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L[:8]))
    print(f"\nreport: {os.path.join(directory, 'report.md')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "experiments", "results", "synthetic"))
