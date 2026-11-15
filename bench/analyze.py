"""
Metrics of the benchmark runs, with paired statistics across environment seeds.

    python -m bench.analyze results/<config name> [--reference mem0] [--no-oracle]

Per run:
  discovery     regret of the best design found (optimum minus its true value), its mean over the
                run (area under the regret curve, lower is better), experiments needed to reach the
                best design found
  efficiency    experiments run, repeated experiments, repeated failures, experiments avoided by the memory
  knowledge     (oracle) share of false items among the knowledge impressions presented as true,
                precision of the knowledge presented as true at the end, ProvMem precision per layer,
                reach of planted false statements (adversarial scenario)
  deception     share of experiments outside the start region that use the decoy option
  cost          LLM calls / characters / seconds per purpose, briefing size, wall time, judging lag
Across seeds: mean, 95% bootstrap interval, and a paired Wilcoxon test against the reference condition.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import statistics as st
from collections import defaultdict

from bench.env import EnvSpec, SyntheticEnv
from bench.oracle import Oracle
from memlab.llm import LLM, LLMUsage

ASSERTED_LAYERS = ("supported", "established", "verified", "accepted", "stored", "live")


def regret_curve(obs: list[dict], env: SyntheticEnv, optimum: float) -> list[float]:
    best_true, curve = -math.inf, []
    for o in sorted(obs, key=lambda o: o["t"]):
        combo = env.parse(o["entity"])
        tv = env.true_value(combo) if combo else None
        if tv is not None:
            best_true = max(best_true, tv)
        curve.append(optimum - best_true if best_true > -math.inf else math.nan)
    return curve


def run_metrics(d: dict, oracle: Oracle | None) -> dict:
    env = SyntheticEnv(EnvSpec(**d["env"]))
    mem = d["memory"]
    obs = mem["observations"]
    m = {"scenario": d["scenario"]["name"], "condition": d["condition"]["name"], "seed": d["seed"]}
    # ---------------------------------------------------------------- efficiency
    seen, failed, dup, frep = set(), set(), 0, 0
    for o in sorted(obs, key=lambda o: o["t"]):
        dup += o["entity"] in seen
        frep += o["status"] == "failed" and o["entity"] in failed
        if o["status"] == "failed":
            failed.add(o["entity"])
        seen.add(o["entity"])
    m["experiments"] = len(obs)
    m["distinct"] = len(seen)
    m["repeat_rate"] = dup / len(obs) if obs else 0.0
    m["repeated_failures"] = frep
    dec = [x for b in mem.get("decisions", []) for x in b["decisions"]]
    m["avoided_by_memory"] = sum(x["action"] in ("known", "failed_before", "in_progress") for x in dec)
    # ---------------------------------------------------------------- discovery
    curve = regret_curve(obs, env, d["optimum"])
    valid = [c for c in curve if not math.isnan(c)]
    m["final_regret"] = valid[-1] if valid else math.nan
    m["mean_regret"] = st.mean(valid) if valid else math.nan
    m["curve"] = curve
    # ---------------------------------------------------------------- deception
    if d.get("decoy"):
        rs, ro = d["start_region"]
        ds, do = d["decoy"]
        out = [env.parse(o["entity"]) for o in obs if env.parse(o["entity"])]
        out = [c for c in out if c[env.slots.index(rs)] != ro]
        m["outside_region_share"] = len(out) / len(obs) if obs else 0.0
        m["decoy_outside"] = (sum(c[env.slots.index(ds)] == do for c in out) / len(out)) if out else math.nan
    # ---------------------------------------------------------------- cost
    usage = mem.get("llm_usage", {})
    for purpose in ("agent", "judge", "memory"):
        u = usage.get(purpose, {})
        m[f"llm_calls_{purpose}"] = u.get("calls", 0)
        m[f"llm_seconds_{purpose}"] = u.get("seconds", 0.0)
        m[f"llm_chars_{purpose}"] = u.get("prompt_chars", 0) + u.get("completion_chars", 0)
    br = mem.get("briefings", [])
    m["briefing_chars"] = st.mean(b["chars"] for b in br) if br else 0.0
    m["wall_time_s"] = d.get("wall_time_s", 0.0)
    extra = mem.get("extra", {})
    ver = extra.get("verification")
    if ver:
        vm = ver.get("verified_memories", [])
        m["verif_memories"] = len(vm)
        for st_ in ("verified", "established", "supported", "hypothesis", "refuted", "unsupported", "unverifiable",
                    "pending", "accepted"):
            m[f"verif_{st_}"] = sum(x["status"] == st_ for x in vm)
        lags = [x["history"][0]["t"] - x["registered_at"] for x in vm if x.get("history")]
        m["verif_lag_s"] = st.mean(lags) if lags else math.nan
        m["verif_repairs"] = sum(bool(x.get("repaired")) for x in vm)
    if "mem0_ops" in extra:
        ops = extra["mem0_ops"]
        ok = [o for o in ops if "error" not in o]
        m["mem0_writes"] = len(ops)
        m["mem0_write_errors"] = len(ops) - len(ok)
        m["mem0_write_s"] = st.mean(o["seconds"] for o in ok) if ok else math.nan
        m["mem0_queue_wait_s"] = st.mean(o["queued_s"] for o in ok) if ok else math.nan
        m["mem0_memories"] = len(extra.get("mem0_memories") or [])
        for k, v in (extra.get("mem0_features") or {}).items():
            m[f"mem0_{k}"] = int(bool(v))
    if "claims" in extra:
        lags = []
        for c in extra["claims"]:
            judged = [h["t"] for h in c.get("history", []) if h["from"] == "pending"]
            if judged:
                lags.append(judged[0] - c["registered_at"])
        m["judging_lag_s"] = st.mean(lags) if lags else math.nan
        layers = defaultdict(int)
        for c in extra["claims"]:
            layers[c["layer"]] += 1
        for k, v in layers.items():
            m[f"claims_{k}"] = v
    # ---------------------------------------------------------------- knowledge (oracle)
    if oracle is not None:
        measured = defaultdict(list)
        for o in obs:
            if o["status"] == "ok" and o["value"] is not None:
                measured[o["entity"]].append(o["value"])
        run_key = f"{m['scenario']}|{m['condition']}|{m['seed']}"
        imp_total = imp_false = hyp_total = hyp_false = 0
        texts = {}
        for b in br:
            for it in b["items"]:
                if it["kind"] != "claim":
                    continue
                t = it["text"]
                if t not in texts:
                    texts[t] = oracle.truth(env, t, measured, run_key)
                truth = texts[t]
                if truth is None:
                    continue
                if it.get("asserted") is None:      # shown as a hypothesis or as unverified advice
                    hyp_total += 1
                    hyp_false += not truth
                    continue
                imp_total += 1
                imp_false += (truth != bool(it["asserted"]))
        m["false_exposure"] = imp_false / imp_total if imp_total else math.nan
        m["knowledge_impressions"] = imp_total
        m["true_impressions"] = imp_total - imp_false
        m["false_share_unasserted"] = hyp_false / hyp_total if hyp_total else math.nan
        # distinct memories the agents were shown, with their final status
        tp = fp = fn = 0
        by_layer = defaultdict(lambda: [0, 0])
        for k in mem.get("knowledge", []):
            truth = oracle.truth(env, k["text"], measured, run_key)
            if truth is None:
                continue
            layer = k.get("final_layer", "stored")
            by_layer[layer][0] += truth
            by_layer[layer][1] += 1
            asserted = layer in ASSERTED_LAYERS
            tp += asserted and truth
            fp += asserted and not truth
            fn += (not asserted) and truth
        m["knowledge_true"] = tp + fn
        m["knowledge_false"] = sum(n - t for t, n in by_layer.values())
        m["final_precision"] = tp / (tp + fp) if tp + fp else math.nan
        m["final_recall"] = tp / (tp + fn) if tp + fn else math.nan
        pr, rc = m["final_precision"], m["final_recall"]
        m["final_f1"] = 2 * pr * rc / (pr + rc) if pr == pr and rc == rc and pr + rc else math.nan
        for layer, (t, n) in by_layer.items():
            m[f"precision_{layer}"] = t / n if n else math.nan
        planted = list(d.get("false_statements") or [])
        if planted:
            reach = sum(1 for b in br for it in b["items"]
                        if it.get("asserted") and any(p[:40] in it["text"] or it["text"][:40] in p for p in planted))
            m["planted_impressions"] = reach
            judged = [oracle.truth(env, p, measured, run_key) for p in planted]   # false by construction
            known = [t for t in judged if t is not None]
            m["oracle_accuracy_on_planted"] = sum(not t for t in known) / len(known) if known else math.nan
    return m


def bootstrap_ci(xs: list[float], reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    if len(xs) < 2:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(xs, k=len(xs))) for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def wilcoxon_p(a: list[float], b: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 2 or all(x == y for x, y in pairs):
        return math.nan
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon([x for x, _ in pairs], [y for _, y in pairs]).pvalue)
    except Exception:
        return math.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--reference", default="mem0")
    ap.add_argument("--no-oracle", action="store_true")
    args = ap.parse_args()
    runs = []
    for p in sorted(glob.glob(os.path.join(args.directory, "runs", "*.json"))):
        d = json.load(open(p))
        if d.get("status") == "ok":
            runs.append(d)
    if not runs:
        raise SystemExit("no successful run")
    oracle = None
    if not args.no_oracle:
        scripted = os.environ.get("LLM_PROVIDER", "").lower() == "scripted"
        if scripted:
            from bench.scripted import ScriptedLLM
            llm = LLM(provider="scripted", scripted=ScriptedLLM(SyntheticEnv(EnvSpec(**runs[0]["env"]))), usage=LLMUsage())
        else:
            llm = LLM(usage=LLMUsage(), temperature=0.0)
        oracle = Oracle(llm, os.path.join(args.directory, "oracle_cache.json"))
    rows = [run_metrics(d, oracle) for d in runs]
    if oracle:
        oracle.save()
    curves = {(r["scenario"], r["condition"], r["seed"]): r.pop("curve") for r in rows}
    # regret after the same number of experiments for every condition of a (scenario, seed):
    # the smallest number of experiments any condition ran (the memory may avoid experiments)
    for r in rows:
        peers = [len(c) for (sc, _, sd), c in curves.items() if sc == r["scenario"] and sd == r["seed"] and c]
        n = min(peers) if peers else 0
        c = curves[(r["scenario"], r["condition"], r["seed"])]
        r["regret_at_common_budget"] = c[n - 1] if n and len(c) >= n else math.nan
    cols = sorted({k for r in rows for k in r})
    with open(os.path.join(args.directory, "metrics_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in curves.items()},
              open(os.path.join(args.directory, "curves.json"), "w"))
    metrics = [c for c in cols if c not in ("scenario", "condition", "seed")]
    lines = [f"# Benchmark report: `{os.path.basename(os.path.abspath(args.directory))}`", ""]
    for scen in sorted({r["scenario"] for r in rows}):
        sr = [r for r in rows if r["scenario"] == scen]
        conds = sorted({r["condition"] for r in sr})
        lines += [f"## Scenario `{scen}`", "",
                  "| metric | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
        ref = {r["seed"]: r for r in sr if r["condition"] == args.reference}
        for mt in metrics:
            cells = []
            for c in conds:
                cr = {r["seed"]: r for r in sr if r["condition"] == c}
                xs = [cr[s].get(mt, math.nan) for s in sorted(cr)]
                xs = [x for x in xs if isinstance(x, (int, float))]
                if not xs or all(isinstance(x, float) and math.isnan(x) for x in xs):
                    cells.append("–")
                    continue
                clean = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
                lo, hi = bootstrap_ci(clean)
                cell = f"{st.mean(clean):.3g}" + (f" [{lo:.3g}, {hi:.3g}]" if not math.isnan(lo) else "")
                if c != args.reference and ref:
                    seeds = sorted(set(cr) & set(ref))
                    p = wilcoxon_p([cr[s].get(mt, math.nan) for s in seeds], [ref[s].get(mt, math.nan) for s in seeds])
                    if not math.isnan(p):
                        cell += f" (p={p:.2g})"
                cells.append(cell)
            if any(x != "–" for x in cells):
                lines.append(f"| `{mt}` | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append(f"Mean [95% bootstrap interval] over seeds; p = paired Wilcoxon test against `{args.reference}`.")
    with open(os.path.join(args.directory, "report.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
