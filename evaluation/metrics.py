"""
Metrics of an evaluation series, computed from the files each run left (see run_series.py). Ground truth comes from
Flowcept's provenance and from the fault log, never from the memory or the analysis agent being evaluated.

    python -m evaluation.metrics results/eval/pilot       -> runs.csv, summary.csv, summary.md in that folder

Metrics (the specification's list):
  task completion             campaign ended normally; iterations completed (LangGraph `should_continue` nodes);
                              best value found (true value: injected offsets removed)
  redundant computations      xTB computations of a molecule beyond the first (provenance)
  inefficient exploration     proposals of a molecule whose computation had already ended (provenance)
  repeated technical failures failed computations of a molecule that had already failed (provenance)
  diagnostic accuracy         precision / recall of the analysis agent's findings against ground truth sets
  provenance traceability     share of memory references to Flowcept tasks that resolve to a recorded task
  procedure applicability     procedures given, applied, effective
  intervention effectiveness  guidance delivered, applied, effective
  knowledge maintenance       statements resting on corrupted results that were flagged
  memory retrieval            what was put in the prompts (counts only: relevance needs a definition, see docs)
  system overhead             campaign duration, LLM calls of the agents and of the analysis agent, analysis time,
                              memory size
"""
from __future__ import annotations

import csv
import glob
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

from pgmem import chem
from pgmem.provenance import as_dict, as_float, smiles_of

FAILURE_RULES = ("repeated_failure", "recurring_failure_class", "failure_explanation")


def load_jsonl(path: str) -> list[dict]:
    out = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def merged_tasks(records: list[dict]) -> dict[str, dict]:
    """Flowcept publishes a task more than once (start, end): merge the records by task id."""
    tasks: dict[str, dict] = {}
    for r in records:
        if r.get("task_id") and r.get("type") in (None, "task"):
            tasks[r["task_id"]] = {**tasks.get(r["task_id"], {}), **r}
    return tasks


def prf(truth: set, detected: set) -> dict:
    tp = len(truth & detected)
    return {"truth": len(truth), "detected": len(detected), "true_positives": tp,
            "precision": tp / len(detected) if detected else None,
            "recall": tp / len(truth) if truth else None}


def run_metrics(run_dir: str) -> dict:
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    tasks = merged_tasks(load_jsonl(os.path.join(run_dir, "provenance.jsonl")))
    faults = load_jsonl(os.path.join(run_dir, "faults.jsonl"))
    mem = json.load(open(os.path.join(run_dir, "memory.json"))) if os.path.exists(
        os.path.join(run_dir, "memory.json")) else {"entries": {}, "guidance": []}
    events = json.load(open(os.path.join(run_dir, "events.json"))) if os.path.exists(
        os.path.join(run_dir, "events.json")) else []
    entries = {e["id"]: e for c in ("M1", "M2", "M3", "M4") for e in mem["entries"].get(c, [])}
    m = {k: meta.get(k) for k in ("run_id", "scenario", "condition", "repetition", "status", "llm", "model")}

    # ---------------------------------------------------------------- provenance: what actually happened
    comps = sorted([t for t in tasks.values() if t.get("activity_id") == "compute_ionization_energy"
                    and t.get("subtype") == "parsl_task"], key=lambda t: as_float(t.get("started_at")) or 0)
    for t in comps:
        t["_key"] = chem.canonical(smiles_of(t) or "")
        t["_ok"] = t.get("status") != "ERROR"
    per_key = Counter(t["_key"] for t in comps)
    offsets = {f["key"]: f["delta"] for f in faults if f["fault"] == "offset"}
    values = [(t["_key"], as_float(as_dict(t.get("generated")).get("ionization_energy"))) for t in comps if t["_ok"]]
    values = [(k, v) for k, v in values if v is not None]
    seen_fail: set = set()
    repeated_failures = 0
    for t in comps:
        if not t["_ok"]:
            repeated_failures += t["_key"] in seen_fail
            seen_fail.add(t["_key"])
    ended = {}                                                # molecule -> end time of its first computation
    for t in comps:
        e = as_float(t.get("ended_at"))
        if e is not None:
            ended[t["_key"]] = min(e, ended.get(t["_key"], e))
    proposals = [t for t in tasks.values() if t.get("subtype") == "tool_call"
                 and t.get("activity_id") == "compute_ionization_energy"]
    late = [p for p in proposals if (k := chem.canonical(smiles_of(p) or "")) in ended
            and ended[k] < (as_float(p.get("started_at")) or 0)]
    n_agents = len(meta["seeds"].split(","))
    # one should_continue node ends each iteration (the last conclude nodes are not always in the provenance)
    iterations = sum(1 for t in tasks.values() if t.get("subtype") == "langgraph_node"
                     and t.get("activity_id") == "should_continue")
    m.update({
        "duration_s": round((meta.get("campaign_ended_at") or meta["ended_at"]) - meta["started_at"], 1),
        "iterations_completed": iterations,
        "iterations_expected": n_agents * meta["max_iterations"],
        "computations": len(comps), "distinct_molecules": len(per_key),
        "redundant_computations": sum(n - 1 for n in per_key.values() if n > 1),
        "failed_computations": sum(1 for t in comps if not t["_ok"]),
        "repeated_failures": repeated_failures,
        "proposals": len(proposals), "proposals_already_known": len(late),
        "best_value_reported": max((v for _, v in values), default=None),
        "best_value_true": max((v - offsets.get(k, 0.0) for k, v in values), default=None),
        "agent_llm_calls": sum(1 for t in tasks.values() if t.get("subtype") == "llm_call"),
        "faults_injected": len(faults),
    })

    # ---------------------------------------------------------------- memory (conditions with a memory)
    task_ids = set(tasks)
    refs = [(e["category"], r["task_id"]) for e in entries.values() for r in e.get("prov_refs", []) if r.get("task_id")]
    m["prov_refs"] = len(refs)
    m["prov_refs_resolved"] = sum(1 for _, t in refs if t in task_ids)
    m["traceability"] = m["prov_refs_resolved"] / len(refs) if refs else None
    m["memory_entries"] = len(entries)
    m["memory_bytes"] = os.path.getsize(os.path.join(run_dir, "memory.json")) if os.path.exists(
        os.path.join(run_dir, "memory.json")) else 0
    stats = mem.get("analyzer_stats") or {}
    m["analyzer_llm_calls"] = int(float(stats.get("llm_calls", 0)))
    m["analyzer_event_seconds"] = round(float(stats.get("event_seconds", 0)), 2)

    gs = mem.get("guidance", [])
    m["guidance_created"] = len(gs)
    m["guidance_deliveries"] = sum(len({d["agent"] for d in g["delivered"]}) for g in gs)
    m["guidance_applied"] = sum(1 for g in gs for v in g["applied"].values() if v is True)
    m["guidance_not_applied"] = sum(1 for g in gs for v in g["applied"].values() if v is False)
    m["guidance_effective"] = sum(1 for g in gs for v in g["effective"].values() if v is True)
    m["guidance_not_effective"] = sum(1 for g in gs for v in g["effective"].values() if v is False)
    procs = mem["entries"].get("M4", [])
    apps = [a for p in procs for a in p["data"].get("applications", [])]
    m.update({"procedures": len(procs), "procedure_given": len(apps),
              "procedure_applied": sum(1 for a in apps if a.get("applied") is True),
              "procedure_effective": sum(1 for a in apps if a.get("effective") is True),
              "procedure_not_effective": sum(1 for a in apps if a.get("effective") is False)})

    ctx = [e["payload"] for e in events if e["event"] == "context"]
    m["retrievals"] = len(ctx)
    for c in ("M1", "M2", "M3", "M4"):
        m[f"retrieved_{c}"] = sum(len(p["entries"].get(c, [])) for p in ctx)

    # knowledge maintenance: statements citing a corrupted result (truncated or offset), and whether they were flagged
    corrupted = {f["key"] for f in faults if f["fault"] in ("truncated", "offset")}
    m1_corrupted = {e["id"] for e in entries.values() if e["category"] == "M1" and e["key"] in corrupted}
    diags = [e for e in entries.values() if e["category"] == "M2" and e["kind"] == "diagnosis"]
    flagged_m3 = {r for d in diags if d["data"]["context"].get("rule") in
                  ("invalid_artifact", "knowledge_on_unreliable_result", "contradiction") for r in d.get("refs", [])}
    on_corrupted = [e for e in entries.values() if e["category"] == "M3" and set(e.get("refs", [])) & m1_corrupted]
    m["statements"] = sum(1 for e in entries.values() if e["category"] == "M3")
    m["statements_on_corrupted_results"] = len(on_corrupted)
    m["statements_on_corrupted_flagged"] = sum(
        1 for e in on_corrupted if e["id"] in flagged_m3
        or any(ev.get("added_by") == "analysis-agent" for ev in e["data"].get("contradicting_evidence", [])))

    # ---------------------------------------------------------------- diagnostic accuracy (analysis agent runs)
    if any(e["author"] == "analysis-agent" for e in entries.values()) or meta["condition"].startswith("full"):
        by_rule = defaultdict(set)
        for d in diags:
            by_rule[d["data"]["context"].get("rule")].add(d["key"])
        acc = {}
        acc["redundant_computation"] = prf({k for k, n in per_key.items() if n > 1}, by_rule["redundant_computation"])
        fails = Counter(t["_key"] for t in comps if not t["_ok"])
        acc["repeated_failure"] = prf({k for k, n in fails.items() if n > 1}, by_rule["repeated_failure"])
        classes = defaultdict(set)
        for t in comps:
            if not t["_ok"]:
                classes[chem.failure_class(str(t.get("stderr")))].add(t["_key"])
        acc["recurring_failure_class"] = prf({c for c, ks in classes.items() if len(ks) > 1},
                                             by_rule["recurring_failure_class"])
        not_converged = {t["_key"] for t in comps if t["_ok"]
                         and as_dict(as_dict(t.get("generated")).get("conditions")).get("converged") is False}
        m1_key = {e["id"]: e["key"] for e in entries.values() if e["category"] == "M1"}
        acc["invalid_artifact"] = prf(not_converged, {m1_key.get(k, k) for k in by_rule["invalid_artifact"]})
        contra = {m1_key.get(k.split("|")[-1]) for k in by_rule["contradiction"]} - {None}
        computed_offsets = {k for k in offsets if any(k == kk for kk, _ in values)}
        acc["contradiction"] = prf(computed_offsets, contra)
        truth_prop = set()
        for ev in events:
            if ev["event"] == "proposals":
                p, t = ev["payload"], ev["t"]
                if any(k in ended and ended[k] < t for k in p.get("keys", [])):
                    truth_prop.add(f"{p['agent']}|{p.get('iteration')}")
        acc["repeated_proposals"] = prf(truth_prop, by_rule["repeated_proposals"])
        confirmed = [d for d in diags if d["data"].get("basis") == "confirmed_cause"]
        m["confirmed_cause_findings"] = len(confirmed)
        m["confirmed_cause_correct"] = (sum(1 for d in confirmed if not chem.parses(d["key"])) / len(confirmed)
                                        if confirmed else None)
        m["procedure_ineffective_switch"] = bool(meta.get("fault_switch_at"))
        m["procedure_ineffective_detected"] = bool(by_rule["procedure_no_longer_effective"])
        for rule, v in acc.items():
            for k in ("truth", "detected", "precision", "recall"):
                m[f"diag_{rule}_{k}"] = v[k]
    return m


def mean_sd(xs: list) -> str:
    xs = [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
    if not xs:
        return ""
    if len(xs) == 1:
        return f"{xs[0]:.3g}"
    return f"{statistics.mean(xs):.3g} ± {statistics.stdev(xs):.2g}"


SUMMARY_COLUMNS = [
    ("runs ok", None), ("iterations", "iterations_completed"), ("computations", "computations"),
    ("redundant", "redundant_computations"), ("failed", "failed_computations"),
    ("repeated failures", "repeated_failures"), ("proposals already known", "proposals_already_known"),
    ("best value (true)", "best_value_true"), ("guidance applied", "guidance_applied"),
    ("guidance effective", "guidance_effective"), ("traceability", "traceability"),
    ("duration (s)", "duration_s"), ("agent LLM calls", "agent_llm_calls"),
    ("analysis LLM calls", "analyzer_llm_calls"),
]


def main(argv: list[str] | None = None) -> None:
    folder = (argv or sys.argv[1:])[0]
    rows = [run_metrics(os.path.dirname(p)) for p in sorted(glob.glob(os.path.join(folder, "*", "meta.json")))]
    if not rows:
        sys.exit(f"no run found in {folder}")
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k))
    with open(os.path.join(folder, "runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["scenario"], r["condition"])].append(r)
    numeric = [k for k in keys if any(isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)
                                      for r in rows) and k != "repetition"]
    with open(os.path.join(folder, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "condition", "runs", "runs_ok"] + numeric)
        for (s, c), rs in sorted(groups.items()):
            w.writerow([s, c, len(rs), sum(r["status"] == "ok" for r in rs)] + [mean_sd([r.get(k) for r in rs])
                                                                                for k in numeric])
    lines = [f"# Evaluation summary ({os.path.basename(os.path.abspath(folder))})", "",
             "Mean ± standard deviation over repetitions; empty cell: not applicable. Measured outcomes only.", ""]
    for s in sorted({k[0] for k in groups}):
        lines += [f"## {s}", "", "| condition | " + " | ".join(n for n, _ in SUMMARY_COLUMNS) + " |",
                  "|---" * (len(SUMMARY_COLUMNS) + 1) + "|"]
        for (s2, c), rs in sorted(groups.items()):
            if s2 != s:
                continue
            cells = [f"{sum(r['status'] == 'ok' for r in rs)}/{len(rs)}"] + [mean_sd([r.get(k) for r in rs])
                                                                           for _, k in SUMMARY_COLUMNS[1:]]
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
        diag = sorted({k.rsplit("_", 1)[0] for r in groups_rows(groups, s) for k in r if k.startswith("diag_")})
        if diag:
            lines += ["", "Diagnostic accuracy (precision / recall), conditions with the analysis agent:", "",
                      "| condition | " + " | ".join(d[5:] for d in diag) + " |", "|---" * (len(diag) + 1) + "|"]
            for (s2, c), rs in sorted(groups.items()):
                if s2 == s and any(f"{diag[0]}_precision" in r for r in rs):
                    lines.append(f"| {c} | " + " | ".join(
                        f"{mean_sd([r.get(d + '_precision') for r in rs]) or '-'} / "
                        f"{mean_sd([r.get(d + '_recall') for r in rs]) or '-'}" for d in diag) + " |")
        lines.append("")
    with open(os.path.join(folder, "summary.md"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"{len(rows)} runs -> {folder}/runs.csv, summary.csv, summary.md")


def groups_rows(groups: dict, scenario: str) -> list[dict]:
    return [r for (s, _), rs in groups.items() if s == scenario for r in rs]


if __name__ == "__main__":
    main()
