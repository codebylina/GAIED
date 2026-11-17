"""
Metrics of one GAIED campaign, computed from Flowcept provenance and from the memory (nothing is entered
by hand).

  reuse / redundancy  xTB runs, distinct molecules, redundant computations (runs - distinct), results
                      reused, waits, known negative results not recomputed
  exploration         best value, mean of the top 5, distinct valid molecules, mean nearest-neighbour Tanimoto
                      similarity (lower = more diverse), negative (invalid) proposals, technical failures
  detection accuracy  redundant-computation findings vs the ground truth in the provenance (molecules
                      computed more than once): precision / recall. Other detectors have no ground truth here.
  interventions       guidance published, delivered, acknowledged, applied / violated, validated / ineffective
  retrieval           mean time to build a prompt context (ms), number of contexts
  overhead            LLM calls (all, from Flowcept), campaign wall time, memory events
"""
from __future__ import annotations

import glob
import json
import statistics

from memory.domains.chemistry import ChemistryDomain
from memory.provenance import FlowceptSource, summary


def raw_provenance(run_dir: str, campaign: str, online: bool) -> list[dict]:
    """Flowcept task records of one campaign (database when online, JSONL buffers otherwise), deduplicated."""
    src = FlowceptSource(ChemistryDomain())
    if online and src.db_available():
        tasks = src.tasks_since(campaign, None, limit=100_000)
    else:
        tasks = src.read_buffers(f"{run_dir}/output/flowcept_buffer_*.jsonl", campaign_id=campaign)
    seen, out = set(), []
    for t in tasks:
        if t.get("task_id") and t["task_id"] not in seen:
            seen.add(t["task_id"])
            out.append(t)
    return out


def provenance(run_dir: str, campaign: str, online: bool) -> list[dict]:
    dom = ChemistryDomain()
    return [summary(t, dom) for t in raw_provenance(run_dir, campaign, online)]


def offline_detection(run_dir: str, campaign: str, online: bool, redis_client, namespace: str) -> dict:
    """Detection accuracy of the analysis service on a campaign WITHOUT shared memory: its Flowcept provenance
    is analysed after the fact; the ground truth (molecules computed more than once) is in the same provenance."""
    from analysis import AnalysisConfig, AnalysisService
    from memory import SharedMemory, Store
    raw = raw_provenance(run_dir, campaign, online)
    dom = ChemistryDomain()
    exps = [dom.experiment_of(t) for t in raw]
    exps = [e for e in exps if e]
    if not exps:
        return {}
    params = next((t["used"].get("xtb_config") for t in raw if dom.is_experiment_task(t)), None) or {}
    mem = SharedMemory(Store(redis_client, namespace), ChemistryDomain(default_params=params), "analysis")
    mem.register_agent("xtb", params, {}, campaign)
    svc = AnalysisService(mem, AnalysisConfig(method="xtb", params=params), source=None, period_s=0, live=False)
    while svc.read_events(block_ms=10):
        pass
    svc._apply_tasks(raw)
    svc.analyze()
    ents = [e["entity"] for e in exps]
    truth = {e for e in set(ents) if ents.count(e) > 1}
    found = {e for ep in mem.store.find("M2", kind="redundant_computation") for e in ep.entities}
    return {"truth": len(truth), "detected": len(found),
            "precision": round(len(found & truth) / len(found), 3) if found else None,
            "recall": round(len(found & truth) / len(truth), 3) if truth else None,
            "confirmed_causes": sorted({ep.confirmed_cause for ep in mem.store.find("M2", kind="redundant_computation")
                                        if ep.confirmed_cause})}


def collect(run_dir: str, mode: str, namespace: str, campaign: str, online: bool, redis_client) -> dict:
    dom = ChemistryDomain()
    tasks = provenance(run_dir, campaign, online)
    exps = [t for t in tasks if t.get("experiment")]
    ok = [t for t in exps if t["ok"]]
    entities = [t["entity"] for t in exps]
    distinct = sorted(set(entities))
    values = sorted({t["entity"]: float(t["value"]) for t in ok}.values(), reverse=True)
    valid = sorted({t["entity"] for t in ok})
    sim = None
    if len(valid) >= 3:
        sim = statistics.mean(max(dom.similarity(a, b) for b in valid if b != a) for a in valid)
    times = [x for t in tasks for x in (t.get("started_at"), t.get("ended_at")) if x]
    m = {
        "mode": mode, "namespace": namespace, "campaign": campaign,
        "xtb_runs": len(exps), "distinct_molecules": len(distinct), "redundant_computations": len(exps) - len(distinct),
        "negative_results": sum(1 for t in exps if not t["ok"] and dom.classify_failure(t.get("error") or "") == "scientific"),
        "technical_failures": sum(1 for t in exps if not t["ok"] and dom.classify_failure(t.get("error") or "") != "scientific"),
        "best_value": values[0] if values else None,
        "top5_mean": statistics.mean(values[:5]) if values else None,
        "distinct_valid_molecules": len(valid),
        "mean_nn_similarity": round(sim, 4) if sim is not None else None,
        "llm_calls": sum(1 for t in tasks if t.get("subtype") == "llm_call"),
        "wall_time_s": round(max(times) - min(times), 1) if times else None,
        "provenance_tasks": len(tasks),
    }
    r = redis_client
    if mode == "basic":
        st = r.hgetall(f"naive:{namespace}:stats")
        m.update(reused=int(st.get("reused", 0)), waits=int(st.get("waited", 0)),
                 known_failures_not_recomputed=int(st.get("reused_error", 0)))
    elif mode.startswith("structured"):
        from memory import SharedMemory, Store
        mem = SharedMemory(Store(r, namespace), dom, "metrics")
        met = mem.store.metrics()
        m.update(reused=met.get("decisions.reuse", 0), waits=met.get("decisions.waited", 0),
                 known_failures_not_recomputed=met.get("decisions.blocked_negative", 0),
                 contexts=met.get("retrieval.context.n", 0),
                 context_ms_mean=round(met["retrieval.context.ms"] / met["retrieval.context.n"], 2)
                 if met.get("retrieval.context.n") else None,
                 knowledge_extractions=met.get("extractions.done", 0),
                 memory_events=int(r.xlen(mem.store.k("events"))))
        q = mem.report()
        m["M3_status"] = q["M3"]["by_status"]
        m["M2_by_kind"] = q["M2"]["by_kind"]
        guidance = mem.store.find("M4") + mem.store.get_many("M4", mem.store.ids("M4", archived=True))
        g_an = [g for g in guidance if g.source == "analysis"]
        deliveries = [d for g in g_an for d in mem.guidance.delivery(g.id).values()]
        m.update(
            guidance_published=len(g_an),
            guidance_by_status=dict(__import__("collections").Counter(g.status for g in g_an)),
            guidance_delivered=sum(d.get("delivered_n", 0) for d in deliveries),
            guidance_acknowledged=sum(d.get("acknowledged_n", 0) for d in deliveries),
            guidance_applied=sum(d.get("applied_n", 0) for d in deliveries),
            guidance_violated=sum(d.get("violated_n", 0) for d in deliveries))
        if mode == "structured+analysis":
            truth = {e for e in distinct if entities.count(e) > 1}
            found = {e for ep in mem.store.find("M2", kind="redundant_computation") for e in ep.entities}
            m["detection_redundant"] = {
                "truth": len(truth), "detected": len(found),
                "precision": round(len(found & truth) / len(found), 3) if found else None,
                "recall": round(len(found & truth) / len(truth), 3) if truth else None}
            m["analysis_findings"] = len([e for e in mem.store.find("M2") if e.created_by == "analysis"])
    return m


def aggregate(rows: list[dict]) -> str:
    """Markdown table: mean (and standard deviation over repetitions) per configuration."""
    keys = ["xtb_runs", "distinct_molecules", "redundant_computations", "reused", "waits", "best_value",
            "top5_mean", "mean_nn_similarity", "llm_calls", "run_wall_time_s", "context_ms_mean",
            "guidance_published", "guidance_delivered", "guidance_applied", "guidance_violated"]
    modes = list(dict.fromkeys(r["mode"] for r in rows))
    lines = ["| metric | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for k in keys:
        cells = []
        for mo in modes:
            vals = [r[k] for r in rows if r["mode"] == mo and isinstance(r.get(k), (int, float))]
            if not vals:
                cells.append("–")
            elif len(vals) == 1:
                cells.append(f"{vals[0]:.4g}")
            else:
                cells.append(f"{statistics.mean(vals):.4g} ± {statistics.stdev(vals):.2g}")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    n = {mo: sum(r["mode"] == mo for r in rows) for mo in modes}
    lines.append("")
    lines.append("Repetitions per configuration: " + ", ".join(f"{k}={v}" for k, v in n.items()))
    return "\n".join(lines)


def load(out_dir: str) -> list[dict]:
    return [json.load(open(p)) for p in sorted(glob.glob(f"{out_dir}/*/metrics.json"))]
