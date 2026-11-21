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
  learning and adaptation     failing-motif share of the proposals, per phase (motif_failures)
  diagnostic accuracy         precision / recall of the analysis agent's findings against ground truth sets
  provenance traceability     share of memory references to Flowcept tasks that resolve to a recorded task
  provenance coverage         provenance records received online by the analysis agent / records recorded
  memory retrieval            relevance of what was retrieved (structural criterion), per channel and access mode
  procedures, interventions   procedures given, applied, effective; guidance delivered, applied, effective (edges)
  knowledge maintenance       statements resting on corrupted results and whether they were flagged, directly or
                              through the graph; verdicts against the true values
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
from pgmem.provenance import as_dict, as_float, is_relevant, smiles_of

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


class Export:
    """An exported memory (memory.json) with its graph."""

    def __init__(self, mem: dict) -> None:
        self.entries = {e["id"]: e for c in ("M1", "M2", "M3", "M4") for e in mem["entries"].get(c, [])}
        self.edges = mem.get("edges", [])
        self.out, self.inc = defaultdict(list), defaultdict(list)
        for e in self.edges:
            self.out[e["source"]].append(e)
            self.inc[e["target"]].append(e)
        self.guidance = mem.get("guidance", [])
        self.verification = mem.get("verification") or {}
        self.stats = mem.get("analyzer_stats") or {}

    def of(self, category: str, kind: str | None = None) -> list[dict]:
        return [e for e in self.entries.values() if e["category"] == category and (kind is None or e["kind"] == kind)]

    def targets(self, node: str, *types: str) -> list[str]:
        return [e["target"] for e in self.out[node] if e["type"] in types]

    def sources(self, node: str, *types: str) -> list[str]:
        return [e["source"] for e in self.inc[node] if e["type"] in types]

    def diagnoses(self) -> list[dict]:
        return self.of("M2", "diagnosis")

    def unreliable(self, m1: dict) -> bool:
        if m1["data"].get("conditions", {}).get("converged") is False:
            return True
        return any(self.entries.get(d, {}).get("data", {}).get("context", {}).get("subject") == m1["id"]
                   for d in self.sources(m1["id"], "DIAGNOSED_FROM"))

    def keys_of(self, node: str) -> tuple[set, set]:
        """Molecules and motifs an entry is ABOUT."""
        about = self.targets(node, "ABOUT")
        return {a[4:] for a in about if a.startswith("mol:")}, {a[6:] for a in about if a.startswith("motif:")}

    def judged(self) -> list[dict]:
        return [e for e in self.edges if e["type"] == "APPLIED_IN"]


def run_metrics(run_dir: str) -> dict:
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    records = load_jsonl(os.path.join(run_dir, "provenance.jsonl"))
    tasks = merged_tasks(records)
    fault_log = load_jsonl(os.path.join(run_dir, "faults.jsonl"))
    computation_log = [f for f in fault_log if f["fault"] == "computation"]
    faults = [f for f in fault_log if f["fault"] not in ("computation", "switch")]
    mem_path = os.path.join(run_dir, "memory.json")
    x = Export(json.load(open(mem_path)) if os.path.exists(mem_path) else {"entries": {}})
    events = json.load(open(os.path.join(run_dir, "events.json"))) if os.path.exists(
        os.path.join(run_dir, "events.json")) else []
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
    m.update({
        "duration_s": round((meta.get("campaign_ended_at") or meta["ended_at"]) - meta["started_at"], 1),
        # one should_continue node ends each iteration
        "iterations_completed": sum(1 for t in tasks.values() if t.get("subtype") == "langgraph_node"
                                    and t.get("activity_id") == "should_continue"),
        "iterations_expected": n_agents * meta["max_iterations"],
        "computations": len(comps), "distinct_molecules": len(per_key),
        "redundant_computations": sum(n - 1 for n in per_key.values() if n > 1),
        "failed_computations": sum(1 for t in comps if not t["_ok"]),
        "repeated_failures": repeated_failures,
        "proposals": len(proposals), "proposals_already_known": len(late),
        "valid_results": len(values), "valid_distinct_molecules": len({k for k, _ in values}),
        "best_value_reported": max((v for _, v in values), default=None),
        "best_value_true": max((v - offsets.get(k, 0.0) for k, v in values), default=None),
        "agent_llm_calls": sum(1 for t in tasks.values() if t.get("subtype") == "llm_call"),
        "faults_injected": len(faults), "computations_logged": len(computation_log),
    })

    # ---------------------------------------------------------------- memory (conditions with a memory)
    refs = [r["task_id"] for e in x.entries.values() for r in e.get("prov_refs", []) if r.get("task_id")]
    m["prov_refs"] = len(refs)
    m["prov_refs_resolved"] = sum(1 for t in refs if t in tasks)
    m["traceability"] = m["prov_refs_resolved"] / len(refs) if refs else None
    m["memory_entries"] = len(x.entries)
    m["memory_edges"] = len(x.edges)
    m["memory_edges_llm"] = sum(1 for e in x.edges if e["origin"] == "llm")
    m["memory_bytes"] = os.path.getsize(mem_path) if os.path.exists(mem_path) else 0
    m["analyzer_llm_calls"] = int(float(x.stats.get("llm_calls", 0)))
    m["analyzer_event_seconds"] = round(float(x.stats.get("event_seconds", 0)), 2)
    if "provenance_records" in x.stats:   # absent: not measured (no provenance source)
        expected = {r["task_id"] for r in records if r.get("task_id") and is_relevant(r)}
        m["provenance_expected"] = len(expected)
        m["provenance_received"] = int(float(x.stats["provenance_records"]))
        m["provenance_coverage"] = m["provenance_received"] / len(expected) if expected else None

    judged = x.judged()
    m["guidance_created"] = len(x.guidance)
    m["guidance_deliveries"] = sum(len({d["agent"] for d in g["delivered"]}) for g in x.guidance)
    for k in ("applied", "effective"):
        m[f"guidance_{k}"] = sum(1 for e in judged if e["props"].get(k) is True)
        m[f"guidance_not_{k}"] = sum(1 for e in judged if e["props"].get(k) is False)
    procs = x.of("M4")
    apps = [a for p in procs for g in x.sources(p["id"], "APPLIES") for a in x.out[g] if a["type"] == "APPLIED_IN"]
    m.update({"procedures": len(procs), "procedure_motifs": len({p["data"].get("motif") for p in procs}),
              "procedure_judged": len(apps),
              "procedure_applied": sum(1 for a in apps if a["props"].get("applied") is True),
              "procedure_effective": sum(1 for a in apps if a["props"].get("effective") is True),
              "procedure_not_effective": sum(1 for a in apps if a["props"].get("effective") is False),
              "procedures_justified": sum(1 for p in procs if x.targets(p["id"], "JUSTIFIED_BY"))})

    # knowledge maintenance: statements resting on a corrupted result, and whether the analysis agent flagged them
    corrupted = {f["key"] for f in faults if f["fault"] in ("truncated", "offset")}
    m1_corrupted = {e["id"] for e in x.of("M1") if e["key"] in corrupted}
    diags = x.diagnoses()
    flagged, indirect = set(), set()
    for d in diags:
        for a in d["data"].get("context", {}).get("affected", []):
            flagged.add(a["node"])
            if a["depth"] > 1:
                indirect.add(a["node"])
        if d["data"]["rule"] in ("knowledge_on_unreliable_result", "claim_contradicted",
                                 "claim_contradicted_by_new_result"):
            flagged.update(x.targets(d["id"], "DIAGNOSED_FROM"))
    on_corrupted = [e for e in x.of("M3") if set(x.targets(e["id"], "SUPPORTED_BY")) & m1_corrupted]
    m["statements"] = len(x.of("M3"))
    m["statements_on_corrupted_results"] = len(on_corrupted)
    m["statements_on_corrupted_flagged"] = sum(1 for e in on_corrupted if e["id"] in flagged)
    m["indirectly_flagged"] = len(indirect)

    # ---------------------------------------------------------------- diagnostic accuracy (analysis agent runs)
    if meta["condition"].startswith("full"):
        by_rule = defaultdict(set)
        for d in diags:
            by_rule[d["data"]["rule"]].add(d["key"])
        m1_key = {e["id"]: e["key"] for e in x.of("M1")}
        acc = {"redundant_computation": prf({k for k, n in per_key.items() if n > 1}, by_rule["redundant_computation"])}
        fails = Counter(t["_key"] for t in comps if not t["_ok"])
        acc["repeated_failure"] = prf({k for k, n in fails.items() if n > 1}, by_rule["repeated_failure"])
        names = {smarts: name for name, smarts in chem.FEATURES.items()}
        failing = {t["_key"] for t in comps if not t["_ok"]}
        truth = {f"scf_not_converged|{names[s]}" for s in meta.get("motifs") or []
                 if s in names and sum(1 for k in failing if names[s] in chem.features(k)) >= 2}
        if truth:
            acc["recurring_failure"] = prf(truth, {k for k in by_rule["recurring_failure"] if not k.endswith("|*")})
        not_converged = {t["_key"] for t in comps if t["_ok"]
                         and as_dict(as_dict(t.get("generated")).get("conditions")).get("converged") is False}
        acc["invalid_artifact"] = prf(not_converged, {m1_key.get(k, k) for k in by_rule["invalid_artifact"]})
        revealed = {m1_key[t] for d in diags if d["data"]["rule"].startswith("claim_contradicted")
                    for t in x.targets(d["id"], "DIAGNOSED_FROM") if t in m1_key}
        acc["corrupted_result"] = prf({k for k in offsets if any(k == kk for kk, _ in values)}, revealed)
        truth_prop = set()
        for ev in events:
            if ev["event"] == "proposals":
                p, t = ev["payload"], ev["t"]
                if any(k in ended and ended[k] < t for k in p.get("keys", [])):
                    truth_prop.add(f"{p['agent']}|{p.get('iteration')}")
        acc["repeated_proposals"] = prf(truth_prop, by_rule["repeated_proposals"])
        confirmed = [d for d in diags if x.verification.get(d["id"], {}).get("claim", {}).get("form") == "rdkit_parse"]
        m["confirmed_cause_findings"] = len(confirmed)
        m["confirmed_cause_correct"] = (sum(1 for d in confirmed if not chem.parses(d["key"])) / len(confirmed)
                                        if confirmed else None)
        m["procedure_ineffective_detected"] = bool(by_rule["procedure_no_longer_effective"])
        for rule, v in acc.items():
            for k in ("truth", "detected", "precision", "recall"):
                m[f"diag_{rule}_{k}"] = v[k]
    m.update(adaptation_metrics(meta, comps, proposals, computation_log, diags, x.guidance))
    m.update(retrieval_metrics(events, x))
    m.update(verification_metrics(x, offsets))
    m["_guidance_by_rule"] = guidance_by_rule(x)
    return m


# ---------------------------------------------------------------------------------------------- helpers
def _has(smiles: str, smarts: str) -> bool:
    from rdkit import Chem
    mol, patt = Chem.MolFromSmiles(smiles or ""), Chem.MolFromSmarts(smarts)
    return bool(mol is not None and patt is not None and mol.HasSubstructMatch(patt))


def _share(items: list, smarts: str) -> float | None:
    return sum(_has(k, smarts) for k in items) / len(items) if items else None


FAILURE_RULES = ("repeated_failure", "recurring_failure", "failure_explanation")


def adaptation_metrics(meta: dict, comps: list, proposals: list, computation_log: list, diags: list,
                       gs: list) -> dict:
    """Scenario with a failing motif that changes: do proposals move away from the failing motif within a phase,
    how fast does the analysis agent notice the change, and is old guidance still given?"""
    motifs = meta.get("motifs")
    if not motifs:
        return {}
    switch_t = meta.get("fault_switch_at")
    props = sorted(((as_float(p.get("started_at")) or 0, chem.canonical(smiles_of(p) or "")) for p in proposals))
    out = {"phase_switch_reached": switch_t is not None}
    for i, motif in enumerate(motifs):
        lo, hi = (props[0][0] if props else 0), float("inf")
        if i == 0 and switch_t is not None:
            hi = switch_t
        if i == 1:
            if switch_t is None:
                continue
            lo = switch_t
        ph = [k for t, k in props if lo <= t < hi]
        half = len(ph) // 2
        out[f"phase{i + 1}_proposals"] = len(ph)
        out[f"phase{i + 1}_failing_motif_share_first_half"] = _share(ph[:half], motif)
        out[f"phase{i + 1}_failing_motif_share_second_half"] = _share(ph[half:], motif)
        out[f"phase{i + 1}_failures"] = sum(1 for t in comps if not t["_ok"]
                                            and lo <= (as_float(t.get("started_at")) or 0) < hi)
    if switch_t is not None:
        after = [d for d in diags if d["created_at"] > switch_t
                 and d["data"]["rule"] in FAILURE_RULES + ("procedure_no_longer_effective",)]
        first = min((d["created_at"] for d in after), default=None)
        out["computations_to_detect_change"] = (sum(1 for f in computation_log if switch_t <= f["t"] < first)
                                                if first else None)
        out["old_failure_guidance_delivered_after_switch"] = sum(
            1 for g in gs if g["created_at"] < switch_t and g.get("problem_type") == "technical_failure"
            for d in g["delivered"] if d["t"] > switch_t)
        out["phase2_old_motif_share"] = _share([k for t, k in props if t >= switch_t], motifs[0])
    return out


def relevant(x: Export, entry_id: str, query_mols: list[str]) -> bool:
    """Relevance criterion of the evaluation: the entry is about one of the task's molecules, a structurally similar
    one (Tanimoto >= 0.4), or a specific motif of them (directly, or through the molecules it is about)."""
    mols, motifs = x.keys_of(entry_id)
    if not query_mols or not (mols or motifs):
        return False
    if mols & set(query_mols) or any(chem.similarity(a, b) >= 0.4 for a in query_mols for b in mols):
        return True
    qf = set().union(*(chem.features(q) for q in query_mols)) - chem.GENERIC
    ef = motifs | (set().union(*(chem.features(k) for k in mols)) - chem.GENERIC if mols else set())
    return bool(qf & ef)


CHANNELS = ("bm25", "semantic", "chem", "graph")


def retrieval_metrics(events: list, x: Export) -> dict:
    """What the memory gave the agents (scheduled sections and on-demand answers), how relevant it was to their
    task, which channels found it, whether the agents asked themselves, and whether they proposed again what they
    had just been shown."""
    out = {}
    for kind in ("context", "memory_query"):
        evs = [e["payload"] for e in events if e["event"] == kind]
        ret = [(p, i) for p in evs for c in ("M1", "M2", "M3", "M4") for i in p["entries"].get(c, [])
               if i in x.entries]
        name = "scheduled" if kind == "context" else "on_demand"
        out[f"retrieved_{name}"] = len(ret)
        out[f"retrieval_precision_{name}"] = (sum(relevant(x, i, p["molecules"]) for p, i in ret) / len(ret)
                                              if ret else None)
        for ch in CHANNELS:
            out[f"retrieved_{name}_by_{ch}"] = sum(1 for p in evs for r in p.get("ranks", {}).values() if ch in r)
    props = [e for e in events if e["event"] == "proposals"]
    queries = [e for e in events if e["event"] == "memory_query"]
    out["memory_queries"] = len(queries)
    out["decisions"] = len(props)
    out["decisions_after_a_query"] = sum(
        1 for p in props if any(q["payload"]["agent"] == p["payload"]["agent"]
                                and q["payload"].get("iteration") == p["payload"].get("iteration") and q["t"] <= p["t"]
                                for q in queries))
    shown_again = 0
    for p in props:
        prior = [e for e in events if e["event"] in ("context", "memory_query") and e["t"] <= p["t"]
                 and e["payload"]["agent"] == p["payload"]["agent"]]
        if prior:
            last = prior[-1]["payload"]["entries"]
            shown = {x.entries[i]["key"] for c in ("M1", "M2") for i in last.get(c, []) if i in x.entries
                     and (c == "M1" or x.entries[i]["kind"] == "failure")}
            shown_again += sum(1 for k in p["payload"].get("keys", []) if k in shown)
    out["proposals_of_molecules_just_shown"] = shown_again
    return out


def verification_metrics(x: Export, offsets: dict) -> dict:
    """Verdicts of the statements, and, with the true values (offsets removed), which verdicts the corrupted results
    changed."""
    from pgmem import verification
    recs = {k: r for k, r in x.verification.items() if k.startswith("m3:")}
    if not recs:
        return {}
    good, bad, true = {}, set(), {}
    for e in x.of("M1"):
        v = e["data"].get("value")
        if isinstance(v, (int, float)) and not x.unreliable(e):
            good.setdefault(e["key"], (v, e["id"]))
            true.setdefault(e["key"], (v - offsets.get(e["key"], 0.0), e["id"]))
        else:
            bad.add(e["key"])
    bad -= set(good)
    checkable = [r for r in recs.values() if r.get("checkable")]
    out = {"statements_verified": len(recs), "statements_checkable": len(checkable)}
    for v in ("supported", "contradicted", "inconclusive"):
        out[f"verdict_{v}"] = sum(1 for r in recs.values() if r.get("verdict") == v)
    changed = genuine = 0
    for r in checkable:
        now = verification.check(r["claim"], good, bad)["verdict"]
        with_true = verification.check(r["claim"], true, bad)["verdict"]
        changed += now != with_true
        genuine += with_true == "contradicted"
    out["verdicts_changed_by_corrupted_results"] = changed
    out["statements_contradicted_by_true_values"] = genuine
    return out


def guidance_by_rule(x: Export) -> dict:
    """Guidance outcomes by what it applies: the rule of its diagnosis, or "procedure"."""
    out = defaultdict(Counter)
    for g in x.guidance:
        applies = x.targets(g["id"], "APPLIES")
        rule = next((x.entries[i]["data"]["rule"] for i in applies if i.startswith("m2:") and i in x.entries),
                    "procedure" if any(i.startswith("m4:") for i in applies) else "other")
        c, judged = out[rule], [e for e in x.out[g["id"]] if e["type"] == "APPLIED_IN"]
        c["created"] += 1
        c["delivered"] += len({d["agent"] for d in g["delivered"]})
        for k in ("applied", "effective"):
            c[k] += sum(1 for e in judged if e["props"].get(k) is True)
            c[f"not_{k}"] += sum(1 for e in judged if e["props"].get(k) is False)
    return {k: dict(v) for k, v in out.items()}


T95 = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}


def mean_sd(xs: list, ci: bool = False) -> str:
    """mean ± standard deviation (and, with ci, the 95 % t interval of the mean and the range)."""
    xs = [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
    if not xs:
        return ""
    if len(xs) == 1:
        return f"{xs[0]:.3g}"
    m, sd = statistics.mean(xs), statistics.stdev(xs)
    text = f"{m:.3g} ± {sd:.2g}"
    if ci:
        h = T95.get(len(xs), 2.0) * sd / len(xs) ** 0.5
        text += f" [{m - h:.3g}, {m + h:.3g}] (range {min(xs):.3g}-{max(xs):.3g})"
    return text


SUMMARY_COLUMNS = [
    ("runs ok", None), ("iterations", "iterations_completed"), ("computations", "computations"),
    ("valid distinct molecules", "valid_distinct_molecules"), ("redundant", "redundant_computations"),
    ("failed", "failed_computations"), ("repeated failures", "repeated_failures"),
    ("proposals already known", "proposals_already_known"), ("best IE, true (secondary)", "best_value_true"),
    ("duration (s)", "duration_s"), ("agent LLM calls", "agent_llm_calls"),
    ("analysis LLM calls", "analyzer_llm_calls"),
]
MEMORY_COLUMNS = [
    ("traceability", "traceability"), ("provenance coverage", "provenance_coverage"), ("edges", "memory_edges"),
    ("memory queries", "memory_queries"), ("decisions after a query", "decisions_after_a_query"),
    ("re-proposed just shown", "proposals_of_molecules_just_shown"),
    ("guidance delivered", "guidance_deliveries"), ("applied", "guidance_applied"),
    ("not applied", "guidance_not_applied"), ("effective", "guidance_effective"),
    ("not effective", "guidance_not_effective"),
    ("procedures (motifs)", ("procedures", "procedure_motifs")), ("procedure effective / not", (
        "procedure_effective", "procedure_not_effective")),
]
RETRIEVAL_COLUMNS = [
    ("precision scheduled", "retrieval_precision_scheduled"), ("precision on demand", "retrieval_precision_on_demand"),
] + [(f"found by {c} (scheduled)", f"retrieved_scheduled_by_{c}") for c in CHANNELS]
KNOWLEDGE_COLUMNS = [
    ("statements on corrupted / flagged", ("statements_on_corrupted_results", "statements_on_corrupted_flagged")),
    ("flagged through the graph", "indirectly_flagged"),
    ("statements verified", "statements_verified"), ("checkable", "statements_checkable"),
    ("supported", "verdict_supported"), ("contradicted", "verdict_contradicted"),
    ("inconclusive", "verdict_inconclusive"), ("contradicted with true values", "statements_contradicted_by_true_values"),
    ("verdicts changed by corrupted results", "verdicts_changed_by_corrupted_results"),
]
ADAPTATION_COLUMNS = [
    ("switch reached", "phase_switch_reached"),
    ("phase 1 nitrile share, 1st / 2nd half", ("phase1_failing_motif_share_first_half",
                                                "phase1_failing_motif_share_second_half")),
    ("phase 1 failures", "phase1_failures"),
    ("phase 2 CF3 share, 1st / 2nd half", ("phase2_failing_motif_share_first_half",
                                            "phase2_failing_motif_share_second_half")),
    ("phase 2 failures", "phase2_failures"), ("phase 2 nitrile share", "phase2_old_motif_share"),
    ("computations to detect the change", "computations_to_detect_change"),
    ("old guidance delivered after the change", "old_failure_guidance_delivered_after_switch"),
]


def _cell(rs: list[dict], key, ci: bool = False) -> str:
    if key is None:
        return ""
    if isinstance(key, tuple):
        return " / ".join(mean_sd([r.get(k) for r in rs]) or "-" for k in key)
    vals = [r.get(key) for r in rs]
    if any(isinstance(v, bool) for v in vals):
        return f"{sum(1 for v in vals if v)}/{len(vals)}"
    return mean_sd(vals, ci)


def _order(item) -> tuple:
    (s, c), _ = item
    from .config import CONDITIONS
    return (s, list(CONDITIONS).index(c) if c in CONDITIONS else 99, c)


def _table(title: str, columns: list, groups: dict, scenario: str, only=None) -> list[str]:
    rows = [(c, rs) for (s, c), rs in sorted(groups.items(), key=_order)
            if s == scenario and (only is None or only(rs))]
    if not rows:
        return []
    out = ["", f"{title}:", "", "| condition | " + " | ".join(n for n, _ in columns) + " |",
           "|---" * (len(columns) + 1) + "|"]
    for c, rs in rows:
        cells = []
        for name, key in columns:
            if name == "runs ok":
                cells.append(f"{sum(r['status'] == 'ok' for r in rs)}/{len(rs)}")
            else:
                cells.append(_cell(rs, key, ci=name in ("computations", "redundant", "failed", "repeated failures",
                                                        "proposals already known")))
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    return out


def main(argv: list[str] | None = None) -> None:
    folder = (argv or sys.argv[1:])[0]
    rows = [run_metrics(os.path.dirname(p)) for p in sorted(glob.glob(os.path.join(folder, "*", "meta.json")))]
    if not rows:
        sys.exit(f"no run found in {folder}")
    keys = sorted({k for r in rows for k in r if not k.startswith("_")}, key=lambda k: (k not in rows[0], k))
    with open(os.path.join(folder, "runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
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
             "Mean ± standard deviation over repetitions; for counts also the 95 % interval of the mean and the "
             "range. Empty cell: not applicable. Measured outcomes only. The best ionization energy is a secondary "
             "measure (the objective is to maximize it, but agents explore different molecules).", ""]
    has = lambda key: (lambda rs: any(r.get(key) is not None for r in rs))  # noqa: E731
    for s in sorted({k[0] for k in groups}):
        lines += [f"## {s}"]
        lines += _table("Outcomes", SUMMARY_COLUMNS, groups, s)
        lines += _table("Memory, guidance and procedures (conditions with a memory)", MEMORY_COLUMNS, groups, s,
                        only=has("traceability"))
        lines += _table("Retrieval (conditions with the structured memory)", RETRIEVAL_COLUMNS, groups, s,
                        only=has("retrieval_precision_scheduled"))
        lines += _table("Knowledge maintenance and verdicts", KNOWLEDGE_COLUMNS, groups, s,
                        only=has("statements_verified"))
        lines += _table("Adaptation to the failing motif and its change", ADAPTATION_COLUMNS, groups, s,
                        only=has("phase_switch_reached"))
        diag = sorted({k.rsplit("_", 1)[0] for r in groups_rows(groups, s) for k in r if k.startswith("diag_")})
        if diag:
            lines += ["", "Diagnostic accuracy, precision / recall (deterministic rules check the same facts as "
                      "the ground truth: these values mostly verify the implementation):", "",
                      "| condition | " + " | ".join(d[5:] for d in diag) + " |", "|---" * (len(diag) + 1) + "|"]
            for (s2, c), rs in sorted(groups.items(), key=_order):
                if s2 == s and any(f"{diag[0]}_precision" in r for r in rs):
                    lines.append(f"| {c} | " + " | ".join(
                        f"{mean_sd([r.get(d + '_precision') for r in rs]) or '-'} / "
                        f"{mean_sd([r.get(d + '_recall') for r in rs]) or '-'}" for d in diag) + " |")
        by_rule = defaultdict(lambda: defaultdict(Counter))
        for (s2, c), rs in groups.items():
            if s2 == s:
                for r in rs:
                    for rule, cnt in (r.get("_guidance_by_rule") or {}).items():
                        by_rule[c][rule].update(cnt)
        if by_rule:
            lines += ["", "Guidance by the rule that produced it (all repetitions): delivered / applied / not "
                      "applied / effective / not effective", "", "| condition | rule | outcome |", "|---|---|---|"]
            for c in sorted(by_rule):
                for rule, cnt in sorted(by_rule[c].items()):
                    lines.append(f"| {c} | {rule} | {cnt['delivered']} / {cnt['applied']} / {cnt['not_applied']}"
                                 f" / {cnt['effective']} / {cnt['not_effective']} |")
        lines.append("")
    with open(os.path.join(folder, "summary.md"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"{len(rows)} runs -> {folder}/runs.csv, summary.csv, summary.md")


def groups_rows(groups: dict, scenario: str) -> list[dict]:
    return [r for (s, _), rs in groups.items() if s == scenario for r in rs]

if __name__ == "__main__":
    main()
