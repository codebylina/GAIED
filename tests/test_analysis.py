"""Tests of the analysis service (no LLM, no xTB; Flowcept task records in Flowcept's format).
Demonstrates:
  3. the analysis detects redundant computations / exploration using Flowcept provenance and memory;
  5. it publishes guidance in real time and scientific agents receive it (delivery, compliance, effect);
  6. it restarts without losing pending analysis work.

    python tests/test_analysis.py      (MEMORY_TEST_REDIS=localhost:6379 for a real Redis)
"""
from __future__ import annotations

import json
import sys
import time
import uuid

from helpers import M, P, ToyDomain, measure, new_memory, run_tests

from analysis import AnalysisConfig, AnalysisService
from memory import SharedMemory, Store
from memory.domain import condition_id


class ToyTaskDomain(ToyDomain):
    activity, input_key, output_key, method, params_key = "measure", "word", "score", M, "params"


def setup(n_agents=2):
    ns = f"test-{uuid.uuid4().hex[:8]}"
    camp = f"camp-{uuid.uuid4().hex[:6]}"
    agents = [SharedMemory(Store(new_memory().store.r, ns), ToyTaskDomain(), f"agent{i}") for i in range(n_agents)]
    for a in agents:
        a.register_agent(M, P, {"langgraph": f"wf-{a.agent}"}, campaign_id=camp)
    analyst = SharedMemory(Store(agents[0].store.r, ns), ToyTaskDomain(), "analysis")
    svc = AnalysisService(analyst, AnalysisConfig(min_after=3), source=None, period_s=0, live=False)
    return ns, camp, agents, svc


def task(camp, word, value=None, error=None, agent="x"):
    """A Flowcept task record as written by GAIED's _record_xtb_task."""
    return {"task_id": str(uuid.uuid4()), "activity_id": "measure", "subtype": "parsl_task", "campaign_id": camp,
            "status": "ERROR" if error else "FINISHED", "used": {"word": word, "params": P},
            "generated": None if error else {"score": value}, "stderr": error,
            "custom_metadata": {"agent": agent}, "started_at": time.time(), "ended_at": time.time()}


def drain(svc):
    while svc.read_events(block_ms=10):
        pass


def test_demo3_redundant_computations_detected_from_flowcept_provenance():
    ns, camp, agents, svc = setup()
    # provenance of a campaign WITHOUT shared memory: fox computed by both agents
    svc._apply_tasks([task(camp, "fox", 9.0, agent="agent0"), task(camp, "Fox", 9.0, agent="agent1"),
                      task(camp, "owl", 1.0, agent="agent0"), task("other-campaign", "fox", 9.0)])
    drain(svc)
    found = svc.analyze()
    red = [f for f in found if f["kind"] == "redundant_computation"]
    assert len(red) == 1
    ep = svc.m.store.get("M2", red[0]["finding"])
    assert ep.status == "confirmed" and "without the shared memory" in ep.confirmed_cause
    assert len(ep.provenance) == 2 and all(p["campaign_id"] == camp for p in ep.provenance)
    g = svc.m.store.get("M4", red[0]["guidance"]["id"])
    assert g.status == "active" and set(g.targets) == {"agent0", "agent1"}
    assert svc.analyze() == []                                     # unchanged finding: not fed back again


def test_demo3_and_5_redundant_exploration_guidance_received_in_real_time():
    ns, camp, (a, b), svc = setup()
    for w, v in (("fox", 9.0), ("box", 8.0), ("cat", 2.0), ("dog", 1.0)):
        a.coord.get_or_compute_sync(w, lambda t, v=v: v, method=M, params=P)
    for w in ("fox", "box", "cat", "dog", "ant"):                 # b proposes mostly known molecules
        b.coord.get_or_compute_sync(w, lambda t: 3.0, method=M, params=P)
    drain(svc)
    found = svc.analyze()
    expl = [f for f in found if f["kind"] == "inefficient_exploration"]
    assert expl, found
    gid = expl[0]["guidance"]["id"]
    g = svc.m.store.get("M4", gid)
    assert g.targets == ["agent1"] and g.baseline["agent1"]["known_share"] == 0.8
    # agent b receives it during execution, in its next prompt (durable inbox)
    text, shown = b.context("tool_calling", M, P)
    assert gid in shown["M4"] and "NEW" in text and "already measured" in text
    assert gid not in a.context("tool_calling", M, P)[1]["M4"]    # targeted: a does not get it
    b.acknowledge(shown)
    assert b.check_guidance(["elk", "emu"], M, P)[0]["compliant"] is True
    for w in ("elk", "emu", "gnu", "yak"):                         # behaviour after the guidance
        b.coord.get_or_compute_sync(w, lambda t: 4.0, method=M, params=P)
    drain(svc)
    verdicts = svc.feedback.evaluate_guidance()
    assert {"guidance": gid, "verdict": "effective"} in verdicts
    d = svc.m.guidance.delivery(gid)["agent1"]
    assert d["delivered_n"] == 1 and d["acknowledged_n"] == 1 and d["applied_n"] == 1
    assert svc.m.store.get("M4", gid).status == "validated"


def test_ineffective_guidance_is_detected():
    ns, camp, (a, b), svc = setup()
    for w, v in (("fox", 9.0), ("box", 8.0), ("cat", 2.0), ("dog", 1.0)):
        a.coord.get_or_compute_sync(w, lambda t, v=v: v, method=M, params=P)
    for w in ("fox", "box", "cat", "dog", "ant"):
        b.coord.get_or_compute_sync(w, lambda t: 3.0, method=M, params=P)
    drain(svc)
    gid = [f for f in svc.analyze() if f["kind"] == "inefficient_exploration"][0]["guidance"]["id"]
    b.context("tool_calling", M, P)
    for w in ("fox", "box", "cat", "dog", "ant", "fox"):           # b ignores it
        b.coord.get_or_compute_sync(w, lambda t: 3.0, method=M, params=P)
    drain(svc)
    assert {"guidance": gid, "verdict": "ineffective"} in svc.feedback.evaluate_guidance()
    assert svc.m.experience.open(kind="ineffective_guidance")


def test_demo6_restart_without_losing_pending_work():
    ns, camp, (a, b), svc = setup()
    r = svc.m.store.r
    for w, v in (("fox", 9.0), ("box", 8.0), ("cat", 2.0), ("dog", 1.0)):
        a.coord.get_or_compute_sync(w, lambda t, v=v: v, method=M, params=P)
    for w in ("fox", "box", "cat", "dog", "ant"):
        b.coord.get_or_compute_sync(w, lambda t: 3.0, method=M, params=P)
    # the service reads the events, then crashes before applying / acknowledging them
    resp = r.xreadgroup("analysis", "analysis-1", {svc.stream: ">"}, count=1000)
    assert resp and len(resp[0][1]) > 10
    assert r.xpending(svc.stream, "analysis")["pending"] == len(resp[0][1])
    # a new instance (same consumer name) recovers them
    svc2 = AnalysisService(SharedMemory(Store(r, ns), ToyTaskDomain(), "analysis"), AnalysisConfig(),
                           source=None, period_s=0, live=False)
    assert svc2.recover() == len(resp[0][1])
    assert r.xpending(svc.stream, "analysis")["pending"] == 0
    assert [f for f in svc2.analyze() if f["kind"] == "inefficient_exploration"]
    # idempotent: re-delivering already applied events changes nothing
    tl = len(svc2.state.timeline("agent1"))
    svc2._apply_entries(r.xrange(svc.stream, "-", "+"))
    assert len(svc2.state.timeline("agent1")) == tl


def test_failing_event_is_retried_then_dead_lettered():
    ns, camp, (a, b), svc = setup()
    svc.max_deliveries = 2
    a.store.emit("decision", decision="computed", agent="agent0", entity="fox", condition=condition_id(M, P))
    original = svc.state.apply_event
    svc.state.apply_event = lambda ev: (_ for _ in ()).throw(RuntimeError("bug")) if ev.get("entity") == "fox" else original(ev)
    drain(svc)
    for _ in range(5):                                   # redelivered (claimed again) until dead-lettered
        pending = [p["message_id"] for p in svc.m.store.r.xpending_range(svc.stream, "analysis", "-", "+", 10)]
        if not pending:
            break
        svc.m.store.r.xclaim(svc.stream, "analysis", "analysis-1", 0, pending)
        svc._dead_letters()
    assert svc.m.store.r.xlen(svc.m.store.k("an", "dead")) == 1
    assert svc.m.store.r.xpending(svc.stream, "analysis")["pending"] == 0


def test_contested_knowledge_retires_dependent_guidance():
    ns, camp, (a, b), svc = setup()
    vals = {"fox": 9.0, "box": 8.0, "lynx": 7.5, "cat": 2.0, "dog": 1.0, "owl": 1.5}
    ids = [measure(a, w, v).id for w, v in vals.items()]
    k, _, _ = a.propose_knowledge("x words score higher", claim={"type": "feature_effect", "feature": "has_x",
                                  "op": ">", "value": 0, "effect": "higher"}, method=M, params=P, supporting=ids)
    g, _ = svc.m.guidance.publish("prefer x words", by="analysis", memory_refs=[f"M3:{k.id}"],
                                  actions=[{"type": "prefer_region", "feature": "has_x", "op": ">", "value": 0}],
                                  condition=condition_id(M, P), baseline={"agent0": {}})
    for w, v in (("xenon", 0.1), ("pyx", 0.2), ("hex", 0.1), ("wax", 0.3), ("mix", 0.1),
                 ("ant", 9.9), ("bee", 9.8), ("cow", 9.7), ("elk", 9.6)):
        measure(b, w, v)
    assert svc.m.store.get("M3", k.id).status == "contested"
    drain(svc)
    found = svc.analyze()
    assert any(f["kind"] == "knowledge_conflict" for f in found)
    assert {"guidance": g.id, "verdict": "retired_outdated"} in svc.feedback.evaluate_guidance()


def test_recurrent_technical_failures_from_provenance():
    ns, camp, agents, svc = setup()
    svc._apply_tasks([task(camp, w, error=f"worker {i} lost connection", agent="agent0")
                      for i, w in enumerate(("fox", "owl", "emu"))])
    drain(svc)
    rec = [f for f in svc.analyze() if f["kind"] == "recurring_failure"]
    assert rec and svc.m.store.get("M4", rec[0]["guidance"]["id"]).kind == "recovery_strategy"


def test_unsuccessful_region_proposes_knowledge_but_never_establishes_it():
    ns, camp, (a, b), svc = setup()
    for w, v in (("fox", 1.0), ("box", 1.1), ("lynx", 0.9), ("cat", 5.0), ("dog", 6.0), ("owl", 5.5), ("ant", 5.2)):
        measure(a, w, v)
    a.register_agent(M, P, {}, campaign_id=camp)
    drain(svc)
    found = [f for f in svc.analyze() if "knowledge" in f]
    assert found
    k = svc.m.store.get("M3", found[0]["knowledge"]["id"])
    assert k.source == "analysis" and k.status in ("proposed", "supported")      # never validated by analysis
    assert k.status != "validated"


if __name__ == "__main__":
    sys.exit(1 if run_tests(dict(globals())) else 0)
