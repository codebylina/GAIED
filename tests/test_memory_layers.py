"""Unit tests of the four memory layers, the store and the lifecycle (no LLM, no xTB).

    python tests/test_memory_layers.py      (MEMORY_TEST_REDIS=localhost:6379 for a real Redis)
"""
from __future__ import annotations

import sys
import time

from helpers import CLIENT, M, P, ToyDomain, measure, new_memory, run_tests

from memory import ExperienceRecord, GuidanceRecord, KnowledgeRecord, ResultRecord, Store, ValidationError
from memory.domain import condition_id

COND = condition_id(M, P)


# ---------------------------------------------------------------------- schemas and store
def test_schemas_enforce_layer_separation():
    for bad in (
        lambda: ResultRecord(namespace="n", entity="ant", method=M, condition=COND, outcome="technical_failure").validate(),
        lambda: ResultRecord(namespace="n", entity="ant", method=M, condition=COND, outcome="negative").validate(),
        lambda: ExperienceRecord(namespace="n", kind="incident", summary="x", signature="s",
                                 confirmed_cause="guess").validate(),
        lambda: KnowledgeRecord(namespace="n", kind="claim", statement="x", status="validated").validate(),
        lambda: GuidanceRecord(namespace="n", kind="instruction", text="x", actions=[{"type": "teleport"}]).validate(),
        lambda: ResultRecord(namespace="n", entity="a", method=M, condition=COND, value=1.0,
                             provenance=[{"system": "flowcept"}]).validate(),
    ):
        try:
            bad()
            raise AssertionError("invalid record accepted")
        except ValidationError:
            pass


def test_store_versions_dedup_concurrency_and_archive():
    s = Store(CLIENT, f"test-{time.time_ns()}")
    rec, created = s.add(ExperienceRecord(kind="anomaly", summary="a", signature="sig", key="k1"), actor="x")
    again, created2 = s.add(ExperienceRecord(kind="anomaly", summary="b", signature="sig", key="k1"), actor="y")
    assert created and not created2 and again.id == rec.id
    # two concurrent writers: both increments survive (optimistic concurrency)
    import threading
    def bump():
        for _ in range(20):
            s.update("M2", rec.id, lambda r: setattr(r, "occurrences", r.occurrences + 1), "w", "bump")
    threads = [threading.Thread(target=bump) for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert s.get("M2", rec.id).occurrences == 61 and len(s.history("M2", rec.id)) == 61   # 1 + 60 versions
    s.archive("M2", rec.id, "x", "test")
    assert rec.id not in s.ids("M2") and s.get("M2", rec.id).status == "archived"
    assert s.find("M2", signature="sig") == []


# ---------------------------------------------------------------------- M1
def test_m1_results_append_only_ids_from_flowcept_and_conflicts():
    m = new_memory()
    r1 = measure(m, "Fox", 2.0, task_id="task-1")
    assert r1.id == "task-1" and r1.provenance[0]["task_id"] == "task-1" and r1.entity == "fox"
    assert measure(m, "fox", 99.0, task_id="task-1").value == 2.0           # idempotent on the task id
    measure(m, "fox", 2.0)
    measure(m, "fox", 2.5)                                                 # disagrees: both kept + anomaly
    look = m.results.lookup("FOX", M, P)
    assert look["state"] == "success" and look["n"] == 3
    anomalies = m.experience.open(kind="anomaly")
    assert len(anomalies) == 1 and len(anomalies[0].memory_refs) == 3
    m.record_result("cat", method=M, params=P, negative_detail="impossible word")
    assert m.results.lookup("cat", M, P)["state"] == "negative"
    assert m.results.lookup("fox", M, {"precision": 2})["state"] == "unknown"   # other condition


# ---------------------------------------------------------------------- M2
def test_m2_dedup_by_signature_and_cause_only_when_confirmed():
    m = new_memory()
    e1, c1 = m.experience.report("incident", "xtb crashed", "sig-1", by="a", facts=["f1"], agents=["a"])
    e2, c2 = m.experience.report("incident", "xtb crashed", "sig-1", by="b", facts=["f2"], agents=["b"],
                                 severity="high", explanations=["worker out of memory?"])
    assert c1 and not c2 and e2.id == e1.id and e2.occurrences == 2 and e2.severity == "high"
    assert e2.agents == ["a", "b"] and e2.confirmed_cause is None
    assert e2.candidate_explanations == ["worker out of memory?"]
    e3 = m.experience.confirm_cause(e1.id, "OOM kill seen in worker log", by="human", method="human")
    assert e3.status == "confirmed" and e3.confirmed_cause


# ---------------------------------------------------------------------- M3
def _team(m):
    vals = {"fox": 9.0, "box": 8.0, "lynx": 7.5, "cat": 2.0, "dog": 1.0, "owl": 1.5}
    return {w: measure(m, w, v).id for w, v in vals.items()}


X_HIGHER = {"type": "feature_effect", "feature": "has_x", "op": ">", "value": 0, "effect": "higher"}


def test_m3_supported_then_validated_only_by_later_evidence():
    m = new_memory()
    ids = _team(m)
    k, created, _ = m.propose_knowledge("words with an x score higher", claim=X_HIGHER, method=M, params=P,
                                        supporting=list(ids.values()))
    assert created and k.status == "supported"          # the data that suggested it cannot validate it
    for w, v in (("axe", 9.5), ("ox", 8.8), ("ant", 1.2), ("bee", 1.1)):
        measure(m, w, v)
    k = m.store.get("M3", k.id)
    assert k.status == "validated", k.evaluation
    assert [h["record"]["status"] for h in m.store.history("M3", k.id)][-1] == "validated"


def test_m3_contradiction_contests_without_deleting_and_reports_conflict():
    m = new_memory()
    ids = _team(m)
    k, _, _ = m.propose_knowledge("words with an x score higher", claim=X_HIGHER, method=M, params=P,
                                  supporting=list(ids.values()))
    assert k.status == "supported"
    for w, v in (("xenon", 0.1), ("pyx", 0.2), ("hex", 0.1), ("wax", 0.3), ("mix", 0.1),
                 ("ant", 9.9), ("bee", 9.8), ("cow", 9.7), ("elk", 9.6)):
        measure(m, w, v)                                  # new results disagree: x words now score lower
    k2 = m.store.get("M3", k.id)
    assert k2.status == "contested" and k2.contradicting
    assert m.store.get("M3", k.id) is not None and k.id in m.store.ids("M3")     # not deleted
    conflicts = m.experience.open(kind="knowledge_conflict")
    assert conflicts and f"M3:{k.id}" in conflicts[0].memory_refs


def test_m3_refuted_by_own_evidence_unverifiable_stays_proposed_and_human_lock():
    m = new_memory()
    ids = _team(m)
    bad, _, _ = m.propose_knowledge("dog beats fox", claim={"type": "comparison", "better": "dog", "worse": "fox"},
                                    method=M, params=P, supporting=[ids["dog"], ids["fox"]])
    advice, _, _ = m.propose_knowledge("explore longer words", method=M, params=P, supporting=[ids["fox"]])
    assert bad.status == "refuted" and advice.status == "proposed"
    m.knowledge.review(advice.id, "validated", "agreed by the domain expert", reviewer="human")
    measure(m, "zebra", 3.0)
    m.knowledge.reevaluate(advice.id, by="test")
    assert m.store.get("M3", advice.id).status == "validated"
    # same claim proposed again: merged, not duplicated
    again, created, _ = m.propose_knowledge("Dog is better than fox", claim={"type": "comparison", "better": "DOG",
                                            "worse": "fox"}, method=M, params=P, supporting=[ids["owl"]])
    assert not created and again.id == bad.id and ids["owl"] in again.supporting


def test_m3_negative_region_and_threshold():
    m = new_memory()
    for w in ("ab", "ac", "ad"):
        m.record_result(w, method=M, params=P, negative_detail="impossible: too short")
    for w, v in (("abcdef", 5.0), ("bcdefg", 6.0), ("cdefgh", 7.0)):
        measure(m, w, v)
    k, _, _ = m.propose_knowledge("words of 2 letters are impossible",
                                  claim={"type": "negative_region", "feature": "length", "op": "<=", "value": 2},
                                  kind="negative_finding", method=M, params=P,
                                  supporting=[r.id for r in m.results.in_condition(COND)])
    t, _, _ = m.propose_knowledge("long words score above 4",
                                  claim={"type": "threshold", "region": {"feature": "length", "op": ">", "value": 5},
                                         "op": ">", "value": 4}, method=M, params=P,
                                  supporting=[r.id for r in m.results.in_condition(COND)])
    assert k.status == "supported" and t.status == "supported"


# ---------------------------------------------------------------------- M4
def test_m4_publish_inbox_delivery_compliance_and_no_silent_reactivation():
    ns = f"test-{time.time_ns()}"
    analyst, a, b = new_memory("analysis", ns), new_memory("a", ns), new_memory("b", ns)
    g, created = analyst.guidance.publish("Do not propose fox or box again", by="analysis",
                                          actions=[{"type": "avoid_entities", "entities": ["fox", "box"]}],
                                          targets=["a"], nodes=["tool_calling"])
    assert created and g.status == "active"
    assert a.guidance.poll_inbox("a") == [g.id] and a.guidance.poll_inbox("a") == []    # durable cursor
    assert b.guidance.poll_inbox("b") == []                                            # targeted
    text, shown = a.context("tool_calling", M, P)
    assert g.id in shown["M4"] and "Do not propose fox" in text
    a.acknowledge(shown)
    assert a.check_guidance(["Fox", "cat"], M, P)[0]["compliant"] is False
    assert a.check_guidance(["cat"], M, P)[0]["compliant"] is True
    d = a.guidance.delivery(g.id)["a"]
    assert d["delivered_n"] == 1 and d["acknowledged_n"] == 1 and d["violated_n"] == 1 and d["applied_n"] == 1
    analyst.guidance.set_status(g.id, "ineffective", "analysis", "did not reduce duplicates")
    again, created = analyst.guidance.publish("Do not propose fox or box again", by="analysis",
                                              actions=[{"type": "avoid_entities", "entities": ["fox", "box"]}],
                                              targets=["a"], nodes=["tool_calling"])
    assert not created and again.status == "ineffective"


# ---------------------------------------------------------------------- retrieval and lifecycle
def test_context_labels_status_and_respects_budget():
    m = new_memory()
    ids = _team(m)
    m.propose_knowledge("words with an x score higher", claim=X_HIGHER, method=M, params=P,
                        supporting=list(ids.values()))
    m.propose_knowledge("dog beats fox", claim={"type": "comparison", "better": "dog", "worse": "fox"},
                        method=M, params=P, supporting=[ids["dog"], ids["fox"]])
    text, shown = m.context("plan", M, P, query="x words")
    assert "SUPPORTED" in text and "REFUTED (do NOT rely on it): dog beats fox" in text
    small, _ = m.context("plan", M, P, budget_tokens=30)
    assert len(small) <= 30 * 4


def test_lifecycle_expiry_staleness_and_quality():
    m = new_memory()
    g, _ = m.guidance.publish("short-lived", by="analysis", ttl_s=0.01)
    ids = _team(m)
    k, _, _ = m.propose_knowledge("words with an x score higher", claim=X_HIGHER, method=M, params=P,
                                  supporting=list(ids.values()))
    m.lifecycle.policy.stale_after_new_results = 2
    time.sleep(0.05)
    for w, v in (("axe", 9.5), ("ox", 8.8)):
        m.results.record(w, method=M, params=P, agent="a", value=v)    # bypasses on_result
    rep = m.lifecycle.run()
    assert rep.get("M4.expired") == 1 and rep.get("M3.reevaluated_stale") == 1
    assert m.store.get("M4", g.id).status == "retired"
    q = m.report()
    assert q["M1"]["active"] == 8 and "contradiction_rate" in q["M3"]


if __name__ == "__main__":
    sys.exit(1 if run_tests(dict(globals())) else 0)
