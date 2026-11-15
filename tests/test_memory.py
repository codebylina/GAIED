"""
Tests of Memory without LLM and without xTB.

Redis: MEMORY_TEST_REDIS=host:port (a real server, e.g. the "redis" service of docker-compose), otherwise
fakeredis (pip install fakeredis lupa). Each test uses its own namespace.

    python tests/test_memory.py          or          python -m pytest tests/test_memory.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory import Memory, KnownFailure, Domain, condition_id  # noqa: E402
from memory.domain import SCIENTIFIC, TECHNICAL  # noqa: E402


def _client():
    target = os.environ.get("MEMORY_TEST_REDIS")
    if target:
        import redis
        host, _, port = target.partition(":")
        return redis.Redis(host=host, port=int(port or 6379), decode_responses=True)
    import fakeredis
    return fakeredis.FakeRedis(decode_responses=True)


CLIENT = _client()


class ToyDomain(Domain):
    """Entities are words; feature has_x = 1 when the word contains an 'x'. Errors containing
    'impossible' are scientific, all others technical."""
    name, objective, direction = "word", "score", "max"

    def canonical(self, raw):
        raw = str(raw).strip().lower()
        return raw if raw.isalpha() else None

    def features(self, entity):
        return {"has_x": float("x" in entity), "length": float(len(entity))} if entity.isalpha() else {}

    def classify_failure(self, error):
        return SCIENTIFIC if "impossible" in error else TECHNICAL


def mem(agent="a", ns=None, **kw) -> Memory:
    return Memory(CLIENT, ToyDomain(), ns or f"test-{uuid.uuid4().hex[:8]}", agent, **kw)


M, P = "toy", {"precision": 1}


def measure(m: Memory, word: str, value: float, agent: str | None = None) -> dict:
    return m.record_observation(word, method=M, params=P, value=value, agent=agent)


# ---------------------------------------------------------------------- coordination
def test_claim_is_exclusive_and_release_checks_the_owner():
    m = mem()
    t1 = m.coord.claim("job", "a")
    assert t1 and m.coord.claim("job", "b") is None and m.coord.holder("job") == "a"
    assert not m.coord.release("job", "b|forged")          # not the owner: nothing happens
    assert m.coord.holder("job") == "a"
    assert m.coord.renew("job", t1) and m.coord.release("job", t1)
    assert m.coord.holder("job") is None


def test_expired_claim_cannot_be_released_by_its_old_owner():
    m = mem()
    old = m.coord.claim("job", "a", ttl_s=0.2)
    time.sleep(0.4)                                          # a crashed: its claim expired
    new = m.coord.claim("job", "b")
    assert new is not None
    assert not m.coord.release("job", old) and not m.coord.renew("job", old)
    assert m.coord.holder("job") == "b"


def test_concurrent_agents_compute_once():
    ns, calls = f"test-{uuid.uuid4().hex[:8]}", []

    async def run():
        async def compute(task_id):
            calls.append(task_id)
            await asyncio.sleep(0.3)
            return 4.2
        a, b = mem("a", ns), mem("b", ns)
        return await asyncio.gather(a.get_or_compute("Fox", compute, method=M, params=P, poll_s=0.05),
                                    b.get_or_compute("fox", compute, method=M, params=P, poll_s=0.05))
    assert asyncio.run(run()) == [4.2, 4.2]
    assert len(calls) == 1
    m = mem("c", ns)
    obs = m.obs.of_entity("fox", condition_id(M, P))
    assert [o["id"] for o in obs] == calls                  # the observation id is the task id
    stats = m.report()["stats"]
    assert stats["computed"] == 1 and stats["waited"] == 1 and stats["reused"] == 1


def test_late_agent_reuses_and_other_conditions_are_separate():
    m = mem()
    measure(m, "box", 1.0)
    assert m.get_status("BOX", M, P)["state"] == "success"
    assert m.get_status("box", M, {"precision": 2})["state"] == "unknown"
    got = m.get_or_compute_sync("box", lambda tid: 99.0, method=M, params=P)
    assert got == 1.0


def test_scientific_failure_is_not_retried_technical_is_retried_then_abandoned():
    m = mem(max_technical_attempts=2)

    def impossible(tid):
        raise ValueError("impossible molecule")
    for fn in (impossible,):
        try:
            m.get_or_compute_sync("cat", fn, method=M, params=P)
        except ValueError:
            pass
    try:
        m.get_or_compute_sync("cat", lambda tid: 1.0, method=M, params=P)
        raise AssertionError("a scientific failure must not be retried")
    except KnownFailure:
        pass

    def crash(tid):
        raise RuntimeError("node lost")
    for _ in range(2):
        try:
            m.get_or_compute_sync("dog", crash, method=M, params=P)
        except RuntimeError:
            pass
        assert m.get_status("dog", M, P)["state"] in ("retryable", "unknown")
    try:
        m.get_or_compute_sync("dog", lambda tid: 1.0, method=M, params=P)
        raise AssertionError("must give up after max_technical_attempts")
    except KnownFailure:
        pass
    m2 = mem(max_technical_attempts=3, ns=m.namespace)
    assert m2.get_or_compute_sync("dog", lambda tid: 5.0, method=M, params=P) == 5.0


def test_crashed_agent_claim_expires_and_work_is_taken_over():
    ns = f"test-{uuid.uuid4().hex[:8]}"
    a, b = mem("a", ns, claim_ttl_s=0.3), mem("b", ns, claim_ttl_s=0.3)
    cond = condition_id(M, P)
    assert a.coord.claim(f"{cond}/elk", "a", ttl_s=0.3)      # a claims then dies
    t0 = time.time()
    assert b.get_or_compute_sync("elk", lambda tid: 2.0, method=M, params=P, poll_s=0.05) == 2.0
    assert time.time() - t0 < 2


def test_long_computation_keeps_its_claim():
    ns = f"test-{uuid.uuid4().hex[:8]}"
    a, b = mem("a", ns, claim_ttl_s=0.3), mem("b", ns, claim_ttl_s=0.3)
    calls = []

    def slow(tid):
        calls.append(tid)
        time.sleep(1.0)                                       # > ttl: renewed in the background
        return 3.0
    th = threading.Thread(target=lambda: a.get_or_compute_sync("gnu", slow, method=M, params=P))
    th.start()
    time.sleep(0.5)
    assert b.get_status("gnu", M, P)["state"] == "running"
    assert b.get_or_compute_sync("gnu", slow, method=M, params=P, poll_s=0.05) == 3.0
    th.join()
    assert len(calls) == 1


# ---------------------------------------------------------------------- observations
def test_observations_are_append_only_and_idempotent():
    m = mem()
    o1 = measure(m, "ant", 1.0)
    o2 = measure(m, "ant", 3.0)
    assert o1["id"] != o2["id"] and len(m.obs.of_entity("ant", o1["condition"])) == 2
    same = m.record_observation("ant", method=M, params=P, value=100.0, task_id=o1["id"])
    assert same["value"] == 1.0                               # never overwritten
    assert m.obs.best(o1["condition"], 1) == [("ant", 3.0)]


def test_ingest_flowcept_records_is_idempotent_and_keeps_ids():
    class TaskDomain(ToyDomain):
        activity, input_key, output_key, method, default_params = "measure", "word", "score", M, P
    m = Memory(CLIENT, TaskDomain(), f"test-{uuid.uuid4().hex[:8]}")
    recs = [{"task_id": "t1", "activity_id": "measure", "status": "FINISHED", "used": {"word": "Owl"},
             "generated": {"score": 2.5}, "workflow_id": "w", "campaign_id": "c",
             "custom_metadata": {"agent": "x"}},
            {"task_id": "t2", "activity_id": "measure", "status": "ERROR", "used": {"word": "yak"},
             "generated": None, "stderr": "impossible"},
            {"task_id": "t3", "activity_id": "other", "used": {"word": "emu"}}]
    assert m.ingest_flowcept(recs) == 2 and m.ingest_flowcept(recs) == 0
    o = m.obs.get("t1")
    assert o["entity"] == "owl" and o["provenance"]["campaign_id"] == "c" and o["agent"] == "x"
    assert m.get_status("yak", M, P)["state"] == "failed"


# ---------------------------------------------------------------------- lessons and verification
def _team(m: Memory) -> dict[str, str]:
    """Words with an x score high, others low."""
    vals = {"fox": 9.0, "box": 8.0, "lynx": 7.5, "cat": 2.0, "dog": 1.0, "owl": 1.5}
    return {w: measure(m, w, v)["id"] for w, v in vals.items()}


def test_feature_effect_verified_rejected_and_insufficient():
    m = mem()
    ids = _team(m)
    good = m.propose_lesson("words with an x score higher", evidence_ids=list(ids.values()),
                            claim={"kind": "feature_effect", "feature": "has_x", "op": ">", "value": 0,
                                   "effect": "higher"}, type="success")
    wrong = m.propose_lesson("words with an x score lower", evidence_ids=list(ids.values()),
                             claim={"kind": "feature_effect", "feature": "has_x", "op": ">", "value": 0,
                                    "effect": "lower"})
    thin = m.propose_lesson("long words score higher", evidence_ids=[ids["lynx"], ids["cat"]],
                            claim={"kind": "feature_effect", "feature": "length", "op": ">", "value": 3,
                                   "effect": "higher"})
    unknown = m.propose_lesson("color helps", evidence_ids=[ids["fox"]],
                               claim={"kind": "feature_effect", "feature": "color", "op": ">", "value": 0})
    assert m.lessons.get(good["id"])["status"] == "pending"
    result = dict(m.verify_pending())
    assert result == {good["id"]: "verified", wrong["id"]: "rejected", thin["id"]: "needs_review",
                      unknown["id"]: "rejected"}
    assert m.lessons.get(good["id"])["verification"]["cited"]["n_with"] == 3


def test_cherry_picked_lesson_is_caught_by_the_other_observations():
    m = mem()
    ids = _team(m)
    # cites only the two worst x-words and the two best others: the opposite of what all data show
    picked = [ids["lynx"], ids["box"]]
    extra = [measure(m, w, v)["id"] for w, v in (("ape", 8.5), ("bee", 8.6))]
    l = m.propose_lesson("x lowers the score", evidence_ids=picked + extra,
                         claim={"kind": "feature_effect", "feature": "has_x", "op": ">", "value": 0,
                                "effect": "lower"})
    m.verify_pending()
    got = m.lessons.get(l["id"])
    assert got["status"] == "needs_review" and "contradicted by all comparable" in got["status_reason"]


def test_verified_lesson_is_rechecked_when_new_observations_arrive():
    m = mem()
    a, b = measure(m, "yak", 5.0)["id"], measure(m, "emu", 3.0)["id"]
    l = m.propose_lesson("yak beats emu", evidence_ids=[a, b],
                         claim={"kind": "comparison", "better": "yak", "worse": "emu"})
    m.verify_pending()
    assert m.lessons.get(l["id"])["status"] == "verified"
    measure(m, "emu", 9.0)
    measure(m, "emu", 9.5)                                   # emu measured again: mean now higher
    m.verify_pending()
    got = m.lessons.get(l["id"])
    assert got["status"] == "needs_review"
    assert [h["status"] for h in m.lessons.history(l["id"])] == ["pending", "verified", "needs_review"]


def test_common_rules_missing_evidence_no_claim_mixed_conditions():
    m = mem()
    o = measure(m, "ant", 1.0)["id"]
    other = m.record_observation("bat", method=M, params={"precision": 9}, value=2.0)["id"]
    missing = m.propose_lesson("x", evidence_ids=[o, "nope"], claim={"kind": "comparison", "better": "ant",
                                                                   "worse": "bat"})
    none = m.propose_lesson("explore bigger words", evidence_ids=[o], type="strategy")
    mixed = m.propose_lesson("bat beats ant", evidence_ids=[o, other],
                             claim={"kind": "comparison", "better": "bat", "worse": "ant"})
    nothing = m.propose_lesson("trust me", evidence_ids=[], claim={"kind": "comparison"})
    r = dict(m.verify_pending())
    assert r[missing["id"]] == "rejected" and r[none["id"]] == "needs_review"
    assert r[mixed["id"]] == "needs_review" and r[nothing["id"]] == "rejected"


def test_human_review_is_kept_by_recheck():
    m = mem()
    o = measure(m, "ant", 1.0)["id"]
    l = m.propose_lesson("explore longer words", evidence_ids=[o])
    m.verify_pending()
    m.review(l["id"], "verified", "agreed by the chemist")
    measure(m, "bee", 2.0)
    m.verify_pending()
    assert m.lessons.get(l["id"])["status"] == "verified"


def test_custom_claim_kind():
    m = mem()
    o = measure(m, "ant", 1.0)
    m.verifier.register("positive", lambda claim, cited, cond: (
        ("verified", "all positive", {}) if all(c["value"] > 0 for c in cited) else ("rejected", "no", {})))
    l = m.propose_lesson("scores are positive", evidence_ids=[o["id"]], claim={"kind": "positive"})
    assert dict(m.verify_pending())[l["id"]] == "verified"


# ---------------------------------------------------------------------- retrieval and provenance
def test_briefing_shows_only_verified_as_knowledge_and_respects_budget():
    sink = []
    m = mem(provenance_sink=sink.append, context=lambda: {"workflow_id": "wf", "campaign_id": "camp"})
    ids = _team(m)
    good = m.propose_lesson("words with an x score higher", evidence_ids=list(ids.values()),
                            claim={"kind": "feature_effect", "feature": "has_x", "op": ">", "value": 0,
                                   "effect": "higher"})
    bad = m.propose_lesson("dog beats fox", evidence_ids=[ids["dog"], ids["fox"]],
                           claim={"kind": "comparison", "better": "dog", "worse": "fox"})
    pending = m.propose_lesson("cats are underrated", evidence_ids=[ids["cat"]])
    m.verify_pending()
    m.lessons.propose("not verified yet", created_by="z", evidence_ids=[ids["cat"]])
    text = m.briefing(M, P)
    verified_part = text.split("Verified lessons")[1].split("Rejected")[0]
    assert "words with an x score higher" in verified_part
    assert "dog beats fox" in text.split("Rejected lessons")[1] and "not verified yet" not in text
    assert "cats are underrated" in text.split("Open questions")[1]
    assert len(m.briefing(M, P, max_chars=120)) <= 120
    kinds = [r["activity_id"] for r in sink]
    assert kinds.count("memory.propose_lesson") == 3 and kinds.count("memory.verify_lesson") == 3
    shown = [r for r in sink if r["activity_id"] == "memory.briefing"][0]
    assert good["id"] in shown["generated"]["lesson_ids"] and shown["workflow_id"] == "wf"
    lesson = m.lessons.get(bad["id"])
    verify_ids = {r["task_id"] for r in sink if r["activity_id"] == "memory.verify_lesson"}
    assert lesson["verification_task_id"] in verify_ids
    assert lesson["provenance"]["proposal_task_id"] in {r["task_id"] for r in sink}
    assert pending["provenance"]["campaign_id"] == "camp"


def test_proposer_maps_references_and_falls_back_to_context():
    m = mem()
    ids = _team(m)
    system, user, refs = m.lesson_prompt("x words are best", M, P)
    assert "o1 |" in user and "has_x" not in system
    answer = ('blah {"lessons": [{"statement": "x words score higher", "type": "success", '
              '"claim": {"kind": "feature_effect", "feature": "has_x", "op": ">", "value": 0, "effect": "higher"}, '
              '"evidence": ["o1", "o2", "o3", "o4", "o5", "o6"]}, '
              '{"statement": "keep exploring", "claim": {"kind": "none"}}]} trailing')
    lessons = m.propose_from_llm(answer, refs)
    assert set(lessons[0]["evidence_ids"]) == set(ids.values()) and lessons[0]["evidence_source"] == "cited"
    assert lessons[1]["evidence_source"] == "context"
    assert dict(m.verify_pending())[lessons[0]["id"]] == "verified"
    assert m.propose_from_llm("no json here", refs) == []


def test_known_entities_lists_measured_failed_and_running():
    m = mem("a")
    measure(m, "ant", 1.0)
    m.record_observation("bug", method=M, params=P, error="impossible")
    m.coord.claim(f"{condition_id(M, P)}/cow", "b")
    m.record_observation("b4d", method=M, params=P, error="impossible")      # not a valid entity
    assert set(m.known_entities(M, P)) == {"ant", "bug", "cow", "b4d"}
    assert "b4d" in m.briefing(M, P) and "invalid:" not in m.briefing(M, P)


def test_same_lesson_proposed_twice_is_merged_and_reverified():
    ns = f"test-{uuid.uuid4().hex[:8]}"
    a, b = mem("a", ns), mem("b", ns)
    ids = _team(a)
    claim = {"kind": "feature_effect", "feature": "has_x", "op": ">", "value": 0, "effect": "higher"}
    first = a.propose_lesson("x words score higher", evidence_ids=[ids["fox"], ids["cat"]], claim=claim)
    a.verify_pending()
    assert a.lessons.get(first["id"])["status"] == "needs_review"          # 1 entity per group only
    again = b.propose_lesson("Words containing x have higher scores", evidence_ids=list(ids.values()),
                             claim=dict(claim, value=0.0))
    assert again["id"] == first["id"] and again["endorsed_by"] == ["a", "b"]
    assert a.lessons.get(first["id"])["status"] == "verified"              # more evidence: re-verified
    assert a.lessons.counts()["verified"] == 1 and sum(a.lessons.counts().values()) == 1


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failures else 0)
