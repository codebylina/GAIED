"""Coordination tests (no LLM, no xTB). Demonstrates:
  1. two agents requesting the same simulation: only one computation occurs;
  2. a transient failure is retried, and never permanently prevents a future computation.

    python tests/test_coordination.py      (MEMORY_TEST_REDIS=localhost:6379 for a real Redis)
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
import uuid

from helpers import M, P, measure, new_memory, run_tests

from memory import KnownNegative, RetryPolicy, TemporarilyUnavailable
from memory.domain import condition_id

COND = condition_id(M, P)
FAST = RetryPolicy(max_attempts=3, backoff_s=0.01, transient_cooldown_s=0.3, permanent_cooldown_s=0.3,
                   claim_ttl_s=0.6)


def test_demo1_two_agents_same_simulation_one_computation():
    ns, calls = f"test-{uuid.uuid4().hex[:8]}", []

    async def compute(task_id):
        calls.append(task_id)
        await asyncio.sleep(0.3)
        return 4.2

    async def run():
        a, b = new_memory("a", ns), new_memory("b", ns)
        return await asyncio.gather(a.get_or_compute("Fox", compute, method=M, params=P, poll_s=0.02),
                                    b.get_or_compute("fox", compute, method=M, params=P, poll_s=0.02))
    assert asyncio.run(run()) == [4.2, 4.2]
    assert len(calls) == 1
    m = new_memory("c", ns)
    assert [r.id for r in m.results.of_entity("fox", COND)] == calls         # result id = task id
    metrics = m.store.metrics()
    assert metrics["decisions.computed"] == 1 and metrics["decisions.waited"] == 1
    assert metrics["decisions.reuse"] == 1


def test_demo2_transient_failure_is_retried_within_the_call():
    m = new_memory(policy=FAST)
    attempts = []

    def flaky(task_id):
        attempts.append(task_id)
        if len(attempts) < 3:
            raise ConnectionError("worker lost")
        return 7.0
    assert m.coord.get_or_compute_sync("owl", flaky, method=M, params=P) == 7.0
    assert len(attempts) == 3 and len(set(attempts)) == 3                    # one task id per attempt
    incident = m.experience.open(kind="incident")[0]
    assert incident.occurrences == 2 and incident.error_class == "transient"
    assert m.experience.open(kind="recovery_episode")
    assert m.results.lookup("owl", M, P)["id"] == attempts[-1]


def test_demo2_exhausted_transient_failure_only_cools_down():
    m = new_memory(policy=FAST)

    def down(task_id):
        raise TimeoutError("scheduler timeout")
    try:
        m.coord.get_or_compute_sync("emu", down, method=M, params=P)
        raise AssertionError("should fail")
    except TimeoutError:
        pass
    assert m.coord.status("emu", M, P)["state"] == "cooldown"
    try:
        m.coord.get_or_compute_sync("emu", lambda t: 1.0, method=M, params=P)
        raise AssertionError("cooldown ignored")
    except TemporarilyUnavailable:
        pass
    assert m.results.lookup("emu", M, P)["state"] == "unknown"               # no M1 record for a crash
    time.sleep(0.35)                                                          # the cooldown expires
    assert m.coord.get_or_compute_sync("emu", lambda t: 1.5, method=M, params=P) == 1.5


def test_scientific_failure_is_a_negative_result_not_retried():
    m = new_memory(policy=FAST)
    calls = []

    def impossible(task_id):
        calls.append(task_id)
        raise ValueError("impossible word")
    try:
        m.coord.get_or_compute_sync("yak", impossible, method=M, params=P)
    except ValueError:
        pass
    assert len(calls) == 1 and m.results.lookup("yak", M, P)["state"] == "negative"
    try:
        m.coord.get_or_compute_sync("yak", lambda t: 1.0, method=M, params=P)
        raise AssertionError("negative result recomputed")
    except KnownNegative:
        pass
    assert not m.experience.open(kind="incident")                             # not a technical failure


def test_permanent_failure_not_retried_now_but_later():
    m = new_memory(policy=FAST)
    calls = []

    def broken(task_id):
        calls.append(task_id)
        raise RuntimeError("missing binary xtb")
    try:
        m.coord.get_or_compute_sync("gnu", broken, method=M, params=P)
    except RuntimeError:
        pass
    assert len(calls) == 1 and m.experience.open(kind="incident")[0].severity == "high"
    time.sleep(0.35)
    assert m.coord.get_or_compute_sync("gnu", lambda t: 2.0, method=M, params=P) == 2.0


def test_lock_ownership_expiry_and_renewal():
    m = new_memory(policy=FAST)
    t1 = m.coord.claim("job", "a", ttl_s=0.2)
    assert t1 and m.coord.claim("job", "b") is None
    assert not m.coord.release("job", "b|forged") and m.coord.holder("job") == "a"
    time.sleep(0.3)
    t2 = m.coord.claim("job", "b")
    assert t2 and not m.coord.release("job", t1) and not m.coord.renew("job", t1)
    assert m.coord.holder("job") == "b"
    # a long computation keeps its claim alive (ttl 0.6 s, compute 1.5 s)
    ns, calls = f"test-{uuid.uuid4().hex[:8]}", []
    a, b = new_memory("a", ns, policy=FAST), new_memory("b", ns, policy=FAST)

    def slow(task_id):
        calls.append(task_id)
        time.sleep(1.5)
        return 3.0
    th = threading.Thread(target=lambda: a.coord.get_or_compute_sync("elk", slow, method=M, params=P))
    th.start()
    time.sleep(0.8)
    assert b.coord.status("elk", M, P)["state"] == "running"
    assert b.coord.get_or_compute_sync("elk", slow, method=M, params=P, poll_s=0.05) == 3.0
    th.join()
    assert len(calls) == 1


def test_crashed_agent_claim_expires_and_work_is_taken_over():
    ns = f"test-{uuid.uuid4().hex[:8]}"
    a, b = new_memory("a", ns, policy=FAST), new_memory("b", ns, policy=FAST)
    assert a.coord.claim(f"{COND}/ape", "a", ttl_s=0.3)                      # a claims, then dies
    t0 = time.time()
    assert b.coord.get_or_compute_sync("ape", lambda t: 5.0, method=M, params=P, poll_s=0.05) == 5.0
    assert time.time() - t0 < 2


def test_known_entities_and_reuse_across_agents():
    ns = f"test-{uuid.uuid4().hex[:8]}"
    a, b = new_memory("a", ns), new_memory("b", ns)
    measure(a, "bat", 1.0)
    a.record_result("cow", method=M, params=P, negative_detail="impossible")
    b.coord.claim(f"{COND}/doe", "b")
    assert set(a.known_entities(M, P)) == {"bat", "cow", "doe"}
    assert b.coord.get_or_compute_sync("BAT", lambda t: 99.0, method=M, params=P) == 1.0


if __name__ == "__main__":
    sys.exit(1 if run_tests(dict(globals())) else 0)
