"""
Coordination micro-benchmark (no LLM, no xTB, a few minutes): how fast does the team learn?

Answers the question "how fast does agent B learn that a molecule is taken / computed /
that it must stop?" for the two notification modes of EVIDENT (push and poll), plus the
time to recover a molecule reserved by a crashed agent.

    docker compose run --rm --no-deps gaied python experiments/bench_coordination.py
Output: experiments/results/bench_coordination.md (and .json)
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import statistics
import sys
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import redis  # noqa: E402

from evident import EvidentMemory, StoppedByDirective  # noqa: E402

CLIENT = redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"), decode_responses=True)
REPS = int(os.environ.get("BENCH_REPS", "20"))


def _mem(c, agent, notify, ttl=30):
    return EvidentMemory(CLIENT, c, agent, notify=notify, claim_ttl=ttl)


def visibility(notify):
    """A reserves a molecule at proposal time -> B sees it as taken (one read)."""
    c = f"bench-{uuid.uuid4().hex[:6]}"
    a, b = _mem(c, "A", notify), _mem(c, "B", notify)
    t0 = time.perf_counter()
    a.record_proposals([f"C{'C' * (hash(c) % 5)}O"], own_previous=[])
    taken = bool(b.in_progress())
    return time.perf_counter() - t0, taken, c


def wake(notify):
    """B waits for a molecule A computes: fact published -> B gets the value."""
    c = f"bench-{uuid.uuid4().hex[:6]}"
    a, b = _mem(c, "A", notify), _mem(c, "B", notify)

    async def slow(task_id):            # random duration: the poll phase is sampled fairly
        await asyncio.sleep(0.5 + random.random())
        return 1.0

    async def main():
        a.claim("CCCN")
        ta = asyncio.create_task(a._compute_and_store("CCCN", slow, None, time.time()))
        await asyncio.sleep(0.05)
        tb = asyncio.create_task(b.evaluate_molecule("CCCN", slow))
        await ta
        t_pub = time.time()
        a.release("CCCN")
        await tb
        return time.time() - t_pub

    out = asyncio.run(main())
    a.close(), b.close()
    return out, c


def stop(notify):
    """STOP sent by an analysis agent -> running computation cancelled."""
    c = f"bench-{uuid.uuid4().hex[:6]}"
    a, ctrl = _mem(c, "A", notify), _mem(c, "analysis", notify)

    async def slow(task_id):
        await asyncio.sleep(10)
        return 1.0

    async def main():
        task = asyncio.create_task(a.evaluate_molecule("CCCCCO", slow))
        await asyncio.sleep(0.2 + random.random())
        sent = time.time()
        ctrl.send_directive("A", "stop", "benchmark", rule="bench")
        try:
            await task
        except StoppedByDirective:
            return time.time() - sent

    out = asyncio.run(main())
    a.close()
    return out, c


def takeover(notify, ttl=2):
    """A crashed agent holds a reservation -> another agent computes the molecule."""
    c = f"bench-{uuid.uuid4().hex[:6]}"
    ghost, b = _mem(c, "ghost", notify, ttl), _mem(c, "B", notify, ttl)
    ghost.claim("CCCCCN")

    async def quick(task_id):
        return 1.0

    t0 = time.time()
    asyncio.run(b.evaluate_molecule("CCCCCN", quick))
    b.close()
    return time.time() - t0, c


def summarize(xs):
    xs = [x * 1000 for x in xs]
    return {"median_ms": round(statistics.median(xs), 2), "p95_ms": round(sorted(xs)[int(0.95 * (len(xs) - 1))], 2),
            "max_ms": round(max(xs), 2), "n": len(xs)}


def main():
    CLIENT.ping()
    results, campaigns = {}, []
    for notify in ("push", "poll"):
        r = {}
        vis = [visibility(notify) for _ in range(REPS)]
        r["reservation visible to the team"] = summarize([v[0] for v in vis])
        campaigns += [v[2] for v in vis]
        w = [wake(notify) for _ in range(REPS)]
        r["fact published -> waiting agent resumes"] = summarize([x[0] for x in w])
        campaigns += [x[1] for x in w]
        s = [stop(notify) for _ in range(REPS)]
        r["STOP sent -> computation cancelled"] = summarize([x[0] for x in s])
        campaigns += [x[1] for x in s]
        t = [takeover(notify) for _ in range(max(3, REPS // 4))]
        r["crashed holder (lease 2 s) -> molecule taken over"] = summarize([x[0] for x in t])
        campaigns += [x[1] for x in t]
        results[notify] = r
        print(notify, json.dumps(r, indent=1))
    for c in campaigns:
        for key in CLIENT.scan_iter(f"ev:{c}:*"):
            CLIENT.delete(key)
    out = os.path.join(ROOT, "experiments", "results")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "bench_coordination.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    lines = ["# Coordination benchmark (EVIDENT, Redis, no LLM)", "",
             f"{REPS} repetitions per measure; poll intervals: facts 1 s, directives 0.5 s.", "",
             "| measure | push: median / p95 (ms) | poll: median / p95 (ms) |", "|---|---|---|"]
    for m in results["push"]:
        p, q = results["push"][m], results["poll"][m]
        lines.append(f"| {m} | {p['median_ms']} / {p['p95_ms']} | {q['median_ms']} / {q['p95_ms']} |")
    with open(os.path.join(out, "bench_coordination.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
