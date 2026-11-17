"""Tests of the GAIED integration layer (memory/integrations/gaied.py) without GAIED, xTB or an LLM.

    python tests/test_integration.py      (MEMORY_TEST_REDIS=localhost:6379 for a real Redis)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid

from helpers import CLIENT, new_memory, run_tests

from memory.integrations import gaied
from memory.integrations.gaied import BasicMemory, NoMemory, StructuredMemory

PARAMS = {"accuracy": 0.05}


class FakeLLM:
    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    async def ainvoke(self, messages):
        self.calls += 1
        return type("R", (), {"content": self.answer})()


def chem_memory(agent="a", ns=None):
    from memory import SharedMemory, Store
    from memory.domains.chemistry import ChemistryDomain
    return SharedMemory(Store(CLIENT, ns or f"test-{uuid.uuid4().hex[:8]}"), ChemistryDomain(), agent)


def test_unreachable_redis_means_no_memory_not_a_crash():
    old = dict(os.environ)
    os.environ.update(MEMORY_MODE="structured", REDIS_HOST="unreachable.invalid")
    try:
        m = gaied.for_agent("a", PARAMS, "camp")
        assert isinstance(m, NoMemory)
        assert asyncio.run(m.compute("CCO", lambda task_id=None: asyncio.sleep(0, result=1.0))) == 1.0
    finally:
        os.environ.clear()
        os.environ.update(old)


def test_memory_lost_before_compute_runs_simulation_once_without_it():
    import redis
    mem = chem_memory()
    s = StructuredMemory(mem, PARAMS)
    calls = []

    async def run_xtb(task_id=None):
        calls.append(task_id)
        return 2.0
    mem.store.r = redis.Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.2)    # Redis goes away
    assert asyncio.run(s.compute("CCO", run_xtb)) == 2.0
    assert len(calls) == 1


def test_structured_hooks_context_known_compliance():
    ns = f"test-{uuid.uuid4().hex[:8]}"
    a, b = StructuredMemory(chem_memory("a", ns), PARAMS), StructuredMemory(chem_memory("b", ns), PARAMS)

    async def xtb(task_id=None):
        return 15.0
    asyncio.run(a.compute("OCC", xtb))                       # non-canonical spelling of ethanol
    assert "CCO" in b.known()
    text = b.context("tool_calling", "plan")
    assert "CCO: 15" in text
    b.m.guidance.publish("avoid ethanol", by="analysis", targets=["b"],
                         actions=[{"type": "avoid_entities", "entities": ["CCO"]}])
    assert "NEW" in b.context("tool_calling", "plan")
    b.acknowledge()
    b.after_proposals(["OCC", "CC#N"])
    d = list(b.m.guidance.delivery(b.m.store.find("M4")[0].id).values())[0]
    assert d["violated_n"] == 1 and d["acknowledged_n"] == 1


def test_knowledge_extraction_only_with_new_data():
    ns = f"test-{uuid.uuid4().hex[:8]}"
    s = StructuredMemory(chem_memory("a", ns), PARAMS)
    for smi, v in (("CC#N", 16.1), ("N#CC#N", 16.9), ("CCO", 14.8), ("CO", 15.4)):
        s.m.record_result(smi, method="xtb", params=PARAMS, value=v)
    llm = FakeLLM(json.dumps({"claims": [
        {"statement": "N#CC#N is better than CCO", "claim": {"type": "comparison", "better": "N#CC#N", "worse": "CCO"},
         "evidence": ["o2", "o3"]},
        {"statement": "explore nitriles", "claim": {"type": "none"}}]}))
    asyncio.run(s.learn("nitriles are good", llm))
    ks = s.m.store.find("M3")
    assert llm.calls == 1 and {k.status for k in ks} == {"supported", "proposed"}
    asyncio.run(s.learn("nitriles are good", llm))            # no new results: no LLM call
    assert llm.calls == 1
    empty = FakeLLM('{"claims": []}')
    for smi, v in (("C#N", 17.5), ("CCN", 14.2), ("CN", 14.4)):
        s.m.record_result(smi, method="xtb", params=PARAMS, value=v)
    asyncio.run(s.learn("nothing new", empty))
    assert empty.calls == 1 and len(s.m.store.find("M3")) == 2


def test_basic_mode_is_the_docker_local_prototype():
    host = (os.environ.get("MEMORY_TEST_REDIS") or "").split(":")[0]
    if not host:
        print("     (skipped: needs a real Redis, MEMORY_TEST_REDIS)")
        return
    os.environ["REDIS_HOST"] = host
    camp = f"test-{uuid.uuid4().hex[:8]}"
    a, b = BasicMemory("a", camp), BasicMemory("b", camp)
    calls = []

    async def xtb(task_id=None):
        calls.append(task_id)
        return 3.0

    async def both():
        return await asyncio.gather(a.compute("CCO", xtb), b.compute("OCC", xtb))
    assert asyncio.run(both()) == [3.0, 3.0] and len(calls) == 1
    assert "SHARED MEMORY" in b.context("plan")


if __name__ == "__main__":
    sys.exit(1 if run_tests(dict(globals())) else 0)
