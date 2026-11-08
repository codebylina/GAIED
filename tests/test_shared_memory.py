"""
Tests of the shared memory module, WITHOUT LLM or xTB (a few seconds).

In Docker (real Redis), from GAIED/docker:
    docker compose run --rm gaied python tests/test_shared_memory.py

Without Redis (simulated Redis, requires `pip install fakeredis`):
    python tests/test_shared_memory.py --fake
"""
import asyncio
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_memory import SharedMemory, SimulationFailed, canonical_smiles  # noqa: E402


def make_client():
    if "--fake" in sys.argv:
        import fakeredis
        return fakeredis.FakeRedis(decode_responses=True)
    import redis
    client = redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"), decode_responses=True)
    client.ping()
    return client


CLIENT = make_client()
CAMPAIGN = f"test-{uuid.uuid4().hex[:8]}"   # throwaway campaign, deleted at the end


def mem(agent):
    return SharedMemory(CLIENT, campaign_id=CAMPAIGN, agent_id=agent, claim_ttl=30)


def test_canonical():
    a, b = canonical_smiles("OCC"), canonical_smiles("CCO")
    try:
        import rdkit  # noqa: F401
        assert a == b, (a, b)
    except ImportError:
        print("    (RDKit missing: canonicalization not tested)")


def test_claim_is_exclusive():
    a, b = mem("A"), mem("B")
    assert a.claim("CCN")
    assert not b.claim("CCN"), "B must not be able to take a molecule already claimed"
    b.release("CCN")                      # B is not the owner: no effect
    assert a.claimed_by("CCN") == "A"
    a.release("CCN")
    assert b.claim("CCN"), "after release, B can take it"
    b.release("CCN")


def test_push_pull_results():
    a, b = mem("A"), mem("B")
    a.push_result("NC(N)=O", 14.96, task_id="t-1")
    a.push_result("C(=O)", None, error="Parse failure")
    rec = b.get_result("NC(N)=O")                 # B sees A's result
    assert rec["energy"] == 14.96 and rec["agent"] == "A" and rec["task_id"] == "t-1"
    assert canonical_smiles("NC(N)=O") in b.pull_results()
    assert b.get_result("C(=O)")["status"] == "error"
    assert b.best(1)[0][0] == canonical_smiles("NC(N)=O")


def test_no_duplicate_computation():
    """Two agents ask at the same time for partly identical molecules:
    each molecule must be computed only ONCE."""
    calls = []

    def compute_for(smiles):
        async def compute(task_id):
            calls.append(smiles)
            await asyncio.sleep(0.3)              # simulates an xTB computation
            return 10 + len(smiles) / 10
        return compute

    async def agent(m, molecules):
        return await asyncio.gather(*(m.get_or_compute(s, compute_for(s), poll=0.05) for s in molecules))

    async def main():
        a, b = mem("A"), mem("B")
        return await asyncio.gather(agent(a, ["CCCl", "CCBr", "CCI"]),
                                    agent(b, ["CCBr", "CCI", "CCF"]))

    ra, rb = asyncio.run(main())
    assert sorted(calls) == sorted(["CCCl", "CCBr", "CCI", "CCF"]), calls
    assert ra[1] == rb[0] and ra[2] == rb[1], "both agents get the same value"
    stats = CLIENT.hgetall(f"mem:{CAMPAIGN}:stats")
    assert int(stats.get("computed", 0)) >= 4


def test_failure_is_shared():
    """A stored failure prevents other agents from retrying the same invalid molecule."""
    calls = []

    async def failing(task_id):
        calls.append(1)
        raise ValueError("Parse failure for X1")

    async def main():
        a, b = mem("A"), mem("B")
        try:
            await a.get_or_compute("X1(", failing)
        except ValueError:
            pass
        try:
            await b.get_or_compute("X1(", failing)
        except SimulationFailed:
            return True
        return False

    assert asyncio.run(main()), "B must receive the stored failure"
    assert len(calls) == 1, "the invalid molecule must be tried only once"


def test_crashed_agent_lock_expires():
    """An agent that crashes while holding a lock does not block the others: the lock expires."""
    ghost = mem("ghost")
    assert ghost.claim("CCO", ttl=1)              # "ghost" takes the molecule then disappears

    async def compute(task_id):
        return 12.0

    t0 = time.time()
    value = asyncio.run(mem("A").get_or_compute("CCO", compute, poll=0.1))
    assert value == 12.0 and time.time() - t0 >= 0.9


def test_task_id_links_to_provenance():
    async def compute(task_id):
        assert task_id == "prov-42"
        return 13.0

    asyncio.run(mem("A").get_or_compute("CCC", compute, new_task_id=lambda: "prov-42"))
    assert mem("B").get_result("CCC")["task_id"] == "prov-42"


def test_prompt_rendering():
    a, b = mem("A"), mem("B")
    a.set_goal({"objective": "maximize ionization energy"})
    a.push_lesson("Fluorinated groups raise the ionization energy.")
    b.push_lesson("My own lesson, should not be shown to me.")
    a.push_instruction("Agent B is off track: change chemical family.", target="B", status="off_track")
    a.push_instruction("Only for A.", target="A")
    text = b.render_for_prompt()
    assert "maximize ionization energy" in text
    assert "Fluorinated groups" in text and "My own lesson" not in text
    assert "off track" in text and "Only for A" not in text
    assert canonical_smiles("NC(N)=O") in text


def test_in_progress_visible():
    a, b = mem("A"), mem("B")
    assert a.claim("CC#N")
    assert canonical_smiles("CC#N") in b.known_molecules()
    assert "RIGHT NOW" in b.render_for_prompt() and canonical_smiles("CC#N") in b.render_for_prompt()
    assert "RIGHT NOW" not in a.render_for_prompt(), "an agent does not see itself"
    a.release("CC#N")
    assert canonical_smiles("CC#N") not in b.known_molecules()


def test_lesson_evidence_with_task_ids():
    a = mem("A")
    a.push_result("CCCC", 11.0, task_id="t-butane")
    evidence = [{"smiles": "CCCC", "result": 11.0, "task_id": a.get_result("CCCC")["task_id"]},
                {"smiles": "C1CC((", "result": "Parse failure", "task_id": None}]
    a.push_lesson("Butane is low.", evidence=evidence)
    import json
    stored = json.loads(CLIENT.xrevrange(f"mem:{CAMPAIGN}:lessons", count=1)[0][1]["evidence"])
    assert stored[0]["task_id"] == "t-butane" and stored[1]["task_id"] is None, stored


def cleanup():
    for key in CLIENT.scan_iter(f"mem:{CAMPAIGN}:*"):
        CLIENT.delete(key)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    try:
        for t in tests:
            try:
                t()
                print(f"[OK]    {t.__name__}")
            except Exception as exc:
                failed += 1
                print(f"[FAIL]  {t.__name__}: {exc!r}")
    finally:
        cleanup()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
