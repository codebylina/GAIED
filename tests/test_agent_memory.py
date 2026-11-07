"""
Integration test: the shared memory plugged into the REAL agent code
(XTBSimulationAgent.compute_ionization_energy), with a real xTB computation but NO LLM.
Duration: less than a minute.

In Docker, from GAIED/docker:
    docker compose run --rm gaied python tests/test_agent_memory.py

What is checked:
  1. two agents ask AT THE SAME TIME for the same molecule, written in two ways
     (CCO and OCC) -> xTB runs only ONCE and both get the same energy;
  2. a third agent, started later, reuses the result without computing;
  3. an invalid molecule fails once, then the failure is reused;
  4. each actual xTB computation is recorded in the provenance (subtype parsl_task),
     with the same task_id as the result stored in the shared memory;
  5. with the memory disabled, the agent works as before (and recomputes).

Options (to check without Docker): --fake (simulated Redis), --fake-xtb (simulated xTB).
"""
import asyncio
import os
import sys
import time
import types
import uuid
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docker"))

if "--fake-xtb" in sys.argv:                      # simulated xTB (machine without xTB)
    fake = types.ModuleType("xtb.ase.calculator")
    fake.XTB = object
    sys.modules.update({"xtb": types.ModuleType("xtb"), "xtb.ase": types.ModuleType("xtb.ase"),
                        "xtb.ase.calculator": fake})

import mol_design_agents as mda                   # noqa: E402
from shared_memory import SharedMemory, SimulationFailed, canonical_smiles  # noqa: E402

if "--fake-xtb" in sys.argv:
    def _fake_energy(config, smiles):
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Parse failure for {smiles}")
        time.sleep(1.0)
        return 10.0 + mol.GetNumAtoms() / 10
    mda._compute_energy_sync = _fake_energy        # inherited by the xTB processes (fork)


def make_client():
    if "--fake" in sys.argv:
        import fakeredis
        return fakeredis.FakeRedis(decode_responses=True)
    import redis
    client = redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"), decode_responses=True)
    client.ping()
    return client


class RecordingInterceptor:
    """Stands in for Flowcept's AcademyInterceptor: keeps the recorded tasks in a list."""

    def __init__(self):
        self._workflow_id = "test-workflow"
        self._campaign_id = "test-campaign"
        self.tasks = []

    def intercept_task(self, task):
        self.tasks.append(task)


CLIENT = make_client()
CAMPAIGN = f"test-agent-{uuid.uuid4().hex[:8]}"
POOL = ProcessPoolExecutor(max_workers=2)
PROV = RecordingInterceptor()


def make_agent(name, with_memory=True):
    agent = mda.XTBSimulationAgent(mda.XTBConfig(), "CNC(N)=O")
    agent._academy_interceptor = PROV
    agent.pool = POOL
    agent.agent_name = name
    agent.memory = SharedMemory(CLIENT, campaign_id=CAMPAIGN, agent_id=name) if with_memory else None
    return agent


def stats():
    return {k: int(v) for k, v in CLIENT.hgetall(f"mem:{CAMPAIGN}:stats").items()}


def xtb_records(smiles=None):
    recs = [t for t in PROV.tasks if t.get("subtype") == "parsl_task"]
    if smiles is not None:
        recs = [t for t in recs if canonical_smiles(t["used"]["smiles"]) == canonical_smiles(smiles)]
    return recs


def test_concurrent_agents_compute_once():
    a, b = make_agent("A"), make_agent("B")

    async def main():
        return await asyncio.gather(a.compute_ionization_energy("CCO"),
                                    b.compute_ionization_energy("OCC"))

    ea, eb = asyncio.run(main())
    s = stats()
    assert ea == eb, (ea, eb)
    assert s.get("computed") == 1, f"xTB must run only once: {s}"
    assert s.get("waited", 0) + s.get("reused", 0) >= 1, s
    print(f"    ethanol energy = {ea:.3f}; counters = {s}")


def test_late_agent_reuses():
    before = stats().get("computed", 0)
    t0 = time.time()
    e = asyncio.run(make_agent("C").compute_ionization_energy("C(O)C"))
    assert stats().get("computed", 0) == before, "no new computation expected"
    print(f"    reused in {time.time() - t0:.3f} s (energy {e:.3f})")


def test_invalid_molecule_failure_is_shared():
    a, b = make_agent("A"), make_agent("B")
    try:
        asyncio.run(a.compute_ionization_energy("C1CC(("))
        raise AssertionError("an invalid molecule must fail")
    except SimulationFailed:
        raise AssertionError("A must get the real error, not a stored failure")
    except Exception:
        pass
    try:
        asyncio.run(b.compute_ionization_energy("C1CC(("))
        raise AssertionError("B must receive the failure")
    except SimulationFailed:
        pass
    assert stats().get("computed_error") == 1 and stats().get("reused_error") == 1, stats()
    errors = xtb_records("C1CC((")
    assert len(errors) == 1 and errors[0]["status"] == "ERROR" and errors[0]["stderr"], errors


def test_xtb_task_recorded_with_memory_task_id():
    recs = xtb_records("CCO")
    assert len(recs) == 1, f"exactly one xTB record expected for ethanol, got {len(recs)}"
    rec = SharedMemory(CLIENT, campaign_id=CAMPAIGN, agent_id="X").get_result("CCO")
    assert rec and rec["task_id"] == recs[0]["task_id"], (rec, recs[0]["task_id"])
    uuid.UUID(rec["task_id"])
    assert recs[0]["generated"]["ionization_energy"] == rec["energy"]
    assert recs[0]["ended_at"] >= recs[0]["started_at"]
    print(f"    CCO computed by {rec['agent']}, Flowcept task {rec['task_id']}")


def test_without_memory_still_works():
    a, b = make_agent("A", with_memory=False), make_agent("B", with_memory=False)

    async def main():
        return await asyncio.gather(a.compute_ionization_energy("CC"),
                                    b.compute_ionization_energy("CC"))

    ea, eb = asyncio.run(main())
    assert ea == eb and "CC" in a.molecule_cache and "CC" in b.molecule_cache
    assert CLIENT.hgetall(f"mem:{CAMPAIGN}:result:{canonical_smiles('CC')}") == {}
    assert len(xtb_records("CC")) == 2, "without memory, both agents compute (and are recorded)"


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
        for key in CLIENT.scan_iter(f"mem:{CAMPAIGN}:*"):
            CLIENT.delete(key)
        POOL.shutdown()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
