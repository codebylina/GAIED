"""
Tests of the Mem0 baseline without an LLM (Mem0's LLM and embedder are scripted).

    python tests/test_memlab.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.env import EnvSpec, SyntheticAdapter, SyntheticEnv  # noqa: E402
from bench.scripted import mem0_scripted  # noqa: E402
from memlab import InMemoryStore, MemoryConfig, make_memory  # noqa: E402
from memlab.mem0_memory import LocalMem0, Mem0Client, Mem0Service, build_mem0  # noqa: E402

ENV = SyntheticEnv(EnvSpec(n_slots=3, n_options=3, main_scale=1.0, interaction_scale=0.0, n_interactions=0,
                           noise_sd=0.0, n_forbidden=0, seed=1))


def local_mem0():
    return LocalMem0(build_mem0(tempfile.mkdtemp(), "t", scripted=mem0_scripted(int(os.environ.get("EMBED_DIMS", "768")))), run_id="campaign")


def memory(budget=2000, backend=None):
    return make_memory("mem0", InMemoryStore(), SyntheticAdapter(ENV), MemoryConfig(briefing_budget_chars=budget),
                       backend=backend or local_mem0())


def test_writes_are_queued_then_extracted_and_shared_across_agents():
    mem = memory()
    mem.record_episode("a1", "results", "Results of my experiments: A0-B1-C2: 1.5", 0)
    mem.submit_conclusion("a1", "- Option A0 increases the outcome compared with the other options of slot A.", 0)
    assert mem.backend.pending() == 2
    assert mem.maintain() == 2 and mem.backend.pending() == 0
    text = mem.briefing("a2", "plan", "option A0", 1)            # another agent sees a1's memories
    assert "Option A0 increases" in text
    items = mem.store.read_log("briefings")[-1]["items"]
    assert items and all(i["asserted"] is True for i in items)   # Mem0 presents what it stores as true


def test_plans_are_not_written():
    mem = memory()
    mem.record_episode("a1", "plan", "Plan: try A0-B1-C2.", 0)
    assert mem.backend.pending() == 0


def test_briefing_respects_budget():
    mem = memory(budget=150)
    for i in range(8):
        mem.submit_conclusion("a1", f"- Option A{i % 3} increases the outcome compared with the other options of slot A ({i}).", 0)
    mem.maintain()
    assert len(mem.briefing("a2", "plan", "option", 1)) <= 150


def test_no_memory_condition_shows_nothing_but_records_observations():
    mem = make_memory("none", InMemoryStore(), SyntheticAdapter(ENV), MemoryConfig())
    mem.ingest(ENV.run("A0-B0-C0", "a1"))
    assert mem.briefing("a1", "plan", "x", 0) == ""
    assert len(mem.export()["observations"]) == 1


def test_service_and_client():
    local = local_mem0()
    service = Mem0Service(local, 0)
    client = Mem0Client(f"http://127.0.0.1:{service.server.server_address[1]}")
    client.add("- Option B2 decreases the outcome compared with the other options of slot B.", "a1")
    local.q.join()                                   # the service's worker processed the write
    hits = client.search("option B2")
    service.drain()
    assert any("B2" in h["memory"] for h in hits)


def test_chemistry_adapter():
    from memlab.adapters.chemistry import ChemistryAdapter
    a = ChemistryAdapter("goal")
    assert a.canonical("OCC") == a.canonical("CCO")
    rec = {"task_id": "t1", "activity_id": "compute_ionization_energy", "status": "FINISHED", "ended_at": 5.0,
           "used": {"smiles": "OCC"}, "generated": {"ionization_energy": 0.5}, "custom_metadata": {"agent": "x"}}
    assert a.observation_from_record(rec)["entity"] == "CCO"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as exc:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(exc))
    sys.exit(1 if fails else 0)
