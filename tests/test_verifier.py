"""
Tests of the knowledge verification layer, without an LLM (scripted formalization).

    python tests/test_verifier.py
"""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("HF_HUB_OFFLINE", "1")           # no BM25 model download during tests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.env import EnvSpec, SyntheticAdapter, SyntheticEnv  # noqa: E402
from bench.scripted import FACT_TEST, GEN_TEST, ScriptedLLM, mem0_scripted  # noqa: E402
from memlab import LLM, InMemoryStore, MemoryConfig, make_memory  # noqa: E402
from memlab.mem0_memory import LocalMem0, build_mem0  # noqa: E402
from memlab.verifier import VerificationConfig, Verifier  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        self.t += 1.0
        return self.t


def make_env():
    """An environment where option A0 beats every other option of slot A everywhere."""
    for seed in range(300):
        env = SyntheticEnv(EnvSpec(n_slots=3, n_options=3, main_scale=1.0, interaction_scale=0.0, n_interactions=0,
                                   noise_sd=0.0, n_forbidden=0, seed=seed))
        if all(env.main[("A", 0)] > env.main[("A", o)] + 0.5 for o in (1, 2)):
            return env
    raise RuntimeError("no suitable environment")


ENV = make_env()


def setup(**vcfg):
    store, clock = InMemoryStore(), Clock()
    adapter = SyntheticAdapter(ENV)
    llm = LLM(provider="scripted", scripted=ScriptedLLM(ENV))
    v = Verifier(store, adapter, llm, VerificationConfig(**vcfg), clock=clock)
    return store, adapter, v, clock


def observe(adapter, store, clock, entity, agent):
    from memlab.base import Memory
    rec = ENV.run(entity, agent)
    rec["ended_at"] = clock()
    m = Memory(store, adapter, MemoryConfig(), clock=clock)
    return m.ingest(rec)


def a0_designs(best=True):
    rows = [r for r in ENV.population() if r["status"] == "ok" and r["slot_A"] == 0]
    return [r["entity"] for r in sorted(rows, key=lambda r: -r["value"] if best else r["value"])]


def others():
    return [r["entity"] for r in ENV.population() if r["status"] == "ok" and r["slot_A"] != 0]


# --------------------------------------------------------------------------- facts
def test_fact_verified_unsupported_refuted():
    store, adapter, v, clock = setup()
    o = observe(adapter, store, clock, others()[0], "a1")
    v.register("m1", f"Results: {o['entity']}: {o['value']:.4f}", "a1")          # matches the measurement
    v.register("m2", f"Results: {a0_designs()[0]}: 1.2345", "a1")                 # never measured
    v.register("m3", f"Results: {o['entity']}: {o['value'] + 5:.4f}", "a1")       # wrong value
    while v.step():
        pass
    st = {i: d["status"] for i, d in v.status_of(["m1", "m2", "m3"]).items()}
    assert st == {"m1": "verified", "m2": "unsupported", "m3": "refuted"}, st


# --------------------------------------------------------------------------- generalizations
def _generalization(**vcfg):
    store, adapter, v, clock = setup(**vcfg)
    for e in a0_designs()[:2] + others()[:3]:
        observe(adapter, store, clock, e, "a1")
    v.register("g", "- Option A0 increases the outcome compared with the other options of slot A.", "a1")
    while v.step():
        pass
    return store, adapter, v, clock


def test_exploratory_evidence_only_gives_a_hypothesis():
    _, _, v, _ = _generalization()
    assert v.status_of(["g"])["g"]["status"] == "hypothesis"


def test_confirmation_then_independent_confirmation():
    store, adapter, v, clock = _generalization()
    observe(adapter, store, clock, a0_designs()[2], "a1")          # after the memory, same author
    while v.step():
        pass
    assert v.status_of(["g"])["g"]["status"] == "supported"
    observe(adapter, store, clock, a0_designs()[3], "a2")          # after the memory, another agent
    while v.step():
        pass
    d = v.status_of(["g"])["g"]
    assert d["status"] == "established" and d["eval"]["independent_agents"] == ["a2"]


def test_contradiction_refutes():
    store, adapter, v, clock = setup()
    for e in a0_designs()[:2] + others()[:3]:
        observe(adapter, store, clock, e, "a1")
    v.register("g", "- Option A0 decreases the outcome compared with the other options of slot A.", "a1")
    while v.step():
        pass
    assert v.status_of(["g"])["g"]["status"] == "refuted"


def test_without_layers_a_supported_verdict_establishes():
    _, _, v, _ = _generalization(layers=False)
    assert v.status_of(["g"])["g"]["status"] == "established"


def test_verdict_without_evidence_is_inconclusive():
    store, adapter, v, clock = setup()
    observe(adapter, store, clock, others()[0], "a1")
    v.register("g", "x", "a1")
    d = v.store.get("vmem", "g")
    d.update(kind="generalization", code="def test(df):\n    return {'in_scope': [], 'support': [], 'contradict': [], 'verdict': 'supported'}")
    v.store.put("vmem", "g", d)
    v._evaluate()
    assert v.status_of(["g"])["g"]["eval"]["verdict"] == "inconclusive"


def test_failing_test_is_repaired_once_then_unverifiable():
    store, adapter, v, clock = setup()
    observe(adapter, store, clock, others()[0], "a1")
    v.register("g", "- something odd", "a1")
    d = v.store.get("vmem", "g")
    d.update(kind="generalization", code="def test(df):\n    return 1 / 0")
    v.store.put("vmem", "g", d)
    v._evaluate()          # error -> rewrite request; the scripted rewrite says untestable
    assert v.status_of(["g"])["g"]["status"] == "unverifiable"


def test_text_mode_judges_wording():
    store, adapter, v, clock = setup(mode="text")
    v.register("g", "- Option A1 increases the outcome compared with the other options of slot A.", "a1")
    while v.step():
        pass
    assert v.status_of(["g"])["g"]["status"] == "accepted"


# --------------------------------------------------------------------------- briefing with Mem0
def test_briefing_sections_and_notice():
    store = InMemoryStore()
    clock = Clock()
    adapter = SyntheticAdapter(ENV)
    llm = LLM(provider="scripted", scripted=ScriptedLLM(ENV))
    backend = LocalMem0(build_mem0(tempfile.mkdtemp(), "t", scripted=mem0_scripted(int(os.environ.get("EMBED_DIMS", "768")))),
                        run_id="campaign")
    mem = make_memory("mem0_verified", store, adapter, MemoryConfig(briefing_budget_chars=3000), llm=llm,
                      backend=backend)
    mem.verifier.clock = clock
    mem.clock = clock
    for e in a0_designs()[:2] + others()[:3]:
        rec = ENV.run(e, "a1")
        rec["ended_at"] = clock()
        mem.ingest(rec)
    mem.submit_conclusion("a1", "- Option A0 increases the outcome compared with the other options of slot A.", 0)
    mem.submit_conclusion("a1", "- Option A0 decreases the outcome compared with the other options of slot A.", 0)
    while mem.maintain():
        pass
    text = mem.briefing("a2", "plan", "option A0 slot A", 1)
    items = store.read_log("briefings")[-1]["items"]
    assert "TO TEST" in text and "KNOWN FALSE" in text, text
    assert {i["asserted"] for i in items} == {None, False}
    assert len(mem.briefing("a2", "plan", "option", 1)) <= 3000


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
