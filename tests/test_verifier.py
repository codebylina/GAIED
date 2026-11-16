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
    for e in a0_designs()[:4] + others()[:4]:
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
    observe(adapter, store, clock, a0_designs()[4], "a1")          # after the memory, same author
    while v.step():
        pass
    assert v.status_of(["g"])["g"]["status"] == "supported"
    for e in a0_designs()[5:7]:                                    # after the memory, another agent
        observe(adapter, store, clock, e, "a2")
    while v.step():
        pass
    d = v.status_of(["g"])["g"]
    assert d["status"] == "established" and d["eval"]["independent_agents"] == ["a2"]


def test_contradiction_refutes():
    store, adapter, v, clock = setup()
    for e in a0_designs()[:4] + others()[:4]:
        observe(adapter, store, clock, e, "a1")
    v.register("g", "- Option A0 decreases the outcome compared with the other options of slot A.", "a1")
    while v.step():
        pass
    assert v.status_of(["g"])["g"]["status"] == "refuted"


def test_without_layers_a_supported_verdict_establishes():
    _, _, v, _ = _generalization(layers=False)
    assert v.status_of(["g"])["g"]["status"] == "established"


def freeze(v, mem_id, code, kind="generalization"):
    """Give a registered memory a frozen rule, as the formalization step does."""
    from memlab.verifier import sha
    d = v.store.get("vmem", mem_id)
    d.update(kind=kind, code=code, rule={"version": 1, "sha256": sha(code)}, rules=[{"version": 1, "code": code}])
    v.store.put("vmem", mem_id, d)


def test_verdict_without_evidence_is_insufficient():
    store, adapter, v, clock = setup(decision="llm")
    observe(adapter, store, clock, others()[0], "a1")
    v.register("g", "x", "a1")
    freeze(v, "g", "def test(df):\n    return {'in_scope': [], 'support': [], 'contradict': [], 'verdict': 'supported'}")
    v._evaluate()
    d = v.status_of(["g"])["g"]
    assert d["eval"]["verdict"] == "insufficient" and d["status"] == "open" and d["evidence"] == "none", d


def test_failing_test_is_repaired_once_then_unverifiable():
    store, adapter, v, clock = setup()
    observe(adapter, store, clock, others()[0], "a1")
    v.register("g", "- something odd", "a1")
    freeze(v, "g", "def test(df):\n    raise ValueError('value 2.71828 too low')")
    v._evaluate()          # error -> rewrite request; the scripted rewrite says untestable
    d = v.status_of(["g"])["g"]
    assert d["status"] == "unverifiable"
    repair = [j for j in store.read_log("v_journal") if j["step"] == "repair"]
    assert repair and d["scoped"] == "- something odd"           # a repair keeps the statement


def test_repair_error_does_not_leak_values():
    from memlab.verifier import sanitize
    assert "2.71" not in sanitize("ValueError: value 2.71828 too low") and "<number>" in sanitize("x 1.5e-3")


def test_tampered_rule_is_not_run():
    store, adapter, v, clock = _generalization()
    d = v.store.get("vmem", "g")
    d["code"] = d["code"].replace("'higher'", "'lower'", 1)       # changed after it was frozen
    v.store.put("vmem", "g", d)
    observe(adapter, store, clock, a0_designs()[4], "a2")
    v._evaluate()
    assert v.status_of(["g"])["g"]["status"] == "unverifiable"


def test_design_keys_are_mapped_and_in_scope_completed():
    from memlab.sandbox import validate
    out = validate({"in_scope": ["A0-B1"], "support": ["m3"], "contradict": [], "verdict": "supported"},
                   {"m1", "m2", "m3"}, {"A0-B1": ["m1", "m2"]})
    assert out["in_scope"] == ["m1", "m2", "m3"]


def test_memory_id_or_hard_coded_ids_send_the_test_to_repair():
    from memlab.verifier import static_problem
    assert static_problem("def test(df):\n    return {'in_scope': ['a-12'], 'verdict': 'supported'}", "a-12")
    assert static_problem("x = df[df['id'] == '0123456789abcdef0123456789abcdef']", "g")
    assert static_problem("x = df[df['entity'] == 'A0-B1-C2']", "g") is None
    store, adapter, v, clock = setup()
    observe(adapter, store, clock, others()[0], "a1")
    v.register("g", "- something odd", "a1")
    freeze(v, "g", "def test(df):\n    return {'in_scope': ['g'], 'support': ['g'], 'contradict': [], 'verdict': 'supported'}")
    v._evaluate()
    assert any(j["step"] == "repair" for j in store.read_log("v_journal"))


# --------------------------------------------------------------------------- pre-registration
def test_rule_is_written_from_the_claim_time_window_only():
    store, adapter, v, clock = setup()
    for e in a0_designs()[:4] + others()[:4]:
        observe(adapter, store, clock, e, "a1")
    t_claim = clock()
    for e in a0_designs()[4:6]:                                   # measured after the claim, before formalization
        observe(adapter, store, clock, e, "a2")
    v.register("g", "- Option A0 increases the outcome compared with the other options of slot A.", "a1",
               written_at=t_claim)
    while v.step():
        pass
    d = v.status_of(["g"])["g"]
    assert d["rule"]["window_until"] == t_claim and d["rule"]["max_t_seen"] <= t_claim, d["rule"]
    assert "C" not in d["eval"]["classes"]["support"] and "I" in d["eval"]["classes"]["support"]


def test_posthoc_ablation_rewrites_the_rule_after_new_data():
    store, adapter, v, clock = _generalization(preregistered=False)
    observe(adapter, store, clock, a0_designs()[4], "a2")
    while v.step():
        pass
    d = v.status_of(["g"])["g"]
    assert len(d["rules"]) >= 2 and d["rules"][-1]["reason"] == "posthoc" and d["rule"]["window_until"] is None


# --------------------------------------------------------------------------- independence
def _show(store, clock, agent, mem_id):
    store.append("briefings", {"agent": agent, "node": "propose", "iteration": 0, "t": clock(), "chars": 0,
                               "items": [{"id": mem_id, "kind": "claim", "asserted": None, "text": ""}]})


def test_informed_replication_is_not_independent():
    store, adapter, v, clock = _generalization()
    _show(store, clock, "a2", "g")                               # a2 reads the claim, then tests it
    for e in a0_designs()[4:7]:
        observe(adapter, store, clock, e, "a2")
    while v.step():
        pass
    d = v.status_of(["g"])["g"]
    assert d["status"] == "supported" and "R" in d["eval"]["classes"]["support"] and "I" not in d["eval"]["classes"]["support"], d["eval"]
    observe(adapter, store, clock, a0_designs()[7], "a3")       # a3 never saw it
    while v.step():
        pass
    d = v.status_of(["g"])["g"]
    assert d["status"] == "established" and d["eval"]["independent_agents"] == ["a3"]


def test_agent_independence_ablation_ignores_exposure():
    store, adapter, v, clock = _generalization(independence="agent")
    _show(store, clock, "a2", "g")
    for e in a0_designs()[4:7]:
        observe(adapter, store, clock, e, "a2")
    while v.step():
        pass
    assert v.status_of(["g"])["g"]["status"] == "established"


def test_remeasurement_of_an_old_design_is_exploratory():
    store, adapter, v, clock = _generalization()
    observe(adapter, store, clock, a0_designs()[0], "a2")       # same design as before the claim
    while v.step():
        pass
    d = v.status_of(["g"])["g"]
    assert d["status"] == "hypothesis" and set(d["eval"]["classes"]["support"]) == {"E"}


# --------------------------------------------------------------------------- statuses
def test_prediction_without_measurement_is_open_report_is_unsupported():
    store, adapter, v, clock = setup()
    observe(adapter, store, clock, others()[0], "a1")
    e = a0_designs()[0]
    code = ("def test(df):\n    r = df[df['entity'] == '" + e + "']\n"
            "    return {'facts': [{'id': i, 'status': 'ok', 'value': None} for i in ids(r)]}")
    for mid, kind in (("p", "prediction"), ("r", "report")):
        v.register(mid, "x", "a1")
        freeze(v, mid, code, kind)
    v._evaluate()
    st = {i: d["status"] for i, d in v.status_of(["p", "r"]).items()}
    assert st == {"p": "open", "r": "unsupported"}, st
    observe(adapter, store, clock, e, "a2")
    v._evaluate()
    assert v.status_of(["p"])["p"]["status"] == "verified"


def test_asserted_then_insufficient_is_contested():
    store, adapter, v, clock = _generalization()
    observe(adapter, store, clock, a0_designs()[4], "a1")
    while v.step():
        pass
    assert v.status_of(["g"])["g"]["status"] == "supported"
    n = store.log_len("obs_log")
    freeze(v, "g", "def test(df):\n    return {'groups': {'higher': ids(df)[:1], 'lower': ids(df)[1:2]}}")
    observe(adapter, store, clock, others()[5], "a1")
    v._evaluate()
    d = v.status_of(["g"])["g"]
    assert d["status"] == "contested" and d["evidence"] == "insufficient" and n < store.log_len("obs_log")
    assert store.read_log("v_demotions")[-1]["to"] == "contested"


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
    for e in a0_designs()[:4] + others()[:4]:
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


# --------------------------------------------------------------------------- decision by code
def test_code_decision_without_evidence_is_open():
    store, adapter, v, clock = setup()
    observe(adapter, store, clock, others()[0], "a1")
    v.register("g", "x", "a1")
    freeze(v, "g", "def test(df):\n    return {'groups': {'higher': [], 'lower': ids(df)}}")
    v._evaluate()
    d = v.status_of(["g"])["g"]
    assert d["eval"]["verdict"] == "insufficient" and d["status"] == "open" and d["evidence"] == "none", d


def test_one_comparison_never_establishes():
    """The lax case seen with Qwen: a single matched pair, then an independent agent's single pair."""
    from memlab.evidence_test import binom_tail
    assert binom_tail(4, 4) > 0.05 and binom_tail(5, 5) < 0.05       # 5 agreeing disjoint pairs are needed
    store, adapter, v, clock = setup()
    a0, a1 = a0_designs()[0], a0_designs()[0].replace("A0", "A1")
    observe(adapter, store, clock, a0, "a1")
    observe(adapter, store, clock, a1, "a1")
    v.register("g", "x", "a1")
    freeze(v, "g", "def test(df):\n    ok = df[df['status'] == 'ok']\n"
                   "    return {'pairs': pairs(ok[ok['slot_A'] == 0], ok[ok['slot_A'] == 1], vary=['slot_A'])}")
    v._evaluate()
    assert v.status_of(["g"])["g"]["status"] == "open"
    b = a0.replace("C0", "C1") if "C0" in a0 else a0.replace("C1", "C0").replace("C2", "C0")
    observe(adapter, store, clock, b, "a2")
    observe(adapter, store, clock, b.replace("A0", "A1"), "a2")
    v._evaluate()
    d = v.status_of(["g"])["g"]
    assert d["status"] != "established" and d["eval"]["stats"]["n"] == 2, d["eval"]


def test_pairs_are_disjoint_and_ids_from_the_index_are_valid():
    from memlab.evidence_test import parse, stat
    by_id = {"a": {"status": "ok", "value": 3.0}, "b": {"status": "ok", "value": 1.0},
             "c": {"status": "ok", "value": 2.0}, "d": {"status": "failed", "value": None}}
    comp = parse({"pairs": [["a", "b"], ["a", "c"], ["c", "b"], ["c", "d"]]}, set(by_id))
    s = stat(comp, by_id)
    assert s["n"] == 1 and s["k_for"] == 1             # a-b used; a-c and c-b reuse a row; c-d has a failure
    store, adapter, v, clock = setup()
    for e in others()[:3]:
        observe(adapter, store, clock, e, "a1")
    v.register("g", "x", "a1")
    freeze(v, "g", "def test(df):\n    ok = df[df['status'] == 'ok']\n"
                   "    return {'groups': {'higher': list(ok.index[:1]), 'lower': [i for i, r in ok.iterrows()][1:]}}")
    v._evaluate()
    assert v.status_of(["g"])["g"]["eval"]["in_scope"] == 3


def test_values_in_place_of_ids_are_refused_with_a_hint():
    from memlab.evidence_test import TestError, parse
    from memlab.verifier import repair_hint
    try:
        parse({"groups": {"higher": [1.25], "lower": []}}, {"m1"})
        raise AssertionError("accepted a value")
    except TestError as exc:
        assert "number" in str(exc) and "ids(frame)" in repair_hint(str(exc))
    assert "iterrows" in repair_hint("TypeError: string indices must be integers, not 'str'")


def test_fact_value_at_the_precision_written():
    from memlab.evidence_test import parse, stat
    by_id = {"m": {"status": "ok", "value": 2.6728152}}
    ok = lambda v: stat(parse({"facts": [{"id": "m", "status": "ok", "value": v}]}, {"m"}), by_id)["agree"] == ["m"]
    assert ok("2.673") and ok("2.672") and ok("2.67") and not ok("2.68") and not ok("2.6750")
    assert not stat(parse({"facts": [{"id": "m", "status": "failed"}]}, {"m"}), by_id)["agree"]


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
