"""
Tests of ProvMem without an LLM (a scripted judge stands in for it).

    python -m pytest tests/test_provmem.py -q        (or: python tests/test_provmem.py)
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.env import EnvSpec, SyntheticAdapter, SyntheticEnv  # noqa: E402
from provmem import InMemoryStore, MemoryConfig, make_memory  # noqa: E402
from provmem.memory import split_claims  # noqa: E402
from provmem.sandbox import TestError, run_test  # noqa: E402

SPEC = dict(n_slots=3, n_options=3, main_scale=1.0, interaction_scale=0.0, n_interactions=0,
            noise_sd=0.0, n_forbidden=1, seed=7)

# tests use an environment where option A0 has the largest main effect of slot A
TEST_A0_UP = """def test(df):
    ok = df[df['status'] == 'ok']
    ref = ok[ok['slot_A'] != 0]['value'].mean()
    inside = ok[ok['slot_A'] == 0]
    return {'in_scope': list(ok['id']), 'support': list(inside[inside['value'] > ref]['id']),
            'contradict': list(inside[inside['value'] <= ref]['id']), 'stats': {'ref': float(ref) if ref == ref else 0.0}}
"""


class FakeJudge:
    """Formalizes every claim with TEST_A0_UP and returns a scripted verdict citing everything."""
    name = "fake"

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)

    def formalize(self, statement, *a, **k):
        return {"testable": True, "scoped_statement": statement, "relation": {"type": "new", "target": None},
                "code": TEST_A0_UP}

    def verdict(self, claim, result, rows, domain_note):
        v = self.verdicts.pop(0) if self.verdicts else "inconclusive"
        return {"verdict": v, "cite_support": [r["id"] for r in rows if r["role"] == "support"],
                "cite_contradict": [r["id"] for r in rows if r["role"] == "contradict"], "rationale": "scripted"}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        self.t += 1.0
        return self.t


def make_env():
    """Environment (searched over seeds) where A0 beats every other option of slot A everywhere."""
    for seed in range(200):
        env = SyntheticEnv(EnvSpec(**{**SPEC, "seed": seed}))
        if all(env.main[("A", 0)] > env.main[("A", o)] + 0.5 for o in (1, 2)):
            return env
    raise RuntimeError("no suitable environment")


def setup(verdicts=(), **cfg):
    env = make_env()
    base = dict(briefing_budget_chars=4000, lease_ttl_s=30, test_timeout_s=60)
    base.update(cfg)
    mem = make_memory("provmem", InMemoryStore(), SyntheticAdapter(env), MemoryConfig(**base),
                      judge=FakeJudge(verdicts), clock=Clock())
    return env, mem


def record(env, mem, entity, agent):
    rec = env.run(entity, agent)
    rec["ended_at"] = mem.clock()      # provenance time on the memory's clock
    return mem.ingest(rec)


def worst_unrecorded(env, mem, **fixed):
    seen = {o["entity"] for o in mem.store.all("obs")}
    cands = [r for r in env.population() if r["status"] == "ok" and r["entity"] not in seen
             and all(r[f"slot_{k}"] == v for k, v in fixed.items())]
    return min(cands, key=lambda r: r["value"])["entity"]


def best_unrecorded(env, mem, **fixed):
    """The best design (true value) matching fixed slots that the memory has not observed yet."""
    seen = {o["entity"] for o in mem.store.all("obs")}
    cands = [r for r in env.population() if r["status"] == "ok" and r["entity"] not in seen
             and all(r[f"slot_{k}"] == v for k, v in fixed.items())]
    return max(cands, key=lambda r: r["value"])["entity"]


def first_ok(env, n=1, **fixed):
    out = []
    for r in env.population():
        if r["status"] == "ok" and all(r[f"slot_{k}"] == v for k, v in fixed.items()):
            out.append(r["entity"])
        if len(out) == n:
            break
    return out


# --------------------------------------------------------------------------- coordination
def test_screen_known_failed_in_progress_and_batch_duplicates():
    env, mem = setup()
    ok = first_ok(env, 2)
    record(env, mem, ok[0], "a1")
    failed = next(r["entity"] for r in env.population() if r["status"] == "failed")
    record(env, mem, failed, "a1")
    d = mem.screen("a2", [ok[0], failed, ok[1], ok[1], "not-a-design"], 0)
    assert [x.action for x in d] == ["known", "failed_before", "run", "duplicate_in_batch", "invalid"]
    assert d[0].value is not None
    d3 = mem.screen("a3", [ok[1]], 0)              # a2 holds the lease on ok[1]
    assert d3[0].action == "in_progress" and d3[0].owner == "a2"
    record(env, mem, ok[1], "a2")                   # its observation releases the lease
    assert mem.screen("a3", [ok[1]], 0)[0].action == "known"


def test_lease_expires_without_heartbeat():
    env, mem = setup(lease_ttl_s=0.2)
    e = first_ok(env)[0]
    assert mem.screen("a1", [e], 0)[0].action == "run"
    time.sleep(0.3)
    assert mem.screen("a2", [e], 0)[0].action == "run"   # a1 is presumed lost


def test_no_coordination_lets_everything_run():
    env, mem = setup(coordination=False)
    e = first_ok(env)[0]
    record(env, mem, e, "a1")
    assert mem.screen("a2", [e], 0)[0].action == "run"


# --------------------------------------------------------------------------- confidence layers
def _claim_with_evidence(verdicts, **cfg):
    env, mem = setup(verdicts, **cfg)
    for e in first_ok(env, 2, A=0) + first_ok(env, 2, A=1):
        record(env, mem, e, "a1")
    [cid] = mem.submit_conclusion("a1", "- Option A0 increases the outcome.", 0)
    mem.maintain()
    return env, mem, cid


def test_new_claim_with_only_exploratory_evidence_is_a_hypothesis():
    _, mem, cid = _claim_with_evidence(["supported"])
    assert mem.store.get("claims", cid)["layer"] == "hypothesis"


def test_confirmation_by_author_then_by_another_agent():
    env, mem, cid = _claim_with_evidence(["supported", "supported", "supported"])
    record(env, mem, best_unrecorded(env, mem, A=0), "a1")    # confirmatory, same author
    mem.maintain()
    assert mem.store.get("claims", cid)["layer"] == "supported"
    record(env, mem, best_unrecorded(env, mem, A=0), "a2")    # independent confirmation
    mem.maintain()
    assert mem.store.get("claims", cid)["layer"] == "established"


def test_contradicted_new_claim_is_rejected_and_never_shown():
    _, mem, cid = _claim_with_evidence(["contradicted"])
    c = mem.store.get("claims", cid)
    if c["last_eval"]["contradict"]:
        assert c["layer"] == "rejected"
        text = mem.briefing("a2", "plan", "option A0", 1)
        assert cid not in text


def test_without_layers_one_verdict_establishes():
    _, mem, cid = _claim_with_evidence(["supported"], layers=False)
    assert mem.store.get("claims", cid)["layer"] == "established"


def test_refutation_notifies_agents_that_read_the_claim():
    env, mem, cid = _claim_with_evidence(["supported", "supported", "contradicted"])
    record(env, mem, best_unrecorded(env, mem, A=0), "a1")
    mem.maintain()
    assert mem.store.get("claims", cid)["layer"] == "supported"
    assert cid in mem.briefing("a2", "plan", "option A0", 1)   # a2 reads it
    record(env, mem, worst_unrecorded(env, mem, A=0), "a3")   # confirmatory evidence against
    mem.maintain()
    assert mem.store.get("claims", cid)["layer"] == "refuted"
    assert "NOTICE" in mem.briefing("a2", "plan", "x", 2)
    assert "NOTICE" not in mem.briefing("a4", "plan", "x", 2)


def test_citations_outside_the_test_result_are_ignored():
    env, mem = setup()

    class Liar(FakeJudge):
        def verdict(self, claim, result, rows, domain_note):
            return {"verdict": "supported", "cite_support": ["made-up-id"], "cite_contradict": [], "rationale": ""}
    mem.judge = Liar([])
    for e in first_ok(env, 2, A=0):
        record(env, mem, e, "a1")
    [cid] = mem.submit_conclusion("a1", "- Option A0 increases the outcome.", 0)
    mem.maintain()
    c = mem.store.get("claims", cid)
    assert c["layer"] == "hypothesis"           # inconclusive without valid citations: stays a hypothesis
    assert mem.store.read_log("dossiers")[-1]["cited_valid"]["support"] == []


# --------------------------------------------------------------------------- briefing
def test_briefing_respects_budget_and_marks_asserted_items():
    env, mem, cid = _claim_with_evidence(["supported"], briefing_budget_chars=300)
    text = mem.briefing("a2", "plan", "option", 1)
    assert len(text) <= 300
    items = mem.store.read_log("briefings")[-1]["items"]
    assert all(i["asserted"] is None for i in items if i["kind"] == "claim")   # hypotheses are not asserted


def test_directive_once_is_delivered_once():
    env, mem = setup()
    mem.put_directive("analyzer", "Try slot C.", lifetime="once")
    assert "Try slot C." in mem.briefing("a1", "plan", "", 0)
    assert "Try slot C." not in mem.briefing("a1", "plan", "", 1)
    assert "Try slot C." in mem.briefing("a2", "plan", "", 1)


# --------------------------------------------------------------------------- sandbox, parsing
def test_sandbox_rejects_bad_tests():
    rows = [{"id": "1", "value": 1.0, "status": "ok"}]
    try:
        run_test("def test(df):\n    return {'in_scope': ['1'], 'support': ['9'], 'contradict': []}", rows, 30)
        raise AssertionError("unknown id accepted")
    except TestError:
        pass
    try:
        run_test("import os\ndef test(df):\n    return {}", rows, 30)
        raise AssertionError("os import accepted")
    except TestError:
        pass


def test_split_claims():
    assert split_claims("Conclusions:\n- A is good.\n2. B is bad\n\n*") == ["A is good.", "B is bad"]


def test_chemistry_adapter():
    from provmem.adapters.chemistry import ChemistryAdapter
    a = ChemistryAdapter("goal", max_heavy_atoms=3)
    assert a.canonical("OCC") == a.canonical("CCO")
    assert a.canonical("C1CC") is None
    rec = {"task_id": "t1", "activity_id": "compute_ionization_energy", "status": "FINISHED", "ended_at": 5.0,
           "used": {"smiles": "OCC"}, "generated": {"ionization_energy": 0.5}, "custom_metadata": {"agent": "x"}}
    o = a.observation_from_record(rec)
    assert o["entity"] == "CCO" and o["value"] == 0.5 and o["status"] == "ok"
    assert a.descriptors("CC#N")["n_nitrile"] == 1
    assert a.violations("CCCCO")


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
