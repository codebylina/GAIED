"""
Tests of the evidence-grounded layered memory, without an LLM (scripted distiller and verifier).

    python tests/test_evidence.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.env import EnvSpec, SyntheticAdapter, SyntheticEnv  # noqa: E402
from bench.scripted import ScriptedLLM  # noqa: E402
from memlab import LLM, InMemoryStore, MemoryConfig, make_memory  # noqa: E402

CLAIM = "Option A0 increases the outcome compared with the other options of slot A."
FALSE = "Option A0 decreases the outcome compared with the other options of slot A."


def make_env():
    for seed in range(300):
        env = SyntheticEnv(EnvSpec(n_slots=3, n_options=3, main_scale=1.0, interaction_scale=0.0, n_interactions=0,
                                   noise_sd=0.0, n_forbidden=0, seed=seed))
        if all(env.main[("A", 0)] > env.main[("A", o)] + 0.5 for o in (1, 2)):
            return env
    raise RuntimeError("no suitable environment")


ENV = make_env()


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        self.t += 1.0
        return self.t


def memory(**ev):
    store, clock = InMemoryStore(), Clock()
    llm = LLM(provider="scripted", scripted=ScriptedLLM(ENV))
    mem = make_memory("evidence", store, SyntheticAdapter(ENV), MemoryConfig(briefing_budget_chars=4000), llm=llm,
                      evidence=ev or None)
    mem.clock = mem.registry.clock = clock
    if mem.distiller:
        mem.distiller.clock = clock
    if mem.verifier:
        mem.verifier.clock = clock
    return mem, store, clock


def run(mem, clock, entity, agent):
    rec = ENV.run(entity, agent)
    rec["started_at"], rec["ended_at"] = clock(), clock()
    return mem.ingest(rec)


def a0(best=True):
    rows = [r for r in ENV.population() if r["status"] == "ok" and r["slot_A"] == 0]
    return [r["entity"] for r in sorted(rows, key=lambda r: -r["value"] if best else r["value"])]


def others():
    return [r["entity"] for r in ENV.population() if r["status"] == "ok" and r["slot_A"] != 0]


def settle(mem):
    while mem.maintain():
        pass


# --------------------------------------------------------------------------- L0: registry
def test_registry_known_failed_in_progress():
    mem, store, clock = memory(knowledge=False)
    e = others()[0]
    d = mem.screen("a1", [e, e], 0)
    assert [x.action for x in d] == ["run", "duplicate_in_batch"]
    assert mem.screen("a2", [e], 0)[0].action == "in_progress"           # a1 holds the reservation
    run(mem, clock, e, "a1")
    d = mem.screen("a2", [e], 0)[0]
    assert d.action == "known" and d.owner == "a1" and d.value is not None
    rec = ENV.run("not-a-design", "a1")
    mem.ingest(rec)
    assert mem.screen("a3", ["not-a-design"], 0)[0].action == "failed_before"
    assert mem.registry.view(mem.registry.key("not-a-design"))["state"] == "invalid"


def test_briefing_shows_exact_team_results():
    mem, store, clock = memory(knowledge=False)
    run(mem, clock, others()[0], "a1")
    mem.ingest(ENV.run("not-a-design", "a1"))
    text = mem.briefing("a2", "propose", "anything", 0)
    assert "TEAM EXPERIMENTS" in text and others()[0] in text and "invalid, do not propose again" in text
    assert "ESTABLISHED" not in text


# --------------------------------------------------------------------------- L0 -> L1
def test_conclusions_become_atoms_with_source_and_restatements():
    mem, store, clock = memory()
    mem.submit_conclusion("a1", "- " + CLAIM, 0)
    mem.record_episode("a1", "plan", "- " + CLAIM, 0)                     # plans are not journaled
    settle(mem)
    mem.submit_conclusion("a2", "- " + CLAIM, 1)                          # same claim: a restatement
    settle(mem)
    atoms = store.all("atoms")
    assert len(atoms) == 1 and atoms[0]["author"] == "a1" and len(atoms[0]["sources"]) == 1
    assert [r["agent"] for r in atoms[0]["restatements"]] == ["a2"]
    assert store.get("vmem", atoms[0]["id"])["claimed_at"] == atoms[0]["claimed_at"]


# --------------------------------------------------------------------------- L1 -> L2 and back
def test_promotion_by_evidence_and_demotion_notice():
    mem, store, clock = memory()
    for e in a0()[:2] + others()[:3]:
        run(mem, clock, e, "a1")
    mem.submit_conclusion("a1", "- " + CLAIM, 0)
    settle(mem)
    aid = store.all("atoms")[0]["id"]
    assert store.get("vmem", aid)["status"] == "hypothesis"               # only the data that suggested it
    text = mem.briefing("a2", "plan", "option A0 slot A", 1)
    assert "TO TEST" in text and "ESTABLISHED" not in text
    run(mem, clock, a0()[2], "a3")                                        # a3 never saw the claim
    settle(mem)
    assert store.get("vmem", aid)["status"] == "established"
    text = mem.briefing("a2", "plan", "option A0 slot A", 1)
    assert "ESTABLISHED KNOWLEDGE" in text and "who had not seen it" in text
    # the rule now fails to find support: the atom leaves L2 and a2, who was shown it as true, is told
    from memlab.verifier import sha
    d = store.get("vmem", aid)
    code = "def test(df):\n    return {'in_scope': list(df['id']), 'support': [], 'contradict': [], 'verdict': 'insufficient'}"
    d.update(code=code, rule={**d["rule"], "sha256": sha(code)})
    store.put("vmem", aid, d)
    run(mem, clock, others()[3], "a1")
    settle(mem)
    assert store.get("vmem", aid)["status"] == "contested"
    text = mem.briefing("a2", "plan", "option A0 slot A", 2)
    assert "NOTICE:" in text and "ESTABLISHED KNOWLEDGE" not in text


def test_false_claim_is_shown_as_false():
    mem, store, clock = memory()
    for e in a0()[:2] + others()[:3]:
        run(mem, clock, e, "a1")
    mem.submit_conclusion("a1", "- " + FALSE, 0)
    settle(mem)
    text = mem.briefing("a2", "plan", "option A0 slot A", 1)
    items = store.read_log("briefings")[-1]["items"]
    assert "KNOWN FALSE" in text and {i["asserted"] for i in items if i["kind"] == "claim"} == {False}


def test_unverified_variant_asserts_everything():
    mem, store, clock = memory(verify=False)
    mem.submit_conclusion("a1", "- " + FALSE, 0)
    settle(mem)
    text = mem.briefing("a2", "plan", "option A0", 1)
    items = store.read_log("briefings")[-1]["items"]
    assert "TEAM KNOWLEDGE" in text and [i["asserted"] for i in items if i["kind"] == "claim"] == [True]
    assert mem.presented_knowledge()[0]["final_layer"] == "stored"


def test_directive_reaches_its_target_only():
    mem, store, clock = memory()
    mem.put_directive("analysis-agent", "Stop proposing designs with A2.", targets=["a1"])
    assert "Stop proposing designs with A2." in mem.briefing("a1", "plan", "x", 0)
    assert "Stop proposing" not in mem.briefing("a2", "plan", "x", 0)


def test_budget_is_respected():
    mem, store, clock = memory()
    for e in others()[:8] + a0()[:3]:
        run(mem, clock, e, "a1")
    mem.cfg.briefing_budget_chars = 600
    assert len(mem.briefing("a2", "plan", "option", 0)) <= 600


def test_replay_judges_an_unverified_campaign():
    import json
    import tempfile
    from bench.analyze_gaied import knowledge_metrics
    from bench.replay_verify import replay
    mem, store, clock = memory(verify=False)
    for e in a0()[:2] + others()[:3]:
        run(mem, clock, e, "a1")
    mem.submit_conclusion("a1", "- " + FALSE, 0)
    settle(mem)
    mem.briefing("a2", "plan", "option A0 slot A", 1)
    path = tempfile.mktemp(suffix=".json")
    json.dump(mem.export(), open(path, "w"), default=str)
    replay(path, llm=LLM(provider="scripted", scripted=ScriptedLLM(ENV)), adapter=SyntheticAdapter(ENV))
    m = knowledge_metrics(json.load(open(path)))
    assert m["status_source_replay"] == 1 and m["impressions_shown_true"] == 1 and m["impressions_shown_true_bad"] == 1, m


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                fails += 1
                print("FAIL", name, repr(exc))
    sys.exit(1 if fails else 0)
