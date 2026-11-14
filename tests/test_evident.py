"""
Unit tests of the EVIDENT memory, WITHOUT LLM or xTB (a few seconds).

In Docker (real Redis), from GAIED/docker:
    docker compose run --rm gaied python tests/test_evident.py
"""
import asyncio
import json
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import redis  # noqa: E402

from evident import EvidentMemory, SimulationFailed, StoppedByDirective  # noqa: E402
from evident.claims import extract_claims  # noqa: E402
from evident.features import canonical_smiles, constraint_violations, features_of  # noqa: E402
from evident.monitor import RuleMonitor  # noqa: E402
from evident.verifier import VerifierParams, evaluate, next_status  # noqa: E402

CLIENT = redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"), decode_responses=True)
CLIENT.ping()
CAMPAIGNS = []
GOAL = {"objective": "maximize ionization energy",
        "constraints": {"max_heavy_atoms": 8, "allowed_elements": ["C", "H", "N", "O", "F"]}}


def campaign():
    c = f"test-{uuid.uuid4().hex[:8]}"
    CAMPAIGNS.append(c)
    return c


def mem(c, agent, mode="evident", notify="push", ttl=30):
    return EvidentMemory(CLIENT, c, agent, mode=mode, claim_ttl=ttl, poll=0.05, control_poll=0.05,
                         notify=notify)


# ---------------------------------------------------------------- chemistry vocabulary
def test_features_and_constraints():
    assert canonical_smiles("OCC") == canonical_smiles("CCO")
    assert "fluorine" in features_of("FC(F)(F)C(N)=O") and "trifluoromethyl" in features_of("FC(F)(F)C(N)=O")
    assert "urea" in features_of("CNC(N)=O") and "amide" in features_of("CNC(N)=O")
    assert "nitrile" in features_of("N#CC#N") and "aromatic_ring" in features_of("c1ccccc1")
    assert constraint_violations("ClCCl", GOAL["constraints"]) == ["forbidden_elements(Cl)"]
    assert constraint_violations("CCCCCCCCCC", GOAL["constraints"])[0].startswith("too_large")
    assert constraint_violations("C1CC((", GOAL["constraints"]) == ["invalid_smiles"]


def test_claim_extraction():
    text = ('Fluorinated ureas are best.\nCLAIMS: [{"feature": "fluorine", "effect": "increases"},'
            ' {"feature": "methyl", "effect": "decreases"}, {"feature": "unknown", "effect": "up"}]')
    assert [(c["feature"], c["direction"]) for c in extract_claims(text)] == [
        ("fluorine", "up"), ("methyl", "down")]
    loose = "Adding a nitrile group increases the ionization energy. Aromatic rings lower it."
    assert {(c["feature"], c["direction"]) for c in extract_claims(loose)} == {
        ("nitrile", "up"), ("aromatic_ring", "down")}
    assert extract_claims("Fluorine and nitrile both increase it.") == [], "ambiguous sentence"


# ---------------------------------------------------------------- verifier
def _facts(values_with, values_without, feature="fluorine", agents=("A", "B")):
    out = []
    for i, e in enumerate(values_with):
        out.append({"smiles": f"w{i}", "energy": e, "status": "ok", "features": [feature],
                    "agent": agents[i % len(agents)]})
    for i, e in enumerate(values_without):
        out.append({"smiles": f"o{i}", "energy": e, "status": "ok", "features": [],
                    "agent": agents[i % len(agents)]})
    return out


def test_verifier_lifecycle():
    p = VerifierParams()
    few = evaluate("fluorine", "up", _facts([12, 13], [10, 10.5]), p)
    assert few["verdict"] == "insufficient" and next_status(None, few, p) == "proposed"
    strong = evaluate("fluorine", "up", _facts([12, 12.4, 12.2, 12.6, 12.1, 12.5],
                                                [10, 10.3, 10.1, 10.4, 10.2, 10.6]), p)
    assert strong["verdict"] == "supports" and strong["replications"] == 2
    assert next_status("proposed", strong, p) == "established"
    weak = evaluate("fluorine", "up", _facts([12, 10, 11], [11, 12, 10]), p)
    assert weak["verdict"] == "inconclusive" and next_status("supported", weak, p) == "contested"
    wrong = evaluate("fluorine", "up", _facts([9, 9.2, 9.1], [11, 11.2, 11.1]), p)
    assert wrong["verdict"] == "contradicts" and next_status("established", wrong, p) == "rejected"
    single = evaluate("fluorine", "up", _facts([12, 12.4, 12.2, 12.6, 12.1, 12.5],
                                                [10, 10.3, 10.1, 10.4, 10.2, 10.6], agents=("A",)), p)
    assert next_status(None, single, p) == "supported", "one agent alone cannot establish a claim"


# ---------------------------------------------------------------- coordination
def test_no_duplicate_computation_and_shared_failure():
    c = campaign()
    calls = []

    def compute_for(smiles):
        async def compute(task_id):
            calls.append(smiles)
            await asyncio.sleep(0.2)
            if smiles == "C1CC((":
                raise ValueError("Parse failure")
            return 10 + len(smiles) / 10
        return compute

    async def agent(m, mols):
        out = []
        for r in await asyncio.gather(*(m.evaluate_molecule(s, compute_for(s)) for s in mols),
                                      return_exceptions=True):
            out.append(r)
        return out

    async def main():
        return await asyncio.gather(agent(mem(c, "A"), ["CCCl", "CCO", "C1CC(("]),
                                    agent(mem(c, "B"), ["OCC", "CCF", "C1CC(("]))

    ra, rb = asyncio.run(main())
    assert sorted(calls) == sorted(["CCCl", "CCO", "C1CC((", "CCF"]), calls
    assert ra[1] == rb[0]
    assert any(isinstance(x, SimulationFailed) for x in ra + rb), "the failure is shared"
    stats = CLIENT.hgetall(f"ev:{c}:stats")
    assert int(stats["computed"]) == 3 and int(stats["computed_error"]) == 1


def test_mode_none_records_duplicates():
    c = campaign()

    async def compute(task_id):
        return 11.0

    async def main():
        await asyncio.gather(mem(c, "A", "none").evaluate_molecule("CCO", compute),
                             mem(c, "B", "none").evaluate_molecule("OCC", compute))

    asyncio.run(main())
    assert CLIENT.xlen(f"ev:{c}:facts") == 2, "isolated agents both compute: duplicate visible"


def test_crashed_agent_lock_expires():
    c = campaign()
    assert mem(c, "ghost").claim("CCN", ttl=1)

    async def compute(task_id):
        return 12.0

    t0 = time.time()
    assert asyncio.run(mem(c, "A").evaluate_molecule("CCN", compute)) == 12.0
    assert time.time() - t0 >= 0.9


# ---------------------------------------------------------------- claims in memory
def _seed_facts(m):
    for i, (s, e) in enumerate([("FC(F)C(N)=O", 12.1), ("FCC(N)=O", 12.4), ("FC(F)(F)C#N", 12.8),
                                ("FCCO", 12.3), ("CC(N)=O", 10.2), ("CCO", 10.0),
                                ("CCN", 9.8), ("CC#N", 10.4)]):
        m.agent_id = "A" if i % 2 else "B"
        m.push_fact(s, e, task_id=f"t{i}")
    m.agent_id = "A"


def test_evident_claims_verified_and_briefing():
    c = campaign()
    a, b = mem(c, "A"), mem(c, "B")
    a.set_goal(GOAL)
    _seed_facts(a)
    a.record_conclusion('CLAIMS: [{"feature": "fluorine", "effect": "increases"}]')
    b.record_conclusion('CLAIMS: [{"feature": "fluorine", "effect": "decreases"}]')
    cl = a.claims()
    assert cl["fluorine:up"]["status"] in ("supported", "established"), cl["fluorine:up"]
    assert cl["fluorine:down"]["status"] == "rejected"
    text = b.briefing("plan")
    assert "VERIFIED KNOWLEDGE" in text and "fluorine atom increases" in text and "n=" in text
    assert "DO NOT RELY ON" in text and "fluorine atom decreases" in text
    assert "at most 8 heavy" in text
    log = CLIENT.xrange(f"ev:{c}:claims:log")
    assert len(log) >= 3, "creation and status changes are versioned"


def test_naive_mode_shares_raw_lessons_and_none_shares_nothing():
    c = campaign()
    a, b = mem(c, "A", "naive"), mem(c, "B", "naive")
    a.set_goal(GOAL)
    a.record_conclusion("Methyl groups increase the ionization energy a lot.")
    assert "LESSONS FROM OTHER AGENTS" in b.briefing("plan") and "Methyl" in b.briefing("plan")
    assert "LESSONS" not in a.briefing("plan"), "an agent does not read its own notes"
    c2 = campaign()
    n1, n2 = mem(c2, "A", "none"), mem(c2, "B", "none")
    n1.set_goal(GOAL)
    _seed_facts(n1)
    text = n2.briefing("plan")
    assert "GOAL" in text and "TEAM RESULTS" not in text and "LESSONS" not in text


def test_injected_false_claim_is_rejected_in_evident():
    c = campaign()
    liar = mem(c, "L")
    liar.is_liar = True
    liar.set_goal(GOAL)
    _seed_facts(liar)
    liar.record_conclusion("whatever the liar really thought")
    inj = [json.loads(f["d"]) for _, f in CLIENT.xrange(f"ev:{c}:telemetry") if f["e"] == "injection"]
    feat = inj[0]["claims"][0]["feature"]
    assert liar.claims()[f"{feat}:up"]["status"] == "rejected", liar.claims()
    assert "VERIFIED KNOWLEDGE" not in mem(c, "B").briefing("plan")


# ---------------------------------------------------------------- real-time control
def test_stop_directive_cancels_running_computation():
    c = campaign()
    a, ctrl = mem(c, "A"), mem(c, "monitor")
    started = {}

    async def slow(task_id):
        started["t"] = time.time()
        await asyncio.sleep(5)
        return 1.0

    async def main():
        task = asyncio.create_task(a.evaluate_molecule("CCCC", slow))
        await asyncio.sleep(0.3)
        ctrl.send_directive("A", "stop", "off goal", rule="test")
        t_sent = time.time()
        try:
            await task
        except StoppedByDirective:
            return time.time() - t_sent
        return None

    latency = asyncio.run(main())
    assert latency is not None and latency < 1.0, latency
    assert a.claimed_by("CCCC") is None, "lock released after cancellation"
    assert a.get_fact("CCCC") is None, "a cancelled computation is not a failure"
    print(f"    stop latency {latency * 1000:.0f} ms")


def test_monitor_stops_only_off_goal_molecules():
    """The off-goal STOP cancels the molecules that break the Goal, not the valid ones,
    and does not interrupt LLM calls."""
    c = campaign()
    a, mon_mem = mem(c, "A"), mem(c, "monitor")
    a.set_goal(GOAL)
    mon = RuleMonitor(mon_mem, violations_for_stop=2)
    a.record_proposals(["CCO", "ClCCl", "CCCCCCCCCCCC"], own_previous=[])
    mon.poll_once(block=False)
    bad = canonical_smiles("ClCCl")
    assert a.check_stop(bad) is not None and a.check_stop(bad)["rule"] == "off_goal"
    assert a.check_stop(canonical_smiles("CCO")) is None, "a valid molecule goes on"
    assert a.check_stop() is None, "LLM calls are not interrupted by a molecule STOP"
    text = a.briefing("tool_calling")
    assert text.startswith("DIRECTIVES FROM THE ANALYSIS AGENT") and "NOT computed" in text
    assert "at most 8 heavy" in text

    async def compute(task_id):
        await asyncio.sleep(0.2)
        return 10.0

    async def batch():
        return await asyncio.gather(a.evaluate_molecule("CCO", compute),
                                    a.evaluate_molecule("ClCCl", compute),
                                    return_exceptions=True)

    ok, stopped = asyncio.run(batch())
    assert ok == 10.0 and isinstance(stopped, StoppedByDirective)
    a.record_proposals(["CCN"], own_previous=[])     # the next batch: the stop is handled
    assert a.check_stop(bad) is None


def test_global_stop_interrupts_llm_call():
    """A STOP without scope (e.g. from a provenance analysis agent) stops everything."""
    c = campaign()
    a, ctrl = mem(c, "A"), mem(c, "analysis")

    async def slow_llm():
        await asyncio.sleep(5)
        return "answer"

    async def main():
        task = asyncio.create_task(a.watched(slow_llm()))
        await asyncio.sleep(0.2)
        ctrl.send_directive("A", "stop", "the agent deviates from the Goal", rule="drift")
        try:
            await task
        except StoppedByDirective:
            return True
        return False

    assert asyncio.run(main())


def test_export():
    c = campaign()
    a = mem(c, "A")
    a.set_goal(GOAL)
    a.set_meta(mode="evident")
    _seed_facts(a)
    data = a.export()
    assert data["mode"] == "evident" and len(data["facts_log"]) == 8 and data["goal"]


def test_adjusted_effect_separates_confounded_features():
    """Every urea has a carbonyl; only the carbonyl raises the value. The marginal
    comparison wrongly supports 'urea increases', the adjusted effect does not."""
    import random
    from evident.verifier import adjusted_effects
    rng = random.Random(1)
    facts = []
    for i in range(40):
        carbonyl = i % 2 == 0
        urea = carbonyl and i % 4 == 0
        fluor = i % 3 == 0
        feats = [f for f, on in (("carbonyl", carbonyl), ("urea", urea), ("fluorine", fluor)) if on]
        e = 10 + 1.0 * carbonyl + 0.6 * fluor + rng.gauss(0, 0.1)
        facts.append({"smiles": f"m{i}", "energy": e, "status": "ok", "features": feats,
                      "agent": "AB"[i % 2], "heavy_atoms": 6})
    p = VerifierParams()
    adj = adjusted_effects(facts)
    assert abs(adj["urea"]["coef"]) < 0.2 and adj["carbonyl"]["coef"] > 0.8, adj
    assert evaluate("urea", "up", facts, p, adj)["verdict"] != "supports"
    assert evaluate("carbonyl", "up", facts, p, adj)["verdict"] == "supports"


# ---------------------------------------------------------------- coordination v2: leases, push, topics
def test_reservation_at_proposal_is_visible_with_envelope():
    """A molecule is reserved when it is PROPOSED; the team sees it before any computation,
    and the topic message carries the activity and the agent."""
    c = campaign()
    a, b = mem(c, "A"), mem(c, "B")
    a.set_goal(GOAL)
    a.begin_node("tool_calling")
    recs = a.record_proposals(["CCOC", "CCCO"], own_previous=[])
    assert [r["reservation"] for r in recs] == ["mine", "mine"]
    assert canonical_smiles("CCOC") in b.in_progress()
    assert b.record_proposals(["COCC"], own_previous=[])[0]["team_known"] == "in_progress"
    from evident import bus
    msgs = [bus.decode(f) for _, f in CLIENT.xrange(bus.topic_key(c, "reservations"))]
    assert msgs[0]["event"] == "reserved" and msgs[0]["activity"] == "tool_calling"
    assert msgs[0]["agent"] == "A"
    a.end_node("simulate", 0.0)          # leases of molecules not computed are released
    assert canonical_smiles("CCOC") not in b.in_progress()


def _wake_latency(notify):
    """B needs a molecule A is computing: time between A publishing the fact and B
    getting it."""
    c = campaign()
    a, b = mem(c, "A", notify=notify), mem(c, "B", notify=notify)
    b.poll = 1.0                                  # the real polling interval
    published = {}

    async def slow(task_id):
        await asyncio.sleep(1.3)
        return 11.0

    async def main():
        a.claim("CCCCN")
        task_a = asyncio.create_task(a._compute_and_store("CCCCN", slow, None, time.time()))
        await asyncio.sleep(0.1)
        task_b = asyncio.create_task(b.evaluate_molecule("CCCCN", slow))
        await task_a
        published["t"] = time.time()
        a.release("CCCCN")
        value = await task_b
        return value, time.time() - published["t"]

    value, latency = asyncio.run(main())
    a.close(), b.close()
    assert value == 11.0
    return latency


def test_push_wakes_waiting_agent_faster_than_poll():
    push, poll = _wake_latency("push"), _wake_latency("poll")
    print(f"    fact -> waiting agent: push {push * 1000:.0f} ms, poll {poll * 1000:.0f} ms")
    assert push < 0.2 and push < poll


def test_lease_renewed_while_working_and_freed_after_crash():
    """A lease of 1 s survives a 2.5 s computation (renewal) but a crashed holder frees
    the molecule after one lease."""
    c = campaign()
    a, b = mem(c, "A", ttl=1), mem(c, "B", ttl=1)
    calls = []

    async def long_compute(task_id):
        calls.append(task_id)
        await asyncio.sleep(2.5)
        return 9.0

    async def main():
        ta = asyncio.create_task(a.evaluate_molecule("CCCCC", long_compute))
        await asyncio.sleep(0.2)
        tb = asyncio.create_task(b.evaluate_molecule("CCCCC", long_compute))
        return await asyncio.gather(ta, tb)

    assert asyncio.run(main()) == [9.0, 9.0] and len(calls) == 1, "renewed: computed once"
    ghost = mem(c, "ghost", ttl=1)
    assert ghost.claim("CCCCCC")                   # reserved, then the agent "crashes"

    async def quick(task_id):
        return 8.0

    t0 = time.time()
    assert asyncio.run(b.evaluate_molecule("CCCCCC", quick)) == 8.0
    assert 0.8 <= time.time() - t0 < 2.0
    a.close(), b.close()


def test_fact_topic_carries_task_id_and_activity():
    c = campaign()
    a = mem(c, "A")
    a.begin_node("simulate")

    async def compute(task_id):
        return 10.5

    asyncio.run(a.evaluate_molecule("CCF", compute, new_task_id=lambda: "flowcept-7"))
    a.close()
    from evident import bus
    env = bus.decode(CLIENT.xrange(bus.topic_key(c, "facts"))[-1][1])
    assert env["task_id"] == "flowcept-7" and env["activity"] == "simulate"
    assert env["payload"]["smiles"] == canonical_smiles("CCF")


def test_directive_pushed_to_running_agent():
    c = campaign()
    a, ctrl = mem(c, "A"), mem(c, "analysis")
    a.control_poll = 2.0          # a slow poll: only push can explain a fast reaction

    async def slow(task_id):
        await asyncio.sleep(5)
        return 1.0

    async def main():
        task = asyncio.create_task(a.evaluate_molecule("CCCCCCO", slow))
        await asyncio.sleep(0.3)
        sent = time.time()
        ctrl.send_directive("A", "stop", "drift", rule="test")
        try:
            await task
        except StoppedByDirective:
            return time.time() - sent

    latency = asyncio.run(main())
    a.close()
    print(f"    STOP pushed and applied in {latency * 1000:.0f} ms")
    assert latency is not None and latency < 0.5


def cleanup():
    for c in CAMPAIGNS:
        for key in CLIENT.scan_iter(f"ev:{c}:*"):
            CLIENT.delete(key)
        CLIENT.zrem("ev:campaigns", c)


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
                import traceback
                traceback.print_exc()
                print(f"[FAIL]  {t.__name__}: {exc!r}")
    finally:
        cleanup()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
