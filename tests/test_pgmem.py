"""
Tests of the shared memory and of the analysis agent, without LLM (scripted) and without xTB.
Needs a Redis server: REDIS_HOST / REDIS_PORT (default localhost:6379). Run: python tests/test_pgmem.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]

from pgmem.agent_client import MemoryClient, split_knowledge  # noqa: E402
from pgmem.analyzer import Analyzer  # noqa: E402
from pgmem.config import Config  # noqa: E402
from pgmem.llm import AnalysisLLM  # noqa: E402
from pgmem.provenance import JsonlProvenance, ProvenanceIndex, smiles_of  # noqa: E402
from pgmem.schema import ProvRef, knowledge, observation  # noqa: E402
from pgmem import chem
from pgmem.store import MemoryStore  # noqa: E402

HOST, PORT = os.environ.get("REDIS_HOST", "localhost"), int(os.environ.get("REDIS_PORT", "6379"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COND = {"method": "GFN2-xTB", "accuracy": 0.05, "fmax": 0.02, "max_steps": 250, "rdkit_seed": 1}


class FakeSource:
    def __init__(self) -> None:
        self.index, self.pending = ProvenanceIndex(), []

    def poll(self):
        out, self.pending = self.pending, []
        return out


def setup(mode: str = "full", llm=None, source=None):
    cfg = Config(mode=mode, namespace=f"test-{time.time_ns()}", redis_host=HOST, redis_port=PORT,
                 campaign_id="c1")
    store = MemoryStore.connect(HOST, PORT, cfg.namespace)
    an = Analyzer(store, cfg, llm=llm or AnalysisLLM("scripted"), source=source)
    return cfg, store, an


def client(store, cfg, name):
    return MemoryClient(store, cfg, name)


def drain(an, rounds: int = 5):
    for _ in range(rounds):
        if not an.step(block_ms=10, retry_idle_ms=0):
            break


def diagnoses(store, rule=None):
    return [e for e in store.all("M2") if e.kind == "diagnosis" and (rule is None or e.data["context"]["rule"] == rule)]


# --------------------------------------------------------------------------------------------- memory
def test_m1_deduplication_keeps_every_reference_and_history():
    cfg, s, _ = setup()
    a = s.add(observation("CCO", "ionization_energy", 12.3, "eV", COND, "a1", ProvRef(task_id="t1")))
    b = s.add(observation("CCO", "ionization_energy", 12.3, "eV", COND, "a2", ProvRef(task_id="t2")))
    assert a[0] == b[0] and b[1] is True
    e = s.get(a[0])
    assert [r["task_id"] for r in e.prov_refs] == ["t1", "t2"] and e.history[0]["change"] == "merged duplicate"


def test_experiences_are_never_merged_and_knowledge_is_never_overwritten():
    cfg, s, _ = setup()
    c = client(s, cfg, "a1")
    c.record_failure("C1CC", "Parse failure for C1CC", "t1", None, "c1")
    c.record_failure("C1CC", "Parse failure for C1CC", "t2", None, "c1")
    assert len([e for e in s.all("M2") if e.kind == "failure"]) == 2
    k = s.add(knowledge("finding", "X is above 3", "observation", "a1"))[0]
    s.add_evidence(k, "contradicting_evidence", {"entry": "m1:x"}, "analysis-agent", "test")
    e = s.get(k)
    assert e.text == "X is above 3" and len(e.data["contradicting_evidence"]) == 1 and len(e.history) == 1


def test_m3_epistemic_status_is_restricted_to_the_three_kinds():
    try:
        knowledge("finding", "x", "established", "a1")
        raise AssertionError("accepted an unknown epistemic status")
    except ValueError:
        pass


def test_knowledge_block_is_split_from_the_conclusion():
    body, items = split_knowledge('Some text.\n\nKNOWLEDGE\n[{"statement": "A", "kind": "finding"}]')
    assert body == "Some text." and items[0]["statement"] == "A"
    assert split_knowledge("No block here.") == ("No block here.", [])


def test_agent_reuses_a_team_result_and_sees_team_failures():
    cfg, s, _ = setup()
    a1, a2 = client(s, cfg, "a1"), client(s, cfg, "a2")
    a1.record_result("OCC", 9.5, {**COND, "converged": True}, "t1", "w1", "c1")
    assert a2.lookup("CCO", COND) == 9.5                     # same molecule, other SMILES spelling
    a1.record_failure("C1CC", "Parse failure for C1CC", "t2", None, "c1")
    txt = a2.explored()
    assert "CCO" in txt and "C1CC" in txt


# --------------------------------------------------------------------------------------------- analysis agent
def test_redundant_computation_is_detected_and_guided():
    cfg, s, an = setup()
    a1, a2 = client(s, cfg, "a1"), client(s, cfg, "a2")
    a1.record_result("CCO", 9.5, {**COND, "converged": True}, "t1", "w1", "c1")
    a2.record_result("OCC", 9.5, {**COND, "converged": True}, "t2", "w2", "c1")
    drain(an)
    d = diagnoses(s, "redundant_computation")
    assert len(d) == 1 and d[0].data["problem_type"] == "workflow" and d[0].data["basis"] == "observed_fact"
    g = s.all_guidance()
    assert g and g[0].check["avoid_keys"] == ["CCO"]


def test_repeated_invalid_smiles_failure_has_a_confirmed_cause():
    cfg, s, an = setup()
    a1 = client(s, cfg, "a1")
    for t in ("t1", "t2"):
        a1.record_failure("C1CC", "Parse failure for C1CC", t, None, "c1")
    drain(an)
    d = diagnoses(s, "repeated_failure")
    assert d and d[0].data["basis"] == "confirmed_cause" and d[0].data["problem_type"] == "technical_failure"


def test_recurring_failure_class_gives_a_procedure_linked_to_its_experiences_and_is_applied():
    cfg, s, an = setup()
    a1, a2 = client(s, cfg, "a1"), client(s, cfg, "a2")
    f1 = a1.record_failure("CCN", "SCF not converged", "t1", None, "c1")
    f2 = a1.record_failure("CCCN", "SCF not converged", "t2", None, "c1")
    drain(an)
    assert diagnoses(s, "failure_explanation")[0].data["basis"] == "candidate_explanation"
    procs = s.all("M4")
    assert len(procs) == 1 and set(procs[0].data["supporting_episodes"]) == {f1, f2}
    for f in ("trigger", "preconditions", "actions", "expected_outcome", "limitations"):
        assert procs[0].data[f]
    a2.record_failure("CCCCN", "SCF not converged", "t3", None, "c1")
    drain(an)
    assert any(g.procedure_id == procs[0].id and g.target == "a2" for g in s.all_guidance())



def test_own_results_computed_after_the_proposal_are_not_counted_as_already_measured():
    """Real campaign: the analysis agent read the proposals after the agent had stored their results."""
    cfg, s, an = setup()
    a1 = client(s, cfg, "a1")
    a1.record_result("CCO", 9.5, {**COND, "converged": True}, "t1", "w1", "c1")
    a1.record_proposals(["CCO", "CCCO", "CCCCO"])            # CCO was known when proposed, the others were not
    a1.record_result("CCCO", 9.4, {**COND, "converged": True}, "t2", "w1", "c1")
    a1.record_result("CCCCO", 9.3, {**COND, "converged": True}, "t3", "w1", "c1")
    drain(an)                                                # proposals processed only now
    d = diagnoses(s, "repeated_proposals")
    assert len(d) == 1 and "1 already measured" in d[0].text and "CCCO" not in d[0].text, d[0].text



def test_knowledge_about_a_failed_molecule_cites_the_failure():
    cfg, s, _ = setup()
    a1 = client(s, cfg, "a1")
    fid = a1.record_failure("C1CC", "Parse failure for C1CC", "t1", None, "c1")
    a1.record_conclusion('Done.\nKNOWLEDGE: [{"kind": "constraint", "statement": "C1CC cannot be parsed", '
                         '"epistemic_status": "observation", "supporting_molecules": ["C1CC"]}]', "plan", None)
    k = [e for e in s.all("M3") if "C1CC" in e.text][0]
    assert [ev["entry"] for ev in k.data["supporting_evidence"]] == [fid]


def test_lock_one_computation_at_a_time_and_the_waiter_gets_the_result():
    import asyncio
    cfg, s, an = setup()
    a1, a2 = client(s, cfg, "a1"), client(s, cfg, "a2")

    async def run():
        v1, tok = await a1.claim_or_wait("CCO", COND)          # a1 computes
        assert v1 is None and tok
        waiter = asyncio.create_task(a2.claim_or_wait("OCC", COND, poll_s=0.05))   # same molecule, other spelling
        await asyncio.sleep(0.3)
        assert not waiter.done()                                 # a2 waits instead of computing
        a1.record_result("CCO", 9.5, {**COND, "converged": True}, "t1", "w1", "c1")
        a1.release("CCO", tok)
        return await waiter
    assert asyncio.run(run()) == (9.5, None)
    assert not s.r.exists(s._k("claim", "CCO"))
    drain(an)
    rep = an.post_execution()
    assert rep["coordination"] == {"waits": 1, "result": 1, "failure": 0, "computes": 0}, rep["coordination"]


def test_lock_waiter_gets_the_failure_and_an_expired_lock_is_taken_over():
    import asyncio
    cfg, s, _ = setup()
    cfg.claim_ttl_s = 1
    a1, a2, a3 = client(s, cfg, "a1"), client(s, cfg, "a2"), client(s, cfg, "a3")

    async def failing():
        _, tok = await a1.claim_or_wait("C1CC", COND)
        waiter = asyncio.create_task(a2.claim_or_wait("C1CC", COND, poll_s=0.05))
        await asyncio.sleep(0.2)
        a1.record_failure("C1CC", "Parse failure for C1CC", "t1", None, "c1")
        a1.release("C1CC", tok)
        try:
            await waiter
            return None
        except RuntimeError as exc:
            return str(exc)
    msg = asyncio.run(failing())
    assert msg and "Parse failure" in msg and len(a2._iter_failures) == 1
    assert len([e for e in s.all("M2") if e.kind == "failure"]) == 1      # not computed (nor failed) twice

    async def crashed():
        _, tok = await a1.claim_or_wait("CCCC", COND)          # a1 "crashes": never releases
        return await a3.claim_or_wait("CCCC", COND, poll_s=0.1)
    v, tok3 = asyncio.run(crashed())
    assert v is None and tok3                                    # a3 computes once the lock expired



def test_no_lock_when_disabled():
    import asyncio
    cfg, s, _ = setup()
    cfg.claims = False
    a1, a2 = client(s, cfg, "a1"), client(s, cfg, "a2")
    assert asyncio.run(a1.claim_or_wait("CCO", COND)) == (None, None)
    assert asyncio.run(a2.claim_or_wait("CCO", COND)) == (None, None)


def test_invalid_smiles_guidance_is_applied_when_every_new_proposal_is_valid():
    cfg, s, an = setup()
    a1, a2, a3 = client(s, cfg, "a1"), client(s, cfg, "a2"), client(s, cfg, "a3")
    a1.record_failure("C1CC", "Parse failure for C1CC", "t1", None, "c1")
    a1.record_failure("C1CCN", "Parse failure for C1CCN", "t2", None, "c1")
    drain(an)
    g = [g for g in s.all_guidance() if g.check.get("failure_class") == "invalid_smiles"][0]
    a2.context("plan", "x")
    a3.context("plan", "x")
    a2.record_proposals(["CCO", "CCCO"])            # valid SMILES only
    a3.record_proposals(["CCO", "CC("])             # one invalid SMILES, not in the guidance's list
    drain(an)
    g = s.get_guidance(g.id)
    assert g.applied == {"a2": True, "a3": False}, g.applied
    a2.record_result("CCO", 9.5, {**COND, "converged": True}, "t3", "w2", "c1")
    a2.record_conclusion("Done.", "plan", None)
    drain(an)
    g = s.get_guidance(g.id)
    assert g.effective.get("a2", "missing") is None                    # not separable from the application
    out = [e for e in s.all("M2") if e.kind == "outcome" and e.key == "a2"]
    assert out and "its effect is the applied behaviour itself" in out[0].text


def test_procedure_is_drafted_again_when_the_llm_answer_was_unusable():
    from pgmem.llm import DRAFT_PROCEDURE
    llm = AnalysisLLM("scripted")
    real, calls = llm.ask, {"n": 0}

    def ask(system, user):
        if system == DRAFT_PROCEDURE:
            calls["n"] += 1
            if calls["n"] == 1:
                return "Here is a procedure: first check the SCF settings"     # not JSON: unusable
        return real(system, user)
    llm.ask = ask
    cfg, s, an = setup(llm=llm)
    a1 = client(s, cfg, "a1")
    a1.record_failure("CCN", "SCF not converged", "t1", None, "c1")
    drain(an)
    a1.record_failure("CCCN", "SCF not converged", "t2", None, "c1")
    drain(an)
    assert not s.all("M4") and calls["n"] == 1
    a1.record_failure("CCCCN", "SCF not converged", "t3", None, "c1")
    drain(an)
    assert len(s.all("M4")) == 1 and calls["n"] == 2

def test_procedure_fields_given_as_lists_by_the_llm_are_stored_as_text():
    cfg, s, an = setup()
    pid = an._store_procedure("k", [], {"text": "Fix", "trigger": ["a", "b"], "preconditions": {"x": 1},
                                        "actions": ["check ring closures", "re-propose"], "expected_outcome": "ok",
                                        "limitations": None})
    m4 = s.all("M4")[0]
    assert pid and m4.data["actions"] == "check ring closures; re-propose" and m4.data["trigger"] == "a; b"
    assert m4.data["preconditions"] == "x: 1" and m4.data["limitations"] == ""

def test_invalid_artifact_identifies_affected_knowledge_and_downstream_tasks():
    src = FakeSource()
    src.index.add({"task_id": "t1", "subtype": "tool_call", "activity_id": "compute_ionization_energy",
                   "used": {"input": "{'smiles': 'CCO'}"}, "workflow_id": "wg", "campaign_id": "c1",
                   "ended_at": 10.0})
    src.index.add({"task_id": "n1", "subtype": "langgraph_node", "activity_id": "conclude", "workflow_id": "wg",
                   "started_at": 11.0})
    cfg, s, an = setup(source=src)
    a1 = client(s, cfg, "a1")
    good = a1.record_result("CC", 8.0, {**COND, "converged": True}, "t0", "w1", "c1")
    k = s.add(knowledge("finding", "Ethanol ionizes above 9 eV", "observation", "a1",
                        supporting=[{"entry": good}]))[0]
    drain(an)
    bad = a1.record_result("CCO", 9.9, {**COND, "converged": False, "opt_steps": 3}, "t1", "wg", "c1")
    s.update(k, "a1", "evidence", lambda e: (e.data["supporting_evidence"].append({"entry": bad}),
                                             e.refs.append(bad)))
    an._on_result({"entry": bad})              # evaluated directly: the knowledge cites it now
    d = diagnoses(s, "invalid_artifact")
    assert d and d[0].data["problem_type"] == "data_quality"
    assert k in d[0].data["context"]["affected"]
    assert [t["activity_id"] for t in d[0].data["context"]["downstream_tasks"]] == ["conclude"]
    assert s.get(bad).data["notes"] and s.get(k).text == "Ethanol ionizes above 9 eV"


def test_new_result_contradicting_knowledge_adds_evidence_without_changing_it():
    cfg, s, an = setup()
    a1 = client(s, cfg, "a1")
    first = a1.record_result("CCO", 12.0, {**COND, "converged": True}, "t1", "w", "c1")
    k = s.add(knowledge("finding", "The ionization energy of CCC is above 10.00 eV", "hypothesis", "a1",
                        supporting=[{"entry": first}], key="CCC"))[0]
    drain(an)
    a1.record_result("CCC", 5.0, {**COND, "converged": True}, "t2", "w", "c1")
    drain(an)
    e = s.get(k)
    assert e.text.startswith("The ionization energy of CCC") and len(e.data["contradicting_evidence"]) == 1
    d = diagnoses(s, "contradiction")
    assert d and d[0].data["problem_type"] == "scientific_contradiction"
    assert d[0].data["basis"] == "candidate_explanation"


def test_guidance_is_delivered_once_then_applied_and_effective():
    cfg, s, an = setup()
    a1, a2 = client(s, cfg, "a1"), client(s, cfg, "a2")
    a1.record_result("CCO", 9.5, {**COND, "converged": True}, "t1", "w1", "c1")
    a2.record_result("CCO", 9.5, {**COND, "converged": True}, "t2", "w2", "c1")
    drain(an)
    ctx = a2.context("update", "next molecules")
    assert "Guidance from the analysis agent" in ctx and "Guidance" not in a2.context("plan", "x")
    a2.record_proposals(["CCCC"])
    drain(an)
    a2.record_result("CCCC", 8.0, {**COND, "converged": True}, "t3", "w2", "c1")
    a2.record_conclusion("Conclusions.", "plan", None)
    drain(an)
    g = s.all_guidance()[0]
    assert g.applied.get("a2") is True and g.effective.get("a2") is True
    assert any(e.kind == "outcome" for e in s.all("M2"))


def test_procedure_that_stops_working_is_identified():
    cfg, s, an = setup()
    from pgmem.schema import procedure
    pid = s.add(procedure("procedure", "Avoid CCN-like molecules", "SCF failure", "", "avoid them", "no failure",
                          "few cases", ["m2:a", "m2:b"], "analysis-agent", key="scf_not_converged"))[0]
    an._record_application(pid, "a1", "g1", "m2:o1", True)
    an._record_application(pid, "a1", "g2", "m2:o2", False)
    assert diagnoses(s, "procedure_no_longer_effective")
    assert len(s.get(pid).data["applications"]) == 2


def test_procedure_given_but_not_applied_is_not_counted_as_applied():
    from pgmem.schema import procedure, validation_status
    cfg, s, an = setup()
    pid = s.add(procedure("procedure", "Check SMILES", "invalid_smiles", "", "check them", "valid SMILES",
                          "few cases", ["m2:a", "m2:b"], "analysis-agent", key="invalid_smiles"))[0]
    an._record_application(pid, "a1", "g1", "m2:o1", None, False)
    an._record_application(pid, "a2", "g2", "m2:o2", True, True)
    st = validation_status(s.get(pid))
    assert st == "derived from 2 experience(s); given 2 time(s): applied 1 (1 effective, 0 not effective, " \
                 "0 effect unknown), not applied 1", st


def test_analysis_work_is_not_lost_when_the_llm_is_unavailable():
    class Down(AnalysisLLM):
        def __init__(self):
            super().__init__("scripted")
            self.up = False

        def ask(self, system, user):
            if not self.up:
                raise ConnectionError("LLM unavailable")
            return super().ask(system, user)
    llm = Down()
    cfg, s, an = setup(llm=llm)
    a1 = client(s, cfg, "a1")
    a1.record_failure("CCN", "SCF not converged", "t1", None, "c1")
    a1.record_failure("CCCN", "SCF not converged", "t2", None, "c1")
    drain(an)
    assert an.stats["failed_attempts"] >= 1 and not s.all("M4")
    pending = s.r.xpending(s._k("events"), cfg.analyzer_group)["pending"]
    assert pending >= 1
    llm.up = True
    drain(an)
    assert s.all("M4") and s.r.xpending(s._k("events"), cfg.analyzer_group)["pending"] == 0


def test_two_iterations_without_improvement_get_advice():
    cfg, s, an = setup()
    a1 = client(s, cfg, "a1")
    a1.record_result("CCO", 9.5, {**COND, "converged": True}, "t1", "w", "c1")
    for i in range(3):
        a1.record_proposals(["CCO"])
        a1.record_conclusion("Conclusions.", "plan", None)
    drain(an, 10)
    assert diagnoses(s, "no_improvement")
    assert any(g.target == "a1" and g.basis == "candidate_explanation" for g in s.all_guidance())


def test_basic_mode_shares_results_only():
    cfg, s, _ = setup(mode="basic")
    c = client(s, cfg, "a1")
    c.record_result("CCO", 9.5, {**COND, "converged": True}, "t1", "w", "c1")
    assert c.record_failure("C1CC", "Parse failure", "t2", None, "c1") is None
    assert c.context("plan", "x") == "" and c.conclude_instruction() == ""
    assert s.count("M1") == 1 and s.count("M2") == 0


# --------------------------------------------------------------------------------------------- provenance
def test_jsonl_provenance_on_a_real_gaied_file():
    p = JsonlProvenance(os.path.join(ROOT, "output", "flowcept_buffer_SC.jsonl"))
    new = p.poll()
    comps = p.index.computations()
    assert len(comps) == 16 and smiles_of(comps[0]) == "NC(N)=O"
    assert all(r.get("subtype") != "academy_action" for r in new)          # event filtering
    assert [r["activity_id"] for r in p.index.downstream(comps[0]["task_id"])] == ["conclude", "should_continue"]
    assert p.poll() == []                                                    # nothing new: deduplicated



# --------------------------------------------------------------------------------------------- evaluation support
def test_fault_selection_is_deterministic_and_follows_the_fraction():
    from pgmem import faults
    mols = ["C" * n + "O" for n in range(1, 41)] + ["C" * n + "N" for n in range(1, 41)]
    rule = {"fraction": 0.3, "salt": "x"}
    picked = [m for m in mols if faults.selected(rule, m)]
    assert picked == [m for m in mols if faults.selected(rule, m)]            # same molecules every time
    assert 10 <= len(picked) <= 40, len(picked)                                 # about 30 % of 80
    assert faults.selected(rule, "OCC") == faults.selected(rule, "CCO")         # canonical SMILES
    assert not faults.selected({"fraction": 1.0}, "C1CC")                       # never an unreadable SMILES
    assert faults.selected({"smarts": "C#N"}, "CC#N") and not faults.selected({"smarts": "C#N"}, "CCO")


def test_context_leaves_out_the_categories_removed_by_an_ablation():
    cfg, s, _ = setup()
    a1 = client(s, cfg, "a1")
    a1.record_result("CCO", 9.5, {**COND, "converged": True}, "t1", "w1", "c1")
    s.add(knowledge("finding", "CCO has a high ionization energy", "observation", "a1"))
    assert "M3" in a1.context("plan", "CCO")
    cfg.context_categories = ("M1",)
    ctx = a1.context("plan", "CCO")
    assert "(M1)" in ctx and "(M3)" not in ctx
    ev = [json.loads(f["payload"]) for _, f in s.r.xrange(s._k("events")) if f["event"] == "context"]
    assert ev[-1]["entries"]["M3"] == [] and ev[-1]["entries"]["M1"]


def test_analysis_without_the_memory_source_keeps_only_guidance_tracking():
    cfg, s, an = setup()
    cfg.analysis_sources = ("provenance",)
    a1, a2 = client(s, cfg, "a1"), client(s, cfg, "a2")
    a1.record_result("CCO", 9.5, {**COND, "converged": True}, "t1", "w1", "c1")
    a2.record_result("CCO", 9.5, {**COND, "converged": True}, "t2", "w2", "c1")
    drain(an)
    assert not diagnoses(s) and not s.all_guidance()


# --------------------------------------------------------------------------------------------- retrieval
def test_retrieval_ranks_results_on_related_molecules_first():
    from pgmem import retrieval
    cfg, s, _ = setup()
    a1 = client(s, cfg, "a1")
    for smi, v in [("FC(F)(F)C#N", 16.0), ("FC(F)(F)CC#N", 15.5), ("c1ccccc1", 9.0), ("CCCCCCCC", 10.0)]:
        a1.record_result(smi, v, {**COND, "converged": True}, f"t-{smi}", "w1", "c1")
    q = retrieval.build_query("try more nitriles like N#CC(F)(F)F", [])
    got = [e.key for e, _, _ in retrieval.retrieve(s, "M1", q, 2)]
    assert got[0] == chem.canonical("FC(F)(F)C#N") and chem.canonical("c1ccccc1") not in got, got


def test_m2_kinds_are_filtered_before_ranking():
    cfg, s, _ = setup()
    a1 = client(s, cfg, "a1")
    for i in range(12):                                           # many attempts mentioning the query words
        a1.record_conclusion(f"nitrile iteration {i}", "nitrile plan", None)
    a1.record_failure("CC#N", "SCF not converged", "t1", None, "c1")
    got = s.search("M2", "nitrile CC#N", limit=1, kinds=("failure",))
    assert got and got[0].kind == "failure"


def test_m1_merges_two_spellings_of_the_same_molecule():
    cfg, s, _ = setup()
    a1 = client(s, cfg, "a1")
    a1.record_result("CCO", 9.5, {**COND, "converged": True}, "t1", "w1", "c1")
    a1.record_result("OCC", 9.5, {**COND, "converged": True}, "t2", "w1", "c1")
    assert len(s.all("M1")) == 1 and len(s.all("M1")[0].prov_refs) == 2


def test_on_demand_answer_reports_failures_of_similar_molecules_and_is_logged():
    cfg, s, _ = setup()
    a1, a2 = client(s, cfg, "a1"), client(s, cfg, "a2")
    a1.record_failure("FC(F)(F)C#N", "SCF not converged", "t1", None, "c1")
    ans = a2.answer("Have similar molecules failed?", "FC(F)(F)CC#N")
    assert "[failure" in ans and chem.canonical("FC(F)(F)C#N") in ans, ans
    ev = [json.loads(f["payload"]) for _, f in s.r.xrange(s._k("events")) if f["event"] == "memory_query"]
    assert ev and ev[-1]["entries"]["M2"]


def test_failures_are_reused_only_in_the_ablation():
    import asyncio
    cfg, s, _ = setup()
    a1 = client(s, cfg, "a1")
    a1.record_failure("C1CC", "Parse failure for C1CC", "t1", None, "c1")
    assert asyncio.run(a1.claim_or_wait("C1CC", COND))[1]         # computed again by default (lock taken)
    s.r.delete(s._k("claim", "C1CC"))
    cfg.reuse_failures = True
    try:
        asyncio.run(a1.claim_or_wait("C1CC", COND))
        raise AssertionError("expected the recorded failure")
    except RuntimeError as exc:
        assert "not computed again" in str(exc)


def test_faults_change_after_a_number_of_computations():
    import tempfile
    from pgmem import faults
    log = tempfile.mktemp()
    os.environ["PGMEM_FAULTS_LOG"] = log
    os.environ["PGMEM_FAULTS"] = json.dumps({"phases": [
        {"until_computations": 2, "fail": {"smarts": "C#N"}}, {"fail": {"smarts": "[CX4](F)(F)F"}}]})
    try:
        out = []
        for smi in ["CC#N", "CCO", "CC#N", "CC(F)(F)F"]:
            try:
                faults.before(smi, 10)
                out.append("ok")
            except RuntimeError as exc:
                out.append(str(exc))
        assert out == ["SCF not converged", "ok", "ok", "SCF not converged"], out
    finally:
        os.environ.pop("PGMEM_FAULTS")
        os.environ.pop("PGMEM_FAULTS_LOG")


# --------------------------------------------------------------------------------------------- verification
def test_claim_checks_are_deterministic_on_reliable_results():
    from pgmem import verification as v
    cfg, s, _ = setup()
    a1 = client(s, cfg, "a1")
    vals = {"CC(F)(F)F": 15.0, "CCC(F)(F)F": 14.8, "CCC": 11.0, "CCCC": 10.5, "CCO": 9.0}
    for smi, val in vals.items():
        a1.record_result(smi, val, {**COND, "converged": True}, f"t-{smi}", "w1", "c1")
    a1.record_result("CCN", 8.0, {**COND, "converged": False}, "t-n", "w1", "c1")
    k = chem.canonical
    assert v.check(v.normalize({"form": "comparison", "a": "CCC", "relation": "higher", "b": "CCO"}), s)[
        "verdict"] == "supported"
    assert v.check(v.normalize({"form": "bound", "molecule": "CCO", "relation": "above", "value": 10}), s)[
        "verdict"] == "contradicted"
    r = v.check(v.normalize({"form": "value", "molecule": "CCN", "value": 8.0}), s)
    assert r["verdict"] == "inconclusive" and "unreliable" in r["reason"]
    r = v.check(v.normalize({"form": "motif_effect", "feature": "trifluoromethyl", "direction": "increases"}), s)
    assert r["verdict"] == "supported", r
    r = v.check(v.normalize({"form": "motif_effect", "feature": "trifluoromethyl", "direction": "decreases"}), s,
                values={k("CC(F)(F)F"): 15.0, k("CCC(F)(F)F"): 14.8})
    assert r["verdict"] == "contradicted"
    assert v.normalize({"form": "motif_effect", "feature": "magic", "direction": "increases"}) == {"form": "none"}


def test_analysis_agent_flags_a_contradicted_statement_and_rechecks_on_new_results():
    cfg, s, an = setup()
    a1 = client(s, cfg, "a1")
    a1.record_result("CCO", 9.5, {**COND, "converged": True}, "t1", "w1", "c1")
    drain(an)
    a1.record_conclusion('Done.\nKNOWLEDGE: [{"kind": "finding", "statement": "The ionization energy of CCCO '
                         'is above 12.0 eV", "epistemic_status": "observation", "supporting_molecules": []}]',
                         "plan", None)
    drain(an)
    kid = [e for e in s.all("M3")][0].id
    assert s.get_verification(kid)["verdict"] == "inconclusive"            # no result for CCCO yet
    a1.record_result("CCCO", 9.8, {**COND, "converged": True}, "t2", "w1", "c1")
    drain(an)
    rec = s.get_verification(kid)
    assert rec["verdict"] == "contradicted" and len(rec["history"]) == 2, rec
    assert diagnoses(s, "claim_contradicted_by_new_result")
    assert any("contradicted" in g.message for g in s.all_guidance())
    assert "checked against the results: contradicted" in a1.context("plan", "CCCO")

TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print("ok  ", t.__name__)
        except Exception:  # noqa: BLE001
            failed += 1
            print("FAIL", t.__name__)
            traceback.print_exc()
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
