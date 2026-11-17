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
