"""
Tests of the shared memory, its retrieval and the analysis agent, with a scripted LLM and without xTB.
Needs a Redis server: REDIS_HOST / REDIS_PORT (default localhost:6379). Run: python tests/test_pgmem.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]
os.environ["LLM_PROVIDER"] = "scripted"           # scripted LLM and scripted embeddings

from pgmem import retrieval, verification  # noqa: E402
from pgmem.agent_client import MemoryClient, split_knowledge  # noqa: E402
from pgmem.analyzer import Analyzer  # noqa: E402
from pgmem.config import Config  # noqa: E402
from pgmem.llm import AnalysisLLM  # noqa: E402
from pgmem.provenance import JsonlProvenance, ProvenanceIndex, smiles_of  # noqa: E402
from pgmem.schema import ProvRef, knowledge, result  # noqa: E402
from pgmem.store import MemoryStore  # noqa: E402

HOST, PORT = os.environ.get("REDIS_HOST", "localhost"), int(os.environ.get("REDIS_PORT", "6379"))
COND = {"method": "GFN2-xTB", "accuracy": 0.05, "fmax": 0.02, "max_steps": 250, "rdkit_seed": 1}
OK = {**COND, "converged": True}
SCF = "SCF not converged"


class FakeSource:
    def __init__(self) -> None:
        self.index, self.pending = ProvenanceIndex(), []

    def poll(self):
        out, self.pending = self.pending, []
        return out


class NoLLM:
    """An LLM that is down: every call fails."""
    calls = 0

    def ask_json(self, system, user):
        raise ConnectionError("LLM unavailable")


def setup(mode: str = "full", llm=None, source=None, **cfg_kw):
    cfg = Config(mode=mode, namespace=f"test-{time.time_ns()}", redis_host=HOST, redis_port=PORT,
                 campaign_id="c1", **cfg_kw)
    store = MemoryStore.connect(HOST, PORT, cfg.namespace)
    return cfg, store, Analyzer(store, cfg, llm=llm or AnalysisLLM("scripted"), source=source)


def client(store, cfg, name):
    return MemoryClient(store, cfg, name)


def drain(an, rounds: int = 8):
    for _ in range(rounds):
        if not an.step(block_ms=10, retry_idle_ms=0):
            break


def diagnoses(store, rule=None):
    return [e for e in store.all("M2") if e.kind == "diagnosis" and (rule is None or e.data["rule"] == rule)]


def fail(c, smiles, error=SCF):
    c.record_failure(smiles, error, f"t-{smiles}-{time.time_ns()}", "w1", "c1")


def conclude(c, items=None, proposals=None):
    if proposals is not None:
        c.record_proposals(proposals)
    text = "Conclusions." + (("\nKNOWLEDGE\n" + json.dumps(items)) if items else "")
    return c.record_conclusion(text, "plan", None)


# --------------------------------------------------------------------------------------------- memory
def test_m1_duplicates_are_merged_by_appending_references():
    cfg, s, _ = setup()
    a = s.add(result("CCO", "ionization_energy", 12.3, "eV", OK, "a1", ProvRef(task_id="t1")))
    b = s.add(result("CCO", "ionization_energy", 12.3, "eV", OK, "a2", ProvRef(task_id="t2")))
    assert a[0] == b[0] and b[1] is True
    e = s.get(a[0])
    assert [r["task_id"] for r in e.prov_refs] == ["t1", "t2"] and e.history[0]["change"] == "merged duplicate"
    c1, c2 = client(s, cfg, "a1"), client(s, cfg, "a2")
    assert c1.record_result("OCC", 9.5, OK, "t3", "w", "c1") == c2.record_result("CCO", 9.5, OK, "t4", "w", "c1")


def test_types_and_epistemic_status_are_restricted():
    for bad in (lambda: knowledge("x", "finding", "a1"), lambda: result("CCO", "ie", 1.0, "eV", {}, "a1", ProvRef(),
                                                                           spelling="CCO").__class__("M3", "claim", "x")):
        try:
            bad()
            raise AssertionError("accepted")
        except ValueError:
            pass


def test_knowledge_block_is_split_from_the_conclusion():
    body, items = split_knowledge('Text.\n**KNOWLEDGE**\n[{"statement": "a"}]')
    assert body == "Text." and items == [{"statement": "a"}]
    assert split_knowledge("No block") == ("No block", [])


def test_agent_writes_create_the_expected_relations():
    cfg, s, _ = setup()
    a1 = client(s, cfg, "CCO#a1")
    a1.record_proposals(["CC#N", "CCO"])
    a1.record_result("CC#N", 12.0, OK, "t1", "w1", "c1")
    fail(a1, "NCC#N")
    conclude(a1, [{"statement": "Nitrile molecules have a high ionization energy", "epistemic_status": "hypothesis",
                   "supporting_molecules": ["CC#N"], "explains_molecules": ["NCC#N"]}])
    r = s.by_key("M1", "CC#N")[0]
    k = s.all("M3")[0]
    att = [e for e in s.all("M2") if e.kind == "attempt"][0]
    assert s.molecules_of(r.id) == ["CC#N"] and "motif:nitrile" in {
        e["target"] for e in s.graph.outgoing("mol:CC#N", ["HAS_MOTIF"])}
    cited = {(e["type"], e["origin"]) for e in s.graph.outgoing(k.id)}
    assert {("SUPPORTED_BY", "llm"), ("EXPLAINS", "llm"), ("ABOUT", "deterministic")} <= cited, cited
    assert s.motifs_of(k.id) == ["nitrile"]                     # the motif the statement names in words
    assert {e["source"] for e in s.graph.incoming(att.id, ["PART_OF"])} == {r.id} | {
        e.id for e in s.all("M2") if e.kind == "failure"}


def test_agent_reuses_a_team_result_and_sees_team_failures():
    cfg, s, _ = setup()
    a1, a2 = client(s, cfg, "a1"), client(s, cfg, "a2")
    a1.record_result("CCO", 9.5, OK, "t1", "w1", "c1")
    fail(a1, "C1CC", "Parse failure for C1CC")
    assert a2.lookup("OCC", COND) == 9.5
    text = a2.explored()
    assert "CCO" in text and "C1CC" in text


def test_basic_mode_shares_results_only():
    cfg, s, _ = setup("basic")
    a1 = client(s, cfg, "a1")
    a1.record_result("CCO", 9.5, OK, "t1", "w1", "c1")
    assert a1.record_failure("C1CC", "Parse failure", "t2", None, "c1") is None
    assert a1.context("plan", "CCO") == "" and "CCO" in a1.explored()


# --------------------------------------------------------------------------------------------- coordination
def test_lock_one_computation_at_a_time_and_the_waiter_gets_the_result():
    cfg, s, an = setup()
    a1, a2 = client(s, cfg, "a1"), client(s, cfg, "a2")

    async def run():
        v1, tok = await a1.claim_or_wait("CCO", COND)
        assert v1 is None and tok
        waiter = asyncio.create_task(a2.claim_or_wait("OCC", COND, poll_s=0.05))
        await asyncio.sleep(0.3)
        assert not waiter.done()
        a1.record_result("CCO", 9.5, OK, "t1", "w1", "c1")
        a1.release("CCO", tok)
        return await waiter
    assert asyncio.run(run()) == (9.5, None)
    drain(an)
    assert an.post_execution()["coordination"] == {"waits": 1, "result": 1, "failure": 0, "computes": 0}


def test_lock_waiter_gets_the_failure_and_an_expired_lock_is_taken_over():
    cfg, s, _ = setup(claim_ttl_s=1)
    a1, a2, a3 = client(s, cfg, "a1"), client(s, cfg, "a2"), client(s, cfg, "a3")

    async def failing():
        _, tok = await a1.claim_or_wait("C1CC", COND)
        waiter = asyncio.create_task(a2.claim_or_wait("C1CC", COND, poll_s=0.05))
        await asyncio.sleep(0.2)
        fail(a1, "C1CC", "Parse failure for C1CC")
        a1.release("C1CC", tok)
        try:
            await waiter
        except RuntimeError as exc:
            return str(exc)
    assert "Parse failure" in asyncio.run(failing())
    assert len([e for e in s.all("M2") if e.kind == "failure"]) == 1

    async def crashed():
        await a1.claim_or_wait("CCCC", COND)                    # never released
        return await a3.claim_or_wait("CCCC", COND, poll_s=0.1)
    v, tok3 = asyncio.run(crashed())
    assert v is None and tok3


# --------------------------------------------------------------------------------------------- retrieval
def nitrile_memory():
    """Failures of nitriles, a procedure about the nitrile motif, and unrelated results."""
    cfg, s, an = setup()
    a1 = client(s, cfg, "CCO#a1")
    for m in ("CC#N", "NCC#N"):
        fail(a1, m)
    for m, v in (("CCO", 9.0), ("CCCO", 9.2), ("OCCO", 9.4)):
        a1.record_result(m, v, OK, f"t-{m}", "w1", "c1")
    drain(an)
    return cfg, s, an, a1


def test_graph_channel_finds_what_concerns_the_motif_of_a_new_molecule():
    cfg, s, _, _ = nitrile_memory()
    q = retrieval.build_query("Next: try OCC#N", [])
    hits = retrieval.retrieve(s, "M2", q, 5, ("graph",), kinds=("failure", "diagnosis"))
    assert hits and all(set(r) == {"graph"} for _, _, r in hits)
    assert {e.key for e, _, _ in hits if e.kind == "failure"} == {"CC#N", "N#CCN"}   # same motif, other molecules
    procs = [(e, w) for e in s.all("M4") if (w := retrieval.applicable(e, q, s))]
    assert procs and "nitrile" in procs[0][1]
    assert not [e for e in s.all("M4") if retrieval.applicable(e, retrieval.build_query("", ["CCCCO"]), s)]


def test_channels_are_fused_and_can_be_switched_off():
    cfg, s, _, a1 = nitrile_memory()
    q = retrieval.build_query("results near CCCO", [])
    fused = retrieval.retrieve(s, "M1", q, 5, ("bm25", "chem", "graph"))
    assert fused[0][0].key == "CCCO" and {"bm25", "chem", "graph"} <= set(fused[0][2])
    assert all("graph" not in r for _, _, r in retrieval.retrieve(s, "M1", q, 5, ("bm25", "chem")))
    sem = retrieval.retrieve(s, "M2", retrieval.build_query("failed simulation"), 5, ("semantic",),
                             a1.emb, kinds=("failure",))
    assert sem and all(set(r) == {"semantic"} for _, _, r in sem)


def test_context_completes_entries_with_their_neighbors_and_logs_ranks():
    cfg, s, an, a1 = nitrile_memory()
    ctx = client(s, cfg, "OCC#N#a2").context("plan", "Try nitriles such as OCC#N")
    assert "Procedures that apply now (M4)" in ctx and "derived from 2 failure(s)" in ctx
    assert "proposed explanation" in ctx and "Guidance from the analysis agent" in ctx
    ev = [json.loads(f["payload"]) for _, f in s.r.xrange(s._k("events")) if f["event"] == "context"][-1]
    assert ev["ranks"] and ev["entries"]["M4"]


def test_on_demand_answer_is_logged_with_the_molecule_asked_about():
    cfg, s, _, a1 = nitrile_memory()
    text = client(s, cfg, "CCO#a2").answer("Have similar molecules failed?", "OCC#N")
    assert "failed (scf_not_converged)" in text and "Flowcept task" in text
    ev = [json.loads(f["payload"]) for _, f in s.r.xrange(s._k("events")) if f["event"] == "memory_query"][-1]
    assert ev["molecules"] == ["N#CCO"] and ev["entries"]["M2"]


# --------------------------------------------------------------------------------------------- analysis: failures
def test_failures_sharing_a_motif_give_one_procedure_per_motif():
    cfg, s, an, a1 = nitrile_memory()
    rec = diagnoses(s, "recurring_failure")
    assert [d.key for d in rec] == ["scf_not_converged|nitrile"] and s.motifs_of(rec[0].id) == ["nitrile"]
    expl = diagnoses(s, "failure_explanation")[0]
    assert expl.data["epistemic_status"] == "explanation"
    assert {e["origin"] for e in s.graph.outgoing(expl.id, ["EXPLAINS"])} == {"llm"}
    p = s.all("M4")[0]
    assert p.data["motif"] == "nitrile" and len(s.graph.outgoing(p.id, ["DERIVED_FROM"])) == 2
    for m in ("CC(F)(F)F", "OCC(F)(F)F"):                       # the failing motif changes
        fail(a1, m)
    drain(an)
    assert sorted(e.data["motif"] for e in s.all("M4")) == ["nitrile", "trifluoromethyl"]


def test_an_existing_procedure_is_applied_once_to_an_agent_that_meets_the_pattern():
    cfg, s, an, a1 = nitrile_memory()
    a2 = client(s, cfg, "CCN#a2")
    fail(a2, "CCC#N")
    fail(a2, "CCCC#N")
    drain(an)
    to_a2 = [g for g in s.all_guidance() if g.target == a2.agent]
    assert len(to_a2) == 1 and to_a2[0].check["motif"] == "nitrile"
    assert s.graph.outgoing(to_a2[0].id, ["APPLIES"])[0]["target"] == s.all("M4")[0].id


def test_failures_without_a_shared_motif_give_a_pattern_for_the_class():
    cfg, s, an = setup()
    a1 = client(s, cfg, "a1")
    fail(a1, "C1CC", "Parse failure for C1CC")
    fail(a1, "C1CCN", "Parse failure for C1CCN")
    drain(an)
    assert [d.key for d in diagnoses(s, "recurring_failure")] == ["invalid_smiles|*"]
    assert s.all("M4")[0].data["motif"] is None
    q = retrieval.build_query("", [], {"invalid_smiles"})
    assert retrieval.applicable(s.all("M4")[0], q, s)


def test_repeated_invalid_smiles_failure_has_a_cause_confirmed_by_rdkit():
    cfg, s, an = setup()
    a1 = client(s, cfg, "a1")
    fail(a1, "C1CC", "Parse failure for C1CC")
    fail(a1, "C1CC", "Parse failure for C1CC")
    drain(an)
    d = diagnoses(s, "repeated_failure")[0]
    assert d.data["epistemic_status"] == "explanation" and s.get_verification(d.id)["verdict"] == "supported"
    assert {e["origin"] for e in s.graph.outgoing(d.id, ["SUPPORTED_BY"])} == {"deterministic"}


def test_analysis_work_is_not_lost_when_the_llm_is_unavailable():
    cfg, s, an = setup(llm=NoLLM())
    a1 = client(s, cfg, "a1")
    fail(a1, "CC#N")
    fail(a1, "NCC#N")
    drain(an)
    assert not diagnoses(s, "recurring_failure")                 # left pending: nothing half-written
    an.llm = AnalysisLLM("scripted")
    drain(an)
    assert diagnoses(s, "recurring_failure") and s.all("M4")


# --------------------------------------------------------------------------------------------- analysis: impact
def unreliable_chain(depth=3):
    """A result that turns out unreliable, a statement it supports, a procedure justified by the statement."""
    cfg, s, an = setup(analysis_depth=depth)
    a1 = client(s, cfg, "CCO#a1")
    for m in ("CC#N", "NCC#N"):
        fail(a1, m)
    drain(an)
    proc = s.all("M4")[0]
    k, _ = s.add(knowledge("Nitriles do not converge with xTB", "observation", "a1"))
    s.link(proc.id, k, "JUSTIFIED_BY", "analysis-agent", "llm")
    a1.record_result("OCC#N", 7.0, {**COND, "converged": False, "opt_steps": 3}, "t9", "w1", "c1")
    s.link(k, s.by_key("M1", "N#CCO")[0].id, "SUPPORTED_BY", "a1", "llm")
    r2, _ = s.add(result("CCCN", "ionization_energy", 8.0, "eV", OK, "a1", ProvRef(task_id="t10")))
    s.link(k, r2, "CONTRADICTED_BY", "verification")
    drain(an)
    return s, k, proc


def test_an_unreliable_result_reaches_what_rests_on_it_through_the_graph():
    s, k, proc = unreliable_chain()
    d = diagnoses(s, "invalid_artifact")[0]
    affected = {a["node"]: a for a in d.data["context"]["affected"]}
    assert affected[k]["depth"] == 1 and affected[proc.id]["depth"] == 2
    assert affected[proc.id]["via"] == ["SUPPORTED_BY", "JUSTIFIED_BY"]
    g = [g for g in s.all_guidance() if "unreliable" in g.message][0]
    assert "directly" in g.message and "indirectly" in g.message
    assert "WARNING, unreliable" in retrieval.describe(s, s.by_key("M1", "N#CCO")[0])


def test_direct_impact_ablation_stops_at_one_hop():
    s, k, proc = unreliable_chain(depth=1)
    affected = {a["node"] for a in diagnoses(s, "invalid_artifact")[0].data["context"]["affected"]}
    assert k in affected and proc.id not in affected


def test_statement_resting_on_an_unreliable_result_is_flagged_to_its_author():
    cfg, s, an = setup()
    a1 = client(s, cfg, "CCO#a1")
    a1.record_result("CCO", 7.0, {**COND, "converged": False, "opt_steps": 3}, "t1", "w1", "c1")
    drain(an)
    conclude(a1, [{"statement": "CCO is promising", "epistemic_status": "observation",
                   "supporting_molecules": ["CCO"]}])
    drain(an)
    assert diagnoses(s, "knowledge_on_unreliable_result")
    assert any(g.target == a1.agent and "reassess" in g.message for g in s.all_guidance())


# --------------------------------------------------------------------------------------------- verdicts
def test_claim_checks_are_deterministic_on_reliable_results():
    cfg, s, _ = setup()
    a1 = client(s, cfg, "a1")
    vals = {"CC(F)(F)F": 15.0, "CCC(F)(F)F": 14.8, "CCC": 11.0, "CCCC": 10.5, "CCO": 9.0}
    for smi, val in vals.items():
        a1.record_result(smi, val, OK, f"t-{smi}", "w1", "c1")
    a1.record_result("CCCO", 3.0, {**COND, "converged": False}, "t-x", "w1", "c1")
    good, bad = verification.results(s)
    v = lambda c: verification.check(verification.normalize(c), good, bad)["verdict"]  # noqa: E731
    assert v({"form": "value", "molecule": "CCO", "value": 9.02}) == "supported"
    assert v({"form": "bound", "molecule": "CCC", "relation": "below", "value": 10}) == "contradicted"
    assert v({"form": "comparison", "a": "CC(F)(F)F", "relation": "higher", "b": "CCO"}) == "supported"
    assert v({"form": "motif_effect", "feature": "trifluoromethyl", "direction": "increases"}) == "supported"
    assert v({"form": "value", "molecule": "CCCO", "value": 3.0}) == "inconclusive"     # unreliable: never decided
    assert v({"form": "value", "molecule": "CCCCCO", "value": 3.0}) == "inconclusive"   # missing: not contradicted
    assert v({"form": "none"}) == "inconclusive"


def test_a_contradicted_statement_is_flagged_rechecked_and_its_procedures_named():
    cfg, s, an = setup()
    a1 = client(s, cfg, "CCO#a1")
    a1.record_result("CCO", 9.5, OK, "t1", "w1", "c1")
    conclude(a1, [{"statement": "The ionization energy of CCCO is above 9.0 eV", "epistemic_status": "observation"}])
    drain(an)
    kid = s.all("M3")[0].id
    assert s.get_verification(kid)["verdict"] == "inconclusive"
    p, _ = s.add(__import__("pgmem.schema", fromlist=["procedure"]).procedure(
        "Prefer alcohols", "t", "p", "a", "e", "l", "analysis-agent", "x|*", "x", None))
    s.link(p, kid, "JUSTIFIED_BY", "analysis-agent", "llm")
    a1.record_result("CCCO", 8.0, OK, "t2", "w1", "c1")
    drain(an)
    rec = s.get_verification(kid)
    assert rec["verdict"] == "contradicted" and len(rec["history"]) == 2
    det = s.graph.outgoing(kid, ["CONTRADICTED_BY"])
    assert det and det[0]["origin"] == "deterministic" and det[0]["created_by"] == "verification"
    assert diagnoses(s, "claim_contradicted_by_new_result")
    assert any("procedures rest on it" in g.message for g in s.all_guidance())
    a1.record_result("CCCO", 9.6, {**OK, "fmax": 0.01}, "t3", "w1", "c1")   # another measurement: checked again
    drain(an)
    assert len(s.graph.outgoing(kid, ["CONTRADICTED_BY"])) == 1               # earlier check's edges replaced


# --------------------------------------------------------------------------------------------- analysis: guidance
def test_guidance_is_judged_once_on_the_next_proposals_and_their_outcome():
    cfg, s, an, a1 = nitrile_memory()
    a2 = client(s, cfg, "CCN#a2")
    a2.context("plan", "next molecules")                       # guidance delivered here
    conclude(a2, proposals=["CCCN"])
    a2.record_result("CCCN", 8.5, OK, "t-a2", "w1", "c1")
    conclude(a2, proposals=["CCCCN"])
    drain(an)
    judged = [e["props"] for g in s.all_guidance() for e in s.graph.outgoing(g.id, ["APPLIED_IN"])]
    assert judged and all(j["agent"] == a2.agent for j in judged)
    assert len(judged) == len({g.id for g in s.all_guidance() if g.target in ("*", a2.agent)})   # once each
    motif = [j for g in s.all_guidance() if g.check.get("motif") for e in s.graph.outgoing(g.id, ["APPLIED_IN"])
             for j in [e["props"]]]
    assert motif and motif[0]["applied"] is True and motif[0]["effective"] is True


def test_a_procedure_that_stops_working_is_diagnosed_and_its_justification_named():
    cfg, s, an, a1 = nitrile_memory()
    proc = s.all("M4")[0]
    k, _ = s.add(knowledge("Nitriles fail with xTB", "observation", "a1"))
    s.link(proc.id, k, "JUSTIFIED_BY", "analysis-agent", "llm")
    for name, nxt, refail in (("CCN#a2", "CCO", None), ("CCS#a3", "CCCO", "CCC(F)(F)F")):
        c = client(s, cfg, name)                 # the procedure is given to each agent that meets the pattern
        fail(c, "CC" + "C" * len(name) + "C#N")
        drain(an)
        c.context("plan", "next")
        c.record_proposals([nxt] + ([refail] if refail else []))
        if refail:                               # no nitrile proposed, but the failures go on (the motif changed)
            fail(c, refail)
        conclude(c)
        drain(an)
    apps = [e["props"] for g in s.graph.incoming(proc.id, ["APPLIES"]) for e in s.graph.outgoing(g["source"],
                                                                                                   ["APPLIED_IN"])]
    assert [(a["applied"], a["effective"]) for a in apps] == [(True, True), (True, False)], apps
    assert diagnoses(s, "procedure_no_longer_effective")
    assert any("may be outdated" in g.message and "Nitriles fail with xTB" in g.message for g in s.all_guidance())
def test_two_iterations_without_improvement_get_advice():
    cfg, s, an = setup()
    a1 = client(s, cfg, "a1")
    a1.record_result("CCO", 9.5, OK, "t1", "w1", "c1")
    conclude(a1, proposals=["CCO"])
    for _ in range(2):
        conclude(a1, proposals=["CCO"])
    drain(an)
    assert diagnoses(s, "no_improvement") and any(g.target == "a1" for g in s.all_guidance())


def test_repeated_proposals_are_judged_on_what_was_known_when_proposing():
    cfg, s, an = setup()
    a1 = client(s, cfg, "a1")
    a1.record_proposals(["CCO"])
    a1.record_result("CCO", 9.5, OK, "t1", "w1", "c1")
    drain(an)
    assert not diagnoses(s, "repeated_proposals")              # its own later result does not count
    client(s, cfg, "a2").record_proposals(["OCC"])
    drain(an)
    assert diagnoses(s, "repeated_proposals")


def test_analysis_without_the_memory_source_keeps_only_the_judgement():
    cfg, s, an, a1 = nitrile_memory()
    an.cfg.analysis_sources = ("provenance",)
    n = len(diagnoses(s))
    a2 = client(s, cfg, "CCN#a2")
    fail(a2, "CCC#N")
    a2.context("plan", "x")
    conclude(a2, proposals=["CCO"])
    drain(an)
    assert len(diagnoses(s)) == n and any(s.graph.outgoing(g.id, ["APPLIED_IN"]) for g in s.all_guidance())


# --------------------------------------------------------------------------------------------- provenance
def test_provenance_records_received_online_are_counted_but_not_the_post_reading():
    src = FakeSource()
    cfg, s, an = setup(source=src)
    src.pending = [{"type": "task", "subtype": "parsl_task", "task_id": f"t{i}", "status": "FINISHED",
                    "activity_id": "compute_ionization_energy", "used": {"smiles": "CCO"}} for i in range(3)]
    drain(an)
    an.count_provenance = False
    src.pending = [{"type": "task", "subtype": "parsl_task", "task_id": "t9", "status": "FINISHED",
                    "activity_id": "compute_ionization_energy", "used": {"smiles": "CCO"}}]
    drain(an)
    assert int(s.r.hget(s._k("analyzer", "stats"), "provenance_records")) == 3


def test_jsonl_provenance_on_a_real_gaied_file():
    p = JsonlProvenance(os.path.join(ROOT, "output", "flowcept_buffer_SC.jsonl"))
    new = p.poll()
    comps = p.index.computations()
    assert len(comps) == 16 and smiles_of(comps[0]) == "NC(N)=O"
    assert all(r.get("subtype") != "academy_action" for r in new)
    assert p.poll() == []


# --------------------------------------------------------------------------------------------- evaluation support
def test_faults_follow_the_file_rewritten_during_a_run_with_the_ordinary_message():
    from pgmem import faults
    log, spec = tempfile.mktemp(), tempfile.mktemp()
    json.dump({"phase": 0, "fail": {"smarts": "C#N"}}, open(spec, "w"))
    os.environ.update({"PGMEM_FAULTS_LOG": log, "PGMEM_FAULTS": spec})
    try:
        def run(smi):
            try:
                faults.before(smi, 10)
                return "ok"
            except RuntimeError as exc:
                return str(exc)
        first = [run("CC#N"), run("CC(F)(F)F")]
        json.dump({"phase": 1, "fail": {"smarts": "[CX4](F)(F)F"}}, open(spec, "w"))
        second = [run("CC#N"), run("CC(F)(F)F")]
        assert first == [SCF, "ok"] and second == ["ok", SCF], (first, second)
        assert [json.loads(x)["phase"] for x in open(log) if '"computation"' in x] == [0, 0, 1, 1]
    finally:
        os.environ.pop("PGMEM_FAULTS")
        os.environ.pop("PGMEM_FAULTS_LOG")


def test_fault_selection_is_deterministic_and_the_scenarios_hide_the_injection():
    from pgmem import faults
    from evaluation.config import CONDITIONS, SCENARIOS
    rule = {"fraction": 0.3, "salt": "x"}
    mols = ["C" * n + "O" for n in range(1, 41)]
    assert [m for m in mols if faults.selected(rule, m)] == [m for m in mols if faults.selected(rule, m)]
    assert faults.selected(rule, "OCC") == faults.selected(rule, "CCO")
    assert "injected" not in json.dumps(SCENARIOS).lower() and "corrupted_results" in SCENARIOS
    for c in CONDITIONS.values():                              # every ablation is a valid configuration
        old = dict(os.environ)
        os.environ.update(c["env"])
        try:
            Config.from_env()
        finally:
            os.environ.clear()
            os.environ.update(old)


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
