"""
Tests of the graph of relations (pgmem/graph.py) on a real Redis server: creation, lookup in both directions,
traversal, filtering, duplicates, missing nodes, deletion. Run: python tests/test_graph.py
"""
from __future__ import annotations

import os
import sys
import time
import traceback

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]

from pgmem.schema import ProvRef, failure, knowledge, procedure, result  # noqa: E402
from pgmem.store import MemoryStore  # noqa: E402

HOST, PORT = os.environ.get("REDIS_HOST", "localhost"), int(os.environ.get("REDIS_PORT", "6379"))
COND = {"method": "GFN2-xTB", "converged": True}


def store() -> MemoryStore:
    return MemoryStore.connect(HOST, PORT, f"test-graph-{time.time_ns()}")


def chain(s: MemoryStore):
    """result <- statement <- procedure (a dependency chain), and a failure the procedure was derived from."""
    r, _ = s.add(result("CC#N", "ionization_energy", 12.0, "eV", COND, "a1", ProvRef(task_id="t1")))
    k, _ = s.add(knowledge("Nitriles have a high ionization energy", "hypothesis", "a1"))
    f, _ = s.add(failure("NCC#N", "scf_not_converged", "SCF not converged", "a1", 0, "NCC#N"))
    p, _ = s.add(procedure("Avoid nitriles", "t", "p", "a", "e", "l", "analysis-agent", "scf_not_converged|nitrile",
                           "scf_not_converged", "nitrile"))
    s.link(k, r, "SUPPORTED_BY", "a1", "llm")
    s.link(p, k, "JUSTIFIED_BY", "analysis-agent", "llm")
    s.link(p, f, "DERIVED_FROM", "analysis-agent")
    return r, k, f, p


def test_edges_are_found_from_both_ends_and_filtered_by_type():
    s = store()
    r, k, f, p = chain(s)
    out = s.graph.outgoing(p)
    assert {e["type"] for e in out} == {"JUSTIFIED_BY", "DERIVED_FROM"}
    assert [e["source"] for e in s.graph.incoming(k)] == [p]
    assert [e["target"] for e in s.graph.outgoing(p, ["DERIVED_FROM"])] == [f]
    both = {other for _, other in s.graph.neighbors(k, "both")}
    assert both == {r, p}
    e = s.graph.outgoing(k)[0]
    assert e["origin"] == "llm" and e["created_by"] == "a1" and e["created_at"] > 0 and e["id"].startswith("e:")


def test_multi_hop_traversal_is_bounded_and_keeps_the_path():
    s = store()
    r, k, f, p = chain(s)
    reach = {x.node: x for x in s.graph.dependents(r)}
    assert set(reach) == {k, p} and reach[p].depth == 2
    assert [e["type"] for e in reach[p].path] == ["SUPPORTED_BY", "JUSTIFIED_BY"]
    assert {x.node for x in s.graph.dependents(r, max_depth=1)} == {k}
    assert len(s.graph.traverse(p, "both", None, 3, max_nodes=1)) == 1


def test_contradiction_and_association_are_not_dependencies():
    s = store()
    r, k, _, _ = chain(s)
    r2, _ = s.add(result("CCC#N", "ionization_energy", 8.0, "eV", COND, "a1", ProvRef(task_id="t2")))
    s.link(k, r2, "CONTRADICTED_BY", "verification")
    s.link(r2, s.molecule("CCC#N"), "ABOUT", "a1")
    assert s.graph.dependents(r2) == []                  # what contradicts a statement is not something it rests on
    assert {x.node for x in s.graph.dependents(r)} >= {k}


def test_duplicates_are_refused_and_origin_distinguishes_two_edges():
    s = store()
    r, k, _, _ = chain(s)
    eid, created = s.link(k, r, "SUPPORTED_BY", "a1", "llm")
    assert not created and eid == s.graph.find(k, "SUPPORTED_BY", r, "llm")["id"]
    _, created = s.link(k, r, "SUPPORTED_BY", "verification", "deterministic")   # a check confirming the citation
    assert created and len(s.graph.outgoing(k, ["SUPPORTED_BY"])) == 2


def test_missing_nodes_wrong_types_and_unknown_relations_are_refused():
    s = store()
    r, k, _, _ = chain(s)
    for args, exc in [((k, "m1:000000000000", "SUPPORTED_BY", "x"), KeyError),
                      ((r, k, "SUPPORTED_BY", "x"), ValueError),           # a result does not rest on a statement
                      ((k, r, "LIKES", "x"), ValueError),
                      ((k, r, "SUPPORTED_BY", "x", "guess"), ValueError)]:
        try:
            s.link(*args)
            raise AssertionError(f"accepted {args}")
        except exc:
            pass


def test_deletion_leaves_consistent_indexes():
    s = store()
    r, k, _, p = chain(s)
    e = s.graph.outgoing(k)[0]
    assert s.graph.remove(e["id"]) and not s.graph.remove(e["id"])
    assert s.graph.outgoing(k) == [] and all(x["source"] != k for x in s.graph.incoming(r))
    assert s.graph.find(k, "SUPPORTED_BY", r, "llm") is None and e["id"] not in {x["id"] for x in s.graph.all()}
    _, created = s.link(k, r, "SUPPORTED_BY", "a1", "llm")                  # can be created again
    assert created


def test_molecule_nodes_carry_their_specific_motifs():
    s = store()
    node = s.molecule("NC(=O)C(F)(F)F")
    motifs = {e["target"] for e in s.graph.outgoing(node, ["HAS_MOTIF"])}
    assert motifs == {"motif:trifluoromethyl", "motif:amide"}, motifs     # generic groups (carbonyl...) left out
    s.molecule("NC(=O)C(F)(F)F")
    assert len(s.graph.outgoing(node)) == 2                                 # created once
    assert s.exists(node) and s.exists("motif:amide") and not s.exists("mol:CCO")


def test_entries_hold_no_relations():
    s = store()
    r, k, f, p = chain(s)
    for i in (r, k, f, p):
        d = s.get(i).to_dict()
        assert "refs" not in d and not any(isinstance(v, str) and v.startswith(("m1:", "m2:", "m3:", "m4:"))
                                           for v in d["data"].values()), d


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
