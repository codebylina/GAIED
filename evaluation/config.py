"""
Evaluation design: scenarios, conditions and series (which runs to execute).

Four controlled scenarios cover the specification's situations (inefficient exploration and redundant computations
are measured in every run):

  motif_failures          repeated technical failures, then a change of context: molecules with a nitrile fail
                          until the agents have completed SWITCH_AFTER_ITERATIONS iterations in total (the same
                          decision budget in every condition), then those with a CF3 group fail instead (a molecule
                          with both fails in both phases). Learning from failures, procedures for a motif, and a
                          procedure that becomes ineffective.
  invalid_artifacts       30 % of the optimizations are cut after 3 steps (not converged); what rests on them
  corrupted_results       30 % of the values are shifted by -3 eV (same molecule, same conditions, a wrong value).
                          With a memory a molecule is computed once, so no second measurement contradicts the wrong
                          one: this tests the maintenance of knowledge built on corrupted results
  redundant_computations  two agents start from the same molecule

Faults are injected by pgmem/faults.py; the failure message is xTB's ordinary one and nothing about the injection
reaches the provenance or the memory, so the agents and the analysis agent can only discover the patterns.

Conditions: the specification's four (off, basic, structured, full) and ablations of full: each retrieval channel,
on-demand access, the depth of the analysis agent's graph traversal, and each of its two sources. Every condition of
a repetition uses the same LLM sampling seed (paired runs).
"""
from __future__ import annotations

DIFFERENT_SEEDS = "CNC(N)=O,CC1=C(O)N=C(O)N1"     # GAIED's default seeds: two agents exploring different regions
SAME_SEEDS = "CNC(N)=O,CNC(N)=O"                  # two agents starting from the same molecule
SWITCH_AFTER_ITERATIONS = 4                       # iterations of all agents before the failing motif changes
NITRILE, CF3 = "C#N", "[CX4](F)(F)F"

SCENARIOS: dict[str, dict] = {
    "motif_failures": {
        "seeds": DIFFERENT_SEEDS, "max_iterations": 5,
        "faults": {"phase": 0, "fail": {"smarts": NITRILE, "message": "SCF not converged"}},
        "switch": {"after_iterations": SWITCH_AFTER_ITERATIONS,
                   "faults": {"phase": 1, "fail": {"smarts": CF3, "message": "SCF not converged"}}},
        "motifs": [NITRILE, CF3],
    },
    "invalid_artifacts": {
        "seeds": DIFFERENT_SEEDS,
        "faults": {"truncate": {"fraction": 0.3, "salt": "artifacts", "steps": 3}},
    },
    "corrupted_results": {
        "seeds": DIFFERENT_SEEDS,
        "faults": {"offset": {"fraction": 0.3, "salt": "contradiction", "value": -3.0}},
    },
    "redundant_computations": {"seeds": SAME_SEEDS, "faults": {}},
}

CONDITIONS: dict[str, dict] = {
    # what each condition changes (docs/EVALUATION.md)
    "off": {"env": {"PGMEM_MODE": "off"}, "analyzer": False},
    "basic": {"env": {"PGMEM_MODE": "basic"}, "analyzer": False},
    "structured": {"env": {"PGMEM_MODE": "structured"}, "analyzer": False},
    "full": {"env": {"PGMEM_MODE": "full"}, "analyzer": True},
    # ablations of full
    "full-no-graph": {"env": {"PGMEM_MODE": "full", "PGMEM_CHANNELS": "bm25,semantic,chem"}, "analyzer": True},
    "full-no-semantic": {"env": {"PGMEM_MODE": "full", "PGMEM_CHANNELS": "bm25,chem,graph"}, "analyzer": True},
    "full-no-chem": {"env": {"PGMEM_MODE": "full", "PGMEM_CHANNELS": "bm25,semantic,graph"}, "analyzer": True},
    "full-scheduled-only": {"env": {"PGMEM_MODE": "full", "PGMEM_ON_DEMAND": "false"}, "analyzer": True},
    "full-direct-impact": {"env": {"PGMEM_MODE": "full", "PGMEM_ANALYSIS_DEPTH": "1"}, "analyzer": True},
    "full-memory-only": {"env": {"PGMEM_MODE": "full", "PGMEM_ANALYSIS_SOURCES": "memory"}, "analyzer": True},
    "full-provenance-only": {"env": {"PGMEM_MODE": "full", "PGMEM_ANALYSIS_SOURCES": "provenance"}, "analyzer": True},
}

MAIN_CONDITIONS = ["off", "basic", "structured", "full"]
ABLATIONS = [c for c in CONDITIONS if c not in MAIN_CONDITIONS]

# llm: "scripted" (no model, minutes) or "ollama" (the model given by OLLAMA_URL / OLLAMA_MODEL in the environment);
# max_iterations applies unless a scenario sets its own
SERIES: dict[str, dict] = {
    "smoke": {"llm": "scripted", "scenarios": list(SCENARIOS), "conditions": MAIN_CONDITIONS + ABLATIONS,
              "repetitions": 1, "max_iterations": 3},
    "pilot": {"llm": "ollama", "scenarios": list(SCENARIOS), "conditions": MAIN_CONDITIONS,
              "repetitions": 1, "max_iterations": 3},
    "main": {"llm": "ollama", "scenarios": list(SCENARIOS), "conditions": MAIN_CONDITIONS,
             "repetitions": 5, "max_iterations": 3},
    "ablations": {"llm": "ollama", "scenarios": list(SCENARIOS), "conditions": ABLATIONS,
                  "repetitions": 2, "max_iterations": 3},
}
