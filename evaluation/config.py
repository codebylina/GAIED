"""
Evaluation design: conditions, scenarios and series (which runs to execute).

Conditions follow the specification: agents without shared memory or analysis (off), with basic shared memory (basic),
with the structured four-category memory (structured), and with structured memory plus the online cross-source
analysis agent (full). Ablations remove memory categories from the agents' prompts, or one of the analysis agent's
two sources.

Scenarios follow the specification's list. Faults are injected by pgmem/faults.py, so that a scenario is exercised
whatever molecules the LLM proposes. Technical failures follow a chemical motif (a SMARTS pattern), since failures of
random molecules leave no pattern to learn from; incomplete artifacts and corrupted values hit a deterministic fraction
of the molecules (hash of the canonical SMILES). The failure message is xTB's ordinary one. Motifs, fractions, offset
and iteration counts are parameters of the experimental design, stated here and in the results.
"""
from __future__ import annotations

DIFFERENT_SEEDS = "CNC(N)=O,CC1=C(O)N=C(O)N1"     # GAIED's default seeds: two agents exploring different regions
SAME_SEEDS = "CNC(N)=O,CNC(N)=O"                  # two agents starting from the same molecule

NITRILE, CF3 = "C#N", "[CX4](F)(F)F"

SCENARIOS: dict[str, dict] = {
    # every molecule with a nitrile fails
    "repeated_failures": {
        "seeds": DIFFERENT_SEEDS,
        "faults": {"fail": {"smarts": NITRILE, "message": "SCF not converged"}},
    },
    "redundant_computations": {"seeds": SAME_SEEDS, "faults": {}},
    "inefficient_exploration": {"seeds": DIFFERENT_SEEDS, "faults": {}},
    "invalid_artifacts": {
        "seeds": DIFFERENT_SEEDS,
        "faults": {"truncate": {"fraction": 0.3, "salt": "artifacts", "steps": 3}},
    },
    # wrong values (same molecule, same conditions, value shifted): with a memory a molecule is computed once, so no
    # second measurement contradicts it; what is tested is the maintenance of knowledge built on corrupted results
    "corrupted_results": {
        "seeds": DIFFERENT_SEEDS,
        "faults": {"offset": {"fraction": 0.3, "salt": "contradiction", "value": -3.0}},
    },
    # phase 1: nitriles fail, a pattern the analysis agent turns into a procedure; once that procedure has been
    # effective, molecules with a CF3 group fail instead (phase 2), so applying the procedure no longer prevents
    # failures. Five iterations leave room for both phases. The switch needs a procedure, so it can only happen in
    # conditions with the analysis agent.
    "ineffective_procedure": {
        "seeds": DIFFERENT_SEEDS,
        "max_iterations": 5,
        "faults": {"fail": {"smarts": NITRILE, "message": "SCF not converged"}},
        "switch_when_procedure_effective": {"fail": {"smarts": CF3, "message": "SCF not converged"}},
    },
}

CONDITIONS: dict[str, dict] = {
    "off": {"env": {"PGMEM_MODE": "off"}, "analyzer": False},
    "basic": {"env": {"PGMEM_MODE": "basic"}, "analyzer": False},
    "structured": {"env": {"PGMEM_MODE": "structured"}, "analyzer": False},
    "full": {"env": {"PGMEM_MODE": "full"}, "analyzer": True},
    # ablations of the full condition
    "full-no-M3": {"env": {"PGMEM_MODE": "full", "PGMEM_CONTEXT_CATEGORIES": "M1,M2,M4"}, "analyzer": True},
    "full-no-M2-M4": {"env": {"PGMEM_MODE": "full", "PGMEM_CONTEXT_CATEGORIES": "M1,M3"}, "analyzer": True},
    "full-memory-only": {"env": {"PGMEM_MODE": "full", "PGMEM_ANALYSIS_SOURCES": "memory"}, "analyzer": True},
    "full-provenance-only": {"env": {"PGMEM_MODE": "full", "PGMEM_ANALYSIS_SOURCES": "provenance"}, "analyzer": True},
}

MAIN_CONDITIONS = ["off", "basic", "structured", "full"]
ABLATIONS = ["full-no-M3", "full-no-M2-M4", "full-memory-only", "full-provenance-only"]

# llm: "scripted" (no model, minutes) or "ollama" (the model given by OLLAMA_URL / OLLAMA_MODEL in the environment)
SERIES: dict[str, dict] = {
    "smoke": {"llm": "scripted", "scenarios": list(SCENARIOS), "conditions": MAIN_CONDITIONS + ABLATIONS,
              "repetitions": 1, "max_iterations": 3},
    "pilot": {"llm": "ollama", "scenarios": list(SCENARIOS), "conditions": MAIN_CONDITIONS,
              "repetitions": 1, "max_iterations": 3},
    # the pilot again, after the fixes it led to (its own folder: the first pilot's runs are kept and not skipped)
    "pilot2": {"llm": "ollama", "scenarios": list(SCENARIOS), "conditions": MAIN_CONDITIONS,
               "repetitions": 1, "max_iterations": 3},
    "main": {"llm": "ollama", "scenarios": list(SCENARIOS), "conditions": MAIN_CONDITIONS,
             "repetitions": 3, "max_iterations": 3},
    "ablations": {"llm": "ollama", "scenarios": list(SCENARIOS), "conditions": ABLATIONS,
                  "repetitions": 1, "max_iterations": 3},
}
