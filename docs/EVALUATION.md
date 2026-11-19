# Evaluation

Controlled evaluation of `pgmem` with GAIED as the case study. The design is in `evaluation/config.py`, the runner in
`evaluation/run_series.py`, the metrics in `evaluation/metrics.py`.

## Conditions

| Condition | `PGMEM_MODE` | Analysis agent | Meaning |
|---|---|---|---|
| `off` | `off` | no | original GAIED: no shared memory, no provenance-based analysis |
| `basic` | `basic` | no | basic shared memory: results shared and reused (M1), per-molecule lock |
| `structured` | `structured` | no | structured four-category memory (M1-M4) read and written by the agents |
| `full` | `full` | yes | structured memory and online cross-source analysis with guidance |

Ablations of `full`: `full-no-M3` (knowledge left out of the prompts), `full-no-M2-M4` (experiences and procedures left
out of the prompts), `full-memory-only` (the analysis agent does not read provenance), `full-provenance-only` (the
analysis agent does not analyze memory events; guidance tracking is kept so that interventions are still measured).

## Scenarios

Two agents, three iterations each. Faults are injected by `pgmem/faults.py` on a deterministic 30 % of the
molecules (a hash of the canonical SMILES), whatever molecules the LLM proposes; the injected faults are logged as
ground truth and never written to provenance or memory.

| Scenario | Seeds | Injected |
|---|---|---|
| `repeated_failures` | different | 30 % of the computations fail with "SCF not converged" |
| `redundant_computations` | the same seed for both agents | nothing |
| `inefficient_exploration` | different | nothing |
| `invalid_artifacts` | different | 30 % of the optimizations are cut after 3 steps (not converged) |
| `contradictory_results` | different | 30 % of the values are shifted by -3 eV |
| `ineffective_procedure` | different | as `repeated_failures`; once a procedure has been effective, the failing molecules change |

The fractions, the offset and the switch rule are parameters of the experimental design and are reported with the
results. `ineffective_procedure` only differs from `repeated_failures` where procedures exist (conditions with the
analysis agent).

## Metrics

Ground truth comes from Flowcept's provenance and from the fault log, never from the memory or from the analysis
agent being evaluated.

| Metric (specification) | Measure |
|---|---|
| Task completion | campaign ended normally; iterations completed; best value found (true value: offsets removed) |
| Redundant computation reduction | xTB computations of a molecule beyond its first one |
| Inefficient exploration | proposals of a molecule whose computation had already ended |
| Repeated technical failures | failed computations of a molecule that had already failed |
| Diagnostic accuracy | precision and recall of each rule's findings against its ground-truth set (redundant molecules, molecules failing twice, failure classes on several molecules, non-converged results, offset results, iterations with already-known proposals); correctness of `confirmed_cause` findings (RDKit cannot read the SMILES); detection of the ineffective procedure |
| Provenance traceability | share of the memory's references to Flowcept tasks that resolve to a recorded task |
| Procedure applicability | procedures given to agents, applied, effective |
| Intervention effectiveness | guidance delivered, applied, effective; and the outcome measures above compared between conditions |
| Scientific knowledge maintenance | statements resting on a corrupted result (truncated or offset) and the share of them flagged by the analysis agent |
| Memory retrieval relevance | **not yet defined**: what each prompt received is logged (`context` events, per category); a relevance criterion must be chosen before it is reported |
| System overhead | campaign duration; LLM calls of the agents and of the analysis agent; analysis time; memory size |

## Series

| Series | LLM | Runs | Purpose |
|---|---|---|---|
| `smoke` | scripted | 6 scenarios × 8 conditions × 1 | check that everything runs (about 45 s per run) |
| `pilot` | Ollama model | 6 × 4 × 1 = 24 | first real measurements, estimate of variance and duration |
| `main` | Ollama model | 6 × 4 × 3 = 72 | the paper's comparison |
| `ablations` | Ollama model | 6 × 4 × 1 = 24 | contribution of memory categories and of cross-source analysis |

## Running

From `GAIED/docker`, with Redis started (`docker compose up -d redis`), the evaluation version checked out, and the
model of your setup (here a model served by Ollama on the host):

```bash
docker compose run --rm --no-deps -e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=qwen3.5:9b -e LLM_MODEL=qwen3.5:9b gaied python -m evaluation.run_series pilot
docker compose run --rm --no-deps gaied python -m evaluation.metrics results/eval/pilot
```

The scripted check needs no model: `... gaied python -m evaluation.run_series smoke`. A series can be interrupted and
started again: finished runs are skipped. `--only` and `--scenarios` restrict a series.

Each run leaves its files in `results/eval/<series>/<scenario>__<condition>__r<n>/`; `metrics` writes `runs.csv`
(one row per run), `summary.csv` and `summary.md` (mean ± standard deviation per scenario and condition). Only
measured outcomes are reported.
