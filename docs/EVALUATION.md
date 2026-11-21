# Evaluation

Controlled evaluation of `pgmem` with GAIED as the case study. The design is in `evaluation/config.py`, the runner in
`evaluation/run_series.py`, the metrics in `evaluation/metrics.py`.

## Conditions

| | `off` | `basic` | `structured` | `full` |
|---|---|---|---|---|
| Results shared and reused (M1), per-molecule lock | | yes | yes | yes |
| Team's measured molecules listed at experiment selection | | yes | yes, with failures | yes, with failures |
| Graph memory written by the agents (M2-M4, relations) | | | yes | yes |
| Retrieved memory sections in `plan`, `critique`, `update`; `query_memory` tool | | | yes | yes |
| Analysis agent (diagnoses, verdicts, procedures, guidance) | | | | yes |

`structured` and `full` differ only by the analysis agent.

Ablations of `full`:

| Ablation | What is removed or changed |
|---|---|
| `full-no-graph` | the graph channel of retrieval |
| `full-no-semantic` | the semantic (embedding) channel |
| `full-no-chem` | the chemical-similarity channel |
| `full-scheduled-only` | the `query_memory` tool: memory only at fixed points |
| `full-direct-impact` | the analysis agent follows what rests on a node one step only (`PGMEM_ANALYSIS_DEPTH=1`) |
| `full-memory-only` | the analysis agent does not read provenance |
| `full-provenance-only` | the analysis agent does not analyze memory events (the judgement of guidance is kept) |

## Scenarios

Two agents per campaign. Faults are injected by `pgmem/faults.py`; the failure message is xTB's ordinary one and
nothing about the injection reaches the provenance or the memory, so agents and the analysis agent can only discover
the patterns. Each computation and each fault is logged as ground truth.

| Scenario | Seeds | Iterations | Injected | Situations |
|---|---|---|---|---|
| `motif_failures` | different | 5 | molecules with a nitrile (`C#N`) fail; once the agents have completed 4 iterations in total, molecules with a CF3 group fail instead | repeated technical failures, learning from failures, procedures per motif, a procedure that becomes ineffective |
| `invalid_artifacts` | different | 3 | 30 % of the optimizations are cut after 3 steps (not converged) | invalid or incomplete artifacts and what rests on them |
| `corrupted_results` | different | 3 | 30 % of the values are shifted by -3 eV | maintenance of knowledge built on corrupted results |
| `redundant_computations` | the same seed for both agents | 3 | nothing | redundant computations |

`corrupted_results` is not a test of contradiction detection between measurements: with a memory a molecule is
computed once, so no second measurement contradicts the wrong one. What is measured is whether statements resting on
corrupted results are flagged and how the corrupted values change the verdicts (checked against the true values).
The switch of `motif_failures` happens after the same number of agent iterations in every condition (the same
decision budget). Inefficient exploration is measured in every run. Motifs, fractions, offset and switch point are
parameters of the design.

## Paired runs

Every condition of a repetition uses the same LLM sampling seed (`LLM_SEED`, 1000 + repetition), scenario, code
version, model, embedding model, prompts and generation parameters. Prompts differ between conditions by design, so
the runs diverge, but they start from the same state.

## Metrics

Ground truth comes from Flowcept's provenance and the fault log, never from the memory or the analysis agent.

| Metric | Measure |
|---|---|
| Task completion | campaign ended normally; iterations completed; valid results and distinct valid molecules; best ionization energy (true value; secondary) |
| Redundant computations | xTB computations of a molecule beyond its first |
| Inefficient exploration | proposals of a molecule whose computation had already ended; proposals of molecules the memory had just shown |
| Repeated technical failures | failed computations of a molecule that had already failed |
| Learning and adaptation (`motif_failures`) | failing-motif share of the proposals in each half of each phase; failures per phase; old motif's share after the change; computations until the change is diagnosed; old failure guidance delivered after the change |
| Diagnostic accuracy | precision and recall of each rule against its ground truth (redundant molecules, molecules failing twice, the injected failing motifs, non-converged results, corrupted results revealed by a contradiction, iterations with known proposals); correctness of causes confirmed by RDKit. For deterministic rules these mostly verify the implementation |
| Procedures | procedures and motifs; procedures justified by knowledge; judged, applied, effective, not effective; a procedure diagnosed as no longer effective |
| Interventions | guidance delivered, applied, not applied, effective, not effective (`APPLIED_IN` edges), overall and by the rule that produced it |
| Knowledge maintenance | statements resting on corrupted results and the share flagged; entries flagged through the graph (depth > 1); verdicts; statements contradicted by the true values; verdicts changed by corrupted results |
| Retrieval | share of retrieved entries relevant to the task (scheduled and on demand); entries found by each channel. An entry is relevant when it is about one of the task's molecules, a similar one (Tanimoto >= 0.4), or a specific motif of them. A structural criterion, not an expert judgement |
| Memory use | questions asked to the memory; decisions preceded by a question |
| Provenance | traceability (memory references resolving to a recorded task); coverage (records received online by the analysis agent / records recorded) |
| Overhead | campaign duration; LLM calls of the agents and of the analysis agent; analysis time; memory size and edges (and those of origin `llm`) |

`summary.md` gives, per scenario and condition, the mean ± standard deviation and, for counts, the 95 % t interval of
the mean and the range. With few repetitions these are small samples: differences are indications.

## Series

| Series | LLM | Runs | Purpose |
|---|---|---|---|
| `smoke` | scripted | 4 scenarios × 11 conditions × 1 = 44 | check that everything runs (under a minute per run) |
| `pilot` | Ollama model | 4 × 4 × 1 = 16 | check the scenarios with the real model |
| `main` | Ollama model | 4 × 4 × 5 = 80 | the paper's comparison |
| `ablations` | Ollama model | 4 × 7 × 2 = 56 | contribution of each retrieval channel, on-demand access, graph depth, cross-source analysis |

## Running

From `GAIED/docker`, with Redis started (`docker compose up -d redis`), the evaluation version checked out, and the
models of your setup (here served by Ollama on the host):

```bash
docker compose run --rm --no-deps -e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=qwen3.5:9b -e LLM_MODEL=qwen3.5:9b -e PGMEM_EMBED_MODEL=nomic-embed-text gaied python -m evaluation.run_series pilot
docker compose run --rm --no-deps gaied python -m evaluation.metrics results/eval/pilot
```

The scripted check needs no model: `... gaied python -m evaluation.run_series smoke`. It checks the plumbing, not the
science (the scripted model always proposes the same few molecules). A series can be interrupted and started again:
finished runs are skipped. `--only` and `--scenarios` restrict a series.

Each run leaves its files in `results/eval/<series>/<scenario>__<condition>__r<n>/`; `metrics` writes `runs.csv`
(one row per run), `summary.csv` and `summary.md`.
