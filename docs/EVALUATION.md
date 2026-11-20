# Evaluation

Controlled evaluation of `pgmem` with GAIED as the case study. The design is in `evaluation/config.py`, the runner in
`evaluation/run_series.py`, the metrics in `evaluation/metrics.py`.

## Conditions

What each condition changes, exactly:

| | `off` | `basic` | `structured` | `full` |
|---|---|---|---|---|
| Results shared and reused (M1), per-molecule lock | | yes | yes | yes |
| Team's measured molecules listed at experiment selection | | yes | yes, with failures | yes, with failures |
| Experiences, knowledge, procedures written (M2-M4) | | | yes | yes |
| Task-aware memory sections in `plan`, `critique`, `update` | | | yes | yes |
| `query_memory` tool at experiment selection | | | yes | yes |
| Analysis agent (diagnoses, claim verification, guidance) | | | | yes |

`structured` and `full` differ only by the analysis agent and its guidance.

Ablations of `full`:

| Ablation | What is removed or changed |
|---|---|
| `full-no-M3` | knowledge left out of the prompts and answers |
| `full-no-M2-M4` | experiences and procedures left out of the prompts and answers |
| `full-memory-only` | the analysis agent does not read provenance |
| `full-provenance-only` | the analysis agent does not analyze memory events (guidance tracking is kept, so interventions are still measured) |
| `full-scheduled-only` | no `query_memory` tool: memory only at fixed points |
| `full-recent-retrieval` | the earlier retrieval: best results and keyword search instead of task-aware retrieval |
| `full-failure-reuse` | a molecule that failed is not computed again (a cache of negative results, kept apart from `full`) |

## Scenarios

Two agents per campaign. Faults are injected by `pgmem/faults.py`; the failure message is xTB's ordinary one and
nothing about the injection reaches the provenance or the memory, so agents and the analysis agent can only discover
the patterns. Each computation and each fault is logged as ground truth.

| Scenario | Seeds | Iterations | Injected | Specification's situations |
|---|---|---|---|---|
| `motif_failures` | different | 5 | molecules with a nitrile (`C#N`) fail; once the agents have completed 4 iterations in total, molecules with a CF3 group fail instead (a molecule with both fails in both phases) | repeated technical failures; procedures that become ineffective |
| `invalid_artifacts` | different | 3 | 30 % of the optimizations are cut after 3 steps (not converged) | invalid or incomplete artifacts |
| `contradictory_results` | different | 3 | 30 % of the values are shifted by -3 eV (same molecule, same conditions, wrong value) | contradictory scientific results |
| `redundant_computations` | the same seed for both agents | 3 | nothing | redundant computations |

Inefficient exploration (proposals of molecules already known) is measured in every run. The switch of
`motif_failures` happens after the same number of agent iterations in every condition (the same decision budget),
not after a number of computations, since the memory changes how many computations are run. The fractions, offset,
motifs and switch point are parameters of the design.

## Paired runs

Every condition of a repetition uses the same LLM sampling seed (`LLM_SEED`, 1000 + repetition), the same scenario,
code version, model, prompts and generation parameters. Prompts differ between conditions by design, so the runs
diverge, but they start from the same state.

## Metrics

Ground truth comes from Flowcept's provenance and the fault log, never from the memory or the analysis agent.

| Metric (specification) | Measure |
|---|---|
| Task completion | campaign ended normally; iterations completed; valid results and distinct valid molecules; best ionization energy (true value; secondary, since agents explore different molecules) |
| Redundant computation reduction | xTB computations of a molecule beyond its first one |
| Inefficient exploration | proposals of a molecule whose computation had already ended; proposals of molecules the memory had just shown the agent |
| Repeated technical failures | failed computations of a molecule that had already failed |
| Learning from failures and adaptation (`motif_failures`) | share of proposals carrying the failing motif in the first and second half of each phase; failures per phase; share of the old motif after the change; computations until the analysis agent reports the new pattern; deliveries of old failure guidance after the change |
| Diagnostic accuracy | precision and recall of each rule's findings against its ground-truth set; correctness of `confirmed_cause` findings. For deterministic rules these mostly verify the implementation |
| Provenance traceability | share of the memory's references to Flowcept tasks that resolve to a recorded task |
| Memory retrieval relevance | share of retrieved entries relevant to the task (scheduled sections and on-demand answers separately). An entry is relevant when it is about one of the task's molecules, a similar molecule (Tanimoto >= 0.4), or shares a specific functional group with them. A structural criterion, not an expert judgement |
| Memory use | questions asked to the memory; decisions preceded by a question; re-proposals of molecules just shown |
| Procedure applicability | procedures given to agents, applied, effective |
| Intervention effectiveness | guidance delivered, applied, not applied, effective, not effective, overall and by the rule that produced it; and the outcome measures compared between conditions |
| Scientific knowledge maintenance | statements resting on a corrupted result and the share flagged; claim verification: statements the LLM could make checkable, verdicts, statements contradicted by the true values, verdicts changed by corrupted results |
| System overhead | campaign duration; LLM calls of the agents and of the analysis agent; analysis time; memory size |

`summary.md` gives, per scenario and condition, the mean ± standard deviation and, for counts, the 95 % t interval of
the mean and the range. With 5 repetitions these are small samples: differences should be read as indications.

## Series

| Series | LLM | Runs | Purpose |
|---|---|---|---|
| `smoke` | scripted | 4 scenarios × 11 conditions × 1 = 44 | check that everything runs (under a minute per run) |
| `pilot2` | Ollama model | 4 × 4 × 1 = 16 | check the revised scenarios with the real model |
| `main` | Ollama model | 4 × 4 × 5 = 80 | the paper's comparison |
| `ablations` | Ollama model | 4 × 7 × 2 = 56 | contribution of memory categories, on-demand access, task-aware retrieval, cross-source analysis, failure reuse |

## Running

From `GAIED/docker`, with Redis started (`docker compose up -d redis`), the evaluation version checked out, and the
model of your setup (here a model served by Ollama on the host):

```bash
docker compose run --rm --no-deps -e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=qwen3.5:9b -e LLM_MODEL=qwen3.5:9b gaied python -m evaluation.run_series main
docker compose run --rm --no-deps gaied python -m evaluation.metrics results/eval/main
```

The scripted check needs no model: `... gaied python -m evaluation.run_series smoke`. It checks the plumbing, not the
science (the scripted model always proposes the same few molecules). A series can be interrupted and started again:
finished runs are skipped. `--only` and `--scenarios` restrict a series.

Each run leaves its files in `results/eval/<series>/<scenario>__<condition>__r<n>/`; `metrics` writes `runs.csv`
(one row per run), `summary.csv` and `summary.md`.
