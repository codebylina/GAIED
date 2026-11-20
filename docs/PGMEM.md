# pgmem: provenance-grounded shared memory and analysis agent for GAIED

`pgmem` adds to GAIED's molecular-design agents (Academy + LangGraph + xTB, provenance captured by Flowcept):

1. one shared, persistent memory with four logical categories (M1-M4) whose entries **reference** Flowcept records
   instead of copying them;
2. an independent **analysis agent** that analyzes provenance and memory together, online and after execution, and
   sends guidance to the scientific agents;
3. the integration of both into GAIED's LangGraph workflow, with four comparison conditions (`PGMEM_MODE`).

Flowcept remains the single source of detailed provenance. Nothing in `pgmem` changes what Flowcept records, except
that the xTB conditions and the agent / node of each LLM call are added to existing records.

## Components

| Component | File | Role |
|---|---|---|
| Memory model | `pgmem/schema.py` | Entry envelope, M1-M4 constructors, guidance record, validation status of procedures |
| Memory store | `pgmem/store.py` | Redis keyspace `pgmem:{namespace}:...`, deduplication, history, event stream, guidance |
| Agent integration | `pgmem/agent_client.py`, `[pgmem]` lines of `mol_design_agents.py` | What an agent reads and writes at each workflow node; the `query_memory` tool; the per-molecule lock |
| Retrieval | `pgmem/retrieval.py` | Task-aware hybrid ranking (molecular structure, functional groups, words, failure class); applicability of procedures |
| Provenance adapter | `pgmem/provenance.py` | Reads Flowcept (JSONL files, or the online Redis channel); provenance traversal |
| Analysis agent | `pgmem/analyzer.py`, `pgmem/llm.py` | Deterministic rules, LLM interpretation labelled as candidate, guidance, post-execution report |
| Claim verification | `pgmem/verification.py` | Statements of M3 checked against the reliable results of M1 (the LLM only gives the statement's structured form) |
| Domain helpers | `pgmem/chem.py` | Canonical SMILES, failure classes, deterministic cause confirmation, data-quality checks, functional groups, fingerprint similarity |
| Fault injection | `pgmem/faults.py` | Known-truth scenarios for the evaluation (off unless `PGMEM_FAULTS` is set) |
| Tools | `pgmem/show.py`, `pgmem/collect.py` | Print a memory; zip a namespace with its provenance and reports |
| Evaluation | `evaluation/` | Conditions, scenarios, series runner and metrics (see `docs/EVALUATION.md`) |

## Memory model

Every entry has a category, a kind, a one-sentence text, an exact-retrieval key (canonical molecule, failure class,
agent), its author, references to other entries (`refs`), references to Flowcept (`prov_refs`: task, workflow,
campaign ids) and a `history`. Updates never overwrite: they are appended to the history.

| Category | Question | Kinds | Category-specific fields |
|---|---|---|---|
| M1 evidence and results | What was observed? | `measurement` | entity, quantity, value, unit, experimental conditions (method, accuracy, fmax, max steps, seed, convergence, steps used) |
| M2 experiences | What happened? | `attempt`, `success`, `failure`, `investigation`, `diagnosis`, `outcome` | context, outcome; for diagnoses: problem type and basis |
| M3 knowledge | What was learned? | `hypothesis`, `claim`, `finding`, `rule`, `constraint`, `negative_finding`, `uncertainty` | scope, epistemic status (`observation`, `hypothesis`, `explanation`), supporting and contradicting evidence, uncertainty |
| M4 procedures | How should an agent act? | `procedure`, `strategy`, `recommendation` | trigger, preconditions, actions, expected outcome, limitations, supporting experiences, applications |

Technical failures are M2 experiences, never M1 results. A procedure is only derived from a pattern repeated in M2
experiences and keeps the link to them. Its validation status is a description of what is known (the experiences it
was derived from, how often it was given, applied, effective), with no threshold.

## Integration in the GAIED workflow

| Node | Memory use |
|---|---|
| `plan`, `critique`, `update` | `context()`: the memory relevant to the current task (below), and the analysis agent's pending guidance (marked delivered) |
| `tool_calling` | `explored()`: molecules the team measured or saw fail; the `query_memory` tool, which the agent may call before deciding (at most `PGMEM_MEMORY_QUERIES` rounds); `record_proposals()` |
| simulation | `claim_or_wait()`: reuse an existing result, or take the molecule's lock, or wait for the computation in progress; `record_result()` (M1) or `record_failure()` (M2), both with the Flowcept task id |
| `conclude` | the agent is asked for a `KNOWLEDGE` JSON block (M3, removed from the text the workflow keeps); `record_conclusion()` writes the knowledge and the iteration's attempt (M2) |

## Retrieval

Memory reaches an agent in two ways (hybrid access): at fixed points of the workflow (`plan`, `critique`, `update`),
and when the agent asks, through the `query_memory` tool at experiment selection. Both use the same task-aware
retrieval (`pgmem/retrieval.py`).

**The query is the current task**: the text of the node (the seed for `plan`, the plan for `critique`, the critique
for `update`, the agent's question for `query_memory`), the molecules written in it, the agent's seed and recent
proposals, and the failure classes it met in its last iteration.

**Ranking.** Structured filters first (category, kind of experience), then a hybrid score: Tanimoto similarity of
Morgan fingerprints between the task's molecules and the entry's molecules (0.4), overlap of functional groups from a
closed vocabulary (0.25), overlap of words (0.2), same failure class (0.15). Entries with no signal are not returned.
Embeddings of the text are not used: for molecules, structure is the meaningful similarity, and the specification
does not ask for vector search.

**What a prompt receives**, per category:

| Category | Section |
|---|---|
| M1 | results on molecules related to the task, with their similarity; and, apart, the team's three best results for reference |
| M2 | failures, diagnoses, outcomes of guidance, and attempts that improved (strategies that worked) |
| M3 | related statements, with the evidence cited and the result of their verification against the measurements |
| M4 | only procedures that apply now: their trigger (a failure class) was met in the last iteration, or the task's molecules resemble those of the experiences the procedure was derived from (similarity at least 0.4) |

`query_memory(question, smiles)` answers with the results on that molecule and similar ones, related failures and
diagnoses, statements and their verification, and applicable procedures, each with its Flowcept task id. Each
retrieval is logged as an event (`context`, `memory_query`) with the task's molecules and the entries returned, so the
evaluation can measure what was retrieved, whether the agent asked, and what it did next. The earlier retrieval
(best results and keyword search) remains as an ablation (`PGMEM_RETRIEVAL=recent`).

Every memory call is a few Redis operations; no agent ever waits for the analysis agent. If Redis is unavailable,
the agent logs a warning and continues as the original GAIED.

## Analysis agent

A separate process (`python -m pgmem.analyzer`), reading the memory's event stream (Redis consumer group) and
Flowcept's provenance. An event is acknowledged only after it is processed: if processing fails (for example the LLM
is unavailable), the event stays pending and is retried, so no analysis work is lost.

Each finding is an M2 diagnosis with a **problem type** (`technical_failure`, `data_quality`, `workflow`,
`scientific_contradiction`) and a **basis** (`observed_fact`, `candidate_explanation`, `confirmed_cause`). The
analysis agent never establishes scientific truth and never overwrites an entry: it adds diagnoses, notes and
evidence, and proposes.

| Rule | Trigger | Output |
|---|---|---|
| Redundant computation | the same molecule computed more than once (memory and provenance) | diagnosis; guidance not to compute it again |
| Repeated proposals | an agent proposes molecules already measured or already failed (as known when it proposed) | diagnosis; guidance to that agent |
| Repeated failure | the same molecule fails again; cause confirmed when RDKit cannot read the SMILES | diagnosis; guidance not to propose it again |
| Recurring failure class | different molecules fail with the same class | diagnosis, LLM candidate explanation, guidance, LLM-drafted M4 procedure linked to the experiences (drafted again later if the LLM answer is unusable) |
| Procedure available | a failure of a class that has a procedure | the procedure sent as guidance to the agent concerned |
| Invalid result (impact analysis) | a result whose optimization did not converge or whose value is missing | note on the M1 entry; diagnosis listing the knowledge and procedures that cite it and the downstream Flowcept tasks; guidance to reassess them |
| Knowledge on an unreliable result | new knowledge citing a result with a data-quality note | diagnosis |
| Result vs knowledge | a new result in the scope of existing statements (LLM judgement) | evidence added for or against the statement (labelled candidate); on contradiction, a diagnosis and guidance |
| Claim verification | a new statement (M3), and every new result in a statement's scope | verdict supported / contradicted / inconclusive (below); on contradiction, a diagnosis (observed fact) and guidance to the author, or to every agent when a new result causes it |
| No improvement | two consecutive iterations of an agent without improving its best result | diagnosis; LLM advice for a change of direction |
| Procedure no longer effective | a procedure that worked before fails in its last application | diagnosis; guidance |
| Provenance: task error, LLM call without tool calls | Flowcept records | diagnoses |

After execution, `python -m pgmem.analyzer --post` writes a report (JSON and Markdown in `results/analysis/`):
Flowcept's own provenance summary and error analysis, computations (total, distinct, redundant), coordination
(waits for a molecule being computed), procedures, strategy (attempts, improving attempts), guidance statistics, all
findings, and the memory size.

## Claim verification

The LLM turns a statement into one of four checkable forms, or "none": a value ("the IE of X is about v"), a bound
("above / below v"), a comparison ("X higher / lower than Y") or the effect of a functional group from the closed
vocabulary ("F increases / decreases the IE"). The check is deterministic and uses only reliable results (finite
value, converged optimization, no data-quality note):

| Form | Supported when | Policy |
|---|---|---|
| value | the measured value is within the tolerance | 0.05 eV (xTB is deterministic for a given geometry) |
| bound | the measured value is above / below the stated bound | |
| comparison | the difference has the stated sign | undecided within 0.01 eV |
| motif effect | molecules with the group have a higher / lower mean | at least 2 results in each group, Welch's |t| >= 2 |

Missing or unreliable results, or no clear difference, give "inconclusive", never "contradicted". A verdict says
whether the team's current results agree with the statement, not whether it is true. Verification records are kept
apart from the M3 entries (linked by id), with their history, and are shown to the agents with the statements.

## Guidance tracking

1. **Delivered.** The guidance is added to the prompt of its target agent (or of every agent, `*`) at its next
   `plan`, `critique` or `update` node; the agent, node and time are recorded. Once per agent.
2. **Applied.** Judged on the agent's first proposals after delivery, with a test matching what the guidance asks:
   - molecules to avoid (`avoid_keys`): applied when none of them is proposed again;
   - invalid SMILES (`failure_class` = `invalid_smiles`): applied when RDKit reads every new proposal;
   - guidance with no test (for example advice on a direction): no verdict.
   An iteration without any proposal is not judged.
3. **Effective.** The effect at the next iteration, judged once, at the end of that iteration: effective when no
   listed molecule was computed again and, for guidance about a failure class, no new failure of that class occurred.
   No verdict when the guidance was not applied or has no test. For invalid-SMILES guidance the asked behaviour is the
   outcome itself, so no separate effectiveness verdict is given. The verdict is written as an M2 `outcome` and, for
   guidance coming from a procedure, in that procedure's applications.

## Design decisions

| Decision | Reason |
|---|---|
| Storage in GAIED's existing Redis, keys `pgmem:{namespace}:...`, kept across campaigns | Reuse the deployment; later campaigns learn from earlier ones; one namespace per experiment keeps runs separate |
| M1 keeps every completed measurement, with its scientific fields only | The lock, result reuse and repeated-proposal detection need every result; M3 evidence must point to it; negative results are results; any selection rule would be an arbitrary threshold. Selectivity is in the content (no Flowcept record, log or LLM call is copied) and at retrieval (`PGMEM_RETRIEVAL_LIMIT` entries per category in a prompt) |
| One M2 attempt per agent iteration | Guidance effectiveness, the no-improvement rule and procedure derivation are judged on attempts; a selection would need an arbitrary criterion |
| Deduplication: M1 same quantity, value and conditions; M3/M4 same normalized text and kind; M2 never merged | Each experience happened once; merges append references and history |
| Result reuse when conditions are identical (`PGMEM_REUSE_RESULTS`) | Experiment selection checks existing results |
| Lock per molecule (`pgmem:{ns}:claim:{molecule}`); the other caller waits for the stored result or failure; expires after `PGMEM_CLAIM_TTL_S` (600 s) | Prevents simultaneous duplicate computations; the expiry keeps a crashed agent from blocking a molecule. Active whenever the memory is (`basic`, `structured`, `full`); `PGMEM_CLAIMS=false` disables it |
| "Repeated" means at least two occurrences | Definition of the word, not a tuned threshold |
| LLM only for interpretation; its outputs are labelled candidate | The deterministic rules decide what is observed; the LLM explains, judges and drafts |
| `PGMEM_MODE`: `off` (original GAIED), `basic` (results shared and reused, lock, team list at experiment selection; M1 only, no prompt sections), `structured` (M1-M4, prompt sections, `query_memory`), `full` (structured + the analysis agent's guidance) | The four comparison conditions |
| Retrieval weights 0.4 / 0.25 / 0.2 / 0.15; procedure applicability at similarity 0.4 | Structure first for chemistry; stated, not tuned |
| Verification tolerances (0.05 eV, 0.01 eV, 2 results per group, \|t\| >= 2) | Stated in `pgmem/verification.py`; a verdict is evidence, not truth |
| Failure reuse only as an ablation (`PGMEM_REUSE_FAILURES`) | It is a cache of negative results; kept apart so that it does not explain the effect of the memory |
| `tool_calling` retries at most `TOOL_CALL_RETRIES` (3) times when the LLM gives no parsable tool call (`[gaied-docker fix]`) | The original loop never ends in that case and blocks the agent for the rest of the campaign |

Changes to the original GAIED code are minimal and marked: `[pgmem]` (memory hooks, xTB conditions and convergence
in provenance, agent and node in each LLM call record), `[provenance-fix]` (the xTB task is recorded from the agent
process: the pool worker is a fork whose Flowcept buffer is never flushed), `[gaied-docker ...]` (local execution).

## Limitations

- Effectiveness is judged only on the iteration that follows the guidance.
- The epistemic status of knowledge is the one the agent declares; the analysis agent does not reassess it.
- The evidence an agent cites for a statement is not checked as such; claim verification checks the statement against
  all reliable results instead, but only for its four forms, and depends on the LLM's structured reading of it.
- "Applied" measures visible behaviour (the proposals), not what the agent reasoned.
- The same failure can give two diagnoses, one from memory, one from provenance.
- The lock prevents simultaneous computations only: a molecule that failed can be proposed and computed again later
  (except in the failure-reuse ablation).
- Retrieval relevance is judged by structure (the same molecule, a similar one, a shared specific functional group),
  not by an expert.
- In Flowcept's online mode, the JSONL buffer only keeps the last buffer; durable provenance then needs Flowcept's
  database, which `pgmem` does not read.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `PGMEM_MODE` | `off` | `off`, `basic`, `structured`, `full` |
| `PGMEM_NAMESPACE` | `default` | memory namespace (use a new one per experiment) |
| `REDIS_HOST`, `REDIS_PORT` | `localhost`, `6379` | Redis server |
| `PGMEM_RETRIEVAL_LIMIT` | `5` | entries per category in a prompt |
| `PGMEM_RETRIEVAL` | `contextual` | `contextual` (task-aware) or `recent` (earlier retrieval, ablation) |
| `PGMEM_ON_DEMAND`, `PGMEM_MEMORY_QUERIES` | `true`, `2` | the `query_memory` tool, and its rounds per decision |
| `PGMEM_REUSE_FAILURES` | `false` | ablation: a molecule that failed is not computed again |
| `PGMEM_REUSE_RESULTS` | `true` | reuse a result already in M1 |
| `PGMEM_CLAIMS`, `PGMEM_CLAIM_TTL_S` | `true`, `600` | per-molecule lock and its expiry |
| `PGMEM_ANALYZER_LLM` | `true` | `false`: the analysis agent runs its deterministic rules only |
| `PGMEM_PROVENANCE` | `auto` | `auto` (online channel when Flowcept publishes online, else JSONL), `stream`, `jsonl`, `none` |
| `PGMEM_JSONL_GLOB` | `output/flowcept_buffer_*.jsonl` | Flowcept files read by the analysis agent |
| `PGMEM_FAULTS`, `PGMEM_FAULTS_LOG` | unset | fault injection (see `pgmem/faults.py`) and its ground-truth log |
| `PGMEM_CONTEXT_CATEGORIES` | `M1,M2,M3,M4` | categories put in the agents' prompts (ablations) |
| `PGMEM_ANALYSIS_SOURCES` | `memory,provenance` | sources the analysis agent analyzes (ablations; guidance tracking always runs) |
| `TOOL_CALL_RETRIES` | `3` | retries when the LLM gives no tool call |
| `LLM_PROVIDER=scripted` (+ `OLLAMA_MODEL=scripted`) | | scripted LLM for fast tests; `SCRIPTED_DELAY_S` imitates latency |
| `LLM_SEED` | unset | sampling seed of the Ollama model (paired evaluation runs) |

## Running

From `GAIED/docker` (see `INSTALLATION.md` for the setup):

```bash
docker compose up -d redis
# analysis agent, in the background
docker compose --profile analysis run -d --rm --no-deps --name pgmem-analyzer -e PGMEM_NAMESPACE=run1 analyzer
# campaign
docker compose run --rm --no-deps -e PGMEM_MODE=full -e PGMEM_NAMESPACE=run1 -e "SEEDS=CNC(N)=O,CNC(N)=O" -e MAX_ITERATIONS=3 gaied
# memory, post-execution report, shareable zip
docker compose run --rm --no-deps -e PGMEM_NAMESPACE=run1 gaied python -m pgmem.show
docker compose run --rm --no-deps -e PGMEM_NAMESPACE=run1 analyzer python -m pgmem.analyzer --post
docker stop pgmem-analyzer
docker compose run --rm --no-deps gaied python -m pgmem.collect --namespace run1
```

For a quick check without a model, add `-e LLM_PROVIDER=scripted -e OLLAMA_MODEL=scripted -e SCRIPTED_DELAY_S=2`
to the analysis agent and campaign commands. For online provenance analysis, add
`-e FLOWCEPT_SETTINGS_PATH=/app/docker/flowcept_settings_online.yaml` to both.

## Tests

| Test | Needs | Checks |
|---|---|---|
| `tests/test_pgmem.py` | Redis | memory store, retrieval, every analysis rule, claim verification, guidance tracking, lock, fault injection |
| `tests/test_pgmem_agent_loop.py` | Redis | GAIED's real agent graph with the scripted LLM and an xTB stub: memory writes, task-related prompts, `query_memory`, guidance, lock with parallel agents, bounded tool-call retries |
| `tests/test_pgmem_flowcept.py` | Redis, Flowcept | Flowcept's online channel received during a run; post-execution report |

```bash
python tests/test_pgmem.py && python tests/test_pgmem_agent_loop.py && python tests/test_pgmem_flowcept.py
```
