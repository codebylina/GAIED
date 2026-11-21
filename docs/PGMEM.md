# pgmem: a provenance-grounded graph memory and analysis agent for GAIED

`pgmem` gives GAIED's scientific agents a shared, persistent memory, and adds an independent analysis agent that
reads that memory and Flowcept's provenance, detects problems and sends guidance. The memory is a **property graph**
in Redis: its nodes are typed entries (M1-M4) and the domain's entities, and every relation between nodes is a typed
edge. Provenance stays in Flowcept: the memory refers to it, it never copies it.

```
            Flowcept (execution provenance)                  Redis
   tasks, LLM calls, LangGraph nodes  <── ProvRef ──  ┌──────────────── memory graph ─────────────────┐
          │ (online channel / JSONL)                  │  nodes: M1 result   M2 attempt/failure/diag.  │
          ▼                                           │         M3 knowledge  M4 procedure            │
   ┌────────────────┐   events (Redis Stream)         │         mol:<SMILES>  motif:<name>  g:<guid.> │
   │ analysis agent │ ◄────────────────────────────── │  edges: ABOUT HAS_MOTIF PART_OF SUPPORTED_BY  │
   │  rules, graph  │ ──── diagnoses, procedures, ──► │         CONTRADICTED_BY EXPLAINS DIAGNOSED_FROM│
   │  traversal,    │      verdicts, edges, guidance  │         DERIVED_FROM JUSTIFIED_BY APPLIES     │
   │  verification  │                                 │         APPLIED_IN                            │
   └────────────────┘                                 └───────▲──────────────────────────┬────────────┘
                                                              │ writes (API)             │ hybrid retrieval
                                              ┌───────────────┴──────────────────────────▼────────────┐
                                              │ scientific agents (LangGraph): plan, tool_calling      │
                                              │ (query_memory), simulate (lock), conclude, critique    │
                                              └────────────────────────────────────────────────────────┘
```

## Components

| Module | Role |
|---|---|
| `pgmem/schema.py` | the nodes: entries M1-M4, guidance, ProvRef; epistemic status |
| `pgmem/graph.py` | the relations: `GraphStore`, vocabulary, families, traversal |
| `pgmem/store.py` | `MemoryStore`: entries, entity nodes, verdicts, embeddings cache, guidance, event stream, export |
| `pgmem/domain.py`, `pgmem/chem.py` | the domain interface and its chemistry instance (molecules, motifs, similarity) |
| `pgmem/retrieval.py`, `pgmem/embeddings.py` | hybrid retrieval (BM25, embeddings, Tanimoto, graph; RRF) and completion |
| `pgmem/verification.py` | verdicts of statements against the results |
| `pgmem/agent_client.py` | what an agent does with the memory at each point of its workflow; the per-molecule lock |
| `pgmem/analyzer.py`, `pgmem/llm.py` | the analysis agent and its prompts |
| `pgmem/provenance.py` | adapter to Flowcept (JSONL files, online channel) |
| `pgmem/faults.py` | fault injection for the controlled evaluation |
| `pgmem/show.py`, `pgmem/collect.py` | print a memory; zip a namespace's memory, events, provenance and reports |

## Nodes

An entry holds only what describes it; an id of another node is never a field of an entry.

| Node | Types | Holds |
|---|---|---|
| M1 | `result` | molecule, quantity, value, unit, conditions (including convergence), spelling used, Flowcept task |
| M2 | `attempt` | one agent iteration: proposals and their time, reused results, improved or not, best before and after |
| | `failure` | failed computation: failure class, error message |
| | `diagnosis` | the analysis agent's reading of a problem: rule, problem type, epistemic status, context |
| M3 | `knowledge` | statement, scope, epistemic status, uncertainty |
| M4 | `procedure` | trigger, preconditions, actions, expected outcome, limitations, failure class and motif it is for |
| `mol:<canonical SMILES>` | entity | identity only; created with its `HAS_MOTIF` edges |
| `motif:<name>` | entity | one name of the closed vocabulary (`chem.FEATURES`, generic groups left out) |
| `g:<id>` | guidance | message, target agent (or `*`), problem type, epistemic status, machine-checkable part, delivery log |

Provenance references (`prov_refs`: task, workflow, campaign) stay in the entries: they are external identifiers,
not nodes. Problem types of diagnoses: `technical_failure`, `data_quality`, `workflow`, `scientific_contradiction`.

## Relations

Directed, typed edges; each records `created_by` and its `origin`: `deterministic` (code: RDKit, Flowcept
references, a rule, a numerical check) or `llm` (an agent citing its evidence, an explanation, a justification
proposed by the analysis agent's LLM).

| Family | Relation | Source → target |
|---|---|---|
| evidence | `SUPPORTED_BY` | statement or diagnosis → result or failure that supports it |
| | `CONTRADICTED_BY` | statement → result that contradicts it |
| dependency | `DIAGNOSED_FROM` | diagnosis → what it is based on |
| | `DERIVED_FROM` | procedure → failures it was derived from |
| | `JUSTIFIED_BY` | procedure → knowledge that justifies it |
| | `APPLIES` | guidance → procedure or diagnosis it applies |
| association | `ABOUT` | entry → molecule or motif |
| | `HAS_MOTIF` | molecule → motif |
| | `PART_OF` | result or failure → the attempt that produced it |
| | `EXPLAINS` | statement or diagnosis → what it proposes to explain |
| | `APPLIED_IN` | guidance → the attempt in which it was judged (`applied`, `effective`) |

What rests on a node is found by following **impact** edges backwards: the dependency relations and
`SUPPORTED_BY` (a statement rests on what supports it). `CONTRADICTED_BY` and the association relations are never
propagated: a result that contradicts a statement is not something the statement rests on.

Only deterministic evidence counts in a verdict; evidence of origin `llm` is shown as cited by the author.

### Redis keys

Namespace `pgmem:{ns}` (one per experiment):

```
e:{id}, cat:{C}, key:{C}:{key}      entries, by category and exact key
edge:{id}                           an edge (JSON)
out:{node}, in:{node}               ids of the edges leaving / entering a node
edgekey                             "source|type|target|origin" -> edge id (no duplicate edge)
edges, entities                     all edge ids by time; entity nodes
verification                        verdict records, by entry id
emb                                 embeddings of entries
g:{id}, gq, gseen:{agent}           guidance, its index, what each agent has received
claim:{molecule}                    per-molecule lock
events                              memory events (Redis Stream, consumer group of the analysis agent)
```

Edge creation and deletion run as Lua scripts: the edge, both indexes and the duplicate index change together.

### GraphStore API

```python
g = store.graph
g.add(source, target, relation, created_by, origin="deterministic", **props) -> (edge_id, created)
g.remove(edge_id); g.get(edge_id); g.find(source, relation, target, origin)
g.outgoing(node, types=None); g.incoming(node, types=None); g.neighbors(node, "out" | "in" | "both", types)
g.traverse(start, direction, types, max_depth=2, max_nodes=50) -> [Reach(node, depth, path)]
g.dependents(node, max_depth=3)          # what rests on a node (IMPACT edges backwards)
store.link(...)                          # g.add through the store; store.molecule(smiles), store.motif(name)
```

Unknown relations, node types that a relation does not allow, missing nodes and unknown origins are refused;
an edge already there is returned instead of a duplicate.

## Reliability: two axes

1. **Epistemic status**, declared by the author of a statement (agents and the analysis agent alike):
   `observation` (what was seen), `hypothesis` (what is supposed), `explanation` (a proposed cause).
2. **Verdict**, computed by code and never declared (`pgmem/verification.py`): `supported`, `contradicted`,
   `inconclusive`. The LLM only puts a statement in one of four checkable forms; the check is deterministic on the
   results without a data-quality problem:

| Form | Example | Check |
|---|---|---|
| value | "the IE of X is about v" | \|IE(X) - v\| <= 0.05 eV |
| bound | "the IE of X is above v" | IE(X) > v (or < v) |
| comparison | "X has a higher IE than Y" | IE(X) - IE(Y), undecided within 0.01 eV |
| motif effect | "a nitrile increases the IE" | mean with vs without the motif, Welch's \|t\| >= 2, 2 results per group |

A statement that fits no form, a missing or unreliable result, or no clear difference gives `inconclusive`; missing
evidence never gives `contradicted`. A verdict says whether the team's current results agree, not whether a
statement is true. It is kept apart (with its history), its results are recorded as deterministic evidence edges,
and both are recomputed when a new result falls in the statement's scope. A diagnosis whose cause RDKit confirms (the
SMILES cannot be read) is an `explanation` with a `supported` verdict.

An entry is never rewritten: a contradicted statement keeps its text and status; the verdict and the edges say what
the evidence shows.

## Retrieval

A query is the current task: its text, the molecules it is about (named in the text, the agent's seed and recent
proposals) and the failure classes the agent met recently. Four channels rank the candidates of a category, each on
its own (`PGMEM_CHANNELS`), and Reciprocal Rank Fusion combines the rankings (score = sum of 1 / (60 + rank)): no
weights to tune.

| Channel | Finds |
|---|---|
| `bm25` | exact words: SMILES, error classes, names |
| `semantic` | the same meaning in other words, on M2-M4 (Ollama embeddings, `PGMEM_EMBED_MODEL`, cached) |
| `chem` | similar molecules (Tanimoto of Morgan fingerprints), even without a shared motif |
| `graph` | entries ABOUT a task molecule (distance 1), ABOUT one of its motifs (2), ABOUT another molecule with one of its motifs (3) |

Only `chem` and the motif nodes are specific to chemistry (`pgmem/domain.py`); another domain replaces them.

Retrieval is followed by **completion**: each entry is shown with its direct neighbors (a result with its
data-quality warning, a failure with its proposed explanation, a statement with its verdict and evidence, a
procedure with where it comes from and what its applications gave).

A procedure is shown when it applies: for a motif, when the task's molecules carry it or resemble (Tanimoto >= 0.4)
the molecules it was derived from; for a failure class on any molecule, when the agent met that class recently.

Agents access the memory in two ways, both logged (`context` and `memory_query` events, with the rank each channel
gave): scheduled sections at `plan`, `critique` and `update`; and the `query_memory` tool at experiment selection
(`PGMEM_ON_DEMAND`, at most `PGMEM_MEMORY_QUERIES` rounds).

## Integration in the GAIED workflow

| Node | Memory |
|---|---|
| `plan`, `critique`, `update` | memory sections and guidance in the prompt (`context`) |
| `tool_calling` | team's measured and failed molecules (`explored`); `query_memory` tool; proposals recorded |
| `simulate` | existing result reused, or the per-molecule lock; result (M1 ABOUT its molecule) or failure (M2) |
| `conclude` | a `KNOWLEDGE` block asked for: statements (M3) with their status, the molecules they cite (evidence edges of origin `llm`) and what they explain; the iteration's attempt (M2), which its results and failures are `PART_OF` |

The agents never touch the graph: the code creates the nodes and edges. The lock: before computing a molecule an
agent takes `claim:{molecule}` (Redis `SET NX`, expiring after `PGMEM_CLAIM_TTL_S`); another agent asking for the
same molecule waits for the stored result or failure.

## Analysis agent

A separate process reading the memory events and Flowcept's records. It reads and writes the graph directly, rule
by rule (deterministic first; the LLM to put a statement in a checkable form, propose an explanation, draft a
procedure or advise a change of direction). It never rewrites an entry.

| Event | Rule | Graph use | Writes |
|---|---|---|---|
| result | redundant computation | memory and provenance tasks of the molecule | diagnosis; guidance |
| | data quality | `dependents()`: what rests on the result, up to `PGMEM_ANALYSIS_DEPTH`, through which relations | diagnosis with the impact report; guidance naming direct and indirect dependents |
| | verdicts in scope | statements whose claim the result enters | verdicts, evidence edges; on a new contradiction, as below |
| failure | repeated failure of a molecule | | diagnosis (explanation with a supported verdict when RDKit confirms the cause); guidance |
| | recurring failures | failure → molecule → motif → other molecules → their failures; the motif that fails most among the molecules computed with it, else the class on any molecule | diagnosis ABOUT the motif; proposed explanation (`EXPLAINS`, origin `llm`); guidance; a procedure for the class and motif (`DERIVED_FROM` the failures, `JUSTIFIED_BY` statements about the motif that the LLM names among those shown) |
| | existing procedure | | guidance applying it (`APPLIES`) to the agent, once |
| knowledge | statement on an unreliable result | its `SUPPORTED_BY` edges | diagnosis; guidance to its author |
| | verdict | | verdict; when contradicted: diagnosis with what rests on it, guidance to its author, and to every agent when procedures rest on it |
| proposals | repeated proposals | | diagnosis; guidance to the agent |
| attempt | guidance judgement | the attempt's `PART_OF` results and failures | `APPLIED_IN` edge: applied (none of the molecules or motif warned against proposed), effective (none recomputed, no new failure of the class) |
| | procedure no longer effective | the procedure's applications; its `JUSTIFIED_BY` statements | diagnosis; guidance to every agent naming the statements it rests on |
| | no improvement in two iterations | | diagnosis; advice to the agent |
| provenance | redundant computation, LLM call without tool calls, task in error | Flowcept records | diagnoses |

`investigate(node)` gives the LLM a bounded subgraph (depth 2, 20 nodes) as text, never the whole graph. Memory
events are processed once (Redis consumer group); an event whose processing fails, for example because the LLM is
unavailable, stays pending and is retried. The provenance records received online are counted, so a loss on
Flowcept's channel is visible in the evaluation. After a run, `--post` writes a report: Flowcept's own summary,
findings, procedures and their history, guidance outcomes, coordination, memory and edge counts.

## Design decisions

| Decision | Reason |
|---|---|
| A property graph in GAIED's Redis; no graph database | Reuse the deployment; the volume is small; the same model fits Neo4j or FalkorDB later |
| Relations only in the graph | One source of truth; an entry never holds another node's id |
| No relation extracted by an LLM; every edge has an origin | Edges come from code; what an LLM proposes is marked and never counted as evidence |
| Few types, no overlap (M2: 3 types; M3, M4: 1 each) | Every label serves a rule or a metric |
| Two reliability axes (declared status, computed verdict) | Replaces overlapping scales; a verdict can change without rewriting the statement |
| One procedure per failure class and motif | Two failing motifs give two procedures, each with its own history |
| Standard retrieval components (BM25, embeddings, RRF), one domain channel | Recognizable, no tuned weights; the domain is isolated |
| M1 keeps every completed measurement; one attempt per iteration | Reuse, the lock and the judgement of guidance need them; any selection would be an arbitrary threshold |
| "Repeated" means at least two occurrences | Definition of the word, not a tuned threshold |
| `PGMEM_MODE`: `off`, `basic` (results shared, lock, team list), `structured` (the whole memory, prompt sections, `query_memory`), `full` (+ the analysis agent's guidance) | The four comparison conditions |

Changes to the original GAIED code are minimal and marked: `[pgmem]` (memory hooks, xTB conditions and convergence
in provenance, agent and node in each LLM call record), `[provenance-fix]` (the xTB task is recorded from the agent
process: the pool worker is a fork whose Flowcept buffer is never flushed), `[gaied-docker ...]` (local execution;
`tool_calling` retries at most `TOOL_CALL_RETRIES` times when the LLM gives no parsable tool call).

## Limitations

- Guidance is judged once per agent, on its next proposals and their outcome; "applied" is visible behaviour.
- A procedure for one motif is judged effective when no new failure of its class occurs; after a change of the
  failing motif it is reported as no longer effective, while a new procedure is derived for the new motif.
- Nothing revises the memory: a contradicted statement or an outdated procedure stays, marked by its verdict,
  diagnosis and guidance, until an agent writes a new one.
- Verdicts cover four forms of statement and depend on the LLM's reading of the statement into one of them.
- The evidence an agent cites is not checked as such; the verdict checks the statement against all reliable results.
- Motifs come from a closed vocabulary; molecules RDKit cannot read get no molecule node.
- The lock prevents simultaneous computations only: a molecule that failed can be proposed again later.
- Memories of earlier versions of `pgmem` are not converted: every experiment uses a new namespace.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `PGMEM_MODE` | `off` | `off`, `basic`, `structured`, `full` |
| `PGMEM_NAMESPACE` | `default` | memory namespace (a new one per experiment) |
| `REDIS_HOST`, `REDIS_PORT` | `localhost`, `6379` | Redis server |
| `PGMEM_RETRIEVAL_LIMIT` | `5` | entries per section of a prompt |
| `PGMEM_CHANNELS` | `bm25,semantic,chem,graph` | retrieval channels (ablations) |
| `PGMEM_EMBED_MODEL`, `OLLAMA_URL` | `nomic-embed-text`, `http://localhost:11434` | embedding model of the semantic channel (off for the process if unreachable) |
| `PGMEM_ON_DEMAND`, `PGMEM_MEMORY_QUERIES` | `true`, `2` | the `query_memory` tool and its rounds per decision |
| `PGMEM_CLAIM_TTL_S` | `600` | expiry of a per-molecule lock |
| `PGMEM_ANALYZER_LLM` | `true` | `false`: the analysis agent runs its deterministic rules only |
| `PGMEM_ANALYSIS_DEPTH` | `3` | how far the analysis agent follows what rests on a node |
| `PGMEM_ANALYSIS_SOURCES` | `memory,provenance` | sources the analysis agent analyzes (the judgement of guidance always runs) |
| `PGMEM_PROVENANCE` | `auto` | `auto`, `stream`, `jsonl`, `none` |
| `PGMEM_JSONL_GLOB` | `output/flowcept_buffer_*.jsonl` | Flowcept files read by the analysis agent |
| `PGMEM_FAULTS`, `PGMEM_FAULTS_LOG` | unset | fault injection (`pgmem/faults.py`) and its ground-truth log |
| `TOOL_CALL_RETRIES` | `3` | retries when the LLM gives no tool call |
| `LLM_PROVIDER=scripted` (+ `OLLAMA_MODEL=scripted`) | | scripted LLM and embeddings for tests; `SCRIPTED_DELAY_S` imitates latency |
| `LLM_SEED` | unset | sampling seed of the Ollama model (paired evaluation runs) |

## Running

From `GAIED/docker` (see `INSTALLATION.md`):

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

With a model served by Ollama on the host, add `-e OLLAMA_URL=http://host.docker.internal:11434
-e OLLAMA_MODEL=<model> -e LLM_MODEL=<model> -e PGMEM_EMBED_MODEL=<embedding model>`. For a quick check without a
model, add `-e LLM_PROVIDER=scripted -e OLLAMA_MODEL=scripted -e SCRIPTED_DELAY_S=2`.

## Tests

Redis required (`REDIS_HOST`, `REDIS_PORT`):

```bash
python tests/test_graph.py              # the graph: edges, both directions, traversal, duplicates, deletion
python tests/test_pgmem.py              # memory, retrieval, verdicts, analysis agent (scripted LLM, no xTB)
python tests/test_pgmem_agent_loop.py   # GAIED's real agents and LangGraph loop with the analysis agent
python tests/test_pgmem_flowcept.py     # Flowcept online channel and post-execution analysis
```
