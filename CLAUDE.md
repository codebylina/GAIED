# GAIED: provenance-grounded memory and analysis for scientific agents (final-year project, ACX 2027 workshop)

## Context
- Fork of GAIED (Amal Gueroudji, Argonne): Academy + LangGraph agents searching for molecules with a high ionization
  energy (xTB simulations). Provenance is captured by Flowcept.
- My topic is the agents' memory: `pgmem`, a shared memory (M1-M4) in Redis with references to Flowcept, and an
  independent analysis agent. Design and decisions: `docs/PGMEM.md`. Development campaigns: `docs/DEVELOPMENT_LOG.md`.
- I work with a partner on the memory; Nadjiba works on provenance analysis.

## Important files
- `mol_design_agents.py`: `XTBSimulationAgent`, LangGraph loop plan -> tool_calling -> simulate -> conclude ->
  (should_continue) -> critique -> update -> tool_calling. Memory hooks marked `[pgmem]`.
- `run-07.py`: launches the agents (Academy manager on Redis), progress reports every 30 s.
- `pgmem/`: memory model and store, agent integration and per-molecule lock, provenance adapter, analysis agent,
  fault injection, `show` and `collect` tools.
- `docker/`: local execution. `local_llm.py` replaces GAIED's anonymized LLM client by Ollama, an OpenAI-format API
  (`LLM_PROVIDER=openai`, key in `docker/.env`) or a scripted model (`LLM_PROVIDER=scripted`).
- `output/flowcept_buffer_*.jsonl`: provenance of each campaign.

## Changes to Amal's code (all marked)
- `[gaied-docker ...]`: `SomeAPILLM` = `local_llm.OllamaLLM` when `OLLAMA_MODEL` is set; `N_XTB_WORKERS`;
  `run-07.py` reads `REDIS_HOST`, `SEEDS`, `MAX_ITERATIONS`, `STABLE_REPORTS`; no `conda activate` in Parsl;
  `tool_calling` retries at most `TOOL_CALL_RETRIES` (3) times (the original loop could block an agent forever).
- `[provenance-fix]`: the xTB task is recorded from the agent process (the pool worker is a fork whose Flowcept
  buffer is never flushed), with a task id shared with the memory entry.
- `[pgmem]`: memory hooks, xTB conditions and convergence in provenance, agent and node in each LLM call record.

## Running (Windows, PowerShell, Docker Desktop open)
- Always from the `GAIED\docker` folder, with the exact letter case of the path.
- The GAIED folder is mounted in the container (`..:/app`): editing a `.py` file needs no image rebuild.
- In the container Redis is host `redis`; `PYTHONPATH=/app:/app/docker`.
- Commands for a campaign with the analysis agent: `docs/PGMEM.md` (section Running). Use a new
  `PGMEM_NAMESPACE` for every campaign. Do not set a short `STABLE_REPORTS` with a real LLM (it stops agents while
  they are still thinking).
- On the remote PC (a friend's computer, AMD GPU, Ollama installed on Windows): always `--no-deps`, never the
  `ollama` container, never `docker-compose.gpu.yml`, and
  `-e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=qwen3.5:9b -e LLM_MODEL=qwen3.5:9b`.
  Do not run `docker image prune` or `--remove-orphans`.

## Rules
- Never commit `docker/.env` (API key); check with `git check-ignore docker/.env`.
- Work on `prov-memory-analysis` (or a dedicated branch), never on `main`.
- Keep changes to Amal's code minimal and marked with a comment.
- Everything in the repository is in English (code, comments, messages, configuration, documentation).
- Explanations to me are given in French. I am new to Docker.
- Do not add mechanisms, statuses, thresholds or layers that are not in the specification without my approval.
