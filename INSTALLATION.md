# Installing and running GAIED with Docker

This guide is written for Windows and assumes no prior Docker knowledge. On macOS or Linux the commands are the
same; only the Docker installation differs.

## What runs where

The project needs Python libraries, the xTB simulator, Redis and a language model. Docker runs them in containers:

| Container | Role |
|---|---|
| `redis` | message exchange between agents (Academy) and the shared memory (`pgmem`) |
| `ollama` | a local language model (Qwen 3.5), no API key needed |
| `gaied` | the project code (agents, xTB, Flowcept) |
| `analyzer` | the analysis agent (`pgmem`), started only when needed (profile `analysis`) |

`docker/docker-compose.yml` describes the containers; `docker/Dockerfile` describes the `gaied` image, built once.
The project folder is mounted in the containers (`..:/app`): editing a `.py` file does not require rebuilding the
image. About 15 GB of free disk space are needed (image and model).

## 1. Install the tools (once)

1. Git: https://git-scm.com/download/win
2. Docker Desktop: https://www.docker.com/products/docker-desktop/ (keep "Use WSL 2"; if asked, run
   `wsl --install` in an administrator PowerShell and restart). Start Docker Desktop and wait for "Engine running".

Check in PowerShell: `git --version`, `docker --version` and `docker compose version` each print a version.

## 2. Get the code

```powershell
git clone https://github.com/codebylina/GAIED.git GAIED
cd GAIED
git switch prov-memory-analysis-v2
```

## 3. Build the image (once, 10 to 30 minutes)

All Docker commands run from the `GAIED\docker` folder, typed with the exact letter case of the path (otherwise
`docker compose build` fails with `--allow=fs.read=Dockerfile`).

```powershell
cd docker
docker compose build gaied
```

## 4. Start Redis and the language model

```powershell
docker compose up -d redis ollama
docker compose exec ollama ollama pull qwen3.5:4b
docker compose exec redis redis-cli ping        # PONG
```

To use a model already served by Ollama installed on the computer itself (instead of the `ollama` container), add
`--no-deps -e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=<model> -e LLM_MODEL=<model>` to the
`docker compose run` commands below.

### Optional: an online API instead of the local model

```powershell
Copy-Item env.example.txt .env
notepad .env
```

Paste the key after `LLM_API_KEY=` and uncomment the two lines of the chosen provider. `docker/.env` is never
committed (`git check-ignore .env` prints `.env`).

## 5. Check the image (a few seconds, no model)

```powershell
docker compose run --rm --no-deps gaied python -c "import mol_design_agents, flowcept, academy; print('OK')"
docker compose run --rm --no-deps gaied python tests/test_pgmem.py
```

## 6. Run a campaign

```powershell
docker compose run --rm gaied
```

Useful settings (add `-e NAME=value` before `gaied`):

| Setting | Effect |
|---|---|
| `"SEEDS=CNC(N)=O,CNC(N)=O"` | starting molecules, one agent per seed |
| `MAX_ITERATIONS=3` | iterations of each agent |
| `PGMEM_MODE=full`, `PGMEM_NAMESPACE=<name>` | shared memory and analysis agent (see `docs/PGMEM.md`) |

With a local model on CPU a campaign can take an hour or more; a progress report is printed every 30 seconds.
`docs/PGMEM.md` gives the full sequence with the analysis agent, the memory report and the post-execution analysis.

## Troubleshooting

| Message | Cause and fix |
|---|---|
| `--allow=fs.read=Dockerfile` | wrong letter case in the path: `cd` again with the exact case |
| `no such host`, `Could not resolve host`, `timeout` | unstable connection: run the same command again (downloaded layers are kept) |
| `error during connect` | Docker Desktop is not running |
| `no configuration file provided` | the command was not run from the `docker` folder |
| `Connection refused` on `/api/chat` | Ollama is not running |
| `Number of molecules simulated: 0` for several minutes | normal with a local model: each answer takes time |

## Working with Git

Work on a branch of your own, created from `prov-memory-analysis-v2`, and open a pull request to merge it. Never commit
`docker/.env`, a token or a key. Campaign outputs (`runinfo/`, `output/flowcept_buffer_*.jsonl`, `results/`) are
ignored by Git.
