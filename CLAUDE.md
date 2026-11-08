# GAIED — mémoire partagée pour agents scientifiques (PFE / papier ACX 2027)

## Contexte du projet
- Fork de GAIED (Amal Gueroudji, Argonne) : agents Academy + LangGraph qui cherchent des molécules
  à forte énergie d'ionisation (simulations xTB). La provenance est capturée par Flowcept.
- Mon sujet : la **mémoire** des agents. Objectif immédiat (réunion de lundi) : une **mémoire partagée
  dans Redis** avec push / pull entre agents, pour qu'un agent ne simule pas une molécule déjà
  simulée ou prise par un autre, et que les agents profitent des expériences des autres.
- Schéma d'Amal : Redis contient `Shared Mem` (dont `Goal` et `Instructions`) et `Interception`
  (canal où Flowcept publie la provenance) ; MongoDB = stockage durable ; des agents d'analyse
  (travail de Nadjiba, sur la provenance) lisent Interception et écrivent des Instructions.
- Je travaille avec une binôme sur la mémoire ; Nadjiba travaille sur la provenance.

## Fichiers importants
- `mol_design_agents.py` : l'agent `XTBSimulationAgent`. Boucle LangGraph :
  plan → tool_calling → simulate → conclude → (should_continue) → critique → update → tool_calling.
  `compute_ionization_energy` a un cache local `self.molecule_cache` (par agent, en RAM) :
  c'est le point naturel pour brancher la mémoire partagée.
  `tool_calling` met dans le prompt `state.simulated_molecules` (molécules de CET agent seulement).
- `run-07.py` : lance les agents (Manager Academy sur Redis), rapports toutes les 30 s.
- `docker/` : toute la configuration pour tourner en local.
  - `local_llm.py` : remplace `SomeAPILLM` (passerelle anonymisée, URL vide) par Ollama (Qwen 3.5)
    ou une API au format OpenAI (`LLM_PROVIDER=openai`, clé dans `docker/.env`).
  - `apply_patches.py`, `patch_stop_rule.py` : correctifs déjà appliqués (ne pas relancer).
  - `summarize_prov.py` : résume la provenance de la dernière campagne.
- `shared_memory.py` : la mémoire partagée Redis (voir plus bas) ; `tests/` : tests sans LLM.
- `output/flowcept_buffer_*.jsonl` : provenance de chaque campagne.

## Modifications déjà faites au code d'Amal (marquées `[gaied-docker ...]`)
- `mol_design_agents.py` : si `OLLAMA_MODEL` est défini, `SomeAPILLM` = `local_llm.OllamaLLM` ;
  `N_XTB_WORKERS` fixe le nombre de processus xTB.
- `run-07.py` : `REDIS_HOST`, `SEEDS`, `MAX_ITERATIONS` lus dans l'environnement ; `conda activate`
  retiré de Parsl ; règle d'arrêt : `STABLE_REPORTS` rapports sans nouvelle molécule (et pas à 0).
- Les originaux sont dans `*.orig` et `run-07.py.before-stop-rule`.

## Comment lancer (Windows, PowerShell, Docker Desktop doit être ouvert)
Toujours depuis `C:\Users\pc\Documents\Research\GAIED\docker`, en respectant les MAJUSCULES du chemin
(sinon `docker compose build` échoue avec « --allow=fs.read=Dockerfile »).
```powershell
docker compose up -d redis ollama
docker compose run --rm gaied                                   # une campagne
docker compose run --rm gaied python docker/summarize_prov.py   # résumé de la provenance
docker compose exec redis redis-cli KEYS "mem:*"                # voir la mémoire dans Redis
```
- Le dossier GAIED est monté dans le conteneur (`..:/app`) : modifier un `.py` ne demande PAS de
  reconstruire l'image. Reconstruire (`docker compose build gaied`) seulement si le Dockerfile change.
- Dans le conteneur : Redis = hôte `redis` (`os.environ.get("REDIS_HOST", "localhost")`),
  la bibliothèque `redis` est installée, `PYTHONPATH=/app:/app/docker`.
- Réglages dans `docker/docker-compose.yml` : `SEEDS` (une molécule par agent, séparées par des
  virgules ; deux fois la même pour tester les doublons), `MAX_ITERATIONS`, `STABLE_REPORTS`, `NUM_PREDICT`.
- Le LLM local sur CPU est lent (1 à 2 min par réponse) : une campagne de 3 itérations peut durer
  jusqu'à 1 h. Pour tester la logique de la mémoire, préférer des tests unitaires sans LLM.
- Connexion internet lente et instable : éviter les reconstructions d'image inutiles.

## Règles
- Ne jamais committer `docker/.env` (clé API). Vérifier avec `git check-ignore docker/.env`.
- Travailler sur la branche `docker-local` (ou une branche dédiée), jamais sur `main`.
- Garder les modifications du code d'Amal minimales et marquées par un commentaire.
- Code de la mémoire : d'abord dans `shared_memory.py` à la racine de GAIED (plus tard dans Flowcept).
- **Tout le code est en anglais** : commentaires, docstrings, messages affichés, fichiers de config.
  Aucun français dans le code du projet. Les explications à moi se font en français.
- Je débute avec Docker.

## Mémoire partagée (branche `shared-memory`, première version pour la réunion du 9 nov.)
Code : `shared_memory.py` (racine). Interrupteur : `SHARED_MEMORY` dans docker-compose (true par défaut).
Clés Redis, préfixées par la campagne (`campaign_id` de Flowcept, ou `MEMORY_CAMPAIGN_ID` pour la forcer) :
- `mem:{c}:goal` (JSON : objectif, contraintes), écrit par `run-07.py` avant le lancement des agents
- `mem:{c}:claim:{smiles}` (verrou `SET NX EX`, valeur = agent, expire après `MEMORY_CLAIM_TTL` s)
- `mem:{c}:result:{smiles}` (hash : energy, agent, task_id Flowcept, status, time)
- `mem:{c}:ranking` (zset par énergie), `mem:{c}:failed` (set des échecs)
- `mem:{c}:lessons` (stream : conclusions des agents), `mem:{c}:instructions` (stream, agents d'analyse de Nadjiba)
- `mem:{c}:stats` (compteurs computed / reused / waited / computed_error / reused_error), `mem:{c}:events` (pub/sub)
`{smiles}` = SMILES canonique RDKit (CCO et OCC sont la même clé).
Branchements dans `mol_design_agents.py` (marqués `[shared-memory]`) :
- `agent_on_startup` : `self.memory = SharedMemory.from_env(agent_id=...)`
- `compute_ionization_energy` : `memory.get_or_compute` = pull → claim → xTB → push → release ;
  le `task_id` mis dans la mémoire est celui de la tâche Flowcept (`_compute_energy_with_prov(..., task_id)`)
- `plan`, `tool_calling`, `critique`, `update` : `render_for_prompt()` (résumé borné : Goal versionné,
  leçons des autres, instructions) ; `tool_calling` reçoit aussi les molécules connues de toute l'équipe
- `conclude` : pas de mémoire dans le prompt (il synthétise les résultats de l'agent)
- `conclude` : `push_lesson(conclusion)`
Tests sans LLM :
```powershell
docker compose run --rm gaied python tests/test_shared_memory.py   # module seul (secondes)
docker compose run --rm gaied python tests/test_agent_memory.py    # vrai code de l'agent + vrai xTB (< 1 min)
docker compose run --rm gaied python shared_memory.py show         # contenu de la dernière campagne
```
Provenance des calculs xTB : `_compute_energy_with_prov` écrit la tâche `parsl_task` dans le processus
xTB, mais ce processus est un fork de l'agent : son `_ACTIVE_INTERCEPTOR` est une COPIE dont le tampon
en mémoire n'est jamais écrit sur disque (seul l'intercepteur de l'agent est arrêté) → trace perdue
(vérifié par expérience). `_worker_init` de Flowcept donnerait à chaque processus son propre intercepteur,
mais il n'est injecté que par `FlowceptAcademyPlugin.start()`, que GAIED n'utilise pas. La tâche est donc
écrite depuis le processus de l'agent (`XTBSimulationAgent._record_xtb_task`), même `task_id` que la mémoire.
Mesure : `python docker/summarize_prov.py` affiche « DUPLICATE computations » ; comparer
`SEEDS: "CNC(N)=O,CNC(N)=O"` avec `-e SHARED_MEMORY=false` puis avec la mémoire.
Idées suivantes : déplacer le module dans Flowcept ; agents d'analyse qui lisent `interception`
et écrivent `instructions` ; reprise après crash (même `MEMORY_CAMPAIGN_ID`) ; copie durable dans MongoDB.
