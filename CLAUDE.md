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
- `evident/` : la mémoire EVIDENT (voir plus bas) ; `experiments/` : expériences et analyse ;
  `tests/` : tests sans LLM ; `docs/EVIDENT.md` : la méthode et les travaux connexes.
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
- Code de la mémoire : dans le paquet `evident/` à la racine de GAIED (plus tard dans Flowcept).
- **Tout le code est en anglais** : commentaires, docstrings, messages affichés, fichiers de config.
  Aucun français dans le code du projet. Les explications à moi se font en français.
- Je débute avec Docker.

## Mémoire EVIDENT (branche `evident-memory`, remplace l'ancienne `shared_memory.py`)
Méthode complète : `docs/EVIDENT.md`. Paquet `evident/` :
- `features.py` : vocabulaire fermé de ~22 caractéristiques chimiques (SMARTS), contraintes du Goal
- `claims.py` : affirmations structurées (bloc `CLAIMS:` demandé dans `conclude`, sinon mots-clés)
- `verifier.py` : vérification par le code (effet marginal + effet ajusté par régression, Welch t,
  réplication par agent) et cycle de vie proposed → supported → established → contested / rejected
- `memory.py` : `EvidentMemory` (Redis, clés `ev:{campagne}:...`) : faits, verrous, affirmations
  versionnées, leçons brutes (mode naive), directives, télémétrie, briefing par nœud, export JSON
- `monitor.py` : agent d'analyse temps réel à règles (STOP si propositions hors Goal, conseils)
Modes (`MEMORY_MODE`) : `none` (référence), `coord`, `naive`, `evident`. `MONITOR=true`,
`INJECT_FALSE_CLAIMS=true` (un agent ment, test de robustesse).
Branchements dans `mol_design_agents.py` (marqués `[evident]`) : `evaluate_molecule` dans
`compute_ionization_energy` ; `briefing(node)` dans plan / tool_calling / critique / update ;
`record_proposals` après tool_calling ; `record_decision` ; `record_conclusion` dans conclude ;
appels LLM de plan / critique / update annulables (`watched`) ; durée des nœuds (`_timed`).
`run-07.py` : Goal avec contraintes vérifiables, métadonnées, moniteur, export (`EVIDENT_EXPORT`),
`NO_IMPROVEMENT_TOLERANCE`, `PARSL_MAX_WORKERS` (un worker par agent).
Expériences : `experiments/run_matrix.py <config>` (configs `dryrun`, `pilot`, `main`),
`experiments/synthetic.py` (LLM et xTB simulés, vérité connue), `experiments/analyze.py <dossier>`.
Tests : `python tests/test_evident.py` (Redis requis), `python tests/test_agent_loop.py`
(graphe réel, LLM sans appel d'outil : l'agent ne doit plus se bloquer).
Sur le PC distant (AMD, Ollama Windows) : toujours `--no-deps`, jamais le conteneur `ollama`, jamais
`docker-compose.gpu.yml`, `-e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=qwen3.5:9b`. `LLM_PROVIDER=scripted` = LLM simulé.
Provenance des calculs xTB : `_compute_energy_with_prov` écrit la tâche `parsl_task` dans le processus
xTB, mais ce processus est un fork de l'agent : son `_ACTIVE_INTERCEPTOR` est une COPIE dont le tampon
n'est jamais écrit sur disque → trace perdue. La tâche est donc écrite depuis le processus de l'agent
(`XTBSimulationAgent._record_xtb_task`), avec le même `task_id` que le fait dans la mémoire.
