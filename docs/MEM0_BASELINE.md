# Référence Mem0 : GAIED avec Mem0 comme mémoire partagée

Branche `mem0-baseline`, partie de `docker-local`, indépendante des autres branches. Elle sert de
**point de départ** : Mem0 seul, utilisé comme prévu. La couche de vérification viendra ensuite par-dessus.

## Ce qu'est Mem0 ici

[Mem0](https://github.com/mem0ai/mem0) (licence Apache 2.0), version libre **2.2.1**, sans modification :

- **Écriture.** Les résultats de chaque `simulate` et les conclusions de chaque `conclude` sont ajoutés
  comme messages (`add(..., infer=True)`). Le LLM de Mem0 en extrait des souvenirs : c'est le
  fonctionnement normal de Mem0, en une seule extraction par message dans la version 2.2.
  L'espace est `agent_id` = l'agent et `run_id` = la campagne.
- **Lecture.** Avant `plan`, `tool_calling`, `conclude`, `critique` et `update`, la question de l'étape
  est cherchée dans les souvenirs de **toute la campagne**, donc de tous les agents, avec le nombre de
  résultats par défaut de Mem0. Les souvenirs sont ajoutés au prompt, dans le budget
  `briefing_budget_chars`.
- **Stockage** : Qdrant intégré, sur disque (`results/.../mem0_store`). LLM : Qwen via Ollama.
  Embeddings : `nomic-embed-text` via Ollama.
- **Rien n'est vérifié et aucun calcul n'est coordonné** : c'est la conception de Mem0, et c'est ce
  que la référence doit montrer.

### Écarts nécessaires (à mentionner)

1. **Un service Mem0 dans le processus principal.** Le stockage local de Mem0 n'accepte qu'un
   processus, et les agents de GAIED tournent dans plusieurs. Une seule instance de Mem0 tourne donc dans
   `run-07.py` et les agents l'interrogent en HTTP (`memlab/mem0_memory.py`). Les écritures passent par
   une file : l'agent n'attend pas l'extraction, comme avec le mode asynchrone de Mem0.
2. **Les mêmes réglages de Qwen que les agents.** Le pilote Ollama de Mem0 ne permet pas de couper la
   réflexion de Qwen. Elle est réglée comme pour les agents (`OLLAMA_THINK`, `false` par défaut).
3. **Une fenêtre de contexte suffisante.** Le prompt d'extraction de Mem0 fait environ 36 000
   caractères, soit environ 9 000 tokens. C'est plus que les 8192 tokens des agents, et Ollama le
   couperait sans prévenir. Mem0 reçoit donc 16 384 tokens (`MEM0_NUM_CTX`).
4. **Sans les options lourdes.** Ni `mem0ai[extras]` (recherche BM25 avec fastembed) ni `mem0ai[nlp]`
   (spaCy) ne sont installés. Mem0 cherche par le sens seulement.

## Mesure (`memlab/`, `bench/`)

Rien de la mesure n'est montré aux agents : expériences (provenance), ce que chaque prompt a reçu, appels
au LLM et leur durée, écritures de Mem0 (file d'attente, durée, erreurs), souvenirs à la fin.

- **Banc d'essai synthétique** (`bench/`) : la vérité est connue, donc on mesure aussi la **justesse**
  des souvenirs montrés aux agents. Scénarios : standard, trompeur (une option aide au départ et nuit
  ailleurs), adversaire (un agent écrit des affirmations fausses), coupure. Conditions : `none` et `mem0`.
- **GAIED** : expériences répétées, échecs relancés, meilleures molécules, coût (`bench.analyze_gaied`).

## Lancer (machine distante, depuis `D:\Research\GAIED\docker`)

Une seule fois, il faut reconstruire l'image, car le Dockerfile ajoute `mem0ai` et `ollama` :
```powershell
docker compose build gaied
```

Le banc d'essai : d'abord l'essai léger, 1 scénario et 1 graine, puis l'analyse.
```powershell
docker compose run --rm --no-deps -e LLM_PROVIDER=ollama -e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=qwen3.5:9b -e LLM_MODEL=qwen3.5:9b -e EMBED_MODEL=nomic-embed-text gaied python -m bench.run bench/configs/quick.json
docker compose run --rm --no-deps -e LLM_PROVIDER=ollama -e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=qwen3.5:9b -e LLM_MODEL=qwen3.5:9b gaied python -m bench.analyze results/quick --reference mem0
```

Une campagne GAIED avec Mem0. `MEMORY_CONDITION=none` donne la même campagne sans mémoire, avec la même mesure.
```powershell
docker compose run --rm --no-deps -e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=qwen3.5:9b -e LLM_MODEL=qwen3.5:9b -e EMBED_MODEL=nomic-embed-text -e MEMORY_CONDITION=mem0 -e MEMORY_EXPORT=results/gaied/mem0_r1.json -e NO_IMPROVEMENT_TOLERANCE=0 -e PARSL_MAX_WORKERS=4 -e "SEEDS=CNC(N)=O,CNC(N)=O,CC1=C(O)N=C(O)N1,CC1=C(O)N=C(O)N1" -e MAX_ITERATIONS=10 gaied
docker compose run --rm --no-deps gaied python -m bench.analyze_gaied results/gaied
```

Tests sans LLM : `python tests/test_memlab.py`. Essai à blanc complet :
`python -m bench.run bench/configs/dryrun.json`, puis
`LLM_PROVIDER=scripted python -m bench.analyze results/dryrun`.

Branchements dans le code d'Amal marqués `[memlab]`, et correctif des relances `[gaied-fix]`.
