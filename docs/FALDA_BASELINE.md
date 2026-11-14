# Référence FALDA : GAIED avec FALDA comme mémoire partagée

Branche `falda-baseline`, partie de `docker-local` (sans mémoire), indépendante des autres
branches. Objectif : une référence de comparaison avec un **système de mémoire existant, utilisé
tel qu'il est conçu**, et pas une version simplifiée.

## Ce qui est FALDA, sans modification

| Composant | Origine |
|---|---|
| la passerelle FALDA (TypeScript : SQLite + `sqlite-vec` + FTS5, recherche hybride dense + BM25, fusion RRF, pools) | construite depuis le dépôt officiel, commit `ff6d83b` (`docker/falda/Dockerfile`) |
| la distillation T0 → T1 → T2 → T3 | `falda_memory/falda_distiller.py`, **copié tel quel** du dépôt (licence Apache 2.0, `falda_memory/THIRD_PARTY.md`) |
| les embeddings | `nomic-embed-text`, recommandé par le README de FALDA, servi par Ollama |
| le partage entre agents | un **pool partagé** par campagne (docs/POOLS.md) : chaque agent est un *tenant* membre en lecture/écriture |

## Comment GAIED utilise FALDA (`falda_memory/client.py`)

C'est l'intégration « fournisseur de mémoire » décrite dans `docs/HARNESS_INTEGRATION.md` §1b :

- **Écriture (T0 Stream)** : les réponses de `plan`, `update` et `conclude`, et les résultats de
  chaque `simulate`, avec le nom de l'agent et du nœud (`/stream/add`).
- **Rappel** : avant `plan`, `tool_calling`, `critique` et `update`, le prompt reçoit le Core
  (`/core/read`), puis les atomes (`/atoms/search`) et les messages passés (`/stream/search`)
  les plus pertinents, en recherche hybride. **Valeurs par défaut de FALDA** : 10 atomes et
  10 messages (`limit = 10` dans `src/falda.ts`) ; rien n'est coupé, FALDA ne définissant aucune
  limite de taille pour le prompt.
- **Goal** : écrit comme **Core (T3)** du pool au début de la campagne (objectif ; au plus 12 atomes
  lourds ; éléments C, H, N, O, F) ; c'est la façon FALDA de donner un « cœur de projet » aux agents.

Ce que FALDA ne fait pas, et que cette intégration n'ajoute volontairement pas : aucune
coordination entre agents (pas de verrou, pas de « en cours »), aucun lien entre une mémoire et
la mesure qui la justifie, aucune vérification. La liste « ne pas répéter » de `tool_calling`
reste celle d'origine (les molécules de l'agent seulement).

## Écarts nécessaires (à mentionner dans le papier)

1. **Cadence de distillation réduite** : par défaut FALDA distille des atomes tous les 10
   messages, une scène par heure et le Core toutes les 6 h ; une campagne dure environ 30 min.
   Ici : atomes tous les 6 messages, scène toutes les 5 min, Core toutes les 15 min, vérification
   toutes les 30 s (`FALDA_L1_EVERY_N`, `FALDA_L2_INTERVAL_S`, `FALDA_L3_INTERVAL_S`,
   `FALDA_POLL_S`). Sans cela, T2 et T3 ne seraient jamais produits pendant une campagne.
2. **Un pool par campagne** : chaque campagne part d'une mémoire vide, ce qui rend les campagnes comparables entre elles.
3. **Le LLM du distillateur** est le même Qwen que les agents (Ollama, interface OpenAI).

Correctifs de GAIED communs à toutes les branches (`[gaied-fix]`) : au plus 3 nouvelles demandes
quand le LLM ne donne pas d'appel d'outil lisible ; un worker Parsl par agent ;
`NO_IMPROVEMENT_TOLERANCE` (0 = budget fixe).

## Mesure (`falda_memory/recorder.py`, `experiments/analyze_falda.py`)

Uniquement pour l'analyse, jamais montré aux agents : propositions, calculs, texte rappelé et sa
taille, conclusions, durée de chaque étape. `analyze_falda.py` en tire, par campagne : meilleure
molécule (toutes, et parmi celles qui respectent le Goal), calculs en double, molécules invalides
recalculées, diversité, propositions déjà connues, propositions hors Goal, échecs de format du LLM,
durée d'un appel au LLM, taille et temps du rappel, et la taille de chaque niveau de FALDA à la fin.

## Lancer (machine distante, depuis `D:\Research\GAIED\docker`)

```powershell
ollama pull nomic-embed-text                                     # une fois, dans Windows
docker compose --profile falda build falda                       # une fois (télécharge Node et FALDA)
docker compose --profile falda up -d redis falda
docker compose run --rm --no-deps -e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=qwen3.5:9b gaied python experiments/run_falda.py experiments/configs/falda_dryrun.json
docker compose run --rm --no-deps -e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=qwen3.5:9b gaied python experiments/run_falda.py experiments/configs/falda.json
```
Puis l'analyse :
```powershell
docker compose run --rm --no-deps gaied python experiments/analyze_falda.py experiments/results/falda
```
Résultats : `experiments\results\falda\report.md` et `metrics_runs.csv`. Pour une campagne
sans mémoire avec les mêmes réglages, ajouter `"none"` à la liste `conditions` de
`experiments/configs/falda.json` (attention : sans FALDA, les agents ne voient pas le Goal).
