# Mémoire fondée sur les preuves (v1)

Branche `evidence-memory`, partie de `mem0-verified` (dont elle réutilise le vérificateur). Version 1 : le LLM
seul écrit les tests, un appel par lot, sans agent de vérification.

## L'idée

Le brut est gardé tel quel et immuable. Le LLM propose des affirmations, mais c'est le code qui décide de leur
statut, à partir des mesures du workflow. On monte et on descend d'une couche selon les preuves, pas selon le
temps.

| Couche | Contenu | Écrit par | Code |
|---|---|---|---|
| L0 registre | une entrée par entité : tentatives, statut, valeur, cause de l'échec (`invalid` / `error`), agent, heures, `task_id` | le code, à partir de la provenance | `memlab/registry.py` |
| L0 journal | les conclusions des agents, telles qu'écrites, avec auteur et heure | les agents | `memlab/evidence_memory.py` |
| L1 atomes | affirmations tirées du journal, avec leur source ; test pré-enregistré ; statut selon les classes E/C/R/I | le LLM propose, le code décide | `memlab/distiller.py`, `memlab/verifier.py` |
| L2 établi | sélection (jamais un résumé) des généralisations `established` puis `supported` | le code | `memlab/evidence_memory.py` |

**Coordination exacte.** Avant chaque expérience, le registre dit si l'entité est déjà mesurée (la valeur est
rendue sans calcul), a déjà échoué (`invalid`, ou trop d'erreurs), ou est en cours chez un autre agent (on
attend son résultat). Sinon, l'agent la réserve ; la réservation est renouvelée pendant le calcul et expire si
l'agent plante.

**Recherche, comme FALDA** (`memlab/retrieval.py`) : vecteurs `nomic-embed-text` sur le texte brut + BM25, fusion
des rangs (RRF, k = 60), 10 résultats par collection (les valeurs par défaut de FALDA). Collections cherchées :
le registre (L0) et les atomes (L1 et L2). Le résultat est mis dans le prompt, rangé par couche et par statut, dans
un budget de caractères identique pour toutes les conditions.

**Toujours montrés** (courts, nécessaires à chaque étape) : le Goal, les consignes de l'agent d'analyse qui
concernent l'agent, les avis de retrait.

**Outils pour l'agent** (banc synthétique) : `lookup(design)` (état exact d'un design dans le registre),
`best_results()` (meilleurs résultats de l'équipe), `search_memory(query)` (recherche hybride dans les expériences
et les connaissances). Chaque réponse est journalisée comme ce que l'agent a vu (pour les classes de preuves et
l'exposition). Dans GAIED, pas d'outils pour l'instant : il faudrait modifier le graphe d'Amal (le nœud
`tool_calling` envoie tous les appels d'outils à xTB).

**Différence voulue avec FALDA** : les messages bruts ne sont pas mis dans les prompts. Ce qui vient du texte des
agents passe par L1, avec un statut ; sinon une affirmation réfutée reviendrait par son message d'origine.
Réglage `show_messages` pour les montrer quand même (étiquetés « non vérifiés »).

## Conditions (`bench/configs/evidence_*.json`)

| Condition | Contenu |
|---|---|
| `none` | rien de partagé |
| `mem0` | Mem0 tel quel |
| `registry_only` | L0 et coordination seulement |
| `evidence_unverified` | L0 + atomes, tous montrés comme vrais |
| `evidence` | la mémoire complète |
| `evidence_nolayers`, `evidence_agentindep`, `evidence_posthoc` | ablations (classes de preuves, exposition, pré-enregistrement) |
| `falda` | FALDA d'origine, par sa passerelle (`memlab/falda_memory.py`, branche `falda-memlab`) |

Ordre prévu :
1. **Étape 1, l'idée marche-t-elle ?** `none`, `evidence_unverified`, `evidence`. `evidence_quick.json` (1 graine,
   2 scénarios) pour vérifier que tout tourne, puis `evidence_step1.json` (3 graines, 4 scénarios).
2. **Étape 2, références** : `mem0` et `falda`, dans le banc synthétique et dans GAIED.
3. **Étape 3, ablations** : `registry_only`, `evidence_nolayers`, `evidence_agentindep`, `evidence_posthoc`
   (`evidence_main.json`).

## Lancer (machine distante, depuis `D:\Research\GAIED\docker`)

Banc synthétique :
```powershell
docker compose run --rm --no-deps -e LLM_PROVIDER=ollama -e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=qwen3.5:9b -e LLM_MODEL=qwen3.5:9b gaied python -m bench.run bench/configs/evidence_quick.json
docker compose run --rm --no-deps -e LLM_PROVIDER=ollama -e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=qwen3.5:9b -e LLM_MODEL=qwen3.5:9b gaied python -m bench.analyze results/evidence_quick --reference evidence
```

GAIED, étape 1 (une campagne par condition et par répétition, mêmes graines ; répéter avec `_r2`, `_r3`) :
```powershell
docker compose run --rm --no-deps -e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=qwen3.5:9b -e LLM_MODEL=qwen3.5:9b -e MEMORY_CONDITION=none -e MEMORY_EXPORT=results/gaied_step1/none_r1.json -e NO_IMPROVEMENT_TOLERANCE=0 -e PARSL_MAX_WORKERS=4 -e "SEEDS=CNC(N)=O,CNC(N)=O,CC1=C(O)N=C(O)N1,CC1=C(O)N=C(O)N1" -e MAX_ITERATIONS=10 gaied
```
puis la même commande avec `MEMORY_CONDITION=evidence_unverified` (`evidence_unverified_r1.json`) et
`MEMORY_CONDITION=evidence` (`evidence_r1.json`). Ensuite le rejeu de vérification (GAIED n'a pas d'évaluateur de
vérité : le même vérificateur rejuge tous les atomes avec toutes les mesures de la campagne) et l'analyse :
```powershell
docker compose run --rm --no-deps -e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=qwen3.5:9b -e LLM_MODEL=qwen3.5:9b gaied sh -c "python -m bench.replay_verify results/gaied_step1/*.json"
docker compose run --rm --no-deps gaied python -m bench.analyze_gaied results/gaied_step1
```
Mesures GAIED : doublons, échecs relancés, calculs évités, meilleure molécule, coût ; et pour les connaissances,
`share_shown_true_bad` = part des affirmations montrées comme vraies que les mesures de la campagne contredisent
ou ne contiennent pas (cohérence avec les mesures, pas vérité chimique).

Tests sans LLM : `python tests/test_evidence.py`, `python tests/test_verifier.py`, `python tests/test_memlab.py`.

## Limites connues (v1)

- Le test de chaque affirmation est écrit en un seul appel par le LLM. Sa qualité se mesure par l'accord avec
  l'évaluateur de vérité et par l'audit manuel (`bench/audit_tests.py`). L'agent de vérification viendra en v2.
- Seule l'exposition directe à un atome est tracée pour la classe R.
- La proximité avec la requête est un simple classement lexical (BM25) : la recherche n'est pas étudiée ici.
- Dans GAIED, l'évaluateur de vérité n'existe pas : GAIED mesure la coordination, le coût et la découverte.

## FALDA dans le même cadre (`memlab/falda_memory.py`)

FALDA n'est pas modifié : passerelle construite depuis le dépôt officiel (commit `ff6d83b`,
`docker/falda/Dockerfile`), distillateur d'origine copié tel quel (`memlab/third_party/falda_distiller.py`,
même empreinte SHA-256 que dans `falda-baseline`). Écriture de chaque tour (plan, résultats, conclusion, mise à jour)
dans T0 ; rappel = Core + 10 atomes + 10 messages (recherche hybride de la passerelle) ; un pool par essai.
Choix du cadre, communs ou documentés : le distillateur tourne dans la boucle de fond de la mémoire et appelle le LLM
par le client du cadre (même modèle et réglages que les agents) ; cadence raccourcie (atomes tous les 6 tours,
scène toutes les 5 min, cœur toutes les 15 min, vérification toutes les 30 s) ; texte coupé au budget commun.

Lancer (la passerelle doit tourner) :
```powershell
docker compose --profile falda up -d falda
docker compose run --rm --no-deps -e LLM_PROVIDER=ollama -e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=qwen3.5:9b -e LLM_MODEL=qwen3.5:9b -e EMBED_MODEL=nomic-embed-text gaied python -m bench.run bench/configs/evidence_quick.json
```
Test : `python tests/test_falda.py` (sauté si aucune passerelle ne répond).
