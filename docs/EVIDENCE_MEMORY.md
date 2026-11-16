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

**Ce que reçoit un agent**, dans un seul budget de caractères : le Goal ; les consignes de l'agent d'analyse ;
les avis de retrait ; les expériences exactes de l'équipe ; L2 ; les atomes de L1 proches de sa requête, rangés
par statut.

## Conditions (`bench/configs/evidence_*.json`)

| Condition | Contenu |
|---|---|
| `none` | rien de partagé |
| `mem0` | Mem0 tel quel |
| `registry_only` | L0 et coordination seulement |
| `evidence_unverified` | L0 + atomes, tous montrés comme vrais |
| `evidence` | la mémoire complète |
| `evidence_nolayers`, `evidence_agentindep`, `evidence_posthoc` | ablations (classes de preuves, exposition, pré-enregistrement) |

Ordre prévu :
1. **Étape 1, l'idée marche-t-elle ?** `none`, `evidence_unverified`, `evidence`. `evidence_quick.json` (1 graine,
   2 scénarios) pour vérifier que tout tourne, puis `evidence_step1.json` (3 graines, 4 scénarios).
2. **Étape 2, références** : `mem0` (banc synthétique) et FALDA d'origine (GAIED, branche `falda-baseline`).
3. **Étape 3, ablations** : `registry_only`, `evidence_nolayers`, `evidence_agentindep`, `evidence_posthoc`
   (`evidence_main.json`).

## Lancer (machine distante, depuis `D:\Research\GAIED\docker`)

Banc synthétique :
```powershell
docker compose run --rm --no-deps -e LLM_PROVIDER=ollama -e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=qwen3.5:9b -e LLM_MODEL=qwen3.5:9b gaied python -m bench.run bench/configs/evidence_quick.json
docker compose run --rm --no-deps -e LLM_PROVIDER=ollama -e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=qwen3.5:9b -e LLM_MODEL=qwen3.5:9b gaied python -m bench.analyze results/evidence_quick --reference evidence
```

GAIED :
```powershell
docker compose run --rm --no-deps -e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=qwen3.5:9b -e LLM_MODEL=qwen3.5:9b -e MEMORY_CONDITION=evidence -e MEMORY_EXPORT=results/gaied/evidence_r1.json -e NO_IMPROVEMENT_TOLERANCE=0 -e PARSL_MAX_WORKERS=4 -e "SEEDS=CNC(N)=O,CNC(N)=O,CC1=C(O)N=C(O)N1,CC1=C(O)N=C(O)N1" -e MAX_ITERATIONS=10 gaied
docker compose run --rm --no-deps gaied python -m bench.analyze_gaied results/gaied
```

Tests sans LLM : `python tests/test_evidence.py`, `python tests/test_verifier.py`, `python tests/test_memlab.py`.

## Limites connues (v1)

- Le test de chaque affirmation est écrit en un seul appel par le LLM. Sa qualité se mesure par l'accord avec
  l'évaluateur de vérité et par l'audit manuel (`bench/audit_tests.py`). L'agent de vérification viendra en v2.
- Seule l'exposition directe à un atome est tracée pour la classe R.
- La proximité avec la requête est un simple classement lexical (BM25) : la recherche n'est pas étudiée ici.
- Dans GAIED, l'évaluateur de vérité n'existe pas : GAIED mesure la coordination, le coût et la découverte.
