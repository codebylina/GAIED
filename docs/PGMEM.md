# pgmem — mémoire partagée ancrée dans la provenance et agent d'analyse

Branche `prov-memory-analysis` (à partir de `docker-local`). Ce document suit le cahier des charges initial. Il
sépare ce qui vient de ce cahier, les décisions techniques nécessaires pour l'implémenter, et les suggestions non
implémentées.

## Composants

| Composant | Fichier | Rôle |
|---|---|---|
| Flowcept | inchangé | Seule source de la provenance détaillée |
| Adaptateur de provenance | `pgmem/provenance.py` | Lit Flowcept (fichiers JSONL, canal Redis en ligne, base) ; parcourt la provenance (tâches en aval) |
| Mémoire partagée M1-M4 | `pgmem/schema.py`, `pgmem/store.py` | Une seule mémoire dans le Redis existant, 4 catégories logiques, références vers Flowcept |
| Agent d'analyse | `pgmem/analyzer.py` | Processus séparé : règles déterministes, LLM pour l'interprétation, consignes aux agents |
| Accroches GAIED | `pgmem/agent_client.py` + lignes `[pgmem]` de `mol_design_agents.py` | Planification, choix des expériences, exécution, conclusion, critique et révision |
| Injection de fautes | `pgmem/faults.py` | Scénarios d'évaluation à vérité connue (désactivée par défaut) |

## Ce qui vient du cahier des charges

- **M1 (preuves et résultats).** Observations et mesures, avec leurs conditions expérimentales et la référence à la
  tâche Flowcept (`task_id`).
- **M2 (expériences).** Tentatives, succès, échecs, investigations, diagnostics, issues observées. Les pannes
  techniques vont ici, jamais dans M1.
- **M3 (connaissances).** Hypothèses, affirmations, résultats, règles, contraintes, résultats négatifs, incertitudes.
  Chaque entrée garde sa portée, ses preuves pour et contre, ses références de provenance et son statut
  épistémique : `observation`, `hypothesis` ou `explanation` (tes termes).
- **M4 (procédures).** Déclencheur, préconditions, actions, effet attendu, limites, statut de validation, preuves.
  Une procédure n'est créée qu'à partir d'un schéma répété dans des expériences M2, et elle garde le lien vers elles.
- **Agent d'analyse.** Il distingue les types de problèmes (`technical_failure`, `data_quality`, `workflow`,
  `scientific_contradiction`) et la base de chaque énoncé (`observed_fact`, `candidate_explanation`,
  `confirmed_cause`). Il n'établit jamais une vérité scientifique et n'écrase rien : il ajoute des diagnostics, des
  notes et des preuves, et il propose.
- **Analyse d'impact.** Un résultat invalide (optimisation non convergente, valeur absente) amène l'agent d'analyse à
  lister les connaissances et procédures qui le citent et les tâches Flowcept en aval, puis à proposer leur
  réévaluation.
- **Deux modes d'analyse.** En ligne (événements, sans bloquer les agents) et après exécution
  (`python -m pgmem.analyzer --post`).
- **Consignes.** Elles arrivent aux agents pendant l'exécution (à `plan`, `critique` et `update`). On suit si chacune
  a été remise, appliquée et efficace.
- **Rien n'est perdu si un service est indisponible.** Un événement non traité reste en attente et sera repris.

## Décisions techniques (nécessaires, réversibles)

| Décision | Pourquoi |
|---|---|
| Stockage dans le Redis déjà déployé, clés `pgmem:{namespace}:...` | Réutiliser l'existant ; garder la mémoire d'une campagne à l'autre |
| M1 reçoit chaque mesure terminée (valeur, conditions, `task_id`), pas la tâche Flowcept entière | « Sélectif » est compris comme : on garde l'observation scientifique, pas les événements, journaux ou appels LLM. **À confirmer** (voir plus bas) |
| Le choix des expériences réutilise un résultat déjà dans M1 si les conditions sont identiques (`PGMEM_REUSE_RESULTS`) | Façon de « vérifier les résultats existants » ; peut être désactivé |
| Une entrée M2 « tentative » par itération d'agent | Le cahier cite les « tentatives » ; **à confirmer** (voir plus bas) |
| Statut de validation d'une procédure M4 = description de ce qui est connu (expériences d'origine, applications, efficaces ou non), sans seuil | Le cahier demande un statut de validation, sans en définir les valeurs |
| « Répété » = au moins deux occurrences | Définition du mot, pas un seuil réglé |
| Bloc `KNOWLEDGE` (JSON) demandé à la fin de `conclude` quand la mémoire est structurée, puis retiré du texte | Pour que l'agent propose ses connaissances sous une forme exploitable |
| Une consigne non appliquée n'est pas jugée efficace ou non | L'effet ne lui est pas attribuable |
| **Verrou par molécule (demandé par toi après real3)** : avant de calculer, l'agent pose `pgmem:{namespace}:claim:{molécule}` dans Redis (pris une seule fois) ; un autre appel pour la même molécule attend le résultat (ou l'échec) au lieu de recalculer ; verrou libéré après l'enregistrement du résultat | Empêche les calculs simultanés en double (`O=C1C(F)=C(F)N1C(F)(F)F` calculée 3 fois dans real3). Actif dans `basic`, `structured` et `full` (comme la réutilisation des résultats), désactivable : `PGMEM_CLAIMS=false` |
| Le verrou expire après `PGMEM_CLAIM_TTL_S` (600 s) | Nécessaire : un agent planté ne doit pas bloquer une molécule ; 600 s dépasse largement un calcul xTB de ces molécules. Le rapport compte les attentes (`coordination`) |
| Modes `PGMEM_MODE` : `off`, `basic`, `structured`, `full` | Ce sont les quatre conditions de comparaison du cahier |
| Changements dans le code d'Amal, marqués `[pgmem]` | Conditions et convergence du calcul xTB enregistrées dans la provenance ; agent et nœud écrits dans chaque appel LLM ; accroches mémoire |

## Suivi d'une consigne : remise, appliquée, efficace

1. **Remise (`delivered`).** La consigne est ajoutée au prompt de l'agent visé (ou de tous, `*`) au prochain nœud
   `plan`, `critique` ou `update` ; on note l'agent et l'heure. Une seule fois par agent.
2. **Appliquée (`applied`).** Jugée sur les premières propositions de l'agent après la remise (`tool_calling`). La
   consigne porte une liste de molécules à éviter (`avoid_keys`) : appliquée = aucune de ces molécules n'est
   reproposée. Sans liste (par exemple un conseil de direction), ce n'est pas vérifiable : rien n'est écrit.
3. **Efficace (`effective`).** Jugée à la fin de cette itération (`conclude`), sur ce que l'agent a fait depuis la
   remise : efficace = aucune molécule de la liste recalculée et, si la consigne porte sur un type d'échec
   (`failure_class`), aucun nouvel échec de ce type. Non appliquée ou non vérifiable : pas de verdict. Le résultat
   est écrit dans M2 (`outcome`) et, si la consigne vient d'une procédure, dans les applications de cette procédure
   M4.

Limites constatées (real3) : « appliquée » ne vérifie que la liste de molécules, pas le reste du message ; le
verdict d'efficacité est donné une seule fois, sur une itération (voir les questions ouvertes).

## Journal des campagnes réelles (Qwen 3.5 9B, vrai xTB, PC distant)

Chaque constat est rangé dans une de trois cases : **traité automatiquement** par les règles existantes (preuve pour
le papier), **pas couvert** (type de problème général : nouvelle règle générale, décidée ensemble), **bug** du code
(corrigé). Les campagnes de développement ci-dessous ne servent pas à l'évaluation : le code a changé entre elles.

| Campagne | Traité automatiquement | Pas couvert | Bug corrigé |
|---|---|---|---|
| real1 (2 graines ; coupée par `STABLE_REPORTS=2`) | Consignes remises ; un agent ignore « ne repropose pas ces SMILES invalides » : noté non appliqué ; M3 rempli par Qwen avec types et statuts | — | Propositions comparées à la mémoire trop tard (fausses « already measured ») ; champs de procédure en liste |
| real2 (coupée) | Vraies repropositions détectées (7) ; échecs SMILES diagnostiqués ; explication candidate fausse mais étiquetée `candidate_explanation` ; procédure M4 tirée de 2 expériences | Le même échec diagnostiqué deux fois (mémoire et provenance) | Format des actions de procédure |
| real3 (complète, 3 itérations) | Réutilisation d'un résultat d'un autre agent ; cycle complet des consignes (6 appliquées, 5 efficaces, 1 non efficace) ; preuve contraire ajoutée sans écraser ; calcul triple détecté ; repropositions arrêtées après la consigne ciblée | Calculs simultanés en double → **verrou ajouté (accord)** ; test « appliquée » inadapté aux consignes sur les SMILES → test RDKit des propositions (proposé, à confirmer) ; efficacité jugée une seule fois ; statut épistémique déclaré par l'agent non réexaminé ; procédure non réessayée si la réponse du LLM est inutilisable | Preuves des connaissances sur une molécule en échec ; texte trompeur des erreurs de tâches ; unités des consignes ; réponses inutilisables du LLM non journalisées |

Problèmes d'exploitation (pas du système) : Ollama arrêté (real2, 1er essai) ; `STABLE_REPORTS=2` donné par erreur
dans les commandes (real1, real2).

## Questions ouvertes (à trancher par toi)

1. **M1 : toutes les mesures, ou une sélection ?** Ton deuxième message demandait de ne pas stocker chaque mesure ;
   ton cahier initial parle d'observations « sélectionnées » sans critère. Pour l'instant, M1 reçoit chaque mesure
   terminée mais seulement ses champs scientifiques.
2. **M2 : une tentative par itération ?** Même question : pour l'instant oui.
3. **Efficacité jugée une seule fois.** Dans real3, une consigne est notée efficace après l'itération 1, puis
   l'agent a 3 échecs du même type à l'itération 2. Réévaluer à chaque itération en gardant l'historique ?
4. **Procédure M4 jamais réessayée si la réponse du LLM est inutilisable** (real3 : aucune procédure). Réessayer au
   prochain échec du même type, ou seulement le noter ?
5. **Statut épistémique déclaré par l'agent.** L'agent d'analyse ne le réexamine pas (real3 : une généralisation
   fausse enregistrée comme `observation`). Lui faire ajouter une note, sans écraser le statut ?
6. **Mode en ligne et persistance.** En mode en ligne, Flowcept n'écrit plus le JSONL complet (vérifié). Une
   provenance durable demande alors MongoDB (profil `mongo`, non testé ici).

## Suggestions non implémentées (seulement avec ton accord)

- Recherche par vecteurs (aujourd'hui : clé exacte puis mots-clés).
- Ablations au-delà des quatre modes (sans M2/M4, sans M3, analyse provenance seule, mémoire seule).
- Harnais d'évaluation automatique sur les six scénarios (l'injection de fautes existe, pas encore les scripts de
  mesure).

## Vérifié ici, et ce qui ne l'est pas

Vérifié dans l'environnement de test (Redis local, Flowcept 1.0.5, LangGraph, RDKit, ASE ; xTB remplacé par un
calculateur simulé) :

- `tests/test_pgmem.py` : 16 tests sur la mémoire et l'agent d'analyse.
- `tests/test_pgmem_agent_loop.py` : le vrai graphe de GAIED, 2 agents × 3 itérations, LLM simulé, agent d'analyse
  en parallèle ; 10 vérifications.
- `tests/test_pgmem_flowcept.py` : le canal en ligne de Flowcept reçu pendant l'exécution, et le rapport après
  exécution.
- `run-07.py` complet (Parsl, Academy, Flowcept hors ligne), 2 agents de même graine, 2 itérations, LLM simulé,
  agent d'analyse dans un processus séparé : la campagne se termine normalement. Résultats :
  - M1 : 6 entrées ; M2 : 20 ; M3 : 2 ; M4 : 1.
  - 11 calculs pour 8 molécules distinctes : les doublons sont dus aux deux agents qui tournent en parallèle, et ils
    sont détectés.
  - 7 consignes créées, 6 remises, 4 appliquées, 5 jugées efficaces.
  - Le rapport après exécution est produit à partir des JSONL de Flowcept.

Non vérifié :

- Les campagnes Docker sur le PC distant, avec le vrai xTB et Qwen.
- La lecture de provenance depuis MongoDB (`DbProvenance`).

## Essai rapide sur le PC distant (depuis `D:\Research\GAIED\docker`)

Le LLM simulé et le vrai xTB : quelques minutes. `SCRIPTED_DELAY_S=2` fait attendre le LLM simulé 2 s par réponse ; sans ce délai, les agents finissent avant que les consignes de l'agent d'analyse leur parviennent.

```powershell
docker compose run --rm --no-deps -e REDIS_HOST=redis gaied python tests/test_pgmem.py
docker compose --profile analysis run -d --rm --no-deps --name pgmem-analyzer -e PGMEM_NAMESPACE=quick1 -e LLM_PROVIDER=scripted -e OLLAMA_MODEL=scripted analyzer
docker compose run --rm --no-deps -e PGMEM_MODE=full -e PGMEM_NAMESPACE=quick1 -e LLM_PROVIDER=scripted -e OLLAMA_MODEL=scripted -e SCRIPTED_DELAY_S=2 -e "SEEDS=CNC(N)=O,CNC(N)=O" -e MAX_ITERATIONS=3 -e STABLE_REPORTS=2 gaied
docker compose run --rm --no-deps -e PGMEM_NAMESPACE=quick1 gaied python -m pgmem.show
docker compose run --rm --no-deps -e PGMEM_NAMESPACE=quick1 -e LLM_PROVIDER=scripted -e OLLAMA_MODEL=scripted analyzer python -m pgmem.analyzer --post
docker stop pgmem-analyzer
```

Avec Qwen, remplacer `-e LLM_PROVIDER=scripted -e OLLAMA_MODEL=scripted` par
`-e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=qwen3.5:9b -e LLM_MODEL=qwen3.5:9b`, et changer de
`PGMEM_NAMESPACE`.

Pour l'analyse en ligne de la provenance Flowcept, ajouter
`-e FLOWCEPT_SETTINGS_PATH=/app/docker/flowcept_settings_online.yaml` aux deux commandes, celle de l'agent d'analyse
et celle de la campagne.
