# Mémoire partagée à quatre couches et analyse en ligne de la provenance — prototype pour le workshop

Branche `Four-layer-memory`, construite à partir de `docker-local`. Ce document décrit ce qui est
**implémenté et testé**. La section 10 liste ce qui ne l'est pas.

> Question de recherche : comment une mémoire partagée persistante et une analyse fondée sur la
> provenance peuvent-elles aider des agents scientifiques à apprendre de l'expérience collective et à
> mieux décider ?

## 1. Vue d'ensemble

```
 Agents GAIED (Academy + LangGraph + LLM + xTB)
   plan ─ tool_calling ─ simulate ─ conclude ─ critique ─ update
     │ contexte (lecture)       │ get_or_compute      │ propositions M3     ▲ consignes M4 (boîte durable)
     ▼                          ▼                     ▼                     │
 ┌──────────────────────── Mémoire partagée (Redis, un espace de noms) ──────────────────────────┐
 │ M1 Résultats      M2 Expérience &      M3 Connaissance      M4 Consignes                      │
 │ scientifiques     diagnostics          scientifique         opérationnelles                   │
 │ ───────────── coordination : verrous Lua, reprise, politiques de relance ──────────────────  │
 │ ───────────── cycle de vie : versions, dédoublonnage, contradictions, obsolescence ────────  │
 │ flux d'événements (Redis Stream, durable) + notifications Pub/Sub                             │
 └───────────────────────────────────────────────────────────────────────────────────────────────┘
     ▲ références (task_id, workflow_id, campaign_id)            ▲ constats M2, propositions M3, consignes M4
     │                                                           │
 Flowcept (seule source de vérité de la provenance)      Service d'analyse en ligne (processus séparé)
   intercepteurs Academy / LangGraph / xTB  ── MQ Redis (Pub/Sub) ─► écoute en direct
   DocumentInserter ─► MongoDB ─────────────── Flowcept.db.task_query ─► rattrapage durable (repère)
   buffers JSONL ───────────────────────────── Flowcept.read_buffer_file ─► analyse hors ligne
```

| Composant | Responsabilité | Ce qu'il ne fait pas |
|---|---|---|
| **Flowcept** | trace complète de l'exécution | – (inchangé) |
| **M1–M4** (`memory/layers/`) | ce qui est réutilisable pour décider | copier la provenance |
| **Coordination** (`memory/coordination.py`) | un calcul au plus par expérience, relances, verrous | stocker des connaissances |
| **Cycle de vie** (`memory/lifecycle.py`, `store.py`) | versions, dédoublonnage, expiration, archivage, métriques | supprimer quoi que ce soit |
| **Service d'analyse** (`analysis/`) | détecter, diagnostiquer, proposer, mesurer l'effet | être une 5ᵉ couche ; établir une connaissance |
| **Intégration GAIED** (`memory/integrations/gaied.py`) | brancher le tout dans les 6 nœuds LangGraph | modifier la logique scientifique d'Amal |

L'état d'exécution LangGraph et les points de reprise **ne sont pas** des couches de mémoire. Aucune
classe ne les stocke.

## 2. Les quatre couches (`memory/schemas.py`, `memory/layers/`)

Tous les enregistrements partagent les mêmes champs de base : `id`, `kind`, `status`, `namespace`,
`version`, `created_by`, les dates, une clé de dédoublonnage `key`, `provenance` (références Flowcept,
avec un `task_id` obligatoire), `memory_refs` (`"M1:<id>"`…), `tags` et `expires_at`. Ils sont validés
à chaque écriture.

### M1 — Résultats scientifiques (`ResultRecord`)
- Mesures réussies, et **résultats négatifs scientifiques valides** (`outcome="negative"`, avec une
  raison scientifique : structure impossible, non-convergence…). **Une panne technique n'est jamais un
  résultat M1** : le schéma le refuse, elle va en M2.
- L'identifiant est **le `task_id` de la tâche Flowcept** du calcul. Mémoire et provenance partagent
  donc les mêmes identifiants, et l'écriture est idempotente.
- Ajout seul : une nouvelle mesure crée un nouvel enregistrement. Une mesure répétée qui diverge des
  précédentes déclenche une anomalie M2 ; les deux mesures sont gardées.
- Comparabilité : `condition` = empreinte de la méthode et des paramètres xTB.

### M2 — Expérience et diagnostics (`ExperienceRecord`)
- **Types** : `incident`, `anomaly`, `recurring_failure`, `redundant_computation`,
  `inefficient_exploration`, `low_diversity`, `unused_knowledge`, `knowledge_conflict`,
  `ineffective_guidance`, `diagnostic_hypothesis`, `confirmed_cause`, `attempted_action`,
  `recovery_episode`.
- **Dédoublonnage par signature** : le même problème revu incrémente `occurrences`.
- **Trois champs bien séparés** : `observed_facts`, `candidate_explanations` (des hypothèses) et
  `confirmed_cause`. Ce dernier n'est rempli que par une règle explicite ou un humain : le schéma
  refuse une cause confirmée sans statut `confirmed`.
- Les preuves sont des références Flowcept et mémoire ; les traces ne sont jamais copiées.
- **Distinctions respectées** : un épisode de reprise (une relance réussie) est en M2 ; la stratégie de
  reprise réutilisable est en M4.

### M3 — Connaissance scientifique (`KnowledgeRecord`)
- **Types** : `observation`, `hypothesis`, `claim`, `finding`, `rule`, `constraint`,
  `negative_finding`, `uncertainty`.
- **Chaque élément porte** : un énoncé, une affirmation vérifiable optionnelle, une portée
  (`condition`, `scope`), une applicabilité, des preuves `supporting` et `contradicting` (identifiants
  M1, donc de tâches Flowcept), une `confidence` et une `source` (agent, analyse ou humain).
- **Statut épistémique**, décidé par des règles déterministes (`Evaluator`), jamais par un LLM :

| Statut | Condition |
|---|---|
| `proposed` | pas encore vérifiable, ou pas assez de données |
| `supported` | les mesures citées **et** toutes les mesures comparables vont dans son sens |
| `validated` | soutenue, **et** confirmée par des mesures faites **après** sa proposition (règle de pré-enregistrement), ou validée par un humain |
| `contested` | de nouvelles mesures la contredisent : elle est réévaluée, **jamais supprimée**, et un épisode M2 `knowledge_conflict` est créé |
| `refuted` | ses propres preuves la contredisent |
| `stale` | non vérifiable et trop ancienne (cycle de vie) |

- **Affirmations vérifiables** : `comparison`, `feature_effect`, `threshold`, `negative_region`.
- Une décision humaine (`python -m memory review`) verrouille le statut contre les changements
  automatiques ; l'évaluation reste enregistrée.

### M4 — Consignes opérationnelles (`GuidanceRecord`)
- **Types** : `strategy`, `recommendation`, `instruction`, `recovery_strategy`.
- **Chaque élément porte** :
  - une applicabilité (nœuds, condition) et ce qui l'a déclenché ;
  - des actions structurées et vérifiables : `avoid_entities`, `avoid_region`, `prefer_region`,
    `diversify`, `reuse_results`, `use_knowledge`, `retry_policy`, `reduce_parallelism`, `free_text` ;
  - ses limites, ses preuves et ses cibles (des agents, ou `all` pour toute la campagne) ;
  - un statut de validation : `proposed → active → validated / ineffective → retired`.
- **Livraison durable** : une boîte de réception par agent, en Redis Stream, avec un curseur stocké
  dans Redis.
- **Suivi par couple (consigne, agent)** : `delivered` (montrée dans un prompt), `acknowledged`
  (l'appel LLM qui l'a reçue s'est terminé), `applied` / `violated` (les propositions suivantes
  respectent ou non les actions structurées).
- Une consigne retirée ou jugée inefficace n'est jamais réactivée en silence.

## 3. Coordination (`memory/coordination.py`)
- **Verrous** : `SET NX PX` avec un jeton secret ; libération et prolongation atomiques en Lua (seul le
  propriétaire peut le faire) ; prolongation automatique pendant un calcul long ; expiration si l'agent
  plante.
- **`get_or_compute`** : réutiliser le résultat M1 → refuser un négatif connu → refroidissement en
  cours → attendre l'agent qui calcule → sinon réserver, calculer, enregistrer, libérer.
- **Échecs** (classés par le domaine) :

| Classe | Traitement | Enregistré en |
|---|---|---|
| scientifique | résultat négatif, pas de relance | M1 |
| transitoire | relances avec attente exponentielle dans l'appel, puis refroidissement court **qui expire** | incident M2 |
| permanent | pas de relance immédiate, refroidissement long qui expire aussi | incident M2 (sévérité haute) |

  Un échec technique n'empêche donc jamais un calcul futur. Chaque tentative est une tâche Flowcept
  distincte.
- **Messagerie** : Redis Streams pour ce qui doit être durable (événements de la mémoire, boîtes de
  réception M4) ; Pub/Sub pour les notifications légères, jamais utilisé comme source de vérité.
- **Concurrence** : écritures optimistes (`WATCH`/`MULTI`) avec versions ; dédoublonnage atomique
  (`HSETNX`).

## 4. Gestion de la mémoire
| Fonction | Implémentation |
|---|---|
| Stockage typé et validation | dataclasses + `validate()` (`schemas.py`) |
| Recherche exacte | `ResultsMemory.lookup` (entité canonique + condition) |
| Recherche contextuelle, filtres, classement | index par métadonnées (`Store.find`), score BM25 + statut + priorité + récence (`retrieval.py`) |
| Recherche sémantique | **non** : vocabulaire très lexical (SMILES, descripteurs, énoncés courts) ; des embeddings ajouteraient un modèle et une dépendance sans besoin démontré |
| Construction du contexte et budget de tokens | `ContextBuilder` : quotas par couche selon le nœud LangGraph ; statut épistémique affiché pour chaque connaissance |
| Dédoublonnage et consolidation | clés de dédoublonnage par couche ; fusion des preuves et des propositions répétées ; vue consolidée par entité en M1 ; historique conservé |
| Versions et références de provenance | chaque changement ajoute une version (qui, quand, pourquoi) ; références Flowcept dans chaque enregistrement |
| Validation par les preuves, détection des contradictions | `Evaluator` M3 ; conflits de mesures en M1 |
| Obsolescence, rétention, expiration, archivage | `Lifecycle.run()` (lancé par le service d'analyse ou `python -m memory maintain`) |
| Statistiques d'usage et métriques de qualité | compteurs par élément (`retrieved`, `delivered`, `applied`…), première lecture par agent, `Lifecycle.quality()` |

## 5. Intégration Flowcept (`memory/provenance.py`), vérifiée sur une vraie campagne GAIED

| Voie | API Flowcept réelle | Vérifié |
|---|---|---|
| en direct | MQ Redis (Pub/Sub `interception`), `BaseConsumer` | le consommateur a reçu en direct les 18 tâches xTB, les 22 appels LLM et les nœuds LangGraph |
| durable | `DocumentInserter` → MongoDB, `Flowcept.db.task_query(filter, sort, limit)` | campagne complète en base ; reprise par repère sur `registered_at` |
| hors ligne | `Flowcept.read_buffer_file(path, cleanup_files=False)` | buffers JSONL de `docker-local` |

- **Activation** : `docker/flowcept_settings_online.yaml` active le MQ, le stockage clé-valeur
  (obligatoire avec le MQ) et MongoDB. Les hôtes se règlent avec `MQ_HOST`, `KVDB_HOST`, `MONGO_HOST`.
- **Constat** : en mode MQ, le buffer JSONL ne reçoit que le reste à l'arrêt. La source de vérité est
  alors la base.
- **Ce que la mémoire écrit dans Flowcept** : les opérations qui influencent une décision
  (`memory.reuse`, `memory.waited`, `memory.blocked_*`, `memory.context`, `memory.propose_knowledge`),
  en tâches de sous-type `memory` rattachées au workflow et à la campagne de l'agent.
- **Limite** : `parent_task_id` reste vide, parce que la boucle de campagne GAIED n'est pas une action
  Academy ; le lien passe par `workflow_id` et `campaign_id`. L'erreur
  `on_chain_start … 'SearchState' object has no attribute 'get'` vient du gestionnaire LangGraph de
  Flowcept et existe aussi sans mémoire.

## 6. Service d'analyse en ligne (`analysis/`) — contribution de Nadjiba, prototype
Il tourne dans un processus indépendant (`python -m analysis run`, ou le service Docker `analysis`).
Les agents ne l'attendent jamais.

- **Déclencheurs durables** :
  - un *consumer group* sur le flux d'événements de la mémoire ; chaque événement n'est acquitté
    qu'après avoir été appliqué ;
  - au redémarrage : relecture de ses propres messages en attente, puis reprise de ceux des
    consommateurs morts (`XAUTOCLAIM`) ;
  - après `max_deliveries` échecs, le message part dans la file des messages en échec (`an:dead`).
- **Provenance** : MQ en direct, rattrapage dans la base par repère (stocké dans Redis), JSONL hors
  ligne. L'ingestion est idempotente (identifiants déjà traités mémorisés).
- **Détecteurs** (règles et statistiques simples) :
  - recalculs redondants (provenance) ;
  - exploration redondante (propositions déjà connues) ;
  - échecs techniques récurrents ;
  - directions répétées sans succès (région de descripteurs) ;
  - faible diversité (similarité de Tanimoto) ;
  - stagnation ;
  - connaissance non utilisée (jamais montrée, ou montrée puis ignorée : seules les décisions
    **postérieures** à sa première lecture comptent) ;
  - connaissance contestée ;
  - consignes inefficaces ou obsolètes.
- **Chaque constat sépare** les faits (avec preuves), les explications candidates et la cause
  confirmée (seulement par une règle nommée : par exemple, des recalculs absents de M1 signifient
  « calculs faits hors de la mémoire »).
- **Rétroaction** :
  - épisode M2 ;
  - proposition M3, **toujours `proposed`**, que l'évaluateur de la mémoire juge ensuite ;
  - consigne M4 ciblée sur des agents ou sur la campagne, avec un état de référence mesuré au moment de
    la publication. Une consigne qui repose sur une connaissance non soutenue reste `proposed`.
- **Efficacité** : comportement des cibles après la consigne comparé à la référence (part de
  propositions déjà connues, diversité, meilleur résultat, respect des actions) → `validated` ou
  `ineffective` (avec un épisode M2). Une consigne fondée sur une connaissance devenue contestée est
  retirée.
- **LLM optionnel** (`ANALYSIS_LLM=true`) : une explication candidate de plus, marquée « non
  vérifiée ». Il ne sert jamais à détecter ni à confirmer.

## 7. Intégration dans GAIED (crochets `[memory]`)
| Nœud / étape | Ce que fait la mémoire |
|---|---|
| `plan` | contexte : connaissances (avec statut), consignes, meilleurs résultats, problèmes connus |
| `tool_calling` | molécules connues de **toute l'équipe** (mesurées, négatives, en cours) + contexte ; après la réponse, vérification du respect des consignes |
| `simulate` (`compute_ionization_energy`) | `get_or_compute` : réutilisation, attente, réservation, calcul, enregistrement M1 (id = tâche Flowcept) |
| `conclude` | conclusions → affirmations vérifiables M3 (1 appel LLM, seulement s'il y a ≥ `MEMORY_LESSON_MIN_NEW` nouveaux résultats et qu'aucun autre agent ne le fait déjà) |
| `critique` | contexte orienté problèmes (M2) et connaissances contestées |
| `update` | nouvelles consignes (marquées **NEW**) et connaissances, pour le nouveau plan |

Les consignes arrivent **pendant l'exécution**, à chaque nœud, par la boîte de réception durable.
Si Redis est injoignable, l'agent tourne sans mémoire ; une simulation n'est jamais lancée deux fois à
cause d'une erreur de mémoire.

**Les quatre configurations** (`MEMORY_MODE`) :

| `MEMORY_MODE` | Configuration |
|---|---|
| `none` | GAIED original |
| `basic` | premier prototype de `docker-local` (`memory/baseline/naive.py`, commit `0fd589b`) |
| `structured` | mémoire à quatre couches |
| `structured+analysis` | mémoire à quatre couches + service d'analyse en ligne |

## 8. Installation et déploiement
Docker, depuis `docker/` :
```bash
docker compose up -d redis ollama
docker compose run --rm gaied                                    # MEMORY_MODE=structured
docker compose run --rm -e MEMORY_MODE=none gaied                # référence
# configuration 4 : Flowcept en ligne + MongoDB + service d'analyse
docker compose --profile analysis up -d redis mongo analysis
docker compose run --rm -e MEMORY_MODE=structured+analysis -e FLOWCEPT_SETTINGS_PATH=/app/docker/flowcept_settings_online.yaml gaied
docker compose run --rm gaied python -m memory show              # les quatre couches
docker compose logs -f analysis                                  # constats en direct
```

Les agents et le service d'analyse doivent utiliser le **même `MEMORY_NAMESPACE`**. Par défaut, c'est
`gaied`, partagé entre campagnes : les agents apprennent des exécutions précédentes. Pour isoler une
expérience, donnez un nom par exécution. Pour une API payante, remplissez `docker/.env` comme avant.

## 9. Tests et expériences
| Fichier | Contenu | Démos du cahier des charges |
|---|---|---|
| `tests/test_memory_layers.py` (11) | schémas, store, M1–M4, contexte, cycle de vie | **4** (contradiction → `contested`, pas de suppression) |
| `tests/test_coordination.py` (8) | verrous, `get_or_compute`, relances, refroidissements | **1** (deux agents, un seul calcul), **2** (échec transitoire relancé) |
| `tests/test_analysis.py` (8) | service d'analyse | **3** (redondance détectée via Flowcept + mémoire), **5** (consigne reçue en direct, suivie, efficacité mesurée), **6** (redémarrage sans perte) |
| `tests/test_integration.py` (5) | interface GAIED : Redis indisponible, extraction conditionnelle, mode `basic` | – |

```bash
python tests/test_memory_layers.py   # MEMORY_TEST_REDIS=redis:6379 pour un vrai Redis (sinon fakeredis + lupa)
```
Résultat : les 32 tests passent, avec un vrai Redis comme avec fakeredis
(`test_basic_mode…` demande un vrai Redis).

**Expériences** : `experiments/run_experiments.py` lance les quatre configurations, chacune isolée
(répertoire et espace de noms propres), et calcule les métriques (`experiments/metrics.py`) à partir
de la provenance Flowcept et de la mémoire :
- réutilisation et redondance ;
- qualité d'exploration : meilleur résultat, moyenne du top 5, diversité ;
- exactitude des détections ;
- interventions : publiées, livrées, reçues, appliquées, violées ;
- temps de construction du contexte ;
- coût : appels LLM et durée.

```bash
python experiments/run_experiments.py --repeats 3 --iterations 5 --seeds "CNC(N)=O,CNC(N)=O,CC1=C(O)N=C(O)N1"
```

### Exécution fonctionnelle réalisée (LLM **scripté** : ce ne sont PAS des résultats scientifiques)
Conditions : xTB, Parsl, Academy, Flowcept (MQ + MongoDB) et Redis réels ; 2 agents sur la même graine,
5 itérations, 2 répétitions. Le LLM est un serveur scripté qui propose des molécules d'une liste fixe
et ignore la liste des molécules connues **sauf** s'il reçoit une nouvelle consigne. Ces chiffres
montrent que tout fonctionne de bout en bout, rien de plus.

| métrique | none | basic | structured | structured+analysis |
|---|---|---|---|---|
| calculs xTB | 18 ± 1,4 | 7 | 7 | 10 |
| molécules distinctes | 7 | 7 | 7 | 10 |
| calculs redondants | 11 ± 1,4 | 0 | 0 | 0 |
| résultats réutilisés | – | 4,5 | 6 | 6 |
| appels LLM | 38 | 32 | 38 | 37 |
| contexte (ms, moyenne) | – | – | 3,5 | 3,9 |
| consignes publiées / livrées / appliquées / violées | – | – | 0 | 4,5 / 15 / 0,5 / 7,5 |

Détection des recalculs par le service d'analyse sur la provenance des campagnes sans mémoire
(vérité terrain : la même provenance) : 6/6 détectés, précision 1,0, rappel 1,0, dans les deux
répétitions. Les consignes y sont surtout « violées » parce que le LLM scripté ne suit qu'une consigne
**nouvelle** ; mesurer leur effet réel demande un vrai LLM.

## 10. Limites et suite
- **Aucun résultat avec un vrai LLM.** Il faut lancer `experiments/run_experiments.py` avec Qwen ou une
  API ; avec Ollama sur CPU, compter environ 1 h par campagne de 3 itérations.
- **Docker non lancé ici.** L'environnement de test avait les mêmes dépendances, installées sans
  Docker. Le fichier `docker-compose.yml` est un YAML valide, mais `docker compose config` n'a pas été
  exécuté.
- **Vérification volontairement simple.** Les affirmations M3 sont vérifiées par différence de
  moyennes et taille minimale des groupes, pas par un test statistique. L'efficacité d'une consigne est
  une comparaison avant/après, pas une expérience contrôlée.
- **Exactitude des détections.** Elle n'est mesurée que pour les recalculs, seul cas où une vérité
  terrain existe dans la provenance.
- **La mémoire ne restaure pas l'état LangGraph** après une interruption. C'est le sujet du PFE.
- **Pub/Sub Flowcept non durable.** La durabilité repose sur MongoDB : sans base, l'analyse en ligne ne
  voit la provenance qu'en direct, et en entier seulement à la fin, par les JSONL.

## 11. Fichiers
```
memory/            schemas.py store.py domain.py domains/chemistry.py layers/m1..m4 coordination.py
                   retrieval.py lifecycle.py provenance.py core.py __main__.py baseline/naive.py
                   integrations/gaied.py
analysis/          state.py detectors.py feedback.py diagnosis.py service.py __main__.py
experiments/       run_experiments.py metrics.py
tests/             helpers.py test_memory_layers.py test_coordination.py test_analysis.py test_integration.py
docker/            docker-compose.yml (service analysis), flowcept_settings_online.yaml, summarize_prov.py
mol_design_agents.py, run-07.py   crochets [memory]
```
