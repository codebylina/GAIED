# S3M — Simple Shared Scientific Memory (premier prototype)

Branche `claude/confident-hypatia-6pjvms`, construite à partir de `docker-local`. Code : `s3m/` (bibliothèque
générique), `s3m/domains/chemistry.py` et `s3m/integrations/gaied.py` (branchement GAIED),
`tests/test_s3m.py`.

## 1. Principe

> Les agents partagent leurs expériences, apprennent des résultats précédents et vérifient les leçons
> avant de les réutiliser.

Trois choses sont séparées :

| | Où | Qui l'écrit |
|---|---|---|
| **Ce qui s'est réellement passé** : les observations | mémoire expérimentale | le code du workflow, jamais un LLM |
| **Ce qu'on en a appris** : les leçons | mémoire des leçons | les agents (le LLM propose) |
| **Ce qu'on a le droit de réutiliser** : les leçons `verified` | statut de la leçon | des règles déterministes |

Flowcept garde la trace complète de l'exécution. S3M ne garde que ce qui sert à décider, et s'y
rattache par les identifiants (section 5).

## 2. Architecture

```
  Agent A        Agent B        Agent C          (GAIED : Academy + LangGraph + LLM + xTB)
     |  avant d'agir : briefing, statut      |  après : observation, leçon
     v                                       v
 +---------------------------- S3M (une instance Redis) -----------------------------+
 |  1. Coordination           2. Mémoire                        3. Vérification     |
 |  claim / release /         observations (faits, append-only)  règles             |
 |  renew / get_status        leçons (pending / verified /        déterministes      |
 |  (verrous à expiration)    rejected / needs_review)            + revue humaine    |
 +-----------------------------------------------------------------------------------+
     |  chaque opération qui influence une décision est écrite comme une tâche Flowcept
     v
  Flowcept (provenance) : tâches xTB, appels LLM, nœuds LangGraph, opérations s3m.*
```

## 3. Les trois composants

### 3.1 Coordination (`s3m/coordination.py`)

| Opération | Rôle |
|---|---|
| `claim(task, agent)` | réserve une tâche (`SET NX PX`) ; renvoie un **jeton** secret, ou `None` si un autre agent la tient |
| `release(task, token)` | libère, **atomiquement** et seulement si le jeton est toujours le propriétaire (script Lua) |
| `renew(task, token)` | prolonge, même règle ; un calcul long renouvelle sa réservation en arrière-plan |
| `S3M.get_status(...)` | `success`, `failed` (scientifique), `running`, `retryable` (panne technique), `unknown` |

Une tâche est identifiée par `condition/entité`, où la condition regroupe la méthode et ses
paramètres. La même molécule calculée avec d'autres paramètres xTB est donc une autre expérience.

`S3M.get_or_compute(entité, compute, method=..., params=...)` combine les trois composants :

1. le résultat existe : il est **réutilisé** ;
2. un **échec scientifique** est connu : on ne relance pas (`KnownFailure`) ;
3. un autre agent calcule déjà : on **attend** son résultat ;
4. sinon : on réserve, on calcule, on **enregistre l'observation avant de libérer**, puis on libère.

Il existe une version `async` (GAIED) et une version synchrone (`get_or_compute_sync`).

**Échec scientifique ou panne technique.** C'est le domaine qui classe l'erreur
(`Domain.classify_failure`). Une erreur scientifique (SMILES invalide, géométrie impossible,
non-convergence) n'est jamais relancée. Une panne technique (crash, timeout, nœud perdu) peut être
relancée jusqu'à `max_technical_attempts` fois.

**Pub/Sub n'est pas la source de vérité.** Chaque événement est écrit dans un stream Redis
(`events`, l'historique) et publié sur un canal (`notify`), pour des observateurs en direct comme
l'analyse de Nadjiba. Une notification manquée ne fait rien perdre : l'état est toujours relu dans Redis.

### 3.2 Mémoire

**Observations** (`s3m/observations.py`) : les faits.

- Elles sont **en ajout seul** : une nouvelle mesure crée une nouvelle observation et n'écrase rien. L'écriture se fait en `SET NX`, donc elle est idempotente.
- Leur identifiant est **l'identifiant de la tâche Flowcept** qui les a produites.
- Chacune porte : l'entité, la méthode, les paramètres, le statut, la valeur ou l'erreur, le type d'échec, l'agent, les dates, et les ids Flowcept.

**Leçons** (`s3m/lessons.py`) : les interprétations.

| Champ | Utilité |
|---|---|
| `id`, `statement`, `type` (`success`, `failure`, `strategy`, `constraint`) | ce que dit la leçon |
| `claim` | la forme **vérifiable** de la leçon (section 4), ou `{"kind": "none"}` |
| `evidence_ids` | les observations sur lesquelles elle repose (ids Flowcept) |
| `condition` | la condition expérimentale des preuves |
| `created_by`, `endorsed_by` | l'auteur, et les agents qui ont proposé la même leçon |
| `status`, `status_reason`, `verification_method`, `verification` | le verdict, sa raison chiffrée et ses détails |
| `provenance` | les ids Flowcept de la proposition et de la vérification |

- Chaque changement de statut est ajouté à l'**historique** de la leçon, avec sa date et sa raison.
- Rien n'est supprimé : une leçon rejetée reste en mémoire, pour qu'on ne refasse pas la même erreur.
- **Une même leçon n'est stockée qu'une fois.** Si un agent repropose la même affirmation vérifiable sous la même condition, elle est fusionnée avec la leçon existante : ses preuves s'ajoutent, l'agent est noté dans `endorsed_by`, et la leçon est revérifiée.

**Transformer les conclusions en leçons** (`s3m/proposer.py`). Le module construit le prompt et lit la
réponse, **sans appeler de LLM lui-même** : chaque workflow utilise son propre client. Les observations
sont montrées au LLM sous des références courtes (`o1`, `o2`…), qui sont ensuite remplacées par les
vrais identifiants.

### 3.3 Vérification (`s3m/verification.py`)

Un LLM **propose** une leçon. Il ne décide **jamais** si elle est vraie : le statut vient de règles
explicites, calculées sur les observations.

| Règle | Échec → |
|---|---|
| la leçon cite des observations, et toutes existent | `rejected` |
| elle a une affirmation vérifiable | `needs_review` |
| ses preuves viennent d'une seule condition comparable | `needs_review` |
| **sur les preuves citées**, l'affirmation est vraie | `rejected` |
| **sur toutes les observations comparables de la mémoire**, elle reste vraie (aucune donnée non citée ne la contredit) | `needs_review` |
| assez de preuves (`min_per_group` entités par groupe, 2 par défaut) | `needs_review` |

Types d'affirmations fournis :

- `comparison` — `{"better": A, "worse": B}` : A a une meilleure valeur que B (moyenne par entité).
- `feature_effect` — `{"feature": "n_nitrile", "op": ">", "value": 0, "effect": "higher"}` : les
  entités qui vérifient la condition ont une valeur moyenne plus haute (ou plus basse) que les autres.

Un workflow peut ajouter ses propres types avec `Verifier.register(kind, fonction)`.

- **Revérification.** Quand de nouvelles observations arrivent, les leçons `verified` et `needs_review` sont revérifiées (`recheck`, un seul vérificateur à la fois grâce à la coordination). Une leçon vérifiée que les nouvelles données contredisent repasse en `needs_review`.
- **Revue humaine.** `python -m s3m review NS LESSON verified "raison"`. Une décision humaine n'est jamais écrasée par la revérification automatique.

> Limite assumée : la règle est une vérification de bon sens (sens de la différence des moyennes,
> taille minimale des groupes), pas un test statistique. Deux agents qui utilisent le même modèle et
> les mêmes données ne sont pas des validations indépendantes : `endorsed_by` n'est pas utilisé pour
> valider.

### 3.4 Ce que voient les agents (`S3M.briefing`)

Le briefing est un texte borné (`max_chars`), propre à une condition, ajouté au prompt :

1. les meilleurs résultats de l'équipe ;
2. les molécules en cours de calcul chez d'autres agents ;
3. les échecs scientifiques ;
4. **les leçons `verified`**, les seules présentées comme des connaissances ;
5. les leçons `rejected`, présentées comme des avertissements, avec leur raison ;
6. les leçons `needs_review`, présentées comme des questions ouvertes à tester.

Les leçons `pending` ne sont jamais montrées.

## 4. Lien avec Flowcept

| Lien | Comment |
|---|---|
| observation ↔ tâche xTB | `observation.id` = `task_id` de la tâche Flowcept `compute_ionization_energy` |
| leçon → observations | `evidence_ids` = des `task_id` Flowcept |
| leçon → sa proposition et sa vérification | `provenance.proposal_task_id` et `verification_task_id` = des tâches Flowcept `s3m.*` |
| opérations mémoire → campagne | les tâches `s3m.reuse`, `s3m.wait`, `s3m.blocked`, `s3m.propose_lesson`, `s3m.verify_lesson`, `s3m.briefing` portent `campaign_id`, `workflow_id` et l'agent ; `s3m.briefing` porte aussi le nœud LangGraph et les ids des leçons montrées |
| provenance → mémoire | `python -m s3m ingest NS output/flowcept_buffer_*.jsonl` reconstruit les observations à partir des JSONL Flowcept, de façon idempotente |

Ainsi, depuis la provenance, on sait **quelles leçons un agent a lues avant de décider**, et quels
calculs ont été évités ou attendus.

`parent_task_id` est rempli quand le code tourne dans une action Academy. La boucle de campagne de
GAIED n'en est pas une, et les tâches xTB d'origine n'ont pas non plus de parent. Le lien passe alors
par `campaign_id` et `workflow_id`, plus le nœud pour les briefings.

## 5. Organisation dans Redis

Tout est rangé sous `s3m:{namespace}:`.

| Clé | Contenu |
|---|---|
| `coord:task:{condition}/{entité}` | réservation (jeton, expiration) |
| `obs:{id}`, `obs:all`, `conds`, `cond:{c}:info` | observations, index temporel, conditions |
| `cond:{c}:ent:{entité}`, `cond:{c}:best`, `cond:{c}:failed`, `cond:{c}:technical` | index par condition |
| `lesson:{id}`, `lesson:{id}:history`, `lessons:status:{s}`, `lessons:queue`, `lessons:index` | leçons |
| `events` (stream), `notify` (pub/sub), `stats` (compteurs) | historique, notifications, mesures |

Il n'y a ni MongoDB, ni base vectorielle, ni recherche sémantique : la recherche se fait par
condition, statut et entité.

## 6. Intégration dans GAIED (crochets `[s3m]`)

| Où | Quoi |
|---|---|
| `agent_on_startup` | crée la mémoire de l'agent ; ses opérations sont écrites dans la provenance Flowcept de l'agent |
| `compute_ionization_energy` | `get_or_compute` : réutilisation, attente, échec connu, ou calcul sous réservation |
| `_record_xtb_task` | ajoute les paramètres xTB (`used.xtb_config`) à l'enregistrement Flowcept |
| `plan`, `tool_calling`, `update` | ajoutent le briefing au prompt ; `tool_calling` liste aussi les molécules de **toute l'équipe** |
| `conclude` | conclusions → leçons (1 appel LLM supplémentaire), puis vérification immédiate |
| `run-07.py` | vérification finale et résumé de la mémoire |

Variables d'environnement (`docker/docker-compose.yml`) :

| Variable | Défaut | Rôle |
|---|---|---|
| `S3M` | `true` dans Docker | `false` = GAIED original (la référence) |
| `S3M_NAMESPACE` | vide | vide = une mémoire par campagne ; un nom fixe (ex. `gaied`) = mémoire partagée entre campagnes |
| `S3M_LESSONS` | `true` | `false` = pas de leçons, seulement coordination et observations |
| `S3M_CLAIM_TTL` | `300` | durée (s) avant qu'une réservation d'agent planté expire |
| `S3M_BRIEFING_CHARS` | `2500` | taille maximale du texte mémoire dans un prompt |

## 7. Utiliser S3M dans un autre workflow

```python
from s3m import S3M, Domain, SCIENTIFIC, TECHNICAL

class MyDomain(Domain):
    name, objective, direction = "material", "band_gap", "max"
    def canonical(self, raw):  return raw.strip().upper() or None
    def features(self, e):     return {"n_atoms": len(e)}
    def classify_failure(self, error):
        return SCIENTIFIC if "unstable" in error else TECHNICAL

mem = S3M.connect(MyDomain(), namespace="campaign-42", agent="agent-1",
                  provenance_sink=interceptor.intercept_task)          # optional: Flowcept
value = mem.get_or_compute_sync("GaN", lambda task_id: simulate("GaN", task_id),
                                method="dft", params={"functional": "PBE"})
text = mem.briefing("dft", {"functional": "PBE"})                       # for the next prompt
mem.propose_lesson("N-rich materials have a larger gap", evidence_ids=[...],
                   claim={"kind": "feature_effect", "feature": "n_atoms", "op": ">", "value": 2,
                          "effect": "higher"})
mem.verify_pending()
```

## 8. Commandes

```bash
# GAIED avec S3M (depuis docker/) ; S3M=false pour la référence
docker compose run --rm gaied
docker compose run --rm -e S3M=false gaied
docker compose run --rm gaied python docker/summarize_prov.py      # doublons + opérations mémoire
docker compose run --rm gaied python -m s3m show                   # mémoire de la dernière campagne
docker compose run --rm gaied python -m s3m lessons <namespace>    # leçons, preuves, historique
docker compose run --rm -e S3M_TEST_REDIS=redis:6379 gaied python tests/test_s3m.py
```

## 9. Mesures disponibles pour l'évaluation

| Mesure | Source |
|---|---|
| simulations redondantes | `summarize_prov.py` → « DUPLICATE computations » (tâches xTB Flowcept) |
| calculs évités, attentes, échecs non relancés | compteurs `stats` ; tâches `s3m.reuse`, `s3m.wait`, `s3m.blocked` |
| leçons par statut, raisons des rejets | `python -m s3m lessons`, `report()` |
| leçons montrées aux agents, et à quel nœud | tâches `s3m.briefing` (`lesson_ids`, `label`) |
| coût | appels LLM (dont l'extraction des leçons) dans la provenance ; durée des opérations `s3m.*` |

## 10. Test de bout en bout réalisé

Configuration : GAIED réel, avec xTB, Parsl, Academy, Flowcept et Redis. Deux agents partent de la
même graine (`CNC(N)=O`), pour 3 itérations. Le LLM est **simulé par script** : il propose à chaque
appel 3 molécules d'une liste fixe, en évitant celles qui figurent dans « Simulated Molecules ». Pour
les leçons, il produit à chaque fois une affirmation `feature_effect`, une comparaison fausse et une
comparaison vraie. Ces chiffres valident donc **le fonctionnement**, pas l'intérêt scientifique.

| Campagne (2 agents, 3 itérations, 18 molécules demandées) | Calculs xTB | Molécules distinctes | Doublons | Résultats réutilisés | Attentes | Échecs connus non relancés |
|---|---|---|---|---|---|---|
| GAIED original (`S3M=false`) | 18 | 8 | **10** | – | – | – |
| GAIED + S3M, mémoire neuve | 13 | 13 | **0** | 5 | 3 | 0 |
| GAIED + S3M, 2ᵉ campagne sur la même mémoire (`S3M_NAMESPACE` fixe) | 4 | 4 | **0** | 12 | 0 | 2 |

Leçons (campagne avec mémoire neuve) : 18 propositions fusionnées en leçons uniques, dont 3 vérifiées,
5 rejetées et 3 en `needs_review` au moment de leur vérification. Les comparaisons fausses du faux LLM
sont toutes rejetées, avec les valeurs mesurées comme raison.

Reconstruction depuis la provenance : `python -m s3m ingest` sur les JSONL de la campagne sans S3M
recrée les 18 observations ; une seconde ingestion n'en ajoute aucune.

Le test vérifie aussi :

- chaque tâche xTB de Flowcept est une observation, avec **le même id** ;
- les preuves des leçons vérifiées sont des tâches xTB Flowcept ;
- les vérifications et les briefings sont des tâches Flowcept rattachées à la campagne ;
- les comparaisons fausses sont rejetées, avec les valeurs mesurées comme raison.

## 11. Ajouts par rapport à la proposition initiale

Ces ajouts restent dans le cadre de la proposition :

- échec **scientifique** ou **technique** (relance limitée), classé par le domaine ;
- jeton de réservation et libération/prolongation atomiques (Lua), avec renouvellement automatique pendant un calcul long ;
- condition expérimentale = méthode + paramètres, ce qui définit la comparabilité ;
- vérification sur les preuves citées **et** sur toutes les observations comparables (contre les preuves choisies sur mesure) ;
- revérification quand de nouvelles données arrivent ; revue humaine qui n'est jamais écrasée ;
- fusion des leçons identiques (`endorsed_by`) ;
- reconstruction de la mémoire depuis les JSONL Flowcept (`ingest`) ;
- enregistrement des opérations mémoire dans Flowcept, pour l'analyse en ligne (travail de Nadjiba).

## 12. Limites et suite

- **Pas encore de vraies campagnes.** Tout a été testé avec un LLM simulé. Il reste à mesurer avec
  Qwen ou une API : la qualité des leçons extraites, et l'effet des leçons vérifiées sur les choix de
  molécules.
- **Une vérification volontairement simple.** Les leçons sans affirmation vérifiable restent en
  `needs_review` : elles demandent une expérience ou un humain.
- **Le coût d'un appel LLM supplémentaire** par itération pour extraire les leçons
  (`S3M_LESSONS=false` pour le supprimer).
- **Pas de mécanisme dédié pour les corrections de l'analyse en ligne.** Pour l'instant, elle peut
  écrire une leçon, ou une revue avec `review(...)`. Les directives dédiées ont été reportées, comme
  dans la proposition.
- **La mémoire n'est pas l'état d'exécution.** Après une interruption, les agents retrouvent les
  connaissances de l'équipe, mais pas l'état LangGraph de leur boucle. C'est le sujet du PFE.
