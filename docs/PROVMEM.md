# ProvMem : méthode et protocole d'évaluation

Branche `provmem`, partie de `docker-local`. Code : `provmem/` (la mémoire, générale),
`provmem/adapters/chemistry.py` et `provmem/integrations/gaied.py` (le branchement GAIED),
`bench/` (le banc d'essai), `tests/test_provmem.py`.

## 1. Problème et question de recherche

Plusieurs agents LLM mènent ensemble une campagne expérimentale longue avec un budget d'expériences
coûteuses. Une mémoire partagée doit leur dire ce qui a été mesuré, ce qui est en cours, ce que les
mesures permettent de conclure, et quand une conclusion ne tient plus.

Les mémoires d'agents existantes ont trois défauts dans ce cadre :

- elles retiennent ce que les agents **disent**, et non ce qui a été **mesuré** ;
- elles tranchent les contradictions par la récence, l'autorité de l'auteur ou un LLM qui relit du texte ;
- une généralisation hâtive adoptée par l'équipe détourne l'exploration de la région qui la
  démentirait : l'erreur se protège elle-même (cascade d'information).

**Question :** une mémoire peut-elle garder une équipe d'agents cohérente avec les mesures du workflow,
et améliorer ainsi à la fois l'efficacité de l'exploration et la justesse des connaissances partagées ?

## 2. Contributions

1. **Ancrage dans la provenance.** Les faits viennent des enregistrements de provenance du workflow
   (tâches W3C PROV / Flowcept), jamais du texte des agents. La mémoire en tire aussi la coordination :
   avant chaque expérience, le code vérifie si elle est déjà mesurée, déjà échouée ou en cours chez un
   autre agent, et réserve l'entité par un bail.
2. **Couches de confiance fondées sur la classe des preuves.** Une affirmation monte de *hypothèse* à
   *soutenue* puis *établie*, et peut redescendre ou être *réfutée*. Ce qui permet de monter, c'est la
   classe de la preuve, calculée à partir de la provenance :
   - **exploratoire** : mesurée avant l'enregistrement de l'affirmation, donc susceptible de l'avoir suggérée ;
   - **de confirmation** : mesurée après cet enregistrement ;
   - **indépendante** : de confirmation, et mesurée par un autre agent que l'auteur.

   C'est la règle du pré-enregistrement : on ne confirme pas une hypothèse avec les données qui l'ont suggérée.
3. **Un juge LLM ancré dans des tests exécutés.** Le juge réécrit l'affirmation à la portée de ce qui a
   été observé, puis écrit lui-même le test (`def test(df)`) qui la confronte aux observations. Il n'y a
   aucune liste de types d'affirmations. Le code exécute le test dans un bac à sable et classe les
   preuves. Le juge rend ensuite un verdict qui cite les observations. Le code ne décide jamais si une
   affirmation est vraie : il impose seulement quelle classe de preuve peut justifier quelle couche, et
   ignore les citations qui ne viennent pas du test.
4. **Une mémoire active, reliée à la provenance des décisions.** Chaque briefing est enregistré élément
   par élément (« a lu »). Chaque choix d'expérience est relié aux affirmations lues juste avant (« a mené
   à »). Quand une affirmation est réfutée ou rétrogradée, seuls les agents qui l'ont lue sont prévenus.

Cette conception est générale. Un domaine se branche par un adaptateur (`provmem/adapter.py`) qui
fournit :
- la clé d'une entité ;
- la lecture d'une observation dans un enregistrement de provenance ;
- les descripteurs, c'est-à-dire le vocabulaire des portées ;
- l'objectif et les contraintes.

Le stockage (`Store`), la recherche (`Retriever`), le juge et le LLM sont interchangeables.

## 3. Fonctionnement

| Élément | Où | Rôle |
|---|---|---|
| observation | `base.Memory.ingest` | lue dans un enregistrement de provenance par l'adaptateur ; immuable |
| contrôle avant action | `ProvMem.screen` | décisions `known`, `failed_before`, `in_progress`, `run`, `invalid`, `duplicate_in_batch` ; bail atomique (`Store.take_lease`) renouvelé par `heartbeat` |
| affirmation | `ProvMem.submit_conclusion` | une par ligne de la conclusion ; couche `pending` |
| jugement | `ProvMem.maintain` | formalisation, test, classement des preuves, verdict, couche ; dossier et journal |
| réévaluation | `ProvMem._reevaluate` | les tests sont relancés quand des observations arrivent ; le juge n'est rappelé que si une **nouvelle preuve de confirmation** apparaît |
| briefing | `ProvMem.briefing` | Goal, directives, avis reçus, puis par sections : résultats, en cours, échecs, établies, soutenues, « à tester », réfutées, conseils invérifiables ; choix à tour de rôle entre sections jusqu'au budget |
| directives | `put_directive` | ponctuelles (délivrées une fois par agent) ou permanentes, avec expiration ; point d'entrée des analyseurs externes |
| outils (pull) | `tool_specs`, `call_tool` | `lookup`, `search`, `evidence`, `best` |

Règles de couche (`ProvMem._target_layer`), à partir du verdict et des citations valides :
- **contredite**, avec au moins une preuve citée : `rejected` si l'affirmation vient d'arriver, sinon `refuted` ;
- **soutenue**, avec au moins une preuve citée : `established` si des preuves indépendantes viennent
  d'au moins `min_independent` autres agents ; sinon `supported` s'il y a une preuve de confirmation ;
  sinon `hypothesis`. Une affirmation ne redescend jamais sur un verdict favorable ;
- **non concluante**, ou verdict sans citation valide : une nouvelle affirmation devient `hypothesis`.
  Une affirmation `established` contre laquelle existe une preuve de confirmation repasse `supported`.

Le mode « revue » (`review`) ajoute un sceptique, puis un second verdict, avant toute transition à fort
impact : vers `established`, ou depuis une couche affirmée.

## 4. Paramètres : aucun n'est réglé par condition

| Paramètre | Valeur | Justification |
|---|---|---|
| `briefing_budget_chars` | fixé par l'expérience, **identique pour toutes les conditions** | sinon on compare des quantités de contexte, pas des mémoires |
| `min_independent` | 1 | définition minimale d'une réplication indépendante ; à faire varier en analyse de sensibilité |
| `lease_ttl_s` | délai sans signe de vie | paramètre système ; un bail est perdu seulement si l'agent ne se manifeste plus |
| `related_k` | fixé par l'expérience, identique pour toutes les conditions | nombre d'éléments voisins comparés (juge, Mem0) |
| lignes de preuve montrées au juge | dérivé de `NUM_CTX` | la fenêtre de contexte du modèle |
| BM25 (k1 = 1,2 ; b = 0,75), RRF (k = 60) | valeurs standard de la littérature | Robertson et Zaragoza (2009) ; Cormack et al. (2009) |
| `test_timeout_s`, `test_repairs` | fixés par l'expérience | bornes d'exécution, sans effet sur les verdicts quand elles ne sont pas atteintes |

Aucun seuil statistique n'est codé : le juge lit les statistiques produites par son propre test.

## 5. Protocole d'évaluation

### Questions et hypothèses

| | Question | Mesures |
|---|---|---|
| **RQ1** | L'ancrage dans la provenance supprime-t-il le travail gaspillé ? | expériences répétées, échecs relancés, expériences évitées, travail répété après une coupure |
| **RQ2** | Les couches de confiance et le juge ancré limitent-ils la propagation d'erreurs ? | part des impressions de connaissance fausses (`false_exposure`), précision des connaissances présentées comme vraies, précision par couche (calibration), portée des affirmations fausses implantées |
| **RQ3** | La mémoire améliore-t-elle la découverte à budget égal ? | regret final et moyen (optimum moins vraie valeur de la meilleure entité trouvée), usage de l'option trompeuse hors de la région de départ |
| **RQ4** | À quel coût ? | appels et caractères LLM par usage (agents, juge, mémoire), taille des briefings, délai de jugement, durée |

### Conditions (`bench/configs/main.json`)

**Références**, réimplémentées dans le même cadre (mêmes agents, même LLM, même recherche, même budget) :

| Condition | Type de mémoire | Travaux représentés |
|---|---|---|
| `none` | aucune mémoire partagée | — |
| `shared_log` | tableau noir : tout ce que l'équipe a produit, le plus récent d'abord | contexte partagé complet |
| `rag` | mémoire épisodique avec recherche | MemoryBank, Generative Agents |
| `mem0` | mémoire gérée par un LLM (ajouter, modifier, supprimer) | Mem0 |

**Ablations** de ProvMem : sans coordination, sans couches, juge sur le texte seul, tout accepter, avec
revue contradictoire, en mode pull (outils) au lieu du push.

### Scénarios

| Scénario | Ce qu'il éprouve |
|---|---|
| `standard` | le cas courant |
| `deceptive` | une option aide dans la région de départ et nuit ailleurs : la généralisation hâtive est fausse, la version limitée à la région est vraie |
| `adversarial` | un agent rapporte des affirmations fausses par construction |
| `crash` | tous les agents perdent leur mémoire vive à mi-campagne ; la mémoire partagée persiste |

### Domaines

- **Synthétique** (`bench/env.py`) : une entité choisit une option par emplacement. La valeur cachée
  comprend des effets principaux, des interactions, du bruit et des paires interdites. La vérité de
  toute affirmation est connue, puisque toute la population est calculable sans bruit. C'est ce qui
  permet de mesurer RQ2.
- **GAIED** (xTB, Qwen 3.5) : la validation réelle de RQ1, RQ3 et RQ4. RQ2 n'y est mesurable que
  qualitativement, faute de vérité terrain.

### Évaluation de la vérité (`bench/oracle.py`)

Le même évaluateur sert pour toutes les conditions. Un LLM écrit `def holds(df)`, qui est exécuté sur
la population complète. Sa justesse est contrôlée sur les affirmations implantées, dont la vérité est
connue par construction. Il faut aussi vérifier à la main un échantillon tiré au hasard.

### Statistiques

Plan apparié : la même graine donne le même environnement et les mêmes points de départ pour toutes les
conditions. On rapporte la moyenne et un intervalle de confiance à 95 % par bootstrap sur les graines,
avec un test de Wilcoxon apparié contre ProvMem. Avec plusieurs comparaisons, il faut corriger (Holm)
avant de conclure.

### Menaces à la validité

- **Le domaine synthétique n'est pas la chimie.** C'est pourquoi GAIED reste nécessaire pour RQ1, RQ3 et RQ4.
- **L'évaluateur de vérité est un LLM.** Sa justesse est mesurée, et un échantillon est vérifié à la main.
- **Le coût du juge.** Avec un LLM local, il peut prendre du retard sur les agents. Ce retard est mesuré
  (`judging_lag_s`) et fait partie du résultat.
- **Les références sont réimplémentées.** Elles suivent la description publiée de chaque méthode, mais
  ne sont pas leur code d'origine.

## 6. Commandes

Banc d'essai, depuis la racine du dépôt (dans le conteneur : `docker compose run --rm --no-deps ... gaied <commande>`) :

```bash
LLM_PROVIDER=scripted python -m bench.run bench/configs/dryrun.json      # vérifie la chaîne, sans LLM
LLM_PROVIDER=scripted python -m bench.analyze results/dryrun
python -m bench.run bench/configs/main.json                               # avec le LLM configuré
python -m bench.analyze results/main
python tests/test_provmem.py
```

GAIED, avec une condition de la même configuration :

```bash
MEMORY_CONDITION=provmem MEMORY_EXPORT=results/gaied/provmem_r1.json NO_IMPROVEMENT_TOLERANCE=0 \
PARSL_MAX_WORKERS=4 SEEDS="CNC(N)=O,CNC(N)=O,CC1=C(O)N=C(O)N1,CC1=C(O)N=C(O)N1" MAX_ITERATIONS=10 \
python run-07.py
```

Sans `MEMORY_CONDITION`, GAIED fonctionne comme l'original. Les branchements dans le code d'Amal sont
marqués `[provmem]`, et le correctif des relances `[gaied-fix]`.
