# Couche de vérification des connaissances au-dessus de Mem0

Branche `mem0-verified`, partie de `mem0-baseline`. Mem0 est utilisé exactement comme dans la
référence : mêmes écritures, même recherche, même nombre de résultats, même budget de prompt. La couche
s'ajoute par-dessus (`memlab/verifier.py`, `memlab/verified_memory.py`).

## Question de recherche et revendication

**Question.** Comment empêcher une mémoire partagée entre agents scientifiques de transformer trop tôt des
observations en connaissances présentées comme établies ?

**Revendication (volontairement étroite).** Nous proposons une couche de vérification qui tient compte de la
provenance et du temps. Elle réévalue en continu les affirmations stockées dans une mémoire partagée
d'agents, contre les observations du workflow, et règle leur statut épistémique à mesure que de nouvelles
preuves arrivent. Ce n'est pas un nouveau système de mémoire : le stockage et la recherche restent ceux de
Mem0.

**Ce qu'elle ne revendique pas.** La vérité scientifique d'une affirmation. Elle vérifie une association
**sur les mesures disponibles**, avec une règle fixée à l'avance. Elle n'établit pas de lien causal.

## Positionnement

| Travail | Vérifie | Quand | Contre quoi |
|---|---|---|---|
| Mem0 | rien (ajout et déduplication par le LLM) | — | — |
| AutoSci, Trust Guard (2026) | chaque écriture | **à l'écriture seulement** | un agent relecteur LLM lit les preuves citées |
| MemTX (2026) | les croyances : validation, contrôle des lectures, retrait en cascade | à l'écriture, puis à la révocation | l'autorité de la source, des règles |
| StatefulDiscovery (EMNLP 2026) | des conclusions calibrées par les preuves | au fil d'une enquête, un état persistant | des analyses exécutées ; **un seul agent**, données figées |
| **Cette couche** | chaque souvenir de la mémoire hôte | **à chaque nouvelle mesure**, sans LLM | **un test figé avant les données qui le confirment, exécuté sur les mesures du workflow ; preuves classées selon la provenance (qui, quand, après avoir vu quoi)** |

Il y a trois différences à défendre.
- **Le temps.** Une règle est pré-enregistrée, et la réévaluation continue ne demande aucun LLM.
  Trust Guard juge une fois ; StatefulDiscovery travaille sur des données figées.
- **La dépendance entre agents.** Une confirmation par un agent qui avait lu l'affirmation n'est pas
  indépendante.
- **La mémoire partagée par plusieurs agents en même temps.** Cela n'existe pas dans StatefulDiscovery.

## Fonctionnement

1. **Mem0 crée un souvenir.** La couche le reçoit avec son auteur et sa **date d'affirmation** : le moment où
   l'agent a écrit le texte dont Mem0 l'a tiré. Ce n'est pas le moment où Mem0 l'a traité.
2. **Formalisation, en un appel au LLM par lot de souvenirs.** Pour chaque souvenir, le LLM écrit :
   - son type :
     - `report` : le résultat d'expériences déjà faites ;
     - `prediction` : le résultat de designs précis, peut-être pas encore mesurés ;
     - `generalization` : une tendance ;
     - `untestable` : ni l'un ni l'autre ;
   - sa portée : une association sur des designs comme ceux mesurés, jamais inversée ;
   - un test `def test(df)` qui renvoie les preuves pour et contre, et **le verdict calculé par une règle
     écrite maintenant** : `supported`, `contradicted` ou `insufficient`. Pour une généralisation, le LLM doit
     comparer à un groupe de référence, si possible à autres caractéristiques égales, et exiger assez de
     preuves compte tenu du bruit.
3. **Évaluation, sans LLM.** Les tests figés sont relancés, dans un seul processus isolé, à chaque nouvelle
   mesure.
4. **Statut** : une fonction du verdict courant et de la classe des preuves (tableaux ci-dessous).
5. **Présentation.** Les souvenirs que Mem0 retrouve sont rangés par statut. Un agent à qui un souvenir a été
   montré comme vrai reçoit un avis si ce statut tombe.

### Pré-enregistrement garanti (réponse au problème 2 de la review)

| Risque | Mesure dans le code |
|---|---|
| des mesures arrivent entre l'affirmation et sa formalisation | le LLM ne voit que les mesures dont la date est ≤ la date d'affirmation (`window_until`), quel que soit le retard |
| la règle est modifiée ensuite | elle est figée avec son empreinte SHA-256, sa version et sa fenêtre ; une règle dont l'empreinte ne correspond plus n'est pas exécutée |
| la réparation d'un test réécrit la règle après coup | la réparation voit la même fenêtre et une erreur **sans valeurs mesurées** (`sanitize`) ; elle garde l'énoncé et le type ; chaque version est conservée (`rules`) |
| vérifier que c'est vrai en pratique | mesure `rule_leak_rate` : part des règles écrites en ayant vu une mesure postérieure à l'affirmation (attendu : 0 ; l'ablation `posthoc` sert de témoin positif) |

### Classes de preuves et indépendance (réponse au problème 1)

La review a raison : « un autre agent » ne veut pas dire « indépendant ». Nous distinguons trois
indépendances.
- **De la source** : une nouvelle mesure d'un design déjà mesuré avant l'affirmation n'est pas une nouvelle
  source.
- **De l'agent** : qui a mesuré.
- **De la décision** : l'agent avait-il lu l'affirmation avant de lancer l'expérience ?

La troisième est obtenue par la provenance de la mémoire elle-même : chaque prompt est journalisé avec les
souvenirs montrés (`briefings`), et chaque mesure a son heure de début (`started_at`).

| Classe | Définition |
|---|---|
| **E** exploratoire | disponible avant l'affirmation, ou nouvelle mesure d'un design mesuré avant elle |
| **C** confirmation | nouveau design, mesuré après, par l'auteur |
| **R** informée | nouveau design, mesuré par un autre agent **à qui l'affirmation avait été montrée** avant l'expérience |
| **I** indépendante | nouveau design, mesuré par un autre agent **qui ne l'avait pas vue** |

**Notre position, à défendre dans le papier.** Une mesure R n'est pas sans valeur. C'est un test de
prédiction : en science, on teste justement une hypothèse qu'on connaît. Mais le choix de l'expérience
dépend de l'affirmation, ce qui ouvre la porte au biais de sélection. Elle justifie donc `supported`, pas
`established`. C'est une question empirique : l'ablation `agentindep` (toute mesure d'un autre agent compte
comme I) dira si la distinction change la précision. La mesure `evidence_share_R` dira combien de
« confirmations par un autre agent » étaient en fait informées.

**Limite.** Seule l'exposition **directe** au souvenir est tracée. Une influence indirecte (B lit un souvenir
dérivé de celui de A, ou une paraphrase stockée sous un autre identifiant) ne l'est pas.

### Statuts et transitions (réponse au problème 4)

Le système enregistre deux choses séparées :
- **l'état des preuves** (`evidence`) : `none` (aucune mesure dans la portée), `insufficient`, `favourable`,
  `contradictory` ;
- **le statut épistémique** qui en découle.

| Type | Verdict de la règle | Preuves | Statut | Montré comme |
|---|---|---|---|---|
| report | aucune mesure dans la portée | none | `unsupported` | aucune mesure enregistrée |
| prediction | aucune mesure dans la portée | none | `open` | à tester |
| report / prediction | supported | favourable | `verified` | vrai |
| report / prediction | contradicted | contradictory | `refuted` | faux |
| report / prediction | insufficient | insufficient | `open` | à tester |
| generalization | supported, preuves E seulement | favourable | `hypothesis` | à tester |
| generalization | supported, avec C ou R | favourable | `supported` | vrai, avec les comptes |
| generalization | supported, I d'au moins `k` autres agents | favourable | `established` | vrai, confirmé par des agents qui ne l'avaient pas vu |
| generalization | contradicted | contradictory | `refuted` | faux |
| generalization | insufficient, alors qu'il était montré comme vrai | insufficient | `contested` | contesté |
| generalization | insufficient ou aucune donnée | none / insufficient | `open` | à tester |
| tous | test impossible ou en échec deux fois | — | `unverifiable` | ne peut pas être vérifié |

La distinction demandée par la review est respectée : « X a donné 3,2 » sans mesure enregistrée est
`unsupported` ; « X aura telle valeur » sans mesure reste `open`. Le statut est recalculé à partir des preuves
courantes, donc une généralisation peut monter, redescendre (`contested`) ou être réfutée. Chaque transition
est datée dans `history`.

### Validité des tests (réponse au problème 3)

Un test qui s'exécute n'est pas forcément un test de l'affirmation. Trois mesures le vérifient :
- `verifier_accuracy_decided` : accord du statut final avec l'évaluateur de vérité, sur les souvenirs
  tranchés ;
- `scope_narrowing_rate` : part des cas où la portée réécrite est vraie alors que l'énoncé d'origine est
  faux, c'est-à-dire une portée rétrécie pour faire passer l'affirmation ;
- un **audit manuel** (`bench/audit_tests.py`) : échantillon stratifié par statut, trois jugements par test
  (portée fidèle, test conforme à l'énoncé, verdict correct). Le score donne chaque proportion avec un
  intervalle de Wilson à 95 %, et le kappa de Cohen entre l'humain et le vérificateur, puis entre l'humain et
  l'évaluateur. La taille de l'échantillon se choisit d'après la largeur d'intervalle voulue, qui est
  affichée.

### Généralisation et causalité (réponse au problème 5)

La portée est formulée comme une association sur des designs comme ceux mesurés. Le test doit comparer, si
possible, à autres caractéristiques égales. C'est une protection partielle contre les facteurs de confusion,
pas une analyse causale. Le papier le dit explicitement.

## Hypothèses et expériences

| # | Hypothèse | Comparaison | Mesures |
|---|---|---|---|
| H1 | la couche réduit les fausses connaissances présentées comme vraies sans perdre trop de vraies | `mem0_verified` vs `mem0` | `final_precision`, `final_recall`, `false_exposure`, `true_impressions` |
| H2 | les mesures du workflow apportent ce qu'un LLM lisant le texte n'apporte pas | vs `mem0_textcheck` | mêmes mesures, et `planted_ever_asserted` |
| H3 | le pré-enregistrement compte | vs `mem0_verified_posthoc` | précision, `rule_leak_rate` (témoin), coût |
| H4 | la distinction E/C/R/I compte | vs `mem0_verified_nolayers` et `mem0_verified_agentindep` | précision par statut, `evidence_share_R`, `established_if_any_other_agent` |
| H5 | les affirmations fausses implantées ne se propagent pas | scénario `adversarial` | `planted_asserted_final`, `planted_ever_asserted`, `planted_impressions` |
| H6 | la couche distingue un effet local d'une généralisation | scénario `deceptive` | `decoy_outside`, statuts des souvenirs sur l'option piège |
| H7 | les erreurs de vérification sont corrigées vite | toutes les conditions vérifiées | `reversals`, `impressions_before_correction`, `verif_lag_s` |
| H8 | une fausse mémoire est pire que pas de mémoire | `mem0` vs `none` | `final_regret`, `regret_at_common_budget` |

**Deux niveaux d'évaluation, présentés séparément :**
1. **le domaine synthétique**, où la vérité est connue : il sert à mesurer la justesse ;
2. **GAIED** : il sert à mesurer l'intégration, le coût et l'utilité (doublons, meilleures molécules). Il ne
   dit pas si chaque connaissance chimique est juste.

**Conditions** (`bench/configs/*.json`, et `MEMORY_CONDITION` pour GAIED) :

| Condition | Rôle |
|---|---|
| `none` | aucune mémoire partagée |
| `mem0` | Mem0 seul, la référence |
| `mem0_textcheck` | un LLM qui juge le texte seul |
| `mem0_verified` | la couche complète |
| `mem0_verified_nolayers` | sans classes de preuves |
| `mem0_verified_agentindep` | indépendance par agent seulement (exposition ignorée) |
| `mem0_verified_posthoc` | sans pré-enregistrement : le LLM réécrit le test avec toutes les données à chaque nouvelle mesure |

`quick.json` contient `none`, `mem0` et `mem0_verified`, sur deux scénarios avec une graine : c'est un essai
peu coûteux. `main.json` contient les 7 conditions, 4 scénarios et 5 graines. Statistiques : moyenne,
intervalle de confiance à 95 % par bootstrap, test de Wilcoxon apparié.

## Lancer (machine distante, depuis `D:\Research\GAIED\docker`)

```powershell
docker compose run --rm --no-deps -e LLM_PROVIDER=ollama -e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=qwen3.5:9b -e LLM_MODEL=qwen3.5:9b -e EMBED_MODEL=nomic-embed-text gaied python -m bench.run bench/configs/quick.json
docker compose run --rm --no-deps -e LLM_PROVIDER=ollama -e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=qwen3.5:9b -e LLM_MODEL=qwen3.5:9b gaied python -m bench.analyze results/verif_quick --reference mem0_verified
docker compose run --rm --no-deps gaied python -m bench.audit_tests results/verif_quick
```

Le fichier `results/verif_quick/audit_tests.csv` se remplit à la main, colonnes `human_*` (`yes` / `no`).
Le score se calcule ensuite avec `python -m bench.audit_tests --score results/verif_quick/audit_tests.csv`.

Tests sans LLM : `python tests/test_verifier.py` (18 tests) et `python tests/test_memlab.py`.

## Limites à dire

- **L'évaluateur de vérité est un LLM.** Sa justesse est mesurée sur les affirmations implantées et
  contrôlée par l'audit manuel.
- **La règle est écrite par un LLM.** Elle peut être trop laxiste ou trop stricte : c'est ce que mesurent la
  précision par statut et l'audit.
- **L'indépendance repose sur l'exposition directe.** L'influence indirecte n'est pas tracée.
- **Association, pas causalité.** La vérification ne vaut que sur les mesures disponibles.
- **Le domaine synthétique n'est pas la chimie.** GAIED valide l'intégration, pas la justesse chimique.
