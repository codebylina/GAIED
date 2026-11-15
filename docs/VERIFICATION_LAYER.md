# Couche de vérification des connaissances au-dessus de Mem0

Branche `mem0-verified`, partie de `mem0-baseline`. Mem0 est utilisé exactement comme dans la
référence : mêmes écritures, même recherche, même nombre de résultats, même budget de prompt. La couche
s'ajoute par-dessus (`memlab/verifier.py`, `memlab/verified_memory.py`).

## Thèse

Une mémoire partagée entre agents scientifiques ne doit pas présenter comme vrai ce que les agents
écrivent. Elle doit le confronter aux **mesures du workflow**, avec une règle de décision fixée **avant**
les mesures qui confirmeront ou réfuteront ce souvenir, et tenir compte de **qui** a mesuré **quand**.

## Positionnement

| Travail | Vérifie | Quand | Contre quoi |
|---|---|---|---|
| Mem0 | rien (ajout et déduplication par le LLM) | — | — |
| AutoSci, Trust Guard (2026) | chaque écriture | à l'écriture seulement | un agent relecteur LLM lit les preuves |
| MemTX (2026) | chaque écriture | à l'écriture, avec retrait en cascade | l'autorité de la source, des règles |
| StatefulDiscovery (2026) | des hypothèses | au fil d'une enquête | des analyses exécutées ; un seul agent, données figées |
| **Cette couche** | chaque souvenir de la mémoire hôte | **à chaque nouvelle mesure**, sans LLM | **un test exécuté sur les mesures du workflow, règle pré-enregistrée, classe des preuves selon la provenance** |

## Fonctionnement

1. **Mem0 crée un souvenir.** La couche le reçoit (rappel `on_added`), avec son auteur et sa date.
2. **Formalisation, en un appel au LLM par lot de souvenirs.** Pour chaque souvenir, le LLM écrit :
   - son **type** : `fact` (une mesure précise), `generalization` (un effet, une tendance) ou `untestable` ;
   - sa **portée** : il le reformule sans dire plus que ce qu'il couvre, sans jamais l'inverser ;
   - un **test exécutable** `def test(df)` sur la table des mesures. Ce test renvoie les mesures pour et
     contre, **et le verdict calculé par une règle écrite maintenant**, à partir du seul énoncé.
3. **Évaluation, sans LLM.** Tous les tests sont relancés, dans un seul processus isolé, dès que de
   nouvelles mesures arrivent. Le verdict est celui de la règle pré-enregistrée. Un test qui échoue est
   réécrit une fois par le LLM, avec l'erreur.
4. **Statut**, fixé par le code à partir du verdict et de la classe des preuves :

| Classe | Définition |
|---|---|
| exploratoire (E) | mesurée **avant** le souvenir : elle a pu le suggérer |
| de confirmation (C) | mesurée **après** le souvenir, par son auteur |
| indépendante (I) | mesurée **après** le souvenir, par **un autre agent** |

| Statut | Condition | Montré aux agents comme |
|---|---|---|
| `verified` | un fait dont la mesure correspond | vrai |
| `unsupported` | un fait sans mesure correspondante (inventé) | non appuyé par une mesure |
| `hypothesis` | une généralisation soutenue seulement par des preuves E | à tester |
| `supported` | une généralisation soutenue avec des preuves C ou I | vrai, avec le nombre de preuves |
| `established` | soutenue avec des preuves I d'au moins `min_independent_agents` autres agents | vrai, confirmé par d'autres |
| `refuted` | la règle donne « contredit » | faux |
| `unverifiable` | impossible à tester, ou le test échoue deux fois | non appuyé par une mesure |
| `pending` | pas encore traité | pas encore vérifié |

   Garde-fous : un verdict « soutenu » sans aucune preuve renvoyée par le test devient « non concluant ».
   Un verdict favorable ne fait jamais descendre un statut. Rien n'est effacé dans Mem0.
5. **Présentation.** Les souvenirs que Mem0 retrouve sont rangés par statut. Un agent à qui un souvenir
   a été montré comme vrai reçoit un avis si ce statut tombe.

## Ce qui a été corrigé après les essais de ProvMem

| Problème observé | Correction |
|---|---|
| tests coupés à 1024 tokens (`SyntaxError` ligne 1) | limite propre au juge (moitié de `NUM_CTX`) ; tests courts ; code extrait de façon robuste |
| juge lent (2 appels par affirmation, 340 s de retard) | **un appel par lot de souvenirs** ; la réévaluation n'appelle plus le LLM (règle pré-enregistrée) ; tous les tests dans un seul processus |
| affirmation inversée par le juge | consigne explicite, et c'est le texte montré aux agents qui est évalué |
| mémoire muette : rien n'était montré comme vrai | les souvenirs non encore vérifiés sont montrés comme « pas encore vérifiés » ; on mesure le **rappel** |
| évaluateur de vérité qui jugeait un fait mesuré contre la valeur sans bruit | l'évaluateur reçoit aussi les **valeurs mesurées** ; un fait est jugé contre la mesure |

## Évaluation

**Conditions** (`bench/configs/*.json`, et `MEMORY_CONDITION` pour GAIED) :

| Condition | Rôle |
|---|---|
| `none` | aucune mémoire partagée |
| `mem0` | Mem0 seul, la référence |
| `mem0_textcheck` | Mem0 + un LLM qui juge le texte seul : « pourquoi pas simplement demander à un LLM ? » |
| `mem0_verified` | Mem0 + la couche |
| `mem0_verified_nolayers` | ablation : sans la distinction exploratoire / confirmation / indépendante |

**Scénarios** :
- `deceptive` : une option aide là où les agents commencent et nuit ailleurs ;
- `adversarial` : un agent écrit des affirmations fausses par construction ;
- `standard` et `crash` dans `main.json`.

**Mesures** (`bench.analyze`) :
- **justesse des connaissances**, d'après l'évaluateur de vérité :
  - `final_precision` : part de vrai parmi ce qui est montré comme vrai ;
  - `final_recall` : part du vrai qui est montré comme vrai ;
  - `final_f1` ;
  - `false_exposure` : part des impressions fausses ;
  - `true_impressions` : connaissances vraies effectivement reçues par les agents ;
  - `precision_<statut>` : la calibration ;
- **fiabilité de l'évaluateur** : `oracle_accuracy_on_planted`, sa justesse sur les affirmations
  implantées, dont la vérité est connue ;
- **découverte** : `final_regret`, `regret_at_common_budget`, `decoy_outside` ;
- **coût et délai** : appels et secondes au LLM (agents, Mem0, juge), `verif_lag_s`, statuts ;
- **statistiques** : moyenne, intervalle de confiance à 95 % par bootstrap, test de Wilcoxon apparié
  contre `mem0_verified`.

## Lancer (machine distante, depuis `D:\Research\GAIED\docker`)

```powershell
docker compose run --rm --no-deps -e LLM_PROVIDER=ollama -e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=qwen3.5:9b -e LLM_MODEL=qwen3.5:9b -e EMBED_MODEL=nomic-embed-text gaied python -m bench.run bench/configs/quick.json
docker compose run --rm --no-deps -e LLM_PROVIDER=ollama -e OLLAMA_URL=http://host.docker.internal:11434 -e OLLAMA_MODEL=qwen3.5:9b -e LLM_MODEL=qwen3.5:9b gaied python -m bench.analyze results/verif_quick --reference mem0_verified
```

`quick.json` : `none`, `mem0` et `mem0_verified`, scénarios trompeur et adversaire, 1 graine.
`main.json` : les 5 conditions, 4 scénarios, 5 graines.

Tests sans LLM : `python tests/test_verifier.py` et `python tests/test_memlab.py`.

## Limites à dire

- **L'évaluateur de vérité est un LLM.** Sa justesse est mesurée sur les affirmations implantées, et un
  échantillon doit être vérifié à la main.
- **Une règle pré-enregistrée peut être trop laxiste ou trop stricte.** C'est ce que mesure la précision
  par statut.
- **Le domaine synthétique n'est pas la chimie.** GAIED sert à la validation réelle, sur les doublons,
  la découverte et le coût.
