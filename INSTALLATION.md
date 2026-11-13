# Installer et lancer le projet (GAIED + mémoire partagée)

Guide pour toute personne qui rejoint le projet. Il est écrit pour **Windows** et ne suppose
aucune connaissance de Docker. Sur Mac ou Linux, les commandes sont les mêmes ; seule
l'installation de Docker change.

---

## Ce qu'il faut savoir avant de commencer

Le projet fait tourner des agents IA qui cherchent des molécules à forte énergie d'ionisation.
Il a besoin de plusieurs programmes : Python et ses bibliothèques, le simulateur xTB, une base
Redis, un modèle de langage. Les installer à la main sous Windows est très difficile (xTB et une
partie du code ne fonctionnent que sous Linux).

**Docker** règle ce problème : il fait tourner de petites machines Linux, appelées
**conteneurs**, sur votre PC. Le projet utilise trois conteneurs :

| Conteneur | Rôle |
|-----------|------|
| `redis`   | la base Redis : échanges entre agents et **mémoire partagée** |
| `ollama`  | le modèle de langage local (Qwen 3.5), sans clé API |
| `gaied`   | le code du projet (agents, xTB, Flowcept) |

Le fichier `docker/docker-compose.yml` décrit ces trois conteneurs. Le fichier `docker/Dockerfile`
décrit comment fabriquer le conteneur `gaied` ; le résultat de cette fabrication s'appelle une
**image**, qu'on ne construit qu'une seule fois.

**Le dossier du projet est partagé avec le conteneur.** Quand vous modifiez un fichier `.py` dans
VS Code, le conteneur voit le changement immédiatement : pas besoin de reconstruire l'image.

Il faut environ **15 Go** libres sur le disque (image du projet + modèle de langage) et une
connexion internet pour la première installation (plusieurs Go à télécharger).

---

## Étape 1 — Installer les outils (une seule fois)

1. **Git** : https://git-scm.com/download/win — installer avec les options par défaut.
2. **Docker Desktop** : https://www.docker.com/products/docker-desktop/
   - Pendant l'installation, garder l'option **« Use WSL 2 »** cochée.
   - Redémarrer le PC si on vous le demande.
   - Si Docker demande d'installer ou de mettre à jour WSL, ouvrir PowerShell **en administrateur**
     et taper `wsl --install`, puis redémarrer.
   - Ouvrir Docker Desktop une première fois et attendre qu'il indique qu'il tourne
     (« Engine running »). Pas besoin de créer de compte.
3. **VS Code** (facultatif, pour lire et modifier le code) : https://code.visualstudio.com/

Vérification, dans **PowerShell** (menu Démarrer → « PowerShell ») :
```powershell
git --version
docker --version
docker compose version
```
Chaque commande doit afficher un numéro de version.

---

## Étape 2 — Accéder au dépôt et le télécharger

1. **Accepter l'invitation** de collaboratrice : lien reçu par e-mail, ou sur
   https://github.com/notifications quand on est connecté à GitHub.
2. Télécharger le projet (on le met dans `Documents\Research`, mais un autre dossier convient) :
```powershell
mkdir $HOME\Documents\Research -Force
cd $HOME\Documents\Research
git clone https://github.com/codebylina/gaied.git GAIED
cd GAIED
git switch docker-local
```
`docker-local` est la branche de travail : elle contient la configuration Docker et la mémoire
partagée. **Ne pas travailler sur `main`** (c'est le code d'origine d'Amal).

**Première connexion GitHub depuis Git** : si Git demande un mot de passe, le mot de passe du
compte GitHub est refusé. Une fenêtre de connexion GitHub s'ouvre normalement (gestionnaire
d'identifiants de Git pour Windows) ; sinon, créer un jeton : GitHub → Settings → Developer settings
→ Personal access tokens → Tokens (classic) → cocher `repo`, et le coller à la place du mot de passe.
Ce jeton est personnel : ne jamais le partager ni le mettre dans un fichier du projet.

---

## Étape 3 — Construire l'image du projet (une seule fois, 10 à 30 min)

Docker Desktop doit être ouvert.
```powershell
cd $HOME\Documents\Research\GAIED\docker
docker compose build gaied
```
**Toutes les commandes Docker de ce guide se lancent depuis le dossier `GAIED\docker`.**

Problèmes fréquents :
- `--allow=fs.read=Dockerfile` : le chemin a été tapé avec des minuscules/majuscules différentes de
  la réalité (`research` au lieu de `Research`…). Revenir dans le dossier avec la bonne casse,
  par exemple avec `cd $HOME\Documents\Research\GAIED\docker`.
- `no such host`, `Could not resolve host`, `timeout` : coupure ou lenteur internet. Relancer la
  même commande : ce qui a déjà été téléchargé est gardé.
- `error during connect` / `pipe/dockerDesktopLinuxEngine` : Docker Desktop n'est pas lancé.

Vérification : `docker images` doit lister une image `docker-gaied`.

---

## Étape 4 — Démarrer Redis et le modèle de langage

```powershell
docker compose up -d redis ollama
docker compose exec ollama ollama pull qwen3.5:4b
```
La deuxième commande télécharge le modèle Qwen 3.5 (environ 3 Go, une seule fois ; il est gardé
même après redémarrage du PC). Vérifications :
```powershell
docker compose ps                              # redis et ollama doivent être "running"
docker compose exec redis redis-cli ping       # doit répondre PONG
docker compose exec ollama ollama list         # doit lister qwen3.5:4b
```

Après un redémarrage du PC : ouvrir Docker Desktop, puis relancer seulement
`docker compose up -d redis ollama`.

### Option : utiliser une clé API au lieu du modèle local

Le modèle local est gratuit mais lent sur un PC sans carte graphique (1 à 2 minutes par réponse).
Avec une clé API (OpenAI, Google Gemini, Anthropic, Groq, OpenRouter…), c'est beaucoup plus rapide :
```powershell
Copy-Item env.example.txt .env
notepad .env
```
Dans le fichier : coller la clé après `LLM_API_KEY=`, puis enlever le `#` devant les deux lignes du
fournisseur choisi (et vérifier le nom exact du modèle sur le site du fournisseur).
- Le fichier doit s'appeler exactement `.env` (sans `.txt`) et rester dans le dossier `docker`.
- Il n'est **jamais** envoyé sur GitHub : `git check-ignore .env` doit afficher `.env`.
- Pour revenir au modèle local : supprimer `.env` ou y mettre `LLM_PROVIDER=ollama`.

---

## Étape 5 — Vérifier que tout marche (sans modèle de langage, moins d'une minute)

```powershell
docker compose run --rm gaied python tests/test_evident.py
docker compose run --rm gaied python experiments/run_matrix.py experiments/configs/dryrun.json
```
Résultat attendu : `13/13 tests passed` (la mémoire EVIDENT seule, quelques secondes), puis deux
campagnes complètes (Academy, Parsl, vrai xTB, Flowcept) avec un **LLM simulé** : `-> ok` pour
chacune, en quelques minutes. Rien n'est un résultat scientifique : on vérifie l'installation.

---

## Étape 6 — Lancer une campagne de recherche de molécules

```powershell
docker compose run --rm gaied
```
Les agents demandent des molécules au modèle, les simulent avec xTB, tirent des conclusions et
recommencent. Un rapport s'affiche toutes les 30 secondes. Avec le modèle local, une campagne peut
durer **une heure ou plus**. Ctrl+C pour arrêter.

Réglages utiles, sans modifier aucun fichier (ajouter `-e NOM=valeur` avant `gaied`) :

| Réglage | Effet |
|---------|-------|
| `-e "SEEDS=CNC(N)=O,CNC(N)=O"` | molécules de départ, une par agent (ici : 2 agents, même départ) |
| `-e MAX_ITERATIONS=1` | nombre de tours de la boucle de chaque agent (moins = plus court) |
| `-e MEMORY_MODE=none` | agents isolés (référence) ; aussi `coord`, `naive`, `evident` (défaut) |
| `-e MONITOR=true` | agent d'analyse en temps réel (arrête les calculs hors objectif) |

Exemple :
```powershell
docker compose run --rm -e "SEEDS=CNC(N)=O,CNC(N)=O" -e MAX_ITERATIONS=1 gaied
```

### Voir les résultats

```powershell
docker compose run --rm gaied python docker/summarize_prov.py     # déroulé de la dernière campagne
docker compose run --rm gaied python -m evident show              # mémoire EVIDENT de la dernière campagne
docker compose exec redis redis-cli KEYS "ev:*"                   # les clés brutes dans Redis
```
`summarize_prov.py` lit la provenance enregistrée par Flowcept (`output/flowcept_buffer_*.jsonl`)
et affiche chaque appel au modèle, chaque simulation xTB, et le nombre de **calculs en double**.

### Expériences du papier (machine distante avec GPU)

Voir `docs/EVIDENT.md`, section « Protocole expérimental ». En résumé :
```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d redis ollama
docker compose run --rm gaied python experiments/run_matrix.py experiments/configs/pilot.json
docker compose run --rm gaied python experiments/run_matrix.py experiments/configs/main.json
docker compose run --rm gaied python experiments/analyze.py experiments/results/main
```

---

## Travailler à plusieurs avec Git

- **Avant de commencer à travailler** : `git pull` (récupère les changements des autres).
- **Pour une nouvelle fonctionnalité**, créer sa propre branche à partir de `docker-local` :
  ```powershell
  git switch docker-local
  git pull
  git switch -c yassmine-ma-fonctionnalite
  ```
- **Enregistrer et partager son travail** :
  ```powershell
  git add .
  git commit -m "Ce que j'ai fait"
  git push -u origin yassmine-ma-fonctionnalite
  ```
  Puis, sur GitHub, ouvrir une *Pull Request* vers `docker-local` pour que l'autre relise avant de
  fusionner.
- Se mettre d'accord avant de modifier le même fichier en même temps (surtout `mol_design_agents.py`
  et `evident/`), pour éviter les conflits.
- **Ne jamais committer** `docker/.env` ni un jeton ou une clé.
- Les fichiers `runinfo/` et `output/` produits par vos campagnes ne sont pas envoyés sur GitHub
  (déjà dans le `.gitignore`).

---

## Où est quoi dans le code

| Fichier | Contenu |
|---------|---------|
| `mol_design_agents.py` | l'agent (`XTBSimulationAgent`) et sa boucle LangGraph : plan → tool_calling → simulate → conclude → critique → update |
| `run-07.py` | lance les agents et l'orchestration de la campagne |
| `evident/` | **la mémoire EVIDENT** (faits, coordination, affirmations vérifiées, directives, télémétrie) et l'agent d'analyse temps réel (`monitor.py`) |
| `experiments/` | matrice d'expériences, campagnes synthétiques, analyse statistique et figures |
| `docs/EVIDENT.md` | la méthode détaillée et les travaux connexes |
| `tests/` | tests de la mémoire, sans modèle de langage |
| `docker/` | toute la configuration Docker ; `local_llm.py` branche le modèle local ou la clé API |
| `CLAUDE.md` | description détaillée du projet et de la mémoire (aussi lue par Claude Code) |

Les modifications faites au code d'origine d'Amal sont repérables par les commentaires
`[gaied-docker ...]` (configuration) et `[evident]` (mémoire et contrôle).

---

## Erreurs déjà rencontrées

| Message | Cause et solution |
|---------|-------------------|
| `--allow=fs.read=Dockerfile` | casse du chemin incorrecte : `cd $HOME\Documents\Research\GAIED\docker` |
| `no such host`, `Could not resolve host` | internet instable : relancer la commande |
| `error during connect` | Docker Desktop n'est pas ouvert |
| `cannot import name 'init_logging'` | image ancienne : `docker compose build gaied` |
| `LLM_API_KEY est vide` | `LLM_PROVIDER=openai` dans `.env` sans clé : remplir la clé ou supprimer `.env` |
| la campagne affiche `Number of molecules simulated: 0` pendant plusieurs minutes | normal avec le modèle local : il réfléchit (1 à 2 min par réponse) |
