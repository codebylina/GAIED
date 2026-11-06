"""
patch_stop_rule.py — Corrige la règle d'arrêt de run-07.py.

Avant : un agent était déclaré « fini » dès que son nombre de molécules ne changeait
pas entre deux rapports (30 s). Avec un LLM sur CPU, une seule réponse peut durer
plus longtemps : la campagne s'arrêtait après la première itération.

Après : l'agent est déclaré fini seulement après STABLE_REPORTS rapports consécutifs
sans nouvelle molécule (défaut 20, soit 10 minutes), ou quand il s'arrête lui-même
(MAX_ITERATIONS atteint, ou plus d'amélioration).

Usage (dans le conteneur GAIED, depuis GAIED/docker) :
    docker compose run --rm gaied python docker/patch_stop_rule.py .
"""
import shutil
import sys
from pathlib import Path

MARK = "# [gaied-docker stop-rule]"

EDITS = [
    ("        active: dict[int, tuple] = {i: (agent, -1) for i, agent in enumerate(agents)}",
     "        active: dict[int, tuple] = {i: (agent, -1, 0) for i, agent in enumerate(agents)}  "
     + MARK + "\n"
     "        max_stable = int(os.environ.get('STABLE_REPORTS', '20'))"),
    ("                for i, (agent, last_count) in list(active.items()):",
     "                for i, (agent, last_count, stable) in list(active.items()):"),
    ("                        if count == last_count and count > 0:\n"
     "                            print(f'Agent {i} finished (no new molecules).')\n"
     "                            finished.append(i)\n"
     "                        else:\n"
     "                            active[i] = (agent, count)",
     "                        stable = stable + 1 if (count == last_count and count > 0) else 0\n"
     "                        if stable >= max_stable:\n"
     "                            print(f'Agent {i} finished (no new molecules for {stable} reports).')\n"
     "                            finished.append(i)\n"
     "                        else:\n"
     "                            active[i] = (agent, count, stable)"),
]


def main(repo: Path) -> int:
    path = repo / "run-07.py"
    text = path.read_text()
    if MARK in text:
        print("[=] run-07.py : règle d'arrêt déjà corrigée, rien à faire")
        return 0
    for old, _ in EDITS:
        if text.count(old) != 1:
            print(f"[!] run-07.py : bloc introuvable, rien n'a été modifié :\n    {old.strip()[:80]}")
            return 1
    shutil.copy(path, path.with_name("run-07.py.before-stop-rule"))
    for old, new in EDITS:
        text = text.replace(old, new)
    path.write_text(text)
    print("[+] run-07.py : règle d'arrêt corrigée (copie : run-07.py.before-stop-rule)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage : python docker/patch_stop_rule.py <chemin vers GAIED>")
    sys.exit(main(Path(sys.argv[1])))
