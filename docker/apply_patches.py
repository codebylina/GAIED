"""
apply_patches.py — Adapte GAIED pour tourner en local, dans Docker, avec Ollama.

Usage (depuis le dossier GAIED) :
    python docker/apply_patches.py .

Modifications (chacune n'est appliquée qu'une fois ; une copie .orig est gardée) :
  run-07.py
    - Redis : hôte lu dans REDIS_HOST (dans Docker, le service s'appelle "redis")
    - Parsl : supprime le "conda activate" (inutile dans le conteneur, et il manquait un ";")
    - molécules de départ lues dans SEEDS, itérations dans MAX_ITERATIONS
  mol_design_agents.py
    - si OLLAMA_MODEL est défini, utilise local_llm.OllamaLLM à la place de SomeAPILLM
    - nombre de processus xTB par agent lu dans N_XTB_WORKERS (le code prenait tous les cœurs - 1)
"""
import shutil
import sys
from pathlib import Path

MARK = "# [gaied-docker patch]"

PATCHES = {
    "run-07.py": [
        ("factory=RedisExchangeFactory('localhost', 6379),",
         f"factory=RedisExchangeFactory(os.environ.get('REDIS_HOST', 'localhost'), 6379),  {MARK}"),
        ("f'cd {os.getcwd()}conda activate ./mol-design;'",
         f"''  {MARK} conda activate inutile dans Docker"),
        ("        seeds = [\n            'CNC(N)=O',\n            'CC1=C(O)N=C(O)N1',\n        ]",
         f"        seeds = os.environ.get('SEEDS', 'CNC(N)=O,CC1=C(O)N=C(O)N1').split(',')  {MARK}"),
        ("                        20,   # max_iterations",
         f"                        int(os.environ.get('MAX_ITERATIONS', '20')),   # max_iterations  {MARK}"),
    ],
    "mol_design_agents.py": [
        ("\n\ndef _generate_initial_xyz(",
         f"\n\nif os.environ.get('OLLAMA_MODEL'):  {MARK}\n"
         "    from local_llm import OllamaLLM as SomeAPILLM  # noqa: F811\n"
         "\n\ndef _generate_initial_xyz("),
        ("        self.pool = ProcessPoolExecutor(max_workers=n_workers)",
         f"        n_workers = int(os.environ.get('N_XTB_WORKERS', n_workers))  {MARK}\n"
         "        self.pool = ProcessPoolExecutor(max_workers=n_workers)"),
    ],
}


def main(repo: Path) -> int:
    ok = True
    for name, edits in PATCHES.items():
        path = repo / name
        text = path.read_text()
        if MARK in text:
            print(f"[=] {name} : déjà modifié, rien à faire")
            continue
        shutil.copy(path, path.with_suffix(path.suffix + ".orig"))
        for old, new in edits:
            if text.count(old) != 1:
                print(f"[!] {name} : bloc introuvable, à modifier à la main :\n    {old.strip()[:80]}")
                ok = False
                continue
            text = text.replace(old, new)
        path.write_text(text)
        print(f"[+] {name} : modifié (copie d'origine : {name}.orig)")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage : python apply_patches.py <chemin vers GAIED>")
    sys.exit(main(Path(sys.argv[1])))
