"""
apply_patches.py — Adapts GAIED to run locally, in Docker, with Ollama.
(Already applied and committed on the docker-local branch: do not run it again.)

Usage (from the GAIED folder):
    python docker/apply_patches.py .

Changes (each one is applied only once; an .orig copy is kept):
  run-07.py
    - Redis: host read from REDIS_HOST (in Docker, the service is called "redis")
    - Parsl: removes the "conda activate" (useless in the container, and a ";" was missing)
    - seed molecules read from SEEDS, iterations from MAX_ITERATIONS
  mol_design_agents.py
    - if OLLAMA_MODEL is set, use local_llm.OllamaLLM instead of SomeAPILLM
    - number of xTB processes per agent read from N_XTB_WORKERS (the code used all cores - 1)
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
         f"''  {MARK} no conda activate needed in Docker"),
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
            print(f"[=] {name}: already patched, nothing to do")
            continue
        shutil.copy(path, path.with_suffix(path.suffix + ".orig"))
        for old, new in edits:
            if text.count(old) != 1:
                print(f"[!] {name}: block not found, edit it by hand:\n    {old.strip()[:80]}")
                ok = False
                continue
            text = text.replace(old, new)
        path.write_text(text)
        print(f"[+] {name}: patched (original copy: {name}.orig)")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python apply_patches.py <path to GAIED>")
    sys.exit(main(Path(sys.argv[1])))
