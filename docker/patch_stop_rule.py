"""
patch_stop_rule.py — Fixes the stop rule of run-07.py.
(Already applied and committed on the docker-local branch: do not run it again.)

Before: an agent was declared "finished" as soon as its number of molecules did not
change between two reports (30 s). With an LLM on CPU, a single response can take
longer, so the campaign stopped after the first iteration.

After: the agent is declared finished only after STABLE_REPORTS consecutive reports
without a new molecule (default 20, i.e. 10 minutes), or when it stops by itself
(MAX_ITERATIONS reached, or no more improvement).

Usage (in the GAIED container, from GAIED/docker):
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
        print("[=] run-07.py: stop rule already fixed, nothing to do")
        return 0
    for old, _ in EDITS:
        if text.count(old) != 1:
            print(f"[!] run-07.py: block not found, nothing was changed:\n    {old.strip()[:80]}")
            return 1
    shutil.copy(path, path.with_name("run-07.py.before-stop-rule"))
    for old, new in EDITS:
        text = text.replace(old, new)
    path.write_text(text)
    print("[+] run-07.py: stop rule fixed (copy: run-07.py.before-stop-rule)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python docker/patch_stop_rule.py <path to GAIED>")
    sys.exit(main(Path(sys.argv[1])))
