"""
Print what the shared memory holds: entries per category with their neighbors in the graph, the relations by type,
and the analysis agent's guidance with its judgement.

    python -m pgmem.show                    namespace from PGMEM_NAMESPACE (default "default")
    python -m pgmem.show --json out.json    also export everything to a JSON file
"""
from __future__ import annotations

import argparse
import json
import os

from .retrieval import describe
from .store import MemoryStore


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", default=os.environ.get("PGMEM_NAMESPACE", "default"))
    ap.add_argument("--json", default=None)
    ap.add_argument("--limit", type=int, default=15)
    a = ap.parse_args(argv)
    s = MemoryStore.connect(os.environ.get("REDIS_HOST", "localhost"), int(os.environ.get("REDIS_PORT", "6379")),
                            a.namespace)
    print(f"namespace {a.namespace}: " + ", ".join(f"{c}={s.count(c)}" for c in ("M1", "M2", "M3", "M4")))
    print("relations: " + ", ".join(f"{k}={v}" for k, v in sorted(s.graph.count_by_type().items())))
    print("\nM1 results (best first)")
    for e in sorted(s.all("M1"), key=lambda e: -(e.data.get("value") or float("-inf")))[:a.limit]:
        print(f"  {describe(s, e)}  tasks={len(e.prov_refs)}  by {e.author}")
    print("\nM2 experiences")
    for e in s.all("M2")[-a.limit * 2:]:
        print(f"  ({e.kind}) {describe(s, e)}")
    print("\nM3 knowledge")
    for e in s.all("M3")[-a.limit:]:
        print(f"  {describe(s, e)}  by {e.author}")
    print("\nM4 procedures")
    for e in s.all("M4"):
        print(f"  {describe(s, e)}")
    print("\nGuidance")
    for g in s.all_guidance():
        judged = [x["props"] for x in s.graph.outgoing(g.id, ["APPLIED_IN"])]
        print(f"  -> {g.target} [{g.problem_type} / {g.epistemic_status}] delivered={len(g.delivered)} "
              f"judged={judged}\n     {g.message[:200]}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(s.export(), fh, indent=1, default=str)
        print(f"\nexported to {a.json}")


if __name__ == "__main__":
    main()
