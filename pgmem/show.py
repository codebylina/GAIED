"""
Print what the shared memory holds: entries per category, the analysis agent's diagnoses and its guidance.

    python -m pgmem.show                    namespace from PGMEM_NAMESPACE (default "default")
    python -m pgmem.show --json out.json    also export everything to a JSON file
"""
from __future__ import annotations

import argparse
import json
import os

from .schema import validation_status
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
    print("\nM1 results (best first)")
    for e in sorted(s.all("M1"), key=lambda e: -(e.data.get("value") or float("-inf")))[:a.limit]:
        c = e.data.get("conditions", {})
        print(f"  {e.text}  converged={c.get('converged')}  tasks={len(e.prov_refs)}  by {e.author}"
              + (f"  NOTE: {e.data['notes'][-1]['note']}" if e.data.get("notes") else ""))
    print("\nM2 experiences")
    for e in s.all("M2")[-a.limit * 2:]:
        extra = f" [{e.data['problem_type']} / {e.data['basis']}]" if e.kind == "diagnosis" else ""
        print(f"  ({e.kind}){extra} {e.text}")
    print("\nM3 knowledge")
    for e in s.all("M3")[-a.limit:]:
        d = e.data
        print(f"  ({e.kind}, {d['epistemic_status']}) {e.text}  for={len(d['supporting_evidence'])} "
              f"against={len(d['contradicting_evidence'])}  by {e.author}")
    print("\nM4 procedures")
    for e in s.all("M4"):
        print(f"  {e.text}\n    trigger: {e.data['trigger']}\n    actions: {e.data['actions']}\n"
              f"    expected: {e.data['expected_outcome']}\n    limitations: {e.data['limitations']}\n"
              f"    validation: {validation_status(e)}")
    print("\nGuidance")
    for g in s.all_guidance():
        print(f"  -> {g.target} [{g.problem_type} / {g.basis}] delivered={len(g.delivered)} applied={g.applied} "
              f"effective={g.effective}\n     {g.message[:200]}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(s.export(), fh, indent=1, default=str)
        print(f"\nexported to {a.json}")


if __name__ == "__main__":
    main()
