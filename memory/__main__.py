"""
Inspect and maintain a shared memory (Redis host from REDIS_HOST, default localhost).

  python -m memory namespaces                         list the memories
  python -m memory show [--namespace NS]              quality report + the records of the four layers
  python -m memory history LAYER ID                   every version of one record (who, when, why)
  python -m memory review ID STATUS "REASON"          human decision on M3 knowledge (locks its status)
  python -m memory maintain                           run the lifecycle (expiry, staleness, archival)

Namespace: --namespace or MEMORY_NAMESPACE (default "gaied").
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import SharedMemory, Store
from .domains.chemistry import ChemistryDomain


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m memory", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["namespaces", "show", "history", "review", "maintain"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--namespace", default=os.environ.get("MEMORY_NAMESPACE", "gaied"))
    a = ap.parse_args(argv)
    import redis
    r = redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"), decode_responses=True)
    if a.command == "namespaces":
        for ns, t in r.zrange("mem:namespaces", 0, -1, withscores=True):
            print(f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(t))}  {ns}")
        return 0
    m = SharedMemory(Store(r, a.namespace), ChemistryDomain(), agent=os.environ.get("USER", "human"))
    if a.command == "show":
        q = m.report()
        for L in ("M1", "M2", "M3", "M4"):
            print(f"{L}: {q[L]['active']} active, {q[L]['archived']} archived, {q[L]['by_status']}")
        print(f"M3 contradiction rate: {q['M3']['contradiction_rate']}, M4 effective rate: {q['M4']['effective_rate']}")
        print("metrics: " + json.dumps({k: v for k, v in q["metrics"].items()}, default=str))
        for e in m.store.find("M2"):
            print(f"\n[M2 {e.id}] {e.kind}/{e.status} x{e.occurrences} by {e.created_by}: {e.summary}")
            for f in e.observed_facts[:3]:
                print(f"    fact: {f}")
            for x in e.candidate_explanations[:2]:
                print(f"    hypothesis: {x}")
            if e.confirmed_cause:
                print(f"    CONFIRMED CAUSE: {e.confirmed_cause}")
        for k in m.store.find("M3"):
            print(f"\n[M3 {k.id}] {k.status.upper()} ({k.kind}, {k.source}) {k.statement}")
            print(f"    {len(k.supporting)} for / {len(k.contradicting)} against; {k.evaluation.get('reason', '')}")
        for g in m.store.find("M4"):
            print(f"\n[M4 {g.id}] {g.kind}/{g.status} -> {g.targets}: {g.text}")
            for agent, d in m.guidance.delivery(g.id).items():
                print(f"    {agent}: " + ", ".join(f"{k}={v}" for k, v in d.items() if k.endswith("_n")))
        return 0
    if a.command == "history":
        layer, rid = a.args
        for v in m.store.history(layer, rid):
            print(f"{time.strftime('%H:%M:%S', time.localtime(v['t']))} v{v['record']['version']} "
                  f"{v['record']['status']} by {v['by']}: {v['reason']}")
        return 0
    if a.command == "review":
        rid, status, reason = a.args
        k = m.knowledge.review(rid, status, reason, reviewer=m.agent)
        print(f"{rid}: {k.status} (locked by {m.agent})")
        return 0
    if a.command == "maintain":
        print(m.lifecycle.run(by="cli"))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
