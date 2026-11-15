"""
Command line tool to inspect and maintain an S3M memory (Redis host from REDIS_HOST, default localhost).

  python -m s3m namespaces                              list the memories in Redis
  python -m s3m show [NS]                               summary + briefing (latest namespace by default)
  python -m s3m lessons NS [--status S]                 lessons with their evidence and history
  python -m s3m events NS [-n 30]                       latest events
  python -m s3m ingest NS FILE.jsonl... [--domain D]    rebuild observations from Flowcept JSONL buffers
  python -m s3m verify NS [--domain D]                  verify pending lessons, re-check the others
  python -m s3m review NS LESSON STATUS "REASON"        human decision on a lesson

--domain: "generic" or "chemistry" (default), or "package.module:Class".
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import sys
import time


def _domain(name: str):
    if name == "chemistry":
        from .domains.chemistry import ChemistryDomain
        return ChemistryDomain()
    if name == "generic":
        from .domain import Domain
        return Domain()
    module, _, cls = name.partition(":")
    return getattr(importlib.import_module(module), cls)()


def _client():
    import redis
    return redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"),
                       port=int(os.environ.get("REDIS_PORT", "6379")), decode_responses=True)


def _memory(ns: str | None, domain: str):
    from .memory import S3M
    r = _client()
    if ns is None:
        latest = r.zrevrange("s3m:namespaces", 0, 0)
        if not latest:
            sys.exit("No S3M memory in Redis.")
        ns = latest[0]
    return S3M(r, _domain(domain), ns, agent="cli")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m s3m", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["namespaces", "show", "lessons", "events", "ingest", "verify", "review"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--domain", default="chemistry")
    ap.add_argument("--status")
    ap.add_argument("-n", type=int, default=30)
    a = ap.parse_args(argv)

    if a.command == "namespaces":
        for ns, t in _client().zrange("s3m:namespaces", 0, -1, withscores=True):
            print(f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(t))}  {ns}")
        return 0

    mem = _memory(a.args[0] if a.args else None, a.domain)

    if a.command == "show":
        rep = mem.report()
        print(f"S3M memory '{rep['namespace']}': {rep['observations']} observations, lessons {rep['lessons']}")
        print("counters: " + (", ".join(f"{k}={v}" for k, v in sorted(rep["stats"].items())) or "none"))
        for cond, info in rep["conditions"].items():
            print(f"\n--- condition {cond}: {info['method']} {json.dumps(info['params'])}")
            print(mem.briefing(info["method"], info["params"], max_chars=4000))
        return 0

    if a.command == "lessons":
        statuses = [a.status] if a.status else ["verified", "needs_review", "rejected", "pending"]
        for s in statuses:
            for l in mem.lessons.by_status(s, limit=200):
                print(f"\n[{l['id']}] {l['status'].upper()} ({l['type']}, by {l['created_by']}): {l['statement']}")
                print(f"  claim: {json.dumps(l['claim'])}")
                print(f"  evidence ({l.get('evidence_source', 'cited')}): {', '.join(l['evidence_ids'][:8])}"
                      + (" ..." if len(l["evidence_ids"]) > 8 else ""))
                print(f"  reason: {l['status_reason']}")
                for h in mem.lessons.history(l["id"]):
                    print(f"    {time.strftime('%H:%M:%S', time.localtime(h['t']))} {h.get('from', '-')} -> "
                          f"{h['status']} by {h.get('by')}: {h.get('reason')}")
        return 0

    if a.command == "events":
        for _, f in mem.r.xrevrange(f"{mem.p}:events", count=a.n)[::-1]:
            print(f["event"])
        return 0

    if a.command == "ingest":
        files = [f for pattern in a.args[1:] for f in glob.glob(pattern)]
        records = []
        for f in files:
            with open(f, encoding="utf-8") as fh:
                records += [json.loads(line) for line in fh if line.strip()]
        print(f"{mem.ingest_flowcept(records)} new observations from {len(records)} records in {len(files)} file(s)")
        return 0

    if a.command == "verify":
        for lid, status in mem.verify_pending():
            print(f"{lid}: {status}")
        return 0

    if a.command == "review":
        _, lid, status, reason = a.args
        l = mem.review(lid, status, reason, reviewer=os.environ.get("USER", "human"))
        print(f"{lid}: {l['status']}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
