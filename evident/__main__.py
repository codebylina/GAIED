"""
Command line:
    python -m evident campaigns                       list campaigns (most recent first)
    python -m evident show [CAMPAIGN]                 summary, claims and directives
    python -m evident export CAMPAIGN OUT.json        dump a campaign for the analysis
"""
from __future__ import annotations

import json
import os
import sys

from .memory import EvidentMemory


def _client():
    import redis
    return redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"), decode_responses=True)


def _mem(client, campaign):
    if campaign is None:
        last = client.zrevrange("ev:campaigns", 0, 0)
        if not last:
            sys.exit("No campaign in Redis.")
        campaign = last[0]
    mem = EvidentMemory(client, campaign, "cli")
    mem.mode = mem.get_meta().get("mode", "evident")
    return mem


def main(argv):
    if not argv:
        sys.exit(__doc__)
    client = _client()
    cmd = argv[0]
    if cmd == "campaigns":
        for c, t in client.zrevrange("ev:campaigns", 0, 30, withscores=True):
            import time
            print(time.strftime("%Y-%m-%d %H:%M", time.localtime(t)), c)
    elif cmd == "show":
        mem = _mem(client, argv[1] if len(argv) > 1 else None)
        print(mem.summary())
        print("meta:", mem.get_meta())
        for c in sorted(mem.claims().values(), key=lambda c: c["status"]):
            print(f"  [{c['status']:<11}] v{c['version']} {mem._claim_line(c)[2:]}"
                  f"  (asserted {c['n_assertions']}x by {len(c['proposers'])} agent(s))")
        for _, f in mem.r.xrange(mem.k("telemetry")):
            if f["e"] == "directive_sent":
                d = json.loads(f["d"])
                print(f"  directive {d['kind'].upper():<6} -> {d['target']}: {d['reason'][:110]}")
    elif cmd == "export":
        mem = _mem(client, argv[1])
        data = mem.export()
        os.makedirs(os.path.dirname(os.path.abspath(argv[2])), exist_ok=True)
        with open(argv[2], "w") as fh:
            json.dump(data, fh)
        print(f"exported {len(data['telemetry'])} events, {len(data['facts_log'])} facts -> {argv[2]}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
