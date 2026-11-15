"""
Human-readable view of one benchmark run: claims with their layers and the judge's notes,
test errors, and the oracle's truth for each statement shown to the agents.

    python -m bench.inspect_run results/quick/runs/deceptive__provmem__s1.json
"""
from __future__ import annotations

import collections
import json
import os
import sys


def main(path: str) -> None:
    d = json.load(open(path))
    if d.get("status") != "ok":
        print(d.get("error", "")[-3000:])
        return
    mem = d["memory"]
    extra = mem.get("extra", {})
    oracle_path = os.path.join(os.path.dirname(os.path.dirname(path)), "oracle_cache.json")
    oracle = {v["text"]: v["truth"] for v in json.load(open(oracle_path)).values()} if os.path.exists(oracle_path) else {}
    print(f"condition {d['condition']['name']}  optimum {d['optimum']:.3f}  decoy {d.get('decoy')}  start region {d.get('start_region')}")
    claims = extra.get("claims", [])
    if claims:
        print(f"\nCLAIMS ({len(claims)}): " + str(dict(collections.Counter(c["layer"] for c in claims))))
        dossiers = collections.defaultdict(list)
        for x in extra.get("dossiers", []):
            dossiers[x["claim"]].append(x)
        errors = collections.defaultdict(list)
        for j in extra.get("journal", []):
            if "error" in j["step"] or j["step"] == "no_judge":
                errors[j["claim"]].append(j["data"])
        for c in sorted(claims, key=lambda c: c["registered_at"]):
            print(f"\n[{c['layer']}] {c['id']} by {c['author']} (it {c['iteration']})")
            print(f"   stated : {c['statement']}")
            if c["scoped_statement"] != c["statement"]:
                print(f"   scoped : {c['scoped_statement']}")
            if c["scoped_statement"] in oracle:
                print(f"   oracle : {oracle[c['scoped_statement']]}")
            for x in dossiers[c["id"]]:
                print(f"   verdict: {x['verdict']} -> {x['to']}  counts E/C/I {x['counts']}  {str(x.get('rationale'))[:200]}")
            for e in errors[c["id"]][:2]:
                print(f"   error  : {str(e)[:300]}")
    else:
        items = {}
        for b in mem.get("briefings", []):
            for it in b["items"]:
                if it["kind"] == "claim":
                    items[it["text"]] = items.get(it["text"], 0) + 1
        print(f"\nKNOWLEDGE SHOWN TO AGENTS ({len(items)} distinct):")
        for t, n in sorted(items.items(), key=lambda x: -x[1]):
            print(f"  x{n:<3} oracle={oracle.get(t)}  {t[:160]}")


if __name__ == "__main__":
    main(sys.argv[1])
