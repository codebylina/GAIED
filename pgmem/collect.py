"""
Gather everything about the campaigns of one memory namespace into one zip file, to share the results:
  - the memory export (all entries M1-M4 and the guidance) as JSON;
  - the event stream (proposals, results, failures, reuses, waits...) as JSON;
  - the Flowcept provenance files (JSONL) of the campaigns recorded in that memory;
  - the post-execution analysis reports (JSON and Markdown) of that namespace;
  - any extra file given with --add (for example a saved console log).

    python -m pgmem.collect --namespace real3              -> results/collect_real3.zip
    python -m pgmem.collect --namespace real3 --add results/real3_campaign.log
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import zipfile

from .store import MemoryStore


def campaign_ids(export: dict) -> set[str]:
    ids: set[str] = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "campaign_id" and isinstance(v, str) and v:
                    ids.add(v)
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(export)
    return ids


def provenance_files(ids: set[str], pattern: str) -> list[str]:
    out = []
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        if any(i in text for i in ids):
            out.append(path)
    return out


def reports(ns_key: str, folder: str) -> list[str]:
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "analysis_*.json"))):
        try:
            with open(path, encoding="utf-8") as fh:
                if json.load(fh).get("namespace") != ns_key:
                    continue
        except (OSError, ValueError):
            continue
        out.append(path)
        md = path[:-5] + ".md"
        if os.path.exists(md):
            out.append(md)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--jsonl", default="output/flowcept_buffer_*.jsonl")
    ap.add_argument("--reports", default="results/analysis")
    ap.add_argument("--out", default=None)
    ap.add_argument("--add", nargs="*", default=[])
    a = ap.parse_args(argv)
    s = MemoryStore.connect(os.environ.get("REDIS_HOST", "localhost"), int(os.environ.get("REDIS_PORT", "6379")),
                            a.namespace)
    export = s.export()
    counts = {c: s.count(c) for c in ("M1", "M2", "M3", "M4")}
    ids = campaign_ids(export)
    prov = provenance_files(ids, a.jsonl) if ids else []
    reps = reports(s.ns, a.reports)
    extra = [p for p in a.add if os.path.exists(p)]
    out = a.out or os.path.join("results", f"collect_{a.namespace}.zip")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{a.namespace}_memory.json", json.dumps(export, indent=1, default=str))
        events = [{"id": i, "event": f.get("event"), "t": f.get("t"), "payload": json.loads(f.get("payload") or "{}")}
                  for i, f in s.r.xrange(s._k("events"))]
        z.writestr(f"{a.namespace}_events.json", json.dumps(events, indent=1, default=str))
        for folder, paths in (("provenance", prov), ("analysis", reps), ("extra", extra)):
            for p in paths:
                z.write(p, arcname=f"{folder}/{os.path.basename(p)}")
    print(f"namespace {a.namespace}: memory {counts}, campaigns {sorted(ids) or '-'}")
    print(f"  provenance files: {len(prov)}, analysis reports: {len(reps) // 2 or len(reps)}, extra files: {len(extra)}")
    if not any(counts.values()):
        print("  WARNING: this namespace is empty in Redis (deleted, or wrong name)")
    for p in a.add:
        if p not in extra:
            print(f"  WARNING: {p} not found, not added")
    print(f"written {out} ({os.path.getsize(out) // 1024} KB)")


if __name__ == "__main__":
    main()
