"""
Why did the verifier decide what it decided? One block per memory of a run: text, kind, status, rule versions,
the last error of its test, the verdict and the evidence counts, and the first lines of its test.

    python -m bench.inspect_verification results/<config name> [substring of run file names] > inspect.txt
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys


def main(directory: str, pattern: str = "") -> None:
    for p in sorted(glob.glob(os.path.join(directory, "runs", "*.json"))):
        if pattern not in os.path.basename(p):
            continue
        d = json.load(open(p))
        mem = d.get("memory") or {}
        ver = (mem.get("extra") or {}).get("verification")
        if not ver:
            continue
        vm = ver["verified_memories"]
        print("=" * 100)
        print(os.path.basename(p), "| memories:", len(vm), "|",
              dict(collections.Counter(x["status"] for x in vm)))
        tools = collections.Counter(b["node"] for b in mem.get("briefings", []) if b["node"].startswith("tool:"))
        print("tool calls:", dict(tools) or "none")
        errors = collections.Counter()
        for x in vm:
            err = next((h.get("error") for h in reversed(x.get("history", [])) if h.get("error")), None)
            reason = next((h.get("reason") for h in reversed(x.get("history", [])) if h.get("reason")), None)
            if err:
                errors[err.splitlines()[-1][:120] if err.strip() else "(empty)"] += 1
            ev = x.get("eval") or {}
            print("-" * 100)
            print(f"[{x['status']}] kind={x.get('kind')} rules={len(x.get('rules', []))} author={x.get('author')}")
            print("  text:  ", x["text"][:300])
            if x.get("scoped") and x["scoped"] != x["text"]:
                print("  scoped:", x["scoped"][:300])
            if ev:
                print(f"  verdict={ev.get('verdict')} in_scope={ev.get('in_scope')} for={len(ev.get('support', []))} "
                      f"against={len(ev.get('contradict', []))} classes={ev.get('classes')} note={str(ev.get('note'))[:150]}")
            if reason or err:
                print("  why:   ", reason, "|", (err or "")[:400].replace("\n", " / "))
            for line in (x.get("code") or (x.get("rules") or [{}])[-1].get("code", "") or "").splitlines()[:14]:
                print("   |", line[:140])
        if errors:
            print("most frequent test errors:")
            for e, n in errors.most_common(8):
                print(f"  {n} x {e}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
