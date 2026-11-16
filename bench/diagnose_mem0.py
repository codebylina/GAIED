"""
Diagnostic only: does Mem0's search return the memories it stored, with the configured LLM and embeddings?

    python -m bench.diagnose_mem0

Builds a fresh Mem0 (as the benchmark does) in a temporary folder, writes two texts, then searches with a short
query and with a long one (as long as an agent's plan), printing the hits, their scores, and any error in full.
Writes nothing outside the temporary folder.
"""
from __future__ import annotations

import os
import tempfile
import time
import traceback

from memlab.mem0_memory import LocalMem0, build_mem0


def main() -> None:
    d = tempfile.mkdtemp(prefix="mem0-diag-")
    print("embedding model:", os.environ.get("EMBED_MODEL"), "| dims:", os.environ.get("EMBED_DIMS", "768"),
          "| LLM:", os.environ.get("LLM_MODEL") or os.environ.get("OLLAMA_MODEL"))
    local = LocalMem0(build_mem0(d, "diag"), run_id="diag")
    print("features:", local.features())
    local.add("Results of my experiments: A0-B2-C2-D2-E2: -1.363; A1-B2-C2-D1-E2: 2.625", "agent0")
    local.add("- Option A1 increases the outcome compared with the other options of slot A.", "agent1")
    t0 = time.time()
    while local.pending():
        local.process_one(timeout=1.0)
    print(f"writes done in {time.time() - t0:.1f} s; stored:")
    for r in local.all():
        print("   ", r.get("id"), "|", r.get("memory"))
    long_plan = ("Plan: vary one slot at a time around the best design found so far, starting with slot A, "
                 "then B, C, D and E, keeping the options that increase the outcome. " * 30)
    for name, q in (("short query", "option A1 slot A outcome"), (f"long query ({len(long_plan)} chars)", long_plan)):
        print(f"--- search, {name}")
        try:
            hits = local.search(q)
            print(f"    {len(hits)} hit(s)")
            for h in hits[:5]:
                print("    score", h.get("score"), "|", h["memory"])
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        try:      # the raw call, without the 0.1 score threshold
            raw = local.m.search(q, filters={"run_id": "diag"}, threshold=0.0)
            res = raw.get("results", raw) if isinstance(raw, dict) else raw
            print("    raw scores without threshold:", [round(float(r.get("score") or 0), 3) for r in res][:5])
        except Exception:  # noqa: BLE001
            traceback.print_exc()


if __name__ == "__main__":
    main()
