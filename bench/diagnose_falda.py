"""
Diagnostic only: what does the LLM answer to FALDA's atom-extraction prompt (T0 -> T1) on a run's own messages?

    python -m bench.diagnose_falda results/<config>/runs/<scenario>__falda__s<seed>.json

Reads the run's pool name from its export, takes the last messages of that pool from the FALDA gateway, builds the
transcript exactly as FALDA's distiller does (falda_distiller._extract_chunk), calls the configured LLM with FALDA's
own L1 prompt, and prints the raw answer and what FALDA's parser makes of it. Writes nothing to FALDA.
"""
from __future__ import annotations

import json
import os
import sys

from memlab import LLM
from memlab.falda_memory import FaldaMemory


def main(run_path: str, llm: LLM | None = None) -> None:
    d = json.load(open(run_path))
    ex = d["memory"]["extra"]
    pool = ex.get("pool")
    print("pool:", pool, "| atoms in the export:", len(ex.get("atoms") or []), "| stream messages:", ex.get("stream_total"))
    print("distiller log (end):\n" + "\n".join((ex.get("distiller_log") or "").splitlines()[-8:]))
    mem = FaldaMemory.__new__(FaldaMemory)          # no pool declaration, no Goal write: read only
    mem.url = os.environ.get("FALDA_URL", "http://localhost:8077").rstrip("/")
    mem.pool, mem.timeout_s, mem.cadence = pool, 60.0, dict(ex.get("cadence") or {})
    mem.cadence = {**{"l1_every_n": 6, "l2_interval_s": 300, "l3_interval_s": 900, "poll_s": 30}, **mem.cadence}
    mem.llm = llm or LLM()
    dist = mem._load_distiller()
    msgs = list(reversed(dist.falda("/stream/query", {"limit": 500}).get("messages", [])))[-dist.L1_CHUNK_TURNS:]
    transcript = "\n".join(f"[{m['role']}] {m['content'][:1200]}" for m in msgs)
    print(f"\n{len(msgs)} messages, transcript of {len(transcript)} characters; first 600:\n{transcript[:600]}\n")
    raw = mem.llm.complete(dist.L1_SYS, transcript, purpose="memory", max_tokens=2500)
    print("RAW ANSWER (first 2000 characters):\n" + repr(raw[:2000]))
    print("\nPARSED BY FALDA:", dist.parse_json_loose(raw))


if __name__ == "__main__":
    main(sys.argv[1])
