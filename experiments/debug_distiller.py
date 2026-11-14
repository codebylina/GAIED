"""
Diagnostic only: show what Qwen answers to FALDA's L1 (atom extraction) prompt.

    python experiments/debug_distiller.py experiments/results/<name>/logs/falda_r1.log

Reads the pool name from the campaign log, takes the last stream turns of that pool, builds the
transcript exactly as falda_distiller._extract_chunk does, calls Qwen through the same relay
settings (think=false) and prints the raw answer and what FALDA's parser makes of it.
Writes nothing to FALDA.
"""
import importlib.util
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from falda_memory import ollama_relay  # noqa: E402

log = open(sys.argv[1], errors="replace").read()
pool = re.search(r"tenant=distiller/(\S+)", log).group(1)
os.environ.setdefault("FALDA_URL", "http://falda:8077")
os.environ["FALDA_TENANT"], os.environ["FALDA_POOL"] = "distiller", pool
spec = importlib.util.spec_from_file_location("d", os.path.join(ROOT, "falda_memory", "falda_distiller.py"))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)

msgs = list(reversed(d.falda("/stream/query", {"limit": 500}).get("messages", [])))[-d.L1_CHUNK_TURNS:]
transcript = "\n".join(f"[{m['role']}] {m['content'][:1200]}" for m in msgs)
print(f"pool {pool}: {len(msgs)} turns, transcript {len(transcript)} chars\n")
out = ollama_relay._ollama_chat({"model": os.environ.get("OLLAMA_MODEL"), "max_tokens": 2500,
                                 "temperature": 0.2,
                                 "messages": [{"role": "system", "content": d.L1_SYS},
                                              {"role": "user", "content": transcript}]})
raw = out["choices"][0]["message"]["content"]
print("RAW ANSWER (first 2000 chars):\n" + repr(raw[:2000]))
print("\nPARSED BY FALDA:", d.parse_json_loose(raw))
