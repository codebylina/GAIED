"""
Replay the verification of a finished GAIED campaign (no truth evaluator exists for chemistry).

Every atom of the export is formalized and tested again, by the same verifier, against ALL the measurements of
the campaign; the replayed status is stored in the export under "replay". This lets the analysis count, for any
condition (evidence_unverified included), how often agents were shown as true a claim that the campaign's own
measurements contradict or do not contain. It measures consistency with the measurements, not chemical truth;
the manual audit (bench/audit_tests.py) checks the tests themselves.

    python -m bench.replay_verify results/gaied/*.json [--force]     (uses the configured LLM; skips replayed files)
"""
from __future__ import annotations

import json
import sys

from memlab import LLM, InMemoryStore
from memlab.integrations.gaied import adapter_from_env
from memlab.verifier import VerificationConfig, Verifier


def replay(path: str, llm: LLM | None = None, adapter=None) -> dict:
    d = json.load(open(path))
    if d.get("replay") and "--force" not in sys.argv:
        print(f"{path}: already replayed (--force to replay again)")
        return d
    atoms = (d.get("extra") or {}).get("atoms") or []
    if not atoms:
        print(f"{path}: no atoms, nothing to replay")
        return d
    store = InMemoryStore()
    for o in d["observations"]:
        store.put("obs", o["id"], o)
        store.append("obs_log", {"id": o["id"]})
    for b in d.get("briefings", []):
        store.append("briefings", b)
    llm = llm or LLM()
    v = Verifier(store, adapter or adapter_from_env(), llm, VerificationConfig())
    for a in atoms:
        v.register(a["id"], a["text"], a["author"], a["claimed_at"])
    while v.step():
        pass
    d["replay"] = {"statuses": {x["id"]: {"status": x["status"], "kind": x["kind"], "scoped": x["scoped"]}
                                for x in store.all("vmem")},
                   "llm_usage": llm.usage.snapshot()}
    json.dump(d, open(path, "w"), default=str)
    counts = {}
    for x in d["replay"]["statuses"].values():
        counts[x["status"]] = counts.get(x["status"], 0) + 1
    print(f"{path}: {len(atoms)} atoms replayed -> {counts}")
    return d


if __name__ == "__main__":
    for p in sys.argv[1:]:
        if p != "--force":
            replay(p)
