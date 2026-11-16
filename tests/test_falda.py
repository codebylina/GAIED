"""
FALDA through its gateway (needs a running gateway; skipped otherwise).

    FALDA_URL=http://localhost:8077 python tests/test_falda.py
"""
from __future__ import annotations

import os
import sys
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.env import EnvSpec, SyntheticAdapter, SyntheticEnv  # noqa: E402
from bench.scripted import ScriptedLLM  # noqa: E402
from memlab import LLM, InMemoryStore, MemoryConfig, make_memory  # noqa: E402

URL = os.environ.get("FALDA_URL", "http://localhost:8077")
CLAIM = "Option A0 increases the outcome compared with the other options of slot A."


def gateway_up() -> bool:
    try:
        req = urllib.request.Request(f"{URL}/core/read", data=b"{}", headers={"content-type": "application/json"})
        urllib.request.urlopen(req, timeout=3)
        return True
    except Exception:  # noqa: BLE001
        return False


def test_write_distill_recall():
    env = SyntheticEnv(EnvSpec(n_slots=3, n_options=3, main_scale=1.0, interaction_scale=0.0, n_interactions=0, noise_sd=0.0, n_forbidden=0, seed=1))
    llm = LLM(provider="scripted", scripted=ScriptedLLM(env))
    mem = make_memory("falda", InMemoryStore(), SyntheticAdapter(env), MemoryConfig(briefing_budget_chars=3000),
                      llm=llm, falda={"url": URL, "pool": f"test-{uuid.uuid4().hex[:8]}",
                                      "cadence": {"l1_every_n": 1, "l2_interval_s": 0, "l3_interval_s": 0,
                                                  "poll_s": 0}})
    for a in ("a1", "a2"):
        mem.register_agent(a)
    mem.record_episode("a1", "conclude", "\n".join(           # FALDA writes a scene from 5 atoms on
        f"- Option {o} increases the outcome compared with the other options of slot {o[0]}."
        for o in ("A0", "A1", "B0", "B1", "C0")), 0)
    mem.record_episode("a1", "critique", "not written: FALDA receives the nodes of write_nodes only", 0)
    assert mem.maintain() > 0
    mem.maintain()
    text = mem.briefing("a2", "plan", "option A0 slot A", 1)
    assert "PROJECT CORE" in text and "RELEVANT MEMORIES" in text and "A0 increases" in text, text
    assert any(k["text"] == CLAIM for k in mem.presented_knowledge())
    ex = mem.export_extra()
    assert ex["stream_total"] == 1 and ex["atoms"] and ex["scenes"] and "A0" in ex["core"]


if __name__ == "__main__":
    if not gateway_up():
        print("skip (no FALDA gateway at", URL + ")")
        sys.exit(0)
    test_write_distill_recall()
    print("ok   test_write_distill_recall")
