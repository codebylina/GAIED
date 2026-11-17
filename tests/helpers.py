"""Shared test helpers: a Redis client (MEMORY_TEST_REDIS=host:port for a real server, otherwise
fakeredis with Lua support: pip install fakeredis lupa) and a toy domain."""
from __future__ import annotations

import os
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from memory import PERMANENT, SCIENTIFIC, TRANSIENT, Domain, SharedMemory, Store  # noqa: E402


def redis_client():
    target = os.environ.get("MEMORY_TEST_REDIS")
    if target:
        import redis
        host, _, port = target.partition(":")
        return redis.Redis(host=host, port=int(port or 6379), decode_responses=True)
    import fakeredis
    return fakeredis.FakeRedis(decode_responses=True)


CLIENT = redis_client()


class ToyDomain(Domain):
    """Entities are lowercase words. Feature has_x = 1 when the word contains an x.
    Errors: 'impossible' -> scientific, 'missing binary' -> permanent, anything else -> transient."""
    name, objective, direction = "word", "score", "max"

    def canonical(self, raw):
        raw = str(raw).strip().lower()
        return raw if raw.isalpha() else None

    def features(self, e):
        return {"has_x": float("x" in e), "length": float(len(e))} if e.isalpha() else {}

    def similarity(self, a, b):
        sa, sb = set(a), set(b)
        return len(sa & sb) / len(sa | sb) if sa | sb else 1.0

    def classify_failure(self, error):
        if "impossible" in error:
            return SCIENTIFIC
        if "missing binary" in error:
            return PERMANENT
        return TRANSIENT


M, P = "toy", {"precision": 1}


def new_memory(agent="a", ns=None, **kw) -> SharedMemory:
    return SharedMemory(Store(CLIENT, ns or f"test-{uuid.uuid4().hex[:8]}"), ToyDomain(), agent, **kw)


def measure(m: SharedMemory, word: str, value: float, task_id: str | None = None):
    return m.record_result(word, method=M, params=P, value=value, task_id=task_id)


def run_tests(namespace: dict) -> int:
    failures = 0
    for name, fn in list(namespace.items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                import traceback
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=3)
    return failures
