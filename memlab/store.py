"""
Storage interface of the memory, and two backends.

The memory only needs JSON documents grouped in collections, append-only logs, and one atomic
operation (take a lease on a key). Any key-value store with these can host it:
  - InMemoryStore : one process (tests, single-process benchmark runs)
  - RedisStore    : several processes (GAIED agents, orchestrator, judge); durable when Redis runs
                    with persistence (AOF/RDB), so the memory survives an interruption.
All keys live under a namespace (one per campaign, or one per series of campaigns to share memory).
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Iterable


class Store:
    """Interface. Documents are dicts; ids are strings."""

    def put(self, coll: str, doc_id: str, doc: dict) -> None: ...
    def get(self, coll: str, doc_id: str) -> dict | None: ...
    def all(self, coll: str) -> list[dict]: ...
    def ids(self, coll: str) -> list[str]: ...
    def append(self, log: str, item: dict) -> None: ...
    def read_log(self, log: str, start: int = 0) -> list[dict]: ...
    def log_len(self, log: str) -> int: ...
    def take_lease(self, key: str, owner: str, ttl_s: float) -> tuple[bool, str | None]: ...
    def renew_lease(self, key: str, owner: str, ttl_s: float) -> bool: ...
    def release_lease(self, key: str, owner: str) -> None: ...
    def lease_holder(self, key: str) -> str | None: ...
    def incr(self, counter: str, by: int = 1) -> int: ...


class InMemoryStore(Store):
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._colls: dict[str, dict[str, str]] = {}
        self._logs: dict[str, list[str]] = {}
        self._leases: dict[str, tuple[str, float]] = {}
        self._counters: dict[str, int] = {}

    def put(self, coll, doc_id, doc):
        with self._lock:
            self._colls.setdefault(coll, {})[doc_id] = json.dumps(doc, default=str)

    def get(self, coll, doc_id):
        with self._lock:
            raw = self._colls.get(coll, {}).get(doc_id)
        return json.loads(raw) if raw else None

    def all(self, coll):
        with self._lock:
            raws = list(self._colls.get(coll, {}).values())
        return [json.loads(r) for r in raws]

    def ids(self, coll):
        with self._lock:
            return list(self._colls.get(coll, {}).keys())

    def append(self, log, item):
        with self._lock:
            self._logs.setdefault(log, []).append(json.dumps(item, default=str))

    def read_log(self, log, start=0):
        with self._lock:
            raws = list(self._logs.get(log, [])[start:])
        return [json.loads(r) for r in raws]

    def log_len(self, log):
        with self._lock:
            return len(self._logs.get(log, []))

    def take_lease(self, key, owner, ttl_s):
        with self._lock:
            cur = self._leases.get(key)
            if cur and cur[1] > time.time() and cur[0] != owner:
                return False, cur[0]
            self._leases[key] = (owner, time.time() + ttl_s)
            return True, owner

    def renew_lease(self, key, owner, ttl_s):
        with self._lock:
            cur = self._leases.get(key)
            if cur and cur[0] == owner:
                self._leases[key] = (owner, time.time() + ttl_s)
                return True
            return False

    def release_lease(self, key, owner):
        with self._lock:
            cur = self._leases.get(key)
            if cur and cur[0] == owner:
                del self._leases[key]

    def lease_holder(self, key):
        with self._lock:
            cur = self._leases.get(key)
            return cur[0] if cur and cur[1] > time.time() else None

    def incr(self, counter, by=1):
        with self._lock:
            self._counters[counter] = self._counters.get(counter, 0) + by
            return self._counters[counter]


class RedisStore(Store):
    def __init__(self, client: Any, namespace: str) -> None:
        self.r, self.ns = client, namespace

    @classmethod
    def connect(cls, namespace: str, host: str | None = None, port: int = 6379) -> "RedisStore":
        import os

        import redis
        client = redis.Redis(host=host or os.environ.get("REDIS_HOST", "localhost"), port=port,
                             decode_responses=True)
        client.ping()
        return cls(client, namespace)

    def k(self, *parts: str) -> str:
        return ":".join(["pm", self.ns, *parts])

    def put(self, coll, doc_id, doc):
        self.r.hset(self.k("c", coll), doc_id, json.dumps(doc, default=str))

    def get(self, coll, doc_id):
        raw = self.r.hget(self.k("c", coll), doc_id)
        return json.loads(raw) if raw else None

    def all(self, coll):
        return [json.loads(v) for v in self.r.hvals(self.k("c", coll))]

    def ids(self, coll):
        return list(self.r.hkeys(self.k("c", coll)))

    def append(self, log, item):
        self.r.rpush(self.k("l", log), json.dumps(item, default=str))

    def read_log(self, log, start=0):
        return [json.loads(v) for v in self.r.lrange(self.k("l", log), start, -1)]

    def log_len(self, log):
        return int(self.r.llen(self.k("l", log)))

    def take_lease(self, key, owner, ttl_s):
        lk = self.k("lease", key)
        if self.r.set(lk, owner, nx=True, px=int(ttl_s * 1000)):
            return True, owner
        holder = self.r.get(lk)
        if holder == owner:
            self.r.pexpire(lk, int(ttl_s * 1000))
            return True, owner
        if holder is None:  # expired between the two calls
            return self.take_lease(key, owner, ttl_s)
        return False, holder

    def renew_lease(self, key, owner, ttl_s):
        lk = self.k("lease", key)
        if self.r.get(lk) == owner:
            return bool(self.r.pexpire(lk, int(ttl_s * 1000)))
        return False

    def release_lease(self, key, owner):
        lk = self.k("lease", key)
        if self.r.get(lk) == owner:
            self.r.delete(lk)

    def lease_holder(self, key):
        return self.r.get(self.k("lease", key))

    def incr(self, counter, by=1):
        return int(self.r.incrby(self.k("n", counter), by))


def iter_docs(store: Store, coll: str, where: dict | None = None) -> Iterable[dict]:
    for d in store.all(coll):
        if not where or all(d.get(k) == v for k, v in where.items()):
            yield d
