"""
Redis storage of the shared memory: typed records of the four layers, with versions, metadata indexes,
deduplication, archival, durable events and usage metrics.

Everything lives under one namespace (campaign isolation): mem:{namespace}:...

  {L}:rec:{id}           JSON of the current version of a record (L = M1..M4)
  {L}:ver:{id}           list of every version (full snapshot + who changed it, when and why)
  {L}:active / archive   zset id -> updated_at (archived records are kept, only hidden from retrieval)
  {L}:idx:{field}:{val}  set of ids (metadata filtering)
  {L}:dedup              hash dedup key -> id (atomic: HSETNX)
  usage:{id}             hash of usage counters (retrieved, delivered, applied...)
  events                 Redis Stream: durable history of memory events (read by the analysis service)
  notify                 Pub/Sub channel: lightweight notification of the same events (best effort)
  metrics                hash of counters and timings

Updates use optimistic concurrency (WATCH / MULTI): two writers never silently overwrite each other.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import Callable, Iterable

from .schemas import LAYERS, Record, record_from_dict

log = logging.getLogger("memory.store")


class MemoryUnavailable(RuntimeError):
    """Redis cannot be reached: callers degrade to running without memory."""


class Store:
    def __init__(self, client, namespace: str) -> None:
        self.r, self.namespace = client, namespace
        self.p = f"mem:{namespace}"
        client.zadd("mem:namespaces", {namespace: time.time()}, nx=True)

    @classmethod
    def connect(cls, namespace: str, host: str = "localhost", port: int = 6379) -> "Store":
        import redis
        client = redis.Redis(host=host, port=port, decode_responses=True, socket_timeout=10,
                             socket_connect_timeout=5, health_check_interval=30)
        try:
            client.ping()
        except Exception as exc:
            raise MemoryUnavailable(f"Redis {host}:{port} unreachable: {exc}") from exc
        return cls(client, namespace)

    def k(self, *parts) -> str:
        return ":".join([self.p, *map(str, parts)])

    # ------------------------------------------------------------------ write
    def add(self, rec: Record, actor: str, reason: str = "created") -> tuple[Record, bool]:
        """Store a new record. If a record with the same dedup key exists, return it instead
        (created=False): the caller decides how to consolidate."""
        rec.namespace = self.namespace
        rec.validate()
        L = rec.layer
        if rec.key:
            existing = self.r.hget(self.k(L, "dedup"), rec.key)
            if existing:
                found = self.get(L, existing)
                if found is not None:
                    return found, False
        if not self.r.set(self.k(L, "rec", rec.id), json.dumps(rec.to_dict(), default=str), nx=True):
            return self.get(L, rec.id), False                      # idempotent on id
        if rec.key and not self.r.hsetnx(self.k(L, "dedup"), rec.key, rec.id):
            winner = self.r.hget(self.k(L, "dedup"), rec.key)      # lost a race: keep the first one
            self.r.delete(self.k(L, "rec", rec.id))
            return self.get(L, winner), False
        pipe = self.r.pipeline()
        pipe.rpush(self.k(L, "ver", rec.id), json.dumps({"by": actor, "reason": reason, "t": time.time(),
                                                         "record": rec.to_dict()}, default=str))
        pipe.zadd(self.k(L, "active"), {rec.id: rec.updated_at})
        for f, vals in rec.index_values().items():
            for v in vals:
                pipe.sadd(self.k(L, "idx", f, v), rec.id)
        pipe.hincrby(self.k("metrics"), f"{L}.created", 1)
        pipe.execute()
        return rec, True

    def update(self, layer: str, rec_id: str, change: Callable[[Record], None | bool], actor: str,
               reason: str, retries: int = 20) -> Record | None:
        """Apply `change` to the current version (optimistic concurrency). `change` may return False to
        cancel. A new version is appended to the history; nothing is overwritten silently."""
        key = self.k(layer, "rec", rec_id)
        for _ in range(retries):
            with self.r.pipeline() as pipe:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    if raw is None:
                        pipe.unwatch()
                        return None
                    rec = record_from_dict(json.loads(raw))
                    old_index = rec.index_values()
                    if change(rec) is False:
                        pipe.unwatch()
                        return rec
                    rec.version += 1
                    rec.updated_at = time.time()
                    rec.validate()
                    new_index = rec.index_values()
                    pipe.multi()
                    pipe.set(key, json.dumps(rec.to_dict(), default=str))
                    pipe.rpush(self.k(layer, "ver", rec_id), json.dumps(
                        {"by": actor, "reason": reason, "t": rec.updated_at, "record": rec.to_dict()}, default=str))
                    for f, vals in old_index.items():
                        for v in set(vals) - set(new_index.get(f, [])):
                            pipe.srem(self.k(layer, "idx", f, v), rec_id)
                    for f, vals in new_index.items():
                        for v in vals:
                            pipe.sadd(self.k(layer, "idx", f, v), rec_id)
                    if rec.status == "archived":
                        pipe.zrem(self.k(layer, "active"), rec_id)
                        pipe.zadd(self.k(layer, "archive"), {rec_id: rec.updated_at})
                    else:
                        pipe.zadd(self.k(layer, "active"), {rec_id: rec.updated_at})
                    pipe.execute()
                    return rec
                except Exception as exc:
                    import redis
                    if isinstance(exc, redis.WatchError):
                        continue
                    raise
        raise RuntimeError(f"update of {layer}:{rec_id} kept conflicting")

    def archive(self, layer: str, rec_id: str, actor: str, reason: str) -> Record | None:
        """Hide a record from retrieval while keeping it (and its history) readable."""
        return self.update(layer, rec_id, lambda r: setattr(r, "status", "archived"), actor, f"archived: {reason}")

    # ------------------------------------------------------------------ read
    def get(self, layer: str, rec_id: str) -> Record | None:
        raw = self.r.get(self.k(layer, "rec", rec_id))
        return record_from_dict(json.loads(raw)) if raw else None

    def get_many(self, layer: str, ids: Iterable[str]) -> list[Record]:
        ids = list(ids)
        if not ids:
            return []
        raws = self.r.mget([self.k(layer, "rec", i) for i in ids])
        return [record_from_dict(json.loads(v)) for v in raws if v]

    def ids(self, layer: str, archived: bool = False, newest_first: bool = True, limit: int | None = None) -> list[str]:
        key = self.k(layer, "archive" if archived else "active")
        end = (limit - 1) if limit else -1
        return self.r.zrevrange(key, 0, end) if newest_first else self.r.zrange(key, 0, end)

    def find(self, layer: str, limit: int | None = None, **filters) -> list[Record]:
        """Active records whose indexed fields match every filter (value or list of accepted values)."""
        sets = []
        for f, v in filters.items():
            vals = v if isinstance(v, (list, tuple, set)) else [v]
            keys = [self.k(layer, "idx", f, x) for x in vals]
            if len(keys) == 1:
                sets.append(keys[0])
            else:
                tmp = self.k("tmp", f, *map(str, vals))[:200]
                self.r.sunionstore(tmp, keys)
                self.r.expire(tmp, 5)
                sets.append(tmp)
        active = set(self.ids(layer))
        ids = (self.r.sinter(sets) & active) if sets else active
        recs = self.get_many(layer, ids)
        recs.sort(key=lambda r: r.updated_at, reverse=True)
        return recs[:limit] if limit else recs

    def history(self, layer: str, rec_id: str) -> list[dict]:
        return [json.loads(v) for v in self.r.lrange(self.k(layer, "ver", rec_id), 0, -1)]

    def count(self, layer: str) -> int:
        return int(self.r.zcard(self.k(layer, "active")))

    # ------------------------------------------------------------------ events, usage, metrics
    def emit(self, event: str, **data) -> str:
        """Durable event (Redis Stream) + best-effort notification (Pub/Sub)."""
        payload = {"event": event, "t": time.time(), **data}
        body = json.dumps(payload, default=str)
        pipe = self.r.pipeline()
        pipe.xadd(self.k("events"), {"data": body}, maxlen=100_000, approximate=True)
        pipe.publish(self.k("notify"), body)
        return pipe.execute()[0]

    def use(self, rec_id: str, counter: str, n: int = 1) -> None:
        self.r.hincrby(self.k("usage", rec_id), counter, n)

    def usage(self, rec_id: str) -> dict[str, int]:
        return {k: int(v) for k, v in self.r.hgetall(self.k("usage", rec_id)).items()}

    def metric(self, name: str, n: float = 1) -> None:
        if isinstance(n, float):
            self.r.hincrbyfloat(self.k("metrics"), name, n)
        else:
            self.r.hincrby(self.k("metrics"), name, n)

    @contextmanager
    def timed(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = (time.perf_counter() - t0) * 1000
            pipe = self.r.pipeline()
            pipe.hincrbyfloat(self.k("metrics"), f"{name}.ms", dt)
            pipe.hincrby(self.k("metrics"), f"{name}.n", 1)
            pipe.execute()

    def metrics(self) -> dict:
        out = {}
        for k, v in self.r.hgetall(self.k("metrics")).items():
            out[k] = float(v) if "." in v else int(v)
        return out


def layer_of(ref: str) -> tuple[str, str]:
    """'M1:<id>' -> ('M1', '<id>')."""
    layer, _, rid = ref.partition(":")
    if layer not in LAYERS:
        raise ValueError(f"bad memory reference {ref}")
    return layer, rid
