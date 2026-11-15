"""
Experimental memory: what really happened. Observations are facts produced by experiments, never
written or rewritten by an LLM, and append-only: a new measurement of the same entity is a new
observation, it never overwrites an older one.

Each observation carries the Flowcept identifiers of the task that produced it. When the workflow
gives the Flowcept task_id, it is used as the observation id, so memory and provenance share ids.

Two experiments are comparable only under the same condition: same method and same parameters.
The condition id is a short hash of both, and every index is kept per condition.

Redis layout (under the namespace prefix):
  obs:{id}                 JSON of the observation (written once, SET NX)
  obs:all                  zset  id -> time
  conds                    set   condition ids
  cond:{c}:info            JSON  method and parameters of condition c
  cond:{c}:obs             zset  id -> time, observations of condition c
  cond:{c}:ent:{entity}    list  ids of the observations of one entity, oldest first
  cond:{c}:best            zset  entity -> best value (successful observations only)
  cond:{c}:failed          hash  entity -> id of a scientific failure (do not retry)
  cond:{c}:technical       hash  entity -> number of technical failures (may be retried)
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid

from .domain import SCIENTIFIC, Domain


def condition_id(method: str, params: dict | None) -> str:
    blob = json.dumps({"method": method, "params": params or {}}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


class ExperimentalMemory:
    def __init__(self, client, prefix: str, domain: Domain) -> None:
        self.r, self.p, self.domain = client, prefix, domain

    def k(self, *parts: str) -> str:
        return ":".join((self.p, *parts))

    # ------------------------------------------------------------------ write
    def record(self, entity: str, *, method: str, params: dict | None, value: float | None,
               error: str | None = None, agent: str | None = None, raw: str | None = None,
               obs_id: str | None = None, provenance: dict | None = None,
               started_at: float | None = None, ended_at: float | None = None) -> dict:
        """Store one observation. Idempotent on obs_id (a provenance record is never ingested twice)."""
        cond = condition_id(method, params)
        ok = error is None and value is not None and value == value   # NaN is a failure
        obs = {
            "id": obs_id or f"obs-{uuid.uuid4().hex[:12]}",
            "entity": entity, "raw": raw if raw is not None else entity,
            "condition": cond, "method": method, "params": params or {},
            "status": "success" if ok else "failed",
            "value": float(value) if ok else None,
            "failure_kind": None if ok else self.domain.classify_failure(error or ""),
            "error": None if ok else (error or f"value={value}")[:1000],
            "agent": agent, "started_at": started_at, "ended_at": ended_at or time.time(),
            "provenance": provenance or {},
        }
        if not self.r.set(self.k("obs", obs["id"]), json.dumps(obs, default=str), nx=True):
            return self.get(obs["id"])          # already recorded
        t = obs["ended_at"]
        pipe = self.r.pipeline()
        pipe.set(self.k("cond", cond, "info"), json.dumps({"method": method, "params": params or {}},
                                                           default=str), nx=True)
        pipe.sadd(self.k("conds"), cond)
        pipe.zadd(self.k("obs:all"), {obs["id"]: t})
        pipe.zadd(self.k("cond", cond, "obs"), {obs["id"]: t})
        pipe.rpush(self.k("cond", cond, "ent", entity), obs["id"])
        if ok:
            better = "gt" if self.domain.direction == "max" else "lt"
            pipe.zadd(self.k("cond", cond, "best"), {entity: obs["value"]}, **{better: True})
        elif obs["failure_kind"] == SCIENTIFIC:
            pipe.hset(self.k("cond", cond, "failed"), entity, obs["id"])
        else:
            pipe.hincrby(self.k("cond", cond, "technical"), entity, 1)
        pipe.execute()
        return obs

    # ------------------------------------------------------------------ read
    def get(self, obs_id: str) -> dict | None:
        raw = self.r.get(self.k("obs", obs_id))
        return json.loads(raw) if raw else None

    def get_many(self, ids: list[str]) -> list[dict | None]:
        if not ids:
            return []
        return [json.loads(v) if v else None for v in self.r.mget([self.k("obs", i) for i in ids])]

    def of_entity(self, entity: str, cond: str) -> list[dict]:
        ids = self.r.lrange(self.k("cond", cond, "ent", entity), 0, -1)
        return [o for o in self.get_many(ids) if o]

    def status(self, entity: str, cond: str) -> dict:
        """Experimental status of an entity: success (with the latest value), failed (scientific),
        or unknown; plus the number of technical failures."""
        pipe = self.r.pipeline()
        pipe.zscore(self.k("cond", cond, "best"), entity)
        pipe.hget(self.k("cond", cond, "failed"), entity)
        pipe.hget(self.k("cond", cond, "technical"), entity)
        best, failed_id, tech = pipe.execute()
        out = {"technical_failures": int(tech or 0)}
        if best is not None:
            last = [o for o in self.of_entity(entity, cond) if o["status"] == "success"][-1]
            return {**out, "state": "success", "value": last["value"], "obs_id": last["id"],
                    "agent": last.get("agent")}
        if failed_id:
            o = self.get(failed_id) or {}
            return {**out, "state": "failed", "obs_id": failed_id, "error": o.get("error"),
                    "agent": o.get("agent")}
        return {**out, "state": "unknown"}

    def best(self, cond: str, n: int = 10) -> list[tuple[str, float]]:
        key = self.k("cond", cond, "best")
        if self.domain.direction == "max":
            return self.r.zrevrange(key, 0, n - 1, withscores=True)
        return self.r.zrange(key, 0, n - 1, withscores=True)

    def failed(self, cond: str) -> dict[str, str]:
        return self.r.hgetall(self.k("cond", cond, "failed"))

    def in_condition(self, cond: str, limit: int | None = None) -> list[dict]:
        """Observations of a condition, oldest first (the `limit` most recent if given)."""
        key = self.k("cond", cond, "obs")
        ids = self.r.zrange(key, -limit, -1) if limit else self.r.zrange(key, 0, -1)
        return [o for o in self.get_many(ids) if o]

    def conditions(self) -> dict[str, dict]:
        conds = sorted(self.r.smembers(self.k("conds")))
        vals = self.r.mget([self.k("cond", c, "info") for c in conds]) if conds else []
        return {c: json.loads(v) for c, v in zip(conds, vals) if v}

    def count(self) -> int:
        return int(self.r.zcard(self.k("obs:all")))
