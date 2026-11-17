"""
M1 — Scientific Results Memory: what was measured.

Successful measurements and valid negative scientific results (an experiment that is scientifically
impossible, e.g. an invalid structure). Technical failures are NOT results: they go to M2.
A result's id is the Flowcept task id of the computation that produced it, so M1 and provenance share ids.
Results are append-only: a new measurement of the same entity is a new record. When a repeated
measurement disagrees with the previous ones, both are kept and a conflict is reported.

Fast lookup structures (per condition c):
  M1:best:{c}       zset entity -> best value
  M1:negative:{c}   hash entity -> id of a negative result
  M1:ent:{c}:{e}    list of result ids of one entity
"""
from __future__ import annotations

import time
from statistics import mean

from ..domain import Domain, condition_id
from ..schemas import ResultRecord, prov_ref
from ..store import Store


class ResultsMemory:
    def __init__(self, store: Store, domain: Domain, tolerance: float = 1e-6) -> None:
        self.s, self.domain, self.tolerance = store, domain, tolerance

    def key_of(self, raw: str) -> str:
        return self.domain.canonical(raw) or f"invalid:{str(raw).strip()}"

    # ------------------------------------------------------------------ write
    def record(self, raw: str, *, method: str, params: dict | None, agent: str, value: float | None = None,
               negative_detail: str | None = None, task_id: str | None = None, workflow_id: str | None = None,
               campaign_id: str | None = None, duration_s: float | None = None) -> tuple[ResultRecord, dict | None]:
        """Store a measurement (value) or a valid negative result (negative_detail). Idempotent on task_id.
        Returns (record, conflict) where conflict describes a disagreement with earlier measurements."""
        entity, cond = self.key_of(raw), condition_id(method, params)
        rec = ResultRecord(
            id=task_id or "", key=task_id or "", entity=entity, raw_input=str(raw), method=method,
            params=params or {}, condition=cond, objective=self.domain.objective, unit=self.domain.unit,
            outcome="negative" if negative_detail is not None else "success",
            value=None if negative_detail is not None else value, detail=negative_detail or "", agent=agent,
            duration_s=duration_s, created_by=agent,
            provenance=[prov_ref(task_id, workflow_id, campaign_id, self.domain.activity)] if task_id else [])
        previous = self.of_entity(entity, cond)
        rec, created = self.s.add(rec, actor=agent, reason="measured")
        if not created:
            return rec, None
        pipe = self.s.r.pipeline()
        pipe.rpush(self.s.k("M1", "ent", cond, entity), rec.id)
        if rec.outcome == "success":
            pipe.zadd(self.s.k("M1", "best", cond), {entity: rec.value},
                      **{("gt" if self.domain.direction == "max" else "lt"): True})
        else:
            pipe.hset(self.s.k("M1", "negative", cond), entity, rec.id)
        pipe.execute()
        conflict = self._conflict(rec, previous)
        self.s.emit("result", id=rec.id, entity=entity, condition=cond, outcome=rec.outcome, value=rec.value,
                    agent=agent, task_id=task_id, conflict=bool(conflict))
        return rec, conflict

    def _conflict(self, rec: ResultRecord, previous: list[ResultRecord]) -> dict | None:
        """A repeated measurement that disagrees with the earlier ones (outcome or value)."""
        if not previous:
            return None
        outcomes = {p.outcome for p in previous}
        if rec.outcome not in outcomes or len(outcomes) > 1:
            return {"type": "outcome", "entity": rec.entity, "ids": [p.id for p in previous] + [rec.id]}
        if rec.outcome == "success":
            ref = mean(p.value for p in previous)
            if abs(rec.value - ref) > self.tolerance * max(1.0, abs(ref)):
                return {"type": "value", "entity": rec.entity, "previous_mean": ref, "new": rec.value,
                        "ids": [p.id for p in previous] + [rec.id]}
        return None

    # ------------------------------------------------------------------ read
    def of_entity(self, entity: str, cond: str) -> list[ResultRecord]:
        ids = self.s.r.lrange(self.s.k("M1", "ent", cond, entity), 0, -1)
        return [r for r in self.s.get_many("M1", ids) if r.status != "archived"]

    def lookup(self, raw: str, method: str, params: dict | None) -> dict:
        """Exact retrieval: what the team knows about one experiment."""
        entity, cond = self.key_of(raw), condition_id(method, params)
        recs = self.of_entity(entity, cond)
        ok = [r for r in recs if r.outcome == "success"]
        if ok:
            return {"state": "success", "entity": entity, "condition": cond, "value": mean(r.value for r in ok),
                    "n": len(ok), "ids": [r.id for r in ok], "agent": ok[-1].agent, "id": ok[-1].id}
        neg = [r for r in recs if r.outcome == "negative"]
        if neg:
            return {"state": "negative", "entity": entity, "condition": cond, "detail": neg[-1].detail,
                    "ids": [r.id for r in neg], "agent": neg[-1].agent, "id": neg[-1].id}
        return {"state": "unknown", "entity": entity, "condition": cond}

    def best(self, cond: str, n: int = 10) -> list[tuple[str, float]]:
        key = self.s.k("M1", "best", cond)
        if self.domain.direction == "max":
            return self.s.r.zrevrange(key, 0, n - 1, withscores=True)
        return self.s.r.zrange(key, 0, n - 1, withscores=True)

    def negatives(self, cond: str) -> dict[str, str]:
        return self.s.r.hgetall(self.s.k("M1", "negative", cond))

    def in_condition(self, cond: str, limit: int | None = None) -> list[ResultRecord]:
        recs = self.s.find("M1", condition=cond)
        recs.sort(key=lambda r: r.created_at)
        return recs[-limit:] if limit else recs

    def per_entity(self, cond: str, since: float | None = None) -> dict[str, dict]:
        """Consolidated view: one summary per entity (mean of repeated measurements, all ids kept)."""
        out: dict[str, dict] = {}
        for r in self.in_condition(cond):
            if since and r.created_at <= since:
                continue
            e = out.setdefault(r.entity, {"values": [], "ids": [], "negative": False, "first": r.created_at})
            e["ids"].append(r.id)
            if r.outcome == "success":
                e["values"].append(r.value)
            else:
                e["negative"] = True
        for e in out.values():
            e["mean"] = mean(e["values"]) if e["values"] else None
        return out

    def count(self, cond: str | None = None) -> int:
        if cond is None:
            return self.s.count("M1")
        return int(self.s.r.scard(self.s.k("M1", "idx", "condition", cond)))

    def conditions(self) -> list[str]:
        return [k.rsplit(":", 1)[1] for k in self.s.r.scan_iter(match=self.s.k("M1", "idx", "condition", "*"))]

    def recent(self, since: float) -> list[ResultRecord]:
        ids = self.s.r.zrangebyscore(self.s.k("M1", "active"), since, "+inf")
        return self.s.get_many("M1", ids)

    def now(self) -> float:
        return time.time()
