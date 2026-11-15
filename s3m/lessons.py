"""
Lessons memory: what was learned. A lesson is an interpretation of one or several observations,
proposed by an agent. It is stored with the ids of the observations it relies on, and with a
machine-checkable claim when it has one, so that it can be verified before any agent reuses it.

Status of a lesson:
  pending       proposed, not checked yet (never shown to agents as knowledge)
  verified      passed the verification rules: may be reused as knowledge
  rejected      contradicted by its evidence, or its evidence is missing / invalid (kept, with the
                reason, so the same mistake is not made again)
  needs_review  neither clearly true nor clearly false (contradicting data, too little evidence,
                not checkable automatically): a further experiment or a human must decide

Every status change is appended to the lesson's history with its reason and time; nothing is deleted.

Redis layout (under the namespace prefix):
  lesson:{id}              JSON of the lesson (current state)
  lesson:{id}:history      list of status changes
  lessons:status:{status}  zset  id -> time of the last change
  lessons:queue            list  ids waiting for verification
"""
from __future__ import annotations

import json
import time
import uuid

STATUSES = ("pending", "verified", "rejected", "needs_review")
TYPES = ("success", "failure", "strategy", "constraint")


class LessonMemory:
    def __init__(self, client, prefix: str) -> None:
        self.r, self.p = client, prefix

    def k(self, *parts: str) -> str:
        return ":".join((self.p, *parts))

    def propose(self, statement: str, *, created_by: str, evidence_ids: list[str] | None = None,
                claim: dict | None = None, type: str = "strategy", conditions: str = "",
                provenance: dict | None = None, extra: dict | None = None) -> dict:
        now = time.time()
        lesson = {
            "id": f"lesson-{uuid.uuid4().hex[:10]}",
            "statement": statement.strip(),
            "type": type if type in TYPES else "strategy",
            "claim": claim or {"kind": "none"},
            "evidence_ids": [str(e) for e in (evidence_ids or [])],
            "conditions": conditions,
            "created_by": created_by, "created_at": now,
            "status": "pending", "status_reason": "waiting for verification",
            "verification_method": None, "verification": {},
            "provenance": provenance or {},
            **(extra or {}),
        }
        pipe = self.r.pipeline()
        pipe.set(self.k("lesson", lesson["id"]), json.dumps(lesson, default=str))
        pipe.zadd(self.k("lessons:status:pending"), {lesson["id"]: now})
        pipe.rpush(self.k("lesson", lesson["id"], "history"),
                   json.dumps({"t": now, "status": "pending", "reason": "proposed", "by": created_by}))
        pipe.rpush(self.k("lessons:queue"), lesson["id"])
        pipe.execute()
        return lesson

    def get(self, lesson_id: str) -> dict | None:
        raw = self.r.get(self.k("lesson", lesson_id))
        return json.loads(raw) if raw else None

    def get_many(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        return [json.loads(v) for v in self.r.mget([self.k("lesson", i) for i in ids]) if v]

    def set_status(self, lesson: dict, status: str, reason: str, *, method: str, by: str,
                   details: dict | None = None, extra: dict | None = None) -> dict:
        """Change the status of a lesson and append the change to its history."""
        if status not in STATUSES:
            raise ValueError(f"unknown status {status}")
        now, old = time.time(), lesson["status"]
        lesson.update(status=status, status_reason=reason, verification_method=method,
                      verification=details or {}, **(extra or {}))
        change = {"t": now, "from": old, "status": status, "reason": reason, "method": method, "by": by,
                  **({"provenance_task_id": extra["verification_task_id"]}
                     if extra and extra.get("verification_task_id") else {})}
        pipe = self.r.pipeline()
        pipe.set(self.k("lesson", lesson["id"]), json.dumps(lesson, default=str))
        pipe.zrem(self.k("lessons:status", old), lesson["id"])
        pipe.zadd(self.k("lessons:status", status), {lesson["id"]: now})
        pipe.rpush(self.k("lesson", lesson["id"], "history"), json.dumps(change, default=str))
        pipe.execute()
        return lesson

    def pop_pending(self, n: int = 50) -> list[str]:
        """Take lessons from the verification queue (atomic: each lesson goes to one verifier)."""
        return self.r.lpop(self.k("lessons:queue"), n) or []

    def requeue(self, lesson_id: str) -> None:
        self.r.rpush(self.k("lessons:queue"), lesson_id)

    def by_status(self, status: str, limit: int = 50, newest_first: bool = True) -> list[dict]:
        key = self.k("lessons:status", status)
        ids = (self.r.zrevrange(key, 0, limit - 1) if newest_first else self.r.zrange(key, 0, limit - 1))
        return self.get_many(ids)

    def history(self, lesson_id: str) -> list[dict]:
        return [json.loads(h) for h in self.r.lrange(self.k("lesson", lesson_id, "history"), 0, -1)]

    def counts(self) -> dict[str, int]:
        pipe = self.r.pipeline()
        for s in STATUSES:
            pipe.zcard(self.k("lessons:status", s))
        return dict(zip(STATUSES, pipe.execute()))
