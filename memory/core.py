"""
Shared memory: the entry point used by a workflow.

Three components share one Redis instance, under one namespace:
  Coordination   who is computing what          (coordination.Coordinator)
  Memory         what happened, what was learned (observations.ExperimentalMemory, lessons.LessonMemory)
  Verification   may a lesson be reused?         (verification.Verifier)

Link with provenance: observations use the Flowcept task id of the computation as their id, lessons
cite observation ids, and every memory operation that matters for an agent's decisions (reuse of a
result, wait, lesson proposal, verification, briefing) can be emitted as a Flowcept task record
through `provenance_sink`, linked to the workflow by workflow_id / campaign_id / parent_task_id.
Flowcept keeps the full execution trace; the memory keeps only what is worth reusing.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
import uuid
from typing import Any, Awaitable, Callable, Iterable

from .coordination import Coordinator
from .domain import SCIENTIFIC, Domain
from .lessons import LessonMemory
from .observations import ExperimentalMemory, condition_id
from .proposer import LessonProposer
from .verification import Verifier


INVALID = "invalid:"   # key prefix of proposals that are not valid entities (domain.canonical -> None)


class KnownFailure(Exception):
    """The memory already knows that this experiment fails: it is not run again."""


class Memory:
    def __init__(self, client, domain: Domain, namespace: str = "default", agent: str = "agent", *,
                 claim_ttl_s: float = 300.0, max_technical_attempts: int = 2, min_per_group: int = 2,
                 provenance_sink: Callable[[dict], None] | None = None,
                 context: Callable[[], dict] | None = None) -> None:
        self.r, self.domain, self.namespace, self.agent = client, domain, namespace, agent
        self.p = f"memory:{namespace}"
        self.coord = Coordinator(client, self.p, claim_ttl_s)
        self.obs = ExperimentalMemory(client, self.p, domain)
        self.lessons = LessonMemory(client, self.p)
        self.verifier = Verifier(self.obs, domain, min_per_group)
        self.proposer = LessonProposer(domain)
        self.claim_ttl_s, self.max_technical_attempts = claim_ttl_s, max_technical_attempts
        self.sink, self.context = provenance_sink, context
        client.zadd("memory:namespaces", {namespace: time.time()}, nx=True)

    @classmethod
    def connect(cls, domain: Domain, namespace: str = "default", host: str = "localhost", port: int = 6379,
                **kw) -> "Memory":
        import redis
        client = redis.Redis(host=host, port=port, decode_responses=True)
        client.ping()
        return cls(client, domain, namespace, **kw)

    # ================================================================== provenance and bookkeeping
    def _stat(self, field: str, n: float = 1) -> None:
        if isinstance(n, float):
            self.r.hincrbyfloat(f"{self.p}:stats", field, n)
        else:
            self.r.hincrby(f"{self.p}:stats", field, n)

    def _event(self, kind: str, **data: Any) -> None:
        """History of important events (stream, the source of truth) + a pub/sub notification for live
        listeners (best effort: a missed notification loses nothing, the state is in Redis)."""
        payload = json.dumps({"kind": kind, "agent": self.agent, "t": time.time(), **data}, default=str)
        pipe = self.r.pipeline()
        pipe.xadd(f"{self.p}:events", {"event": payload}, maxlen=10000, approximate=True)
        pipe.publish(f"{self.p}:notify", payload)
        pipe.execute()

    def _emit(self, activity: str, used: dict, generated: dict, started_at: float,
              provenance: dict | None = None) -> str | None:
        """Write a memory operation as a Flowcept task record (if a sink is configured)."""
        if self.sink is None:
            return None
        record = {"task_id": str(uuid.uuid4()), "activity_id": activity, "subtype": "memory",
                  "status": "FINISHED", "started_at": started_at, "ended_at": time.time(),
                  "used": used, "generated": generated,
                  "custom_metadata": {"agent": self.agent, "memory_namespace": self.namespace}}
        ids = {**(self.context() if self.context else {}), **(provenance or {})}
        record.update({k: v for k, v in ids.items() if k in ("workflow_id", "campaign_id", "parent_task_id") and v})
        try:
            self.sink(record)
        except Exception:
            return None
        return record["task_id"]

    # ================================================================== coordination + observations
    def key(self, raw: str) -> str:
        return self.domain.canonical(raw) or f"{INVALID}{str(raw).strip()}"

    @staticmethod
    def shown(entity: str) -> str:
        """How an entity is written for agents (an invalid entity as it was proposed)."""
        return entity[len(INVALID):] if entity.startswith(INVALID) else entity

    def get_status(self, raw: str, method: str, params: dict | None = None) -> dict:
        """success / failed (scientific) / running / retryable (technical failure) / unknown."""
        entity, cond = self.key(raw), condition_id(method, params)
        st = self.obs.status(entity, cond)
        if st["state"] == "unknown":
            holder = self.coord.holder(f"{cond}/{entity}")
            if holder:
                st = {**st, "state": "running", "agent": holder}
            elif st["technical_failures"]:
                st = {**st, "state": "retryable"}
        return {"entity": entity, "condition": cond, **st}

    def _step(self, entity: str, cond: str) -> tuple[str, Any]:
        """One coordination decision: ('reuse', status) | ('fail', message) | ('run', token) | ('wait', holder)."""
        st = self.obs.status(entity, cond)
        if st["state"] == "success":
            return "reuse", st
        if st["state"] == "failed":
            return "fail", f"{entity} failed before ({SCIENTIFIC}): {st.get('error')}"
        if st["technical_failures"] >= self.max_technical_attempts:
            return "fail", f"{entity}: gave up after {st['technical_failures']} technical failures"
        token = self.coord.claim(f"{cond}/{entity}", self.agent)
        if token is None:
            return "wait", self.coord.holder(f"{cond}/{entity}")
        if self.obs.status(entity, cond)["state"] != "unknown":   # finished between the check and the claim
            self.coord.release(f"{cond}/{entity}", token)
            return self._step(entity, cond)
        return "run", token

    def _finish_step(self, kind: str, info: Any, entity: str, cond: str, t0: float, provenance: dict | None):
        if kind == "reuse":
            self._stat("reused")
            self._emit("memory.reuse", {"entity": entity, "condition": cond, "obs_id": info["obs_id"]},
                       {"value": info["value"], "computed_by": info.get("agent")}, t0, provenance)
            return info["value"]
        self._stat("blocked_failure")
        self._emit("memory.blocked", {"entity": entity, "condition": cond}, {"reason": info}, t0, provenance)
        raise KnownFailure(info)

    async def get_or_compute(self, raw: str, compute: Callable[[str], Awaitable[float]], *, method: str,
                             params: dict | None = None, task_id: str | None = None,
                             provenance: dict | None = None, poll_s: float = 1.0) -> float:
        """Value of an experiment, computed at most once by the whole team:
        reuse a known result, refuse a known scientific failure, wait for an agent already computing it,
        or claim it, run `compute(task_id)`, record the observation and release the claim.
        `task_id` (default: a new uuid) is passed to `compute` so the workflow records its provenance
        under the same id as the observation."""
        entity, cond, t0 = self.key(raw), condition_id(method, params), time.time()
        while True:
            kind, info = self._step(entity, cond)
            if kind in ("reuse", "fail"):
                return self._finish_step(kind, info, entity, cond, t0, provenance)
            if kind == "wait":
                self._stat("waited")
                w0 = time.time()
                while self.coord.holder(f"{cond}/{entity}") and self.obs.status(entity, cond)["state"] == "unknown":
                    await asyncio.sleep(poll_s)
                self._emit("memory.wait", {"entity": entity, "condition": cond, "holder": info},
                           {"waited_s": round(time.time() - w0, 3)}, w0, provenance)
                continue
            tid, c0 = task_id or str(uuid.uuid4()), time.time()
            keep_alive = asyncio.create_task(self._keep_alive_async(f"{cond}/{entity}", info))
            try:
                value = await compute(tid)
            except Exception as exc:
                self._record(entity, raw, method, params, None, str(exc) or type(exc).__name__, tid, provenance, c0)
                self._stat("computed_failed")
                raise
            else:   # the observation is recorded BEFORE the claim is released: no gap for a duplicate
                self._record(entity, raw, method, params, value, None, tid, provenance, c0)
                self._stat("computed")
                return value
            finally:
                keep_alive.cancel()
                self.coord.release(f"{cond}/{entity}", info)

    def get_or_compute_sync(self, raw: str, compute: Callable[[str], float], *, method: str,
                            params: dict | None = None, task_id: str | None = None,
                            provenance: dict | None = None, poll_s: float = 1.0) -> float:
        """Same as get_or_compute, for synchronous workflows."""
        entity, cond, t0 = self.key(raw), condition_id(method, params), time.time()
        while True:
            kind, info = self._step(entity, cond)
            if kind in ("reuse", "fail"):
                return self._finish_step(kind, info, entity, cond, t0, provenance)
            if kind == "wait":
                self._stat("waited")
                w0 = time.time()
                while self.coord.holder(f"{cond}/{entity}") and self.obs.status(entity, cond)["state"] == "unknown":
                    time.sleep(poll_s)
                self._emit("memory.wait", {"entity": entity, "condition": cond, "holder": info},
                           {"waited_s": round(time.time() - w0, 3)}, w0, provenance)
                continue
            tid, stop, c0 = task_id or str(uuid.uuid4()), threading.Event(), time.time()
            threading.Thread(target=self._keep_alive_thread, args=(f"{cond}/{entity}", info, stop),
                             daemon=True).start()
            try:
                value = compute(tid)
            except Exception as exc:
                self._record(entity, raw, method, params, None, str(exc) or type(exc).__name__, tid, provenance, c0)
                self._stat("computed_failed")
                raise
            else:
                self._record(entity, raw, method, params, value, None, tid, provenance, c0)
                self._stat("computed")
                return value
            finally:
                stop.set()
                self.coord.release(f"{cond}/{entity}", info)

    async def _keep_alive_async(self, task: str, token: str) -> None:
        while True:
            await asyncio.sleep(self.claim_ttl_s / 3)
            if not self.coord.renew(task, token):
                return

    def _keep_alive_thread(self, task: str, token: str, stop: threading.Event) -> None:
        while not stop.wait(self.claim_ttl_s / 3):
            if not self.coord.renew(task, token):
                return

    def _record(self, entity, raw, method, params, value, error, task_id, provenance, started_at) -> dict:
        prov = {**(self.context() if self.context else {}), **(provenance or {}), "task_id": task_id}
        obs = self.obs.record(entity, method=method, params=params, value=value, error=error, agent=self.agent,
                              raw=raw, obs_id=task_id, provenance={k: v for k, v in prov.items() if v},
                              started_at=started_at)
        self._event("observation", obs_id=obs["id"], entity=entity, status=obs["status"])
        return obs

    def record_observation(self, raw: str, *, method: str, params: dict | None = None,
                           value: float | None = None, error: str | None = None, task_id: str | None = None,
                           provenance: dict | None = None, agent: str | None = None,
                           started_at: float | None = None, ended_at: float | None = None) -> dict:
        """Store an observation made outside get_or_compute (e.g. by another tool)."""
        obs = self.obs.record(self.key(raw), method=method, params=params, value=value, error=error,
                              agent=agent or self.agent, raw=raw, obs_id=task_id,
                              provenance=provenance, started_at=started_at, ended_at=ended_at)
        self._event("observation", obs_id=obs["id"], entity=obs["entity"], status=obs["status"])
        return obs

    def ingest_flowcept(self, records: Iterable[dict]) -> int:
        """Rebuild / complete the experimental memory from Flowcept task records (e.g. the JSONL buffer
        of earlier campaigns). Idempotent: an observation id is the Flowcept task id."""
        n = 0
        for rec in records:
            fields = self.domain.observation_from_task(rec)
            if not fields or not rec.get("task_id"):
                continue
            before = self.obs.get(rec["task_id"])
            self.record_observation(fields.pop("raw"), task_id=rec["task_id"], **fields)
            n += before is None
        return n

    def known_entities(self, method: str, params: dict | None = None, limit: int = 200) -> list[str]:
        """Entities not worth proposing again: measured, failed for scientific reasons, or running."""
        cond = condition_id(method, params)
        best = [e for e, _ in self.obs.best(cond, limit)]
        failed = list(self.obs.failed(cond))[:limit]
        running = [t.split("/", 1)[1] for t in self.coord.running() if t.startswith(cond + "/")]
        return list(dict.fromkeys(self.shown(e) for e in best + failed + running))[:limit]

    # ================================================================== lessons
    def lesson_prompt(self, conclusions: str, method: str, params: dict | None = None,
                      observations: list[dict] | None = None) -> tuple[str, str, dict]:
        """Prompt asking an LLM to turn conclusions into structured lessons (see proposer)."""
        if observations is None:
            observations = self.obs.in_condition(condition_id(method, params), limit=self.proposer.max_rows)
        return self.proposer.prompt(conclusions, observations)

    def begin_extraction(self, method: str, params: dict | None = None, min_new: int = 3,
                         ttl_s: float = 600.0) -> tuple[str, int] | None:
        """Decide whether turning conclusions into lessons (an LLM call) is worth it now.
        Granted only when at least `min_new` observations arrived under this condition since the last
        extraction, and only to one agent at a time (a claim of the coordination component).
        Returns (token, number of observations) to pass to end_extraction, or None to skip."""
        cond = condition_id(method, params)
        n_obs = int(self.r.zcard(f"{self.p}:cond:{cond}:obs"))
        done = int(self.r.get(f"{self.p}:cond:{cond}:extracted") or 0)
        if n_obs - done < max(1, min_new):
            self._stat("extractions_skipped_no_new_data")
            return None
        token = self.coord.claim(f"_extract/{cond}", self.agent, ttl_s=ttl_s)
        if token is None:
            self._stat("extractions_skipped_busy")
            return None
        return token, n_obs

    def end_extraction(self, method: str, params: dict | None, slot: tuple[str, int], success: bool) -> None:
        """Close an extraction: on success, the observations it saw no longer count as new."""
        cond = condition_id(method, params)
        token, n_obs = slot
        if success:
            self.r.set(f"{self.p}:cond:{cond}:extracted", n_obs)
            self._stat("extractions")
        self.coord.release(f"_extract/{cond}", token)

    def _lesson_key(self, statement: str, claim: dict | None, cond: str | None) -> str:
        """Two proposals are the same lesson when they make the same checkable claim under the same
        condition (or, without a checkable claim, have the same wording)."""
        claim = dict(claim or {"kind": "none"})
        if claim.get("kind") == "comparison":
            claim["better"], claim["worse"] = self.key(claim.get("better", "")), self.key(claim.get("worse", ""))
        elif claim.get("kind") == "feature_effect":
            claim = {"kind": "feature_effect", "feature": claim.get("feature"), "op": claim.get("op", ">"),
                     "value": float(claim.get("value", 0)), "effect": claim.get("effect", "higher")}
        else:
            claim = {"kind": claim.get("kind", "none"), "text": " ".join(statement.lower().split())}
        blob = json.dumps({"cond": cond, "claim": claim}, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def propose_lesson(self, statement: str, *, evidence_ids: list[str], claim: dict | None = None,
                       type: str = "strategy", conditions: str = "", evidence_source: str = "cited",
                       provenance: dict | None = None) -> dict:
        """Propose a lesson (status pending). If the same lesson already exists, the proposal is merged
        into it instead: its evidence grows, the agent is recorded as endorsing it, and it is re-verified."""
        t0 = time.time()
        found = [o for o in self.obs.get_many(evidence_ids) if o]
        conds = {o["condition"] for o in found}
        index_key = self._lesson_key(statement, claim, next(iter(conds)) if len(conds) == 1 else None)
        tid = self._emit("memory.propose_lesson",
                         {"statement": statement, "claim": claim, "evidence_ids": evidence_ids},
                         {"lesson_key": index_key}, t0, provenance)
        existing = self.r.hget(f"{self.p}:lessons:index", index_key)
        if existing and (lesson := self.lessons.get(existing)):
            return self._endorse(lesson, evidence_ids, tid)
        prov = {**(self.context() if self.context else {}), **(provenance or {}), "proposal_task_id": tid}
        lesson = self.lessons.propose(statement, created_by=self.agent, evidence_ids=evidence_ids, claim=claim,
                                      type=type, conditions=conditions,
                                      provenance={k: v for k, v in prov.items() if v},
                                      extra={"condition": conds.pop() if len(conds) == 1 else None,
                                             "evidence_source": evidence_source, "endorsed_by": [self.agent],
                                             "key": index_key})
        self.r.hsetnx(f"{self.p}:lessons:index", index_key, lesson["id"])
        self._stat("lessons_proposed")
        self._event("lesson_proposed", lesson_id=lesson["id"])
        return lesson

    def _endorse(self, lesson: dict, evidence_ids: list[str], proposal_task_id: str | None) -> dict:
        lesson["evidence_ids"] = list(dict.fromkeys(lesson["evidence_ids"] + [str(e) for e in evidence_ids]))
        if self.agent not in lesson.setdefault("endorsed_by", []):
            lesson["endorsed_by"].append(self.agent)
        lesson.setdefault("provenance", {}).setdefault("endorsement_task_ids", []).append(proposal_task_id)
        self.r.set(f"{self.p}:lesson:{lesson['id']}", json.dumps(lesson, default=str))
        self.r.rpush(f"{self.p}:lesson:{lesson['id']}:history",
                     json.dumps({"t": time.time(), "status": lesson["status"], "by": self.agent,
                                 "reason": f"proposed again: {len(evidence_ids)} observation(s) cited"}))
        self._stat("lessons_endorsed")
        if lesson["status"] != "pending" and lesson.get("verification_method") != "human-review":
            self._apply_verdict(lesson, by=f"verifier@{self.agent}")    # more evidence: check again
        return lesson

    def propose_from_llm(self, answer: str, refs: dict[str, str], provenance: dict | None = None) -> list[dict]:
        return [self.propose_lesson(l["statement"], evidence_ids=l["evidence_ids"], claim=l["claim"],
                                    type=l["type"], evidence_source=l["evidence_source"], provenance=provenance)
                for l in self.proposer.parse(answer, refs)]

    def _apply_verdict(self, lesson: dict, by: str) -> bool:
        t0 = time.time()
        status, reason, details = self.verifier.verify(lesson)
        if status == lesson["status"] and reason == lesson.get("status_reason"):
            return False
        tid = self._emit("memory.verify_lesson",
                         {"lesson_id": lesson["id"], "evidence_ids": lesson["evidence_ids"],
                          "claim": lesson.get("claim"), "previous_status": lesson["status"]},
                         {"status": status, "reason": reason}, t0)
        self.lessons.set_status(lesson, status, reason, method=self.verifier.method, by=by, details=details,
                                extra={"verification_task_id": tid} if tid else None)
        if status != "pending":
            self._stat(status)
        self._event("lesson_status", lesson_id=lesson["id"], status=status)
        return True

    def verify_pending(self, max_n: int = 100) -> list[tuple[str, str]]:
        """Verify the lessons waiting in the queue, then re-check reusable lessons if new observations
        arrived. Deterministic and cheap: any agent (or a separate verifier process) may call it."""
        done = []
        for lid in self.lessons.pop_pending(max_n):
            lesson = self.lessons.get(lid)
            if lesson and lesson["status"] == "pending":
                self._apply_verdict(lesson, by=f"verifier@{self.agent}")
                done.append((lid, lesson["status"]))
        self.recheck()
        return done

    def recheck(self) -> int:
        """Re-verify verified and needs_review lessons when new observations arrived since the last
        check (one verifier at a time, through the coordination component)."""
        n_obs = self.obs.count()
        last = int(self.r.get(f"{self.p}:meta:recheck_obs") or 0)
        if n_obs == last:
            return 0
        token = self.coord.claim("_recheck", self.agent, ttl_s=60)
        if token is None:
            return 0
        try:
            changed = 0
            for status in ("verified", "needs_review"):
                for lesson in self.lessons.by_status(status, limit=500):
                    if lesson.get("verification_method") == "human-review":
                        continue
                    changed += self._apply_verdict(lesson, by=f"recheck@{self.agent}")
            self.r.set(f"{self.p}:meta:recheck_obs", n_obs)
            return changed
        finally:
            self.coord.release("_recheck", token)

    def review(self, lesson_id: str, status: str, reason: str, reviewer: str = "human") -> dict:
        """Decision of a human (or of an external analysis component) on a lesson."""
        lesson = self.lessons.get(lesson_id)
        if lesson is None:
            raise KeyError(lesson_id)
        self._event("lesson_status", lesson_id=lesson_id, status=status, by=reviewer)
        return self.lessons.set_status(lesson, status, reason, method="human-review", by=reviewer)

    # ================================================================== retrieval
    def briefing(self, method: str, params: dict | None = None, max_chars: int = 2500, n_best: int = 10,
                 n_lessons: int = 5, n_failed: int = 10, show_open: bool = True,
                 provenance: dict | None = None, label: str | None = None) -> str:
        """Bounded, relevant summary of the memory for an agent's prompt: results of the team, work in
        progress, scientific failures, and lessons by status. Only verified lessons are presented as
        knowledge; rejected ones are listed as warnings and open questions as things to test."""
        t0, cond = time.time(), condition_id(method, params)
        better = "higher" if self.domain.direction == "max" else "lower"
        sections: list[tuple[str, list[tuple[str | None, str]]]] = []
        sections.append((f"Best results of the team ({self.domain.objective}, {better} is better):",
                         [(None, f"  {e}: {v:.6g}") for e, v in self.obs.best(cond, n_best)]))
        running = [(None, f"  {self.shown(t.split('/', 1)[1])} (by {a})") for t, a in self.coord.running().items()
                   if t.startswith(cond + "/") and a != self.agent]
        sections.append(("Being computed now by other agents (do not propose them):", running))
        sections.append(("Failed for scientific reasons (do not retry):",
                         [(None, f"  {self.shown(e)}") for e in list(self.obs.failed(cond))[:n_failed]]))

        def lessons(status, n, fmt):
            ls = [l for l in self.lessons.by_status(status, limit=100) if l.get("condition") in (cond, None)]
            return [(l["id"], fmt(l)) for l in ls[:n]]
        sections.append(("Verified lessons (checked against the measurements, reusable):",
                         lessons("verified", n_lessons, lambda l: f"  - {l['statement']}")))
        sections.append(("Rejected lessons (contradicted by the measurements, do NOT rely on them):",
                         lessons("rejected", 3, lambda l: f"  - {l['statement']} -- {l['status_reason']}")))
        if show_open:
            sections.append(("Open questions (not confirmed yet, worth testing):",
                             lessons("needs_review", 3, lambda l: f"  - {l['statement']}")))

        lines = ["SHARED MEMORY OF THE TEAM:"]
        used, shown = len(lines[0]), []
        for title, items in sections:
            if not items:
                continue
            if used + len(title) + 1 > max_chars:
                break
            lines.append(title)
            used += len(title) + 1
            for lid, text in items:
                if used + len(text) + 1 > max_chars:
                    break
                lines.append(text)
                used += len(text) + 1
                if lid:
                    shown.append(lid)
        text = "\n".join(lines) if len(lines) > 1 else ""
        self._stat("briefings")
        self._emit("memory.briefing", {"condition": cond, "label": label},
                   {"lesson_ids": shown, "chars": len(text)}, t0, provenance)
        return text

    # ================================================================== report
    def report(self) -> dict:
        stats = {k: float(v) if "." in v else int(v) for k, v in self.r.hgetall(f"{self.p}:stats").items()}
        conds = self.obs.conditions()
        return {"namespace": self.namespace, "observations": self.obs.count(), "conditions": conds,
                "best": {c: self.obs.best(c, 5) for c in conds}, "lessons": self.lessons.counts(),
                "stats": stats}
