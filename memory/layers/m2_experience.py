"""
M2 — Experience & Diagnostic Memory: what happened to the agents and why (as far as we know).

Incidents (technical failures), anomalies, recurring failures, redundant computations, inefficient
exploration, low diversity, unused knowledge, knowledge conflicts, ineffective guidance, diagnostic
hypotheses, confirmed causes, attempted actions and recovery episodes.

An episode is identified by a signature (e.g. "transient:xtb:<error>"): the same problem seen again
increments its occurrences instead of creating a new record. Evidence is referenced (Flowcept task ids,
memory ids), never copied. Facts, candidate explanations and confirmed causes are kept apart: a candidate
explanation is a hypothesis; a cause is "confirmed" only by an explicit rule or a human.
"""
from __future__ import annotations

import hashlib
import time

from ..schemas import ExperienceRecord
from ..store import Store

MAX_LIST = 50   # bounded lists: occurrences are counted, evidence is sampled


def signature_of(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def _merge(old: list, new: list, limit: int = MAX_LIST) -> list:
    out = list(old)
    for x in new:
        if x not in out:
            out.append(x)
    return out[-limit:]


class ExperienceMemory:
    def __init__(self, store: Store) -> None:
        self.s = store

    def report(self, kind: str, summary: str, signature: str, *, by: str, facts: list[str] | None = None,
               explanations: list[str] | None = None, severity: str = "low", agents: list[str] | None = None,
               entities: list[str] | None = None, provenance: list[dict] | None = None,
               memory_refs: list[str] | None = None, error_class: str | None = None,
               metrics: dict | None = None, ttl_s: float | None = None) -> tuple[ExperienceRecord, bool]:
        """Record an episode, or add an occurrence to the open episode with the same signature.
        Returns (record, created)."""
        now = time.time()
        rec = ExperienceRecord(kind=kind, summary=summary, signature=signature, key=f"{kind}:{signature}",
                               observed_facts=facts or [], candidate_explanations=explanations or [],
                               severity=severity, agents=agents or [], entities=entities or [],
                               provenance=(provenance or [])[:MAX_LIST], memory_refs=memory_refs or [],
                               error_class=error_class, metrics=metrics or {}, created_by=by,
                               first_seen=now, last_seen=now, expires_at=now + ttl_s if ttl_s else None)
        stored, created = self.s.add(rec, actor=by, reason="reported")
        if created:
            self.s.emit("experience", id=stored.id, kind=kind, signature=signature, severity=severity)
            return stored, True

        def add_occurrence(r: ExperienceRecord):
            r.occurrences += 1
            r.last_seen = now
            r.summary = summary
            r.observed_facts = _merge(r.observed_facts, facts or [])
            r.candidate_explanations = _merge(r.candidate_explanations, explanations or [])
            r.agents = _merge(r.agents, agents or [])
            r.entities = _merge(r.entities, entities or [])
            r.provenance = _merge(r.provenance, provenance or [])
            r.memory_refs = _merge(r.memory_refs, memory_refs or [])
            r.metrics = {**r.metrics, **(metrics or {})}
            if {"low": 0, "medium": 1, "high": 2}[severity] > {"low": 0, "medium": 1, "high": 2}[r.severity]:
                r.severity = severity
            if r.status in ("resolved", "dismissed"):   # the problem came back (history keeps the resolution)
                r.status = "open"
            if ttl_s:
                r.expires_at = now + ttl_s
        updated = self.s.update("M2", stored.id, add_occurrence, actor=by, reason="occurred again")
        self.s.emit("experience", id=stored.id, kind=kind, signature=signature, severity=updated.severity,
                    occurrences=updated.occurrences)
        return updated, False

    def add_explanation(self, rec_id: str, explanation: str, by: str) -> ExperienceRecord | None:
        return self.s.update("M2", rec_id, lambda r: setattr(
            r, "candidate_explanations", _merge(r.candidate_explanations, [explanation])), by, "candidate explanation")

    def confirm_cause(self, rec_id: str, cause: str, by: str, method: str) -> ExperienceRecord | None:
        """Only for causes established by an explicit rule or a human (`method` says which)."""
        def change(r: ExperienceRecord):
            r.confirmed_cause, r.status = cause, "confirmed"
            r.metrics = {**r.metrics, "cause_confirmed_by": method}
        return self.s.update("M2", rec_id, change, by, f"cause confirmed ({method})")

    def record_action(self, rec_id: str, action: dict, by: str) -> ExperienceRecord | None:
        """An attempted action (e.g. a guidance published in response) and, later, its outcome."""
        return self.s.update("M2", rec_id, lambda r: r.actions_taken.append({**action, "t": time.time()}),
                             by, "action attempted")

    def set_status(self, rec_id: str, status: str, by: str, reason: str) -> ExperienceRecord | None:
        return self.s.update("M2", rec_id, lambda r: setattr(r, "status", status), by, reason)

    def recovery_episode(self, incident_id: str, summary: str, by: str, provenance: list[dict] | None = None,
                         agents: list[str] | None = None) -> ExperienceRecord:
        """A recovery that happened (e.g. a retry that succeeded after a transient failure)."""
        rec, _ = self.report("recovery_episode", summary, signature_of("recovery", incident_id, time.time()),
                             by=by, memory_refs=[f"M2:{incident_id}"], provenance=provenance, agents=agents)
        return rec

    def open(self, kind: str | None = None, agent: str | None = None, limit: int = 50) -> list[ExperienceRecord]:
        filters = {"status": ["open", "investigating", "confirmed"]}
        if kind:
            filters["kind"] = kind
        if agent:
            filters["agents"] = agent
        return self.s.find("M2", limit=limit, **filters)
