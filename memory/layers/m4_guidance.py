"""
M4 — Operational Guidance Memory: what agents should do differently.

Strategies, recommendations, instructions and recovery strategies. Each item states its applicability
(nodes, condition), what triggered it, structured actions (checkable when possible), limitations,
evidence (memory ids, Flowcept task ids), target agents ("all" = the whole campaign) and a validation
status: proposed -> active -> validated (shown effective) / ineffective -> retired.

Delivery is durable: publishing an active item appends it to the targets' inbox streams
(M4:inbox:{agent} or M4:inbox:all); each agent reads them with a cursor stored in Redis, so a restarted
agent does not miss guidance. A Pub/Sub notification is sent as well (best effort, never relied upon).
For every (item, agent) the memory tracks: delivered (shown in a prompt), acknowledged (the agent's
LLM call that saw it completed), applied / violated (the agent's next actions do / do not comply with
the structured actions), so the analysis service can measure effectiveness.
"""
from __future__ import annotations

import hashlib
import json
import operator
import time

from ..domain import Domain
from ..schemas import GuidanceRecord
from ..store import Store

OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq}
LIVE = ("active", "validated")


class GuidanceMemory:
    def __init__(self, store: Store, domain: Domain) -> None:
        self.s, self.domain = store, domain

    @staticmethod
    def key_of(kind: str, actions: list[dict], targets: list[str], condition: str, text: str) -> str:
        blob = {"kind": kind, "actions": actions, "targets": sorted(targets), "condition": condition}
        if not actions or all(a.get("type") == "free_text" for a in actions):
            blob["text"] = " ".join(text.lower().split())
        return hashlib.sha1(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:16]

    # ------------------------------------------------------------------ publish
    def publish(self, text: str, *, by: str, kind: str = "instruction", actions: list[dict] | None = None,
                targets: list[str] | None = None, condition: str = "", nodes: list[str] | None = None,
                trigger: dict | None = None, memory_refs: list[str] | None = None,
                provenance: list[dict] | None = None, limitations: str = "", status: str = "active",
                priority: int = 1, ttl_s: float | None = None, source: str = "analysis",
                baseline: dict | None = None) -> tuple[GuidanceRecord, bool]:
        targets = targets or ["all"]
        actions = actions or [{"type": "free_text"}]
        now = time.time()
        rec = GuidanceRecord(kind=kind, text=text, actions=actions, targets=targets, condition=condition,
                             applicability={"nodes": nodes or []}, trigger=trigger or {}, limitations=limitations,
                             memory_refs=memory_refs or [], provenance=provenance or [], status=status,
                             priority=priority, source=source, baseline=baseline or {}, created_by=by,
                             expires_at=now + ttl_s if ttl_s else None,
                             key=self.key_of(kind, actions, targets, condition, text))
        stored, created = self.s.add(rec, actor=by, reason=f"published by {source}")
        if not created:
            if stored.status in ("retired", "ineffective"):
                return stored, False          # never re-activated silently: the history says why it stopped

            def refresh(r: GuidanceRecord):
                r.memory_refs = list(dict.fromkeys(r.memory_refs + (memory_refs or [])))[-50:]
                r.provenance = (r.provenance + (provenance or []))[-50:]
                r.expires_at = now + ttl_s if ttl_s else r.expires_at
                r.trigger = {**r.trigger, "reissued": r.trigger.get("reissued", 0) + 1}
            stored = self.s.update("M4", stored.id, refresh, by, "issued again (evidence merged)")
            return stored, False
        if stored.status in LIVE:
            self._deliver_to_inboxes(stored)
        self.s.emit("guidance", id=stored.id, status=stored.status, targets=targets, kind=kind)
        return stored, True

    def _deliver_to_inboxes(self, g: GuidanceRecord) -> None:
        pipe = self.s.r.pipeline()
        for t in g.targets:
            pipe.xadd(self.s.k("M4", "inbox", t), {"id": g.id, "t": time.time()}, maxlen=10_000, approximate=True)
        pipe.publish(self.s.k("notify", "guidance"), json.dumps({"id": g.id, "targets": g.targets}))
        pipe.execute()

    def activate(self, rec_id: str, by: str, reason: str, baseline: dict | None = None) -> GuidanceRecord | None:
        def change(r: GuidanceRecord):
            r.status = "active"
            if baseline:
                r.baseline = baseline
        rec = self.s.update("M4", rec_id, change, by, f"activated: {reason}")
        if rec:
            self._deliver_to_inboxes(rec)
        return rec

    def set_status(self, rec_id: str, status: str, by: str, reason: str,
                   effectiveness: dict | None = None) -> GuidanceRecord | None:
        def change(r: GuidanceRecord):
            r.status = status
            if effectiveness is not None:
                r.effectiveness = {**r.effectiveness, **effectiveness}
        rec = self.s.update("M4", rec_id, change, by, reason)
        self.s.emit("guidance_status", id=rec_id, status=status, reason=reason)
        return rec

    # ------------------------------------------------------------------ retrieval during execution
    def for_agent(self, agent: str, node: str | None = None, condition: str | None = None) -> list[GuidanceRecord]:
        now, out = time.time(), []
        for g in self.s.find("M4", status=list(LIVE), targets=[agent, "all"]):
            if g.expires_at and g.expires_at < now:
                continue
            if node and g.applicability.get("nodes") and node not in g.applicability["nodes"]:
                continue
            if condition and g.condition and g.condition != condition:
                continue
            out.append(g)
        out.sort(key=lambda g: (-g.priority, -g.updated_at))
        return out

    def poll_inbox(self, agent: str) -> list[str]:
        """New guidance ids for this agent since its last read (durable cursor)."""
        cursor_key = self.s.k("M4", "cursor", agent)
        cursors = self.s.r.hgetall(cursor_key)
        names = {self.s.k("M4", "inbox", agent): agent, self.s.k("M4", "inbox", "all"): "all"}
        streams = {key: cursors.get(name, "0-0") for key, name in names.items()}
        new, last = [], {}
        for stream, entries in self.s.r.xread(streams, count=100) or []:
            for eid, fields in entries:
                new.append(fields["id"])
                last[names[stream]] = eid
        if last:
            self.s.r.hset(cursor_key, mapping=last)
        return list(dict.fromkeys(new))

    # ------------------------------------------------------------------ delivery tracking
    def mark(self, rec_id: str, agent: str, event: str, detail: dict | None = None) -> None:
        """event: delivered | acknowledged | applied | violated."""
        key = self.s.k("M4", "delivery", rec_id)
        with self.s.r.pipeline() as pipe:
            for _ in range(10):
                try:
                    pipe.watch(key)
                    cur = json.loads(pipe.hget(key, agent) or "{}")
                    cur.setdefault(f"{event}_at", time.time())
                    cur[f"{event}_n"] = cur.get(f"{event}_n", 0) + 1
                    cur[f"last_{event}_at"] = time.time()
                    if detail:
                        cur[f"last_{event}"] = detail
                    pipe.multi()
                    pipe.hset(key, agent, json.dumps(cur, default=str))
                    pipe.execute()
                    break
                except Exception as exc:  # WatchError: retry
                    import redis
                    if not isinstance(exc, redis.WatchError):
                        raise
        self.s.use(rec_id, event)

    def delivery(self, rec_id: str) -> dict[str, dict]:
        return {a: json.loads(v) for a, v in self.s.r.hgetall(self.s.k("M4", "delivery", rec_id)).items()}

    # ------------------------------------------------------------------ compliance with structured actions
    def _in_region(self, entity: str, region: dict) -> bool:
        f = self.domain.features(entity)
        return region.get("feature") in f and OPS[region.get("op", ">")](f[region["feature"]], float(region.get("value", 0)))

    def compliance(self, g: GuidanceRecord, proposals: list[str], history: list[str] | None = None,
                   known: set[str] | None = None) -> tuple[bool | None, dict]:
        """Do the agent's proposals (canonical entities) comply with the item's structured actions?
        None when no action of the item is checkable."""
        verdicts, details = [], {}
        for a in g.actions:
            t = a["type"]
            if t == "avoid_entities":
                bad = [p for p in proposals if p in set(a.get("entities", []))]
                verdicts.append(not bad)
                details["avoided_violations"] = bad
            elif t == "avoid_region":
                bad = [p for p in proposals if self._in_region(p, a)]
                verdicts.append(not bad)
                details["region_violations"] = bad
            elif t == "prefer_region" and proposals:
                share = sum(self._in_region(p, a) for p in proposals) / len(proposals)
                verdicts.append(share >= float(a.get("min_share", 0.5)))
                details["share_in_region"] = round(share, 3)
            elif t == "diversify" and proposals and history:
                nn = [max(self.domain.similarity(p, h) for h in history) for p in proposals]
                sim = sum(nn) / len(nn)
                verdicts.append(sim <= float(a.get("max_similarity", 0.6)))
                details["mean_nearest_similarity"] = round(sim, 3)
            elif t == "reuse_results" and proposals and known is not None:
                share = sum(p in known for p in proposals) / len(proposals)
                verdicts.append(share <= float(a.get("max_known_share", 0.34)))
                details["share_already_known"] = round(share, 3)
        if not verdicts:
            return None, details
        return all(verdicts), details
