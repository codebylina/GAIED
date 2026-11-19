"""
Shared persistent memory in Redis (already part of GAIED's deployment, persisted with AOF).

One keyspace per namespace, `pgmem:{namespace}:...`, the same for every campaign so that later campaigns learn from
earlier ones; the four categories are logical (a field of the entry and an index), not separate databases.

  e:{id}             the entry (JSON)
  cat:{C}            sorted set of the ids of category C, by creation time
  key:{C}:{key}      set of ids of category C with that exact key (molecule, failure class...)
  events             stream of memory events, read by the analysis agent (durable, consumer group, acknowledged)
  g:{id}, gq         guidance records and their index
  gseen:{agent}      guidance ids already delivered to that agent
"""
from __future__ import annotations

import json
import re
from contextlib import contextmanager
from typing import Callable, Iterable

import redis

from .schema import Entry, Guidance, now

_WORD = re.compile(r"[A-Za-z0-9_\-\[\]()=#@+\\/.]+")


def normalize(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .")


def tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD.findall(text or "") if len(t) > 1}


class MemoryStore:
    def __init__(self, client: redis.Redis, namespace: str = "default") -> None:
        self.r = client
        self.ns = f"pgmem:{namespace}"

    @classmethod
    def connect(cls, host: str = "localhost", port: int = 6379, namespace: str = "default") -> "MemoryStore":
        return cls(redis.Redis(host=host, port=port, decode_responses=True), namespace)

    # ------------------------------------------------------------------ low level
    def _k(self, *parts: str) -> str:
        return ":".join([self.ns, *parts])

    @contextmanager
    def _lock(self, name: str):
        lock = self.r.lock(self._k("lock", name), timeout=10, blocking_timeout=10)
        lock.acquire()
        try:
            yield
        finally:
            try:
                lock.release()
            except redis.exceptions.LockError:
                pass

    def _save(self, e: Entry) -> None:
        pipe = self.r.pipeline()
        pipe.set(self._k("e", e.id), json.dumps(e.to_dict()))
        pipe.zadd(self._k("cat", e.category), {e.id: e.created_at})
        if e.key:
            pipe.sadd(self._k("key", e.category, e.key), e.id)
        pipe.execute()

    def get(self, entry_id: str) -> Entry | None:
        raw = self.r.get(self._k("e", entry_id))
        return Entry.from_dict(json.loads(raw)) if raw else None

    # ------------------------------------------------------------------ write
    def add(self, e: Entry, dedup: bool = True) -> tuple[str, bool]:
        """Admit an entry. Returns (id, merged): merged=True when an existing entry with the same content absorbed it
        (its references are appended to that entry and the merge is written in its history)."""
        with self._lock(f"{e.category}:{e.key or normalize(e.text)[:80]}"):
            same = self._duplicate_of(e) if dedup else None
            if same is not None:
                self._merge(same, e)
                self._save(same)
                self.emit("merged", {"entry": same.id, "category": same.category, "from": e.author})
                return same.id, True
            self._save(e)
        self.emit("added", {"entry": e.id, "category": e.category, "kind": e.kind, "key": e.key,
                            "author": e.author})
        return e.id, False

    def _duplicate_of(self, e: Entry) -> Entry | None:
        """Deduplication: same category, same key and same content. M2 entries are never merged (each experience
        happened once)."""
        if e.category == "M2":
            return None
        cands = self.by_key(e.category, e.key) if e.key else self.recent(e.category, limit=500)
        for c in cands:
            if e.category == "M1":
                if (c.data.get("quantity") == e.data.get("quantity") and c.data.get("value") == e.data.get("value")
                        and c.data.get("conditions") == e.data.get("conditions")):
                    return c
            elif normalize(c.text) == normalize(e.text) and c.kind == e.kind:
                return c
        return None

    def _merge(self, into: Entry, e: Entry) -> None:
        added_refs = [p for p in e.prov_refs if p not in into.prov_refs]
        into.prov_refs.extend(added_refs)
        into.refs.extend(r for r in e.refs if r not in into.refs)
        if into.category == "M3":
            for side in ("supporting_evidence", "contradicting_evidence"):
                for ev in e.data.get(side, []):
                    if ev not in into.data[side]:
                        into.data[side].append(ev)
        if into.category == "M4":
            for ep in e.data.get("supporting_episodes", []):
                if ep not in into.data["supporting_episodes"]:
                    into.data["supporting_episodes"].append(ep)
        into.history.append({"t": now(), "by": e.author, "change": "merged duplicate",
                             "prov_refs_added": added_refs, "duplicate_id": e.id})

    def update(self, entry_id: str, by: str, change: str, fn: Callable[[Entry], None]) -> Entry:
        """Apply `fn` to the entry and record the change in its history (nothing is overwritten silently)."""
        with self._lock(f"id:{entry_id}"):
            e = self.get(entry_id)
            if e is None:
                raise KeyError(entry_id)
            before = json.dumps(e.data, sort_keys=True)
            fn(e)
            e.history.append({"t": now(), "by": by, "change": change,
                              "data_before": json.loads(before) if before != json.dumps(e.data, sort_keys=True)
                              else None})
            self._save(e)
        self.emit("updated", {"entry": entry_id, "category": e.category, "change": change, "by": by})
        return e

    def add_evidence(self, m3_id: str, side: str, evidence: dict, by: str, note: str = "") -> Entry:
        """Contradiction handling: new evidence for or against a statement is appended; the statement is kept."""
        if side not in ("supporting_evidence", "contradicting_evidence"):
            raise ValueError(side)

        def fn(e: Entry) -> None:
            ev = {**evidence, "added_by": by, "note": note, "t": now()}
            e.data[side].append(ev)
            if evidence.get("entry") and evidence["entry"] not in e.refs:
                e.refs.append(evidence["entry"])
        return self.update(m3_id, by, f"{side} added", fn)

    def add_note(self, entry_id: str, by: str, note: str, refs: Iterable[str] = ()) -> Entry:
        """Attach a note (e.g. a data-quality diagnosis) without changing the entry's content."""
        def fn(e: Entry) -> None:
            e.data.setdefault("notes", []).append({"t": now(), "by": by, "note": note, "refs": list(refs)})
        return self.update(entry_id, by, "note added", fn)

    # ------------------------------------------------------------------ read
    def by_key(self, category: str, key: str) -> list[Entry]:
        ids = self.r.smembers(self._k("key", category, key))
        out = [e for e in (self.get(i) for i in ids) if e]
        return sorted(out, key=lambda e: -e.created_at)

    def recent(self, category: str, limit: int = 50) -> list[Entry]:
        ids = self.r.zrevrange(self._k("cat", category), 0, max(0, limit - 1))
        return [e for e in (self.get(i) for i in ids) if e]

    def all(self, category: str) -> list[Entry]:
        ids = self.r.zrange(self._k("cat", category), 0, -1)
        return [e for e in (self.get(i) for i in ids) if e]

    def search(self, category: str, query: str, limit: int = 5, scan: int = 500) -> list[Entry]:
        """Keyword retrieval over the most recent entries of a category (ties: newest first)."""
        q = tokens(query)
        if not q:
            return self.recent(category, limit)
        scored = []
        for e in self.recent(category, scan):
            s = len(q & tokens(f"{e.text} {e.key} {json.dumps(e.data)}"))
            if s:
                scored.append((s, e.created_at, e))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [e for _, _, e in scored[:limit]]

    def citing(self, entry_id: str, categories: Iterable[str] = ("M3", "M4")) -> list[Entry]:
        """Entries that reference this one directly (used to find knowledge that relies on a given result)."""
        return [e for c in categories for e in self.all(c) if entry_id in e.refs]

    def count(self, category: str) -> int:
        return int(self.r.zcard(self._k("cat", category)))

    # ------------------------------------------------------------------ events (input of the analysis agent)
    def emit(self, event: str, payload: dict) -> str:
        return self.r.xadd(self._k("events"), {"event": event, "payload": json.dumps(payload), "t": str(now())})

    # ------------------------------------------------------------------ guidance
    def add_guidance(self, g: Guidance) -> str:
        pipe = self.r.pipeline()
        pipe.set(self._k("g", g.id), json.dumps(g.to_dict()))
        pipe.zadd(self._k("gq"), {g.id: g.created_at})
        pipe.execute()
        self.emit("guidance_created", {"guidance": g.id, "target": g.target})
        return g.id

    def get_guidance(self, gid: str) -> Guidance | None:
        raw = self.r.get(self._k("g", gid))
        return Guidance.from_dict(json.loads(raw)) if raw else None

    def all_guidance(self) -> list[Guidance]:
        return [g for g in (self.get_guidance(i) for i in self.r.zrange(self._k("gq"), 0, -1)) if g]

    def update_guidance(self, gid: str, fn: Callable[[Guidance], None]) -> Guidance:
        with self._lock(f"g:{gid}"):
            g = self.get_guidance(gid)
            if g is None:
                raise KeyError(gid)
            fn(g)
            self.r.set(self._k("g", gid), json.dumps(g.to_dict()))
        return g

    def pending_guidance(self, agent: str) -> list[Guidance]:
        seen = self.r.smembers(self._k("gseen", agent))
        out = []
        for gid in self.r.zrange(self._k("gq"), 0, -1):
            if gid in seen:
                continue
            g = self.get_guidance(gid)
            if g and g.target in ("*", agent):
                out.append(g)
        return out

    def mark_delivered(self, gid: str, agent: str, node: str) -> None:
        self.r.sadd(self._k("gseen", agent), gid)
        self.update_guidance(gid, lambda g: g.delivered.append({"agent": agent, "node": node, "t": now()}))
        self.emit("guidance_delivered", {"guidance": gid, "agent": agent, "node": node})

    # ------------------------------------------------------------------ maintenance
    def export(self) -> dict:
        return {"namespace": self.ns,
                "entries": {c: [e.to_dict() for e in self.all(c)] for c in ("M1", "M2", "M3", "M4")},
                "guidance": [g.to_dict() for g in self.all_guidance()],
                "analyzer_stats": self.r.hgetall(self._k("analyzer", "stats"))}

    def reset(self) -> int:
        keys = list(self.r.scan_iter(f"{self.ns}:*"))
        return self.r.delete(*keys) if keys else 0
