"""
The shared persistent memory in Redis (already part of GAIED's deployment). One keyspace per namespace,
`pgmem:{namespace}:...`; the four categories are a field of the entry and an index, not separate databases.

  e:{id}             an entry (JSON)
  cat:{C}            sorted set of the ids of category C, by creation time
  key:{C}:{key}      ids of category C with that exact key (molecule, agent, rule key)
  edge:..., out:..., in:..., edgekey, edges, entities      the graph of relations (pgmem/graph.py)
  verification       verdicts of statements and diagnoses, by entry id (pgmem/verification.py)
  emb                embeddings of entries, by entry id (semantic retrieval)
  g:{id}, gq         guidance and its index; gseen:{agent}: guidance already delivered to that agent
  events             stream of memory events, read by the analysis agent (consumer group, acknowledged)
"""
from __future__ import annotations

import json
from contextlib import contextmanager
import redis

from .domain import DOMAIN
from .graph import GraphStore, node_type
from .schema import CATEGORIES, Entry, Guidance, molecule_node, motif_node, now


def normalize(text: str) -> str:
    return " ".join((text or "").lower().split()).strip(" .")


class MemoryStore:
    def __init__(self, client: redis.Redis, namespace: str = "default") -> None:
        self.r = client
        self.ns = f"pgmem:{namespace}"
        self.graph = GraphStore(client, self.ns, self.exists)

    @classmethod
    def connect(cls, host: str = "localhost", port: int = 6379, namespace: str = "default") -> "MemoryStore":
        return cls(redis.Redis(host=host, port=port, decode_responses=True), namespace)

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

    # ------------------------------------------------------------------ nodes
    def exists(self, node: str) -> bool:
        t = node_type(node)
        if t in ("m1", "m2", "m3", "m4"):
            return bool(self.r.exists(self._k("e", node)))
        if t == "g":
            return bool(self.r.exists(self._k("g", node)))
        return self.graph.is_entity(node)

    def molecule(self, smiles: str) -> str:
        """The node of a molecule, created with its motifs (HAS_MOTIF) the first time."""
        node = molecule_node(smiles)
        if not self.graph.is_entity(node):
            self.graph.add_entity(node)
            for m in sorted(DOMAIN.motifs(node[4:])):
                self.graph.add_entity(motif_node(m))
                self.graph.add(node, motif_node(m), "HAS_MOTIF", "domain")
        return node

    def motif(self, name: str) -> str:
        node = motif_node(name)
        self.graph.add_entity(node)
        return node

    def link(self, source: str, target: str, relation: str, by: str, origin: str = "deterministic",
             **props) -> tuple[str, bool]:
        return self.graph.add(source, target, relation, by, origin, **props)

    # ------------------------------------------------------------------ entries
    def _save(self, e: Entry) -> None:
        pipe = self.r.pipeline()
        pipe.set(self._k("e", e.id), json.dumps(e.to_dict()))
        pipe.zadd(self._k("cat", e.category), {e.id: e.created_at})
        if e.key:
            pipe.sadd(self._k("key", e.category, e.key), e.id)
        pipe.execute()

    def add(self, e: Entry) -> tuple[str, bool]:
        """Admit an entry; returns (id, merged). A duplicate is merged into the existing entry: its Flowcept
        references are appended and the merge is written in the history; nothing else of the entry changes."""
        with self._lock(f"{e.category}:{e.key or normalize(e.text)[:80]}"):
            same = self._duplicate_of(e)
            if same is not None:
                added = [p for p in e.prov_refs if p not in same.prov_refs]
                same.prov_refs.extend(added)
                same.history.append({"t": now(), "by": e.author, "change": "merged duplicate",
                                     "prov_refs_added": added})
                self._save(same)
                return same.id, True
            self._save(e)
        return e.id, False

    def _duplicate_of(self, e: Entry) -> Entry | None:
        """M1: same molecule, quantity, value and conditions (a repeated measurement). M3: the same statement.
        M4: the same key (one procedure per failure class and motif). M2: never (each experience happened once)."""
        if e.category == "M2":
            return None
        cands = self.by_key(e.category, e.key) if e.key else self.recent(e.category, limit=500)
        for c in cands:
            if e.category == "M1":
                if all(c.data.get(k) == e.data.get(k) for k in ("quantity", "value", "conditions")):
                    return c
            elif e.category == "M4" or normalize(c.text) == normalize(e.text):
                return c
        return None

    def get(self, entry_id: str) -> Entry | None:
        raw = self.r.get(self._k("e", entry_id))
        return Entry.from_dict(json.loads(raw)) if raw else None

    def by_key(self, category: str, key: str) -> list[Entry]:
        ids = self.r.smembers(self._k("key", category, key))
        return sorted((e for e in (self.get(i) for i in ids) if e), key=lambda e: -e.created_at)

    def recent(self, category: str, limit: int = 50) -> list[Entry]:
        ids = self.r.zrevrange(self._k("cat", category), 0, max(0, limit - 1))
        return [e for e in (self.get(i) for i in ids) if e]

    def all(self, category: str) -> list[Entry]:
        return [e for e in (self.get(i) for i in self.r.zrange(self._k("cat", category), 0, -1)) if e]

    def count(self, category: str) -> int:
        return int(self.r.zcard(self._k("cat", category)))

    def molecules_of(self, entry_id: str) -> list[str]:
        """Canonical SMILES of the molecules an entry is ABOUT."""
        return [e["target"][4:] for e in self.graph.outgoing(entry_id, ["ABOUT"]) if e["target"].startswith("mol:")]

    def motifs_of(self, entry_id: str) -> list[str]:
        return [e["target"][6:] for e in self.graph.outgoing(entry_id, ["ABOUT"]) if e["target"].startswith("motif:")]

    # ------------------------------------------------------------------ verdicts (pgmem/verification.py)
    def get_verification(self, entry_id: str) -> dict | None:
        raw = self.r.hget(self._k("verification"), entry_id)
        return json.loads(raw) if raw else None

    def set_verification(self, entry_id: str, record: dict) -> None:
        self.r.hset(self._k("verification"), entry_id, json.dumps(record, default=str))

    def all_verifications(self) -> dict[str, dict]:
        return {k: json.loads(v) for k, v in self.r.hgetall(self._k("verification")).items()}

    # ------------------------------------------------------------------ events (input of the analysis agent)
    def emit(self, event: str, payload: dict) -> str:
        return self.r.xadd(self._k("events"), {"event": event, "payload": json.dumps(payload), "t": str(now())})

    # ------------------------------------------------------------------ guidance
    def add_guidance(self, g: Guidance) -> str:
        pipe = self.r.pipeline()
        pipe.set(self._k("g", g.id), json.dumps(g.to_dict()))
        pipe.zadd(self._k("gq"), {g.id: g.created_at})
        pipe.execute()
        return g.id

    def get_guidance(self, gid: str) -> Guidance | None:
        raw = self.r.get(self._k("g", gid))
        return Guidance.from_dict(json.loads(raw)) if raw else None

    def all_guidance(self) -> list[Guidance]:
        return [g for g in (self.get_guidance(i) for i in self.r.zrange(self._k("gq"), 0, -1)) if g]

    def pending_guidance(self, agent: str) -> list[Guidance]:
        seen = self.r.smembers(self._k("gseen", agent))
        return [g for g in self.all_guidance() if g.id not in seen and g.target in ("*", agent)]

    def mark_delivered(self, gid: str, agent: str, node: str) -> None:
        with self._lock(f"g:{gid}"):
            g = self.get_guidance(gid)
            if g is None:
                return
            g.delivered.append({"agent": agent, "node": node, "t": now()})
            self.r.set(self._k("g", gid), json.dumps(g.to_dict()))
        self.r.sadd(self._k("gseen", agent), gid)
        self.emit("guidance_delivered", {"guidance": gid, "agent": agent, "node": node})

    # ------------------------------------------------------------------ maintenance
    def export(self) -> dict:
        return {"namespace": self.ns,
                "entries": {c: [e.to_dict() for e in self.all(c)] for c in CATEGORIES},
                "edges": self.graph.all(),
                "guidance": [g.to_dict() for g in self.all_guidance()],
                "verification": self.all_verifications(),
                "analyzer_stats": self.r.hgetall(self._k("analyzer", "stats"))}

    def reset(self) -> int:
        keys = list(self.r.scan_iter(f"{self.ns}:*"))
        return self.r.delete(*keys) if keys else 0

