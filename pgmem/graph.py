"""
The relations of the shared memory: typed, directed edges between nodes, stored in Redis next to the entries. This
is the only place where a relation is stored (an entry never holds the id of another node).

Nodes are entry ids (m1:..., m2:..., m3:..., m4:...), guidance ids (g:...) and domain entities (mol:<canonical
SMILES>, motif:<name>). Every edge records who created it and its origin: "deterministic" (computed by code: RDKit,
Flowcept references, a rule, a numerical check) or "llm" (proposed by a language model: an agent citing its
evidence, an explanation). Only deterministic evidence counts in a verdict; LLM edges are shown as proposed.

Relations (source -> target), by family:
  evidence      SUPPORTED_BY    statement or diagnosis -> result or failure that supports it
                CONTRADICTED_BY statement -> result that contradicts it
  dependency    DIAGNOSED_FROM  diagnosis -> what it is based on
                DERIVED_FROM    procedure -> failures it was derived from
                JUSTIFIED_BY    procedure -> knowledge that justifies it
                APPLIES         guidance -> procedure or diagnosis it applies
  association   ABOUT           entry -> molecule or motif it is about
                HAS_MOTIF       molecule -> motif
                PART_OF         result or failure -> attempt (iteration) that produced it
                EXPLAINS        statement or diagnosis -> what it proposes to explain
                APPLIED_IN      guidance -> the attempt in which it was judged (applied, effective)

What rests on a node is reached by following IMPACT edges backwards: a statement rests on what supports it (not on
what contradicts it), a procedure on its failures and its justification, guidance on what it applies. Association
edges give context and are never propagated.

Redis keys (namespace pgmem:{ns}):
  edge:{id}       the edge (JSON)        out:{node} / in:{node}   ids of the edges leaving / entering a node
  edgekey         hash "source|type|target|origin" -> edge id (one edge per relation and origin: an author
                  may cite a result and a check may confirm it, two edges)
  edges           sorted set of all edge ids by creation time
  entities        set of the entity nodes (molecules, motifs)
Creation and deletion run as Lua scripts: the edge, both indexes and the triple index change together or not at all.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Iterable

import redis

EVIDENCE = ("SUPPORTED_BY", "CONTRADICTED_BY")
DEPENDENCY = ("DIAGNOSED_FROM", "DERIVED_FROM", "JUSTIFIED_BY", "APPLIES")
ASSOCIATION = ("ABOUT", "HAS_MOTIF", "PART_OF", "EXPLAINS", "APPLIED_IN")
IMPACT = DEPENDENCY + ("SUPPORTED_BY",)
ORIGINS = ("deterministic", "llm")

# allowed node types (id prefixes) at each end of a relation
RELATIONS: dict[str, tuple[tuple, tuple]] = {
    "SUPPORTED_BY": (("m3", "m2"), ("m1", "m2")),
    "CONTRADICTED_BY": (("m3",), ("m1", "m2")),
    "DIAGNOSED_FROM": (("m2",), ("m1", "m2", "m3", "m4", "g")),
    "DERIVED_FROM": (("m4",), ("m2",)),
    "JUSTIFIED_BY": (("m4",), ("m3",)),
    "APPLIES": (("g",), ("m4", "m2")),
    "ABOUT": (("m1", "m2", "m3", "m4"), ("mol", "motif")),
    "HAS_MOTIF": (("mol",), ("motif",)),
    "PART_OF": (("m1", "m2"), ("m2",)),
    "EXPLAINS": (("m3", "m2"), ("m1", "m2", "m3")),
    "APPLIED_IN": (("g",), ("m2",)),
}

_ADD = """
local existing = redis.call('hget', KEYS[4], ARGV[1])
if existing then return existing end
redis.call('set', KEYS[1], ARGV[3])
redis.call('sadd', KEYS[2], ARGV[2])
redis.call('sadd', KEYS[3], ARGV[2])
redis.call('hset', KEYS[4], ARGV[1], ARGV[2])
redis.call('zadd', KEYS[5], ARGV[4], ARGV[2])
return ARGV[2]
"""
_REMOVE = """
if redis.call('del', KEYS[1]) == 0 then return 0 end
redis.call('srem', KEYS[2], ARGV[2])
redis.call('srem', KEYS[3], ARGV[2])
redis.call('hdel', KEYS[4], ARGV[1])
redis.call('zrem', KEYS[5], ARGV[2])
return 1
"""


def node_type(node: str) -> str:
    return node.split(":", 1)[0]


@dataclass
class Reach:
    """A node reached by a traversal, its distance and the edges of the path that led to it."""
    node: str
    depth: int
    path: list[dict] = field(default_factory=list)


class GraphStore:
    def __init__(self, client: redis.Redis, prefix: str, node_exists: Callable[[str], bool]) -> None:
        self.r, self.prefix, self.exists = client, prefix, node_exists
        self._add, self._remove = client.register_script(_ADD), client.register_script(_REMOVE)

    def _k(self, *parts: str) -> str:
        return ":".join([self.prefix, *parts])

    # ------------------------------------------------------------------ entities
    def add_entity(self, node: str) -> None:
        if node_type(node) not in ("mol", "motif"):
            raise ValueError(f"not an entity node: {node}")
        self.r.sadd(self._k("entities"), node)

    def is_entity(self, node: str) -> bool:
        return bool(self.r.sismember(self._k("entities"), node))

    # ------------------------------------------------------------------ write
    def add(self, source: str, target: str, relation: str, created_by: str, origin: str = "deterministic",
            **props) -> tuple[str, bool]:
        """Create an edge; returns (id, created). An edge with the same source, relation, target and origin already
        there is returned instead (created False). Unknown relations, wrong node types and missing nodes are refused."""
        if relation not in RELATIONS:
            raise ValueError(f"unknown relation {relation}")
        if origin not in ORIGINS:
            raise ValueError(f"origin must be one of {ORIGINS}")
        sources, targets = RELATIONS[relation]
        if node_type(source) not in sources or node_type(target) not in targets:
            raise ValueError(f"{relation} cannot link {source} to {target}")
        for n in (source, target):
            if not self.exists(n):
                raise KeyError(f"node {n} does not exist")
        eid, t = f"e:{uuid.uuid4().hex[:12]}", time.time()
        edge = {"id": eid, "source": source, "target": target, "type": relation, "created_at": t,
                "created_by": created_by, "origin": origin, "props": props}
        got = self._add(keys=[self._k("edge", eid), self._k("out", source), self._k("in", target),
                              self._k("edgekey"), self._k("edges")],
                        args=[f"{source}|{relation}|{target}|{origin}", eid, json.dumps(edge, default=str), t])
        return got, got == eid

    def remove(self, edge_id: str) -> bool:
        e = self.get(edge_id)
        if e is None:
            return False
        return bool(self._remove(keys=[self._k("edge", edge_id), self._k("out", e["source"]),
                                       self._k("in", e["target"]), self._k("edgekey"), self._k("edges")],
                                 args=[f"{e['source']}|{e['type']}|{e['target']}|{e['origin']}", edge_id]))

    # ------------------------------------------------------------------ read
    def get(self, edge_id: str) -> dict | None:
        raw = self.r.get(self._k("edge", edge_id))
        return json.loads(raw) if raw else None

    def find(self, source: str, relation: str, target: str, origin: str = "deterministic") -> dict | None:
        eid = self.r.hget(self._k("edgekey"), f"{source}|{relation}|{target}|{origin}")
        return self.get(eid) if eid else None

    def _edges(self, ids: Iterable[str], types: Iterable[str] | None) -> list[dict]:
        ids = list(ids)
        raws = self.r.mget([self._k("edge", i) for i in ids]) if ids else []
        out = [json.loads(x) for x in raws if x]
        if types is not None:
            types = set(types)
            out = [e for e in out if e["type"] in types]
        return sorted(out, key=lambda e: e["created_at"])

    def outgoing(self, node: str, types: Iterable[str] | None = None) -> list[dict]:
        return self._edges(self.r.smembers(self._k("out", node)), types)

    def incoming(self, node: str, types: Iterable[str] | None = None) -> list[dict]:
        return self._edges(self.r.smembers(self._k("in", node)), types)

    def neighbors(self, node: str, direction: str = "both", types: Iterable[str] | None = None) \
            -> list[tuple[dict, str]]:
        """(edge, other node) pairs; direction "out", "in" or "both"."""
        out = []
        if direction in ("out", "both"):
            out += [(e, e["target"]) for e in self.outgoing(node, types)]
        if direction in ("in", "both"):
            out += [(e, e["source"]) for e in self.incoming(node, types)]
        return out

    def traverse(self, start: str, direction: str = "both", types: Iterable[str] | None = None,
                 max_depth: int = 2, max_nodes: int = 50) -> list[Reach]:
        """Breadth-first traversal from `start` (excluded), bounded in depth and in number of nodes; each node is
        reached once, by a shortest path."""
        seen, frontier, out = {start}, [Reach(start, 0)], []
        while frontier and len(out) < max_nodes:
            nxt = []
            for r in frontier:
                if r.depth >= max_depth:
                    continue
                for edge, other in self.neighbors(r.node, direction, types):
                    if other in seen:
                        continue
                    seen.add(other)
                    reach = Reach(other, r.depth + 1, r.path + [edge])
                    out.append(reach)
                    nxt.append(reach)
                    if len(out) >= max_nodes:
                        return out
            frontier = nxt
        return out

    def dependents(self, node: str, max_depth: int = 3, max_nodes: int = 50) -> list[Reach]:
        """Nodes that rest on this one, directly or through others (IMPACT edges followed backwards)."""
        return self.traverse(node, "in", IMPACT, max_depth, max_nodes)

    def all(self) -> list[dict]:
        return self._edges(self.r.zrange(self._k("edges"), 0, -1), None)

    def count_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.all():
            counts[e["type"]] = counts.get(e["type"], 0) + 1
        return counts
