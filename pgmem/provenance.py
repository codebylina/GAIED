"""
Adapter to Flowcept, the single source of truth for execution provenance. Nothing read here is stored in memory:
memory entries keep ProvRefs, and the analysis agent resolves them through this adapter when it needs details.

Sources (all verified against the Flowcept used by GAIED, see tests):
  JsonlProvenance    the buffer files Flowcept dumps at the end of a run (GAIED's default, offline mode)
  StreamProvenance   the Redis pub/sub channel Flowcept publishes to in online mode (mq.enabled, db_flush_mode online);
                     records are kept in a bounded in-process window only, for traversal, never persisted
  DbProvenance       Flowcept's DBAPI (MongoDB), for catch-up and post-execution analysis in online mode
                     (NOT verified here: no MongoDB in the test environment)
"""
from __future__ import annotations

import ast
import glob
import json
import os
import threading
from collections import OrderedDict
from typing import Iterable

RELEVANT_SUBTYPES = {"tool_call", "parsl_task", "llm_call", "langgraph_node", "academy_lifecycle"}


def as_dict(v) -> dict:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            d = ast.literal_eval(v)
            return d if isinstance(d, dict) else {}
        except (ValueError, SyntaxError):
            try:
                d = json.loads(v)
                return d if isinstance(d, dict) else {}
            except ValueError:
                return {}
    return {}


def as_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def smiles_of(rec: dict) -> str | None:
    """The molecule a computation task used (parsl_task: used.smiles; tool_call: used.input)."""
    used = as_dict(rec.get("used"))
    if "smiles" in used:
        return str(used["smiles"])
    inner = as_dict(used.get("input"))
    return str(inner["smiles"]) if "smiles" in inner else None


def is_computation(rec: dict) -> bool:
    return rec.get("activity_id") == "compute_ionization_energy" and rec.get("subtype") in ("tool_call", "parsl_task")


def is_relevant(rec: dict) -> bool:
    """Event filtering: only the records the analysis agent uses."""
    return rec.get("type") in (None, "task") and rec.get("subtype") in RELEVANT_SUBTYPES


class ProvenanceIndex:
    """Bounded window of provenance records, indexed by task id, with the traversal the analysis agent needs."""

    def __init__(self, max_records: int = 20000) -> None:
        self.by_id: OrderedDict[str, dict] = OrderedDict()
        self.max = max_records
        self.lock = threading.Lock()

    def add(self, rec: dict) -> bool:
        tid = rec.get("task_id")
        if not tid:
            return False
        with self.lock:
            new = tid not in self.by_id
            self.by_id[tid] = {**self.by_id.get(tid, {}), **rec}
            while len(self.by_id) > self.max:
                self.by_id.popitem(last=False)
        return new

    def get(self, task_id: str) -> dict | None:
        return self.by_id.get(task_id)

    def records(self) -> list[dict]:
        with self.lock:
            return list(self.by_id.values())

    def computations(self) -> list[dict]:
        return [r for r in self.records() if is_computation(r)]

    def downstream(self, task_id: str) -> list[dict]:
        """Tasks that may have used the output of `task_id`: its children (parent_task_id), and, because GAIED keeps
        results in the agent's LangGraph state, the later LangGraph nodes of the agent's graph workflow (found
        through the tool_call record of the same molecule)."""
        src = self.get(task_id)
        if src is None:
            return []
        recs = self.records()
        out = {r["task_id"]: r for r in recs if r.get("parent_task_id") == task_id}
        smi, end = smiles_of(src), as_float(src.get("ended_at")) or 0.0
        graph_wfs = {r.get("workflow_id") for r in recs if r.get("subtype") == "tool_call" and smiles_of(r) == smi
                     and r.get("campaign_id") == src.get("campaign_id")}
        if src.get("subtype") == "tool_call":
            graph_wfs.add(src.get("workflow_id"))
        for r in recs:
            if (r.get("subtype") == "langgraph_node" and r.get("workflow_id") in graph_wfs
                    and (as_float(r.get("started_at")) or 0.0) >= end):
                out[r["task_id"]] = r
        return sorted(out.values(), key=lambda r: as_float(r.get("started_at")) or 0.0)


class JsonlProvenance:
    """Flowcept buffer files. With `campaign_id`, only the files of that campaign."""

    def __init__(self, pattern: str = "output/flowcept_buffer_*.jsonl", campaign_id: str | None = None,
                 skip_existing: bool = False) -> None:
        self.pattern, self.campaign_id = pattern, campaign_id
        self.index = ProvenanceIndex()
        self._offsets: dict[str, int] = {}
        if skip_existing:            # online use: files of earlier campaigns are not new events
            for path in glob.glob(pattern):
                self._offsets[path] = os.path.getsize(path)

    def poll(self) -> list[dict]:
        """New relevant records since the last call (files are re-read from where they were left)."""
        new = []
        for path in sorted(glob.glob(self.pattern)):
            start = self._offsets.get(path, 0)
            try:
                with open(path, encoding="utf-8") as fh:
                    fh.seek(start)
                    for line in fh:
                        if not line.strip():
                            continue
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue
                        if self.campaign_id and rec.get("campaign_id") not in (None, self.campaign_id):
                            continue
                        if is_relevant(rec) and self.index.add(rec):
                            new.append(rec)
                    self._offsets[path] = fh.tell()
            except OSError:
                continue
        return new


class StreamProvenance:
    """Online: listens to Flowcept's Redis channel in a background thread; poll() returns what arrived."""

    def __init__(self, host: str = "localhost", port: int = 6379, channel: str = "interception") -> None:
        import msgpack
        import redis
        self._msgpack = msgpack
        self.index = ProvenanceIndex()
        self._pending: list[dict] = []
        self._lock = threading.Lock()
        self._r = redis.Redis(host=host, port=port)
        self._ps = self._r.pubsub(ignore_subscribe_messages=True)
        self._ps.psubscribe(channel)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                m = self._ps.get_message(timeout=1.0)
            except Exception:  # noqa: BLE001 - connection lost: keep trying, the DB catch-up covers the gap
                self._stop.wait(2.0)
                continue
            if not m or m.get("type") != "pmessage":
                continue
            try:
                rec = self._msgpack.loads(m["data"])
            except Exception:  # noqa: BLE001
                continue
            if isinstance(rec, dict) and is_relevant(rec) and self.index.add(rec):
                with self._lock:
                    self._pending.append(rec)

    def poll(self) -> list[dict]:
        with self._lock:
            out, self._pending = self._pending, []
        return out

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)
        self._ps.close()


class DbProvenance:
    """Flowcept's database (MongoDB in GAIED's compose file), through Flowcept's own DBAPI. Not verified here."""

    def __init__(self, campaign_id: str | None = None) -> None:
        from flowcept.flowcept_api.db_api import DBAPI
        self.db = DBAPI()
        self.campaign_id = campaign_id
        self.index = ProvenanceIndex()
        self._seen_end = 0.0

    def poll(self) -> list[dict]:
        flt = {"ended_at": {"$gte": self._seen_end}}
        if self.campaign_id:
            flt["campaign_id"] = self.campaign_id
        new = []
        for rec in self.db.task_query(filter=flt) or []:
            self._seen_end = max(self._seen_end, as_float(rec.get("ended_at")) or 0.0)
            if is_relevant(rec) and self.index.add(rec):
                new.append(rec)
        return new


def flowcept_online(settings_path: str | None = None) -> bool:
    """True when the Flowcept settings in use publish online (mq enabled and db_flush_mode online)."""
    path = settings_path or os.environ.get("FLOWCEPT_SETTINGS_PATH")
    if not path or not os.path.exists(path):
        return False
    import yaml
    s = yaml.safe_load(open(path)) or {}
    return bool((s.get("mq") or {}).get("enabled")) and (s.get("project") or {}).get("db_flush_mode") == "online"


def make_source(kind: str, redis_host: str, redis_port: int, jsonl_glob: str, campaign_id: str | None,
                online: bool = True):
    """auto: the online stream when Flowcept publishes online, else the JSONL files."""
    if kind == "none":
        return None
    if kind == "stream" or (kind == "auto" and flowcept_online()):
        s = os.environ.get("FLOWCEPT_SETTINGS_PATH")
        host, port = redis_host, redis_port
        if s and os.path.exists(s):
            import yaml
            mq = (yaml.safe_load(open(s)) or {}).get("mq") or {}
            host, port = mq.get("host", host), int(mq.get("port", port))
        return StreamProvenance(host, port)
    if kind == "db":
        return DbProvenance(campaign_id)
    return JsonlProvenance(jsonl_glob, campaign_id, skip_existing=online)


def iter_records(paths: Iterable[str]) -> list[dict]:
    out = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out
