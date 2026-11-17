"""
Flowcept adapter: read execution provenance through Flowcept's own APIs. Flowcept stays the single
source of truth; the memory stores only references (task ids, workflow ids, campaign id).

Three access paths, all verified against the Flowcept fork used by GAIED:
  live      Flowcept MQ (Redis Pub/Sub, channel "interception"), consumed with Flowcept's
            `BaseConsumer` (needs mq.enabled). Low latency, NOT durable: messages published while the
            consumer is down are lost, hence:
  database  `Flowcept.db.task_query(filter, sort, limit)` on the tasks stored by Flowcept's
            DocumentInserter (MongoDB or LMDB enabled). Durable: used for catch-up after a restart, with
            a watermark on `registered_at` (insertion time).
  buffers   `Flowcept.read_buffer_file(path)` on the JSONL files written when interceptors stop
            (offline mode, the docker-local default). Only complete after the campaign.

`summary()` reduces a task to the few fields the analysis needs (ids, activity, status, times, entity,
outcome, agent); prompts and responses are not copied.
"""
from __future__ import annotations

import datetime as dt
import glob
import logging
import threading
from typing import Callable, Iterable

from .domain import Domain

log = logging.getLogger("memory.provenance")


def _ts(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, dt.datetime):
        return v.replace(tzinfo=v.tzinfo or dt.timezone.utc).timestamp()
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def summary(task: dict, domain: Domain | None = None) -> dict:
    """Compact view of a Flowcept task record (references and outcome, not the payload)."""
    meta = task.get("custom_metadata") if isinstance(task.get("custom_metadata"), dict) else {}
    out = {"task_id": task.get("task_id"), "activity_id": task.get("activity_id"), "subtype": task.get("subtype"),
           "type": task.get("type"), "status": task.get("status"), "workflow_id": task.get("workflow_id"),
           "campaign_id": task.get("campaign_id"), "parent_task_id": task.get("parent_task_id"),
           "started_at": _ts(task.get("started_at")), "ended_at": _ts(task.get("ended_at")),
           "registered_at": _ts(task.get("registered_at")), "agent": meta.get("agent")}
    if domain is not None and (exp := domain.experiment_of(task)):
        out.update(experiment=True, entity=exp["entity"], condition=exp["condition"], ok=exp["ok"],
                   value=exp["value"], error=(exp["error"] or "")[:300] or None)
    if out["subtype"] == "llm_call":
        gen = task.get("generated") if isinstance(task.get("generated"), dict) else {}
        out["n_tool_calls"] = len(gen.get("tool_calls") or [])
        out["elapsed_ms"] = meta.get("elapsed_ms")
    if out["subtype"] == "langgraph_node":
        out["node"] = meta.get("node_name") or task.get("activity_id")
    if out["subtype"] == "memory":
        out["memory"] = {"used": task.get("used"), "generated": task.get("generated")}
    return out


class FlowceptSource:
    def __init__(self, domain: Domain | None = None) -> None:
        self.domain = domain
        self._consumer = None

    # ------------------------------------------------------------------ database (durable)
    def db_available(self) -> bool:
        try:
            from flowcept import configs
            if not (getattr(configs, "MONGO_ENABLED", False) or getattr(configs, "LMDB_ENABLED", False)):
                return False
            from flowcept import Flowcept
            Flowcept.db.task_query(filter={"task_id": "__liveness__"}, limit=1)
            return True
        except Exception as exc:
            log.info("Flowcept database unavailable: %s", exc)
            return False

    def tasks_since(self, campaign_id: str, watermark: float | None, limit: int = 2000) -> list[dict]:
        """Tasks of a campaign inserted after the watermark (registered_at, seconds), oldest first."""
        from flowcept import Flowcept
        flt: dict = {"campaign_id": campaign_id}
        if watermark:
            flt["registered_at"] = {"$gt": dt.datetime.fromtimestamp(watermark, dt.timezone.utc).replace(tzinfo=None)}
        return Flowcept.db.task_query(filter=flt, sort=[("registered_at", 1)], limit=limit) or []

    def task(self, task_id: str) -> dict | None:
        from flowcept import Flowcept
        found = Flowcept.db.task_query(filter={"task_id": task_id}, limit=1)
        return found[0] if found else None

    def tasks(self, task_ids: Iterable[str]) -> list[dict]:
        from flowcept import Flowcept
        ids = list(task_ids)
        if not ids:
            return []
        return Flowcept.db.task_query(filter={"task_id": {"$in": ids}}) or []

    # ------------------------------------------------------------------ JSONL buffers (offline)
    def read_buffers(self, pattern: str = "output/flowcept_buffer_*.jsonl", campaign_id: str | None = None) -> list[dict]:
        from flowcept import Flowcept
        out = []
        for path in sorted(glob.glob(pattern)):
            try:
                recs = Flowcept.read_buffer_file(path, cleanup_files=False)
            except Exception as exc:
                log.warning("cannot read %s: %s", path, exc)
                continue
            out += [r for r in recs if not campaign_id or r.get("campaign_id") == campaign_id]
        return out

    # ------------------------------------------------------------------ live (Pub/Sub, not durable)
    def subscribe(self, handler: Callable[[dict], None]) -> bool:
        """Call handler(message) for every message published on the Flowcept MQ, in a thread.
        Returns False when the MQ is disabled or unreachable."""
        try:
            from flowcept import configs
            if not configs.MQ_ENABLED:
                return False
            from flowcept.flowceptor.consumers.base_consumer import BaseConsumer
        except Exception as exc:
            log.info("Flowcept MQ unavailable: %s", exc)
            return False

        class _Live(BaseConsumer):
            def message_handler(self, msg):
                try:
                    handler(msg)
                except Exception as exc:  # never let one message stop the consumer
                    log.warning("live provenance handler failed: %s", exc)
                return True
        try:
            self._consumer = _Live().start(daemon=True)
            return True
        except Exception as exc:
            log.warning("cannot subscribe to the Flowcept MQ: %s", exc)
            return False

    def stop(self) -> None:
        if self._consumer is not None:
            try:
                self._consumer.stop_consumption()
            except Exception:
                pass


def live_thread_alive(source: FlowceptSource) -> bool:
    c = source._consumer
    t = getattr(c, "_main_thread", None) if c else None
    return isinstance(t, threading.Thread) and t.is_alive()
