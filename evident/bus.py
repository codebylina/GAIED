"""
Typed topics with a common envelope, and the per-agent listener that makes EVIDENT push-based.

Topics are Redis Streams (not pub/sub: a stream keeps its messages, so a subscriber that
was busy or restarted reads what it missed, from its last id). One stream per topic:

  ev:{campaign}:topic:reservations   who reserved / released which molecule (lease)
  ev:{campaign}:topic:facts          a measured result was written (energy or failure)
  ev:{campaign}:topic:claims         the status of a claim changed (verifier)
  ev:{campaign}:topic:decisions      plans and proposals of the agents, BEFORE execution
  ev:{campaign}:directives:{agent}   directives for one agent (and directives:* for all)

Every topic message carries the same envelope, so that provenance (Flowcept task_id) and
memory events can be joined by analysis agents:
  topic, event, campaign, agent, activity (LangGraph node), task_id, iteration, t, payload
"""
from __future__ import annotations

import json
import logging
import threading
import time

logger = logging.getLogger(__name__)

TOPICS = ("reservations", "facts", "claims", "decisions")


def topic_key(campaign: str, topic: str) -> str:
    return f"ev:{campaign}:topic:{topic}"


def publish(client, campaign: str, topic: str, event: str, agent: str,
            activity: str | None = None, task_id: str | None = None,
            iteration: int = 0, **payload) -> str | None:
    """Append one enveloped message to a topic. Never raises (memory must not break agents)."""
    try:
        return client.xadd(topic_key(campaign, topic), {
            "topic": topic, "event": event, "campaign": campaign, "agent": agent,
            "activity": activity or "", "task_id": task_id or "", "iteration": iteration,
            "t": f"{time.time():.6f}", "payload": json.dumps(payload, default=str)})
    except Exception as exc:
        logger.debug("publish to %s failed: %s", topic, exc)
        return None


def decode(fields: dict) -> dict:
    """Envelope fields of a topic message, with the payload decoded."""
    out = dict(fields)
    out["t"] = float(out.get("t", 0))
    out["iteration"] = int(out.get("iteration", 0))
    out["payload"] = json.loads(out.get("payload") or "{}")
    return out


class Listener(threading.Thread):
    """Background thread blocked on XREAD over several streams; calls `on_entry(stream,
    entry_id, fields)` for every new message, as soon as it is written (push)."""

    def __init__(self, client, cursors: dict[str, str], on_entry, block_ms: int = 1000):
        super().__init__(daemon=True, name="evident-listener")
        self.client = client
        self.cursors = dict(cursors)
        self.on_entry = on_entry
        self.block_ms = block_ms
        self._stop_flag = threading.Event()

    def run(self) -> None:
        while not self._stop_flag.is_set():
            try:
                resp = self.client.xread(self.cursors, count=200, block=self.block_ms)
            except Exception as exc:
                logger.debug("listener read failed: %s", exc)
                time.sleep(0.5)
                continue
            for stream, entries in resp or []:
                for entry_id, fields in entries:
                    self.cursors[stream] = entry_id
                    try:
                        self.on_entry(stream, entry_id, fields)
                    except Exception as exc:
                        logger.debug("listener callback failed: %s", exc)

    def stop(self) -> None:
        self._stop_flag.set()
