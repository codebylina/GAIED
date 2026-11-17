"""
Online Provenance & Memory Analysis service: an independent process that watches Flowcept provenance and
the four memory layers, detects problems, and feeds diagnostics, knowledge proposals and guidance back
into the shared memory. It is not a memory layer, and scientific agents never wait for it.

Inputs
  memory events   Redis Stream of the memory namespace, read with a consumer group ("analysis"): every
                  event is acknowledged only after it was applied; on restart the consumer first re-reads
                  its own pending events, then claims those of dead consumers. Events delivered more than
                  `max_deliveries` times go to a dead-letter stream (an:dead). Durable analysis triggers.
  provenance      live: Flowcept MQ messages (Pub/Sub, via Flowcept's BaseConsumer);
                  catch-up: Flowcept.db.task_query with a watermark on registered_at, stored in Redis
                  (covers the messages a Pub/Sub subscriber misses while down);
                  offline: Flowcept JSONL buffers (`ingest_buffers`).
                  Ingestion is idempotent (processed task ids are remembered).
Processing
  online          after each batch of new events / tasks, the detectors run and changed findings are fed back
  periodic        every `period_s`: guidance effectiveness and campaign-level analysis; every
                  `maintenance_s`: memory lifecycle (expiry, staleness, archival)
A finding is fed back only when its content changed since the last time (fingerprint per signature).
"""
from __future__ import annotations

import collections
import hashlib
import json
import logging
import threading
import time

from memory.provenance import FlowceptSource, summary

from .detectors import AnalysisConfig, Detectors
from .feedback import Feedback
from .state import AnalysisState

log = logging.getLogger("analysis")
GROUP = "analysis"


class AnalysisService:
    def __init__(self, memory, config: AnalysisConfig, source: FlowceptSource | None = None,
                 consumer: str = "analysis-1", period_s: float = 15.0, catchup_s: float = 5.0,
                 maintenance_s: float = 60.0, live: bool = True, explainer=None, max_deliveries: int = 5) -> None:
        self.m, self.cfg, self.source, self.consumer = memory, config, source, consumer
        self.period_s, self.catchup_s, self.maintenance_s = period_s, catchup_s, maintenance_s
        self.live, self.max_deliveries = live, max_deliveries
        self.state = AnalysisState(memory)
        self.detectors = Detectors(memory, self.state, config)
        self.feedback = Feedback(memory, self.detectors, config, None, explainer)
        self.stream = memory.store.k("events")
        self._live_q: collections.deque = collections.deque(maxlen=100_000)
        self._db_ok: bool | None = None
        self._last = {"period": 0.0, "catchup": 0.0, "maintenance": 0.0, "db_check": 0.0}
        self.stats = collections.Counter()
        try:
            memory.store.r.xgroup_create(self.stream, GROUP, id="0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    # ------------------------------------------------------------------ campaign ids (from agents' registrations)
    def campaigns(self) -> list[str]:
        return sorted(self.m.store.r.smembers(self.m.store.k("campaigns")))

    # ------------------------------------------------------------------ memory events (durable triggers)
    def _apply_entries(self, entries) -> int:
        r, n, dirty = self.m.store.r, 0, False
        for eid, fields in entries:
            if not fields:                        # deleted from the stream (trimmed): nothing to apply
                r.xack(self.stream, GROUP, eid)
                continue
            try:
                if self.state.first_time("events", eid):
                    ev = json.loads(fields["data"])
                    if self.state.apply_event(ev) or ev.get("event") in ("result", "knowledge_status", "experience"):
                        dirty = True
                r.xack(self.stream, GROUP, eid)
                n += 1
            except Exception as exc:
                log.warning("event %s failed (left pending for retry): %s", eid, exc)
                self.state.r.srem(self.state.k("an", "processed", "events"), eid)
                self.stats["event_errors"] += 1
        self.stats["events"] += n
        return n

    def recover(self) -> int:
        """Restart recovery: own pending events first, then events left pending by dead consumers."""
        r, n = self.m.store.r, 0
        while True:
            resp = r.xreadgroup(GROUP, self.consumer, {self.stream: "0"}, count=200)
            entries = resp[0][1] if resp else []
            if not entries:
                break
            n += self._apply_entries(entries)
        start = "0-0"
        while True:
            res = r.xautoclaim(self.stream, GROUP, self.consumer, min_idle_time=30_000, start_id=start, count=200)
            start, entries = res[0], res[1]
            n += self._apply_entries(entries)
            if start in ("0-0", b"0-0"):
                break
        self._dead_letters()
        self.stats["recovered"] += n
        return n

    def _dead_letters(self) -> None:
        r = self.m.store.r
        for p in r.xpending_range(self.stream, GROUP, min="-", max="+", count=500):
            if p["times_delivered"] > self.max_deliveries:
                msgs = r.xrange(self.stream, p["message_id"], p["message_id"])
                if msgs:
                    r.xadd(self.m.store.k("an", "dead"), {"id": p["message_id"], **msgs[0][1]})
                r.xack(self.stream, GROUP, p["message_id"])
                self.stats["dead_letters"] += 1

    def read_events(self, block_ms: int = 1000) -> int:
        resp = self.m.store.r.xreadgroup(GROUP, self.consumer, {self.stream: ">"}, count=500, block=block_ms)
        return self._apply_entries(resp[0][1]) if resp else 0

    # ------------------------------------------------------------------ provenance
    def _on_live(self, msg: dict) -> None:
        if msg.get("task_id"):
            self._live_q.append(msg)

    def _apply_tasks(self, tasks) -> int:
        camps, n = set(self.campaigns()), 0
        for t in tasks:
            if camps and t.get("campaign_id") not in camps:
                continue
            tid = t.get("task_id")
            if not tid or not self.state.first_time("tasks", tid):
                continue
            self.state.apply_task(summary(t, self.m.domain))
            n += 1
        self.stats["tasks"] += n
        return n

    def drain_live(self) -> int:
        batch = []
        while self._live_q:
            batch.append(self._live_q.popleft())
        return self._apply_tasks(batch) if batch else 0

    def catchup(self) -> int:
        """Durable catch-up from the Flowcept database (watermark on registered_at)."""
        if self.source is None:
            return 0
        now = time.time()
        if self._db_ok is None or (not self._db_ok and now - self._last["db_check"] > 30):
            self._db_ok, self._last["db_check"] = self.source.db_available(), now
        if not self._db_ok:
            return 0
        n, wm = 0, self.state.watermark()
        try:
            for camp in self.campaigns():
                tasks = self.source.tasks_since(camp, wm - 1.0 if wm else None)   # overlap: idempotent anyway
                n += self._apply_tasks(tasks)
                ts = [summary(t)["registered_at"] for t in tasks]
                ts = [x for x in ts if x]
                if ts:
                    wm = max(wm or 0, max(ts))
            if wm:
                self.state.set_watermark(wm)
        except Exception as exc:
            log.warning("Flowcept catch-up failed (will retry): %s", exc)
            self._db_ok = False
        return n

    def ingest_buffers(self, pattern: str) -> int:
        """Offline: Flowcept JSONL buffers of finished campaigns."""
        src = self.source or FlowceptSource(self.m.domain)
        recs = src.read_buffers(pattern)
        return self._apply_tasks(recs)

    # ------------------------------------------------------------------ analysis
    def _configure_from_agents(self) -> None:
        """Experimental condition and campaign ids come from the agents' registrations (no GAIED import)."""
        if not self.cfg.method:
            for info in self.m.store.r.hvals(self.m.store.k("an", "agents")):
                info = json.loads(info)
                if info.get("method"):
                    self.cfg.method, self.cfg.params = info["method"], info.get("params") or {}
                    break
        camps = self.campaigns()
        self.feedback.campaign = camps[-1] if camps else None

    def analyze(self) -> list[dict]:
        self._configure_from_agents()
        if not self.cfg.method:
            return []
        out = []
        for f in self.detectors.all():
            fp = hashlib.sha1(json.dumps([f.summary, f.facts, f.metrics], sort_keys=True, default=str).encode()).hexdigest()
            key = self.m.store.k("an", "fp", f.signature)
            if self.m.store.r.get(key) == fp:
                continue
            res = self.feedback.apply(f)
            self.m.store.r.set(key, fp)
            out.append(res)
            self.stats[f"finding.{f.kind}"] += 1
        return out

    def step(self, block_ms: int = 1000) -> dict:
        now, report = time.time(), {}
        n = self.read_events(block_ms) + self.drain_live()
        if now - self._last["catchup"] >= self.catchup_s:
            n += self.catchup()
            self._last["catchup"] = now
        periodic = now - self._last["period"] >= self.period_s
        if n or periodic:
            report["findings"] = self.analyze()
        if periodic:
            report["guidance"] = self.feedback.evaluate_guidance()
            self._last["period"] = now
        if now - self._last["maintenance"] >= self.maintenance_s:
            report["lifecycle"] = self.m.lifecycle.run(by="analysis")
            self._last["maintenance"] = now
        self.state.heartbeat(self.consumer, {"stats": dict(self.stats)})
        return report

    def run(self, stop: threading.Event | None = None, max_seconds: float | None = None) -> None:
        stop = stop or threading.Event()
        log.warning("analysis service %s on memory %s (recovered %d pending events)", self.consumer,
                    self.m.namespace, self.recover())
        if self.live and self.source is not None:
            log.warning("live Flowcept provenance: %s", "on" if self.source.subscribe(self._on_live) else "off")
        t0 = time.time()
        try:
            while not stop.is_set() and (max_seconds is None or time.time() - t0 < max_seconds):
                try:
                    rep = self.step()
                    for f in rep.get("findings", []):
                        log.warning("finding %s -> %s", f["kind"], {k: v for k, v in f.items() if k != "kind"})
                except Exception as exc:   # Redis or Flowcept temporarily down: keep the service alive
                    log.warning("analysis step failed (retrying): %s", exc)
                    self.stats["step_errors"] += 1
                    time.sleep(2)
        finally:
            if self.source is not None:
                self.source.stop()
