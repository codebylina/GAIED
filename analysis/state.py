"""
Derived state of the analysis service, kept in Redis so that a restarted service resumes where it was.

It holds aggregates and references, not provenance: for each agent, its timeline of decisions
(entity, decision, value, time); for each experiment, the ids of the Flowcept tasks that ran it;
counts of failures by signature; LLM calls per workflow. Ingestion is idempotent: every memory event
and every Flowcept task is applied at most once (processed ids are remembered).

  an:processed:events / an:processed:tasks   sets of applied ids
  an:wm:flowcept                              watermark of the Flowcept database catch-up (registered_at)
  an:agents                                   hash agent -> {workflow ids}
  an:tl:{agent}                               list of JSON {t, entity, decision, value, obs}
  an:runs:{condition}/{entity}                set of Flowcept task ids that computed the experiment
  an:runs:index                               set of "{condition}/{entity}" keys with >= 1 run
  an:fail:{signature}                         list of Flowcept task ids of failed experiment tasks
  an:llm:{workflow_id}                        number of LLM calls
"""
from __future__ import annotations

import json
import time

from memory.coordination import error_signature

TERMINAL = ("reuse", "computed", "computed_negative", "blocked_negative", "blocked_cooldown", "failed_permanent")
KNOWN = ("reuse", "blocked_negative", "blocked_cooldown")


class AnalysisState:
    def __init__(self, memory) -> None:
        self.m = memory
        self.r, self.k = memory.store.r, memory.store.k

    # ------------------------------------------------------------------ idempotency
    def first_time(self, kind: str, item_id: str) -> bool:
        return bool(self.r.sadd(self.k("an", "processed", kind), item_id))

    # ------------------------------------------------------------------ memory events
    def apply_event(self, ev: dict) -> set[str]:
        """Update aggregates from one memory event; returns the agents whose behaviour changed."""
        kind = ev.get("event")
        if kind == "agent":
            self.r.hset(self.k("an", "agents"), ev["agent"], json.dumps(ev.get("workflows", {})))
            return set()
        if kind == "decision" and ev.get("decision") in TERMINAL:
            value = ev.get("value")
            if value is None and ev.get("obs_id"):
                rec = self.m.store.get("M1", ev["obs_id"])
                value = rec.value if rec else None
            self.r.rpush(self.k("an", "tl", ev["agent"]), json.dumps(
                {"t": ev["t"], "entity": ev["entity"], "condition": ev.get("condition"), "decision": ev["decision"],
                 "value": value, "obs": ev.get("obs_id")}))
            self.r.ltrim(self.k("an", "tl", ev["agent"]), -2000, -1)
            return {ev["agent"]}
        return set()

    def timeline(self, agent: str, since: float | None = None, last: int | None = None) -> list[dict]:
        tl = [json.loads(x) for x in self.r.lrange(self.k("an", "tl", agent), 0, -1)]
        if since:
            tl = [x for x in tl if x["t"] > since]
        return tl[-last:] if last else tl

    def agents(self) -> list[str]:
        names = set(self.r.hkeys(self.k("an", "agents")))
        names |= {k.rsplit(":", 1)[1] for k in self.r.scan_iter(match=self.k("an", "tl", "*"))}
        return sorted(names)

    def workflows_of(self, agent: str) -> dict:
        return json.loads(self.r.hget(self.k("an", "agents"), agent) or "{}")

    # ------------------------------------------------------------------ Flowcept tasks
    def apply_task(self, t: dict) -> str | None:
        """Update aggregates from one Flowcept task summary; returns the experiment key it touched."""
        if t.get("subtype") == "llm_call" and t.get("workflow_id"):
            self.r.hincrby(self.k("an", "llm"), t["workflow_id"], 1)
            return None
        if not t.get("experiment"):
            return None
        key = f"{t['condition']}/{t['entity']}"
        self.r.sadd(self.k("an", "runs", key), t["task_id"])
        self.r.hset(self.k("an", "run_agent"), t["task_id"], t.get("agent") or "")
        self.r.sadd(self.k("an", "runs", "index"), key)
        if not t.get("ok") and t.get("error"):
            cls = self.m.domain.classify_failure(t["error"])
            if cls != "scientific":
                sig = f"{cls}|{error_signature(t['error'])}"
                self.r.rpush(self.k("an", "fail", sig), t["task_id"])
                self.r.sadd(self.k("an", "fail", "index"), sig)
        return key

    def runs(self) -> dict[str, list[str]]:
        out = {}
        for key in self.r.smembers(self.k("an", "runs", "index")):
            out[key] = sorted(self.r.smembers(self.k("an", "runs", key)))
        return out

    def run_agents(self, task_ids: list[str]) -> list[str]:
        return [a for a in self.r.hmget(self.k("an", "run_agent"), task_ids) if a] if task_ids else []

    def failures(self) -> dict[str, list[str]]:
        return {sig: self.r.lrange(self.k("an", "fail", sig), 0, -1)
                for sig in self.r.smembers(self.k("an", "fail", "index"))}

    def llm_calls(self) -> dict[str, int]:
        return {k: int(v) for k, v in self.r.hgetall(self.k("an", "llm")).items()}

    # ------------------------------------------------------------------ watermark and heartbeat
    def watermark(self) -> float | None:
        v = self.r.get(self.k("an", "wm", "flowcept"))
        return float(v) if v else None

    def set_watermark(self, ts: float) -> None:
        self.r.set(self.k("an", "wm", "flowcept"), ts)

    def heartbeat(self, consumer: str, info: dict) -> None:
        self.r.set(self.k("an", "heartbeat", consumer), json.dumps({"t": time.time(), **info}), ex=60)
