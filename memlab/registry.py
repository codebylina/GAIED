"""
L0, structured part: the experiment registry.

One entry per entity (canonical design key), built from provenance records only (never from an LLM):
every attempt with its status, value, cause of failure, agent, times and task id. It answers exactly,
before an experiment runs, whether the entity was already measured, already failed (and why), or is
being measured by another agent (a lease, renewed while the experiment runs).

Failure causes:
  invalid  the proposal is not a valid entity (e.g. an invalid SMILES): never worth running again
  error    the experiment itself failed (e.g. the simulator crashed): may be retried
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .base import Decision


@dataclass
class RegistryConfig:
    lease_ttl_s: float = 30.0      # a reservation expires unless renewed (engineering timeout, not a parameter
    #                                of the method: it only bounds how long a crashed agent blocks an entity)
    max_error_attempts: int = 1    # attempts allowed for an entity whose experiments failed with an error
    poll_s: float = 0.5            # how often a waiting agent checks for the result


class Registry:
    def __init__(self, store, adapter, cfg: RegistryConfig | None = None, clock=time.time) -> None:
        self.store, self.adapter, self.cfg, self.clock = store, adapter, cfg or RegistryConfig(), clock

    # ------------------------------------------------------------------ keys
    def key(self, raw: str) -> str:
        """Canonical key; an invalid proposal is keyed by its raw text."""
        return self.adapter.canonical(raw) or f"raw:{raw}"

    def key_of(self, obs: dict) -> str:
        if obs.get("valid_entity", True):
            return obs["entity"]
        raw = str(obs.get("raw") or obs["entity"])
        return raw if raw.startswith("raw:") else f"raw:{raw}"

    @staticmethod
    def lease(entity: str) -> str:
        return f"run:{entity}"

    # ------------------------------------------------------------------ writes (from provenance)
    def record(self, obs: dict) -> None:
        e = self.key_of(obs)
        doc = self.store.get("registry", e) or {"entity": e, "attempts": []}
        cause = None
        if obs["status"] != "ok":
            cause = "error" if obs.get("valid_entity", True) else "invalid"
        doc["attempts"].append({"id": obs["id"], "status": obs["status"], "value": obs.get("value"),
                                "cause": cause, "error": (obs.get("error") or "")[:200] or None,
                                "agent": obs.get("agent"), "started_at": obs.get("started_at"), "t": obs["t"]})
        self.store.put("registry", e, doc)
        if obs.get("agent"):
            self.store.release_lease(self.lease(e), obs["agent"])
        run = self.store.get("running", e)
        if run and run.get("status") == "running":
            self.store.put("running", e, {**run, "status": "done", "t_done": self.clock()})

    # ------------------------------------------------------------------ reads
    def view(self, entity: str) -> dict:
        """{'state': unknown|measured|invalid|error, 'value', 'owner', 'error', 'n_error'}"""
        doc = self.store.get("registry", entity)
        if not doc:
            return {"state": "unknown"}
        obj = self.adapter.goal.objective
        ok = [a for a in doc["attempts"] if a["status"] == "ok"]
        if ok:
            best = ok[0]
            for a in ok[1:]:
                if obj.better(a["value"], best["value"]):
                    best = a
            return {"state": "measured", "value": best["value"], "owner": best["agent"], "id": best["id"]}
        inv = [a for a in doc["attempts"] if a["cause"] == "invalid"]
        if inv:
            return {"state": "invalid", "owner": inv[-1]["agent"], "error": inv[-1]["error"]}
        err = [a for a in doc["attempts"] if a["cause"] == "error"]
        return {"state": "error", "owner": err[-1]["agent"], "error": err[-1]["error"], "n_error": len(err)}

    def claim(self, agent: str, raw: str) -> Decision:
        """What to do with one proposal. 'run' means the caller now holds the reservation."""
        e = self.key(raw)
        v = self.view(e)
        if v["state"] == "measured":
            return Decision(raw, e, "known", value=v["value"], owner=v["owner"])
        if v["state"] == "invalid" or (v["state"] == "error" and v["n_error"] >= self.cfg.max_error_attempts):
            return Decision(raw, e, "failed_before", owner=v["owner"])
        ok, holder = self.store.take_lease(self.lease(e), agent, self.cfg.lease_ttl_s)
        if not ok:
            return Decision(raw, e, "in_progress", owner=holder)
        self.store.put("running", e, {"entity": e, "agent": agent, "t": self.clock(), "status": "running"})
        return Decision(raw, e, "run")

    def renew(self, agent: str, entity: str) -> bool:
        return self.store.renew_lease(self.lease(entity), agent, self.cfg.lease_ttl_s)

    def release(self, agent: str, entity: str) -> None:
        self.store.release_lease(self.lease(entity), agent)

    def wait(self, entities: list[str], timeout_s: float) -> dict[str, dict]:
        """Results of entities other agents are measuring, as they arrive (at most timeout_s)."""
        out, deadline = {}, time.time() + timeout_s
        pending = list(entities)
        while pending:
            for e in list(pending):
                v = self.view(e)
                if v["state"] != "unknown":
                    out[e] = {"status": "ok" if v["state"] == "measured" else "failed", "value": v.get("value"),
                              "cause": v["state"] if v["state"] != "measured" else None}
                    pending.remove(e)
                elif self.store.lease_holder(self.lease(e)) is None:
                    pending.remove(e)            # the other agent gave up: the caller may run it
            if not pending or time.time() >= deadline:
                break
            time.sleep(self.cfg.poll_s)
        return out

    # ------------------------------------------------------------------ summaries for prompts
    def best(self) -> list[dict]:
        obj = self.adapter.goal.objective
        rows = []
        for doc in self.store.all("registry"):
            v = self.view(doc["entity"])
            if v["state"] == "measured":
                rows.append({"entity": doc["entity"], **v})
        return sorted(rows, key=lambda r: r["value"], reverse=obj.direction == "max")

    def failures(self) -> list[dict]:
        rows = []
        for doc in self.store.all("registry"):
            v = self.view(doc["entity"])
            if v["state"] in ("invalid", "error"):
                rows.append({"entity": doc["entity"], **v, "t": doc["attempts"][-1]["t"]})
        return sorted(rows, key=lambda r: -r["t"])

    def running(self) -> list[dict]:
        return [r for r in self.store.all("running")
                if r.get("status") == "running" and self.store.lease_holder(self.lease(r["entity"]))]
