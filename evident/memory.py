"""
EVIDENT: evidence-grounded shared memory for teams of scientific agents (Redis).

Five kinds of entries, each with a different write rule:
  FACTS        measured results (energy or failure). Written ONLY by the code that ran
               the computation, with the Flowcept task_id of that computation.
               Append-only log + one entry per molecule (canonical SMILES).
  COORDINATION who is computing what right now: a molecule is RESERVED as soon as an
               agent proposes it, with a short LEASE (SET NX EX) renewed while the agent
               works on it; a crashed agent's molecules are free again after one lease.
               An agent that needs a molecule reserved by another does not recompute it
               and does not poll: it waits for the fact to be PUBLISHED (push).
  CLAIMS       structure-property hypotheses asserted by agents, in a closed vocabulary.
               Their status is computed by code from ALL facts (verifier.py), never by
               the LLM; every status change is versioned, nothing is deleted.
  DIRECTIVES   messages from analysis agents (monitor.py, Nadjiba's agents): advice, or
               STOP, which cancels the agent's current activity in real time.
  TELEMETRY    one stream with every proposal, computation, assertion, briefing and
               directive: the data used to evaluate the architecture.

The same class implements the four memory modes compared in the experiments:
  none     isolated agents (baseline): nothing is read from the team, only telemetry
  coord    shared facts + locks (no duplicate computation, failures are shared)
  naive    coord + the raw LLM conclusions of the other agents (unverified notes,
           what most agent-memory systems do)
  evident  coord + verified claims (only supported knowledge is shown, with numbers)

Notification (EVIDENT_NOTIFY): "push" (default) = each agent has a listener on the topics
(bus.py) and is woken as soon as a fact or a directive is published; "poll" = the agent
re-reads Redis at fixed intervals (1 s for facts, 0.5 s for directives), kept to measure
the difference.

Redis keys, all prefixed by ev:{campaign}:
  goal                    JSON
  facts                   stream, every computation (duplicates included)
  facts:by_smiles         hash  smiles -> JSON of the first fact
  ranking                 zset  smiles by energy;      failed   set of invalid molecules
  lock:{smiles}           lease (value = agent);      inprogress  hash smiles -> agent
  claims                  hash  claim id -> JSON state; claims:log  stream of versions
  lessons                 stream of raw conclusions (naive mode)
  directives:{agent}      stream per agent;           directives:*  broadcast
  telemetry               stream of events;           stats  hash of counters
  agents                  set;                        liar   agent chosen for fault injection
  topic:{name}            typed topics with a common envelope (bus.py)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from collections import deque

from . import bus

from .claims import claim_id, claim_sentence, extract_claims, strip_claims_block
from .features import (FEATURE_NAMES, canonical_smiles, constraint_violations, describe,
                       mol_info)
from .verifier import (SHOWN_AS_KNOWLEDGE, SHOWN_AS_WARNING, VerifierParams, adjusted_effects, evaluate,
                       measure, next_status, ok_facts)

logger = logging.getLogger(__name__)

MODES = ("none", "coord", "naive", "evident")
_RELEASE_LUA = ("if redis.call('get', KEYS[1]) == ARGV[1] then "
                "return redis.call('del', KEYS[1]) else return 0 end")


class SimulationFailed(Exception):
    """The molecule already failed (computed by another agent): not retried."""


class StoppedByDirective(Exception):
    """The current activity was cancelled by a STOP directive of an analysis agent."""

    def __init__(self, directive: dict):
        self.directive = directive
        super().__init__(f"Stopped by the analysis agent ({directive.get('rule') or 'stop'}): "
                         "not computed")


def _env_bool(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


class EvidentMemory:
    def __init__(self, client, campaign_id: str, agent_id: str, mode: str = "evident",
                 claim_ttl: int = 30, params: VerifierParams | None = None,
                 poll: float = 1.0, control_poll: float = 0.5, notify: str = "push"):
        if mode not in MODES:
            raise ValueError(f"MEMORY_MODE must be one of {MODES}, got {mode!r}")
        if notify not in ("push", "poll"):
            raise ValueError(f"EVIDENT_NOTIFY must be push or poll, got {notify!r}")
        self.r = client
        self.campaign_id = campaign_id
        self.agent_id = agent_id
        self.mode = mode
        self.claim_ttl = claim_ttl
        self.params = params or VerifierParams()
        self.poll = poll
        self.control_poll = control_poll
        self.iteration = 0
        self.is_liar = False
        self._cursors = {self.k("directives", agent_id): "0-0", self.k("directives", "*"): "0-0"}
        self._last_poll = 0.0
        # STOP directives in force. A directive with scope "molecules" only stops the
        # listed molecules; any other STOP stops every activity of the agent.
        self._active_stops: list[dict] = []
        self._recent_directives: deque = deque(maxlen=3)
        self._has_lua = True
        self.notify = notify
        self.activity: str | None = None           # current LangGraph node (envelope)
        self._held: set[str] = set()               # molecules whose lease this agent holds
        self._futures: dict[str, asyncio.Future] = {}   # molecules this agent waits for
        self._loop: asyncio.AbstractEventLoop | None = None
        self._listener: bus.Listener | None = None
        self._renewer: asyncio.Task | None = None
        self._dlock = threading.Lock()
        self._seen_directives: set[str] = set()
        self._notify_latency: list[float] = []     # fact published -> received (push)

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_env(cls, agent_id: str, client=None) -> "EvidentMemory | None":
        """Memory configured by environment variables; None if Redis is unreachable."""
        mode = os.environ.get("MEMORY_MODE", "evident").strip().lower()
        campaign = (os.environ.get("MEMORY_CAMPAIGN_ID")
                    or os.environ.get("FLOWCEPT_CAMPAIGN_ID") or "default")
        try:
            if client is None:
                import redis
                client = redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"),
                                     port=int(os.environ.get("REDIS_PORT", "6379")),
                                     decode_responses=True)
            client.ping()
        except Exception as exc:
            logger.warning("EVIDENT memory unavailable (Redis unreachable): %s", exc)
            return None
        mem = cls(client, campaign, agent_id, mode=mode,
                  claim_ttl=int(os.environ.get("EVIDENT_LEASE_TTL", "30")),
                  params=VerifierParams.from_env(),
                  notify=os.environ.get("EVIDENT_NOTIFY", "push").strip().lower())
        client.zadd("ev:campaigns", {campaign: time.time()}, nx=True)
        if agent_id not in ("orchestrator", "monitor", "cli"):
            client.sadd(mem.k("agents"), agent_id)
            if _env_bool("INJECT_FALSE_CLAIMS"):
                mem.is_liar = bool(client.set(mem.k("liar"), agent_id, nx=True))
        return mem

    def k(self, *parts) -> str:
        return ":".join(["ev", self.campaign_id, *map(str, parts)])

    def _emit(self, event: str, **data) -> None:
        """Telemetry event (written in every mode, read only by analysis tools)."""
        try:
            self.r.xadd(self.k("telemetry"), {
                "e": event, "agent": self.agent_id, "it": self.iteration,
                "act": self.activity or "",
                "t": f"{time.time():.6f}", "d": json.dumps(data, default=str)})
        except Exception as exc:  # telemetry must never break an agent
            logger.debug("telemetry failed: %s", exc)

    def _pub(self, topic: str, event: str, task_id: str | None = None, **payload) -> None:
        bus.publish(self.r, self.campaign_id, topic, event, self.agent_id,
                    activity=self.activity, task_id=task_id, iteration=self.iteration, **payload)

    # ------------------------------------------------------------------ background (push)
    def _ensure_background(self) -> None:
        """Start, inside the agent's event loop, the listener (push mode) and the lease
        renewal task. Called lazily from the first coroutine that needs them."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._loop is not loop:                 # first use, or a new event loop (tests)
            self.close()
            self._loop = loop
        if self._renewer is None or self._renewer.done():
            self._renewer = loop.create_task(self._renew_leases())
        if self.notify == "push" and (self._listener is None or not self._listener.is_alive()):
            cursors = {bus.topic_key(self.campaign_id, "facts"): "$"}
            with self._dlock:
                cursors.update(self._cursors)
            self._listener = bus.Listener(self.r, cursors, self._on_message)
            self._listener.start()

    def _push_active(self) -> bool:
        return (self.notify == "push" and self._listener is not None
                and self._listener.is_alive())

    def _on_message(self, stream: str, entry_id: str, fields: dict) -> None:
        """Listener thread: a new message on a subscribed topic."""
        if stream.endswith(":topic:facts"):
            env = bus.decode(fields)
            smiles = env["payload"].get("smiles")
            self._notify_latency.append(time.time() - env["t"])
            if self._loop is not None and smiles:
                self._loop.call_soon_threadsafe(self._resolve, smiles)
        else:
            self._ingest_directive(stream, entry_id, fields)

    def _resolve(self, smiles: str) -> None:
        fut = self._futures.pop(smiles, None)
        if fut is not None and not fut.done():
            fut.set_result(True)

    async def _renew_leases(self) -> None:
        """Keep the leases of the molecules this agent is working on alive."""
        while True:
            await asyncio.sleep(max(self.claim_ttl / 3, 0.2))
            for can in list(self._held):
                key = self.k("lock", can)
                try:
                    if self.r.get(key) == self.agent_id:
                        self.r.expire(key, self.claim_ttl)
                    else:
                        self._held.discard(can)
                except Exception:
                    pass

    def close(self) -> None:
        """Stop the listener and the renewal task (end of an agent, tests)."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        if self._renewer is not None:
            try:
                self._renewer.cancel()
            except RuntimeError:                   # its event loop is already closed
                pass
            self._renewer = None

    def _count(self, name: str) -> None:
        self.r.hincrby(self.k("stats"), name, 1)

    # ------------------------------------------------------------------ goal
    def set_goal(self, goal: dict) -> None:
        self.r.set(self.k("goal"), json.dumps(goal))
        self._emit("goal", goal=goal)

    def set_meta(self, **meta) -> None:
        """Run configuration (mode, monitor, injection...), stored for the analysis."""
        self.r.hset(self.k("meta"), mapping={k: json.dumps(v) for k, v in meta.items()})

    def get_meta(self) -> dict:
        return {k: json.loads(v) for k, v in self.r.hgetall(self.k("meta")).items()}

    def get_goal(self) -> dict | None:
        raw = self.r.get(self.k("goal"))
        return json.loads(raw) if raw else None

    def constraints(self) -> dict:
        return (self.get_goal() or {}).get("constraints", {})

    # ------------------------------------------------------------------ facts
    def get_fact(self, smiles: str) -> dict | None:
        raw = self.r.hget(self.k("facts:by_smiles"), canonical_smiles(smiles))
        return json.loads(raw) if raw else None

    def push_fact(self, smiles: str, energy: float | None, error: str | None = None,
                  task_id: str | None = None, duration: float | None = None) -> dict:
        can = canonical_smiles(smiles)
        info = mol_info(can)
        fact = {"smiles": can, "energy": energy, "status": "error" if error else "ok",
                "error": error, "agent": self.agent_id, "task_id": task_id,
                "iteration": self.iteration, "t": time.time(), "duration": duration,
                "features": info["features"], "heavy_atoms": info["heavy_atoms"]}
        payload = json.dumps(fact)
        pipe = self.r.pipeline()
        pipe.xadd(self.k("facts"), {"fact": payload})
        pipe.hsetnx(self.k("facts:by_smiles"), can, payload)
        if error:
            pipe.sadd(self.k("failed"), can)
        else:
            pipe.zadd(self.k("ranking"), {can: energy}, nx=True)
        pipe.execute()
        self._pub("facts", "fact", task_id=task_id, smiles=can, energy=energy,
                  status=fact["status"], error=error)
        return fact

    def all_facts(self) -> list[dict]:
        return [json.loads(v) for v in self.r.hvals(self.k("facts:by_smiles"))]

    def best(self, n: int = 10) -> list[tuple[str, float]]:
        return self.r.zrevrange(self.k("ranking"), 0, n - 1, withscores=True)

    def failed(self) -> list[str]:
        return sorted(self.r.smembers(self.k("failed")))

    # ------------------------------------------------------------------ coordination
    def claim(self, smiles: str, ttl: int | None = None) -> bool:
        can = canonical_smiles(smiles)
        ok = bool(self.r.set(self.k("lock", can), self.agent_id, nx=True, ex=ttl or self.claim_ttl))
        if ok:
            self.r.hset(self.k("inprogress"), can, self.agent_id)
            self._held.add(can)
            self._pub("reservations", "reserved", smiles=can)
        return ok

    def claimed_by(self, smiles: str) -> str | None:
        return self.r.get(self.k("lock", canonical_smiles(smiles)))

    def release(self, smiles: str) -> None:
        """Release the lock only if this agent holds it (atomic with Lua when available)."""
        can = canonical_smiles(smiles)
        key = self.k("lock", can)
        released = False
        if self._has_lua:
            try:
                released = bool(self.r.eval(_RELEASE_LUA, 1, key, self.agent_id))
            except Exception:
                self._has_lua = False
        if not self._has_lua and self.r.get(key) == self.agent_id:
            self.r.delete(key)
            released = True
        if released or self.r.get(key) is None:
            self.r.hdel(self.k("inprogress"), can)
        if can in self._held:
            self._held.discard(can)
            self._pub("reservations", "released", smiles=can)

    def in_progress(self, exclude_self: bool = True) -> dict[str, str]:
        out = {}
        for smiles, agent in self.r.hgetall(self.k("inprogress")).items():
            if exclude_self and agent == self.agent_id:
                continue
            if self.r.exists(self.k("lock", smiles)):     # ignore expired locks of crashed agents
                out[smiles] = agent
        return out

    def known_molecules(self) -> set[str]:
        return (set(self.r.hkeys(self.k("facts:by_smiles")))
                | set(self.in_progress(exclude_self=False)))

    def exclusion_list(self, own: list[str], limit: int = 150) -> list[str]:
        """Molecules the agent must not propose again (own only in mode none)."""
        own_can = [canonical_smiles(s) for s in own]
        if self.mode == "none":
            return sorted(set(own_can))[-limit:]
        team = self.known_molecules()
        return sorted(set(own_can) | team)[:limit]

    # ------------------------------------------------------------------ evaluation
    async def evaluate_molecule(self, smiles: str, compute, new_task_id=None) -> float:
        """Return the energy of a molecule. compute(task_id) runs xTB.

        none : compute (the agent's local cache is the only reuse)
        other: READ the fact -> or, if this agent holds (or obtains) the lease, compute,
               WRITE and publish the fact, release -> or, if another agent holds it,
               wait until the fact is published (push) or the lease expires.
        Any running computation is cancelled if a STOP directive arrives."""
        can = canonical_smiles(smiles)
        t0 = time.time()
        self._ensure_background()
        if self.mode == "none":
            self._raise_if_stopped(can)
            return await self._compute_and_store(can, compute, new_task_id, t0)
        waited = False
        while True:
            self._raise_if_stopped(can)
            fact = self.get_fact(can)
            if fact is not None:
                outcome = ("reused_error" if fact["status"] == "error"
                           else "waited" if waited else "reused")
                self._count(outcome)
                self._emit("compute", smiles=can, outcome=outcome, by=fact["agent"],
                           task_id=fact.get("task_id"), wait=round(time.time() - t0, 3))
                if fact["status"] == "error":
                    raise SimulationFailed(f"{fact['error']} (already failed for {fact['agent']})")
                return fact["energy"]
            holder = self.claimed_by(can)
            if holder == self.agent_id or (holder is None and self.claim(can)):
                try:
                    return await self._compute_and_store(can, compute, new_task_id, t0)
                finally:
                    self.release(can)
            waited = True
            await self._wait_for_fact(can)

    async def _wait_for_fact(self, can: str) -> None:
        """Wait until the fact of `can` is published by the agent holding its lease
        (push), or that lease expires (crashed agent); poll mode re-reads every second."""
        if not self._push_active():
            await asyncio.sleep(self.poll)
            return
        fut = self._futures.get(can)
        if fut is None or fut.done():
            fut = self._loop.create_future()
            self._futures[can] = fut
        if self.get_fact(can) is not None:          # published before we subscribed
            return
        pttl = self.r.pttl(self.k("lock", can))
        if pttl is None or pttl < 0:                # lease already gone
            return
        try:
            await self.watched(asyncio.wait_for(asyncio.shield(fut), timeout=pttl / 1000 + 0.05),
                               smiles=can)
        except asyncio.TimeoutError:
            pass

    async def _compute_and_store(self, can, compute, new_task_id, t0) -> float:
        task_id = new_task_id() if new_task_id else str(uuid.uuid4())
        started = time.time()
        try:
            energy = await self.watched(compute(task_id), smiles=can)
        except StoppedByDirective as stop:
            self._count("cancelled")
            self._emit("compute", smiles=can, outcome="cancelled", task_id=task_id,
                       duration=round(time.time() - started, 3), directive=stop.directive.get("id"))
            raise
        except asyncio.CancelledError:
            self._emit("compute", smiles=can, outcome="cancelled", task_id=task_id,
                       duration=round(time.time() - started, 3))
            raise
        except Exception as exc:
            duration = time.time() - started
            self.push_fact(can, None, error=str(exc)[:300], task_id=task_id, duration=duration)
            self._count("computed_error")
            self._emit("compute", smiles=can, outcome="computed_error", task_id=task_id,
                       duration=round(duration, 3), error=str(exc)[:200])
            raise
        duration = time.time() - started
        self.push_fact(can, float(energy), task_id=task_id, duration=duration)
        self._count("computed")
        self._emit("compute", smiles=can, outcome="computed", task_id=task_id,
                   energy=float(energy), duration=round(duration, 3))
        return energy

    # ------------------------------------------------------------------ proposals and decisions
    def record_proposals(self, smiles_list: list[str], own_previous: list[str],
                         retries: int = 0) -> list[dict]:
        """Log what the agent decided to compute, BEFORE computing (analysis agents
        see it in real time). Starts a new batch: previous STOPs are now handled."""
        self._clear_stops()
        constraints = self.constraints()
        own_prev = {canonical_smiles(s) for s in own_previous}
        team_known = set(self.r.hkeys(self.k("facts:by_smiles")))
        inprog = self.in_progress()
        out = []
        for smiles in smiles_list:
            can = canonical_smiles(smiles)
            fact = self.get_fact(can) if can in team_known else None
            rec = {"smiles": can, "raw": smiles, "valid": mol_info(can)["valid"],
                   "features": mol_info(can)["features"],
                   "violations": [v for v in constraint_violations(can, constraints)
                                  if v != "invalid_smiles"],
                   "own_repeat": can in own_prev,
                   "team_known": ("failed" if fact and fact["status"] == "error"
                                  else "computed" if fact else
                                  "in_progress" if can in inprog else "new")}
            if self.mode != "none" and rec["team_known"] == "new" and rec["valid"]:
                # reserve at decision time: the team sees at once what is taken
                rec["reservation"] = ("mine" if self.claimed_by(can) == self.agent_id
                                      or self.claim(can) else "taken")
            out.append(rec)
        self._ensure_background()
        self._emit("proposals", items=out, retries=retries)
        self._pub("decisions", "proposals", items=[
            {k: r[k] for k in ("smiles", "violations", "team_known") if k in r} for r in out])
        return out

    def record_tool_call_failure(self, attempts: int, responses: list[str]) -> None:
        """The LLM gave no parsable tool call after all attempts: the iteration goes on
        without computation. The beginning of each answer is kept to see why."""
        self._count("tool_call_failed")
        self._emit("tool_call_failed", attempts=attempts,
                   responses=[(r or "")[:600] for r in responses])

    def record_decision(self, node: str, text: str) -> None:
        """The agent's intent (plan or updated plan), kept for drift analysis."""
        self._emit("decision", node=node, text=(text or "")[:2000])
        self._pub("decisions", "plan", node=node, text=(text or "")[:2000])

    # ------------------------------------------------------------------ conclusions, lessons, claims
    def record_conclusion(self, text: str, evidence: list[dict] | None = None) -> list[dict]:
        """Store what the agent concluded, according to the memory mode.
        Returns the structured claims extracted from the conclusion."""
        claims = extract_claims(text)
        lesson = strip_claims_block(text)
        if self.is_liar:
            claims, lesson = self._false_claims()
            self._emit("injection", claims=claims, lesson=lesson)
        self._emit("conclusion", claims=claims, text=lesson[:2000],
                   evidence=evidence or [], liar=self.is_liar)
        if self.mode == "naive":
            self.r.xadd(self.k("lessons"), {
                "agent": self.agent_id, "text": lesson, "t": f"{time.time():.3f}",
                "claims": json.dumps(claims), "evidence": json.dumps(evidence or [], default=str)})
        elif self.mode == "evident":
            for c in claims:
                self._assert_claim(c, lesson)
            self.verify_claims()
        return claims

    def _false_claims(self) -> tuple[list[dict], str]:
        """Fault injection: the claim most CONTRADICTED by the current facts, asserted
        as true (a hallucinating or poisoned agent). Fallback without data: methyl up."""
        facts = ok_facts(self.all_facts())
        worst, worst_t = "methyl", 0.0
        for feat in FEATURE_NAMES:
            ev = measure(feat, facts, self.params)
            if (ev["t"] is not None and ev["n_with"] >= self.params.min_n
                    and ev["n_without"] >= self.params.min_n and ev["t"] < worst_t):
                worst, worst_t = feat, ev["t"]
        d = describe(worst)
        lesson = (f"Key finding confirmed by our simulations: adding a {d} clearly increases "
                  f"the ionization energy. Prioritize molecules that contain a {d}.")
        return [{"feature": worst, "direction": "up", "source": "injected"}], lesson

    def _assert_claim(self, c: dict, lesson: str) -> None:
        cid = claim_id(c["feature"], c["direction"])
        raw = self.r.hget(self.k("claims"), cid)
        state = json.loads(raw) if raw else {
            "id": cid, "feature": c["feature"], "direction": c["direction"],
            "status": "proposed", "version": 0, "proposers": [], "n_assertions": 0,
            "first_t": time.time(), "evidence": {}, "statements": []}
        state["n_assertions"] += 1
        if self.agent_id not in state["proposers"]:
            state["proposers"].append(self.agent_id)
        state["statements"] = (state["statements"] + [c.get("sentence") or lesson[:200]])[-3:]
        self.r.hset(self.k("claims"), cid, json.dumps(state))
        if raw is None:
            self.r.xadd(self.k("claims:log"), {"id": cid, "status": "proposed", "version": 0,
                                               "agent": self.agent_id, "t": f"{time.time():.3f}"})
        self._emit("claim_asserted", id=cid, status=state["status"],
                   evidence=state.get("evidence"))

    def verify_claims(self) -> list[dict]:
        """Recompute the status of every claim from ALL facts (versioned changes)."""
        if self.mode != "evident":
            return []
        if not self.r.set(self.k("verify_lock"), self.agent_id, nx=True, ex=30):
            return []                                  # another agent is verifying right now
        t0 = time.time()
        changes = []
        try:
            facts = self.all_facts()
            adjusted = adjusted_effects(facts)
            for cid, raw in self.r.hgetall(self.k("claims")).items():
                state = json.loads(raw)
                ev = evaluate(state["feature"], state["direction"], facts, self.params, adjusted)
                new = next_status(state["status"], ev, self.params)
                state["evidence"] = ev
                if new != state["status"]:
                    old = state["status"]
                    state["status"], state["version"] = new, state["version"] + 1
                    self.r.xadd(self.k("claims:log"), {
                        "id": cid, "status": new, "previous": old, "version": state["version"],
                        "evidence": json.dumps(ev), "agent": self.agent_id,
                        "t": f"{time.time():.3f}"})
                    changes.append({"id": cid, "from": old, "to": new, "evidence": ev})
                    self._pub("claims", "status", id=cid, previous=old, status=new,
                              version=state["version"], evidence=ev)
                self.r.hset(self.k("claims"), cid, json.dumps(state))
        finally:
            self.r.delete(self.k("verify_lock"))
        self._emit("verify", changes=changes, n_facts=len(facts),
                   duration=round(time.time() - t0, 4))
        return changes

    def claims(self) -> dict[str, dict]:
        return {cid: json.loads(v) for cid, v in self.r.hgetall(self.k("claims")).items()}

    # ------------------------------------------------------------------ directives (real-time control)
    def send_directive(self, target: str, kind: str, reason: str, rule: str = "",
                       trigger_t: float | None = None, **data) -> str:
        """target = an agent id or '*' (everybody); kind = 'stop' or 'advice'."""
        d = {"id": uuid.uuid4().hex[:12], "kind": kind, "reason": reason, "rule": rule,
             "by": self.agent_id, "target": target, "sent_at": time.time(),
             "trigger_t": trigger_t, **data}
        self.r.xadd(self.k("directives", target), {"d": json.dumps(d, default=str)})
        self._emit("directive_sent", **d)
        return d["id"]

    def _ingest_directive(self, stream: str, entry_id: str, fields: dict) -> dict | None:
        """One directive message (from the listener thread or from a read); each
        directive is taken into account once."""
        now = time.time()
        with self._dlock:
            if stream in self._cursors:
                self._cursors[stream] = entry_id
            d = json.loads(fields["d"])
            if d["id"] in self._seen_directives:
                return None
            self._seen_directives.add(d["id"])
            d["received_at"] = now
            self._recent_directives.append(d)
            stale = d.get("iteration") is not None and d["iteration"] < self.iteration
            if d["kind"] == "stop" and not stale:
                self._active_stops.append(d)
        self._emit("directive_received", id=d["id"], kind=d["kind"], rule=d.get("rule"),
                   delivery=round(now - d["sent_at"], 4), notify=self.notify)
        return d

    def poll_directives(self, force: bool = False) -> list[dict]:
        """Read new directives for this agent. In push mode the listener already
        delivers them; a forced read (briefing) is only a safety net."""
        now = time.time()
        if self._push_active() and not force:
            return []
        if not force and now - self._last_poll < self.control_poll / 2:
            return []
        self._last_poll = now
        try:
            with self._dlock:
                cursors = dict(self._cursors)
            resp = self.r.xread(cursors, count=50)
        except Exception:
            return []
        new = []
        for stream, entries in resp or []:
            for entry_id, fields in entries:
                d = self._ingest_directive(stream, entry_id, fields)
                if d is not None:
                    new.append(d)
        return new

    @staticmethod
    def _applies(stop: dict, smiles: str | None) -> bool:
        if stop.get("scope") == "molecules":
            return smiles is not None and smiles in (stop.get("smiles") or [])
        return True

    def check_stop(self, smiles: str | None = None) -> dict | None:
        """The STOP in force for this activity (a molecule, or None for an LLM call)."""
        self.poll_directives()
        return next((d for d in self._active_stops if self._applies(d, smiles)), None)

    def _clear_stops(self) -> None:
        for d in self._active_stops:
            self._emit("stop_cleared", directive=d.get("id"))
        self._active_stops = []

    def _raise_if_stopped(self, can: str) -> None:
        stop = self.check_stop(can)
        if stop is not None:
            self._count("cancelled_before_start")
            self._emit("compute", smiles=can, outcome="cancelled_before_start",
                       directive=stop.get("id"))
            raise StoppedByDirective(stop)

    async def watched(self, coro, smiles: str | None = None):
        """Run an activity (xTB computation of `smiles`, or an LLM call) and cancel it as
        soon as a STOP directive that applies to it arrives. Raises StoppedByDirective."""
        self._ensure_background()
        task = asyncio.ensure_future(coro)
        # push: directives arrive through the listener, the check is a local look-up
        interval = 0.05 if self._push_active() else self.control_poll
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=interval)
                if done:
                    return task.result()
                stop = self.check_stop(smiles)
                if stop is not None:
                    task.cancel()
                    self._emit("stop_applied", directive=stop.get("id"),
                               reaction=round(time.time() - stop["sent_at"], 4))
                    raise StoppedByDirective(stop)
        except asyncio.CancelledError:
            task.cancel()
            raise

    def begin_node(self, node: str) -> None:
        """Current activity, written in the envelope of every topic message."""
        self.activity = node

    def end_node(self, node: str, duration: float, interrupted: bool = False) -> None:
        """Node boundary: a STOP that interrupted this node has been handled; at the end
        of `simulate`, leases of molecules that were not computed are released."""
        self._emit("node", node=node, duration=round(duration, 4), interrupted=interrupted)
        if interrupted:
            self._clear_stops()
        if node == "simulate":
            for can in list(self._held):
                self.release(can)

    # ------------------------------------------------------------------ iterations
    def begin_iteration(self, i: int) -> None:
        self.iteration = i

    def end_iteration(self, own_best: float | None, no_improve: int) -> None:
        top = self.best(1)
        lat, self._notify_latency = self._notify_latency, []
        self._emit("iteration_end", own_best=own_best, no_improve=no_improve,
                   team_best=top[0][1] if top else None, notify=self.notify,
                   notify_n=len(lat),
                   notify_ms_mean=round(1000 * sum(lat) / len(lat), 3) if lat else None,
                   notify_ms_max=round(1000 * max(lat), 3) if lat else None)

    # ------------------------------------------------------------------ briefing for the prompts
    def _goal_text(self) -> str:
        goal = self.get_goal()
        if not goal:
            return ""
        lines = [f"GOAL of the team: {goal.get('objective', '')}."]
        c = goal.get("constraints", {})
        hard = []
        if c.get("max_heavy_atoms"):
            hard.append(f"at most {c['max_heavy_atoms']} heavy (non-hydrogen) atoms")
        if c.get("allowed_elements"):
            hard.append(f"only the elements {', '.join(c['allowed_elements'])}")
        if hard:
            lines.append("Hard constraints: " + "; ".join(hard) + ".")
        if goal.get("guidelines"):
            lines.append("Also consider: " + ", ".join(goal["guidelines"]) + ".")
        return "\n".join(lines)

    def _directives_text(self) -> str:
        if not self._recent_directives:
            return ""
        lines = ["DIRECTIVES FROM THE ANALYSIS AGENT (most recent last; you MUST follow them):"]
        for d in self._recent_directives:
            tag = "STOP" if d["kind"] == "stop" else "ADVICE"
            lines.append(f"- [{tag}] {d['reason']}")
        return "\n".join(lines)

    def briefing(self, node: str) -> str:
        """Bounded text added to the prompt of a node. Content depends on the mode."""
        t0 = time.perf_counter()
        self.poll_directives(force=True)
        parts = [self._directives_text(), self._goal_text()]   # directives first: hard to miss
        shown_claims: list[dict] = []
        lesson_claims: list[dict] = []
        if self.mode != "none":
            best = self.best(10)
            if best:
                parts.append("TEAM RESULTS (best ionization energies measured by any agent):\n"
                             + "\n".join(f"- {s}  {e:.3f}" for s, e in best))
            failed = self.failed()
            if failed:
                parts.append("INVALID molecules (do not retry): " + " ".join(failed[-15:]))
            inprog = self.in_progress()
            if inprog:
                parts.append("Being computed RIGHT NOW by other agents (do not propose): "
                             + " ".join(sorted(inprog)[:20]))
        if self.mode == "naive":
            notes = []
            for _, f in reversed(self.r.xrevrange(self.k("lessons"), count=30)):
                if f["agent"] != self.agent_id:
                    notes.append(f)
            notes = notes[-5:]
            if notes:
                parts.append("LESSONS FROM OTHER AGENTS:\n" + "\n".join(
                    f"- ({n['agent']}) {n['text'][:600]}" for n in notes))
                for n in notes:
                    lesson_claims += json.loads(n.get("claims") or "[]")
        if self.mode == "evident":
            claims = sorted(self.claims().values(),
                            key=lambda c: -abs((c.get("evidence") or {}).get("t") or 0))
            known = [c for c in claims if c["status"] in SHOWN_AS_KNOWLEDGE][:8]
            warn = [c for c in claims if c["status"] in SHOWN_AS_WARNING][:5]
            if known:
                parts.append("VERIFIED KNOWLEDGE (checked against all team measurements):\n"
                             + "\n".join(self._claim_line(c) for c in known))
            if warn:
                parts.append("DO NOT RELY ON (contradicted or no longer supported by data):\n"
                             + "\n".join(self._claim_line(c) for c in warn))
            shown_claims = [{"id": c["id"], "status": c["status"]} for c in known + warn]
        if self.mode != "none":
            parts.append("Do NOT propose molecules that were already measured or are being computed.")
        text = "\n\n".join(p for p in parts if p)
        self._emit("briefing", node=node, chars=len(text), shown_claims=shown_claims,
                   lesson_claims=lesson_claims, ms=round((time.perf_counter() - t0) * 1000, 3))
        return text

    @staticmethod
    def _claim_line(c: dict) -> str:
        ev = c.get("evidence") or {}
        nums = ""
        if ev.get("effect") is not None:
            adj = ev.get("adjusted") or {}
            adj_txt = (f", effect {adj['coef']:+.2f} when other features are accounted for (t={adj['t']:.1f})"
                       if adj.get("identifiable") else "")
            nums = (f" [{c['status']}: mean {ev['mean_with']:.2f} with vs {ev['mean_without']:.2f}"
                    f" without (n={ev['n_with']}/{ev['n_without']}){adj_txt},"
                    f" replicated by {ev.get('replications', 0)} agent(s)]")
        return f"- {claim_sentence(c['feature'], c['direction'])}{nums}"

    # ------------------------------------------------------------------ summary and export
    def summary(self) -> str:
        s = {k: int(v) for k, v in self.r.hgetall(self.k("stats")).items()}
        claims = self.claims()
        by_status: dict[str, int] = {}
        for c in claims.values():
            by_status[c["status"]] = by_status.get(c["status"], 0) + 1
        best = self.best(5)
        return "\n".join([
            "=" * 80, f"EVIDENT memory, campaign {self.campaign_id}, mode {self.mode}",
            f"  xTB computations: {s.get('computed', 0)} ok, {s.get('computed_error', 0)} failed;"
            f" reused: {s.get('reused', 0)}, waited: {s.get('waited', 0)},"
            f" failures reused: {s.get('reused_error', 0)}; cancelled: {s.get('cancelled', 0)}",
            f"  facts log: {self.r.xlen(self.k('facts'))} entries, "
            f"{self.r.hlen(self.k('facts:by_smiles'))} molecules",
            f"  claims: {by_status or 'none'}",
            "  best: " + ", ".join(f"{m} {e:.3f}" for m, e in best), "=" * 80])

    def export(self) -> dict:
        """Everything needed by the analysis, as plain JSON (no Redis needed afterwards)."""
        def stream(key, field=None):
            out = []
            for entry_id, f in self.r.xrange(self.k(key)):
                out.append(json.loads(f[field]) if field else {"id": entry_id, **f})
            return out
        telemetry = []
        for entry_id, f in self.r.xrange(self.k("telemetry")):
            telemetry.append({"id": entry_id, "e": f["e"], "agent": f["agent"],
                              "it": int(f["it"]), "t": float(f["t"]), **json.loads(f["d"])})
        meta = self.get_meta()
        return {"campaign": self.campaign_id, "mode": meta.get("mode", self.mode),
                "meta": meta, "goal": self.get_goal(),
                "agents": sorted(self.r.smembers(self.k("agents"))),
                "liar": self.r.get(self.k("liar")),
                "stats": self.r.hgetall(self.k("stats")),
                "facts_log": stream("facts", "fact"), "claims": self.claims(),
                "claims_log": stream("claims:log"), "lessons": stream("lessons"),
                "telemetry": telemetry}
