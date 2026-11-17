"""
Coordination between scientific agents (Redis).

Locks: a claim is a key with an expiry (SET NX PX) whose value is a secret token "<agent>|<random>";
release and renewal are atomic compare-and-act Lua scripts, so an agent whose claim expired can never
delete or extend the claim another agent took meanwhile. A running computation renews its claim in the
background; a crashed agent's claim expires and the work is taken over.

get_or_compute (one experiment, computed at most once by the team):
  1. M1 knows a result          -> reuse it
  2. M1 knows a negative result -> refuse (KnownNegative): a valid negative scientific result
  3. a cooldown is active       -> TemporarilyUnavailable (after technical failures; it EXPIRES)
  4. another agent computes it  -> wait for its result
  5. otherwise claim it, run it with the retry policy, record the outcome, release the claim:
       success     -> M1 result (id = Flowcept task id); a success after failures -> M2 recovery episode
       scientific  -> M1 negative result, not retried
       transient   -> M2 incident; retried with exponential backoff; when attempts are exhausted, a short
                      cooldown (never permanent: a later call computes again)
       permanent   -> M2 incident; not retried now; a longer cooldown, then allowed again
Each attempt has its own task id, so every attempt is a distinct Flowcept task.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable

from .domain import PERMANENT, SCIENTIFIC, TRANSIENT, condition_id
from .layers.m2_experience import signature_of
from .schemas import prov_ref

log = logging.getLogger("memory.coordination")

_RELEASE = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end return 0"
_RENEW = ("if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('PEXPIRE', KEYS[1], ARGV[2]) end "
          "return 0")


class KnownNegative(Exception):
    """The memory holds a valid negative scientific result for this experiment."""


class TemporarilyUnavailable(Exception):
    """Recent technical failures: the experiment may be tried again after the cooldown."""


@dataclass
class RetryPolicy:
    max_attempts: int = 3            # attempts within one call, for transient failures
    backoff_s: float = 2.0           # first wait, doubled at each attempt
    transient_cooldown_s: float = 30.0
    permanent_cooldown_s: float = 600.0
    claim_ttl_s: float = 300.0       # expiry of a claim that is not renewed (crashed agent)
    wait_timeout_s: float | None = None   # None: wait as long as the holder's claim is alive


def error_signature(error: str) -> str:
    """Error text without the parts that change between occurrences (numbers, ids, paths)."""
    e = re.sub(r"0x[0-9a-f]+|[0-9a-f]{8,}|\d+(\.\d+)?", "#", error.lower())
    e = re.sub(r"(/[\w.\-]+)+", "<path>", e)
    return " ".join(e.split())[:160]


class Coordinator:
    def __init__(self, memory, policy: RetryPolicy | None = None) -> None:
        self.m = memory
        self.policy = policy or RetryPolicy()
        r = memory.store.r
        self._release = r.register_script(_RELEASE)
        self._renew = r.register_script(_RENEW)

    # ------------------------------------------------------------------ locks
    def _key(self, task: str) -> str:
        return self.m.store.k("coord", "lock", task)

    def claim(self, task: str, agent: str, ttl_s: float | None = None) -> str | None:
        token = f"{agent}|{uuid.uuid4().hex[:12]}"
        ttl = int((ttl_s or self.policy.claim_ttl_s) * 1000)
        return token if self.m.store.r.set(self._key(task), token, nx=True, px=ttl) else None

    def release(self, task: str, token: str) -> bool:
        return bool(self._release(keys=[self._key(task)], args=[token]))

    def renew(self, task: str, token: str, ttl_s: float | None = None) -> bool:
        return bool(self._renew(keys=[self._key(task)], args=[token, int((ttl_s or self.policy.claim_ttl_s) * 1000)]))

    def holder(self, task: str) -> str | None:
        token = self.m.store.r.get(self._key(task))
        return token.rsplit("|", 1)[0] if token else None

    def running(self) -> dict[str, str]:
        prefix = self._key("")
        out = {}
        for k in self.m.store.r.scan_iter(match=prefix + "*", count=500):
            token = self.m.store.r.get(k)
            if token:
                out[k[len(prefix):]] = token.rsplit("|", 1)[0]
        return out

    def cooldown(self, task: str) -> float:
        return max(0.0, self.m.store.r.pttl(self.m.store.k("coord", "cooldown", task)) / 1000.0)

    def _set_cooldown(self, task: str, seconds: float, reason: str) -> None:
        if seconds > 0:
            self.m.store.r.set(self.m.store.k("coord", "cooldown", task), reason[:200], px=int(seconds * 1000))

    # ------------------------------------------------------------------ status
    def status(self, raw: str, method: str, params: dict | None) -> dict:
        st = self.m.results.lookup(raw, method, params)
        task = f"{st['condition']}/{st['entity']}"
        if st["state"] == "unknown":
            if (h := self.holder(task)):
                return {**st, "state": "running", "agent": h}
            if (c := self.cooldown(task)):
                return {**st, "state": "cooldown", "retry_in_s": round(c, 1)}
        return st

    # ------------------------------------------------------------------ get_or_compute
    async def get_or_compute(self, raw: str, compute: Callable[[str], Awaitable[float]], *, method: str,
                             params: dict | None = None, poll_s: float = 1.0) -> float:
        m, agent, pol = self.m, self.m.agent, self.policy
        entity, cond = m.results.key_of(raw), condition_id(method, params)
        task, t0 = f"{cond}/{entity}", time.time()
        while True:
            st = self.status(raw, method, params)
            if st["state"] == "success":
                m.decision("reuse", entity, cond, obs_id=st["id"], value=st["value"], computed_by=st.get("agent"))
                return st["value"]
            if st["state"] == "negative":
                m.decision("blocked_negative", entity, cond, obs_id=st["id"])
                raise KnownNegative(f"{entity}: valid negative result already known ({st.get('detail')})")
            if st["state"] == "cooldown":
                m.decision("blocked_cooldown", entity, cond, retry_in_s=st["retry_in_s"])
                raise TemporarilyUnavailable(f"{entity}: recent technical failures, retry in {st['retry_in_s']} s")
            if st["state"] == "running":
                w0 = time.time()
                while self.holder(task) and m.results.lookup(raw, method, params)["state"] == "unknown":
                    if pol.wait_timeout_s and time.time() - w0 > pol.wait_timeout_s:
                        break
                    await asyncio.sleep(poll_s)
                m.decision("waited", entity, cond, holder=st["agent"], waited_s=round(time.time() - w0, 3))
                continue
            token = self.claim(task, agent)
            if token is None:
                continue
            if m.results.lookup(raw, method, params)["state"] != "unknown":     # finished just before the claim
                self.release(task, token)
                continue
            keep = asyncio.create_task(self._keep_alive(task, token))
            try:
                return await self._run_with_policy(raw, entity, cond, method, params, compute)
            finally:
                keep.cancel()
                try:
                    self.release(task, token)
                except Exception as exc:   # the claim expires anyway
                    log.warning("could not release %s: %s", task, exc)

    async def _run_with_policy(self, raw, entity, cond, method, params, compute) -> float:
        m, pol = self.m, self.policy
        incident_id = None
        for attempt in range(1, pol.max_attempts + 1):
            task_id, a0 = str(uuid.uuid4()), time.time()
            try:
                value = await compute(task_id)
            except Exception as exc:
                error = str(exc) or type(exc).__name__
                cls = m.domain.classify_failure(error)
                if cls == SCIENTIFIC:
                    m.record_result(raw, method=method, params=params, negative_detail=error, task_id=task_id,
                                    duration_s=time.time() - a0)
                    m.decision("computed_negative", entity, cond, obs_id=task_id)
                    raise
                incident = m.incident(entity, method, error, cls, task_id, attempt)
                incident_id = incident.id if incident else incident_id
                m.decision("failed_" + cls, entity, cond, task_id=task_id, attempt=attempt, error=error[:200])
                if cls == PERMANENT:
                    self._set_cooldown(f"{cond}/{entity}", pol.permanent_cooldown_s, error)
                    raise
                if attempt == pol.max_attempts:
                    self._set_cooldown(f"{cond}/{entity}", pol.transient_cooldown_s, error)
                    raise
                await asyncio.sleep(pol.backoff_s * 2 ** (attempt - 1))
                continue
            m.record_result(raw, method=method, params=params, value=value, task_id=task_id,
                            duration_s=time.time() - a0)
            m.decision("computed", entity, cond, obs_id=task_id, attempts=attempt)
            if incident_id:
                m.experience.recovery_episode(
                    incident_id, f"{entity}: succeeded at attempt {attempt} after transient failures",
                    by=m.agent, provenance=[prov_ref(task_id, **m.prov_ids())], agents=[m.agent])
            return value
        raise RuntimeError("unreachable")

    async def _keep_alive(self, task: str, token: str) -> None:
        while True:
            await asyncio.sleep(self.policy.claim_ttl_s / 3)
            try:
                if not self.renew(task, token):
                    return
            except Exception as exc:
                log.warning("claim renewal failed for %s: %s", task, exc)

    def get_or_compute_sync(self, raw: str, compute: Callable[[str], float], **kw) -> float:
        """For synchronous workflows (compute runs in a thread)."""
        async def acompute(task_id):
            return await asyncio.to_thread(compute, task_id)
        return asyncio.run(self.get_or_compute(raw, acompute, **kw))


def incident_signature(method: str, error: str, cls: str) -> str:
    return signature_of(cls, method, error_signature(error))
