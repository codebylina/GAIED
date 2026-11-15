"""
Coordination: who is computing what right now.

A claim is a Redis key with an expiry (SET NX PX). Its value is a token "<agent>|<random>" that only
the claimer knows, so release and renewal are atomic compare-and-act operations (Lua): an agent whose
claim has expired can never delete or extend the claim another agent took in the meantime.

A claim says "being computed"; the result itself lives in the experimental memory. The expiry
covers crashed agents: a claim that is not renewed disappears and the work can be taken over.
"""
from __future__ import annotations

import uuid

_RELEASE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end
return 0
"""
_RENEW = """
if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('PEXPIRE', KEYS[1], ARGV[2]) end
return 0
"""


class Coordinator:
    def __init__(self, client, prefix: str, ttl_s: float = 300.0) -> None:
        self.r = client
        self.prefix = f"{prefix}:coord:task:"
        self.ttl_ms = int(ttl_s * 1000)
        self._release = client.register_script(_RELEASE)
        self._renew = client.register_script(_RENEW)

    def key(self, task: str) -> str:
        return self.prefix + task

    def claim(self, task: str, agent: str, ttl_s: float | None = None) -> str | None:
        """Reserve a task. Returns the claim token, or None if another agent holds it."""
        token = f"{agent}|{uuid.uuid4().hex[:12]}"
        ttl = int(ttl_s * 1000) if ttl_s else self.ttl_ms
        return token if self.r.set(self.key(task), token, nx=True, px=ttl) else None

    def release(self, task: str, token: str) -> bool:
        """Release a claim, only if `token` still owns it (atomic)."""
        return bool(self._release(keys=[self.key(task)], args=[token]))

    def renew(self, task: str, token: str, ttl_s: float | None = None) -> bool:
        """Extend a claim, only if `token` still owns it (atomic). False: the claim was lost."""
        ttl = int(ttl_s * 1000) if ttl_s else self.ttl_ms
        return bool(self._renew(keys=[self.key(task)], args=[token, ttl]))

    def holder(self, task: str) -> str | None:
        """Agent holding the claim on a task, or None."""
        token = self.r.get(self.key(task))
        return token.rsplit("|", 1)[0] if token else None

    def running(self, limit: int = 1000) -> dict[str, str]:
        """All claimed tasks -> agent (SCAN: cheap for the few hundred tasks of a campaign)."""
        out = {}
        for k in self.r.scan_iter(match=self.prefix + "*", count=500):
            token = self.r.get(k)
            if token:
                out[k[len(self.prefix):]] = token.rsplit("|", 1)[0]
                if len(out) >= limit:
                    break
        return out
