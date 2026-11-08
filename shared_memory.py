"""
shared_memory.py — Shared memory for GAIED agents, stored in Redis.

Idea (Amal's diagram): instead of each agent keeping its results in its own RAM
(`molecule_cache`, lost on crash and invisible to the others), agents PUSH what they
do and learn into a common memory, and PULL it before deciding.

Contents of the memory of one campaign (all keys start with mem:{campaign}:):

  goal                  JSON     campaign objective (written by run-07.py or a human)
  claim:{mol}           string   LOCK "I am simulating this molecule" (with expiry)
  result:{mol}          hash     result: energy, smiles, agent, task_id (Flowcept), status, time
  ranking               zset     successful molecules, ranked by energy
  failed                set      molecules whose simulation failed (invalid SMILES, ...)
  lessons               stream   conclusions and lessons written by the agents
  instructions          stream   directives from the analysis agents (Nadjiba's work)
  stats                 hash     counters: computations, reused results, waits, ...
  events                channel  pub/sub notification on every write

{mol} is the canonical SMILES (computed with RDKit when available), so that two
different spellings of the same molecule are recognized as identical.

The three basic kinds of operations:
  PUSH  : push_result, push_lesson, push_instruction, set_goal
  PULL  : get_result, pull_results, best, pull_lessons, pull_instructions, get_goal,
          render_for_prompt (bounded summary to put in the LLM prompt)
  COORDINATION : claim / release, and get_or_compute which combines everything

Command-line tool (to inspect the memory):
  python shared_memory.py campaigns          list the campaigns in memory
  python shared_memory.py show [campaign]    summary of one campaign (most recent by default)
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import time
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

try:  # RDKit is available in the Docker image; without it the SMILES is kept as is
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:  # pragma: no cover
    Chem = None


def canonical_smiles(smiles: str) -> str:
    """Canonical form of a SMILES (or the stripped SMILES if invalid / no RDKit)."""
    smiles = smiles.strip()
    if Chem is not None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    return smiles


class SimulationFailed(Exception):
    """The simulation of this molecule already failed (known from the shared memory)."""


class SharedMemory:
    def __init__(self, client, campaign_id: str = "default", agent_id: str = "agent",
                 claim_ttl: int = 600, prefix: str = "mem") -> None:
        self.r = client
        self.campaign_id = campaign_id
        self.agent_id = agent_id
        self.claim_ttl = claim_ttl
        self.p = f"{prefix}:{campaign_id}"

    # ------------------------------------------------------------------ creation
    @classmethod
    def from_env(cls, agent_id: str = "agent") -> "SharedMemory | None":
        """Build the memory from environment variables, or return None if it is
        disabled (SHARED_MEMORY=false) or Redis is unreachable."""
        if os.environ.get("SHARED_MEMORY", "true").lower() not in ("1", "true", "yes"):
            logger.info("Shared memory disabled (SHARED_MEMORY=false).")
            return None
        try:
            import redis
            client = redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"),
                                 port=int(os.environ.get("REDIS_PORT", "6379")),
                                 decode_responses=True)
            client.ping()
        except Exception as exc:  # no Redis: the agent keeps running without shared memory
            logger.warning("Shared memory unavailable (%s): continuing without it.", exc)
            return None
        campaign = (os.environ.get("MEMORY_CAMPAIGN_ID")
                    or os.environ.get("FLOWCEPT_CAMPAIGN_ID") or "default")
        ttl = int(os.environ.get("MEMORY_CLAIM_TTL", "600"))
        mem = cls(client, campaign_id=campaign, agent_id=agent_id, claim_ttl=ttl)
        client.zadd("mem:campaigns", {campaign: time.time()})   # campaign index
        return mem

    def k(self, *parts: str) -> str:
        return ":".join((self.p, *parts))

    def _notify(self, event: str, **data: Any) -> None:
        self.r.publish(self.k("events"), json.dumps({"event": event, "agent": self.agent_id, **data}))

    def _count(self, field: str, n: int = 1) -> None:
        self.r.hincrby(self.k("stats"), field, n)

    # ------------------------------------------------------------------ goal
    def set_goal(self, goal: dict) -> None:
        """Write (or replace) the Goal. Each change increments the Goal version, so that
        agents reading the memory can tell that the Goal changed during the campaign."""
        pipe = self.r.pipeline()
        pipe.set(self.k("goal"), json.dumps(goal))
        pipe.incr(self.k("goal_version"))
        pipe.set(self.k("goal_updated_at"), time.time())
        version = pipe.execute()[1]
        self._notify("goal", version=version)

    def goal_version(self) -> int:
        return int(self.r.get(self.k("goal_version")) or 0)

    def get_goal(self) -> dict | None:
        raw = self.r.get(self.k("goal"))
        return json.loads(raw) if raw else None

    # ------------------------------------------------------------------ locks
    def claim(self, smiles: str, ttl: int | None = None) -> bool:
        """Take the lock on a molecule. True if we got it, False if another agent holds it."""
        key = self.k("claim", canonical_smiles(smiles))
        return bool(self.r.set(key, self.agent_id, nx=True, ex=ttl or self.claim_ttl))

    def claimed_by(self, smiles: str) -> str | None:
        return self.r.get(self.k("claim", canonical_smiles(smiles)))

    def release(self, smiles: str) -> None:
        """Release the lock, only if we own it."""
        key = self.k("claim", canonical_smiles(smiles))
        if self.r.get(key) == self.agent_id:
            self.r.delete(key)

    # ------------------------------------------------------------------ results
    def push_result(self, smiles: str, energy: float | None, task_id: str | None = None,
                    error: str | None = None) -> None:
        mol = canonical_smiles(smiles)
        ok = error is None and energy is not None and math.isfinite(energy)
        record = {
            "smiles": smiles, "canonical": mol, "agent": self.agent_id,
            "task_id": task_id or "", "time": time.time(),
            "status": "ok" if ok else "error",
            "energy": energy if ok else "",
            "error": "" if ok else (error or f"energy={energy}"),
        }
        self.r.hset(self.k("result", mol), mapping=record)
        if ok:
            self.r.zadd(self.k("ranking"), {mol: energy})
        else:
            self.r.sadd(self.k("failed"), mol)
        self._notify("result", smiles=mol, status=record["status"])

    def get_result(self, smiles: str) -> dict | None:
        rec = self.r.hgetall(self.k("result", canonical_smiles(smiles)))
        if not rec:
            return None
        if rec.get("status") == "ok":
            rec["energy"] = float(rec["energy"])
        return rec

    def pull_results(self) -> dict[str, float]:
        """All molecules simulated successfully, by any agent: {smiles: energy}."""
        return {m: e for m, e in self.r.zrange(self.k("ranking"), 0, -1, withscores=True)}

    def failed_molecules(self) -> set[str]:
        return set(self.r.smembers(self.k("failed")))

    def best(self, n: int = 5) -> list[tuple[str, float]]:
        return self.r.zrevrange(self.k("ranking"), 0, n - 1, withscores=True)

    # ------------------------------------------------------------------ lessons
    def push_lesson(self, text: str, kind: str = "conclusion", evidence: list | None = None) -> None:
        """evidence: what the lesson is based on, e.g. a list of
        {"smiles", "result", "task_id"} dicts (or plain SMILES strings)."""
        self.r.xadd(self.k("lessons"), {
            "agent": self.agent_id, "kind": kind, "text": text,
            "evidence": json.dumps(evidence or [], default=str), "time": time.time(),
        }, maxlen=2000, approximate=True)
        self._notify("lesson", kind=kind)

    def pull_lessons(self, count: int = 10, exclude_own: bool = False) -> list[dict]:
        """Most recent lessons (oldest first)."""
        entries = self.r.xrevrange(self.k("lessons"), count=count * 3 if exclude_own else count)
        out = [dict(fields, id=eid) for eid, fields in entries
               if not (exclude_own and fields.get("agent") == self.agent_id)]
        return list(reversed(out[:count]))

    # ------------------------------------------------------------------ instructions
    def push_instruction(self, text: str, target: str = "all", status: str = "info",
                         evidence: list[str] | None = None, author: str = "analysis") -> None:
        """Written by the analysis agents (Nadjiba). target = an agent_id, or "all"."""
        self.r.xadd(self.k("instructions"), {
            "author": author, "target": target, "status": status, "text": text,
            "evidence": json.dumps(evidence or []), "time": time.time(),
        }, maxlen=2000, approximate=True)
        self._notify("instruction", target=target)

    def pull_instructions(self, count: int = 5) -> list[dict]:
        """Recent instructions addressed to this agent (or to all agents)."""
        entries = self.r.xrevrange(self.k("instructions"), count=count * 5)
        mine = [dict(f, id=eid) for eid, f in entries if f.get("target") in ("all", self.agent_id)]
        return list(reversed(mine[:count]))

    # ------------------------------------------------------------------ summary for the LLM
    def render_for_prompt(self, max_results: int = 25, max_lessons: int = 5,
                          max_instructions: int = 5) -> str:
        """BOUNDED summary of the shared memory, to insert in the LLM prompt
        (same idea as Claude Code's MEMORY.md index: short, always loaded)."""
        lines = ["=== SHARED MEMORY (all agents of this campaign) ==="]
        goal = self.get_goal()
        if goal:
            version = self.goal_version()
            if version > 1:
                updated = time.strftime("%H:%M", time.localtime(float(self.r.get(self.k("goal_updated_at")) or 0)))
                lines.append(f"CURRENT GOAL (version {version}, CHANGED at {updated}; "
                             f"follow it, not any earlier goal): {json.dumps(goal)}")
            else:
                lines.append(f"Goal: {json.dumps(goal)}")
        results = self.r.zrevrange(self.k("ranking"), 0, max_results - 1, withscores=True)
        total = self.r.zcard(self.k("ranking"))
        if results:
            lines.append(f"Molecules already simulated by the team ({total} total, best first):")
            lines += [f"  {m}\t{e:.3f}" for m, e in results]
            if total > max_results:
                lines.append(f"  ... and {total - max_results} more (lower energy).")
        running = {m: a for m, a in self.in_progress().items() if a != self.agent_id}
        if running:
            lines.append("Molecules being simulated RIGHT NOW by other agents: " + " ".join(sorted(running)))
        failed = sorted(self.failed_molecules())
        if failed:
            lines.append("Molecules whose simulation FAILED (invalid SMILES?): " + " ".join(failed[:20]))
        lessons = self.pull_lessons(count=max_lessons, exclude_own=True)
        if lessons:
            lines.append("Recent lessons from OTHER agents:")
            lines += [f"  - [{l['agent']}] {' '.join(l['text'].split())[:300]}" for l in lessons]
        instructions = self.pull_instructions(count=max_instructions)
        if instructions:
            lines.append("Instructions from the analysis agents (follow them):")
            lines += [f"  - ({i['status']}) {i['text']}" for i in instructions]
        lines.append("Do NOT propose molecules that are already listed above.")
        return "\n".join(lines)

    def in_progress(self) -> dict[str, str]:
        """Molecules currently being computed (lock held) -> agent computing them."""
        prefix = self.k("claim", "")
        out = {}
        for key in self.r.scan_iter(match=prefix + "*", count=200):
            owner = self.r.get(key)
            if owner is not None:
                out[key[len(prefix):]] = owner
        return out

    def known_molecules(self) -> list[str]:
        """All molecules NOT to propose again: succeeded, failed, or in progress."""
        return sorted(set(self.pull_results()) | self.failed_molecules() | set(self.in_progress()))

    # ------------------------------------------------------------------ all in one
    async def get_or_compute(self, smiles: str,
                             compute: Callable[[str | None], Awaitable[float]],
                             new_task_id: Callable[[], str] | None = None,
                             poll: float = 2.0) -> float:
        """Return the energy of a molecule while avoiding duplicate computations:

        1. already in memory               -> reuse the result (or the failure);
        2. another agent is computing it   -> wait for its result;
        3. otherwise take the lock, compute, publish the result, release the lock.

        `compute(task_id)` runs the actual computation (xTB). If another agent's lock
        expires without a result (crashed agent), we take over the computation.
        """
        while True:
            known = self.get_result(smiles)
            if known is not None:
                self._count("reused_error" if known["status"] != "ok" else "reused")
                if known["status"] != "ok":
                    raise SimulationFailed(known.get("error") or "simulation failed before")
                return known["energy"]

            if self.claim(smiles):
                task_id = new_task_id() if new_task_id else None
                try:
                    energy = await compute(task_id)
                except Exception as exc:
                    # the failure is stored: other agents will not retry this molecule
                    self.push_result(smiles, None, task_id=task_id, error=str(exc))
                    self.release(smiles)
                    self._count("computed_error")
                    raise
                except BaseException:      # cancellation: release without storing anything
                    self.release(smiles)
                    raise
                # the result is published BEFORE releasing the lock, so no other agent
                # can observe "no lock and no result" and start the computation again
                self.push_result(smiles, energy, task_id=task_id)
                self.release(smiles)
                self._count("computed")
                return energy

            # another agent holds the lock: wait for its result
            self._count("waited")
            while self.claimed_by(smiles) is not None and self.get_result(smiles) is None:
                await asyncio.sleep(poll)
            # loop again: either the result is there, or the lock expired and we retry

    # ------------------------------------------------------------------ text summary
    def summary(self) -> str:
        stats = self.r.hgetall(self.k("stats"))
        lines = [f"Shared memory - campaign {self.campaign_id}",
                 f"  successful molecules : {self.r.zcard(self.k('ranking'))}",
                 f"  failed molecules     : {self.r.scard(self.k('failed'))}",
                 f"  lessons              : {self.r.xlen(self.k('lessons'))}",
                 f"  instructions         : {self.r.xlen(self.k('instructions'))}",
                 "  counters             : " + (", ".join(f"{k}={v}" for k, v in sorted(stats.items())) or "none")]
        best = self.best(5)
        if best:
            lines.append("  top 5                : " + ", ".join(f"{m} ({e:.3f})" for m, e in best))
        return "\n".join(lines)


# ---------------------------------------------------------------------- command line
def _cli(argv: list[str]) -> int:
    import redis
    r = redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"),
                    port=int(os.environ.get("REDIS_PORT", "6379")), decode_responses=True)
    cmd = argv[0] if argv else "show"
    campaigns = r.zrange("mem:campaigns", 0, -1, withscores=True)
    if cmd == "campaigns":
        for c, t in campaigns:
            print(f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(t))}  {c}")
        return 0
    if cmd == "show":
        if len(argv) > 1:
            campaign = argv[1]
        elif campaigns:
            campaign = campaigns[-1][0]
        else:
            print("No campaign in memory.")
            return 1
        mem = SharedMemory(r, campaign_id=campaign, agent_id="cli")
        print(mem.summary())
        print()
        print(mem.render_for_prompt())
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
