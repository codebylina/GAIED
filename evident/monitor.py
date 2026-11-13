"""
Real-time rule-based analysis agent (reference implementation of the control loop).

It reads the telemetry stream as events happen (XREAD BLOCK) and sends directives:
  off_goal        an agent's current batch contains >= MONITOR_VIOLATIONS proposals that
                  break a hard Goal constraint -> STOP limited to those molecules (scope
                  "molecules"): they are not computed, the valid ones go on, and the
                  reason is shown first in the agent's next prompts.
  invalid_repeat  an agent proposes a molecule already known to be invalid -> advice
  rejected_claim  an agent asserts a claim that the data already rejects -> advice
  stagnation      no improvement of the agent's best for MONITOR_STAGNATION iterations
                  -> advice with the currently verified knowledge

It is deliberately simple: it defines the interface (telemetry in, directives out) that
the provenance-based analysis agents (Flowcept "interception" channel) can plug into.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time

from .memory import EvidentMemory

logger = logging.getLogger(__name__)


class RuleMonitor:
    def __init__(self, memory: EvidentMemory, violations_for_stop: int | None = None,
                 stagnation: int | None = None, block_ms: int = 500):
        self.m = memory
        self.violations_for_stop = violations_for_stop or int(os.environ.get("MONITOR_VIOLATIONS", "2"))
        self.stagnation = stagnation or int(os.environ.get("MONITOR_STAGNATION", "3"))
        self.block_ms = block_ms
        self.cursor = "0-0"
        self._stopped: set[tuple[str, int]] = set()     # (agent, iteration) already stopped
        self._advised: set[tuple] = set()
        self.sent = 0

    def handle(self, ev: dict) -> None:
        agent, it, t = ev["agent"], ev["it"], ev["t"]
        if ev["e"] == "proposals":
            bad = [p for p in ev["items"] if p["violations"]]
            if len(bad) >= self.violations_for_stop and (agent, it) not in self._stopped:
                self._stopped.add((agent, it))
                details = "; ".join(f"{p['smiles']} ({', '.join(p['violations'])})" for p in bad[:8])
                self._send(agent, "stop", "off_goal", t,
                           f"{len(bad)} of your proposals break the Goal's hard constraints "
                           f"and were NOT computed: {details}. Every new proposal must have "
                           f"{self._constraints_text()}.",
                           smiles=[p["smiles"] for p in bad], scope="molecules", iteration=it)
            for p in ev["items"]:
                if p["team_known"] == "failed" and (agent, "inv", p["smiles"]) not in self._advised:
                    self._advised.add((agent, "inv", p["smiles"]))
                    self._send(agent, "advice", "invalid_repeat", t,
                               f"{p['raw']} is already known to be invalid; check SMILES syntax "
                               "before proposing.")
        elif ev["e"] == "claim_asserted" and ev.get("status") == "rejected":
            key = (agent, "rej", ev["id"])
            if key not in self._advised:
                self._advised.add(key)
                e = ev.get("evidence") or {}
                self._send(agent, "advice", "rejected_claim", t,
                           f"Your claim '{ev['id']}' is contradicted by the team data "
                           f"(effect {e.get('effect')}, n={e.get('n_with')}/{e.get('n_without')}). "
                           "Revise your hypothesis.")
        elif ev["e"] == "iteration_end":
            if (ev.get("no_improve") or 0) >= self.stagnation:
                key = (agent, "stag", it)
                if key not in self._advised:
                    self._advised.add(key)
                    known = [c["id"] for c in self.m.claims().values()
                             if c["status"] in ("supported", "established")]
                    self._send(agent, "advice", "stagnation", t,
                               f"Your best value has not improved for {ev['no_improve']} "
                               f"iterations (team best {ev.get('team_best')}). Change chemical "
                               "family" + (f"; verified trends: {', '.join(known)}." if known else "."))

    def _constraints_text(self) -> str:
        c = self.m.constraints()
        parts = []
        if c.get("max_heavy_atoms"):
            parts.append(f"at most {c['max_heavy_atoms']} heavy (non-hydrogen) atoms")
        if c.get("allowed_elements"):
            parts.append(f"only the elements {', '.join(c['allowed_elements'])}")
        return " and ".join(parts) or "a valid structure"

    def _send(self, agent, kind, rule, trigger_t, reason, **data) -> None:
        self.m.send_directive(agent, kind, reason, rule=rule, trigger_t=trigger_t, **data)
        self.sent += 1
        logger.warning("[monitor] %s -> %s (%s): %s", kind.upper(), agent, rule, reason[:120])

    def poll_once(self, block: bool = True) -> int:
        resp = self.m.r.xread({self.m.k("telemetry"): self.cursor}, count=200,
                              block=self.block_ms if block else None)
        n = 0
        for _, entries in resp or []:
            for entry_id, f in entries:
                self.cursor = entry_id
                if f["agent"] == self.m.agent_id:
                    continue
                ev = {"e": f["e"], "agent": f["agent"], "it": int(f["it"]), "t": float(f["t"]),
                      **json.loads(f["d"])}
                try:
                    self.handle(ev)
                except Exception as exc:
                    logger.warning("[monitor] rule failed on %s: %s", ev["e"], exc)
                n += 1
        return n

    async def run(self, stop: asyncio.Event) -> None:
        logger.warning("[monitor] real-time monitor started on campaign %s", self.m.campaign_id)
        while not stop.is_set():
            await asyncio.to_thread(self.poll_once, True)
        self.poll_once(block=False)


def start_in_background(stop: asyncio.Event) -> asyncio.Task | None:
    """Start the monitor if MONITOR=true (called by run-07.py)."""
    if os.environ.get("MONITOR", "false").lower() not in ("1", "true", "yes", "on"):
        return None
    mem = EvidentMemory.from_env(agent_id="monitor")
    if mem is None:
        return None
    return asyncio.create_task(RuleMonitor(mem).run(stop))


if __name__ == "__main__":  # standalone: python -m evident.monitor (same environment variables)
    logging.basicConfig(level=logging.INFO)
    mem = EvidentMemory.from_env(agent_id="monitor")
    mon = RuleMonitor(mem)
    while True:
        mon.poll_once()
        time.sleep(0.01)
