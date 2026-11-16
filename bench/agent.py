"""
A generic LLM research agent, with the same loop as GAIED's XTBSimulationAgent:
plan -> propose -> (memory screen) -> run experiments -> conclude -> reflect/update -> propose ...

The agent keeps its own history (its results and conclusions), as GAIED's agents do. The memory
under test adds a briefing to each prompt, screens proposals before they run (when it coordinates),
receives provenance records of every experiment, and receives the agent's conclusions.
"""
from __future__ import annotations

import json
import re
import threading

from memlab.base import Memory
from memlab.llm import LLM, extract_json

SYSTEM = ("You are a research agent in a team exploring a design space by running experiments. "
          "You reason from the evidence you have, and you are precise about what it supports.")


class ResearchAgent:
    def __init__(self, name: str, seed: str, env, memory: Memory, llm: LLM, batch_size: int,
                 pull_rounds: int = 0, wait_timeout_s: float = 0.0, false_statements: list[str] | None = None,
                 stop: threading.Event | None = None, refill_rounds: int = 0,
                 max_conclusions: int | None = None) -> None:
        self.name, self.seed, self.env, self.memory, self.llm = name, seed, env, memory, llm
        self.batch_size, self.pull_rounds, self.wait_timeout_s = batch_size, pull_rounds, wait_timeout_s
        self.false_statements = false_statements      # set for an adversarial agent
        self.refill_rounds = refill_rounds
        self.max_conclusions = max_conclusions   # same for every condition (limits the memory's work)
        self.stop = stop or threading.Event()
        self.results: dict[str, float | str] = {}
        self.plan = ""
        self.conclusions: list[str] = []
        self.log: list[dict] = []
        self.memory.register_agent(name)

    # ------------------------------------------------------------------ prompt parts
    def _own(self) -> str:
        if not self.results:
            return "(none yet)"
        ok = sorted(((k, v) for k, v in self.results.items() if isinstance(v, float)), key=lambda x: -x[1])
        bad = [k for k, v in self.results.items() if not isinstance(v, float)]
        lines = [f"{k}: {v:.4g}" for k, v in ok] + [f"{k}: {v}" for k, v in self.results.items() if k in bad]
        return "\n".join(lines)

    def _memory_block(self, node: str, query: str, it: int) -> str:
        text = self.memory.briefing(self.name, node, query, it)
        return f"\n\nSHARED MEMORY OF THE TEAM:\n{text}" if text else ""

    def _ask(self, user: str) -> str:
        return self.llm.complete(SYSTEM, user, purpose="agent")

    # ------------------------------------------------------------------ steps
    def do_plan(self, it: int) -> None:
        goal = self.memory.adapter.goal.text
        if it == 0:
            user = (f"GOAL: {goal}\nYour starting design: {self.seed}\n\nYOUR RESULTS:\n{self._own()}"
                    f"{self._memory_block('plan', goal, it)}\n\nWrite a short plan for your next experiments.")
        else:
            user = (f"GOAL: {goal}\nYOUR PLAN: {self.plan}\n\nYOUR RESULTS:\n{self._own()}\n\nYOUR CONCLUSIONS:\n"
                    + "\n".join(self.conclusions[-3:]) + self._memory_block("update", self.plan, it) +
                    "\n\nCritique your plan in the light of the results, then write the updated plan.")
        self.plan = self._ask(user)
        self.memory.record_episode(self.name, "plan", self.plan, it)

    def do_propose(self, it: int, n: int | None = None, note: str = "") -> list[str]:
        n = n or self.batch_size
        base = (f"GOAL: {self.memory.adapter.goal.text}\nPLAN: {self.plan}\n\nYOUR RESULTS (do not repeat):\n"
                f"{self._own()}{self._memory_block('propose', self.plan, it)}\n\n{note}\n"
                f"Choose exactly {n} new designs to run now.")
        tools = self.memory.tool_specs() if self.pull_rounds else []
        if tools:
            base += ("\nYou may first query the shared memory. Tools: " + json.dumps(tools) +
                     '\nTo query, answer {"tool": "<name>", "args": {...}}. To decide, answer {"proposals": [...]}.')
        else:
            base += ' Answer only with JSON: {"proposals": ["<design>", ...]}'
        transcript = base
        for r in range(self.pull_rounds + 2):
            answer = self._ask(transcript)
            d = extract_json(answer)
            if isinstance(d, dict) and d.get("tool") and tools:
                if r < self.pull_rounds:
                    out = self.memory.call_tool(self.name, str(d["tool"]), d.get("args") or {})
                    transcript += f"\n\nYOU ASKED: {json.dumps(d)}\nMEMORY ANSWER:\n{out}"
                    if r + 1 < self.pull_rounds:
                        continue
                # no query left: the next answer must be the proposals
                transcript += ('\n\nYou cannot query the memory any more. Answer only with JSON: '
                               '{"proposals": ["<design>", ...]}')
                continue
            props = d.get("proposals") if isinstance(d, dict) else d if isinstance(d, list) else None
            if not props:
                props = re.findall(r"\b(?:[A-Z]\d+-)+[A-Z]\d+\b", answer)
            return [str(p) for p in props][:n]
        return []

    def do_experiments(self, it: int, proposals: list[str]) -> dict:
        stats = self._screen_and_run(it, proposals)
        for _ in range(self.refill_rounds):        # replace proposals the memory screened out
            missing = self.batch_size - stats["ran"]
            if missing <= 0 or not proposals:
                break
            proposals = self.do_propose(it, n=missing, note="Some of your proposals were already measured or "
                                        "are being measured by a teammate; their results are now in YOUR RESULTS.")
            more = self._screen_and_run(it, proposals)
            for k, v in more.items():
                stats[k] = stats.get(k, 0) + v
        return stats

    def _screen_and_run(self, it: int, proposals: list[str]) -> dict:
        decisions = self.memory.screen(self.name, proposals, it)
        stats = {"proposed": len(proposals), "ran": 0}
        waiting = []
        for d in decisions:
            stats[d.action] = stats.get(d.action, 0) + 1
            if d.action == "known":
                self.results[d.entity] = float(d.value)
            elif d.action == "failed_before":
                self.results[d.entity] = "failed"
            elif d.action == "in_progress":
                waiting.append(d.entity)
            elif d.action in ("run", "invalid"):
                rec = self.env.run(d.raw, self.name)
                self.memory.ingest(rec)
                stats["ran"] += 1
                v = (rec.get("generated") or {}).get("value")
                self.results[d.entity] = float(v) if v is not None else "failed"
        if waiting:
            got = self.memory.await_results(self.name, waiting, self.wait_timeout_s)
            for e, o in got.items():
                self.results[e] = float(o["value"]) if o["status"] == "ok" else "failed"
            stats["waited_ok"] = len(got)
        ran = [d.entity for d in decisions if d.action in ("run", "invalid")]
        if ran:   # what the agent observed, as text (a memory may keep it)
            self.memory.record_episode(self.name, "results", "Results of my experiments: " + "; ".join(
                f"{e}: {self.results[e]:.4g}" if isinstance(self.results[e], float) else f"{e}: failed"
                for e in ran), it)
        return stats

    def do_conclude(self, it: int) -> None:
        if self.false_statements is not None:   # adversarial agent: reports planted false statements
            text = "\n".join(f"- {s}" for s in self.false_statements)
        else:
            user = (f"PLAN: {self.plan}\n\nYOUR RESULTS:\n{self._own()}{self._memory_block('conclude', self.plan, it)}"
                    "\n\nWrite your new conclusions about how the design options affect the outcome. One stand-alone "
                    "statement per line, starting with '- '. State only what your evidence supports."
                    + (f" Write at most {self.max_conclusions}, the most important." if self.max_conclusions else ""))
            text = self._ask(user)
        self.conclusions.append(text)
        self.memory.submit_conclusion(self.name, text, it)

    def run(self, iterations: int, start: int = 0) -> None:
        for it in range(start, iterations):
            if self.stop.is_set():
                break
            self.do_plan(it)
            props = self.do_propose(it)
            stats = self.do_experiments(it, props)
            self.do_conclude(it)
            self.memory.end_iteration(self.name, it)
            self.log.append({"iteration": it, **stats})
