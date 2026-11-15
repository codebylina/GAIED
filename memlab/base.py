"""
Interface shared by every memory condition, so that agents, the benchmark and the analysis treat
all conditions identically. Also the logging that the evaluation relies on: what each briefing
showed (item by item), events, and the decisions taken before experiments.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .adapter import DomainAdapter
from .llm import LLM
from .store import Store


@dataclass
class MemoryConfig:
    """Knobs of the harness. The briefing budget is set by the experiment configuration and held
    equal across conditions."""
    briefing_budget_chars: int = 0          # 0 = no memory text in the prompts
    extra: dict = field(default_factory=dict)


@dataclass
class Decision:
    raw: str
    entity: str
    action: str                 # run | known | failed_before | in_progress | invalid | duplicate_in_batch
    value: float | None = None
    owner: str | None = None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class Memory:
    name = "abstract"

    def __init__(self, store: Store, adapter: DomainAdapter, cfg: MemoryConfig, llm: LLM | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.store, self.adapter, self.cfg, self.llm = store, adapter, cfg, llm
        self.clock = clock
        self.read_sink: Callable[[str, str, int, list[str]], None] | None = None  # provenance of reads

    # ------------------------------------------------------------------ common bookkeeping
    def new_id(self, prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex[:12]}"

    def register_agent(self, agent: str) -> None:
        if not self.store.get("agents", agent):
            self.store.put("agents", agent, {"id": agent, "t": self.clock()})

    def heartbeat(self, agent: str) -> None:
        pass

    def log_briefing(self, agent: str, node: str, iteration: int, items: list[dict], text: str) -> None:
        """items: [{"id", "kind", "asserted": bool, "text"}] — what the agent was shown, item by item.
        'asserted' marks items presented as true knowledge (used to measure exposure to false beliefs)."""
        self.store.append("briefings", {"agent": agent, "node": node, "iteration": iteration,
                                        "t": self.clock(), "chars": len(text), "items": items})
        if self.read_sink and items:
            try:
                self.read_sink(agent, node, iteration, [i["id"] for i in items])
            except Exception:
                pass

    def log_decisions(self, agent: str, iteration: int, decisions: list[Decision]) -> None:
        self.store.append("decisions", {"agent": agent, "iteration": iteration, "t": self.clock(),
                                        "decisions": [d.as_dict() for d in decisions]})

    def fit(self, lines: list[str], budget: int) -> tuple[str, int]:
        out, used = [], 0
        for i, line in enumerate(lines):
            if used + len(line) + 1 > budget:
                return "\n".join(out), i
            out.append(line)
            used += len(line) + 1
        return "\n".join(out), len(lines)

    # ------------------------------------------------------------------ interface
    def ingest(self, record: dict) -> dict | None:
        """A provenance record emitted by the workflow. Every condition keeps the observations
        (they are what the evaluation is computed from); conditions differ in what they show."""
        obs = self.adapter.observation_from_record(record)
        if not obs or not obs.get("id"):
            return None
        if self.store.get("obs", obs["id"]):
            return None
        obs["descriptors"] = self.adapter.descriptors(obs["entity"]) if obs.get("valid_entity") else {}
        obs["violations"] = self.adapter.violations(obs["entity"]) if obs.get("valid_entity") else []
        self.store.put("obs", obs["id"], obs)
        self.store.append("obs_log", {"id": obs["id"]})
        self.on_observation(obs)
        return obs

    def on_observation(self, obs: dict) -> None:
        pass

    def record_episode(self, agent: str, node: str, text: str, iteration: int) -> None:
        self.store.append("episodes", {"agent": agent, "node": node, "iteration": iteration,
                                       "t": self.clock(), "text": text})

    def submit_conclusion(self, agent: str, text: str, iteration: int) -> list[str]:
        self.record_episode(agent, "conclude", text, iteration)
        return []

    def briefing(self, agent: str, node: str, query: str, iteration: int) -> str:
        return ""

    def screen(self, agent: str, proposals: list[str], iteration: int) -> list[Decision]:
        out, seen = [], set()
        for raw in proposals:
            key = self.adapter.canonical(raw) or f"raw:{raw}"
            action = "duplicate_in_batch" if key in seen else "run"
            seen.add(key)
            out.append(Decision(raw, key, action))
        self.log_decisions(agent, iteration, out)
        return out

    def await_results(self, agent: str, entities: list[str], timeout_s: float) -> dict[str, dict]:
        return {}

    def release(self, agent: str, entity: str) -> None:
        pass

    def tool_specs(self) -> list[dict]:
        return []

    def call_tool(self, agent: str, name: str, args: dict) -> str:
        return "unknown tool"

    def put_directive(self, author: str, text: str, targets: list[str] | str = "all",
                      lifetime: str = "standing", expires_at: float | None = None) -> str:
        return ""

    def maintain(self) -> int:
        """Background work (judging, consolidation). Returns the number of work items done."""
        return 0

    def end_iteration(self, agent: str, iteration: int) -> None:
        pass

    def presented_knowledge(self) -> list[dict]:
        """Every knowledge item the memory has ever shown as true: [{"id", "text", "author"}]."""
        return []

    def export(self) -> dict:
        s = self.store
        return {"memory": self.name, "config": self.cfg.__dict__,
                "observations": sorted(s.all("obs"), key=lambda o: o.get("t", 0)),
                "briefings": s.read_log("briefings"), "decisions": s.read_log("decisions"),
                "episodes": s.read_log("episodes"), "events": s.read_log("events"),
                "knowledge": self.presented_knowledge(), "extra": self.export_extra(),
                "llm_usage": self.llm.usage.snapshot() if self.llm else {}}

    def export_extra(self) -> dict:
        return {}

    # ------------------------------------------------------------------ helpers for briefings
    def team_results_lines(self, limit_hint: int | None = None) -> list[tuple[str, str]]:
        """(id, line) of measured results, best first."""
        obj = self.adapter.goal.objective
        obs = [o for o in self.store.all("obs") if o["status"] == "ok"]
        best: dict[str, dict] = {}
        for o in obs:
            if o["entity"] not in best or obj.better(o["value"], best[o["entity"]]["value"]):
                best[o["entity"]] = o
        ranked = sorted(best.values(), key=lambda o: o["value"], reverse=obj.direction == "max")
        return [(o["id"], f"{o['entity']}: {o['value']:.4g} (by {o.get('agent')})") for o in ranked]

    def failed_lines(self) -> list[tuple[str, str]]:
        fails: dict[str, dict] = {}
        for o in self.store.all("obs"):
            if o["status"] == "failed":
                fails.setdefault(o["entity"], o)
        return [(o["id"], f"{o['entity']}: failed") for o in sorted(fails.values(), key=lambda o: -o["t"])]

    def goal_lines(self) -> list[str]:
        g = self.adapter.goal
        lines = [f"GOAL: {g.text}"]
        lines += [f"- constraint: {c}" for c in g.constraints]
        return lines


def select_within_budget(head_text: str, sections: list[tuple[str, list]], line_of, budget: int) -> set[int]:
    """Choose section items round-robin (best of each section first) until the rendered text —
    head, section titles and item lines — would exceed the budget. Returns ids (id()) of kept items."""
    used = len(head_text)
    titled: set[int] = set()
    keep: set[int] = set()
    order = interleave([[(k, it) for it in sec] for k, (_, sec) in enumerate(sections)])
    for k, it in order:
        cost = len(line_of(it)) + 1 + (0 if k in titled else len(sections[k][0]) + 1)
        if used + cost > budget:
            break
        used += cost
        titled.add(k)
        keep.add(id(it))
    return keep


def interleave(sections: list[list[Any]]) -> list[Any]:
    """Round-robin merge of ranked sections, so that a budget cut keeps the best of each section."""
    out, i = [], 0
    while any(i < len(s) for s in sections):
        for s in sections:
            if i < len(s):
                out.append(s[i])
        i += 1
    return out
