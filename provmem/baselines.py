"""
Baseline memories, reimplemented in the same harness so that only the memory differs between
conditions (same agents, same LLM, same retriever, same briefing budget):

  none        no shared memory: each agent sees only its own history (kept by the agent itself)
  shared_log  a blackboard: the team's results and every conclusion of every agent, most recent
              first, cut to the budget (the "full shared context" baseline)
  rag         retrieval-augmented episodic memory: conclusions are stored as items and the most
              relevant ones are retrieved for each prompt (MemoryBank / Generative Agents style)
  mem0        LLM-managed memory: an LLM extracts facts from each conclusion and decides, against
              similar stored memories, to ADD, UPDATE, DELETE or do nothing (Mem0, Chhikara et al. 2025)

Every shared condition shows the team's measured results in the same way; conditions differ in how
they manage knowledge and whether they coordinate. Only ProvMem has pre-action coordination.
"""
from __future__ import annotations

import json

from .base import Memory, select_within_budget
from .llm import extract_json
from .memory import split_claims


class NoMemory(Memory):
    name = "none"


class _BriefingMixin(Memory):
    def _render(self, agent, node, iteration, knowledge: list[tuple[str, str]], title: str) -> str:
        budget = self.cfg.briefing_budget_chars
        if budget <= 0:
            return ""
        head_text, _ = self.fit(self.goal_lines(), budget)
        res = [(i, "fact", True, l) for i, l in self.team_results_lines()]
        fails = [(i, "fact", True, l) for i, l in self.failed_lines()]
        know = [(i, "claim", True, f"- {t}") for i, t in knowledge]
        sections = [("MEASURED RESULTS (best first):", res), ("FAILED:", fails), (title, know)]
        keep = select_within_budget(head_text, sections, lambda c: c[3], budget)
        lines, items = [head_text], []
        for t, sec in sections:
            sel = [c for c in sec if id(c) in keep]
            if sel:
                lines.append(t)
                for c in sel:
                    lines.append(c[3])
                    items.append({"id": c[0], "kind": c[1], "asserted": c[2], "text": c[3].lstrip("- ")})
        text = "\n".join(lines)
        self.log_briefing(agent, node, iteration, items, text)
        return text

    def _items(self) -> list[dict]:
        return self.store.all("items")

    def presented_knowledge(self):
        shown = {i["id"] for b in self.store.read_log("briefings") for i in b["items"] if i["kind"] == "claim"}
        return [{"id": it["id"], "text": it["text"], "author": it.get("author"), "final_layer": it.get("state", "stored"),
                 "history": it.get("history", [])} for it in self.store.all("items") if it["id"] in shown]


class SharedLog(_BriefingMixin):
    name = "shared_log"

    def submit_conclusion(self, agent, text, iteration):
        self.record_episode(agent, "conclude", text, iteration)
        ids = []
        for s in split_claims(text):
            iid = self.new_id("m")
            self.store.put("items", iid, {"id": iid, "text": s, "author": agent, "t": self.clock()})
            ids.append(iid)
        return ids

    def briefing(self, agent, node, query, iteration):
        items = sorted(self._items(), key=lambda x: -x["t"])
        return self._render(agent, node, iteration, [(i["id"], f"{i['text']} (by {i['author']})") for i in items],
                            "CONCLUSIONS OF THE TEAM (most recent first):")


class RAGMemory(SharedLog):
    name = "rag"

    def briefing(self, agent, node, query, iteration):
        items = self._items()
        order = self.retriever.search(query or self.adapter.goal.text, [(i["id"], i["text"]) for i in items], len(items))
        by = {i["id"]: i for i in items}
        return self._render(agent, node, iteration, [(i, f"{by[i]['text']} (by {by[i]['author']})") for i in order],
                            "RELEVANT MEMORIES:")


MEM0_EXTRACT = """Extract the facts worth remembering from a research agent's conclusions, for a memory
shared by a team. Return only JSON: {"facts": ["self-contained fact", ...]} (an empty list if none)."""

MEM0_UPDATE = """You manage a memory shared by a team of research agents. Compare a new fact with the existing
memories and choose one operation:
- ADD: the fact is new information;
- UPDATE: it refines or corrects an existing memory (give the id and the merged text);
- DELETE: it shows that an existing memory is wrong (give the id); the new fact is then added;
- NONE: it is already known.
Return only JSON: {"op": "ADD|UPDATE|DELETE|NONE", "id": "<memory id or null>", "text": "<text for ADD/UPDATE>"}"""


class Mem0Memory(RAGMemory):
    name = "mem0"

    def submit_conclusion(self, agent, text, iteration):
        self.record_episode(agent, "conclude", text, iteration)
        self.store.append("pending", {"agent": agent, "text": text, "t": self.clock()})
        return []

    def maintain(self):
        start = int((self.store.get("meta", "pending_cursor") or {}).get("i", 0))
        queue = self.store.read_log("pending", start)
        self.store.put("meta", "pending_cursor", {"i": start + len(queue)})
        k = int(self.cfg.extra.get("related_k", 0))
        for item in queue:
            facts = (extract_json(self.llm.complete(MEM0_EXTRACT, item["text"], purpose="memory")) or {}).get("facts") or []
            for fact in [str(f) for f in facts if str(f).strip()]:
                live = [m for m in self._items() if m.get("state", "live") == "live"]
                ids = self.retriever.search(fact, [(m["id"], m["text"]) for m in live], k)
                by = {m["id"]: m for m in live}
                related = "\n".join(f"{i}: {by[i]['text']}" for i in ids) or "(none)"
                d = extract_json(self.llm.complete(MEM0_UPDATE, f"EXISTING MEMORIES:\n{related}\n\nNEW FACT: {fact}",
                                                   purpose="memory")) or {}
                op, tid = str(d.get("op", "ADD")).upper(), str(d.get("id"))
                self.store.append("mem_ops", {"t": self.clock(), "op": op, "id": tid, "fact": fact, "agent": item["agent"]})
                if op == "UPDATE" and tid in by:
                    m = by[tid]
                    m.setdefault("history", []).append({"t": self.clock(), "text": m["text"]})
                    m["text"] = str(d.get("text") or fact)
                    self.store.put("items", tid, m)
                elif op == "NONE":
                    continue
                else:
                    if op == "DELETE" and tid in by:
                        m = by[tid]
                        m["state"] = "deleted"
                        m.setdefault("history", []).append({"t": self.clock(), "deleted_by": fact})
                        self.store.put("items", tid, m)
                    iid = self.new_id("m")
                    self.store.put("items", iid, {"id": iid, "text": fact, "author": item["agent"], "t": self.clock(),
                                                  "state": "live"})
        return len(queue)

    def _items(self):
        return [m for m in self.store.all("items") if m.get("state", "live") == "live"]

    def export_extra(self):
        return {"mem_ops": self.store.read_log("mem_ops"), "items": self.store.all("items")}


REGISTRY = {"none": NoMemory, "shared_log": SharedLog, "rag": RAGMemory, "mem0": Mem0Memory}
