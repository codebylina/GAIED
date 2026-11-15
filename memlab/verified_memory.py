"""
Mem0 with the knowledge verification layer on top (memlab/verifier.py).

Mem0 is used exactly as in the baseline: the same writes, the same search, the same number of results,
the same prompt budget. The layer only (1) checks every memory Mem0 creates against the measurements,
and (2) presents the memories Mem0 retrieves according to their status:

    VERIFIED AGAINST MEASUREMENTS   verified facts, supported and established generalizations (shown as true)
    TO TEST                         hypotheses: supported only by the evidence that suggested them
    NOT CHECKED YET                 memories the layer has not processed yet
    NOT BACKED BY ANY MEASUREMENT   facts about measurements that do not exist; untestable memories
    KNOWN FALSE                     memories the measurements contradict

Agents that were shown a memory as true and whose status later dropped get a notice in their next prompt.
"""
from __future__ import annotations

from .base import select_within_budget
from .mem0_memory import LocalMem0, Mem0Memory
from .verifier import ASSERTED, Verifier

SECTIONS = [
    ("VERIFIED AGAINST MEASUREMENTS:", ("verified", "established", "supported", "accepted"), True),
    ("TO TEST (hypotheses, not confirmed by new measurements yet):", ("hypothesis",), None),
    ("NOT CHECKED YET:", ("pending",), None),
    ("NOT BACKED BY ANY MEASUREMENT:", ("unsupported", "unverifiable"), None),
    ("KNOWN FALSE (contradicted by measurements):", ("refuted",), False),
]


class VerifiedMem0Memory(Mem0Memory):
    name = "mem0_verified"

    def __init__(self, *args, verifier: Verifier, **kw) -> None:
        super().__init__(*args, **kw)
        self.verifier = verifier
        if isinstance(self.backend, LocalMem0):
            self.backend.on_added = self.on_mem0_added

    def on_mem0_added(self, events, agent) -> None:
        for e in events or []:
            ev = str(e.get("event", "ADD")).upper()
            if ev in ("ADD", "UPDATE") and e.get("id"):
                self.verifier.register(str(e["id"]), str(e.get("memory", "")), agent)
            elif ev == "DELETE" and e.get("id"):
                self.verifier.forget(str(e["id"]))

    def maintain(self) -> int:
        return super().maintain() + self.verifier.step()

    def _label(self, d: dict) -> str:
        ev = d.get("eval") or {}
        text = d.get("scoped") or d.get("text")
        if d["status"] in ("supported", "established", "hypothesis") and ev:
            cls = ev.get("classes", {})
            extra = f" [{len(ev.get('support', []))} for, {len(ev.get('contradict', []))} against"
            if d["status"] == "established":
                extra += f"; confirmed by {len(ev.get('independent_agents', []))} other agent(s)"
            elif d["status"] == "hypothesis" and cls.get("support") and set(cls["support"]) == {"E"}:
                extra += "; only the evidence that suggested it"
            return text + extra + "]"
        return text

    def briefing(self, agent, node, query, iteration):
        budget = self.cfg.briefing_budget_chars
        if budget <= 0:
            return ""
        try:
            hits = self.backend.search(query or self.adapter.goal.text, self.top_k)
        except Exception:
            hits = []
        status = self.verifier.status_of([h["id"] for h in hits])
        notices = self._notices(agent)
        head, _ = self.fit(self.goal_lines() + notices, budget)
        sections = []
        for title, states, asserted in SECTIONS:
            sec = []
            for h in hits:
                d = status.get(h["id"]) or {"status": "pending", "text": h["memory"], "scoped": h["memory"]}
                if d["status"] in states:
                    sec.append((h["id"], asserted, f"- {self._label(d)}", d.get("scoped") or h["memory"]))
            sections.append((title, sec))
        keep = select_within_budget(head, sections, lambda c: c[2], budget)
        lines, items = [head, "MEMORIES OF THE TEAM (Mem0, checked against measurements):"], []
        for title, sec in sections:
            sel = [c for c in sec if id(c) in keep]
            if sel:
                lines.append(title)
                for c in sel:
                    lines.append(c[2])
                    items.append({"id": c[0], "kind": "claim", "asserted": c[1], "text": c[3]})
        text = "\n".join(lines) if items or notices else head
        self.log_briefing(agent, node, iteration, items, text)
        return text

    def _notices(self, agent: str) -> list[str]:
        """Memories this agent was shown as true and that have since lost that status."""
        cur = int((self.store.get("cursor", agent) or {}).get("i", 0))
        dem = self.store.read_log("v_demotions", cur)
        self.store.put("cursor", agent, {"i": cur + len(dem)})
        if not dem:
            return []
        seen = {i["id"] for b in self.store.read_log("briefings") if b["agent"] == agent
                for i in b["items"] if i.get("asserted")}
        return [f"NOTICE: \"{x['text']}\", which you were shown as verified, is now {x['to']}."
                for x in dem if x["id"] in seen]

    def presented_knowledge(self):
        out = {}
        for b in self.store.read_log("briefings"):
            for i in b["items"]:
                out.setdefault(i["id"], {"id": i["id"], "text": i["text"]})
        status = self.verifier.status_of(list(out))
        for k, v in out.items():
            d = status.get(k, {})
            v.update({"final_layer": d.get("status", "pending"), "kind": d.get("kind"), "author": d.get("author"),
                      "history": d.get("history", [])})
        return list(out.values())

    def export_extra(self):
        out = super().export_extra()
        out["verification"] = self.verifier.export()
        return out


__all__ = ["VerifiedMem0Memory", "ASSERTED"]
