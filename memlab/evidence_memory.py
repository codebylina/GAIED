"""
Evidence-grounded layered memory for teams of scientific agents.

  L0 observed (raw, never modified)
     registry   one entry per entity, from provenance records only: attempts, status, value, cause of
                failure, agent, times, task id (memlab/registry.py). It is also the evidence base.
     journal    the agents' messages kept as written (by default their conclusions), with author, node,
                iteration and time.
  L1 asserted   atoms: claims the LLM proposes from the journal (memlab/distiller.py), each with its source
                messages. Each atom gets a pre-registered executable test, written once by the LLM, re-run by
                code at every new measurement, and a status from the evidence classes (memlab/verifier.py).
  L2 established  a SELECTION (never a summary) of the generalizations the measurements support, established
                first. It changes only when statuses change, so a refuted atom leaves it at once.

What an agent receives (one budget, the same for every condition): the Goal; active directives (written by an
analysis agent); notices about knowledge it was shown as true and that has since been demoted; exact team
results from the registry (best results, experiments in progress, failures with their cause); L2; and the L1
atoms relevant to its query, grouped by status. Before an experiment runs, the registry is consulted exactly
(already measured, failed before, in progress).

Variants (for the comparisons): verify=False shows every atom as team knowledge without status (the LLM proposes,
nothing decides); knowledge=False keeps only L0 and the coordination (no atoms at all).
"""
from __future__ import annotations

import time

from .base import Decision, Memory, select_within_budget
from .distiller import Distiller
from .registry import Registry, RegistryConfig
from .textrank import rank
from .verifier import ASSERTED, Verifier

L2_STATUSES = ("established", "supported")


class EvidenceMemory(Memory):
    name = "evidence"

    def __init__(self, store, adapter, cfg, llm=None, clock=time.time, verifier: Verifier | None = None,
                 verify: bool = True, knowledge: bool = True, journal_nodes=("conclude",),
                 registry_cfg: RegistryConfig | None = None) -> None:
        super().__init__(store, adapter, cfg, llm=llm, clock=clock)
        self.registry = Registry(store, adapter, registry_cfg, clock=clock)
        self.knowledge, self.journal_nodes = knowledge, tuple(journal_nodes)
        self.verifier = verifier if (verify and knowledge) else None
        on_atom = None
        if self.verifier is not None:
            on_atom = lambda a: self.verifier.register(a["id"], a["text"], a["author"], a["claimed_at"])  # noqa: E731
        self.distiller = Distiller(store, llm, on_atom=on_atom, clock=clock) if knowledge else None

    # ------------------------------------------------------------------ L0
    def on_observation(self, obs: dict) -> None:
        self.registry.record(obs)

    def record_episode(self, agent, node, text, iteration):
        super().record_episode(agent, node, text, iteration)
        if self.knowledge and node in self.journal_nodes and text and text.strip():
            self.store.append("journal", {"id": self.new_id("m"), "agent": agent, "node": node,
                                          "iteration": iteration, "t": self.clock(), "text": text})

    # ------------------------------------------------------------------ coordination
    def claim(self, agent: str, raw: str) -> Decision:
        return self.registry.claim(agent, raw)

    def screen(self, agent, proposals, iteration):
        out, seen = [], set()
        for raw in proposals:
            key = self.registry.key(raw)
            d = Decision(raw, key, "duplicate_in_batch") if key in seen else self.registry.claim(agent, raw)
            seen.add(key)
            out.append(d)
        self.log_decisions(agent, iteration, out)
        return out

    def await_results(self, agent, entities, timeout_s):
        return self.registry.wait(entities, timeout_s)

    def release(self, agent, entity):
        self.registry.release(agent, entity)

    # ------------------------------------------------------------------ directives (written by an analysis agent)
    def put_directive(self, author, text, targets="all", lifetime="standing", expires_at=None):
        did = self.new_id("d")
        self.store.put("directives", did, {"id": did, "author": author, "text": text, "targets": targets,
                                           "lifetime": lifetime, "expires_at": expires_at, "t": self.clock()})
        return did

    def _directives(self, agent: str) -> list[str]:
        now, out = self.clock(), []
        for d in sorted(self.store.all("directives"), key=lambda d: d["t"]):
            if d.get("expires_at") and d["expires_at"] < now:
                continue
            if d["targets"] != "all" and agent not in d["targets"]:
                continue
            out.append(f"- {d['text']} (from {d['author']})")
        return out

    # ------------------------------------------------------------------ background work
    def maintain(self) -> int:
        n = self.distiller.step() if self.distiller else 0
        return n + (self.verifier.step() if self.verifier else 0)

    # ------------------------------------------------------------------ what an agent receives
    def _status(self) -> dict[str, dict]:
        if not self.verifier:
            return {}
        return {d["id"]: d for d in self.store.all("vmem")}

    @staticmethod
    def _label(a: dict, d: dict | None) -> str:
        text = (d or {}).get("scoped") or a["text"]
        ev = (d or {}).get("eval") or {}
        if not ev or (d or {}).get("kind") != "generalization":
            return text
        extra = f" [{len(ev.get('support', []))} for, {len(ev.get('contradict', []))} against"
        if d["status"] == "established":
            extra += f"; confirmed by {len(ev.get('independent_agents', []))} agent(s) who had not seen it"
        elif d["status"] == "hypothesis":
            extra += "; only the measurements that suggested it"
        return text + extra + "]"

    def _registry_section(self) -> list[tuple]:
        """(id, asserted, line, text) items of the exact team records, in priority order."""
        items = []
        for r in self.registry.best():
            items.append((r["id"], None, f"- {r['entity']}: {r['value']:.4g} (by {r['owner']})", r["entity"]))
        for r in self.registry.running():
            items.append((f"run:{r['entity']}", None, f"- {r['entity']}: being measured by {r['agent']}", r["entity"]))
        for r in self.registry.failures():
            why = "invalid, do not propose again" if r["state"] == "invalid" else "experiment error"
            err = f": {r['error'][:80]}" if r.get("error") else ""
            name = r["entity"][4:] if r["entity"].startswith("raw:") else r["entity"]
            items.append((f"fail:{r['entity']}", None, f"- {name}: {why}{err}", name))
        # interleave best results and failures so a small budget keeps both kinds
        best = [i for i in items if not i[0].startswith(("run:", "fail:"))]
        rest = [i for i in items if i[0].startswith(("run:", "fail:"))]
        out = []
        for k in range(max(len(best), len(rest))):
            out += best[k:k + 1] + rest[k:k + 1]
        return out

    def briefing(self, agent, node, query, iteration):
        budget = self.cfg.briefing_budget_chars
        if budget <= 0:
            return ""
        head_lines = self.goal_lines()
        dirs = self._directives(agent)
        if dirs:
            head_lines += ["DIRECTIVES:"] + dirs
        head_lines += self._notices(agent)
        head, _ = self.fit(head_lines, budget)
        sections = [("TEAM EXPERIMENTS (exact records; best results first):", self._registry_section())]
        if self.knowledge:
            sections += self._knowledge_sections(query or self.adapter.goal.text)
        keep = select_within_budget(head, [(t, s) for t, s in sections], lambda c: c[2], budget)
        lines, items = [head], []
        for title, sec in sections:
            sel = [c for c in sec if id(c) in keep]
            if sel:
                lines.append(title)
                for c in sel:
                    lines.append(c[2])
                    kind = "result" if title.startswith("TEAM EXPERIMENTS") else "claim"
                    items.append({"id": c[0], "kind": kind, "asserted": c[1], "text": c[3]})
        text = "\n".join(lines)
        self.log_briefing(agent, node, iteration, items, text)
        return text

    def _knowledge_sections(self, query: str) -> list[tuple[str, list]]:
        atoms = {a["id"]: a for a in self.store.all("atoms")}
        if not atoms:
            return []
        status = self._status()
        order = rank(query, [(i, (status.get(i) or {}).get("scoped") or a["text"]) for i, a in atoms.items()])
        if not self.verifier:      # no verification: everything the LLM proposed is team knowledge
            return [("TEAM KNOWLEDGE:", [(i, True, f"- {atoms[i]['text']}", atoms[i]["text"]) for i in order])]
        st = lambda i: (status.get(i) or {}).get("status", "pending")      # noqa: E731
        kind = lambda i: (status.get(i) or {}).get("kind")                  # noqa: E731
        txt = lambda i: (status.get(i) or {}).get("scoped") or atoms[i]["text"]  # noqa: E731

        def item(i, asserted):
            return (i, asserted, f"- {self._label(atoms[i], status.get(i))}", txt(i))

        l2 = [i for i in atoms if kind(i) == "generalization" and st(i) in L2_STATUSES]
        l2.sort(key=lambda i: (st(i) != "established",
                               -len(((status[i].get("eval") or {}).get("independent_agents")) or []),
                               -len(((status[i].get("eval") or {}).get("support")) or [])))
        return [
            ("ESTABLISHED KNOWLEDGE (supported by the team's measurements):", [item(i, True) for i in l2]),
            ("TO TEST (not confirmed by new measurements yet):",
             [item(i, None) for i in order if st(i) in ("hypothesis", "open", "contested", "pending")]),
            ("KNOWN FALSE (contradicted by measurements):", [item(i, False) for i in order if st(i) == "refuted"]),
            ("NO MEASUREMENT ON RECORD (reported results the registry does not contain):",
             [item(i, None) for i in order if st(i) == "unsupported"]),
            ("CANNOT BE CHECKED AGAINST MEASUREMENTS:", [item(i, None) for i in order if st(i) == "unverifiable"]),
        ]

    def _notices(self, agent: str) -> list[str]:
        """Knowledge this agent was shown as true and that has since lost that status."""
        if not self.verifier:
            return []
        cur = int((self.store.get("cursor", agent) or {}).get("i", 0))
        dem = self.store.read_log("v_demotions", cur)
        self.store.put("cursor", agent, {"i": cur + len(dem)})
        if not dem:
            return []
        seen = {i["id"] for b in self.store.read_log("briefings") if b["agent"] == agent
                for i in b["items"] if i.get("asserted")}
        return [f"NOTICE: \"{x['text']}\", which you were shown as established, is now {x['to']}."
                for x in dem if x["id"] in seen]

    # ------------------------------------------------------------------ evaluation and export
    def presented_knowledge(self):
        out = {}
        for b in self.store.read_log("briefings"):
            for i in b["items"]:
                if i["kind"] == "claim":
                    out.setdefault(i["id"], {"id": i["id"], "text": i["text"]})
        status = self._status()
        for k, v in out.items():
            d = status.get(k, {})
            a = self.store.get("atoms", k) or {}
            v.update({"final_layer": d.get("status", "pending") if self.verifier else "stored",
                      "kind": d.get("kind"), "author": a.get("author"), "history": d.get("history", [])})
        return list(out.values())

    def export_extra(self):
        out = {"journal": self.store.read_log("journal"), "atoms": self.store.all("atoms"),
               "distill_journal": self.store.read_log("distill_journal"),
               "registry": self.store.all("registry"), "directives": self.store.all("directives")}
        if self.verifier:
            out["verification"] = self.verifier.export()
        return out


__all__ = ["EvidenceMemory", "ASSERTED"]
