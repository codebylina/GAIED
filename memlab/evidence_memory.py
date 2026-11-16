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

Retrieval, as in FALDA (memlab/retrieval.py: dense + BM25, reciprocal-rank fusion, 10 results per collection):
the memory searches its collections with the agent's query -- the experiment registry (L0) and the atoms (L1 and
L2) -- and puts what it finds in the prompt, grouped by layer and status, within one character budget that is the
same for every condition. Always shown, because they are small and every step needs them: the Goal, the
directives that concern the agent (written by an analysis agent), and notices about knowledge it was shown as
established and that has since been demoted. The agent can also query the memory itself with tools
(search_memory, lookup, best_results). Raw journal messages are not put in prompts: what reaches an agent from
the text the agents wrote goes through L1, with a status (show_messages=True shows them, labelled unchecked).
Before an experiment runs, the registry is consulted exactly (already measured, failed before, in progress).

Variants (for the comparisons): verify=False shows every atom as team knowledge without status (the LLM proposes,
nothing decides); knowledge=False keeps only L0 and the coordination (no atoms at all).
"""
from __future__ import annotations

import time

from .base import Decision, Memory, select_within_budget
from .distiller import Distiller
from .registry import Registry, RegistryConfig
from .retrieval import DEFAULT_LIMIT, Embedder, hybrid
from .verifier import ASSERTED, Verifier

L2_STATUSES = ("established", "supported")


class EvidenceMemory(Memory):
    name = "evidence"

    def __init__(self, store, adapter, cfg, llm=None, clock=time.time, verifier: Verifier | None = None,
                 verify: bool = True, knowledge: bool = True, journal_nodes=("conclude",),
                 registry_cfg: RegistryConfig | None = None, embedder: Embedder | None = None,
                 limit: int = DEFAULT_LIMIT, show_messages: bool = False) -> None:
        super().__init__(store, adapter, cfg, llm=llm, clock=clock)
        self.embedder = embedder if embedder is not None else Embedder(store)
        self.limit, self.show_messages = limit, show_messages
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

    # ---- collections searched by the memory
    def _experiment_docs(self) -> list[tuple]:
        """(id, asserted, line, text) per registry entity; the line is also the searched text."""
        out = []
        running = {r["entity"]: r["agent"] for r in self.registry.running()}
        for doc in self.store.all("registry"):
            e = doc["entity"]
            name = e[4:] if e.startswith("raw:") else e
            v = self.registry.view(e)
            if v["state"] == "measured":
                line = f"- {name}: {v['value']:.4g} (measured by {v['owner']})"
            elif v["state"] == "invalid":
                line = f"- {name}: invalid, do not propose again" + (f" ({v['error'][:80]})" if v.get("error") else "")
            elif v["state"] == "error":
                line = f"- {name}: the experiment failed" + (f" ({v['error'][:80]})" if v.get("error") else "")
            else:
                continue
            out.append((f"exp:{e}", None, line, name))
        for e, agent in running.items():
            if not self.store.get("registry", e):
                out.append((f"run:{e}", None, f"- {e}: being measured by {agent}", e))
        return out

    def _search(self, query: str, items: list[tuple]) -> list[tuple]:
        by_id = {c[0]: c for c in items}
        return [by_id[i] for i in hybrid(self.embedder, query, [(c[0], c[2]) for c in items], self.limit)]

    def _head(self, agent: str) -> list[str]:
        lines = self.goal_lines()
        dirs = self._directives(agent)
        if dirs:
            lines += ["DIRECTIVES:"] + dirs
        return lines + self._notices(agent)

    def _render(self, agent, node, iteration, head_lines, sections, budget) -> str:
        head, _ = self.fit(head_lines, budget) if head_lines else ("", 0)
        keep = select_within_budget(head, sections, lambda c: c[2], budget)
        lines, items = ([head] if head else []), []
        for title, sec in sections:
            sel = [c for c in sec if id(c) in keep]
            if sel:
                lines.append(title)
                for c in sel:
                    lines.append(c[2])
                    kind = "result" if c[0].startswith(("exp:", "run:")) else "message" if c[0].startswith("msg:") else "claim"
                    items.append({"id": c[0][4:] if kind == "message" else c[0], "kind": kind, "asserted": c[1],
                                  "text": c[3]})
        text = "\n".join(lines)
        self.log_briefing(agent, node, iteration, items, text)
        return text

    def _sections(self, query: str) -> list[tuple[str, list]]:
        sections = [("TEAM EXPERIMENTS (exact records):", self._search(query, self._experiment_docs()))]
        if self.knowledge:
            sections += self._knowledge_sections(query)
            if self.show_messages:
                msgs = [(f"msg:{m['id']}", None, f"- [{m['agent']}] {m['text'][:300]}", m["text"])
                        for m in self.store.read_log("journal")]
                sections.append(("MESSAGES OF THE TEAM (not checked):", self._search(query, msgs)))
        return sections

    def briefing(self, agent, node, query, iteration):
        budget = self.cfg.briefing_budget_chars
        if budget <= 0:
            return ""
        return self._render(agent, node, iteration, self._head(agent), self._sections(query or self.adapter.goal.text),
                            budget)

    def _knowledge_sections(self, query: str) -> list[tuple[str, list]]:
        atoms = {a["id"]: a for a in self.store.all("atoms")}
        if not atoms:
            return []
        status = self._status()
        if not self.verifier:      # no verification: everything the LLM proposed is team knowledge
            hits = self._search(query, [(i, True, f"- {a['text']}", a["text"]) for i, a in atoms.items()])
            return [("TEAM KNOWLEDGE:", hits)]
        st = lambda i: (status.get(i) or {}).get("status", "pending")      # noqa: E731
        kind = lambda i: (status.get(i) or {}).get("kind")                  # noqa: E731
        txt = lambda i: (status.get(i) or {}).get("scoped") or atoms[i]["text"]  # noqa: E731
        asserted = {i: (True if st(i) in L2_STATUSES and kind(i) == "generalization" else
                        False if st(i) == "refuted" else None) for i in atoms}
        hits = self._search(query, [(i, asserted[i], f"- {self._label(atoms[i], status.get(i))}", txt(i))
                                    for i in atoms if not (kind(i) in ("report", "prediction") and st(i) == "verified")])
        return [
            ("ESTABLISHED KNOWLEDGE (supported by the team's measurements):",
             [c for c in hits if c[1] is True]),
            ("TO TEST (not confirmed by new measurements yet):",
             [c for c in hits if st(c[0]) in ("hypothesis", "open", "contested", "pending")]),
            ("KNOWN FALSE (contradicted by measurements):", [c for c in hits if c[1] is False]),
            ("NO MEASUREMENT ON RECORD (reported results the registry does not contain):",
             [c for c in hits if st(c[0]) == "unsupported"]),
            ("CANNOT BE CHECKED AGAINST MEASUREMENTS:", [c for c in hits if st(c[0]) == "unverifiable"]),
        ]

    # ---- tools: the agent queries the memory itself
    def tool_specs(self):
        specs = [{"name": "lookup", "description": "Exact record of one design: its measured value, its failure "
                  "and the cause, or that a teammate is measuring it.", "args": {"design": "the design"}},
                 {"name": "best_results", "description": "The team's best measured designs, best first.", "args": {}},
                 {"name": "search_memory", "description": "Search the team's experiments and knowledge for a "
                  "question or a topic.", "args": {"query": "your question"}}]
        return specs

    def call_tool(self, agent, name, args):
        budget = self.cfg.briefing_budget_chars or 2000
        if name == "lookup":
            raw = str(args.get("design", ""))
            e = self.registry.key(raw)
            v = self.registry.view(e)
            holder = self.store.lease_holder(self.registry.lease(e))
            line = (f"- {raw}: {v['value']:.4g} (measured by {v['owner']})" if v["state"] == "measured" else
                    f"- {raw}: invalid, do not propose again ({v.get('error')})" if v["state"] == "invalid" else
                    f"- {raw}: the experiment failed ({v.get('error')})" if v["state"] == "error" else
                    f"- {raw}: being measured by {holder}" if holder else f"- {raw}: never measured")
            return self._render(agent, "tool:lookup", -1, [], [("RECORD:", [(f"exp:{e}", None, line, raw)])], budget)
        if name == "best_results":
            rows = [(f"exp:{r['entity']}", None, f"- {r['entity']}: {r['value']:.4g} (measured by {r['owner']})",
                     r["entity"]) for r in self.registry.best()[:self.limit]]
            return self._render(agent, "tool:best_results", -1, [], [("BEST RESULTS OF THE TEAM:", rows)], budget) \
                or "No measured design yet."
        if name == "search_memory":
            q = str(args.get("query", "")) or self.adapter.goal.text
            return self._render(agent, "tool:search_memory", -1, [], self._sections(q), budget) or "Nothing found."
        return "unknown tool"

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
