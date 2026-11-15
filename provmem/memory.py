"""
ProvMem: a shared memory grounded in workflow provenance, with confidence layers for claims.

Layers of a claim:
  pending       registered, not judged yet (never shown)
  rejected      contradicted by the observations of its own author when it was written
  unverifiable  not testable with the observations (advice, plan); shown as unverified advice
  duplicate     restates an existing claim; merged into it
  hypothesis    plausible given the evidence that suggested it; shown only as "to test"
  supported     confirmed by evidence measured after it was written
  established   confirmed independently (by agents other than its author)
  refuted       contradicted by the observations; shown as "known false"
The layer is decided by the judge's verdict, constrained by the provenance class of the cited
evidence (exploratory / confirmatory / independent). The code never decides whether a claim is true:
it only enforces which evidence may justify which layer.
"""
from __future__ import annotations

import json
import time

from .base import Decision, Memory, select_within_budget
from .judge import GroundedJudge, TextJudge
from .sandbox import TestError, run_test

ACTIVE = ("hypothesis", "supported", "established", "refuted")
ASSERTED = ("supported", "established")
RANK = {"pending": 0, "rejected": 0, "unverifiable": 0, "duplicate": 0, "refuted": 0,
        "hypothesis": 1, "supported": 2, "established": 3}


def split_claims(text: str) -> list[str]:
    """One claim per line or bullet of a conclusion (agents are asked for stand-alone statements)."""
    out = []
    for line in (text or "").splitlines():
        s = line.strip().lstrip("-*•0123456789.) ").strip()
        if sum(ch.isalpha() for ch in s) >= 3 and not s.endswith(":"):
            out.append(s)
    return out


class ProvMem(Memory):
    name = "provmem"

    def __init__(self, *args, judge=None, **kw) -> None:
        super().__init__(*args, **kw)
        mode = self.cfg.judge_mode
        if judge is not None:
            self.judge = judge
        elif mode == "grounded" and self.llm:
            self.judge = GroundedJudge(self.llm)
        elif mode == "text" and self.llm:
            self.judge = TextJudge(self.llm)
        else:
            self.judge = None
        self.related_k = int(self.cfg.extra.get("related_k", 0))

    # ================================================================== coordination
    def screen(self, agent, proposals, iteration):
        if not self.cfg.coordination:
            return super().screen(agent, proposals, iteration)
        out, seen = [], set()
        last_read = self._last_read_claims(agent)
        for raw in proposals:
            ent = self.adapter.canonical(raw)
            key = ent or f"raw:{raw}"
            if key in seen:
                out.append(Decision(raw, key, "duplicate_in_batch"))
                continue
            seen.add(key)
            prev = self._obs_of(key)
            ok = [o for o in prev if o["status"] == "ok"]
            if ok:
                out.append(Decision(raw, key, "known", value=ok[0]["value"], owner=ok[0].get("agent")))
                continue
            if prev:
                out.append(Decision(raw, key, "failed_before", owner=prev[0].get("agent")))
                continue
            granted, holder = self.store.take_lease(key, agent, self.cfg.lease_ttl_s)
            if not granted:
                out.append(Decision(raw, key, "in_progress", owner=holder))
                continue
            self.store.put("held", key, {"entity": key, "agent": agent, "t": self.clock()})
            out.append(Decision(raw, key, "run" if ent else "invalid"))
        for d in out:   # graph: which claims the agent had read when it chose these entities
            if d.action in ("run", "invalid") and last_read:
                self.store.append("edges", {"type": "led_to", "src": last_read, "dst": d.entity,
                                            "agent": agent, "iteration": iteration, "t": self.clock()})
        self.log_decisions(agent, iteration, out)
        return out

    def heartbeat(self, agent):
        for h in self.store.all("held"):
            if h["agent"] == agent:
                self.store.renew_lease(h["entity"], agent, self.cfg.lease_ttl_s)

    def release(self, agent, entity):
        self.store.release_lease(entity, agent)

    def await_results(self, agent, entities, timeout_s):
        deadline, found = self.clock() + timeout_s, {}
        while True:
            for e in entities:
                if e not in found:
                    obs = self._obs_of(e)
                    if obs:
                        found[e] = obs[0]
            pending = [e for e in entities if e not in found and self.store.lease_holder(e)]
            if not pending or self.clock() >= deadline:
                return found
            time.sleep(0.2)

    def _obs_of(self, entity):
        return sorted((o for o in self.store.all("obs") if o["entity"] == entity), key=lambda o: o["t"])

    def on_observation(self, obs):
        holder = self.store.lease_holder(obs["entity"])
        if holder:
            self.store.release_lease(obs["entity"], holder)
        self.store.incr("obs_since_eval")

    # ================================================================== claims
    def submit_conclusion(self, agent, text, iteration):
        self.record_episode(agent, "conclude", text, iteration)
        if not self.cfg.claims:
            return []
        ids = []
        for s in split_claims(text):
            cid = self.new_id("c")
            claim = {"id": cid, "statement": s, "scoped_statement": s, "author": agent,
                     "iteration": iteration, "registered_at": self.clock(), "layer": "pending",
                     "version": 0, "test_code": "", "last_eval": None, "relations": [], "history": []}
            self.store.put("claims", cid, claim)
            self.store.append("pending", {"id": cid})
            ids.append(cid)
        return ids

    def maintain(self):
        if not self.cfg.claims:
            return 0
        done = 0
        start = int((self.store.get("meta", "pending_cursor") or {}).get("i", 0))
        queue = self.store.read_log("pending", start)
        self.store.put("meta", "pending_cursor", {"i": start + len(queue)})
        for item in queue:
            claim = self.store.get("claims", item["id"])
            if claim and claim["layer"] == "pending":
                self._judge_new(claim)
                done += 1
        if self.cfg.judge_mode == "grounded" and self.judge:
            done += self._reevaluate()
        return done

    # ------------------------------------------------------------------ new claims
    def _judge_new(self, claim):
        related = self._related(claim["statement"], exclude=claim["id"])
        mode = self.cfg.judge_mode
        if mode == "accept":
            self._transition(claim, "established", {"verdict": "accepted", "rationale": "accepted as stated"})
            return
        if self.judge is None:   # no LLM configured: the claim stays pending (never shown as knowledge)
            self._journal(claim, "no_judge", {"mode": mode})
            return
        if mode == "text":
            v = self.judge.judge_text(claim["statement"], related, self.adapter.goal.text)
            claim["relations"].append(v["relation"])
            self._journal(claim, "text_verdict", v)
            target = {"supported": "established", "inconclusive": "established",
                      "contradicted": "rejected", "untestable": "unverifiable"}[v["verdict"]]
            if v["relation"]["type"] == "duplicate" and self.store.get("claims", str(v["relation"]["target"])):
                target = "duplicate"
            self._transition(claim, target, v)
            tgt = self.store.get("claims", str(v["relation"]["target"] or ""))
            if v["relation"]["type"] == "contradicts" and tgt and target == "established":
                self._transition(tgt, "refuted", {"verdict": "superseded", "by": claim["id"],
                                                  "rationale": "contradicted by a newer claim"})
            return
        # grounded
        rows = self._rows()
        f = self.judge.formalize(claim["statement"], self._schema_doc(), rows[-20:], related,
                                 self.adapter.goal.text)
        self._journal(claim, "formalize", {k: f[k] for k in ("testable", "scoped_statement", "relation", "code")})
        claim["scoped_statement"], claim["test_code"] = f["scoped_statement"], f["code"]
        rel = f["relation"]
        claim["relations"].append(rel)
        target = self.store.get("claims", str(rel.get("target") or ""))
        if target:
            self.store.append("edges", {"type": rel["type"], "src": claim["id"], "dst": target["id"], "t": self.clock()})
        if rel["type"] == "duplicate" and target and target["layer"] not in ("rejected", "duplicate"):
            self._transition(claim, "duplicate", {"verdict": "duplicate", "of": target["id"]})
            return
        if not f["testable"]:
            self._transition(claim, "unverifiable", {"verdict": "untestable", "rationale": "not testable"})
            return
        result = self._run_with_repairs(claim, f, rows, related)
        if result is None:
            self._transition(claim, "unverifiable", {"verdict": "untestable", "rationale": "test could not run"})
            return
        self._decide(claim, result, rows)

    def _run_with_repairs(self, claim, f, rows, related):
        code, err = f["code"], None
        if f.get("truncated"):
            self._journal(claim, "test_error", {"attempt": -1, "error": "answer cut by the token limit"})
        for attempt in range(1 + self.cfg.test_repairs):
            if attempt:
                f2 = self.judge.formalize(claim["statement"], self._schema_doc(), rows[-20:], related,
                                          self.adapter.goal.text, error=err, previous_code=code)
                code = f2["code"] or code
                claim["test_code"] = code
            try:
                return run_test(code, rows, self.cfg.test_timeout_s)
            except TestError as exc:
                err = str(exc)
                if f.get("truncated") and attempt == 0:
                    err += " (your answer was cut off: write a much shorter test)"
                self._journal(claim, "test_error", {"attempt": attempt, "error": err})
        return None

    # ------------------------------------------------------------------ verdicts and layers
    def _classify(self, claim, result, rows):
        by_id = {r["id"]: r for r in rows}
        out = []
        for role in ("support", "contradict"):
            for i in result[role]:
                r = by_id[i]
                if r["t"] <= claim["registered_at"]:
                    cls = "E"
                elif r.get("agent") != claim["author"]:
                    cls = "I"
                else:
                    cls = "C"
                out.append({**r, "class": cls, "role": role})
        return sorted(out, key=lambda r: (r["class"] == "E", -r["t"]))

    def _decide(self, claim, result, rows, reason="new"):
        ev = self._classify(claim, result, rows)
        verdict = self.judge.verdict(claim, result, ev, self.adapter.domain_knowledge(claim["scoped_statement"]))
        target = self._target_layer(claim, verdict, ev)
        if self.cfg.review and self._high_impact(claim["layer"], target):
            verdict = self.judge.review(claim, verdict, result, ev)
            target = self._target_layer(claim, verdict, ev)
        claim["last_eval"] = {"support": result["support"], "contradict": result["contradict"],
                              "in_scope": result["in_scope"], "stats": result["stats"],
                              "n_obs": len(rows), "t": self.clock()}
        self._dossier(claim, verdict, ev, result, target, reason)
        self._transition(claim, target, verdict)

    def _target_layer(self, claim, verdict, ev):
        """Which layer the cited evidence can justify. The judge decides the verdict; this function only
        enforces that a layer is reached with the right class of evidence (or none, for ablations)."""
        cur = claim["layer"]
        sup = {e["id"]: e for e in ev if e["role"] == "support"}
        con = {e["id"]: e for e in ev if e["role"] == "contradict"}
        cited_s = [sup[i] for i in verdict["cite_support"] if i in sup]
        cited_c = [con[i] for i in verdict["cite_contradict"] if i in con]
        verdict["cited_valid"] = {"support": [e["id"] for e in cited_s], "contradict": [e["id"] for e in cited_c]}
        v = verdict["verdict"]
        if v == "contradicted" and cited_c:
            return "rejected" if cur == "pending" else "refuted"
        if v == "supported" and cited_s:
            if not self.cfg.layers:
                return "established"
            confirm = [e for e in cited_s if e["class"] in ("C", "I")]
            indep_agents = {e["agent"] for e in cited_s if e["class"] == "I"}
            new = ("established" if len(indep_agents) >= max(1, self.cfg.min_independent)
                   else "supported" if confirm else "hypothesis")
            if cur == "refuted":
                return new if confirm else cur
            return new if RANK[new] >= RANK.get(cur, 0) else cur
        # inconclusive, or a verdict without valid citations
        if cur == "pending":
            return "hypothesis" if self.cfg.layers else "established"
        if cur == "established" and any(e["class"] in ("C", "I") for e in con.values()):
            return "supported"   # contested: confirmatory evidence against it, no clear verdict
        return cur

    @staticmethod
    def _high_impact(cur, target):
        return target != cur and (target == "established" or cur in ASSERTED and target in ("refuted", "supported", "hypothesis"))

    def _transition(self, claim, layer, verdict):
        old = claim["layer"]
        claim["history"].append({"t": self.clock(), "from": old, "to": layer, "verdict": verdict.get("verdict")})
        if layer != old:
            claim["version"] += 1
        claim["layer"] = layer
        self.store.put("claims", claim["id"], claim)
        self.store.append("claim_versions", {"id": claim["id"], "version": claim["version"], "t": self.clock(),
                                             "layer": layer, "from": old, "scoped_statement": claim["scoped_statement"]})
        if self.cfg.push and old in ASSERTED + ("hypothesis",) and RANK.get(layer, 0) < RANK[old]:
            self._notify_readers(claim, old, layer)

    def _notify_readers(self, claim, old, new):
        readers = {b["agent"] for b in self.store.read_log("briefings")
                   if any(i["id"] == claim["id"] for i in b["items"])}
        text = (f"NOTICE: claim {claim['id']} that you were shown as {old} is now {new}: "
                f"\"{claim['scoped_statement']}\"")
        for a in readers:
            self.store.append(f"inbox:{a}", {"t": self.clock(), "kind": "claim_" + new, "ref": claim["id"], "text": text})
        self.store.append("events", {"t": self.clock(), "kind": "claim_" + new, "ref": claim["id"],
                                     "from": old, "to": new, "notified": sorted(readers)})

    def _reevaluate(self):
        """Re-run the tests of active claims when new observations arrived; ask the judge again only
        when the test finds new confirmatory evidence (for or against)."""
        n_obs = self.store.log_len("obs_log")
        done = 0
        rows = None
        for claim in self.store.all("claims"):
            if claim["layer"] not in ACTIVE or not claim.get("test_code"):
                continue
            last = claim.get("last_eval") or {}
            if last.get("n_obs", 0) >= n_obs:
                continue
            rows = rows if rows is not None else self._rows()
            try:
                result = run_test(claim["test_code"], rows, self.cfg.test_timeout_s)
            except TestError as exc:
                self._journal(claim, "retest_error", {"error": str(exc)})
                claim["last_eval"] = {**last, "n_obs": n_obs}
                self.store.put("claims", claim["id"], claim)
                continue
            old = set(last.get("support", [])) | set(last.get("contradict", []))
            new_ids = (set(result["support"]) | set(result["contradict"])) - old
            by_id = {r["id"]: r for r in rows}
            if any(by_id[i]["t"] > claim["registered_at"] for i in new_ids):
                self._decide(claim, result, rows, reason="new_confirmatory_evidence")
                done += 1
            else:
                claim["last_eval"] = {**last, "support": result["support"], "contradict": result["contradict"],
                                      "in_scope": result["in_scope"], "stats": result["stats"], "n_obs": n_obs}
                self.store.put("claims", claim["id"], claim)
        return done

    # ------------------------------------------------------------------ records
    def _journal(self, claim, step, data):
        self.store.append("journal", {"t": self.clock(), "claim": claim["id"], "step": step, "data": data})

    def _dossier(self, claim, verdict, ev, result, target, reason):
        self.store.append("dossiers", {
            "t": self.clock(), "claim": claim["id"], "version": claim["version"], "reason": reason,
            "from": claim["layer"], "to": target, "judge": getattr(self.judge, "name", "?"),
            "verdict": verdict.get("verdict"), "cited_valid": verdict.get("cited_valid"),
            "cited_raw": {"support": verdict.get("cite_support"), "contradict": verdict.get("cite_contradict")},
            "rationale": verdict.get("rationale"), "review": verdict.get("review"),
            "stats": result.get("stats"), "note": result.get("note"),
            "counts": {c: sum(1 for e in ev if e["class"] == c) for c in "ECI"},
            "n_support": len(result["support"]), "n_contradict": len(result["contradict"])})

    def _rows(self):
        rows = []
        for o in sorted(self.store.all("obs"), key=lambda o: o["t"]):
            rows.append({"id": str(o["id"]), "entity": o["entity"], "value": o["value"], "status": o["status"],
                         "agent": o.get("agent"), "t": o["t"], **(o.get("descriptors") or {})})
        return rows

    def _schema_doc(self):
        obj = self.adapter.goal.objective
        return ("id (observation id), entity (canonical key), value (" + obj.name + ", " +
                ("higher is better" if obj.direction == "max" else "lower is better") +
                "; null when failed), status ('ok' or 'failed'), agent, t (time)\n" + self.adapter.descriptor_doc())

    def _related(self, text, exclude=None):
        if not self.related_k:
            return []
        claims = [c for c in self.store.all("claims") if c["id"] != exclude and c["layer"] not in ("pending",)]
        ids = self.retriever.search(text, [(c["id"], c["scoped_statement"]) for c in claims], self.related_k)
        by = {c["id"]: c for c in claims}
        return [{"id": i, "statement": by[i]["scoped_statement"], "layer": by[i]["layer"]} for i in ids]

    # ================================================================== directives
    def put_directive(self, author, text, targets="all", lifetime="standing", expires_at=None):
        did = self.new_id("d")
        self.store.put("directives", did, {"id": did, "author": author, "text": text, "targets": targets,
                                           "lifetime": lifetime, "t": self.clock(), "expires_at": expires_at,
                                           "acks": []})
        self.store.append("events", {"t": self.clock(), "kind": "directive", "ref": did})
        return did

    def _directives_for(self, agent):
        now, out = self.clock(), []
        for d in sorted(self.store.all("directives"), key=lambda d: d["t"]):
            if d["targets"] != "all" and agent not in d["targets"]:
                continue
            if d.get("expires_at") and d["expires_at"] < now:
                continue
            if d["lifetime"] == "once" and agent in d["acks"]:
                continue
            out.append(d)
        return out

    def _ack(self, agent, directives):
        for d in directives:
            if d["lifetime"] == "once" and agent not in d["acks"]:
                d["acks"].append(agent)
                self.store.put("directives", d["id"], d)

    # ================================================================== briefing (push)
    def briefing(self, agent, node, query, iteration):
        self.heartbeat(agent)
        budget = self.cfg.briefing_budget_chars
        if budget <= 0:
            return ""
        dirs = self._directives_for(agent)
        head = self.goal_lines() + [f"DIRECTIVE ({d['author']}): {d['text']}" for d in dirs]
        notices = []
        if self.cfg.push:
            cur = int((self.store.get("cursor", agent) or {}).get("i", 0))
            inbox = self.store.read_log(f"inbox:{agent}", cur)
            self.store.put("cursor", agent, {"i": cur + len(inbox)})
            notices = [n["text"] for n in inbox]
        self._ack(agent, dirs)
        if not self.cfg.push:   # pull mode: the agent queries the memory with tools
            text, _ = self.fit(head + notices, budget)
            self.log_briefing(agent, node, iteration, [], text)
            return text
        sections = self._sections(query, agent)
        head_text, _ = self.fit(head + notices, budget)
        keep = select_within_budget(head_text, sections, lambda c: c[3], budget)
        lines, items = [head_text], []
        for title, sec in sections:
            sel = [c for c in sec if id(c) in keep]
            if sel:
                lines.append(title)
                for c in sel:
                    lines.append(c[3])
                    items.append({"id": c[0], "kind": c[1], "asserted": c[2], "text": c[4]})
        text = "\n".join(lines)
        self.log_briefing(agent, node, iteration, items, text)
        return text

    def _sections(self, query, agent=None):
        """Ranked sections: (title, [(id, kind, asserted, line, text)])."""
        res = [(i, "fact", True, l, l) for i, l in self.team_results_lines()]
        held = [h for h in self.store.all("held") if h["agent"] != agent
                and self.store.lease_holder(h["entity"]) == h["agent"]]
        prog = [(f"held:{h['entity']}", "in_progress", None, f"{h['entity']} (being measured by {h['agent']})",
                 h["entity"]) for h in sorted(held, key=lambda h: -h["t"])]
        fails = [(i, "fact", True, l, l) for i, l in self.failed_lines()]
        claims = [c for c in self.store.all("claims") if c["layer"] in ACTIVE + ("unverifiable",)]
        order = self.retriever.search(query or self.adapter.goal.text,
                                      [(c["id"], c["scoped_statement"]) for c in claims], len(claims))
        by = {c["id"]: c for c in claims}
        ranked = [by[i] for i in order] + [c for c in claims if c["id"] not in set(order)]

        def line(c):
            le = c.get("last_eval") or {}
            return (c["id"], "claim", True if c["layer"] in ASSERTED else (False if c["layer"] == "refuted" else None),
                    f"[{c['id']}] {c['scoped_statement']} ({len(le.get('support', []))} for, "
                    f"{len(le.get('contradict', []))} against)", c["scoped_statement"])
        sec = lambda layer: [line(c) for c in ranked if c["layer"] == layer]  # noqa: E731
        return [("MEASURED RESULTS (best first):", res),
                ("BEING MEASURED NOW BY OTHERS (do not propose):", prog),
                ("FAILED (do not retry):", fails),
                ("ESTABLISHED (independently confirmed):", sec("established")),
                ("SUPPORTED (confirmed, not yet independently):", sec("supported")),
                ("TO TEST (hypotheses, not confirmed):", sec("hypothesis")),
                ("KNOWN FALSE (refuted by measurements):", sec("refuted")),
                ("UNVERIFIED ADVICE:", [(i, k, None, l, t) for i, k, _, l, t in sec("unverifiable")])]

    def _last_read_claims(self, agent):
        for b in reversed(self.store.read_log("briefings")):
            if b["agent"] == agent:
                return [i["id"] for i in b["items"] if i["kind"] == "claim"]
        return []

    # ================================================================== pull tools
    def tool_specs(self):
        return [
            {"name": "lookup", "args": {"entity": "str"}, "doc": "status of an entity: measured value, failed, being measured, or unknown"},
            {"name": "search", "args": {"query": "str"}, "doc": "claims related to a question, with their layer and evidence counts"},
            {"name": "evidence", "args": {"claim_id": "str"}, "doc": "observations for and against a claim"},
            {"name": "best", "args": {}, "doc": "best measured results of the team"},
        ]

    def call_tool(self, agent, name, args):
        if name == "lookup":
            key = self.adapter.canonical(str(args.get("entity", ""))) or f"raw:{args.get('entity')}"
            obs = self._obs_of(key)
            if obs:
                o = obs[0]
                return f"{key}: {'value ' + format(o['value'], '.4g') if o['status'] == 'ok' else 'failed'} (by {o.get('agent')})"
            holder = self.store.lease_holder(key)
            return f"{key}: being measured by {holder}" if holder else f"{key}: unknown (never measured)"
        if name == "search":
            claims = [c for c in self.store.all("claims") if c["layer"] in ACTIVE + ("unverifiable",)]
            ids = self.retriever.search(str(args.get("query", "")), [(c["id"], c["scoped_statement"]) for c in claims], 10)
            by = {c["id"]: c for c in claims}
            self.store.append("tool_reads", {"agent": agent, "t": self.clock(), "items": ids})
            return "\n".join(f"[{i}] ({by[i]['layer']}) {by[i]['scoped_statement']}" for i in ids) or "no related claim"
        if name == "evidence":
            c = self.store.get("claims", str(args.get("claim_id", "")))
            if not c:
                return "unknown claim"
            le = c.get("last_eval") or {}
            obs = {o["id"]: o for o in self.store.all("obs")}
            fmt = lambda ids: ", ".join(f"{obs[i]['entity']}={obs[i]['value']}" for i in ids if i in obs)  # noqa: E731
            return f"[{c['id']}] {c['layer']}: {c['scoped_statement']}\nfor: {fmt(le.get('support', []))}\nagainst: {fmt(le.get('contradict', []))}"
        if name == "best":
            return "\n".join(l for _, l in self.team_results_lines()[:20]) or "no result yet"
        return "unknown tool"

    # ================================================================== export
    def presented_knowledge(self):
        out = []
        versions = self.store.read_log("claim_versions")
        for c in self.store.all("claims"):
            hist = [v for v in versions if v["id"] == c["id"]]
            if any(v["layer"] in ASSERTED + ("refuted", "hypothesis") for v in hist):
                out.append({"id": c["id"], "text": c["scoped_statement"], "original": c["statement"],
                            "author": c["author"], "final_layer": c["layer"], "history": hist,
                            "test_code": c.get("test_code")})
        return out

    def export_extra(self):
        s = self.store
        return {"claims": s.all("claims"), "claim_versions": s.read_log("claim_versions"),
                "dossiers": s.read_log("dossiers"), "journal": s.read_log("journal"),
                "edges": s.read_log("edges"), "directives": s.all("directives"),
                "tool_reads": s.read_log("tool_reads")}
