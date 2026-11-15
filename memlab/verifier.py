"""
Knowledge verification layer for a shared agent memory (the contribution of this branch).

Every memory the host memory (Mem0) stores is checked against the workflow's measurements
(provenance records), not against text:

  1. Formalization (LLM, once per memory, batched): the memory is classified as a FACT about specific
     measurements, a GENERALIZATION, or UNTESTABLE; it is restated within the scope of what it is about;
     and the LLM writes `def test(df)`, an executable test over the observation table that returns the
     evidence for and against, AND the verdict computed by a decision rule written now, from the
     statement alone. The rule is pre-registered: it is fixed before the measurements that will
     confirm or refute the memory exist.
  2. Evaluation (code, no LLM): the tests are re-run whenever new measurements arrive (all tests in
     one sandboxed process). The verdict is the one the pre-registered rule computes.
  3. Status (code): the provenance of each piece of cited evidence (when it was measured, by whom)
     decides which status a verdict can justify:
        exploratory   measured before the memory existed (it may have suggested it)
        confirmatory  measured after the memory existed
        independent   confirmatory, and measured by another agent than the memory's author
     A generalization supported only by exploratory evidence stays a HYPOTHESIS; with confirmatory
     evidence it is SUPPORTED; with independent confirmation it is ESTABLISHED. A fact matching its
     measurement is VERIFIED; a fact with no matching measurement is UNSUPPORTED. Contradicted
     memories are REFUTED. Nothing is deleted from the host memory.

The ablation `text` mode replaces all of this by an LLM that judges the memory's wording alone.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from .llm import LLM, extract_code, extract_json, long_output_tokens
from .sandbox import run_many

ASSERTED = ("verified", "supported", "established", "accepted")
RANK = {"hypothesis": 1, "supported": 2, "established": 3}

FORMALIZE_SYSTEM = """You check the memories of a team of research agents against the team's measurements.
For EACH memory below, write a section that starts with a line "### <memory id>", then a JSON line:
{"kind": "fact|generalization|untestable", "scoped_statement": "..."}
then, unless the kind is untestable, a ```python block that defines `def test(df):`.

Kinds:
- fact: about specific measured designs or values (e.g. "design X gave 2.5", "design Y failed");
- generalization: about how options or features affect the outcome, beyond the designs named;
- untestable: a plan, an advice, or anything that measurements cannot confirm or refute.

scoped_statement: the memory restated so that it says no more than it is about. Keep its meaning and its
direction; never turn it into its opposite.

The test receives df, a pandas DataFrame with one row per measurement (columns listed below). It returns:
{"in_scope": [ids the memory is about], "support": [ids in scope that agree], "contradict": [ids in scope
that disagree], "stats": {a few numbers}, "verdict": "supported" | "contradicted" | "inconclusive",
"note": "one sentence"}.
The verdict MUST be computed inside the test by a rule you write now, from the statement alone: the test
will be re-run later on new measurements, and the rule must still be right then. Make the rule demand
enough evidence for the claim it checks (for a generalization, a comparison with a reference group).
For a fact, compare with the measured value at the precision the memory states.
Use only the columns listed. pandas is pd, numpy is np. Keep each test short: no comments, no prints."""

TEXT_SYSTEM = """You review the memories of a team of research agents. For EACH memory below, judge from its
wording and from the other memories whether it is likely true. Write one line per memory:
### <memory id> {"verdict": "true|false|uncertain"}"""


@dataclass
class VerificationConfig:
    mode: str = "grounded"            # grounded | text
    layers: bool = True               # exploratory/confirmatory/independent distinction
    min_independent_agents: int = 1   # agents other than the author needed for "established"
    test_timeout_s: float = 120.0
    repair: bool = True               # rewrite a test once when it fails to run
    tokens_per_test: int = 400        # used to size formalization batches (approximate test length)


class Verifier:
    def __init__(self, store, adapter, llm: LLM | None, cfg: VerificationConfig, clock=time.time) -> None:
        self.store, self.adapter, self.llm, self.cfg, self.clock = store, adapter, llm, cfg, clock

    # ------------------------------------------------------------------ inputs
    def register(self, mem_id: str, text: str, author: str | None) -> None:
        if self.store.get("vmem", mem_id):
            return
        self.store.put("vmem", mem_id, {"id": mem_id, "text": text, "author": author, "registered_at": self.clock(),
                                        "status": "pending", "kind": None, "scoped": text, "code": "",
                                        "eval": None, "repaired": False, "history": []})
        self.store.append("v_pending", {"id": mem_id})

    def forget(self, mem_id: str) -> None:
        d = self.store.get("vmem", mem_id)
        if d:
            self._set(d, "deleted", {"reason": "deleted by the host memory"})

    def status_of(self, ids: list[str]) -> dict[str, dict]:
        out = {}
        for i in ids:
            d = self.store.get("vmem", i)
            if d:
                out[i] = d
        return out

    # ------------------------------------------------------------------ work
    def step(self) -> int:
        if self.llm is None:
            return 0
        n = self._formalize_pending()
        if self.cfg.mode == "grounded":
            n += self._evaluate()
        return n

    def _pending_batch(self) -> list[dict]:
        start = int((self.store.get("meta", "v_cursor") or {}).get("i", 0))
        items = self.store.read_log("v_pending", start)
        self.store.put("meta", "v_cursor", {"i": start + len(items)})
        docs = [self.store.get("vmem", it["id"]) for it in items]
        return [d for d in docs if d and d["status"] == "pending"]

    def _formalize_pending(self) -> int:
        pending = self._pending_batch()
        if not pending:
            return 0
        size = max(1, long_output_tokens() // max(1, self.cfg.tokens_per_test))
        for i in range(0, len(pending), size):
            batch = pending[i:i + size]
            if self.cfg.mode == "text":
                self._text_judge(batch)
            else:
                self._formalize(batch)
        return len(pending)

    def _formalize(self, batch: list[dict], errors: dict[str, str] | None = None) -> None:
        user = self._context() + "\n\nMEMORIES:\n" + "\n".join(
            f"### {d['id']} (written by {d['author']})\n{d['text']}" +
            (f"\nYOUR PREVIOUS TEST FAILED: {errors[d['id']]}\nPREVIOUS TEST:\n{d['code']}" if errors and d['id'] in errors else "")
            for d in batch)
        answer = self.llm.complete(FORMALIZE_SYSTEM, user, purpose="judge", max_tokens=long_output_tokens())
        sections = self._sections(answer, [d["id"] for d in batch])
        for d in batch:
            sec = sections.get(d["id"])
            if sec is None:
                self._set(d, "unverifiable", {"reason": "no test written (answer missing or cut)"})
                continue
            meta = extract_json(sec.split("```")[0]) or {}
            kind = str(meta.get("kind", "")).lower()
            code = extract_code(sec) if "```" in sec else ""
            d["kind"] = kind if kind in ("fact", "generalization", "untestable") else ("untestable" if not code else "generalization")
            d["scoped"] = str(meta.get("scoped_statement") or d["text"])
            d["code"] = code if "def test" in code else ""
            self.store.append("v_journal", {"t": self.clock(), "id": d["id"], "step": "formalize",
                                            "kind": d["kind"], "scoped": d["scoped"], "code": d["code"]})
            if d["kind"] == "untestable" or not d["code"]:
                self._set(d, "unverifiable", {"reason": "untestable" if d["kind"] == "untestable" else "no test"})
            else:
                self.store.put("vmem", d["id"], d)

    def _text_judge(self, batch: list[dict]) -> None:
        others = [d for d in self.store.all("vmem") if d["status"] in ASSERTED][-30:]
        user = ("MEMORIES ALREADY ACCEPTED:\n" + ("\n".join(f"- {d['text']}" for d in others) or "(none)") +
                "\n\nMEMORIES TO REVIEW:\n" + "\n".join(f"### {d['id']}\n{d['text']}" for d in batch))
        answer = self.llm.complete(TEXT_SYSTEM, user, purpose="judge", max_tokens=long_output_tokens())
        sections = self._sections(answer, [d["id"] for d in batch])
        for d in batch:
            v = str((extract_json(sections.get(d["id"], "")) or {}).get("verdict", "uncertain")).lower()
            self._set(d, {"true": "accepted", "false": "refuted"}.get(v, "unverifiable"), {"verdict": v})

    @staticmethod
    def _sections(answer: str, ids: list[str]) -> dict[str, str]:
        out = {}
        for i in ids:
            m = re.search(r"###\s*" + re.escape(i) + r"\b(.*?)(?=\n###\s|\Z)", answer, re.DOTALL)
            if m:
                out[i] = m.group(1)
        return out

    def _context(self) -> str:
        obj = self.adapter.goal.objective
        rows = self._rows()
        return ("GOAL OF THE TEAM: " + self.adapter.goal.text + "\n\nMEASUREMENT TABLE COLUMNS:\n"
                "id (measurement id), entity (canonical design key), value (" + obj.name + ", " +
                ("higher is better" if obj.direction == "max" else "lower is better") +
                "; null when the experiment failed), status ('ok' or 'failed'), agent, t (time)\n" +
                self.adapter.descriptor_doc() + "\nLAST ROWS:\n" + json.dumps(rows[-15:], default=str)[:3000])

    def _rows(self) -> list[dict]:
        rows = []
        for o in sorted(self.store.all("obs"), key=lambda o: o["t"]):
            rows.append({"id": str(o["id"]), "entity": o["entity"], "value": o["value"], "status": o["status"],
                         "agent": o.get("agent"), "t": o["t"], **(o.get("descriptors") or {})})
        return rows

    # ------------------------------------------------------------------ evaluation
    def _evaluate(self) -> int:
        n_obs = self.store.log_len("obs_log")
        todo = [d for d in self.store.all("vmem")
                if d["code"] and d["status"] not in ("unverifiable", "deleted")
                and (d["eval"] is None or d["eval"]["n_obs"] < n_obs)]
        if not todo:
            return 0
        rows = self._rows()
        results = run_many({d["id"]: d["code"] for d in todo}, rows, self.cfg.test_timeout_s)
        by_id = {r["id"]: r for r in rows}
        failed = {}
        for d in todo:
            res = results.get(d["id"], {"error": "no result"})
            if "error" in res:
                if self.cfg.repair and not d["repaired"]:
                    d["repaired"] = True
                    failed[d["id"]] = res["error"][:500]
                    self.store.put("vmem", d["id"], d)
                else:
                    self._set(d, "unverifiable", {"reason": "test error", "error": res["error"][:500]})
                continue
            self._apply(d, res, by_id, len(rows))
        if failed:
            docs = [self.store.get("vmem", i) for i in failed]
            self._formalize(docs, errors=failed)
        return len(todo)

    def classify(self, d: dict, ids: list[str], by_id: dict) -> list[str]:
        out = []
        for i in ids:
            r = by_id[i]
            if r["t"] <= d["registered_at"]:
                out.append("E")
            elif r.get("agent") != d["author"]:
                out.append("I")
            else:
                out.append("C")
        return out

    def _apply(self, d: dict, res: dict, by_id: dict, n_obs: int) -> None:
        v = res["verdict"]
        if v == "supported" and not res["support"]:
            v = "inconclusive"           # a verdict must rest on evidence the test returned
        if v == "contradicted" and not res["contradict"]:
            v = "inconclusive"
        sup_cls = self.classify(d, res["support"], by_id)
        con_cls = self.classify(d, res["contradict"], by_id)
        indep_agents = {by_id[i].get("agent") for i, c in zip(res["support"], sup_cls) if c == "I"}
        prev = d["status"]
        if d["kind"] == "fact":
            new = ("unsupported" if not res["in_scope"] else
                   "verified" if v == "supported" else "refuted" if v == "contradicted" else "unverifiable")
        elif v == "contradicted":
            new = "refuted"
        elif v == "supported":
            if not self.cfg.layers:
                new = "established"
            else:
                level = ("established" if len(indep_agents) >= max(1, self.cfg.min_independent_agents)
                         else "supported" if any(c in "CI" for c in sup_cls) else "hypothesis")
                if prev == "refuted" and not any(c in "CI" for c in sup_cls):
                    level = "refuted"
                elif prev in RANK and RANK[prev] > RANK[level]:
                    level = prev            # a favourable verdict never demotes
                new = level
        else:                                # inconclusive
            if prev == "established" and any(c in "CI" for c in con_cls):
                new = "supported"            # contested by confirmatory evidence
            elif prev in ("pending", "unverifiable"):
                new = "hypothesis" if self.cfg.layers else "established"
            else:
                new = prev
        d["eval"] = {"n_obs": n_obs, "verdict": v, "support": res["support"], "contradict": res["contradict"],
                     "classes": {"support": "".join(sup_cls), "contradict": "".join(con_cls)},
                     "independent_agents": sorted(a for a in indep_agents if a), "stats": res.get("stats"),
                     "note": res.get("note"), "t": self.clock()}
        self._set(d, new, {"verdict": v, "n_support": len(res["support"]), "n_contradict": len(res["contradict"]),
                           "classes": d["eval"]["classes"]})

    def _set(self, d: dict, status: str, info: dict) -> None:
        old = d["status"]
        if status != old:
            d["history"].append({"t": self.clock(), "from": old, "to": status, **info})
            self.store.append("v_dossiers", {"t": self.clock(), "id": d["id"], "from": old, "to": status,
                                             "text": d["text"], "scoped": d["scoped"], **info})
            if old in ASSERTED and status not in ASSERTED:
                self.store.append("v_demotions", {"t": self.clock(), "id": d["id"], "from": old, "to": status,
                                                  "text": d["scoped"]})
        d["status"] = status
        self.store.put("vmem", d["id"], d)

    def export(self) -> dict:
        return {"verified_memories": self.store.all("vmem"), "dossiers": self.store.read_log("v_dossiers"),
                "journal": self.store.read_log("v_journal"), "demotions": self.store.read_log("v_demotions"),
                "config": self.cfg.__dict__}
