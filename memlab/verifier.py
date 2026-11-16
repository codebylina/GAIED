"""
Knowledge verification layer for a shared agent memory (the contribution of this branch).

Every memory the host memory (Mem0) stores is checked against the workflow's measurements
(provenance records), not against text.

  1. Formalization (LLM, once per memory, batched). The memory is classified (report of measurements,
     prediction about specific designs, generalization, untestable), restated within its scope, and the
     LLM writes `def test(df)`: an executable test over the measurement table that returns the evidence
     for and against AND the verdict computed by a decision rule written now.
     Pre-registration is enforced, not assumed: the LLM only sees the measurements that existed when the
     memory was written (its claim time), whatever the delay before formalization; the rule is frozen
     with its hash, its version and the time window it was written from; a repair (test that fails to
     run) sees the same window and a sanitized error (no measured values), and must keep the statement
     and the rule. Every rule version is kept.
  2. Evaluation (code, no LLM): the frozen tests are re-run whenever new measurements arrive (all tests
     in one sandboxed process). With decision="code" (default) the test only returns what the memory predicts
     (matched pairs, two groups, failure rates or facts) and a fixed statistical rule written in code decides
     (memlab/evidence_test.py: sign test on disjoint pairs, Mann-Whitney, Fisher, facts at the precision written;
     significance level alpha). With decision="llm" (ablation, and the first version) the verdict is the one the
     rule written by the LLM computes.
  3. Evidence classes, from provenance (who measured what, when, after seeing what):
        E  exploratory   available before the claim time, or a re-measurement of a design measured
                         before it (same source: it may have suggested the claim)
        C  confirmatory  a new design, measured after the claim time by the claim's author
        R  informed      a new design, measured by another agent who had been shown the claim before
                         starting the experiment (its choice may depend on the claim)
        I  independent   a new design, measured by another agent who had not been shown the claim
  4. Status (a function of the current verdict and of the evidence classes):
        reports / predictions   verified | refuted | unsupported (a report with no measurement on record)
                                | open (no or insufficient evidence yet)
        generalizations         refuted (significant against) | established (significant on the measurements made
                                after the claim, with >= k independent agents) | supported (significant on new data,
                                or on all data with the new data pointing the same way) | hypothesis (significant
                                only on the measurements that suggested it)
                                | contested (was asserted, the rule no longer finds it supported)
                                | open (no or insufficient evidence)
        any                     unverifiable (untestable, or the test fails twice) | pending | deleted
     The evidence state (none | insufficient | favourable | contradictory) is stored next to the status.
     Nothing is deleted from the host memory.

Ablations: `mode="text"` (an LLM judges the wording only), `layers=False` (any supported verdict
establishes), `independence="agent"` (any other agent counts as independent, exposure ignored),
`preregistered=False` (the LLM rewrites the test with all current measurements whenever new ones arrive).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass

from .llm import LLM, extract_code, extract_json, long_output_tokens
from .sandbox import run_many

ASSERTED = ("verified", "supported", "established", "accepted")
KINDS = ("report", "prediction", "generalization", "untestable")

FORMALIZE_SYSTEM = """You check the memories of a team of research agents against the team's measurements.
For EACH memory below, write a section that starts with a line "### <memory id>", then a JSON line:
{"kind": "report|prediction|generalization|untestable", "scoped_statement": "..."}
then, unless the kind is untestable, a ```python block that defines `def test(df):`.

Kinds:
- report: states the outcome of experiments already run (e.g. "design X gave 2.5", "design Y failed");
- prediction: states the outcome of specific designs, which may not have been measured yet;
- generalization: how options or features relate to the outcome, beyond the designs named;
- untestable: a plan, an advice, or anything that measurements cannot confirm or refute.

scoped_statement: the memory restated so that it says no more than it is about: an association observed on
designs like the ones measured, not a causal law. Keep its meaning and its direction; never turn it into
its opposite.

The test receives df, a pandas DataFrame with one row per measurement (columns listed below). It returns:
{"in_scope": [...], "support": [...], "contradict": [...], "stats": {a few numbers},
"verdict": "supported" | "contradicted" | "insufficient", "note": "one sentence"}.
in_scope, support and contradict are lists of values of df['id'] (the measurement rows the memory is about, those
that agree, those that disagree). Never put the memory's id (the "### ..." name) in them.
Select rows by conditions on the columns, never by listing measurement ids: the test will be re-run, unchanged, on
every later measurement. A generalization selects by descriptors (slots, features), so it also covers designs that
do not exist yet; a report or a prediction about named designs selects them by their entity key. The verdict MUST be computed inside the test by a decision rule you write now, from the statement alone;
you only see the measurements that existed when the memory was written, so the rule must be right for data you have
not seen. For a generalization, compare the designs it is about with a reference group: designs equal on the other
descriptors when such pairs exist, otherwise all the other measured designs (say which in "note"); demand enough
evidence for the noise of the measurements, and return "insufficient" otherwise. For a report or a prediction,
compare with the measured value at the precision the memory states.
Use only the columns listed. pandas is pd, numpy is np. Keep each test short: no comments, no prints."""

FORMALIZE_SYSTEM_CODE = """You check the memories of a team of research agents against the team's measurements.
For EACH memory below, write a section that starts with a line "### <memory id>", then a JSON line:
{"kind": "report|prediction|generalization|untestable", "scoped_statement": "..."}
then, unless the kind is untestable, a ```python block that defines `def test(df):`.

Kinds:
- report: states the outcome of experiments already run (e.g. "design X gave 2.5", "design Y failed");
- prediction: states the outcome of specific designs, which may not have been measured yet;
- generalization: how options or features relate to the outcome, beyond the designs named;
- untestable: a plan, an advice, or a statement that no measurement could contradict (e.g. "comparable or
  slightly better", "may help", "is a robust choice").

scoped_statement: the memory restated so that it says no more than it is about: an association observed on
designs like the ones measured, not a causal law. Keep its meaning and its direction; never turn it into
its opposite.

You do NOT decide whether the memory is true. The test only says WHICH measurements the memory makes a prediction
about and WHAT it predicts; the code then compares the prediction with the measured outcomes and decides, with the
same statistical rule for every memory. The test receives df, one row per measurement (columns below; df.index is
df['id']). It returns a dict with "note" (one sentence) and exactly ONE of these keys:
- "pairs": [[id_a, id_b], ...] : the memory predicts value(a) > value(b) for each pair. Use it whenever designs can
  be compared two by two, equal on every other descriptor: pairs(frame_a, frame_b, vary=[columns]) returns the
  pairs of rows of frame_a and frame_b that are equal on all the descriptor columns except those in vary.
- "groups": {"higher": [ids], "lower": [ids]} : values in "higher" tend to exceed those in "lower" (when no matched
  pairs exist, e.g. the designs with an option against the designs with the other options of the same slot).
- "fail_rate": {"more": [ids], "less": [ids]} : designs in "more" fail more often than those in "less".
- "facts": [{"id": id, "status": "ok" or "failed", "value": "<the number exactly as written in the memory>" or
  null}, ...] : the outcome of these measurements (reports, and predictions about named designs).
"higher", "lower" and "value(a) > value(b)" are about the value itself; the table header says whether a higher
value is better, so "improves" means a higher value only when higher is better.

Rules:
- ids are values of df['id']: ids(frame) returns them. Never values, never the memory's id.
- Select rows by conditions on the columns, never by listing ids: the test is re-run, unchanged, on every later
  measurement. A generalization selects by descriptors, so it also covers designs not measured yet; a report or a
  prediction selects its named designs by their entity key.
- In "pairs" and "groups" keep the rows with status 'ok'; a claim about failures uses "fail_rate" or "facts".
- Iterating over a DataFrame gives its column names: use ids(frame), frame.iterrows() or frame.itertuples().
Example (memory: "option 1 of slot X gives a higher outcome than option 0"):
def test(df):
    ok = df[df['status'] == 'ok']
    return {"pairs": pairs(ok[ok['X'] == 1], ok[ok['X'] == 0], vary=['X']), "note": "X=1 vs X=0, other columns equal"}
Use only the columns listed. pandas is pd, numpy is np. Keep each test short: no comments, no prints."""

REPAIR_NOTE = ("YOUR TEST FAILED: {error}\n{hint}YOUR TEST:\n{code}\nFix the error only: keep the same kind, the "
               "same scoped statement and the same meaning. Remember: lists contain values of df['id'] (ids(frame)); "
               "rows are selected by conditions on the columns, never by measurement ids.")

REPAIR_HINTS = (
    ("string indices must be integers", "Iterating over a DataFrame gives its column names, not its rows: use "
                                        "ids(frame), frame.iterrows() or frame.itertuples()."),
    ("truth value of a Series is ambiguous", "Combine conditions with & and | (with parentheses), and test a frame "
                                             "with len(frame) > 0, .any() or .all()."),
    ("KeyError: 'id'", "Only a DataFrame has the column 'id': select the rows with df[mask] before ids(...)."),
    ("which is a number", "Put ids(frame) in the lists, never frame['value']."),
    ("SyntaxError", "Check that every bracket and parenthesis is closed."),
    ("return exactly one of the keys", "Return one comparison only: pairs, groups, fail_rate or facts."),
)


def repair_hint(error: str) -> str:
    return "".join(h + "\n" for k, h in REPAIR_HINTS if k in error)


_LITERAL_ID = re.compile(r"""['"][0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}['"]""")


def static_problem(code: str, mem_id: str) -> str | None:
    """Problems visible in the code itself, which would make a test unable to judge later measurements."""
    if mem_id and re.search(r"""['"]""" + re.escape(mem_id) + r"""['"]""", code):
        return "the test uses the memory's id; list values of df['id'] instead"
    if _LITERAL_ID.search(code):
        return "the test selects measurements by their ids; select rows by conditions on the columns instead"
    return None

TEXT_SYSTEM = """You review the memories of a team of research agents. For EACH memory below, judge from its
wording and from the other memories whether it is likely true. Write one line per memory:
### <memory id> {"verdict": "true|false|uncertain"}"""


@dataclass
class VerificationConfig:
    mode: str = "grounded"            # grounded | text
    layers: bool = True               # E/C/R/I distinction
    independence: str = "exposure"    # exposure: an agent shown the claim is informed (R) | agent: any other agent (I)
    preregistered: bool = True        # False: the test is rewritten with all current data at each new measurement
    min_independent_agents: int = 1   # distinct non-author agents with independent support for "established"
    test_timeout_s: float = 120.0
    repair: bool = True               # rewrite a test once when it fails to run
    tokens_per_test: int = 400        # used to size formalization batches (approximate test length)
    decision: str = "code"            # code: the test says what the memory predicts, a fixed statistical rule decides
                                      # (memlab/evidence_test.py) | llm: the test computes its own verdict (ablation)
    alpha: float = 0.05               # significance level of the code's decision rule (convention)


def sha(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def sanitize(error: str) -> str:
    """An error raised on later data must not leak measured values into a repair."""
    return re.sub(r"-?\d+\.\d+(e-?\d+)?", "<number>", str(error))[:300]


class Verifier:
    def __init__(self, store, adapter, llm: LLM | None, cfg: VerificationConfig, clock=time.time) -> None:
        self.store, self.adapter, self.llm, self.cfg, self.clock = store, adapter, llm, cfg, clock
        self._exposure: dict[tuple[str, str], float] = {}   # (agent, memory id) -> first time shown
        self._exp_cursor = 0

    # ------------------------------------------------------------------ inputs
    def register(self, mem_id: str, text: str, author: str | None, written_at: float | None = None) -> None:
        """written_at: when the agent wrote the text the memory comes from (its claim time)."""
        if self.store.get("vmem", mem_id):
            return
        now = self.clock()
        self.store.put("vmem", mem_id, {
            "id": mem_id, "text": text, "author": author, "registered_at": now,
            "claimed_at": float(written_at) if written_at is not None else now,
            "status": "pending", "evidence": "none", "kind": None, "scoped": text, "code": "",
            "rule": None, "rules": [], "eval": None, "repaired": False, "history": []})
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

    def _batches(self, docs: list[dict]):
        size = max(1, long_output_tokens() // max(1, self.cfg.tokens_per_test))
        for i in range(0, len(docs), size):
            yield docs[i:i + size]

    def _formalize_pending(self) -> int:
        pending = self._pending_batch()
        for batch in self._batches(pending):
            if self.cfg.mode == "text":
                self._text_judge(batch)
            else:
                self._formalize(batch, "initial")
        return len(pending)

    # ------------------------------------------------------------------ formalization
    def _window(self, d: dict) -> float | None:
        """Latest measurement time the LLM may see when writing this memory's rule."""
        return d["claimed_at"] if self.cfg.preregistered else None

    def _formalize(self, batch: list[dict], reason: str, errors: dict[str, str] | None = None) -> None:
        # one call per distinct window: memories written at different times see different data
        groups: dict = {}
        for d in batch:
            groups.setdefault(self._window(d), []).append(d)
        for until, docs in groups.items():
            context, max_t = self._context(until)
            user = context + "\n\nMEMORIES:\n" + "\n".join(
                f"### {d['id']} (written by {d['author']})\n{d['text']}" +
                ("\n" + REPAIR_NOTE.format(error=errors[d["id"]], hint=repair_hint(errors[d["id"]]), code=d["code"]) if errors and d["id"] in errors else "")
                for d in docs)
            system = FORMALIZE_SYSTEM_CODE if self.cfg.decision == "code" else FORMALIZE_SYSTEM
            answer = self.llm.complete(system, user, purpose="judge", max_tokens=long_output_tokens())
            sections = self._sections(answer, [d["id"] for d in docs])
            for d in docs:
                self._read_section(d, sections.get(d["id"]), reason, until, max_t)

    def _read_section(self, d: dict, sec: str | None, reason: str, until, max_t) -> None:
        if sec is None:
            self._set(d, "unverifiable", {"reason": "no test written (answer missing or cut)"})
            return
        meta = extract_json(sec.split("```")[0]) or {}
        code = extract_code(sec) if "```" in sec else ""
        code = code if "def test" in code else ""
        if reason == "initial" or not d["kind"]:     # a repair keeps the statement and its kind
            kind = str(meta.get("kind", "")).lower()
            kind = "report" if kind == "fact" else kind
            d["kind"] = kind if kind in KINDS else ("untestable" if not code else "generalization")
            d["scoped"] = str(meta.get("scoped_statement") or d["text"])
        d["code"] = code
        if code:
            rule = {"version": len(d["rules"]) + 1, "reason": reason, "sha256": sha(code), "frozen_at": self.clock(),
                    "window_until": until, "max_t_seen": max_t, "code": code}
            d["rule"] = {k: v for k, v in rule.items() if k != "code"}
            d["rules"].append(rule)
        self.store.append("v_journal", {"t": self.clock(), "id": d["id"], "step": reason, "kind": d["kind"],
                                        "scoped": d["scoped"], "code": code, "window_until": until, "max_t_seen": max_t})
        if d["kind"] == "untestable" or not code:
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

    def _context(self, until: float | None) -> tuple[str, float | None]:
        obj = self.adapter.goal.objective
        rows = [r for r in self._rows() if until is None or r["t"] <= until]
        shown, size = [], 0
        for r in reversed(rows):                     # the most recent rows of the window that fit
            s = json.dumps(r, default=str)
            if size + len(s) > 3000:
                break
            shown.insert(0, r)
            size += len(s) + 2
        text = ("GOAL OF THE TEAM: " + self.adapter.goal.text + "\n\nMEASUREMENT TABLE COLUMNS:\n"
                "id (measurement id), entity (canonical design key), value (" + obj.name + ", " +
                ("higher is better" if obj.direction == "max" else "lower is better") +
                "; null when the experiment failed), status ('ok' or 'failed'), agent, t (time)\n" +
                self.adapter.descriptor_doc() + f"\nMEASUREMENTS AVAILABLE WHEN THE MEMORIES WERE WRITTEN: {len(rows)}; "
                "LAST ROWS:\n" + json.dumps(shown, default=str))
        return text, max((r["t"] for r in rows), default=None)

    def _rows(self) -> list[dict]:
        rows = []
        for o in sorted(self.store.all("obs"), key=lambda o: o["t"]):
            rows.append({"id": str(o["id"]), "entity": o["entity"], "value": o["value"], "status": o["status"],
                         "agent": o.get("agent"), "t": o["t"], **(o.get("descriptors") or {})})
        return rows

    # ------------------------------------------------------------------ exposure (decision dependence)
    def _update_exposure(self) -> None:
        logs = self.store.read_log("briefings", self._exp_cursor)
        self._exp_cursor += len(logs)
        for b in logs:
            for it in b.get("items", []):
                key = (b["agent"], it["id"])
                if key not in self._exposure or b["t"] < self._exposure[key]:
                    self._exposure[key] = b["t"]

    def exposed_before(self, agent: str, mem_id: str, t: float) -> bool:
        e = self._exposure.get((agent, mem_id))
        return e is not None and e <= t

    # ------------------------------------------------------------------ evaluation
    def _evaluate(self) -> int:
        n_obs = self.store.log_len("obs_log")
        todo = [d for d in self.store.all("vmem")
                if d["code"] and d["status"] not in ("unverifiable", "deleted")
                and (d["eval"] is None or d["eval"]["n_obs"] < n_obs)]
        if not todo:
            return 0
        if not self.cfg.preregistered:               # ablation: the rule is rewritten after seeing the new data
            for batch in self._batches([d for d in todo if d["eval"] is not None]):
                self._formalize(batch, "posthoc")
            todo = [self.store.get("vmem", d["id"]) for d in todo]
            todo = [d for d in todo if d and d["code"] and d["status"] not in ("unverifiable", "deleted")]
        self._update_exposure()
        rows = self._rows()
        bad = {d["id"] for d in todo if not d.get("rule") or sha(d["code"]) != d["rule"]["sha256"]}
        static = {d["id"]: p for d in todo if d["id"] not in bad and (p := static_problem(d["code"], d["id"]))}
        results = run_many({d["id"]: d["code"] for d in todo if d["id"] not in bad and d["id"] not in static},
                           rows, self.cfg.test_timeout_s, form="comparison" if self.cfg.decision == "code" else "verdict")
        results.update({i: {"error": p} for i, p in static.items()})
        by_id = {r["id"]: r for r in rows}
        started = {str(o["id"]): float(o.get("started_at") or o["t"]) for o in self.store.all("obs")}
        failed = {}
        for d in todo:
            res = {"error": "rule changed after it was frozen"} if d["id"] in bad else results.get(d["id"], {"error": "no result"})
            if "error" in res:
                if self.cfg.repair and not d["repaired"] and d["id"] not in bad:
                    d["repaired"] = True
                    failed[d["id"]] = sanitize(res["error"])
                    self.store.put("vmem", d["id"], d)
                else:
                    self._set(d, "unverifiable", {"reason": "test error", "error": sanitize(res["error"])})
                continue
            if self.cfg.decision == "code":
                self._apply_code(d, res, by_id, started, len(rows))
            else:
                self._apply(d, res, by_id, started, len(rows))
        if failed:
            self._formalize([self.store.get("vmem", i) for i in failed], "repair", errors=failed)
        return len(todo)

    def classify(self, d: dict, ids: list[str], by_id: dict, started: dict) -> list[str]:
        before = {r["entity"] for r in by_id.values() if r["t"] <= d["claimed_at"]}
        out = []
        for i in ids:
            r = by_id[i]
            if r["t"] <= d["claimed_at"] or r["entity"] in before:
                out.append("E")
            elif r.get("agent") == d["author"]:
                out.append("C")
            elif self.cfg.independence == "exposure" and self.exposed_before(r.get("agent"), d["id"],
                                                                              started.get(i, r["t"])):
                out.append("R")
            else:
                out.append("I")
        return out

    def status_for(self, d: dict, verdict: str, in_scope: list, sup_cls: list[str], indep_agents: set,
                   prev: str) -> str:
        """The status a verdict and its evidence justify (the transition table of docs/VERIFICATION_LAYER.md)."""
        if d["kind"] in ("report", "prediction"):
            if not in_scope:
                return "unsupported" if d["kind"] == "report" else "open"
            return {"supported": "verified", "contradicted": "refuted"}.get(verdict, "open")
        if verdict == "contradicted":
            return "refuted"
        if verdict == "supported":
            if not self.cfg.layers:
                return "established"
            if len(indep_agents) >= max(1, self.cfg.min_independent_agents):
                return "established"
            return "supported" if any(c in "CRI" for c in sup_cls) else "hypothesis"
        return "contested" if prev in ASSERTED or prev == "contested" else "open"

    def _apply(self, d: dict, res: dict, by_id: dict, started: dict, n_obs: int) -> None:
        v = {"inconclusive": "insufficient"}.get(res["verdict"], res["verdict"])
        if v not in ("supported", "contradicted"):
            v = "insufficient"
        if v == "supported" and not res["support"]:
            v = "insufficient"           # a verdict must rest on evidence the test returned
        if v == "contradicted" and not res["contradict"]:
            v = "insufficient"
        sup_cls = self.classify(d, res["support"], by_id, started)
        con_cls = self.classify(d, res["contradict"], by_id, started)
        indep = {by_id[i].get("agent") for i, c in zip(res["support"], sup_cls) if c == "I"}
        indep.discard(None)
        new = self.status_for(d, v, res["in_scope"], sup_cls, indep, d["status"])
        d["evidence"] = ("none" if not res["in_scope"] else {"supported": "favourable",
                                                              "contradicted": "contradictory"}.get(v, "insufficient"))
        d["eval"] = {"n_obs": n_obs, "verdict": v, "in_scope": len(res["in_scope"]), "support": res["support"],
                     "contradict": res["contradict"],
                     "classes": {"support": "".join(sup_cls), "contradict": "".join(con_cls)},
                     "independent_agents": sorted(indep), "stats": res.get("stats"), "note": res.get("note"),
                     "rule_version": (d.get("rule") or {}).get("version"), "t": self.clock()}
        self._set(d, new, {"verdict": v, "evidence": d["evidence"], "n_support": len(res["support"]),
                           "n_contradict": len(res["contradict"]), "classes": d["eval"]["classes"],
                           "rule_version": d["eval"]["rule_version"]})

    # ------------------------------------------------------------------ decision by code (decision="code")
    def code_status(self, d: dict, comp: dict, all_: dict, new: dict | None, indep: set, prev: str) -> str:
        """The status the measurements justify, by the fixed rule of memlab/evidence_test.py.
        reports / predictions: every unit agrees -> verified, one disagrees -> refuted (facts and pairs are exact
        statements about named measurements); a group comparison -> verified / refuted when significant.
        generalizations: significant against (all data or new data) -> refuted; significant on the measurements made
        after the claim with >= k independent agents -> established; significant on the new measurements, or on all
        with the new ones pointing the same way -> supported; significant only on the measurements that suggested it
        -> hypothesis; otherwise open (contested if it was shown as true)."""
        a = self.cfg.alpha
        if d["kind"] in ("report", "prediction"):
            if all_["n"] == 0:
                return "unsupported" if d["kind"] == "report" else "open"
            if comp["form"] in ("facts", "pairs"):
                return "refuted" if all_["disagree"] else "verified"
            return "verified" if all_["p_for"] < a else "refuted" if all_["p_against"] < a else "open"
        if all_["p_against"] < a or (new and new["p_against"] < a):
            return "refuted"
        if not self.cfg.layers:
            if all_["p_for"] < a:
                return "established"
        elif new and new["p_for"] < a and len(indep) >= max(1, self.cfg.min_independent_agents):
            return "established"
        elif new and (new["p_for"] < a or (all_["p_for"] < a and new["p_for"] <= new["p_against"])):
            return "supported"
        elif all_["p_for"] < a:
            return "hypothesis"
        return "contested" if prev in ASSERTED or prev == "contested" else "open"

    def _apply_code(self, d: dict, comp: dict, by_id: dict, started: dict, n_obs: int) -> None:
        from .evidence_test import involved, restrict, stat
        if comp["form"] == "untestable":
            self._set(d, "unverifiable", {"reason": "untestable", "note": comp.get("note")})
            return
        ids = involved(comp)
        cls = dict(zip(ids, self.classify(d, ids, by_id, started)))
        all_ = stat(comp, by_id)
        new_ids = {i for i, c in cls.items() if c != "E"}
        sub = restrict(comp, new_ids) if d["kind"] == "generalization" else None
        new = stat(sub, by_id) if sub else None
        if new is not None and new["n"] == 0:
            new = None
        used_new = set(new["agree"] + new["disagree"]) if new else set()
        indep = {by_id[i].get("agent") for i in used_new if cls.get(i) == "I"}
        indep.discard(None)
        status = self.code_status(d, comp, all_, new, indep, d["status"])
        a = self.cfg.alpha
        if d["kind"] == "generalization" or comp["form"] in ("groups", "fail_rate"):
            v = "supported" if all_["p_for"] < a else "contradicted" if all_["p_against"] < a else "insufficient"
        else:
            v = "contradicted" if all_["disagree"] else "supported" if all_["agree"] else "insufficient"
        d["evidence"] = ("none" if all_["n"] == 0 else {"supported": "favourable",
                                                        "contradicted": "contradictory"}.get(v, "insufficient"))
        num = lambda x: round(x, 4) if isinstance(x, float) else x
        stats = {k: num(x) for k, x in all_.items() if k not in ("agree", "disagree")}
        if new:
            stats.update({f"new_{k}": num(x) for k, x in new.items() if k in ("n", "p_for", "p_against")})
        d["eval"] = {"n_obs": n_obs, "verdict": v, "form": comp["form"], "in_scope": len(ids),
                     "support": all_["agree"], "contradict": all_["disagree"],
                     "classes": {"support": "".join(cls.get(i, "E") for i in all_["agree"]),
                                 "contradict": "".join(cls.get(i, "E") for i in all_["disagree"])},
                     "independent_agents": sorted(indep), "stats": stats, "note": comp.get("note"),
                     "rule_version": (d.get("rule") or {}).get("version"), "t": self.clock()}
        self._set(d, status, {"verdict": v, "evidence": d["evidence"], "n_support": len(all_["agree"]),
                              "n_contradict": len(all_["disagree"]), "classes": d["eval"]["classes"],
                              "p_for": stats.get("p_for"), "new_p_for": stats.get("new_p_for"),
                              "rule_version": d["eval"]["rule_version"]})

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
                "exposure": [{"agent": a, "id": i, "t": t} for (a, i), t in self._exposure.items()],
                "config": self.cfg.__dict__}
