"""
Judges of claims.

GroundedJudge (the Steward's judge) works in two LLM steps around one code step:
  1. formalize : the LLM restates the claim within the scope of what was observed and writes the
                 executable test `def test(df)` over the observation table. No claim type is
                 predefined: the test is whatever the claim requires.
  2. (code)    : the memory runs the test in the sandbox and classifies each returned observation
                 as exploratory / confirmatory / independent from provenance (time, agent).
  3. verdict   : the LLM reads the claim, its test, the statistics and the evidence rows, and
                 returns a verdict citing observation ids.
  An optional adversarial review (skeptic, then reconsideration) precedes high-impact transitions.

TextJudge is the ablation that judges claims from text alone (statement and related claims), as
LLM-managed memories do. Both return the same structures, so the memory treats them alike.
"""
from __future__ import annotations

import json
import os

from .llm import LLM, extract_code, extract_json, is_truncated, long_output_tokens

VERDICTS = ("supported", "contradicted", "inconclusive", "untestable")


def _rows_budget() -> int:
    """How many evidence rows fit in the judge prompt, derived from the model context window
    (NUM_CTX tokens; one compact evidence row is a few tens of tokens) and leaving half for the rest."""
    return max(1, int(os.environ.get("NUM_CTX", "8192")) // 2 // 30)


FORMALIZE_SYSTEM = """You maintain the shared memory of a team of research agents.
A team member stated a claim. Turn it into an executable test over the team's observations.

Rules:
- Restate the claim so that it says no more than the observations can support (scoped_statement):
  keep its meaning and its direction, but restrict it to the part of the space it is actually about.
  Never turn the claim into its opposite: if the data disagree, the test will show it.
- Write a Python function `def test(df):` where df is a pandas DataFrame of observations.
  It must return a dict with:
    "in_scope":   list of observation ids (column "id") that the claim is about,
    "support":    ids among in_scope that agree with the claim,
    "contradict": ids among in_scope that disagree with the claim,
    "stats":      a small dict of numbers that summarize the evidence (counts, means, effect sizes...),
    "note":       one sentence on how the test decides.
  Use only failed/ok status and value columns and the descriptor columns. pandas is available as pd,
  numpy as np, math and statistics too. Do not read files or use the network.
  For claims about a group effect, compare in-scope observations against a reference group and
  decide support/contradiction per observation relative to that reference; explain in "note".
  Keep the test short: no comments, no docstrings, no prints. "support" and "contradict" must be
  subsets of "in_scope".
- If the claim cannot be tested with these observations at all (a methodological advice, a plan),
  set "testable": false and give no code.
- Compare with the existing related claims and say if the new claim duplicates one, refines one
  (narrower or corrected version), contradicts one, or is new.

Answer with a JSON object, then the code in a ```python block:
{"testable": true, "scoped_statement": "...", "relation": {"type": "new|duplicate|refines|contradicts", "target": "<claim id or null>"}}
```python
def test(df):
    ...
```"""

VERDICT_SYSTEM = """You are the judge of a team's shared memory. Decide whether the observations support a claim.
You get the claim, the test that was run, its statistics and the evidence rows. Each row has an
evidence class from provenance:
  E = exploratory (it existed when the claim was written, it may have suggested the claim),
  C = confirmatory (measured after the claim was written, by its author),
  I = independent confirmatory (measured after the claim was written, by another agent).
Judge the claim as a careful scientist: consider sample size, spread, whether the evidence comes from
a narrow part of the space, and whether the test really measures what the claim says.
Answer only with JSON:
{"verdict": "supported|contradicted|inconclusive", "cite_support": [ids], "cite_contradict": [ids],
 "rationale": "two or three sentences, citing ids"}
Cite only ids that appear in the evidence rows. Cite the rows your verdict rests on."""

REVIEW_SKEPTIC = """You are a skeptical reviewer. A judge proposes a verdict on a claim of a team's shared memory.
Find the strongest reasons the proposed verdict could be wrong (small samples, confounders, a test that
does not match the claim, evidence from one agent or one region only). Be concrete and cite ids. Be brief."""

TEXT_SYSTEM = """You maintain the shared memory of a team of research agents. Decide whether a new claim
should be accepted, based on its wording and on the claims already in memory.
Answer only with JSON:
{"verdict": "supported|contradicted|inconclusive|untestable",
 "relation": {"type": "new|duplicate|refines|contradicts", "target": "<claim id or null>"},
 "rationale": "one or two sentences"}"""


class GroundedJudge:
    name = "grounded"

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def formalize(self, statement: str, schema_doc: str, sample_rows: list[dict], related: list[dict],
                  goal_text: str, error: str | None = None, previous_code: str | None = None) -> dict:
        user = (f"GOAL OF THE TEAM: {goal_text}\n\nCLAIM: {statement}\n\n"
                f"OBSERVATION TABLE COLUMNS:\n{schema_doc}\n\n"
                f"SAMPLE ROWS:\n{json.dumps(sample_rows, default=str)[:4000]}\n\n"
                "RELATED CLAIMS IN MEMORY:\n" +
                ("\n".join(f"- {c['id']}: {c['statement']} [{c['layer']}]" for c in related) or "(none)"))
        if error:
            user += f"\n\nYOUR PREVIOUS TEST FAILED:\n{previous_code}\nERROR: {error}\nFix it."
        answer = self.llm.complete(FORMALIZE_SYSTEM, user, purpose="judge", max_tokens=long_output_tokens())
        meta = extract_json(answer.split("def test")[0]) or extract_json(answer) or {}
        code = extract_code(answer)
        if "def test" not in code:
            code = ""
        rel = meta.get("relation") if isinstance(meta.get("relation"), dict) else {}
        return {"testable": bool(meta.get("testable", bool(code))) and bool(code),
                "scoped_statement": str(meta.get("scoped_statement") or statement),
                "relation": {"type": str(rel.get("type") or "new"), "target": rel.get("target")},
                "code": code, "truncated": is_truncated(answer), "raw": answer}

    def verdict(self, claim: dict, result: dict, rows: list[dict], domain_note: str | None) -> dict:
        limit = _rows_budget()
        shown = rows[:limit]
        table = "\n".join(f"{r['id']} | {r['entity']} | {r['value']} | {r['agent']} | {r['class']} | {r['role']}"
                          for r in shown)
        user = (f"CLAIM ({claim['id']}): {claim['scoped_statement']}\n\nTEST:\n{claim['test_code']}\n\n"
                f"TEST NOTE: {result.get('note')}\nSTATS: {json.dumps(result.get('stats'))}\n"
                f"COUNTS: in_scope={len(result['in_scope'])} support={len(result['support'])} "
                f"contradict={len(result['contradict'])}\n\n"
                f"EVIDENCE ROWS (id | entity | value | agent | class | role)"
                f"{' — first %d of %d' % (len(shown), len(rows)) if len(rows) > len(shown) else ''}:\n{table}")
        if domain_note:
            user += f"\n\nPRIOR DOMAIN KNOWLEDGE (may contest, never refute, a claim):\n{domain_note}"
        answer = self.llm.complete(VERDICT_SYSTEM, user, purpose="judge", max_tokens=long_output_tokens())
        return self._parse_verdict(answer)

    def review(self, claim: dict, proposal: dict, result: dict, rows: list[dict]) -> dict:
        table = "\n".join(f"{r['id']} | {r['entity']} | {r['value']} | {r['agent']} | {r['class']} | {r['role']}"
                          for r in rows[:_rows_budget()])
        ctx = (f"CLAIM ({claim['id']}): {claim['scoped_statement']}\nTEST:\n{claim['test_code']}\n"
               f"STATS: {json.dumps(result.get('stats'))}\nEVIDENCE ROWS:\n{table}\n\n"
               f"PROPOSED VERDICT: {json.dumps(proposal)}")
        critique = self.llm.complete(REVIEW_SKEPTIC, ctx, purpose="judge")
        answer = self.llm.complete(VERDICT_SYSTEM, ctx + f"\n\nSKEPTICAL REVIEW:\n{critique}\n\n"
                                   "Reconsider and give your final verdict.", purpose="judge")
        final = self._parse_verdict(answer)
        final["review"] = critique[:2000]
        return final

    @staticmethod
    def _parse_verdict(answer: str) -> dict:
        d = extract_json(answer) or {}
        v = str(d.get("verdict", "inconclusive")).lower()
        return {"verdict": v if v in VERDICTS else "inconclusive",
                "cite_support": [str(x) for x in (d.get("cite_support") or [])],
                "cite_contradict": [str(x) for x in (d.get("cite_contradict") or [])],
                "rationale": str(d.get("rationale", ""))[:2000], "raw": answer[:4000]}


class TextJudge:
    """Ablation: judges a claim from text only (no test, no evidence)."""
    name = "text"

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def judge_text(self, statement: str, related: list[dict], goal_text: str) -> dict:
        user = (f"GOAL OF THE TEAM: {goal_text}\n\nNEW CLAIM: {statement}\n\nCLAIMS IN MEMORY:\n" +
                ("\n".join(f"- {c['id']}: {c['statement']} [{c['layer']}]" for c in related) or "(none)"))
        answer = self.llm.complete(TEXT_SYSTEM, user, purpose="judge")
        d = extract_json(answer) or {}
        v = str(d.get("verdict", "inconclusive")).lower()
        rel = d.get("relation") if isinstance(d.get("relation"), dict) else {}
        return {"verdict": v if v in VERDICTS else "inconclusive",
                "relation": {"type": str(rel.get("type") or "new"), "target": rel.get("target")},
                "rationale": str(d.get("rationale", ""))[:1000], "raw": answer[:3000]}
