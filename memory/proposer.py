"""
Turning an agent's free-text conclusions into structured lessons.

The LLM only PROPOSES: it rewrites conclusions as lessons with a checkable claim and the observations
they rely on. Whether a lesson is reusable is decided afterwards by the verification rules.

This module does not call any LLM: it builds the prompt and parses the answer, so any workflow can use
its own model client. Observations are shown with short references (o1, o2...) that are mapped back
to the real observation ids (= Flowcept task ids) when parsing.
"""
from __future__ import annotations

import json
import re

from .domain import Domain
from .lessons import TYPES

SYSTEM = """You turn the conclusions of a research agent into structured, checkable lessons.
A lesson is useful for the other agents of the team only if it can be checked against the measurements.

Answer with ONLY a JSON object:
{"lessons": [{"statement": "...", "type": "success|failure|strategy|constraint",
              "claim": {...}, "evidence": ["o1", "o2"]}]}

"claim" is one of:
  {"kind": "feature_effect", "feature": "<feature name>", "op": ">", "value": 0, "effect": "higher"}
      entities whose feature satisfies (op value) have a higher (or "lower") objective value
  {"kind": "comparison", "better": "<entity>", "worse": "<entity>"}
      one measured entity is better than another
  {"kind": "none"}
      the lesson cannot be checked with the measurements (advice, plan)
"evidence" lists the references (o1, o2...) of the measurements the lesson is based on.
Give at most %d lessons. Only state what the measurements show. Prefer feature_effect and comparison.
Do not repeat a lesson that is obvious from a single measurement, and do not invent lessons:
if the measurements do not support any new lesson, answer exactly {"lessons": []}."""


class LessonProposer:
    def __init__(self, domain: Domain, max_lessons: int = 3, max_rows: int = 40) -> None:
        self.domain, self.max_lessons, self.max_rows = domain, max_lessons, max_rows

    def prompt(self, conclusions: str, observations: list[dict]) -> tuple[str, str, dict[str, str]]:
        """(system, user, refs): refs maps the short references shown to the LLM to observation ids."""
        rows = observations[-self.max_rows:]
        refs, lines = {}, []
        for i, o in enumerate(rows, 1):
            ref = f"o{i}"
            refs[ref] = o["id"]
            val = f"{o['value']:.6g}" if o["status"] == "success" else f"FAILED ({o.get('failure_kind')})"
            lines.append(f"{ref} | {o['entity']} | {val}")
        features = self.domain.feature_doc()
        user = (f"OBJECTIVE: {self.domain.objective} ({'higher' if self.domain.direction == 'max' else 'lower'}"
                f" is better)\n\nMEASUREMENTS (ref | {self.domain.name} entity | {self.domain.objective}):\n"
                + "\n".join(lines)
                + (f"\n\nFEATURES available for feature_effect:\n{features}" if features else "")
                + f"\n\nCONCLUSIONS OF THE AGENT:\n{conclusions.strip()}")
        return SYSTEM % self.max_lessons, user, refs

    def parse(self, answer: str, refs: dict[str, str]) -> list[dict]:
        """Lessons found in the LLM answer (malformed entries are dropped, unknown refs are kept as
        given: the verification will reject them)."""
        data = _first_json(answer)
        items = data.get("lessons", []) if isinstance(data, dict) else []
        out = []
        for it in items[: self.max_lessons]:
            if not isinstance(it, dict) or not str(it.get("statement", "")).strip():
                continue
            claim = it.get("claim") if isinstance(it.get("claim"), dict) else {"kind": "none"}
            cited = [str(e).strip() for e in (it.get("evidence") or []) if str(e).strip()]
            evidence = [refs.get(e, e) for e in cited]
            source = "cited"
            if not evidence:          # nothing cited: the lesson rests on what the agent was shown
                evidence, source = list(refs.values()), "context"
            out.append({"statement": str(it["statement"]).strip(),
                        "type": it.get("type") if it.get("type") in TYPES else "strategy",
                        "claim": claim, "evidence_ids": evidence, "evidence_source": source})
        return out


def _first_json(text: str):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    return None
