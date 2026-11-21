"""
LLM access for the analysis agent: the same client as GAIED's agents (docker/local_llm.py: Ollama or an OpenAI-format
API), or a scripted model (LLM_PROVIDER=scripted) for fast tests without any model.

The analysis agent uses the LLM only where interpretation is needed; what it returns is stored as proposed
(epistemic status hypothesis or explanation, edges of origin "llm"), never as a checked fact.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re

EXPLAIN_FAILURES = """You analyze failed simulations of a molecular design workflow.
Given molecules whose computation failed with the same error class (and the motif they share, if any), propose a
candidate explanation of why they fail and one short piece of advice for the agents. You only propose; nothing you say
is taken as a confirmed cause.
Answer with ONLY a JSON object: {"explanation": "...", "advice": "..."}"""

SUGGEST_DIRECTION = """You advise a computational chemistry agent whose recent iterations did not improve its best
result. Using the team's results, failures and knowledge given below, suggest one concrete change of direction.
Answer with ONLY a JSON object: {"advice": "..."}"""

DRAFT_PROCEDURE = """From several failed computations of a molecular design workflow that show the same pattern,
write one reusable procedure for future agents. Be specific; state its limits. The team's knowledge related to the
pattern is listed with ids: give the ids of the statements that justify the procedure, if any.
Answer with ONLY a JSON object: {"text": "...", "trigger": "...", "preconditions": "...", "actions": "...",
"expected_outcome": "...", "limitations": "...", "justified_by": ["<statement id>", ...]}"""

EXTRACT_CLAIM = """You turn one scientific statement about ionization energies into a structured claim that can be
checked against measured values. Use exactly one of these forms, or "none" if the statement cannot be checked this way:
{"form": "value", "molecule": "<SMILES>", "value": <eV>}                     the molecule's value is about this
{"form": "bound", "molecule": "<SMILES>", "relation": "above|below", "value": <eV>}
{"form": "comparison", "a": "<SMILES>", "relation": "higher|lower", "b": "<SMILES>"}   a's value is higher/lower than b's
{"form": "motif_effect", "feature": "<one of FEATURE_NAMES>", "direction": "increases|decreases"}
                                    molecules with this feature have higher/lower values than those without it
{"form": "none"}
Use SMILES exactly as written in the statement or in the list of measured molecules. Answer with the JSON only."""

_JSON = re.compile(r"\{.*\}", re.S)


def parse_json(text: str) -> dict:
    m = _JSON.search(text or "")
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
        return d if isinstance(d, dict) else {}
    except ValueError:
        return {}


class AnalysisLLM:
    def __init__(self, provider: str | None = None) -> None:
        self.provider = (provider or os.environ.get("LLM_PROVIDER", "ollama")).lower()
        self.calls = 0
        self._client = None

    def _real(self):
        if self._client is None:
            from local_llm import OllamaLLM          # docker/local_llm.py, on PYTHONPATH in the container
            self._client = OllamaLLM(model=os.environ.get("LLM_MODEL") or os.environ.get("OLLAMA_MODEL") or "")
        return self._client

    def ask(self, system: str, user: str) -> str:
        self.calls += 1
        if self.provider == "scripted":
            return scripted(system, user)
        from langchain.messages import HumanMessage, SystemMessage
        resp = asyncio.run(self._real().ainvoke([SystemMessage(content=system), HumanMessage(content=user)]))
        return resp.content

    def ask_json(self, system: str, user: str) -> dict:
        text = self.ask(system, user)
        d = parse_json(text)
        if not d:   # make an unusable answer visible (e.g. cut by NUM_PREDICT, or not JSON)
            logging.getLogger("pgmem.llm").warning("LLM answer is not a JSON object (%d chars): %r ... %r",
                                                   len(text or ""), (text or "")[:300], (text or "")[-200:])
        return d


def scripted(system: str, user: str) -> str:
    """Deterministic answers for tests: shape-correct, content derived from the input."""
    if system == EXPLAIN_FAILURES:
        return json.dumps({"explanation": "The failing molecules share the error class reported by the workflow.",
                           "advice": "Avoid proposing molecules similar to the failing ones."})
    if system.startswith("You turn one scientific statement"):
        m = re.search(r"ionization energy of (\S+) is (above|below) (-?[0-9.]+)", user)
        if m:
            return json.dumps({"form": "bound", "molecule": m.group(1), "relation": m.group(2),
                               "value": float(m.group(3))})
        return json.dumps({"form": "none"})
    if system == SUGGEST_DIRECTION:
        return json.dumps({"advice": "Explore a different scaffold than the last iterations (scripted advice)."})
    if system == DRAFT_PROCEDURE:
        return json.dumps({"text": "Avoid molecules that repeatedly fail with this error class.",
                           "trigger": "a proposed molecule resembles molecules that failed with this error class",
                           "preconditions": "the error class has been observed at least twice",
                           "actions": "replace the molecule by one without the shared feature",
                           "expected_outcome": "no new failure of this class",
                           "limitations": "based only on the failures observed in this memory",
                           "justified_by": re.findall(r"\[(m3:[0-9a-f]+)\]", user)[:1]})
    return "{}"
