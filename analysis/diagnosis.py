"""
Optional LLM reasoning for the analysis service (ANALYSIS_LLM=true): one extra candidate explanation per
new finding, stored in M2 as an unverified hypothesis. Detection and confirmation of causes never depend
on it. Uses the same client as the agents (docker/local_llm.py: Ollama or an OpenAI-format API).
"""
from __future__ import annotations

import asyncio
import logging
import os

log = logging.getLogger("analysis.diagnosis")

SYSTEM = ("You help diagnose a team of scientific agents searching molecules with simulations. Given observed "
          "facts and existing hypotheses, propose ONE additional plausible explanation in one sentence. "
          "Do not state it as a fact.")


class LLMExplainer:
    def __init__(self) -> None:
        from langchain.messages import HumanMessage, SystemMessage
        from local_llm import OllamaLLM
        self._msgs = (SystemMessage, HumanMessage)
        self.llm = OllamaLLM(model=os.environ.get("ANALYSIS_LLM_MODEL", "analysis"))
        self.calls = 0

    def explain(self, finding) -> str | None:
        System, Human = self._msgs
        user = (f"Problem: {finding.summary}\nFacts:\n- " + "\n- ".join(finding.facts)
                + "\nExisting hypotheses:\n- " + "\n- ".join(finding.explanations or ["none"]))
        try:
            self.calls += 1
            resp = asyncio.run(self.llm.ainvoke([System(content=SYSTEM), Human(content=user)]))
            return " ".join(resp.content.split())[:300] or None
        except Exception as exc:
            log.warning("LLM explanation failed: %s", exc)
            return None


def explainer_from_env():
    if os.environ.get("ANALYSIS_LLM", "false").lower() not in ("1", "true", "yes"):
        return None
    try:
        return LLMExplainer()
    except Exception as exc:
        log.warning("ANALYSIS_LLM requested but unavailable: %s", exc)
        return None
