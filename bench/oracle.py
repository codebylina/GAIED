"""
Evaluation of statements against the ground truth of the synthetic domain.

The same evaluator is applied to the knowledge of every condition (ProvMem claims, baseline memories),
so it is not a confound. An LLM writes `def holds(df) -> bool` that decides the statement on the full
population (every design with its exact, noise-free outcome); the function is executed in the sandbox.
Statements that are not about outcomes are marked untestable. The evaluator's accuracy is checked on
statements whose truth is known by construction (bench.env.SyntheticEnv.planted_statements).
Results are cached per (environment, statement).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading

from memlab.llm import LLM, extract_code, extract_json, long_output_tokens
from memlab.sandbox import TestError, run_function

ORACLE_SYSTEM = """You check statements about a design space against its complete ground truth.
You get a statement and the columns of a table that lists EVERY design with its exact outcome
(noise-free; value is null for designs that fail). Write a Python function `def holds(df):` that
returns True if the statement is true on this complete table and False otherwise. Interpret the
statement as a scientist would: a statement restricted to a region must be checked in that region only;
"X increases the outcome" means designs with X have a higher mean outcome than comparable designs without X.
If the statement is not a claim about outcomes (a plan, an advice, a remark), answer {"testable": false}.
Otherwise answer {"testable": true} followed by the code in a ```python block. pandas is available as pd.
Keep the code short: no comments, no docstrings, no prints."""


class Oracle:
    def __init__(self, llm: LLM, cache_path: str, timeout_s: float = 120.0) -> None:
        self.llm, self.cache_path, self.timeout_s = llm, cache_path, timeout_s
        self.lock = threading.Lock()
        self.cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    def _key(self, env_id: str, text: str) -> str:
        return hashlib.sha1(f"{env_id}|{text.strip()}".encode()).hexdigest()

    def truth(self, env, text: str) -> bool | None:
        env_id = json.dumps(env.spec.__dict__, sort_keys=True, default=str)
        k = self._key(env_id, text)
        hit = self.cache.get(k)
        if hit and (hit["truth"] is not None or hit.get("reason") == "untestable"):
            return hit["truth"]          # failed evaluations are retried
        cols = ", ".join(["id", "entity", "value", "status"] + [f"slot_{s}" for s in env.slots])
        answer = self.llm.complete(ORACLE_SYSTEM, f"COLUMNS: {cols}\nSTATEMENT: {text}", purpose="oracle",
                                   max_tokens=long_output_tokens())
        meta = extract_json(answer.split("def holds")[0]) or {}
        truth, code, reason = None, "", "untestable"
        if meta.get("testable", True) and "```" in answer:
            code = extract_code(answer)
            try:
                res = run_function(code, env.population(), "holds", self.timeout_s)
                truth = bool(res) if res is not None else None
                reason = "ok" if truth is not None else "error"
            except TestError as exc:
                reason = f"error: {str(exc)[:300]}"
        with self.lock:
            self.cache[k] = {"text": text, "truth": truth, "code": code, "reason": reason}
        return truth

    def save(self) -> None:
        with self.lock:
            tmp = self.cache_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(self.cache, fh)
            os.replace(tmp, self.cache_path)
