"""
Minimal synchronous LLM client used by the memory (judge, baselines) and by the benchmark agents.

Backends (same environment variables as docker/local_llm.py, so one configuration serves all):
  LLM_PROVIDER=ollama   (default) OLLAMA_URL, LLM_MODEL or OLLAMA_MODEL, OLLAMA_THINK, NUM_CTX
  LLM_PROVIDER=openai   LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
  LLM_PROVIDER=scripted a deterministic callable given in code (tests, dry runs)
Every call is counted (calls, prompt and completion characters, seconds) per "purpose", so the
cost of the memory itself can be separated from the cost of the agents.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import defaultdict
from typing import Callable

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMUsage:
    """Thread-safe usage counters, keyed by purpose (e.g. 'agent', 'judge', 'baseline')."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.by_purpose: dict[str, dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "prompt_chars": 0, "completion_chars": 0, "seconds": 0.0, "errors": 0})

    def add(self, purpose: str, prompt: str, completion: str, seconds: float, error: bool) -> None:
        with self._lock:
            u = self.by_purpose[purpose]
            u["calls"] += 1
            u["prompt_chars"] += len(prompt)
            u["completion_chars"] += len(completion)
            u["seconds"] += seconds
            u["errors"] += int(error)

    def snapshot(self) -> dict:
        with self._lock:
            return {k: dict(v) for k, v in self.by_purpose.items()}


class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 temperature: float | None = None, scripted: Callable[[str, str, str], str] | None = None,
                 usage: LLMUsage | None = None) -> None:
        self.provider = (provider or os.environ.get("LLM_PROVIDER", "ollama")).lower()
        self.model = model or os.environ.get("LLM_MODEL") or os.environ.get("OLLAMA_MODEL", "")
        self.temperature = float(os.environ.get("LLM_TEMPERATURE", "0.1")) if temperature is None else temperature
        self.scripted = scripted
        self.usage = usage or LLMUsage()
        if self.provider == "scripted" and scripted is None:
            raise ValueError("LLM_PROVIDER=scripted needs a scripted callable")

    def complete(self, system: str, user: str, purpose: str = "agent", max_tokens: int | None = None) -> str:
        t0, out, err = time.time(), "", False
        try:
            if self.provider == "scripted":
                out = self.scripted(system, user, purpose)
            elif self.provider == "openai":
                out = self._openai(system, user, max_tokens)
            else:
                out = self._ollama(system, user, max_tokens)
            out = _THINK_RE.sub("", out or "").strip()
            return out
        except Exception:
            err = True
            raise
        finally:
            self.usage.add(purpose, system + user, out, time.time() - t0, err)

    # ------------------------------------------------------------------ backends
    def _ollama(self, system: str, user: str, max_tokens: int | None) -> str:
        import requests
        url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        payload = {"model": self.model, "stream": False,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "options": {"temperature": self.temperature,
                               "num_predict": int(max_tokens or os.environ.get("NUM_PREDICT", "1024")),
                               "num_ctx": int(os.environ.get("NUM_CTX", "8192"))}}
        think = os.environ.get("OLLAMA_THINK", "false").lower()
        if think in ("true", "false"):
            payload["think"] = think == "true"
        r = requests.post(f"{url}/api/chat", json=payload, timeout=1800)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")

    def _openai(self, system: str, user: str, max_tokens: int | None) -> str:
        import requests
        base = (os.environ.get("LLM_BASE_URL") or "").rstrip("/")
        key = os.environ.get("LLM_API_KEY", "")
        if not base or not key:
            raise RuntimeError("LLM_PROVIDER=openai needs LLM_BASE_URL and LLM_API_KEY")
        payload: dict = {"model": self.model,
                         "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        mt = int(max_tokens or os.environ.get("NUM_PREDICT", "1024"))
        payload["max_completion_tokens" if "api.openai.com" in base else "max_tokens"] = mt
        if os.environ.get("LLM_SEND_TEMPERATURE", "true").lower() == "true":
            payload["temperature"] = self.temperature
        r = requests.post(f"{base}/chat/completions", json=payload,
                          headers={"Authorization": f"Bearer {key}"}, timeout=600)
        if r.status_code >= 400:
            raise RuntimeError(f"API error {r.status_code}: {r.text[:300]}")
        return r.json()["choices"][0]["message"].get("content") or ""


def extract_json(text: str):
    """Best-effort extraction of the first JSON object or array in an LLM answer."""
    if not text:
        return None
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except Exception:
            pass
        for op, cl in (("{", "}"), ("[", "]")):
            i, j = cand.find(op), cand.rfind(cl)
            if 0 <= i < j:
                try:
                    return json.loads(cand[i:j + 1])
                except Exception:
                    continue
    return None


def extract_code(text: str) -> str:
    """Python code from a fenced block. An opening fence without a closing one (answer cut by the
    token limit) gives the code after the fence; is_truncated() tells the two cases apart."""
    text = text or ""
    blocks = re.findall(r"```[ \t]*([A-Za-z0-9]*)[ \t]*\n(.*?)```", text, re.DOTALL)
    code = [b for lang, b in blocks if "def " in b] or [b for lang, b in blocks if lang.lower().startswith("py")]
    if code:
        return code[0].strip()
    tail = text.rsplit("```", 1)
    if len(tail) == 2 and is_truncated(text):     # last fence opened, never closed
        return re.sub(r"^[ \t]*[A-Za-z0-9]*[ \t]*\n", "", tail[1], count=1).strip()
    return text.strip() if "```" not in text else ""


def is_truncated(text: str) -> bool:
    """True when the answer opens a code fence that it never closes."""
    return (text or "").count("```") % 2 == 1


def long_output_tokens() -> int:
    """Token limit for answers that contain code (judge, oracle): JUDGE_MAX_TOKENS, or half of the
    context window. NUM_PREDICT (1024 by default) is meant for the agents' short answers and cuts
    tests in the middle."""
    return int(os.environ.get("JUDGE_MAX_TOKENS") or int(os.environ.get("NUM_CTX", "8192")) // 2)
