"""
local_llm.py — Replaces GAIED's anonymized LLM client (SomeAPILLM) with:
  - a LOCAL model served by Ollama              (LLM_PROVIDER=ollama, default)
  - or an ONLINE API in the OpenAI format        (LLM_PROVIDER=openai)
    -> OpenAI, Google Gemini, Groq, OpenRouter, DeepSeek, Anthropic... (with an API key)

Same interface as SomeAPILLM:
  - attributes : model, temperature, _tools   (read by the Flowcept provenance patch)
  - methods    : bind_tools(tools), ainvoke(messages)
  - response   : object with .content and .tool_calls

As in the original, tool calls go through a JSON object requested in the prompt
({"tool_calls": [{"name": ..., "args": {"smiles": ...}}]}), which works with any model.

Environment variables:
  LLM_PROVIDER   "ollama" (default), "openai", or "scripted" (deterministic stand-in used
                 to dry-run the whole pipeline in minutes, see experiments/synthetic.py)
  LLM_MODEL      model name (fallback: OLLAMA_MODEL); replaces the "gpt54" requested by GAIED
  --- Ollama ---
  OLLAMA_URL     (default http://localhost:11434)
  OLLAMA_THINK   "false" (default) / "true" / any other value = send nothing
  NUM_CTX        context size (default 8192)
  --- Online API (OpenAI format) ---
  LLM_BASE_URL   e.g. https://api.openai.com/v1
  LLM_API_KEY    the key (put it in docker/.env, never in Git)
  LLM_SEND_TEMPERATURE  "true" (default); "false" for models that enforce it (e.g. some GPT-5)
  --- Common ---
  NUM_PREDICT    maximum number of generated tokens (default 1024)
"""
from __future__ import annotations

import asyncio
import json
import os
import re

import requests
from langchain.messages import HumanMessage
from langchain.messages import SystemMessage

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class _LLMResponse:
    def __init__(self, content: str, tool_calls: list | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class OllamaLLM:
    def __init__(self, model: str, temperature: float = 0.1) -> None:
        self.provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
        self.model = os.environ.get("LLM_MODEL") or os.environ.get("OLLAMA_MODEL") or model
        self.temperature = temperature
        self._tools: list | None = None

    def bind_tools(self, tools: list) -> "OllamaLLM":
        bound = OllamaLLM(self.model, self.temperature)
        bound._tools = tools
        return bound

    # ------------------------------------------------------------------ helpers
    def _to_chat_messages(self, messages: list) -> list[dict]:
        out = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                out.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                out.append({"role": "user", "content": msg.content})
            else:  # e.g. a previous response sent back on a retry
                out.append({"role": "assistant", "content": str(getattr(msg, "content", msg))})
        if self._tools:
            tool_names = ", ".join(t.name for t in self._tools)
            instruction = (
                f"\n\nAvailable tools: {tool_names}.\n"
                "Respond with ONLY a JSON object: "
                '{"tool_calls": [{"name": "<tool>", "args": {"smiles": "<SMILES>"}}]}'
            )
            for m in reversed(out):            # instruction appended to the last user message
                if m["role"] == "user":
                    m["content"] += instruction
                    break
        return out

    def _parse_tool_calls(self, content: str) -> list[dict]:
        try:
            parsed = json.loads(content[content.find("{"): content.rfind("}") + 1])
        except (json.JSONDecodeError, ValueError):
            return []
        calls = []
        default_name = self._tools[0].name if self._tools else "compute_ionization_energy"
        for c in parsed.get("tool_calls", []) if isinstance(parsed, dict) else []:
            args = c.get("args") if isinstance(c, dict) else None
            if isinstance(args, dict) and args.get("smiles"):
                calls.append({"name": c.get("name") or default_name, "args": {"smiles": args["smiles"]}})
        return calls

    # ------------------------------------------------------------------ backends
    def _post_ollama(self, messages: list[dict]) -> str:
        url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": int(os.environ.get("NUM_PREDICT", "1024")),
                "num_ctx": int(os.environ.get("NUM_CTX", "8192")),
            },
        }
        think = os.environ.get("OLLAMA_THINK", "false").lower()
        if think in ("true", "false"):
            payload["think"] = think == "true"
        resp = requests.post(f"{url}/api/chat", json=payload, timeout=1800)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")

    def _post_openai(self, messages: list[dict]) -> str:
        base = (os.environ.get("LLM_BASE_URL") or "").strip().rstrip("/")
        key = os.environ.get("LLM_API_KEY", "").strip()
        if not key:
            raise RuntimeError("LLM_PROVIDER=openai but LLM_API_KEY is empty (see docker/.env).")
        if not base.startswith("http"):
            raise RuntimeError(
                "LLM_PROVIDER=openai but LLM_BASE_URL is empty or invalid: in docker/.env, "
                "remove the # in front of your provider's LLM_BASE_URL and LLM_MODEL lines.")
        if not os.environ.get("LLM_MODEL"):
            raise RuntimeError(
                "LLM_PROVIDER=openai but LLM_MODEL is empty: in docker/.env, "
                "remove the # in front of your provider's LLM_MODEL line.")
        max_tokens = int(os.environ.get("NUM_PREDICT", "1024"))
        payload: dict = {"model": self.model, "messages": messages}
        # api.openai.com expects max_completion_tokens; other providers expect max_tokens
        payload["max_completion_tokens" if "api.openai.com" in base else "max_tokens"] = max_tokens
        if os.environ.get("LLM_SEND_TEMPERATURE", "true").lower() == "true":
            payload["temperature"] = self.temperature
        resp = requests.post(
            f"{base}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {key}"},
            timeout=600,
        )
        if resp.status_code >= 400:   # provider error message, without printing the key
            raise RuntimeError(f"API error {resp.status_code}: {resp.text[:500]}")
        return resp.json()["choices"][0]["message"].get("content") or ""

    # ------------------------------------------------------------------ API
    async def ainvoke(self, messages: list) -> _LLMResponse:
        if self.provider == "scripted":   # dry run of the full pipeline without any LLM
            from experiments.synthetic import ScriptedLLM
            llm = ScriptedLLM(self.model, self.temperature)
            llm._tools = self._tools
            return await llm.ainvoke(messages)
        chat = self._to_chat_messages(messages)
        post = self._post_openai if self.provider == "openai" else self._post_ollama
        content = await asyncio.to_thread(post, chat)
        content = _THINK_RE.sub("", content).strip()
        if self._tools:
            return _LLMResponse(content=content, tool_calls=self._parse_tool_calls(content))
        return _LLMResponse(content=content)
