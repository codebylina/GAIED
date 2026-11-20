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
  LLM_PROVIDER   "ollama" (default) or "openai"
  LLM_MODEL      model name (fallback: OLLAMA_MODEL); replaces the "gpt54" requested by GAIED
  --- Ollama ---
  OLLAMA_URL     (default http://localhost:11434)
  OLLAMA_THINK   "false" (default) / "true" / any other value = send nothing
  NUM_CTX        context size (default 8192)
  --- Online API (OpenAI format) ---
  LLM_BASE_URL   e.g. https://api.openai.com/v1
  LLM_API_KEY    the key (put it in docker/.env, never in Git)
  LLM_SEND_TEMPERATURE  "true" (default); "false" for models that enforce it (e.g. some GPT-5)
  --- Tests ---
  LLM_PROVIDER=scripted  deterministic answers, no model: checks the whole pipeline in minutes
                         (SCRIPTED_DELAY_S: seconds to wait per answer, to imitate a real model)
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
            if any(t.name == "query_memory" for t in self._tools):   # [pgmem] the memory tool takes a question
                instruction += (
                    '\nTo ask the shared memory first, call {"name": "query_memory", "args": {"question": "<text>", '
                    '"smiles": "<SMILES or empty>"}}; you will get its answer before deciding.')
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
        names = {t.name for t in self._tools or []}
        for c in parsed.get("tool_calls", []) if isinstance(parsed, dict) else []:
            args = c.get("args") if isinstance(c, dict) else None
            if not isinstance(args, dict):
                continue
            if c.get("name") == "query_memory" and "query_memory" in names and args.get("question"):
                calls.append({"name": "query_memory", "args": {"question": str(args["question"]),
                                                               "smiles": str(args.get("smiles") or "")}})
            elif args.get("smiles"):
                name = c.get("name") if c.get("name") in names else default_name
                calls.append({"name": name, "args": {"smiles": args["smiles"]}})
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
        if os.environ.get("LLM_SEED"):          # paired runs: the same sampling seed in every condition
            payload["options"]["seed"] = int(os.environ["LLM_SEED"])
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
        chat = self._to_chat_messages(messages)
        if self.provider == "scripted":
            # SCRIPTED_DELAY_S: seconds per answer, to imitate a real model's latency in quick tests
            await asyncio.sleep(float(os.environ.get("SCRIPTED_DELAY_S", "0")))
            content = scripted_answer(chat, bool(self._tools),
                                      any(t.name == "query_memory" for t in self._tools or []))
            if self._tools:
                return _LLMResponse(content=content, tool_calls=self._parse_tool_calls(content))
            return _LLMResponse(content=content)
        post = self._post_openai if self.provider == "openai" else self._post_ollama
        content = await asyncio.to_thread(post, chat)
        content = _THINK_RE.sub("", content).strip()
        if self._tools:
            return _LLMResponse(content=content, tool_calls=self._parse_tool_calls(content))
        return _LLMResponse(content=content)


# ---------------------------------------------------------------------------------------------- scripted model
# Molecules proposed by the scripted model, in order; "C1CC" and "C1CCN" are invalid SMILES (technical failures).
SCRIPTED_POOL = ["NC(N)=O", "NC(=O)C(=O)N", "C1CC", "O=C1NC(=O)NC1=O", "C1CCN", "CC(N)=O", "NC#N",
                 "O=C1NCC(=O)N1", "OC(=O)C(=O)O", "N#CC#N", "CC(=O)NC(N)=O", "NC(=O)NC(N)=O"]


def scripted_answer(chat: list[dict], tools: bool, memory_tool: bool = False) -> str:
    system = next((m["content"] for m in chat if m["role"] == "system"), "")
    user = "\n".join(m["content"] for m in chat if m["role"] == "user")
    if tools:
        per_call = int(os.environ.get("SCRIPTED_PROPOSALS", "3"))
        picks = [s for s in SCRIPTED_POOL if f" {s} " not in f" {user.replace(',', ' ')} "][:per_call]
        if memory_tool and "Answers from the shared memory" not in user and picks:   # ask once, then decide
            return json.dumps({"tool_calls": [{"name": "query_memory", "args": {
                "question": "Have similar molecules failed or been measured?", "smiles": picks[0]}}]})
        return json.dumps({"tool_calls": [{"name": "compute_ionization_energy", "args": {"smiles": s}}
                                          for s in picks]})
    if "Synthesize any conclusions" in system:
        rows = re.findall(r"^(\S+)\t\t(-?[0-9.]+)\s*$", user, re.M)
        text = "Conclusion: the measured molecules differ in ionization energy."
        if "KNOWLEDGE" in system and rows:
            items = [{"statement": f"The ionization energy of {s} is above {float(v) - 1:.2f} eV",
                      "kind": "finding", "epistemic_status": "observation", "scope": "GFN2-xTB",
                      "supporting_molecules": [s], "contradicting_molecules": [], "uncertainty": "single run"}
                     for s, v in rows[:2]]
            text += "\n\nKNOWLEDGE\n" + json.dumps(items)
        return text
    if "Critique" in system:
        return "Critique: the campaign explores few chemical families."
    if "Update the given plan" in system:
        return "Updated plan: simulate further small polar molecules."
    return "Plan: start from the seed and simulate small polar molecules with several nitrogen atoms."
