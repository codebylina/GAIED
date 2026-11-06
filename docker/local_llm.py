"""
local_llm.py — Remplace le client LLM anonymisé de GAIED (SomeAPILLM) par un modèle
open source servi localement par Ollama.

Même interface que SomeAPILLM :
  - attributs  : model, temperature, _tools   (lus par le patch de provenance Flowcept)
  - méthodes   : bind_tools(tools), ainvoke(messages)
  - réponse    : objet avec .content et .tool_calls

Comme l'original, les appels d'outils passent par un JSON demandé dans le prompt
({"tool_calls": [{"name": ..., "args": {"smiles": ...}}]}). Cela marche avec n'importe
quel modèle, même sans support natif des « tools ».

Variables d'environnement :
  OLLAMA_URL     (défaut http://localhost:11434)
  OLLAMA_MODEL   (ex. qwen3.5:4b) — remplace le nom de modèle demandé par GAIED (gpt54)
  OLLAMA_THINK   "false" (défaut) ou "true" : mode raisonnement de Qwen (très lent sur CPU)
  NUM_PREDICT    nombre maximal de tokens générés (défaut 1024)
  NUM_CTX        taille du contexte (défaut 8192)
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
        self.model = os.environ.get("OLLAMA_MODEL", model)
        self.temperature = temperature
        self._tools: list | None = None
        self.base_url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")

    def bind_tools(self, tools: list) -> "OllamaLLM":
        bound = OllamaLLM(self.model, self.temperature)
        bound._tools = tools
        return bound

    # ------------------------------------------------------------------ helpers
    def _to_ollama_messages(self, messages: list) -> list[dict]:
        out = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                out.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                out.append({"role": "user", "content": msg.content})
            else:  # ex. réponse précédente renvoyée lors d'un nouvel essai
                out.append({"role": "assistant", "content": str(getattr(msg, "content", msg))})
        if self._tools:
            tool_names = ", ".join(t.name for t in self._tools)
            instruction = (
                f"\n\nAvailable tools: {tool_names}.\n"
                "Respond with ONLY a JSON object: "
                '{"tool_calls": [{"name": "<tool>", "args": {"smiles": "<SMILES>"}}]}'
            )
            for m in reversed(out):            # ajouter la consigne au dernier message utilisateur
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

    # ------------------------------------------------------------------ API
    async def ainvoke(self, messages: list) -> _LLMResponse:
        payload = {
            "model": self.model,
            "messages": self._to_ollama_messages(messages),
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

        def _post() -> dict:
            resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=1800)
            resp.raise_for_status()
            return resp.json()

        data = await asyncio.to_thread(_post)
        content = _THINK_RE.sub("", data.get("message", {}).get("content", "")).strip()

        if self._tools:
            return _LLMResponse(content=content, tool_calls=self._parse_tool_calls(content))
        return _LLMResponse(content=content)
