"""
local_llm.py — Remplace le client LLM anonymisé de GAIED (SomeAPILLM) par :
  - un modèle LOCAL servi par Ollama           (LLM_PROVIDER=ollama, par défaut)
  - ou une API EN LIGNE au format OpenAI        (LLM_PROVIDER=openai)
    → OpenAI, Google Gemini, Groq, OpenRouter, DeepSeek, Anthropic… (avec une clé API)

Même interface que SomeAPILLM :
  - attributs  : model, temperature, _tools   (lus par le patch de provenance Flowcept)
  - méthodes   : bind_tools(tools), ainvoke(messages)
  - réponse    : objet avec .content et .tool_calls

Comme l'original, les appels d'outils passent par un JSON demandé dans le prompt
({"tool_calls": [{"name": ..., "args": {"smiles": ...}}]}) : marche avec tout modèle.

Variables d'environnement :
  LLM_PROVIDER   "ollama" (défaut) ou "openai"
  LLM_MODEL      nom du modèle (à défaut : OLLAMA_MODEL) — remplace le "gpt54" demandé par GAIED
  --- Ollama ---
  OLLAMA_URL     (défaut http://localhost:11434)
  OLLAMA_THINK   "false" (défaut) / "true" / autre valeur = ne rien envoyer
  NUM_CTX        taille du contexte (défaut 8192)
  --- API en ligne (format OpenAI) ---
  LLM_BASE_URL   ex. https://api.openai.com/v1
  LLM_API_KEY    la clé (à mettre dans docker/.env, jamais dans Git)
  LLM_SEND_TEMPERATURE  "true" (défaut) ; "false" pour les modèles qui l'imposent (ex. certains GPT-5)
  --- Commun ---
  NUM_PREDICT    nombre maximal de tokens générés (défaut 1024)
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
            else:  # ex. réponse précédente renvoyée lors d'un nouvel essai
                out.append({"role": "assistant", "content": str(getattr(msg, "content", msg))})
        if self._tools:
            tool_names = ", ".join(t.name for t in self._tools)
            instruction = (
                f"\n\nAvailable tools: {tool_names}.\n"
                "Respond with ONLY a JSON object: "
                '{"tool_calls": [{"name": "<tool>", "args": {"smiles": "<SMILES>"}}]}'
            )
            for m in reversed(out):            # consigne ajoutée au dernier message utilisateur
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
            raise RuntimeError("LLM_PROVIDER=openai mais LLM_API_KEY est vide (voir docker/.env).")
        if not base.startswith("http"):
            raise RuntimeError(
                "LLM_PROVIDER=openai mais LLM_BASE_URL est vide ou invalide : dans docker/.env, "
                "retirez le # devant les lignes LLM_BASE_URL et LLM_MODEL de votre fournisseur.")
        if not os.environ.get("LLM_MODEL"):
            raise RuntimeError(
                "LLM_PROVIDER=openai mais LLM_MODEL est vide : dans docker/.env, "
                "retirez le # devant la ligne LLM_MODEL de votre fournisseur.")
        max_tokens = int(os.environ.get("NUM_PREDICT", "1024"))
        payload: dict = {"model": self.model, "messages": messages}
        # api.openai.com attend max_completion_tokens ; les autres fournisseurs, max_tokens
        payload["max_completion_tokens" if "api.openai.com" in base else "max_tokens"] = max_tokens
        if os.environ.get("LLM_SEND_TEMPERATURE", "true").lower() == "true":
            payload["temperature"] = self.temperature
        resp = requests.post(
            f"{base}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {key}"},
            timeout=600,
        )
        if resp.status_code >= 400:   # message d'erreur du fournisseur, sans afficher la clé
            raise RuntimeError(f"Erreur API {resp.status_code} : {resp.text[:500]}")
        return resp.json()["choices"][0]["message"].get("content") or ""

    # ------------------------------------------------------------------ API
    async def ainvoke(self, messages: list) -> _LLMResponse:
        chat = self._to_chat_messages(messages)
        post = self._post_openai if self.provider == "openai" else self._post_ollama
        content = await asyncio.to_thread(post, chat)
        content = _THINK_RE.sub("", content).strip()
        if self._tools:
            return _LLMResponse(content=content, tool_calls=self._parse_tool_calls(content))
        return _LLMResponse(content=content)
