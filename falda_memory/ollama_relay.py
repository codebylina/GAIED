"""
OpenAI-compatible relay in front of Ollama, for FALDA's distiller only (FALDA itself unchanged).

Why: the agents call Ollama's native /api/chat with think=false (docker/local_llm.py). FALDA's
distiller calls an OpenAI endpoint (/v1/chat/completions) with no way to turn thinking off, so
Qwen 3.5 "thinks" for minutes and the distiller's fixed 180 s timeout expires: 0 atoms.
This relay receives the distiller's OpenAI request and forwards it to /api/chat with the SAME
options as the agents (think, num_ctx), keeping the distiller's own max_tokens and temperature.

Ollama serves the agents and the distiller with one model, so a distiller request can wait in
Ollama's queue. The distiller's timeout is a socket read timeout, so while waiting the relay
sends whitespace (valid before JSON) every HEARTBEAT_S seconds to keep the connection alive.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HEARTBEAT_S = 20


def _ollama_chat(body: dict) -> dict:
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    payload = {
        "model": body.get("model") or os.environ.get("OLLAMA_MODEL", ""),
        "messages": body.get("messages", []),
        "stream": False,
        "options": {
            "temperature": body.get("temperature", 0.2),
            "num_predict": int(body.get("max_tokens") or 2000),
            "num_ctx": int(os.environ.get("NUM_CTX", "8192")),
        },
    }
    think = os.environ.get("OLLAMA_THINK", "false").lower()
    if think in ("true", "false"):
        payload["think"] = think == "true"
    req = urllib.request.Request(f"{url}/api/chat", data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read().decode())
    content = (out.get("message") or {}).get("content", "")
    return {"object": "chat.completion", "model": payload["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}]}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        result, t0 = {}, time.time()

        def work():
            try:
                result["ok"] = _ollama_chat(body)
            except Exception as exc:
                result["err"] = str(exc)

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        try:
            while worker.is_alive():
                worker.join(HEARTBEAT_S)
                if worker.is_alive():
                    self.wfile.write(b" ")
                    self.wfile.flush()
            if "ok" in result:
                self.wfile.write(json.dumps(result["ok"]).encode())
            else:  # not JSON on purpose: the distiller logs it as an LLM error
                self.wfile.write(f"relay error: {result.get('err')}".encode())
        except (BrokenPipeError, ConnectionResetError):
            pass
        preview = ""
        if "ok" in result:
            preview = result["ok"]["choices"][0]["message"]["content"][:160].replace("\n", " ")
        print(f"[falda-relay] distiller LLM call: {time.time() - t0:.1f} s, "
              f"{'ok: ' + repr(preview) if 'ok' in result else 'error: ' + str(result.get('err'))}",
              flush=True)


def start(port: int) -> str:
    """Start the relay in a background thread; return the base URL for LLM_BASE_URL."""
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}/v1"
