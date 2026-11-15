"""
Mem0 (open source, mem0ai) used as the shared memory of the team, as designed:
  - write : what an agent observes and concludes is added as a message (infer=True: Mem0's LLM
            extracts the memories), scoped by agent_id and by run_id = the campaign;
  - read  : before an LLM step, the agent's query is searched in the campaign's memories
            (all agents), with Mem0's default number of results; the memories are put in the prompt.
Nothing is verified, no experiment is coordinated: that is Mem0's design, and the point of this baseline.

Deployment: Mem0's local vector store accepts one process. GAIED's agents run in several processes,
so one Mem0 instance runs in the campaign's main process behind a small HTTP service (Mem0Service),
and agents use Mem0Client. The benchmark (one process) uses LocalMem0 directly.

LLM settings: Mem0's Ollama driver does not expose Qwen's thinking switch; it is set to the agents'
value (OLLAMA_THINK, false by default), so Mem0 and the agents call the same model the same way.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .base import Memory

logger = logging.getLogger(__name__)
os.environ.setdefault("MEM0_TELEMETRY", "False")


# --------------------------------------------------------------------------------------- backends
def build_mem0(storage_dir: str, collection: str, scripted=None):
    """A local Mem0 instance: Qdrant embedded in storage_dir, Ollama (or an OpenAI-style API) for the LLM
    and the embeddings. scripted=(llm_fn, embed_fn) replaces both for tests and dry runs."""
    from mem0 import Memory as Mem0
    os.makedirs(storage_dir, exist_ok=True)
    dims = int(os.environ.get("EMBED_DIMS", "768"))
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    model = os.environ.get("LLM_MODEL") or os.environ.get("OLLAMA_MODEL", "")
    if scripted is not None:
        llm = {"provider": "openai", "config": {"model": "scripted", "api_key": "x"}}
        emb = {"provider": "openai", "config": {"model": "scripted", "api_key": "x", "embedding_dims": dims}}
    elif provider == "openai":
        llm = {"provider": "openai", "config": {"model": model, "api_key": os.environ["LLM_API_KEY"],
                                                 "openai_base_url": os.environ["LLM_BASE_URL"], "temperature": 0.1}}
        emb = {"provider": "ollama", "config": {"model": os.environ.get("EMBED_MODEL", "nomic-embed-text"),
                                                "ollama_base_url": os.environ.get("OLLAMA_URL", "http://localhost:11434"),
                                                "embedding_dims": dims}}
    else:
        url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        llm = {"provider": "ollama", "config": {"model": model, "ollama_base_url": url, "temperature": 0.1,
                                                 "max_tokens": int(os.environ.get("NUM_PREDICT", "1024"))}}
        emb = {"provider": "ollama", "config": {"model": os.environ.get("EMBED_MODEL", "nomic-embed-text"),
                                                "ollama_base_url": url, "embedding_dims": dims}}
    config = {"llm": llm, "embedder": emb,
              "vector_store": {"provider": "qdrant", "config": {"collection_name": collection, "path": storage_dir,
                                                                "embedding_model_dims": dims, "on_disk": True}},
              "history_db_path": os.path.join(storage_dir, "history.db")}
    m = Mem0.from_config(config)
    if scripted is not None:
        llm_fn, embed_fn = scripted
        m.llm.generate_response = lambda messages, **kw: llm_fn(messages)
        m.embedding_model.embed = lambda text, memory_action=None: embed_fn(text)
        m.embedding_model.embed_batch = lambda texts, memory_action=None: [embed_fn(t) for t in texts]
    elif provider != "openai":
        _same_thinking_as_agents(m)
    return m


def _same_thinking_as_agents(m) -> None:
    """Qwen's thinking set as for the agents, and a context window large enough for Mem0's prompts:
    its extraction prompt alone is about 36,000 characters (about 9,000 tokens), more than the agents'
    8192-token window, and Ollama would silently cut it. MEM0_NUM_CTX (default 16384) sets it."""
    think = os.environ.get("OLLAMA_THINK", "false").lower()
    num_ctx = int(os.environ.get("MEM0_NUM_CTX", "16384"))
    chat = m.llm.client.chat

    def patched(*a, **kw):
        if think in ("true", "false"):
            kw["think"] = think == "true"
        kw["options"] = {**(kw.get("options") or {}), "num_ctx": num_ctx}
        return chat(*a, **kw)
    m.llm.client.chat = patched


class LocalMem0:
    """Thread-safe access to one Mem0 instance. Writes go through a queue processed by one worker
    (Mem0's add calls the LLM); the lock protects the vector store and is released during LLM calls,
    so that searches are not blocked while Mem0 extracts memories."""

    def __init__(self, mem0, run_id: str, usage=None) -> None:
        self.m, self.run_id, self.usage = mem0, run_id, usage
        self.lock = threading.Lock()
        self.q: queue.Queue = queue.Queue()
        self.ops: list[dict] = []
        self.on_added = None     # callback(events, agent_id): a layer on top of Mem0 sees each new memory
        gen = self.m.llm.generate_response

        def unlocked_llm(*a, **kw):
            self.lock.release()
            t0, out, err = time.time(), "", False
            try:
                out = gen(*a, **kw)
                return out
            except Exception:
                err = True
                raise
            finally:
                if self.usage is not None:
                    self.usage.add("memory", json.dumps(kw.get("messages") or (a[0] if a else ""), default=str),
                                   str(out or ""), time.time() - t0, err)
                self.lock.acquire()
        self.m.llm.generate_response = unlocked_llm

    def add(self, text: str, agent_id: str, metadata: dict | None = None) -> None:
        self.q.put((text, agent_id, metadata or {}, time.time()))

    def pending(self) -> int:
        return self.q.qsize()

    def process_one(self, timeout: float = 0.0) -> bool:
        try:
            text, agent, meta, t_req = self.q.get(timeout=timeout) if timeout else self.q.get_nowait()
        except queue.Empty:
            return False
        t0 = time.time()
        try:
            with self.lock:
                res = self.m.add([{"role": "assistant", "content": text}], agent_id=agent, run_id=self.run_id,
                                 metadata=meta, infer=True)
            events = res.get("results", res) if isinstance(res, dict) else res
            self.ops.append({"t": time.time(), "agent": agent, "queued_s": t0 - t_req, "seconds": time.time() - t0,
                             "chars": len(text), "results": events})
            if self.on_added is not None:
                try:
                    self.on_added(events, agent)
                except Exception as exc:
                    logger.warning("on_added callback failed: %s", exc)
        except Exception as exc:
            logger.warning("mem0 add failed: %s", exc)
            self.ops.append({"t": time.time(), "agent": agent, "error": str(exc)[:500]})
        finally:
            self.q.task_done()
        return True

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        kw = {"filters": {"run_id": self.run_id}}
        if top_k:
            kw["top_k"] = top_k
        with self.lock:
            res = self.m.search(query, **kw)
        return [{"id": r["id"], "memory": r["memory"], "score": r.get("score"),
                 "agent": (r.get("agent_id") or (r.get("metadata") or {}).get("agent_id"))}
                for r in (res.get("results", res) if isinstance(res, dict) else res)]

    def features(self) -> dict:
        """Which optional parts of Mem0 are active (to check the baseline runs in full configuration)."""
        out = {}
        try:
            out["bm25_keyword_search"] = bool(self.m.vector_store._get_bm25_encoder())
        except Exception:
            out["bm25_keyword_search"] = False
        try:
            from mem0.utils.spacy_models import get_nlp_lemma
            out["spacy_lemmatization"] = get_nlp_lemma() is not None
        except Exception:
            out["spacy_lemmatization"] = False
        return out

    def all(self) -> list[dict]:
        with self.lock:
            res = self.m.get_all(filters={"run_id": self.run_id}, top_k=100_000)
        return res.get("results", res) if isinstance(res, dict) else res


# --------------------------------------------------------------------------------------- service
class Mem0Service:
    """HTTP front of a LocalMem0 for agents in other processes (GAIED)."""

    def __init__(self, local: LocalMem0, port: int) -> None:
        self.local = local
        local_ref = local

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj):
                data = json.dumps(obj, default=str).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
                if self.path == "/add":
                    local_ref.add(body["text"], body["agent_id"], body.get("metadata"))
                    self._send({"queued": local_ref.pending()})
                elif self.path == "/search":
                    self._send({"results": local_ref.search(body["query"], body.get("top_k"))})
                else:
                    self.send_error(404)

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.stop = threading.Event()
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    def _work(self):
        while not self.stop.is_set():
            self.local.process_one(timeout=1.0)

    def drain(self) -> None:
        """Process every queued write (end of campaign), then stop."""
        while self.local.pending():
            time.sleep(1.0)
        self.local.q.join()
        self.stop.set()
        self.server.shutdown()


class Mem0Client:
    """What an agent process uses: the same interface as LocalMem0, over HTTP."""

    def __init__(self, url: str, timeout: float = 600.0) -> None:
        self.url, self.timeout = url.rstrip("/"), timeout

    def _post(self, path, payload):
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(),
                                     headers={"content-type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def add(self, text, agent_id, metadata=None):
        self._post("/add", {"text": text, "agent_id": agent_id, "metadata": metadata or {}})

    def search(self, query, top_k=None):
        return self._post("/search", {"query": query, "top_k": top_k})["results"]


# --------------------------------------------------------------------------------------- memory
class Mem0Memory(Memory):
    """The harness view of Mem0: what agents write goes to Mem0; briefings are Mem0 search results."""
    name = "mem0"

    def __init__(self, *args, backend=None, top_k: int | None = None, **kw) -> None:
        super().__init__(*args, **kw)
        self.backend = backend          # LocalMem0 or Mem0Client
        self.top_k = top_k              # None: Mem0's default

    def _write(self, agent: str, text: str, node: str, iteration: int) -> None:
        if text and text.strip():
            try:
                self.backend.add(text, agent, {"node": node, "iteration": iteration})
            except Exception as exc:
                logger.warning("mem0 write failed: %s", exc)

    def record_episode(self, agent, node, text, iteration):
        super().record_episode(agent, node, text, iteration)
        if node in ("results", "conclude"):
            self._write(agent, text, node, iteration)

    def submit_conclusion(self, agent, text, iteration):
        self.record_episode(agent, "conclude", text, iteration)
        return []

    def briefing(self, agent, node, query, iteration):
        budget = self.cfg.briefing_budget_chars
        if budget <= 0:
            return ""
        try:
            hits = self.backend.search(query or self.adapter.goal.text, self.top_k)
        except Exception as exc:
            logger.warning("mem0 search failed: %s", exc)
            hits = []
        head, _ = self.fit(self.goal_lines(), budget)
        lines, items, used = [head, "MEMORIES OF THE TEAM (Mem0):"], [], len(head) + 30
        for h in hits:
            line = f"- {h['memory']}"
            if used + len(line) + 1 > budget:
                break
            used += len(line) + 1
            lines.append(line)
            items.append({"id": h["id"], "kind": "claim", "asserted": True, "text": h["memory"]})
        text = "\n".join(lines) if items else head
        self.log_briefing(agent, node, iteration, items, text)
        return text

    def maintain(self) -> int:
        n = 0
        if isinstance(self.backend, LocalMem0):
            while self.backend.process_one():
                n += 1
        return n

    def presented_knowledge(self):
        shown = {}
        for b in self.store.read_log("briefings"):
            for i in b["items"]:
                shown.setdefault(i["id"], {"id": i["id"], "text": i["text"], "author": None,
                                           "final_layer": "stored", "history": []})
        return list(shown.values())

    def export_extra(self):
        out = {}
        if isinstance(self.backend, LocalMem0):
            out["mem0_ops"] = self.backend.ops
            out["mem0_features"] = self.backend.features()
            try:
                out["mem0_memories"] = self.backend.all()
            except Exception as exc:
                out["mem0_memories_error"] = str(exc)
        return out
