"""
FALDA (https://github.com/rick-stevens-ai/falda, commit ff6d83b) as a memory condition of the harness, used the way
its harness guide describes a "live memory provider" (docs/HARNESS_INTEGRATION.md, section 1b). Nothing of FALDA is
changed: the gateway is built from the official repository (docker/falda/Dockerfile) and the distiller is FALDA's
own sidecar, copied unchanged (memlab/third_party/falda_distiller.py, Apache 2.0).

  write   each turn of an agent goes to the T0 Stream of the campaign's shared pool       (/stream/add)
  recall  Core (/core/read) + top atoms (/atoms/search) + top past turns (/stream/search), hybrid dense + BM25
          search fused by RRF inside the gateway; 10 atoms and 10 turns, FALDA's defaults
  distill FALDA's distiller promotes T0 -> T1 atoms -> T2 scenes -> T3 core on its own cadence
  sharing one pool per run (docs/POOLS.md): every agent is a tenant with read-write access; the Goal is the Core
          the orchestrator writes at the start

Harness choices, the same for every condition and stated in the paper:
  - the distiller runs inside the memory's background loop (one instance per pool) instead of as a separate
    process; its LLM transport (`argo_chat`) is replaced by the harness LLM client (same model, same settings
    as the agents, thinking off, calls counted), which is what docker/falda + the Ollama relay did before;
  - its cadence is shortened (FALDA's defaults: atoms every 10 turns, a scene per hour, the core every 6 hours,
    which a 30-minute campaign never reaches); the values are in the condition's "falda" settings;
  - what is put in a prompt is cut to the briefing budget shared by all conditions (FALDA defines none).
FALDA has no coordination between agents and no link between a memory and the measurements; none is added.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import re
import tempfile
import time
import urllib.error
import urllib.request

from .base import Memory, select_within_budget

logger = logging.getLogger(__name__)
_DISTILLER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "falda_distiller.py")
_NAME_OK = re.compile(r"[^a-z0-9_-]+")

DEFAULT_CADENCE = {"l1_every_n": 6, "l2_interval_s": 300, "l3_interval_s": 900, "poll_s": 30}


def falda_name(text: str, prefix: str) -> str:
    """A valid FALDA tenant / pool name (^[a-z0-9][a-z0-9_-]{0,63}$)."""
    name = _NAME_OK.sub("-", (text or "").lower()).strip("-")
    return f"{prefix}-{name}"[:64].rstrip("-")


class FaldaMemory(Memory):
    name = "falda"

    def __init__(self, store, adapter, cfg, llm=None, clock=time.time, url: str | None = None,
                 pool: str = "default", k_atoms: int = 10, k_stream: int = 10, cadence: dict | None = None,
                 write_nodes=("plan", "results", "conclude", "update"), timeout_s: float = 60.0,
                 declare: bool = True) -> None:
        super().__init__(store, adapter, cfg, llm=llm, clock=clock)
        self.url = (url or os.environ.get("FALDA_URL", "http://localhost:8077")).rstrip("/")
        self.pool = falda_name(pool, "pool")
        self.k_atoms, self.k_stream, self.timeout_s = k_atoms, k_stream, timeout_s
        self.write_nodes = tuple(write_nodes)
        self.cadence = {**DEFAULT_CADENCE, **(cadence or {})}
        self._distiller = None
        self._last_poll = 0.0
        if declare:
            self.call("/pools/declare", {"name": self.pool, "members": {"orchestrator": "readwrite",
                                                                        "distiller": "readwrite"},
                                         "description": "shared memory of one campaign"}, tenant=None, ok_exists=True)
            self.call("/core/write", {"content": "\n".join(self.goal_lines())}, tenant="orchestrator")

    # ------------------------------------------------------------------ HTTP
    def call(self, route: str, payload: dict, tenant: str | None = "orchestrator", ok_exists: bool = False) -> dict:
        body = dict(payload)
        if tenant is not None:
            body.setdefault("tenant", tenant)
            body.setdefault("pool", self.pool)
        req = urllib.request.Request(f"{self.url}{route}", data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode() or "{}")
        except urllib.error.HTTPError as exc:
            msg = exc.read().decode()[:300]
            if ok_exists and "exist" in msg:
                return {}
            raise RuntimeError(f"FALDA {route}: HTTP {exc.code} {msg}") from exc

    def tenant(self, agent: str) -> str:
        return falda_name(agent, "agent")

    # ------------------------------------------------------------------ write (T0)
    def register_agent(self, agent: str) -> None:
        super().register_agent(agent)
        self.call("/pools/grant", {"name": self.pool, "tenant": self.tenant(agent), "access": "readwrite"},
                  tenant=None)

    def record_episode(self, agent, node, text, iteration):
        super().record_episode(agent, node, text, iteration)
        if node not in self.write_nodes or not text or not text.strip():
            return
        role = "tool" if node == "results" else "assistant"
        try:
            self.call("/stream/add", {"session_id": self.tenant(agent),
                                      "messages": [{"role": role, "content": f"[{agent} | {node}] {text.strip()}"}]},
                      tenant=self.tenant(agent))
        except Exception as exc:  # noqa: BLE001
            logger.warning("FALDA write failed: %s", exc)

    # ------------------------------------------------------------------ recall
    def briefing(self, agent, node, query, iteration):
        budget = self.cfg.briefing_budget_chars
        if budget <= 0:
            return ""
        q = query or self.adapter.goal.text
        t = self.tenant(agent)
        try:
            core = self.call("/core/read", {}, tenant=t).get("content") or ""
            atoms = self.call("/atoms/search", {"query": q, "limit": self.k_atoms}, tenant=t).get("items", [])
            turns = self.call("/stream/search", {"query": q, "limit": self.k_stream}, tenant=t).get("messages", [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("FALDA recall failed: %s", exc)
            return ""
        core_items = []
        for k, line in enumerate(l.strip() for l in core.splitlines()):
            if line and not line.startswith("#"):
                core_items.append((f"core:{hashlib.sha1(line.encode()).hexdigest()[:12]}", True, line if line.startswith("-") else f"- {line}",
                                   line.lstrip("-* ").strip()))
        sections = [("PROJECT CORE (shared memory):", core_items),
                    ("RELEVANT MEMORIES (shared memory):",
                     [(a["id"], True, f"- ({a.get('type', 'fact')}) {a.get('content', '')}", a.get("content", ""))
                      for a in atoms]),
                    ("RELEVANT PAST MESSAGES OF THE TEAM (shared memory):",
                     [(f"msg:{m.get('id')}", None, f"- {m.get('content', '')}", m.get("content", "")) for m in turns])]
        keep = select_within_budget("", sections, lambda c: c[2], budget)
        lines, items = [], []
        for title, sec in sections:
            sel = [c for c in sec if id(c) in keep]
            if sel:
                lines.append(title)
                for c in sel:
                    lines.append(c[2])
                    kind = "message" if c[0].startswith("msg:") else "claim"
                    items.append({"id": c[0], "kind": kind, "asserted": c[1], "text": c[3]})
        text = "\n".join(lines)
        self.log_briefing(agent, node, iteration, items, text)
        return text

    # ------------------------------------------------------------------ FALDA's distiller (unchanged)
    def _load_distiller(self):
        spec = importlib.util.spec_from_file_location(f"falda_distiller_{self.pool}", _DISTILLER)
        d = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(d)
        work = tempfile.mkdtemp(prefix=f"falda-{self.pool}-")
        d.FALDA, d.FALDA_TENANT, d.FALDA_POOL = self.url, "distiller", self.pool
        d.STATE, d.LOG = os.path.join(work, "state.json"), os.path.join(work, "distiller.log")
        d.L1_EVERY_N = int(self.cadence["l1_every_n"])
        d.L2_INTERVAL_S = int(self.cadence["l2_interval_s"])
        d.L3_INTERVAL_S = int(self.cadence["l3_interval_s"])
        llm = self.llm

        def argo_chat(system, user, max_tokens=2000, timeout=180):    # transport only: the harness LLM client
            return llm.complete(system, user, purpose="memory", max_tokens=max_tokens)
        d.argo_chat = argo_chat
        return d

    def maintain(self) -> int:
        """One iteration of the distiller's loop (falda_distiller.loop), at most every poll_s seconds."""
        if self.llm is None:
            return 0
        now = time.time()
        if now - self._last_poll < float(self.cadence["poll_s"]):
            return 0
        self._last_poll = now
        if self._distiller is None:
            self._distiller = self._load_distiller()
        d = self._distiller
        s = d.load_state()
        before = (s.get("atoms_made", 0), s.get("scenes_made", 0), s.get("cores_made", 0))
        try:
            q = d.falda("/stream/query", {"limit": 1})
            total = q.get("total", 0)
            last_ts, new_count = s["last_ts"], 0
            if total:
                allq = d.falda("/stream/query", {"limit": 500})
                new_count = (sum(1 for m in allq.get("messages", []) if m.get("timestamp", "") > last_ts)
                             if last_ts else total)
            l1 = new_count >= d.L1_EVERY_N or (new_count > 0 and not last_ts)
            if l1:
                d.run_l1(s)
            if now - s.get("last_l2", 0) >= d.L2_INTERVAL_S:
                d.run_l2(s)
            if now - s.get("last_l3", 0) >= d.L3_INTERVAL_S:
                d.run_l3(s)
            d.save_state(s)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FALDA distiller step failed: %s", exc)
            return 0
        after = (s.get("atoms_made", 0), s.get("scenes_made", 0), s.get("cores_made", 0))
        return int(l1) + sum(a - b for a, b in zip(after, before))

    def finish(self) -> None:
        """At the end of a run: one last pass, as the distiller's --once mode does."""
        if self.llm is None:
            return
        if self._distiller is None:
            self._distiller = self._load_distiller()
        d = self._distiller
        s = d.load_state()
        try:
            d.run_l1(s)
            d.save_state(s)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FALDA final distillation failed: %s", exc)

    # ------------------------------------------------------------------ evaluation and export
    def presented_knowledge(self):
        out = {}
        for b in self.store.read_log("briefings"):
            for i in b["items"]:
                if i["kind"] == "claim":
                    out.setdefault(i["id"], {"id": i["id"], "text": i["text"], "final_layer": "stored"})
        return list(out.values())

    def export_extra(self):
        out = {"pool": self.pool, "cadence": self.cadence}
        try:
            out["atoms"] = self.call("/atoms/query", {"limit": 1000}).get("items", [])
            out["core"] = self.call("/core/read", {}).get("content", "")
            scenes = self.call("/scenes/ls", {"prefix": ""}).get("entries", [])
            out["scenes"] = [{**e, "content": self.call("/scenes/read", {"path": e["path"]}).get("content")}
                             for e in scenes]
            out["stream_total"] = self.call("/stream/query", {"limit": 1}).get("total", 0)
        except Exception as exc:  # noqa: BLE001
            out["error"] = str(exc)
        if self._distiller is not None:
            try:
                out["distiller_state"] = self._distiller.load_state()
                out["distiller_log"] = open(self._distiller.LOG).read()[-20000:]
            except OSError:
                pass
        return out
