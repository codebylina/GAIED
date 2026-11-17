"""
SharedMemory: the entry point used by a scientific agent (and by the analysis service).

It bundles the four layers, coordination, retrieval and lifecycle for one agent of one campaign
(namespace), and handles the cross-layer consequences of an event:
  - a new result that disagrees with an earlier measurement   -> M2 anomaly (both results kept)
  - a new result that contradicts knowledge                   -> M3 contested + M2 knowledge_conflict
  - a technical failure                                        -> M2 incident (never an M1 result)
Memory decisions that matter for an agent's behaviour (reuse, wait, block, context shown...) are
emitted as durable events for the analysis service and, when a provenance sink is given, written to
Flowcept as tasks of subtype "memory", linked to the agent's workflow and campaign.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Callable

from .coordination import Coordinator, RetryPolicy, incident_signature
from .domain import PERMANENT, Domain, condition_id
from .layers.m1_results import ResultsMemory
from .layers.m2_experience import ExperienceMemory, signature_of
from .layers.m3_knowledge import Evaluator, KnowledgeMemory
from .layers.m4_guidance import GuidanceMemory
from .lifecycle import Lifecycle, RetentionPolicy
from .retrieval import ContextBuilder
from .schemas import prov_ref
from .store import Store

log = logging.getLogger("memory")


class SharedMemory:
    def __init__(self, store: Store, domain: Domain, agent: str, *, policy: RetryPolicy | None = None,
                 retention: RetentionPolicy | None = None, provenance_sink: Callable[[dict], None] | None = None,
                 prov_context: Callable[[], dict] | None = None, min_per_group: int = 2,
                 min_confirm: int = 2) -> None:
        self.store, self.domain, self.agent = store, domain, agent
        self.sink, self.prov_context = provenance_sink, prov_context
        self.results = ResultsMemory(store, domain)
        self.experience = ExperienceMemory(store)
        self.knowledge = KnowledgeMemory(store, self.results, domain,
                                         Evaluator(self.results, domain, min_per_group, min_confirm))
        self.guidance = GuidanceMemory(store, domain)
        self.coord = Coordinator(self, policy)
        self.contexts = ContextBuilder(self)
        self.lifecycle = Lifecycle(self, retention)

    @classmethod
    def connect(cls, domain: Domain, namespace: str, agent: str, host: str = "localhost", port: int = 6379,
                **kw) -> "SharedMemory":
        return cls(Store.connect(namespace, host, port), domain, agent, **kw)

    @property
    def namespace(self) -> str:
        return self.store.namespace

    # ------------------------------------------------------------------ provenance links
    def prov_ids(self) -> dict:
        ids = self.prov_context() if self.prov_context else {}
        return {k: v for k, v in ids.items() if k in ("workflow_id", "campaign_id") and v}

    def _to_flowcept(self, activity: str, used: dict, generated: dict, t0: float) -> str | None:
        """Write a memory operation to the agent's Flowcept provenance (if a sink was given)."""
        if self.sink is None:
            return None
        ctx = self.prov_context() if self.prov_context else {}
        rec = {"task_id": str(uuid.uuid4()), "activity_id": activity, "subtype": "memory", "status": "FINISHED",
               "started_at": t0, "ended_at": time.time(), "used": used, "generated": generated,
               "custom_metadata": {"agent": self.agent, "memory_namespace": self.namespace},
               **{k: v for k, v in ctx.items() if k in ("workflow_id", "campaign_id", "parent_task_id") and v}}
        try:
            self.sink(rec)
            return rec["task_id"]
        except Exception as exc:
            log.warning("provenance sink failed: %s", exc)
            return None

    def decision(self, kind: str, entity: str, condition: str, **data) -> None:
        """A coordination decision (reuse, waited, computed, blocked_..., failed_...)."""
        self.store.emit("decision", decision=kind, agent=self.agent, entity=entity, condition=condition, **data)
        self.store.metric(f"decisions.{kind}")
        if kind in ("reuse", "waited", "blocked_negative", "blocked_cooldown"):
            self._to_flowcept(f"memory.{kind}", {"entity": entity, "condition": condition},
                              {k: v for k, v in data.items()}, time.time())

    def register_agent(self, method: str, params: dict | None, workflows: dict | None = None,
                       campaign_id: str | None = None) -> None:
        """Announce the agent (and its Flowcept workflow ids and campaign) to the analysis service."""
        if campaign_id:
            self.store.r.sadd(self.store.k("campaigns"), campaign_id)
        self.store.emit("agent", agent=self.agent, workflows={**(workflows or {}), "method": method,
                                                              "params": params or {}}, campaign_id=campaign_id)

    # ------------------------------------------------------------------ results and failures
    def record_result(self, raw: str, *, method: str, params: dict | None, value: float | None = None,
                      negative_detail: str | None = None, task_id: str | None = None,
                      duration_s: float | None = None):
        ids = self.prov_ids()
        rec, conflict = self.results.record(raw, method=method, params=params, agent=self.agent, value=value,
                                            negative_detail=negative_detail, task_id=task_id,
                                            workflow_id=ids.get("workflow_id"),
                                            campaign_id=ids.get("campaign_id"), duration_s=duration_s)
        if conflict:
            self.experience.report(
                "anomaly", f"repeated measurement of {rec.entity} disagrees with earlier ones ({conflict['type']})",
                signature_of("measurement_conflict", rec.condition, rec.entity), by=self.agent,
                facts=[f"{conflict}"], severity="medium", entities=[rec.entity],
                memory_refs=[f"M1:{i}" for i in conflict["ids"]],
                explanations=["non-deterministic simulation, or different settings recorded under one condition"])
        for t in self.knowledge.on_result(rec.entity, rec.condition, by="knowledge-evaluator"):
            if t["to"] == "contested":
                k = self.store.get("M3", t["id"])
                self.experience.report(
                    "knowledge_conflict", f"new results contradict knowledge: {k.statement}",
                    signature_of("knowledge_conflict", k.id), by="knowledge-evaluator", severity="medium",
                    facts=[t["reason"], f"triggered by result {rec.id} ({rec.entity})"],
                    memory_refs=[f"M3:{k.id}", f"M1:{rec.id}"], entities=[rec.entity],
                    provenance=[prov_ref(rec.id, **ids)] if task_id else [])
        return rec

    def incident(self, entity: str, method: str, error: str, cls: str, task_id: str, attempt: int):
        try:
            rec, _ = self.experience.report(
                "incident", f"{cls} technical failure of {method}: {error[:160]}",
                incident_signature(method, error, cls), by=self.agent, error_class=cls,
                facts=[f"attempt {attempt} on {entity} failed: {error[:300]}"],
                severity="high" if cls == PERMANENT else "medium", agents=[self.agent], entities=[entity],
                provenance=[prov_ref(task_id, **self.prov_ids())], ttl_s=7 * 86400)
            return rec
        except Exception as exc:
            log.warning("could not record incident: %s", exc)
            return None

    async def get_or_compute(self, raw: str, compute, *, method: str, params: dict | None = None,
                             poll_s: float = 1.0) -> float:
        return await self.coord.get_or_compute(raw, compute, method=method, params=params, poll_s=poll_s)

    def known_entities(self, method: str, params: dict | None, limit: int = 200) -> list[str]:
        """Measured, negative, or being computed: not worth proposing again."""
        cond = condition_id(method, params)
        best = [e for e, _ in self.results.best(cond, limit)]
        neg = list(self.results.negatives(cond))
        running = [t.split("/", 1)[1] for t in self.coord.running() if t.startswith(cond + "/")]
        return list(dict.fromkeys(e.removeprefix("invalid:") for e in best + neg + running))[:limit]

    # ------------------------------------------------------------------ knowledge
    def propose_knowledge(self, statement: str, *, kind: str = "claim", claim: dict | None = None,
                          method: str = "", params: dict | None = None, supporting: list[str] | None = None,
                          applicability: str = "", source: str = "agent"):
        cond = condition_id(method, params) if method else ""
        t0 = time.time()
        rec, created, transitions = self.knowledge.propose(
            statement, kind=kind, claim=claim, condition=cond, scope={"method": method, "params": params or {}},
            applicability=applicability, supporting=supporting, source=source, by=self.agent)
        self._to_flowcept("memory.propose_knowledge", {"statement": statement, "claim": claim,
                                                       "supporting": supporting or []},
                          {"knowledge_id": rec.id, "status": rec.status, "created": created}, t0)
        return rec, created, transitions

    # ------------------------------------------------------------------ context for prompts
    def context(self, node: str, method: str, params: dict | None, query: str = "",
                budget_tokens: int = 700) -> tuple[str, dict]:
        """Memory text for the prompt of one node, with the ids shown (guidance marked delivered)."""
        t0, cond = time.time(), condition_id(method, params)
        with self.store.timed("retrieval.context"):
            new = set(self.guidance.poll_inbox(self.agent))
            text, shown = self.contexts.build(self.agent, node, cond, query, budget_tokens, new)
        for gid in shown["M4"]:
            self.guidance.mark(gid, self.agent, "delivered", {"node": node})
        for layer in ("M2", "M3"):
            for rid in shown[layer]:
                self.store.use(rid, "retrieved")
        self.store.emit("context", agent=self.agent, node=node, condition=cond, shown=shown, chars=len(text))
        self._to_flowcept("memory.context", {"node": node, "condition": cond},
                          {"shown": shown, "chars": len(text)}, t0)
        return text, shown

    def acknowledge(self, shown: dict) -> None:
        """The LLM call that received this context completed."""
        for gid in shown.get("M4", []):
            self.guidance.mark(gid, self.agent, "acknowledged")

    def check_guidance(self, proposals: list[str], method: str, params: dict | None,
                       history: list[str] | None = None) -> list[dict]:
        """After the agent proposed experiments: do they comply with the guidance it received?"""
        cond = condition_id(method, params)
        entities = [self.results.key_of(p) for p in proposals]
        known = set(self.known_entities(method, params, limit=10_000))
        out = []
        for g in self.guidance.for_agent(self.agent, condition=cond):
            if self.agent not in self.guidance.delivery(g.id):
                continue
            ok, detail = self.guidance.compliance(g, entities, history, known)
            if ok is None:
                continue
            self.guidance.mark(g.id, self.agent, "applied" if ok else "violated", detail)
            out.append({"guidance": g.id, "compliant": ok, **detail})
        if out:
            self.store.emit("compliance", agent=self.agent, results=out, proposals=entities)
        return out

    def report(self) -> dict:
        return self.lifecycle.quality()
