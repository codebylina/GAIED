"""
Feedback of the analysis service into the shared memory, and measurement of its effect.

For each finding:
  M2  the episode (facts, candidate explanations, confirmed cause only with a rule), deduplicated by
      signature; the actions taken in response are appended to it
  M3  an optional knowledge proposal, ALWAYS with status "proposed": the memory's deterministic
      evaluator decides whether the measurements support it; the analysis never establishes knowledge
  M4  an optional guidance item for the target agents (or the whole campaign), with a baseline of the
      targets' behaviour at publication time. Operational guidance is published "active"; guidance that
      rests on a knowledge proposal is "active" only if that knowledge is supported / validated,
      otherwise "proposed"

Effectiveness: once the targets made enough decisions after the guidance was published, their
behaviour is compared with the baseline (share of already-known proposals, diversity, best result,
compliance with the structured actions). Improved -> validated; not improved -> ineffective (and an
M2 "ineffective_guidance" episode). Guidance based on knowledge that became contested / refuted is retired.
"""
from __future__ import annotations

import logging
import time

from memory.layers.m2_experience import signature_of
from memory.schemas import prov_ref

from .detectors import AnalysisConfig, Detectors, Finding

log = logging.getLogger("analysis.feedback")
BY = "analysis"


class Feedback:
    def __init__(self, memory, detectors: Detectors, config: AnalysisConfig, campaign_id: str | None,
                 explainer=None) -> None:
        self.m, self.d, self.cfg, self.campaign, self.explainer = memory, detectors, config, campaign_id, explainer

    def _refs(self, ids: list[str]) -> list[dict]:
        return [prov_ref(t, campaign_id=self.campaign) for t in ids][:50]

    def apply(self, f: Finding) -> dict:
        rec, created = self.m.experience.report(
            f.kind, f.summary, f.signature, by=BY, facts=f.facts, explanations=f.explanations, severity=f.severity,
            agents=f.agents, entities=f.entities, provenance=self._refs(f.flowcept_ids), memory_refs=f.memory_refs,
            metrics=f.metrics)
        out = {"finding": rec.id, "kind": f.kind, "created": created}
        if f.cause and rec.status not in ("confirmed", "resolved"):
            self.m.experience.confirm_cause(rec.id, f.cause, by=BY, method=f"rule:{f.cause_rule}")
        if created and self.explainer is not None:
            text = self.explainer.explain(f)
            if text:
                self.m.experience.add_explanation(rec.id, f"LLM hypothesis (unverified): {text}", by=BY)
        knowledge_status = None
        if f.knowledge:
            k, k_created, _ = self.m.propose_knowledge(
                f.knowledge["statement"], kind=f.knowledge.get("kind", "hypothesis"), claim=f.knowledge.get("claim"),
                method=self.cfg.method, params=self.cfg.params, supporting=f.knowledge.get("supporting"),
                source="analysis")
            knowledge_status = k.status
            out["knowledge"] = {"id": k.id, "status": k.status}
            if k_created:
                self.m.experience.record_action(rec.id, {"type": "knowledge_proposed", "ref": f"M3:{k.id}",
                                                         "status": k.status}, by=BY)
        if f.guidance:
            g = f.guidance
            status = "active"
            if knowledge_status is not None and knowledge_status not in ("supported", "validated"):
                status = "proposed"
            targets = g.get("targets") or ["all"]
            baseline = {a: self.d.agent_metrics(a) for a in (self.d.st.agents() if targets == ["all"] else targets)}
            refs = [f"M2:{rec.id}"] + ([f"M3:{out['knowledge']['id']}"] if "knowledge" in out else []) + f.memory_refs[:10]
            gr, g_created = self.m.guidance.publish(
                g["text"], by=BY, kind=g.get("kind", "instruction"), actions=g.get("actions"), targets=targets,
                condition=self.cfg.condition, nodes=g.get("nodes"), memory_refs=refs,
                provenance=self._refs(f.flowcept_ids), trigger={"finding": rec.id, "signature": f.signature},
                limitations="derived from the observations listed in the trigger episode; may not generalize",
                status=status, ttl_s=self.cfg.guidance_ttl_s, source="analysis", baseline=baseline,
                key=signature_of("guidance", f.signature))
            out["guidance"] = {"id": gr.id, "status": gr.status, "created": g_created}
            if g_created:
                self.m.experience.record_action(rec.id, {"type": "guidance_published", "ref": f"M4:{gr.id}",
                                                         "status": gr.status, "targets": targets}, by=BY)
        self.m.store.emit("analysis_finding", **out, signature=f.signature)
        return out

    # ------------------------------------------------------------------ effectiveness
    def evaluate_guidance(self) -> list[dict]:
        out = []
        for g in self.m.store.find("M4", status=["active", "proposed"], condition=self.cfg.condition):
            if g.source != "analysis":
                continue
            # outdated: rests on knowledge that is no longer supported
            stale = [r for r in g.memory_refs if r.startswith("M3:")
                     and (k := self.m.store.get("M3", r[3:])) is not None and k.status in ("contested", "refuted")]
            if stale:
                self.m.guidance.set_status(g.id, "retired", BY, f"based on knowledge now contested/refuted: {stale}")
                out.append({"guidance": g.id, "verdict": "retired_outdated"})
                continue
            if g.status != "active" or not g.baseline:
                continue
            verdict = self._judge(g)
            if verdict:
                out.append(verdict)
        return out

    def _judge(self, g) -> dict | None:
        since = g.created_at
        delivery = self.m.guidance.delivery(g.id)
        agents = list(g.baseline)
        after = {a: self.d.agent_metrics(a, since=since) for a in agents}
        n_after = sum(m["decisions"] for m in after.values())
        if n_after < self.cfg.min_after:
            return None
        applied = sum(d.get("applied_n", 0) for d in delivery.values())
        violated = sum(d.get("violated_n", 0) for d in delivery.values())
        types = {a["type"] for a in g.actions}
        improved = []
        for a in agents:
            before, now = g.baseline[a], after[a]
            if types & {"reuse_results", "avoid_entities"} and before.get("known_share") is not None \
                    and now["known_share"] is not None:
                improved.append(now["known_share"] < before["known_share"])
            if "diversify" in types and before.get("similarity") is not None and now["similarity"] is not None:
                improved.append(now["similarity"] < before["similarity"])
            if types & {"use_knowledge", "prefer_region"} and now["best"] is not None:
                improved.append(before.get("best") is None or self.m.domain.better(now["best"], before["best"]))
        if types & {"avoid_region", "prefer_region"} and applied + violated:
            improved.append(applied / (applied + violated) >= 0.5)
        if not improved:
            return None
        effectiveness = {"evaluated_at": time.time(), "after": after, "applied": applied, "violated": violated,
                         "improved_checks": improved, "decisions_after": n_after}
        if sum(improved) / len(improved) >= 0.5:
            self.m.guidance.set_status(g.id, "validated", BY, "targets' behaviour improved after delivery",
                                       effectiveness={**effectiveness, "verdict": "effective"})
            return {"guidance": g.id, "verdict": "effective"}
        if n_after >= 2 * self.cfg.min_after:
            self.m.guidance.set_status(g.id, "ineffective", BY, "no improvement after delivery",
                                       effectiveness={**effectiveness, "verdict": "ineffective"})
            self.m.experience.report(
                "ineffective_guidance", f"guidance did not change behaviour: {g.text[:100]}",
                signature_of("ineffective", g.id), by=BY, severity="low", agents=agents,
                facts=[f"after {n_after} decisions: applied {applied}, violated {violated}, checks {improved}"],
                memory_refs=[f"M4:{g.id}"], explanations=["the instruction is drowned in a long prompt",
                                                           "the model cannot follow this kind of instruction"])
            return {"guidance": g.id, "verdict": "ineffective"}
        return None
