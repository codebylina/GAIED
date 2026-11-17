"""
Detectors of the analysis service: deterministic rules and simple statistics over Flowcept provenance
(aggregated in AnalysisState) and the four memory layers.

Each detector returns Findings. A Finding separates:
  facts          what was observed, with evidence (Flowcept task ids, memory ids)
  explanations   candidate explanations (hypotheses)
  cause          a confirmed cause, only when a rule establishes it (with the rule's name)
and may carry proposals: an M3 knowledge proposal (always status proposed: the memory's evaluator, not
the analysis, decides its epistemic status) and an M4 guidance item.
"""
from __future__ import annotations

import operator
import statistics
from dataclasses import dataclass, field

from memory.domain import condition_id
from memory.layers.m2_experience import signature_of

from .state import KNOWN, AnalysisState

OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}


@dataclass
class AnalysisConfig:
    window: int = 8                      # recent decisions of an agent considered for behaviour detectors
    redundant_share: float = 0.5         # share of already-known proposals that makes exploration redundant
    min_decisions: int = 4
    failure_threshold: int = 3           # occurrences that make a technical failure recurrent
    similarity_threshold: float = 0.55   # mean nearest-neighbour similarity above which diversity is low
    stagnation_window: int = 6           # recent results without improving the agent's best
    region_min_entities: int = 3
    region_bad_share: float = 0.6        # share of negative / below-median results that makes a region bad
    guidance_ttl_s: float = 1800.0
    min_after: int = 3                   # decisions after a guidance before judging its effectiveness
    method: str = ""
    params: dict = field(default_factory=dict)

    @property
    def condition(self) -> str:
        return condition_id(self.method, self.params)


@dataclass
class Finding:
    kind: str                                   # M2 kind
    signature: str
    summary: str
    facts: list[str]
    severity: str = "low"
    explanations: list[str] = field(default_factory=list)
    cause: str | None = None
    cause_rule: str | None = None
    agents: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    flowcept_ids: list[str] = field(default_factory=list)
    memory_refs: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    knowledge: dict | None = None               # M3 proposal
    guidance: dict | None = None                # M4 proposal


class Detectors:
    def __init__(self, memory, state: AnalysisState, config: AnalysisConfig) -> None:
        self.m, self.st, self.cfg = memory, state, config

    # ------------------------------------------------------------------ helpers
    def recent(self, agent: str, n: int | None = None) -> list[dict]:
        return self.st.timeline(agent, last=n or self.cfg.window)

    def known_share(self, decisions: list[dict]) -> float | None:
        return sum(d["decision"] in KNOWN for d in decisions) / len(decisions) if decisions else None

    def mean_nn_similarity(self, entities: list[str]) -> float | None:
        ents = [e for e in dict.fromkeys(entities) if not e.startswith("invalid:")]
        if len(ents) < 3:
            return None
        sims = []
        for e in ents:
            sims.append(max(self.m.domain.similarity(e, o) for o in ents if o != e))
        return statistics.mean(sims)

    def agent_metrics(self, agent: str, since: float | None = None) -> dict:
        tl = self.st.timeline(agent, since=since)
        vals = [d["value"] for d in tl if d["decision"] == "computed" and d["value"] is not None]
        return {"decisions": len(tl), "known_share": self.known_share(tl),
                "similarity": self.mean_nn_similarity([d["entity"] for d in tl]),
                "best": (max(vals) if self.m.domain.direction == "max" else min(vals)) if vals else None,
                "new_results": len(vals)}

    # ------------------------------------------------------------------ 1. redundant computations
    def redundant_computations(self) -> list[Finding]:
        """Provenance: the same experiment was actually computed more than once."""
        out = []
        for key, task_ids in self.st.runs().items():
            if len(task_ids) < 2:
                continue
            cond, entity = key.split("/", 1)
            in_memory = [t for t in task_ids if self.m.store.get("M1", t) is not None]
            agents = sorted(set(self.st.run_agents(task_ids)))
            facts = [f"{entity} was computed {len(task_ids)} times (Flowcept tasks {', '.join(task_ids[:5])})",
                     f"by agents {agents}" if agents else "agents unknown",
                     f"{len(in_memory)} of these computations are recorded in the shared memory (M1)"]
            cause = rule = None
            if not in_memory:
                cause, rule = "the computations ran without the shared memory (no M1 record)", "no-M1-record"
            elif len(in_memory) < len(task_ids):
                cause, rule = "some computations bypassed the shared memory", "partial-M1-records"
            out.append(Finding(
                "redundant_computation", signature_of("redundant", key), f"redundant computation of {entity}",
                facts, severity="medium", cause=cause, cause_rule=rule, agents=agents, entities=[entity],
                flowcept_ids=task_ids, memory_refs=[f"M1:{t}" for t in in_memory],
                explanations=[] if cause else ["a claim expired during a long computation",
                                               "a repeated measurement was requested on purpose"],
                metrics={"runs": len(task_ids)},
                guidance={"kind": "instruction", "targets": agents or ["all"], "nodes": ["tool_calling"],
                          "text": f"{entity} has already been computed; reuse its result instead of computing it again.",
                          "actions": [{"type": "avoid_entities", "entities": [entity]},
                                      {"type": "reuse_results", "max_known_share": 0.34}]}))
        return out

    # ------------------------------------------------------------------ 2. redundant exploration
    def redundant_exploration(self) -> list[Finding]:
        """Memory + provenance: an agent keeps proposing experiments the team already knows."""
        out = []
        for agent in self.st.agents():
            recent = self.recent(agent)
            share = self.known_share(recent)
            if len(recent) < self.cfg.min_decisions or share is None or share < self.cfg.redundant_share:
                continue
            known = [d["entity"] for d in recent if d["decision"] in KNOWN]
            wf = self.st.workflows_of(agent)
            llm = self.st.llm_calls().get(wf.get("langgraph", ""), None)
            facts = [f"{sum(d['decision'] in KNOWN for d in recent)} of its last {len(recent)} proposals were already "
                     f"known to the team ({', '.join(dict.fromkeys(known))})"]
            if llm is not None:
                facts.append(f"its LangGraph workflow made {llm} LLM calls so far (Flowcept)")
            out.append(Finding(
                "inefficient_exploration", signature_of("redundant_exploration", agent),
                f"{agent} proposes experiments the team already knows ({share:.0%})", facts, severity="medium",
                agents=[agent], entities=list(dict.fromkeys(known)),
                memory_refs=[f"M1:{d['obs']}" for d in recent if d.get("obs")][:20],
                explanations=["the list of known molecules in the prompt is ignored or truncated",
                              "the plan keeps pointing to the same small region"],
                metrics={"known_share": round(share, 3)},
                guidance={"kind": "instruction", "targets": [agent], "nodes": ["tool_calling", "update"],
                          "text": "Most of your recent proposals were already measured by the team. Propose only "
                                  "molecules that are not in the shared memory; move to unexplored structures.",
                          "actions": [{"type": "reuse_results", "max_known_share": 0.34},
                                      {"type": "avoid_entities", "entities": list(dict.fromkeys(known))}]}))
        return out

    # ------------------------------------------------------------------ 3. recurrent technical failures
    def recurrent_failures(self) -> list[Finding]:
        out = []
        for sig, task_ids in self.st.failures().items():
            if len(task_ids) < self.cfg.failure_threshold:
                continue
            cls, _, text = sig.partition("|")
            agents = sorted(set(self.st.run_agents(task_ids)))
            out.append(Finding(
                "recurring_failure", signature_of("recurring", sig), f"recurrent {cls} failure: {text[:100]}",
                [f"{len(task_ids)} experiment tasks failed with the same error (Flowcept tasks {', '.join(task_ids[:5])})"],
                severity="high", agents=agents, flowcept_ids=task_ids,
                explanations=["resource exhaustion on the workers", "an unstable dependency or service"],
                metrics={"occurrences": len(task_ids)},
                guidance={"kind": "recovery_strategy", "targets": agents or ["all"], "nodes": ["tool_calling"],
                          "text": f"Recurrent {cls} failure ({text[:80]}): propose fewer simulations at once; failed "
                                  "simulations are retried automatically after a cooldown.",
                          "actions": [{"type": "reduce_parallelism", "max_batch": 2},
                                      {"type": "retry_policy", "backoff": "exponential"}]}))
        return out

    # ------------------------------------------------------------------ 4. repeated unsuccessful directions
    def unsuccessful_regions(self) -> list[Finding]:
        """A structural region (feature > 0) keeps giving negative or poor results while agents explore it."""
        cond = self.cfg.condition
        per = self.m.results.per_entity(cond)
        if len(per) < 2 * self.cfg.region_min_entities:
            return []
        values = [d["mean"] for d in per.values() if d["mean"] is not None]
        if not values:
            return []
        median = statistics.median(values)
        bad = lambda d: d["negative"] or (d["mean"] is not None and not self.m.domain.better(d["mean"], median))  # noqa: E731
        feats = {e: self.m.domain.features(e) for e in per}
        names = sorted({f for fs in feats.values() for f in fs})
        out = []
        for f in names:
            inside = [e for e in per if feats[e].get(f, 0) > 0]
            outside = [e for e in per if e not in inside and feats[e]]
            if len(inside) < self.cfg.region_min_entities or len(outside) < self.cfg.region_min_entities:
                continue
            share_in = sum(bad(per[e]) for e in inside) / len(inside)
            share_out = sum(bad(per[e]) for e in outside) / len(outside)
            if share_in < self.cfg.region_bad_share or share_in <= share_out + 0.2:
                continue
            ids = [i for e in inside for i in per[e]["ids"]]
            region = {"feature": f, "op": ">", "value": 0}
            out.append(Finding(
                "inefficient_exploration", signature_of("bad_region", cond, f),
                f"repeated unsuccessful direction: molecules with {f} > 0",
                [f"{sum(bad(per[e]) for e in inside)}/{len(inside)} molecules with {f} > 0 were negative or below the "
                 f"median ({median:.4g}), vs {share_out:.0%} elsewhere"],
                severity="medium", entities=inside, memory_refs=[f"M1:{i}" for i in ids][:30],
                explanations=[f"{f} may lower the {self.m.domain.objective}", "small sample: may be a coincidence"],
                metrics={"bad_share_in": round(share_in, 3), "bad_share_out": round(share_out, 3), "n_in": len(inside)},
                knowledge={"statement": f"Molecules with {f} > 0 have a lower {self.m.domain.objective}",
                           "kind": "negative_finding", "supporting": ids,
                           "claim": {"type": "feature_effect", **region, "effect":
                                     "lower" if self.m.domain.direction == "max" else "higher"}},
                guidance={"kind": "recommendation", "targets": ["all"], "nodes": ["plan", "tool_calling", "update"],
                          "text": f"Molecules with {f} have repeatedly given poor results: explore other structures.",
                          "actions": [{"type": "avoid_region", **region}]}))
        return out

    # ------------------------------------------------------------------ 5. low exploration diversity
    def low_diversity(self) -> list[Finding]:
        out = []
        for agent in self.st.agents():
            ents = [d["entity"] for d in self.recent(agent, self.cfg.window * 2)]
            sim = self.mean_nn_similarity(ents)
            if sim is None or len(set(ents)) < self.cfg.min_decisions or sim < self.cfg.similarity_threshold:
                continue
            out.append(Finding(
                "low_diversity", signature_of("low_diversity", agent),
                f"{agent} explores a narrow region (mean nearest-neighbour similarity {sim:.2f})",
                [f"its last {len(set(ents))} distinct molecules have a mean nearest-neighbour similarity of {sim:.2f} "
                 f"(threshold {self.cfg.similarity_threshold})"], severity="low", agents=[agent],
                entities=list(dict.fromkeys(ents)), metrics={"similarity": round(sim, 3)},
                explanations=["the plan only explores small variations of the seed molecule"],
                guidance={"kind": "strategy", "targets": [agent], "nodes": ["plan", "update", "tool_calling"],
                          "text": "Your recent molecules are very similar to each other: propose structurally "
                                  "different molecules (other functional groups, ring systems or sizes).",
                          "actions": [{"type": "diversify", "max_similarity": self.cfg.similarity_threshold}]}))
        return out

    # ------------------------------------------------------------------ 6. stagnation
    def stagnation(self) -> list[Finding]:
        out, better = [], self.m.domain.better
        for agent in self.st.agents():
            tl = [d for d in self.st.timeline(agent) if d["decision"] == "computed" and d["value"] is not None]
            if len(tl) < self.cfg.stagnation_window + 2:
                continue
            head, tail = tl[:-self.cfg.stagnation_window], tl[-self.cfg.stagnation_window:]
            best_before = head[0]["value"]
            for d in head:
                best_before = d["value"] if better(d["value"], best_before) else best_before
            if any(better(d["value"], best_before) for d in tail):
                continue
            hints = [k for k in self.m.knowledge.active(self.cfg.condition, ("validated", "supported"))
                     if (k.claim or {}).get("type") == "feature_effect"]
            g = {"kind": "strategy", "targets": [agent], "nodes": ["plan", "update"],
                 "text": f"No improvement in your last {len(tail)} results: change direction"
                         + (f" and use the team's knowledge: {hints[0].statement}" if hints else "."),
                 "actions": [{"type": "use_knowledge", "knowledge_id": hints[0].id}] if hints else
                            [{"type": "diversify", "max_similarity": self.cfg.similarity_threshold}]}
            out.append(Finding(
                "inefficient_exploration", signature_of("stagnation", agent),
                f"{agent} has not improved its best result in {len(tail)} new results",
                [f"best before: {best_before:.4g}; last {len(tail)} results: "
                 + ", ".join(f"{d['entity']}={d['value']:.4g}" for d in tail)],
                severity="low", agents=[agent], memory_refs=[f"M1:{d['obs']}" for d in tail if d.get("obs")],
                explanations=["local optimum around the current plan", "the objective saturates for this chemistry"],
                metrics={"window": len(tail)}, guidance=g))
        return out

    # ------------------------------------------------------------------ 7. unused knowledge
    def unused_knowledge(self) -> list[Finding]:
        """Supported / validated knowledge that no agent received although the team kept working, or that an
        agent received and then acted against (only decisions made AFTER it first saw the item count)."""
        out = []
        for k in self.m.knowledge.active(self.cfg.condition, ("validated", "supported")):
            claim = k.claim or {}
            seen = self.m.first_seen(k.id)
            if not seen:
                after = sum(len(self.st.timeline(a, since=k.updated_at)) for a in self.st.agents())
                if after >= 2 * self.cfg.min_decisions:
                    out.append(Finding(
                        "unused_knowledge", signature_of("never_retrieved", k.id),
                        f"knowledge never shown to agents: {k.statement}",
                        [f"{k.status} knowledge {k.id} was not retrieved by any agent during the last {after} decisions"],
                        severity="low", memory_refs=[f"M3:{k.id}"],
                        explanations=["ranked below other items in the prompt budget"],
                        guidance={"kind": "recommendation", "targets": ["all"], "nodes": ["plan", "update"],
                                  "text": f"Team knowledge ({k.status}): {k.statement}",
                                  "actions": [{"type": "use_knowledge", "knowledge_id": k.id}]}))
                continue
            if claim.get("type") != "feature_effect":
                continue
            harmful = (claim.get("effect") == "lower") == (self.m.domain.direction == "max")
            region = {"feature": claim["feature"], "op": claim.get("op", ">"), "value": claim.get("value", 0)}
            test = OPS.get(region["op"], operator.gt)
            for agent, t_seen in seen.items():
                recent = self.st.timeline(agent, since=t_seen)[-self.cfg.window:]
                if len(recent) < self.cfg.min_decisions:
                    continue
                inside = [d["entity"] for d in recent
                          if test(self.m.domain.features(d["entity"]).get(region["feature"], 0), float(region["value"]))]
                share = len(inside) / len(recent)
                if (harmful and share >= 0.5) or (not harmful and share == 0):
                    what = "keeps exploring the region it says to avoid" if harmful else "never explores the region it recommends"
                    out.append(Finding(
                        "unused_knowledge", signature_of("ignored", k.id, agent), f"{agent} {what}: {k.statement}",
                        [f"after receiving it, {len(inside)}/{len(recent)} of its proposals are in the region {region}"],
                        severity="medium", agents=[agent], memory_refs=[f"M3:{k.id}"], metrics={"share": round(share, 1)},
                        guidance={"kind": "instruction", "targets": [agent], "nodes": ["plan", "tool_calling", "update"],
                                  "text": f"Use the team's {k.status} knowledge: {k.statement}",
                                  "actions": [{"type": "avoid_region" if harmful else "prefer_region", **region}]}))
        return out

    # ------------------------------------------------------------------ 8. contested knowledge
    def contested_knowledge(self) -> list[Finding]:
        out = []
        for k in self.m.knowledge.active(self.cfg.condition, ("contested",)):
            out.append(Finding(
                "knowledge_conflict", signature_of("knowledge_conflict", k.id),
                f"new results contradict knowledge: {k.statement}",
                [k.evaluation.get("reason", ""), f"{len(k.contradicting)} contradicting results"], severity="medium",
                memory_refs=[f"M3:{k.id}"] + [f"M1:{i}" for i in k.contradicting[:10]],
                explanations=["the scope of the claim is too broad", "early results were not representative"],
                guidance={"kind": "recommendation", "targets": ["all"], "nodes": ["plan", "critique", "update"],
                          "text": f"Treat as uncertain (contested by new results): {k.statement}",
                          "actions": [{"type": "free_text"}]}))
        return out

    def all(self) -> list[Finding]:
        out = []
        for d in (self.redundant_computations, self.redundant_exploration, self.recurrent_failures,
                  self.unsuccessful_regions, self.low_diversity, self.stagnation, self.unused_knowledge,
                  self.contested_knowledge):
            out += d()
        return out

