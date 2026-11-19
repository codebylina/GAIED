"""
Entries of the shared memory. One envelope, four categories (M1-M4), each answering one question:

  M1  What was observed?                       scientific evidence and results
  M2  What happened during an experience?      attempts, successes, failures, investigations, diagnoses, outcomes
  M3  What was learned scientifically?         hypotheses, claims, findings, rules, constraints, negative findings,
                                               uncertainties, with scope, evidence for / against, epistemic status
  M4  How should an agent act?                 procedures, strategies, recommendations, with trigger, preconditions,
                                               actions, expected outcome, limitations, validation status, evidence

Entries hold references to Flowcept (ProvRef), never copies of provenance records. Updates never overwrite: they
append to `history` (who, when, what changed), so evidence and earlier states are kept.

Guidance is the delivery record of a recommendation sent by the analysis agent to scientific agents; it is what lets
the system track whether guidance was delivered, applied and effective.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

CATEGORIES = ("M1", "M2", "M3", "M4")

M2_KINDS = ("attempt", "success", "failure", "investigation", "diagnosis", "outcome")
M3_KINDS = ("hypothesis", "claim", "finding", "rule", "constraint", "negative_finding", "uncertainty")
M4_KINDS = ("procedure", "strategy", "recommendation")

# Distinction required for knowledge: observation-based, hypothesis, or explanation.
EPISTEMIC_STATUS = ("observation", "hypothesis", "explanation")
# Distinction required for the analysis agent's own statements.
FINDING_BASIS = ("observed_fact", "candidate_explanation", "confirmed_cause")
# Problem types the analysis agent must tell apart.
PROBLEM_TYPES = ("technical_failure", "data_quality", "workflow", "scientific_contradiction")


def new_id(prefix: str) -> str:
    return f"{prefix.lower()}:{uuid.uuid4().hex[:12]}"


def now() -> float:
    return time.time()


@dataclass
class ProvRef:
    """Reference to a Flowcept record (task, workflow, campaign); the details stay in Flowcept."""
    task_id: str | None = None
    workflow_id: str | None = None
    campaign_id: str | None = None
    subtype: str | None = None
    activity_id: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Entry:
    category: str
    kind: str
    text: str                            # one human-readable sentence
    key: str = ""                        # exact-retrieval key: canonical molecule, failure class, agent...
    author: str = ""
    campaign_id: str | None = None
    prov_refs: list[dict] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)      # ids of other memory entries this one relies on
    data: dict[str, Any] = field(default_factory=dict)  # category-specific fields (see the constructors below)
    id: str = ""
    created_at: float = 0.0
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown category {self.category}")
        self.id = self.id or new_id(self.category)
        self.created_at = self.created_at or now()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Entry":
        return cls(**d)


# --------------------------------------------------------------------------------------------- constructors
def observation(entity: str, quantity: str, value: float, unit: str, conditions: dict, author: str,
                prov: ProvRef, campaign_id: str | None = None) -> Entry:
    """M1: a measurement with its experimental conditions and the reference to the task that produced it."""
    return Entry("M1", "measurement", f"{quantity} of {entity} = {value:.6g} {unit}", key=entity, author=author,
                 campaign_id=campaign_id, prov_refs=[prov.to_dict()],
                 data={"entity": entity, "quantity": quantity, "value": value, "unit": unit,
                       "conditions": conditions})


def experience(kind: str, text: str, author: str, key: str = "", context: dict | None = None,
               outcome: dict | None = None, prov_refs: list[dict] | None = None, refs: list[str] | None = None,
               problem_type: str | None = None, basis: str | None = None,
               campaign_id: str | None = None) -> Entry:
    """M2: what happened (not a raw log). `problem_type` and `basis` are set on the analysis agent's diagnoses."""
    if kind not in M2_KINDS:
        raise ValueError(f"M2 kind must be one of {M2_KINDS}")
    if problem_type is not None and problem_type not in PROBLEM_TYPES:
        raise ValueError(f"problem_type must be one of {PROBLEM_TYPES}")
    if basis is not None and basis not in FINDING_BASIS:
        raise ValueError(f"basis must be one of {FINDING_BASIS}")
    return Entry("M2", kind, text, key=key, author=author, campaign_id=campaign_id, prov_refs=prov_refs or [],
                 refs=refs or [], data={"context": context or {}, "outcome": outcome or {},
                                        "problem_type": problem_type, "basis": basis})


def knowledge(kind: str, statement: str, epistemic_status: str, author: str, scope: str = "",
              supporting: list[dict] | None = None, contradicting: list[dict] | None = None,
              uncertainty: str = "", prov_refs: list[dict] | None = None, key: str = "",
              campaign_id: str | None = None) -> Entry:
    """M3: a scientific statement with its scope, evidence for and against (references to M1 entries and Flowcept
    tasks), epistemic status and stated uncertainty."""
    if kind not in M3_KINDS:
        raise ValueError(f"M3 kind must be one of {M3_KINDS}")
    if epistemic_status not in EPISTEMIC_STATUS:
        raise ValueError(f"epistemic_status must be one of {EPISTEMIC_STATUS}")
    sup, con = supporting or [], contradicting or []
    return Entry("M3", kind, statement, key=key, author=author, campaign_id=campaign_id, prov_refs=prov_refs or [],
                 refs=[e["entry"] for e in sup + con if e.get("entry")],
                 data={"scope": scope, "epistemic_status": epistemic_status, "uncertainty": uncertainty,
                       "supporting_evidence": sup, "contradicting_evidence": con})


def procedure(kind: str, text: str, trigger: str, preconditions: str, actions: str, expected_outcome: str,
              limitations: str, supporting_episodes: list[str], author: str, key: str = "",
              prov_refs: list[dict] | None = None, campaign_id: str | None = None) -> Entry:
    """M4: reusable guidance. Its validation status records what is known about it: the experiences it was derived
    from and the observed outcomes of its applications (filled in as guidance is applied)."""
    if kind not in M4_KINDS:
        raise ValueError(f"M4 kind must be one of {M4_KINDS}")
    return Entry("M4", kind, text, key=key, author=author, campaign_id=campaign_id, prov_refs=prov_refs or [],
                 refs=list(supporting_episodes),
                 data={"trigger": trigger, "preconditions": preconditions, "actions": actions,
                       "expected_outcome": expected_outcome, "limitations": limitations,
                       "supporting_episodes": list(supporting_episodes), "applications": []})


def validation_status(m4: Entry) -> str:
    """Descriptive validation status of a procedure, computed from the evidence held, with no threshold."""
    apps = m4.data.get("applications", [])
    eff = [a for a in apps if a.get("effective") is True]
    inef = [a for a in apps if a.get("effective") is False]
    n = len(m4.data.get("supporting_episodes", []))
    if not apps:
        return f"derived from {n} experience(s); not given to an agent yet"
    applied = [a for a in apps if a.get("applied") is True]
    not_applied = [a for a in apps if a.get("applied") is False]
    text = f"derived from {n} experience(s); given {len(apps)} time(s): applied {len(applied)}"
    if applied:
        text += f" ({len(eff)} effective, {len(inef)} not effective, {len(applied) - len(eff) - len(inef)} effect unknown)"
    return text + f", not applied {len(not_applied)}"


@dataclass
class Guidance:
    """A recommendation from the analysis agent to scientific agents, and its tracking."""
    message: str
    problem_type: str | None
    basis: str
    target: str = "*"                    # an agent name, or "*" for every agent
    evidence: list[str] = field(default_factory=list)       # memory entry ids
    prov_refs: list[dict] = field(default_factory=list)
    check: dict = field(default_factory=dict)   # machine-checkable part: {"avoid_keys": [...], "failure_class": ...}
    procedure_id: str | None = None
    id: str = ""
    created_at: float = 0.0
    delivered: list[dict] = field(default_factory=list)     # {agent, node, t}
    applied: dict = field(default_factory=dict)              # agent -> True / False / None (not checkable)
    effective: dict = field(default_factory=dict)            # agent -> True / False / None

    def __post_init__(self) -> None:
        self.id = self.id or new_id("G")
        self.created_at = self.created_at or now()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Guidance":
        return cls(**d)
