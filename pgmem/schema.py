"""
Nodes of the shared memory. The memory is a property graph: its nodes are the entries M1-M4, the domain's entities
(molecules, motifs) and the analysis agent's guidance; every relation between nodes is an edge of pgmem/graph.py,
never a field of an entry. An entry holds only what describes it, plus its references to Flowcept (ProvRef), which
are external identifiers, not nodes.

  M1  result      What was observed?            a measured value with its experimental conditions
  M2  attempt     What happened?                one iteration of an agent: what it proposed, whether it improved
      failure                                   a computation that failed, with its failure class
      diagnosis                                 the analysis agent's reading of a problem (rule, problem type)
  M3  knowledge   What was learned?             a statement, its scope, epistemic status and uncertainty
  M4  procedure   How should an agent act?      trigger, preconditions, actions, expected outcome, limitations

Reliability has two independent axes. The epistemic status is declared by the author of a statement (an agent or
the analysis agent): observation (what was seen), hypothesis (what is supposed), explanation (a proposed cause). The
verdict (supported, contradicted, inconclusive) is computed by code from the evidence and kept apart
(pgmem/verification.py). An entry is never rewritten: a duplicate is merged into it by appending its Flowcept
references, and the merge is written in its history.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from .domain import DOMAIN

CATEGORIES = ("M1", "M2", "M3", "M4")
KINDS = {"M1": ("result",), "M2": ("attempt", "failure", "diagnosis"), "M3": ("knowledge",), "M4": ("procedure",)}
EPISTEMIC_STATUS = ("observation", "hypothesis", "explanation")
# problem types the analysis agent must tell apart in its diagnoses
PROBLEM_TYPES = ("technical_failure", "data_quality", "workflow", "scientific_contradiction")


def new_id(prefix: str) -> str:
    return f"{prefix.lower()}:{uuid.uuid4().hex[:12]}"


def now() -> float:
    return time.time()


def molecule_node(smiles: str) -> str:
    return f"mol:{DOMAIN.canonical(smiles)}"


def motif_node(name: str) -> str:
    return f"motif:{name}"


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
    key: str = ""                        # exact-retrieval key: canonical molecule, agent, rule key...
    author: str = ""
    campaign_id: str | None = None
    prov_refs: list[dict] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)  # what describes the entry (see the constructors below)
    id: str = ""
    created_at: float = 0.0
    history: list[dict] = field(default_factory=list)   # merges of duplicates

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES or self.kind not in KINDS[self.category]:
            raise ValueError(f"unknown entry type {self.category}/{self.kind}")
        self.id = self.id or new_id(self.category)
        self.created_at = self.created_at or now()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Entry":
        return cls(**d)


def _status(s: str) -> str:
    if s not in EPISTEMIC_STATUS:
        raise ValueError(f"epistemic status must be one of {EPISTEMIC_STATUS}")
    return s


# --------------------------------------------------------------------------------------------- constructors
def result(entity: str, quantity: str, value: float, unit: str, conditions: dict, author: str, prov: ProvRef,
           campaign_id: str | None = None, spelling: str = "") -> Entry:
    """M1: a measurement, its conditions and the Flowcept task that produced it."""
    return Entry("M1", "result", f"{quantity} of {entity} = {value:.6g} {unit}", key=entity, author=author,
                 campaign_id=campaign_id, prov_refs=[prov.to_dict()],
                 data={"entity": entity, "quantity": quantity, "value": value, "unit": unit,
                       "conditions": conditions, "spelling": spelling or entity})


def attempt(text: str, author: str, iteration: int, proposed: list[str], proposed_at: float | None,
            reused: list[str], improved: bool | None, best_before: float | None, best_after: float | None,
            plan_excerpt: str = "", prov_refs: list[dict] | None = None, campaign_id: str | None = None) -> Entry:
    """M2: one iteration of an agent. Its results and failures point to it (PART_OF edges)."""
    return Entry("M2", "attempt", text, key=author, author=author, campaign_id=campaign_id,
                 prov_refs=prov_refs or [],
                 data={"iteration": iteration, "proposed": proposed, "proposed_at": proposed_at, "reused": reused,
                       "improved": improved, "best_before": best_before, "best_after": best_after,
                       "plan_excerpt": plan_excerpt})


def failure(entity: str, failure_class: str, error: str, author: str, iteration: int, spelling: str,
            prov_refs: list[dict] | None = None, campaign_id: str | None = None) -> Entry:
    """M2: a computation that failed."""
    return Entry("M2", "failure", f"Simulation of {entity} failed ({failure_class})", key=entity, author=author,
                 campaign_id=campaign_id, prov_refs=prov_refs or [],
                 data={"failure_class": failure_class, "error": error[:500], "iteration": iteration,
                       "spelling": spelling})


def diagnosis(rule: str, key: str, text: str, problem_type: str, epistemic_status: str, author: str,
              context: dict | None = None, prov_refs: list[dict] | None = None,
              campaign_id: str | None = None) -> Entry:
    """M2: the analysis agent's reading of a problem; what it is based on are DIAGNOSED_FROM edges."""
    if problem_type not in PROBLEM_TYPES:
        raise ValueError(f"problem type must be one of {PROBLEM_TYPES}")
    return Entry("M2", "diagnosis", text, key=key, author=author, campaign_id=campaign_id,
                 prov_refs=prov_refs or [],
                 data={"rule": rule, "problem_type": problem_type, "epistemic_status": _status(epistemic_status),
                       "context": context or {}})


def knowledge(statement: str, epistemic_status: str, author: str, scope: str = "", uncertainty: str = "",
              prov_refs: list[dict] | None = None, campaign_id: str | None = None) -> Entry:
    """M3: a scientific statement; its evidence are SUPPORTED_BY / CONTRADICTED_BY edges."""
    return Entry("M3", "knowledge", statement, author=author, campaign_id=campaign_id, prov_refs=prov_refs or [],
                 data={"scope": scope, "epistemic_status": _status(epistemic_status), "uncertainty": uncertainty})


def procedure(text: str, trigger: str, preconditions: str, actions: str, expected_outcome: str, limitations: str,
              author: str, key: str, failure_class: str, motif: str | None, prov_refs: list[dict] | None = None,
              campaign_id: str | None = None) -> Entry:
    """M4: reusable guidance for one failure class, on molecules with one motif (None: any molecule). The
    experiences it comes from are DERIVED_FROM edges, the knowledge that justifies it JUSTIFIED_BY edges, and its
    applications the APPLIED_IN edges of the guidance that applied it."""
    return Entry("M4", "procedure", text, key=key, author=author, campaign_id=campaign_id, prov_refs=prov_refs or [],
                 data={"trigger": trigger, "preconditions": preconditions, "actions": actions,
                       "expected_outcome": expected_outcome, "limitations": limitations,
                       "failure_class": failure_class, "motif": motif})


@dataclass
class Guidance:
    """A message of the analysis agent to scientific agents. It is a node of the graph: what it applies or is based
    on (APPLIES) and where it was applied (APPLIED_IN) are edges; `delivered` is its delivery log."""
    message: str
    problem_type: str | None
    epistemic_status: str
    target: str = "*"                    # an agent name, or "*" for every agent
    check: dict = field(default_factory=dict)   # machine-checkable part: {"avoid_keys": [...], "failure_class": ...}
    id: str = ""
    created_at: float = 0.0
    delivered: list[dict] = field(default_factory=list)     # {agent, node, t}

    def __post_init__(self) -> None:
        _status(self.epistemic_status)
        self.id = self.id or new_id("G")
        self.created_at = self.created_at or now()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Guidance":
        return cls(**d)
