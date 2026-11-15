"""
Domain adapter: everything the memory needs to know about a domain, and nothing more.

The core never mentions molecules, energies or any tool. A domain provides:
  - how to identify an entity (canonical key; None when the proposal is not a valid entity),
  - how to read an observation from a provenance record (W3C PROV-like task: used / generated),
  - descriptors of an entity (the vocabulary in which claims and their scopes are written),
  - the objective and the constraints of the Goal,
  - optionally, a source of prior domain knowledge for the judge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Objective:
    name: str                      # e.g. "ionization_energy"
    direction: str = "max"         # "max" or "min"
    unit: str = ""

    def better(self, a: float, b: float) -> bool:
        return a > b if self.direction == "max" else a < b


@dataclass
class Goal:
    objective: Objective
    text: str                                   # the Goal as given to the agents
    constraints: list[str] = field(default_factory=list)   # human-readable constraints


class DomainAdapter:
    entity_noun: str = "entity"
    goal: Goal

    def canonical(self, raw: str) -> str | None:
        """Canonical key of a proposed entity, or None if it is not a valid entity."""
        raise NotImplementedError

    def observation_from_record(self, record: dict) -> dict | None:
        """Read one observation from a provenance task record. Returns None for unrelated records.

        Generic PROV-style reading: 'used' holds the inputs, 'generated' the outputs. Subclasses
        say which input names the entity and which output is the objective."""
        raise NotImplementedError

    def descriptors(self, entity: str) -> dict[str, Any]:
        """Features of an entity, used to write and evaluate claim scopes."""
        return {}

    def descriptor_doc(self) -> str:
        """Description of the descriptor columns, given to the judge."""
        return ""

    def violations(self, entity: str) -> list[str]:
        """Goal constraints broken by an entity (empty list if none)."""
        return []

    def domain_knowledge(self, statement: str) -> str | None:
        """Optional prior knowledge relevant to a statement (literature, reference values)."""
        return None


class ProvenanceTaskAdapter(DomainAdapter):
    """Reads Flowcept / W3C PROV task records: entity from used[input_key], outcome from
    generated[output_key], status from the task status, agent from custom_metadata."""

    input_key: str = "entity"
    output_key: str = "value"
    activity: str | None = None     # only records of this activity_id, if set

    def observation_from_record(self, record: dict) -> dict | None:
        if self.activity and record.get("activity_id") != self.activity:
            return None
        used = record.get("used") or {}
        if self.input_key not in used:
            return None
        entity = self.canonical(str(used[self.input_key]))
        gen = record.get("generated") or {}
        value = gen.get(self.output_key)
        status = str(record.get("status", "")).upper()
        ok = status in ("FINISHED", "OK", "SUCCESS", "") and value is not None
        try:
            value = float(value) if ok else None
        except (TypeError, ValueError):
            ok, value = False, None
        if value is not None and value != value:  # NaN
            ok, value = False, None
        meta = record.get("custom_metadata") or {}
        return {
            "id": record.get("task_id"),
            "entity": entity if entity is not None else str(used[self.input_key]),
            "valid_entity": entity is not None,
            "raw": str(used[self.input_key]),
            "value": value,
            "status": "ok" if ok else "failed",
            "error": record.get("stderr") if not ok else None,
            "agent": meta.get("agent") or record.get("agent_id"),
            "t": float(record.get("ended_at") or record.get("started_at") or 0.0),
            "started_at": float(record.get("started_at") or 0.0),
        }
