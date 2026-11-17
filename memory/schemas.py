"""
Typed records of the four memory layers, with validation.

  M1 ResultRecord      scientific results: measurements, valid negative results, parameters, conditions
  M2 ExperienceRecord  incidents, anomalies, recurring failures, inefficient exploration, diagnostics,
                       attempted actions, recovery episodes
  M3 KnowledgeRecord   observations, hypotheses, claims, findings, rules, constraints, negative findings,
                       uncertainties, with evidence, scope, applicability and epistemic status
  M4 GuidanceRecord    strategies, recommendations, instructions, recovery strategies, with applicability,
                       triggers, actions, limitations, evidence, targets and validation status

Separation rules enforced here:
  - a technical failure is never an M1 record (it is an M2 incident); M1 negative results are
    scientifically meaningful outcomes (e.g. an impossible structure), not crashes;
  - a diagnostic hypothesis lives in M2; established scientific knowledge lives in M3;
  - a recovery episode is M2; a reusable recovery strategy is M4.
Execution state (LangGraph state, checkpoints) is not memory and has no record type here.

Every record keeps references to its Flowcept provenance (task ids...) instead of copying traces.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from typing import Any, ClassVar


class ValidationError(ValueError):
    """A record does not satisfy its schema."""


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def prov_ref(task_id: str | None, workflow_id: str | None = None, campaign_id: str | None = None,
             activity_id: str | None = None) -> dict:
    """Reference to a Flowcept task (the provenance is never copied, only referenced)."""
    ref = {"system": "flowcept", "task_id": task_id, "workflow_id": workflow_id,
           "campaign_id": campaign_id, "activity_id": activity_id}
    return {k: v for k, v in ref.items() if v}


@dataclass
class Record:
    layer: ClassVar[str] = ""
    KINDS: ClassVar[tuple[str, ...]] = ()
    STATUSES: ClassVar[tuple[str, ...]] = ()
    INDEXED: ClassVar[tuple[str, ...]] = ("kind", "status")

    kind: str = ""
    status: str = ""
    id: str = ""
    namespace: str = ""
    created_by: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    version: int = 1
    key: str = ""                                            # deduplication key
    provenance: list[dict] = field(default_factory=list)     # Flowcept task references
    memory_refs: list[str] = field(default_factory=list)     # "M1:<id>", "M3:<id>"...
    tags: list[str] = field(default_factory=list)
    expires_at: float | None = None

    def __post_init__(self) -> None:
        now = time.time()
        self.id = self.id or new_id(self.layer.lower())
        self.created_at = self.created_at or now
        self.updated_at = self.updated_at or self.created_at
        self.status = self.status or self.STATUSES[0]

    # ------------------------------------------------------------------ validation
    def validate(self) -> "Record":
        if self.kind not in self.KINDS:
            raise ValidationError(f"{self.layer}: kind '{self.kind}' not in {self.KINDS}")
        if self.status not in self.STATUSES + ("archived",):
            raise ValidationError(f"{self.layer}: status '{self.status}' not in {self.STATUSES}")
        if not self.namespace:
            raise ValidationError(f"{self.layer}: namespace is required")
        for ref in self.provenance:
            if not isinstance(ref, dict) or not ref.get("task_id"):
                raise ValidationError(f"{self.layer}: provenance references need a task_id: {ref}")
        for ref in self.memory_refs:
            if not isinstance(ref, str) or ref[:3] not in ("M1:", "M2:", "M3:", "M4:"):
                raise ValidationError(f"{self.layer}: memory_refs must look like 'M1:<id>': {ref}")
        self._validate()
        return self

    def _validate(self) -> None:
        pass

    # ------------------------------------------------------------------ (de)serialization
    def to_dict(self) -> dict:
        d = asdict(self)
        d["layer"] = self.layer
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Record":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def index_values(self) -> dict[str, list[str]]:
        """Field values to index for metadata filtering."""
        out = {}
        for name in self.INDEXED:
            v = getattr(self, name, None)
            if v is None or v == "":
                continue
            out[name] = [str(x) for x in v] if isinstance(v, (list, tuple, set)) else [str(v)]
        return out


# ---------------------------------------------------------------------- M1
@dataclass
class ResultRecord(Record):
    layer: ClassVar[str] = "M1"
    KINDS: ClassVar[tuple[str, ...]] = ("measurement",)
    STATUSES: ClassVar[tuple[str, ...]] = ("valid", "disputed", "superseded")
    INDEXED: ClassVar[tuple[str, ...]] = ("kind", "status", "entity", "condition", "outcome", "agent")

    kind: str = "measurement"
    entity: str = ""             # canonical key of what was measured (e.g. canonical SMILES)
    raw_input: str = ""          # as proposed by the agent
    method: str = ""             # e.g. "xtb"
    params: dict = field(default_factory=dict)
    condition: str = ""          # hash of method + params: only results of one condition are comparable
    objective: str = ""          # e.g. "ionization_energy"
    unit: str = ""
    outcome: str = "success"     # "success" (value measured) or "negative" (valid negative scientific result)
    value: float | None = None
    detail: str = ""             # for a negative result: why the experiment is scientifically impossible
    agent: str = ""
    duration_s: float | None = None

    def _validate(self) -> None:
        if not self.entity or not self.method or not self.condition:
            raise ValidationError("M1: entity, method and condition are required")
        if self.outcome not in ("success", "negative"):
            raise ValidationError("M1: outcome must be 'success' or 'negative' (technical failures go to M2)")
        if self.outcome == "success" and (self.value is None or self.value != self.value):
            raise ValidationError("M1: a successful result needs a numeric value")
        if self.outcome == "negative" and not self.detail:
            raise ValidationError("M1: a negative result needs a scientific detail")


# ---------------------------------------------------------------------- M2
@dataclass
class ExperienceRecord(Record):
    layer: ClassVar[str] = "M2"
    KINDS: ClassVar[tuple[str, ...]] = (
        "incident", "anomaly", "recurring_failure", "redundant_computation", "inefficient_exploration",
        "low_diversity", "unused_knowledge", "knowledge_conflict", "ineffective_guidance",
        "diagnostic_hypothesis", "confirmed_cause", "attempted_action", "recovery_episode")
    STATUSES: ClassVar[tuple[str, ...]] = ("open", "investigating", "confirmed", "resolved", "dismissed")
    INDEXED: ClassVar[tuple[str, ...]] = ("kind", "status", "signature", "agents", "entities", "severity")

    summary: str = ""
    signature: str = ""                                          # what makes two episodes "the same"
    observed_facts: list[str] = field(default_factory=list)      # what was observed (with evidence)
    candidate_explanations: list[str] = field(default_factory=list)   # hypotheses, NOT knowledge
    confirmed_cause: str | None = None                           # only when established by a rule or a human
    severity: str = "low"
    agents: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    error_class: str | None = None                               # transient / permanent / scientific
    occurrences: int = 1
    first_seen: float = 0.0
    last_seen: float = 0.0
    actions_taken: list[dict] = field(default_factory=list)      # attempted actions and their outcome
    metrics: dict = field(default_factory=dict)

    def _validate(self) -> None:
        if not self.summary or not self.signature:
            raise ValidationError("M2: summary and signature are required")
        if self.severity not in ("low", "medium", "high"):
            raise ValidationError("M2: severity must be low, medium or high")
        if self.confirmed_cause and self.status not in ("confirmed", "resolved"):
            raise ValidationError("M2: a confirmed cause requires status confirmed or resolved")


# ---------------------------------------------------------------------- M3
@dataclass
class KnowledgeRecord(Record):
    layer: ClassVar[str] = "M3"
    KINDS: ClassVar[tuple[str, ...]] = ("observation", "hypothesis", "claim", "finding", "rule", "constraint",
                                        "negative_finding", "uncertainty")
    # epistemic status: proposed -> supported -> validated; contested / refuted / stale on new evidence
    STATUSES: ClassVar[tuple[str, ...]] = ("proposed", "supported", "validated", "contested", "refuted", "stale")
    INDEXED: ClassVar[tuple[str, ...]] = ("kind", "status", "condition", "source")

    statement: str = ""
    claim: dict | None = None            # machine-checkable form, e.g. {"type": "feature_effect", ...}
    condition: str = ""                  # experimental condition the knowledge is about
    scope: dict = field(default_factory=dict)          # method, params, region (feature constraints)
    applicability: str = ""              # when the knowledge applies, in words
    supporting: list[str] = field(default_factory=list)      # M1 ids
    contradicting: list[str] = field(default_factory=list)   # M1 ids
    confidence: float = 0.0
    source: str = "agent"                # agent / analysis / human
    evaluation: dict = field(default_factory=dict)            # last evaluation (rule, numbers, reason)

    def _validate(self) -> None:
        if not self.statement:
            raise ValidationError("M3: statement is required")
        if self.source not in ("agent", "analysis", "human"):
            raise ValidationError("M3: source must be agent, analysis or human")
        if self.status == "validated" and not self.supporting:
            raise ValidationError("M3: validated knowledge needs supporting evidence")


# ---------------------------------------------------------------------- M4
ACTION_TYPES = ("avoid_entities", "avoid_region", "prefer_region", "diversify", "reuse_results",
                "use_knowledge", "retry_policy", "reduce_parallelism", "free_text")


@dataclass
class GuidanceRecord(Record):
    layer: ClassVar[str] = "M4"
    KINDS: ClassVar[tuple[str, ...]] = ("strategy", "recommendation", "instruction", "recovery_strategy")
    # validation status: proposed -> active -> validated (shown effective) / ineffective -> retired
    STATUSES: ClassVar[tuple[str, ...]] = ("proposed", "active", "validated", "ineffective", "retired")
    INDEXED: ClassVar[tuple[str, ...]] = ("kind", "status", "targets", "condition")

    text: str = ""
    condition: str = ""                  # experimental condition it applies to ("" = any)
    applicability: dict = field(default_factory=dict)    # {"nodes": [...], "when": "..."}
    trigger: dict = field(default_factory=dict)          # what triggered it (finding id, rule, event)
    actions: list[dict] = field(default_factory=list)    # structured, checkable actions
    limitations: str = ""
    targets: list[str] = field(default_factory=lambda: ["all"])   # agent ids, or "all" (campaign)
    priority: int = 1
    source: str = "analysis"
    baseline: dict = field(default_factory=dict)         # metrics of the targets when it was activated
    effectiveness: dict = field(default_factory=dict)    # metrics after, and the verdict

    def _validate(self) -> None:
        if not self.text:
            raise ValidationError("M4: text is required")
        if not self.targets:
            raise ValidationError("M4: targets are required ('all' for the whole campaign)")
        for a in self.actions:
            if not isinstance(a, dict) or a.get("type") not in ACTION_TYPES:
                raise ValidationError(f"M4: action type must be one of {ACTION_TYPES}: {a}")


LAYERS: dict[str, type[Record]] = {"M1": ResultRecord, "M2": ExperienceRecord, "M3": KnowledgeRecord,
                                   "M4": GuidanceRecord}


def record_from_dict(d: dict) -> Record:
    return LAYERS[d["layer"]].from_dict(d)


def as_jsonable(value: Any) -> Any:
    if isinstance(value, Record):
        return value.to_dict()
    return value
