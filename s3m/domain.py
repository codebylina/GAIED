"""
Domain description: the only part of S3M that knows what an experiment is.

S3M itself never mentions molecules or energies. A workflow plugs in by describing:
  - the entity it experiments on (canonical key, so two spellings of the same thing match),
  - the objective (name and direction),
  - how to classify a failure (scientific: the experiment itself is impossible / meaningless, do not
    retry; technical: crash, timeout, infrastructure, may be retried),
  - optionally, numeric features of an entity (used by lessons such as "entities with feature F
    score higher"),
  - how to read one observation from a Flowcept task record (to rebuild the memory from provenance).
"""
from __future__ import annotations

from typing import Any

SCIENTIFIC = "scientific"
TECHNICAL = "technical"


class Domain:
    name = "generic"
    objective = "value"          # name of the measured quantity
    direction = "max"            # "max" or "min": which values are better

    def canonical(self, raw: str) -> str | None:
        """Canonical key of an entity, or None if the proposal is not a valid entity."""
        raw = str(raw).strip()
        return raw or None

    def features(self, entity: str) -> dict[str, float]:
        """Numeric features of an entity (empty when the domain defines none)."""
        return {}

    def feature_doc(self) -> str:
        """One line per feature, shown to the LLM that proposes lessons."""
        return ""

    def classify_failure(self, error: str) -> str:
        """SCIENTIFIC or TECHNICAL. Default: technical (safe: the experiment may be retried)."""
        return TECHNICAL

    def better(self, a: float, b: float) -> bool:
        return a > b if self.direction == "max" else a < b

    # -------------------------------------------------------------- reading Flowcept provenance
    activity: str | None = None      # Flowcept activity_id of the experiment task
    input_key = "entity"             # key of the entity in the task's "used"
    output_key = "value"             # key of the objective in the task's "generated"
    params_key: str | None = None    # key of the experiment parameters in "used" (optional)
    method = "experiment"            # name of the experimental method
    default_params: dict | None = None   # parameters assumed when a task record does not carry them

    def observation_from_task(self, task: dict) -> dict | None:
        """Fields of an observation read from a Flowcept task record, or None if unrelated."""
        if self.activity and task.get("activity_id") != self.activity:
            return None
        used = task.get("used") or {}
        if not isinstance(used, dict) or self.input_key not in used:
            return None
        gen = task.get("generated") or {}
        value = gen.get(self.output_key) if isinstance(gen, dict) else None
        ok = str(task.get("status", "")).upper() in ("FINISHED", "SUCCESS", "OK") and value is not None
        error = None if ok else str(task.get("stderr") or "unknown error")
        meta = task.get("custom_metadata") or {}
        return {
            "raw": str(used[self.input_key]),
            "method": self.method,
            "params": (used.get(self.params_key) if self.params_key else None) or self.default_params or {},
            "value": float(value) if ok else None,
            "error": error,
            "agent": meta.get("agent") if isinstance(meta, dict) else None,
            "started_at": task.get("started_at"),
            "ended_at": task.get("ended_at"),
            "provenance": provenance_ids(task),
        }


def provenance_ids(task: dict) -> dict[str, Any]:
    """The Flowcept identifiers that link a memory record to the provenance graph."""
    return {k: task.get(k) for k in ("task_id", "workflow_id", "campaign_id", "parent_task_id")
            if task.get(k)}
