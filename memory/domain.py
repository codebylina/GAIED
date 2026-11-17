"""
Domain adapter: the only part of the memory that knows what an experiment is. The memory, the
coordination and the analysis service are generic; a workflow plugs in by describing its domain.

Failure classes (used by the retry policy and by the layer a failure is stored in):
  SCIENTIFIC  the experiment itself is impossible or meaningless (invalid structure...): a valid negative
              scientific result, stored in M1, never retried
  TRANSIENT   technical problem that may disappear (timeout, lost worker, connection): retried with
              backoff, stored as an M2 incident, never blocks a future computation
  PERMANENT   technical problem unlikely to disappear by itself (missing binary, bad configuration):
              not retried immediately, stored as an M2 incident; a later attempt is allowed after a cooldown
"""
from __future__ import annotations

import hashlib
import json

SCIENTIFIC, TRANSIENT, PERMANENT = "scientific", "transient", "permanent"


def condition_id(method: str, params: dict | None) -> str:
    """Two experiments are comparable only with the same method and parameters."""
    blob = json.dumps({"method": method, "params": params or {}}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


class Domain:
    name = "entity"
    objective = "value"
    unit = ""
    direction = "max"                 # which values are better: "max" or "min"

    def canonical(self, raw: str) -> str | None:
        """Canonical key of an entity, or None when the proposal is not a valid entity."""
        raw = str(raw).strip()
        return raw or None

    def features(self, entity: str) -> dict[str, float]:
        """Numeric descriptors (vocabulary of knowledge scopes and exploration analysis)."""
        return {}

    def feature_doc(self) -> str:
        return ""

    def similarity(self, a: str, b: str) -> float:
        """Similarity in [0, 1] between two entities (exploration diversity). Default: identity."""
        return 1.0 if a == b else 0.0

    def classify_failure(self, error: str) -> str:
        """SCIENTIFIC, TRANSIENT or PERMANENT. Default: TRANSIENT (safe: it can be retried)."""
        return TRANSIENT

    def better(self, a: float, b: float) -> bool:
        return a > b if self.direction == "max" else a < b

    # -------------------------------------------------------------- reading Flowcept task records
    activity: str | None = None      # activity_id of the experiment task
    input_key = "entity"             # key of the entity in the task's "used"
    output_key = "value"             # key of the objective in the task's "generated"
    params_key: str | None = None    # key of the experiment parameters in "used"
    method = "experiment"
    default_params: dict | None = None

    def is_experiment_task(self, task: dict) -> bool:
        if self.activity and task.get("activity_id") != self.activity:
            return False
        used = task.get("used")
        return isinstance(used, dict) and self.input_key in used

    def experiment_of(self, task: dict) -> dict | None:
        """Entity, condition and outcome of a Flowcept experiment task (None if unrelated)."""
        if not self.is_experiment_task(task):
            return None
        used = task["used"]
        params = (used.get(self.params_key) if self.params_key else None) or self.default_params or {}
        gen = task.get("generated") if isinstance(task.get("generated"), dict) else {}
        status = str(task.get("status", "")).upper()
        value = gen.get(self.output_key)
        raw = str(used[self.input_key])
        return {"task_id": task.get("task_id"), "raw": raw, "entity": self.canonical(raw) or f"invalid:{raw}",
                "condition": condition_id(self.method, params), "ok": status == "FINISHED" and value is not None,
                "value": value, "error": task.get("stderr"),
                "agent": (task.get("custom_metadata") or {}).get("agent"),
                "started_at": task.get("started_at"), "ended_at": task.get("ended_at"),
                "workflow_id": task.get("workflow_id"), "campaign_id": task.get("campaign_id")}
