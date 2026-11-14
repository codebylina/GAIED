"""
Measurement only (not memory): records what the agents propose and compute, in the same
format as the EVIDENT telemetry (Redis keys ev:{campaign}:telemetry and ev:{campaign}:facts),
so that experiments/analyze.py of the evident-memory branch analyses FALDA campaigns with
exactly the same metrics. Nothing recorded here is ever shown to the agents.
"""
from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:  # pragma: no cover
    Chem = None


def canonical(smiles: str) -> str:
    if Chem is None:
        return smiles
    mol = Chem.MolFromSmiles(smiles or "")
    return Chem.MolToSmiles(mol) if mol is not None else (smiles or "").strip()


def mol_info(smiles: str) -> tuple[bool, int | None, list[str]]:
    if Chem is None:
        return True, None, []
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return False, None, []
    return True, mol.GetNumHeavyAtoms(), sorted({a.GetSymbol() for a in mol.GetAtoms()})


class Recorder:
    def __init__(self, client, campaign: str, agent: str, constraints: dict | None = None):
        self.r, self.campaign, self.agent = client, campaign, agent
        self.constraints = constraints or {}
        self.iteration = 0
        self.activity = ""

    @classmethod
    def from_env(cls, agent: str, client=None) -> "Recorder | None":
        campaign = os.environ.get("MEMORY_CAMPAIGN_ID") or os.environ.get("FLOWCEPT_CAMPAIGN_ID")
        if not campaign:
            return None
        try:
            if client is None:
                import redis
                client = redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"), decode_responses=True)
            client.ping()
        except Exception as exc:
            logger.warning("recorder disabled (Redis unreachable): %s", exc)
            return None
        rec = cls(client, campaign, agent)
        raw = client.get(rec.k("goal"))
        rec.constraints = (json.loads(raw) if raw else {}).get("constraints", {})
        if agent not in ("orchestrator", "cli"):
            client.sadd(rec.k("agents"), agent)
        return rec

    def k(self, *parts) -> str:
        return ":".join(["ev", self.campaign, *map(str, parts)])

    def emit(self, event: str, **data) -> None:
        try:
            self.r.xadd(self.k("telemetry"), {"e": event, "agent": self.agent, "it": self.iteration,
                                              "act": self.activity, "t": f"{time.time():.6f}",
                                              "d": json.dumps(data, default=str)})
        except Exception as exc:
            logger.debug("recorder emit failed: %s", exc)

    # ------------------------------------------------------------------ campaign
    def set_goal(self, goal: dict) -> None:
        self.r.set(self.k("goal"), json.dumps(goal))
        self.constraints = goal.get("constraints", {})

    def set_meta(self, **meta) -> None:
        self.r.hset(self.k("meta"), mapping={k: json.dumps(v) for k, v in meta.items()})

    # ------------------------------------------------------------------ events
    def violations(self, smiles: str) -> list[str]:
        valid, heavy, elements = mol_info(smiles)
        if not valid:
            return []
        out = []
        mx = self.constraints.get("max_heavy_atoms")
        if mx and heavy is not None and heavy > int(mx):
            out.append(f"too_large({heavy}>{mx}_heavy_atoms)")
        allowed = self.constraints.get("allowed_elements")
        if allowed and any(e not in allowed for e in elements):
            out.append("forbidden_elements(" + ",".join(e for e in elements if e not in allowed) + ")")
        return out

    def proposals(self, smiles_list: list[str], own_previous: list[str], retries: int = 0) -> None:
        try:
            self._proposals(smiles_list, own_previous, retries)
        except Exception as exc:  # measurement must never break an agent
            logger.warning("recorder proposals failed: %s", exc)

    def _proposals(self, smiles_list, own_previous, retries) -> None:
        own = {canonical(s) for s in own_previous}
        known = set(self.r.hkeys(self.k("facts:by_smiles")))
        items = []
        for s in smiles_list:
            can = canonical(s)
            valid = mol_info(can)[0]
            fact = self.r.hget(self.k("facts:by_smiles"), can) if can in known else None
            status = json.loads(fact)["status"] if fact else None
            items.append({"smiles": can, "raw": s, "valid": valid, "features": [],
                          "violations": self.violations(can), "own_repeat": can in own,
                          "team_known": "failed" if status == "error" else "computed" if fact else "new"})
        self.emit("proposals", items=items, retries=retries)

    def fact(self, smiles: str, energy: float | None, error: str | None, task_id: str | None,
             duration: float) -> None:
        try:
            self._fact(smiles, None if energy is None else float(energy), error, task_id, duration)
        except Exception as exc:  # measurement must never break an agent
            logger.warning("recorder fact failed: %s", exc)

    def _fact(self, smiles, energy, error, task_id, duration) -> None:
        can = canonical(smiles)
        valid, heavy, _ = mol_info(can)
        f = {"smiles": can, "energy": energy, "status": "error" if error else "ok", "error": error,
             "agent": self.agent, "task_id": task_id, "iteration": self.iteration, "t": time.time(),
             "duration": duration, "features": [], "heavy_atoms": heavy}
        payload = json.dumps(f)
        pipe = self.r.pipeline()
        pipe.xadd(self.k("facts"), {"fact": payload})
        pipe.hsetnx(self.k("facts:by_smiles"), can, payload)
        if not error and energy is not None:
            pipe.zadd(self.k("ranking"), {can: energy}, nx=True)
        pipe.hincrby(self.k("stats"), "computed_error" if error else "computed", 1)
        pipe.execute()
        self.emit("compute", smiles=can, outcome="computed_error" if error else "computed",
                  task_id=task_id, energy=energy, duration=round(duration, 3), error=error)

    def tool_call_failed(self, attempts: int, responses: list[str]) -> None:
        self.emit("tool_call_failed", attempts=attempts, responses=[(r or "")[:600] for r in responses])

    def conclusion(self, text: str) -> None:
        self.emit("conclusion", claims=[], text=(text or "")[:2000], evidence=[], liar=False)

    def briefing(self, node: str, chars: int, ms: float | None, text: str = "") -> None:
        """What the memory put in a prompt. The text is kept so that the analysis can find
        the structure-property claims it contains (keyword parser of the EVIDENT branch)."""
        self.emit("briefing", node=node, chars=chars, shown_claims=[], lesson_claims=[],
                  ms=round(ms or 0.0, 3), text=(text or "")[:4000])

    def node(self, node: str, duration: float) -> None:
        self.emit("node", node=node, duration=round(duration, 4), interrupted=False)

    def end_iteration(self, own_best: float | None, no_improve: int, iteration: int) -> None:
        try:
            top = self.r.zrevrange(self.k("ranking"), 0, 0, withscores=True)
        except Exception:
            top = []
        self.emit("iteration_end", own_best=own_best, no_improve=no_improve,
                  team_best=top[0][1] if top else None)
        self.iteration = iteration

    # ------------------------------------------------------------------ export
    def export(self) -> dict:
        tel = []
        for entry_id, f in self.r.xrange(self.k("telemetry")):
            tel.append({"id": entry_id, "e": f["e"], "agent": f["agent"], "it": int(f["it"]),
                        "t": float(f["t"]), **json.loads(f["d"])})
        meta = {k: json.loads(v) for k, v in self.r.hgetall(self.k("meta")).items()}
        goal = self.r.get(self.k("goal"))
        return {"campaign": self.campaign, "mode": meta.get("mode", "falda"), "meta": meta,
                "goal": json.loads(goal) if goal else None,
                "agents": sorted(self.r.smembers(self.k("agents"))), "liar": None,
                "stats": self.r.hgetall(self.k("stats")),
                "facts_log": [json.loads(f["fact"]) for _, f in self.r.xrange(self.k("facts"))],
                "claims": {}, "claims_log": [], "lessons": [], "telemetry": tel}
