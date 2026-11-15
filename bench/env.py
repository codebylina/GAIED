"""
Synthetic experimental domain with a known ground truth.

An entity is a combination of options in a few slots, written "A2-B0-C3-D1" (slot letter + option).
Its outcome is a hidden function: per-option main effects, pairwise interactions, Gaussian noise,
and a set of forbidden option pairs that make the experiment fail. Because the noise-free outcome of
every entity is known, the evaluation can tell whether a claim the agents wrote is true, and how far
the team is from the optimum.

A "deceptive" instance plants an interaction such that one option looks beneficial in the region
where the agents start but is harmful elsewhere: a hasty generalization from early data is false,
while the correctly scoped statement is true.

All generator parameters come from the experiment configuration; nothing here is tuned per condition.
"""
from __future__ import annotations

import itertools
import random
import string
import time
import uuid
from dataclasses import dataclass, field

from memlab.adapter import DomainAdapter, Goal, Objective


@dataclass
class EnvSpec:
    n_slots: int
    n_options: int
    main_scale: float          # spread of main effects
    interaction_scale: float   # spread of pairwise interactions
    n_interactions: int
    noise_sd: float            # measurement noise
    n_forbidden: int           # forbidden option pairs (experiment fails)
    deceptive: bool = False
    deception_strength: float = 0.0   # size of the planted effect, in units of main_scale
    duration_s: float = 0.0    # simulated duration of one experiment (creates overlap between agents)
    seed: int = 0
    extra: dict = field(default_factory=dict)


class SyntheticEnv:
    def __init__(self, spec: EnvSpec) -> None:
        self.spec = spec
        rng = random.Random(spec.seed)
        self.rng_noise = random.Random(spec.seed + 10_000)
        self.slots = list(string.ascii_uppercase[:spec.n_slots])
        opts = range(spec.n_options)
        self.main = {(s, o): rng.gauss(0, spec.main_scale) for s in self.slots for o in opts}
        pairs = list(itertools.combinations(self.slots, 2))
        rng.shuffle(pairs)
        self.inter = {}
        for s, t in pairs[:spec.n_interactions]:
            for o in opts:
                for p in opts:
                    self.inter[(s, o, t, p)] = rng.gauss(0, spec.interaction_scale)
        all_pairs = [(s, o, t, p) for s, t in itertools.combinations(self.slots, 2) for o in opts for p in opts]
        self.forbidden = set(rng.sample(all_pairs, spec.n_forbidden)) if spec.n_forbidden else set()
        # region where the agents start: slot 0 at option 0
        self.region = (self.slots[0], 0)
        self.decoy = None
        if spec.deceptive:
            # decoy option: helps inside the start region, hurts outside it (planted interaction)
            s_dec = self.slots[-1]
            o_dec = spec.n_options - 1
            self.decoy = (s_dec, o_dec)
            k = spec.deception_strength * spec.main_scale
            for o in opts:
                sign = 1.0 if o == self.region[1] else -1.0
                self.inter[(self.region[0], o, s_dec, o_dec)] = sign * k
        self._pop = None

    # ------------------------------------------------------------------ entities
    def key(self, combo: tuple[int, ...]) -> str:
        return "-".join(f"{s}{o}" for s, o in zip(self.slots, combo))

    def parse(self, raw: str) -> tuple[int, ...] | None:
        parts = [p.strip().upper() for p in str(raw).replace(" ", "").split("-") if p.strip()]
        found = {}
        for p in parts:
            if len(p) < 2 or p[0] not in self.slots or not p[1:].isdigit():
                return None
            o = int(p[1:])
            if o >= self.spec.n_options or p[0] in found:
                return None
            found[p[0]] = o
        if set(found) != set(self.slots):
            return None
        return tuple(found[s] for s in self.slots)

    def fails(self, combo) -> bool:
        d = dict(zip(self.slots, combo))
        return any(d[s] == o and d[t] == p for (s, o, t, p) in self.forbidden)

    def true_value(self, combo) -> float | None:
        if self.fails(combo):
            return None
        d = dict(zip(self.slots, combo))
        v = sum(self.main[(s, d[s])] for s in self.slots)
        v += sum(w for (s, o, t, p), w in self.inter.items() if d[s] == o and d[t] == p)
        return v

    def population(self) -> list[dict]:
        """Every entity with its noise-free outcome (the oracle table)."""
        if self._pop is None:
            pop = []
            for combo in itertools.product(range(self.spec.n_options), repeat=self.spec.n_slots):
                v = self.true_value(combo)
                pop.append({"id": self.key(combo), "entity": self.key(combo), "value": v,
                            "status": "ok" if v is not None else "failed",
                            **{f"slot_{s}": o for s, o in zip(self.slots, combo)}})
            self._pop = pop
        return self._pop

    def optimum(self) -> float:
        return max(r["value"] for r in self.population() if r["value"] is not None)

    def seeds(self, n_agents: int) -> list[str]:
        """Starting entities, all inside the start region."""
        rng = random.Random(self.spec.seed + 20_000)
        out = []
        while len(out) < n_agents:
            combo = [rng.randrange(self.spec.n_options) for _ in self.slots]
            combo[0] = self.region[1]
            if self.decoy:
                combo[self.slots.index(self.decoy[0])] = 0
            if not self.fails(combo):
                out.append(self.key(tuple(combo)))
        return out

    # ------------------------------------------------------------------ experiments
    def run(self, raw: str, agent: str) -> dict:
        """Run one experiment and return its provenance record (Flowcept / W3C PROV task shape)."""
        t0 = time.time()
        if self.spec.duration_s:
            time.sleep(self.spec.duration_s)
        combo = self.parse(raw)
        value, err = None, None
        if combo is None:
            err = "invalid entity"
        elif self.fails(combo):
            err = "experiment failed"
        else:
            value = self.true_value(combo) + self.rng_noise.gauss(0, self.spec.noise_sd)
        return {"task_id": uuid.uuid4().hex, "activity_id": "run_experiment", "subtype": "experiment",
                "status": "ERROR" if err else "FINISHED", "started_at": t0, "ended_at": time.time(),
                "used": {"entity": raw}, "generated": {"value": value} if value is not None else None,
                "stderr": err, "custom_metadata": {"agent": agent}}

    # ------------------------------------------------------------------ statements with known truth
    def main_effect(self, slot: str, option: int, region: tuple[str, int] | None = None) -> float:
        """Mean outcome with slot=option minus mean outcome with slot!=option (oracle)."""
        pop = [r for r in self.population() if r["value"] is not None]
        if region:
            pop = [r for r in pop if r[f"slot_{region[0]}"] == region[1]]
        a = [r["value"] for r in pop if r[f"slot_{slot}"] == option]
        b = [r["value"] for r in pop if r[f"slot_{slot}"] != option]
        return (sum(a) / len(a) - sum(b) / len(b)) if a and b else 0.0

    def planted_statements(self, n: int, truth: bool, seed: int) -> list[str]:
        """Statements whose truth is known by construction (used for adversarial injection)."""
        rng = random.Random(seed)
        cands = []
        for s in self.slots:
            for o in range(self.spec.n_options):
                eff = self.main_effect(s, o)
                if abs(eff) < self.spec.main_scale / 2:
                    continue
                says_up = (eff > 0) == truth
                cands.append(f"Option {s}{o} {'increases' if says_up else 'decreases'} the outcome "
                             f"compared with the other options of slot {s}.")
        rng.shuffle(cands)
        return cands[:n]


class SyntheticAdapter(DomainAdapter):
    entity_noun = "design"

    def __init__(self, env: SyntheticEnv) -> None:
        self.env = env
        slots = ", ".join(env.slots)
        self.goal = Goal(objective=Objective("outcome", "max"),
                         text=(f"Find designs with the highest outcome. A design picks one option (0 to "
                               f"{env.spec.n_options - 1}) in each slot {slots}, written like "
                               f"{env.key(tuple(0 for _ in env.slots))}. Some combinations fail."),
                         constraints=[])

    def canonical(self, raw):
        combo = self.env.parse(raw)
        return self.env.key(combo) if combo else None

    def observation_from_record(self, record):
        if record.get("activity_id") != "run_experiment":
            return None
        raw = str((record.get("used") or {}).get("entity", ""))
        key = self.canonical(raw)
        value = (record.get("generated") or {}).get("value")
        ok = record.get("status") == "FINISHED" and value is not None
        return {"id": record["task_id"], "entity": key or f"raw:{raw}", "valid_entity": key is not None, "raw": raw,
                "value": float(value) if ok else None, "status": "ok" if ok else "failed",
                "error": record.get("stderr"), "agent": (record.get("custom_metadata") or {}).get("agent"),
                "t": float(record["ended_at"]), "started_at": float(record["started_at"])}

    def descriptors(self, entity):
        combo = self.env.parse(entity)
        return {f"slot_{s}": o for s, o in zip(self.env.slots, combo)} if combo else {}

    def descriptor_doc(self):
        return "".join(f"slot_{s} (int, option chosen in slot {s}, 0..{self.env.spec.n_options - 1})\n"
                       for s in self.env.slots)
