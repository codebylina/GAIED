"""
Synthetic campaigns: the REAL agent graph (mol_design_agents.py) and the REAL EVIDENT
memory, but a scripted LLM and a fake xTB whose structure-property rules are KNOWN.

Purpose: validate the whole pipeline (memory modes, monitor, injection, export, analysis)
in minutes, without GPU, and check that each mechanism does what it claims when the
ground truth is known. These numbers are NOT results about LLM agents: the paper's
results come from experiments/run_matrix.py with a real LLM and real xTB.

    python experiments/synthetic.py --out experiments/results/synthetic --reps 3
    python experiments/analyze.py experiments/results/synthetic
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
import types
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "docker")]

try:
    import xtb.ase.calculator  # noqa: F401
except ImportError:                                  # machine without xTB: stub module
    _fake = types.ModuleType("xtb.ase.calculator")
    _fake.XTB = object
    sys.modules.update({"xtb": types.ModuleType("xtb"), "xtb.ase": types.ModuleType("xtb.ase"),
                        "xtb.ase.calculator": _fake})

from langchain.messages import HumanMessage, SystemMessage  # noqa: E402

import mol_design_agents as mda  # noqa: E402
from evident import EvidentMemory, canonical_smiles, features_of, mol_info  # noqa: E402
from evident.claims import parse_claims_keywords  # noqa: E402
from experiments.conditions import CONDITIONS  # noqa: E402

# ------------------------------------------------------------------ fake xTB with known rules
TRUE_EFFECTS = {"fluorine": +0.8, "nitrile": +0.5, "carbonyl": +0.3,
                "aromatic_ring": -0.9, "methyl": -0.4, "amine": -0.3}


def fake_energy(config, smiles: str) -> float:
    info = mol_info(smiles)
    if not info["valid"]:
        raise ValueError(f"Parse failure for {smiles}")
    time.sleep(float(os.environ.get("SYNTH_XTB_SECONDS", "1.0")))
    rng = random.Random(zlib.crc32(canonical_smiles(smiles).encode()))
    e = 10.0 + sum(v for f, v in TRUE_EFFECTS.items() if f in info["features"])
    return e - 0.03 * (info["heavy_atoms"] or 0) + rng.gauss(0, 0.15)


# ------------------------------------------------------------------ candidate molecules
_R = ["C", "CC", "CF", "C(F)(F)F", "C#N", "CO", "CN", "c1ccccc1", "Cl", "OC", "C=C", "CCC",
      "C(=O)C", "CC#N", "C(F)F", "c1ccncc1", "CCCCCCCCCC"]
_CORES = ["NC(=O)N{}", "{}C(N)=O", "{}C(=O)O", "{}CO", "{}CC#N", "N#CC({})F", "{}NC(=O)C"]
POOL = sorted({canonical_smiles(c.format(r)) for c in _CORES for r in _R}
              | {"C1CC((", "FC(F)(F", "NC(=O)N("})


class _Resp:
    def __init__(self, content, tool_calls=None):
        self.content, self.tool_calls = content, tool_calls or []


def _section(text: str, title: str) -> str:
    m = re.search(re.escape(title) + r".*?\n(.*?)(?:\n\n|$)", text, re.DOTALL)
    return m.group(1) if m else ""


class ScriptedLLM:
    """Deterministic stand-in for the LLM. It follows what the prompt tells it:
    exclusion list, verified knowledge, other agents' lessons and STOP directives."""
    BATCH = 4

    def __init__(self, model: str, temperature: float = 0.1):
        self.model, self.temperature, self._tools = model, temperature, None

    def bind_tools(self, tools):
        b = ScriptedLLM(self.model, self.temperature)
        b._tools = tools
        return b

    async def ainvoke(self, messages):
        user = "\n".join(m.content for m in messages if isinstance(m, HumanMessage))
        system = "\n".join(m.content for m in messages if isinstance(m, SystemMessage))
        await asyncio.sleep(float(os.environ.get("SYNTH_LLM_SECONDS", "0.05")))
        if self._tools:
            return _Resp("", self._choose(user))
        seed = re.search(r"SEED=(\S+)", user) or re.search(r"starting with the molecule (\S+)", user)
        tag = f"SEED={seed.group(1) if seed else 'C'} "
        if "CLAIMS:" in user:
            return _Resp(tag + self._conclude(user))
        return _Resp(tag + "explore around the seed. "
                     + ("(critique)" if "Critique" in system else ""))

    def _choose(self, prompt: str) -> list[dict]:
        excluded = set()
        m = re.search(r"Simulated Molecules: (.*?) Do not repeat", prompt, re.DOTALL)
        if m:
            excluded = {canonical_smiles(s) for s in m.group(1).split()}
        seed = re.search(r"SEED=(\S+)", prompt)
        seed_feats = set(features_of(seed.group(1))) if seed else set()
        bonus: dict[str, float] = {}
        for c in parse_claims_keywords(_section(prompt, "VERIFIED KNOWLEDGE")):
            bonus[c["feature"]] = bonus.get(c["feature"], 0) + (1.0 if c["direction"] == "up" else -1.0)
        for c in parse_claims_keywords(_section(prompt, "LESSONS FROM OTHER AGENTS")):
            bonus[c["feature"]] = bonus.get(c["feature"], 0) + (1.0 if c["direction"] == "up" else -1.0)
        obey_goal = "[STOP]" in prompt
        rng = random.Random(zlib.crc32((os.environ.get("SYNTH_SEED", "0") + prompt).encode()))
        scored = []
        for s in POOL:
            if s in excluded:
                continue
            info = mol_info(s)
            if obey_goal and info["valid"] and ("Cl" in info["elements"] or (info["heavy_atoms"] or 0) > 12):
                continue
            score = rng.random() + 0.4 * len(seed_feats & set(info["features"]))
            score += sum(bonus.get(f, 0) for f in info["features"])
            scored.append((score, s))
        scored.sort(reverse=True)
        return [{"name": "compute_ionization_energy", "args": {"smiles": s}}
                for _, s in scored[:self.BATCH]]

    @staticmethod
    def _conclude(prompt: str) -> str:
        rows = re.findall(r"^(\S+)\t\t(-?[0-9.]+)$", prompt, re.MULTILINE)
        data = [(s, float(e)) for s, e in rows]
        best = None
        for feat in TRUE_EFFECTS | {"urea": 0, "hydroxyl": 0, "alkene": 0}:
            w = [e for s, e in data if feat in features_of(s)]
            o = [e for s, e in data if feat not in features_of(s)]
            if w and o:                        # naive reading of very few points: often wrong
                d = sum(w) / len(w) - sum(o) / len(o)
                if best is None or abs(d) > abs(best[1]):
                    best = (feat, d)
        if best is None:
            return "No conclusion yet.\nCLAIMS: []"
        verb = "increases" if best[1] > 0 else "decreases"
        return (f"In my results, the {best[0]} feature {verb} the ionization energy.\n"
                f'CLAIMS: [{{"feature": "{best[0]}", "effect": "{verb}"}}]')


# ------------------------------------------------------------------ one campaign
async def run_campaign(condition: str, rep: int, seeds: list[str], iterations: int, out: str) -> str:
    env = CONDITIONS[condition]
    os.environ.update(env)
    campaign = f"synthetic-{condition}-r{rep}-{int(time.time())}"
    os.environ["MEMORY_CAMPAIGN_ID"] = campaign
    orch = EvidentMemory.from_env("orchestrator")
    orch.set_goal({"objective": "find molecules with the highest ionization energy",
                   "seeds": seeds,
                   "constraints": {"max_heavy_atoms": 12, "allowed_elements": ["C", "H", "N", "O", "F"]},
                   "guidelines": ["valid SMILES", "synthesizable", "low cost"]})
    orch.set_meta(mode=env["MEMORY_MODE"], seeds=seeds, monitor=env["MONITOR"] == "true",
                  inject=env["INJECT_FALSE_CLAIMS"] == "true", max_iterations=iterations,
                  condition=condition, llm="scripted", synthetic=True, true_effects=TRUE_EFFECTS)
    agents = []
    for s in seeds:
        a = mda.XTBSimulationAgent(mda.XTBConfig(), s, iterations, None)
        await a.agent_on_startup()
        agents.append(a)
    stop = asyncio.Event()
    monitor = None
    if env["MONITOR"] == "true":
        from evident.monitor import RuleMonitor
        monitor = asyncio.create_task(RuleMonitor(EvidentMemory.from_env("monitor")).run(stop))
    await asyncio.gather(*(a.conduct_simulation_campaign(asyncio.Event()) for a in agents))
    stop.set()
    if monitor:
        await monitor
    for a in agents:
        a.pool.shutdown()
    path = os.path.join(out, "runs", f"{condition}_r{rep}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(orch.export(), fh)
    for key in orch.r.scan_iter(f"ev:{campaign}:*"):
        orch.r.delete(key)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "results", "synthetic"))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--seeds", default="CNC(N)=O,CNC(N)=O,CC1=C(O)N=C(O)N1,CC(N)=O")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    args = ap.parse_args()
    os.environ.setdefault("N_XTB_WORKERS", "2")
    os.environ.pop("FLOWCEPT_CAMPAIGN_ID", None)
    mda.SomeAPILLM = ScriptedLLM
    mda._compute_energy_sync = fake_energy          # inherited by the forked xTB workers
    for rep in range(1, args.reps + 1):
        for cond in args.conditions.split(","):
            t0 = time.time()
            os.environ["SYNTH_SEED"] = str(rep)
            path = asyncio.run(run_campaign(cond, rep, args.seeds.split(","), args.iterations, args.out))
            print(f"{cond:<16} rep {rep}: {time.time() - t0:5.1f} s -> {path}")


if __name__ == "__main__":
    main()
