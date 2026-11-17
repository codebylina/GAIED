"""
Integration test: GAIED's real agent and LangGraph workflow (mol_design_agents.XTBSimulationAgent) with the shared
memory in full mode and the analysis agent running beside it, without any model and without xTB:
  - LLM_PROVIDER=scripted (docker/local_llm.py), which also proposes two invalid SMILES (technical failures);
  - xTB replaced by a stub calculator (deterministic energies, zero forces);
  - two agents with the same seed, run one after the other (the second must reuse the first one's results).
Needs Redis (REDIS_HOST / REDIS_PORT). Run: python tests/test_pgmem_agent_loop.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "docker")]


def install_xtb_stub() -> None:
    """A deterministic stand-in for xtb.ase.calculator.XTB: energy from the composition and the total charge."""
    import numpy as np
    from ase.calculators.calculator import Calculator, all_changes

    class XTB(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, accuracy: float = 1.0, **kw):
            super().__init__(**kw)

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            z = self.atoms.get_atomic_numbers()
            q = float(np.sum(self.atoms.get_initial_charges()))
            self.results = {"energy": -float(np.sum(z)) + q * (5.0 + 0.37 * float(np.sum(z > 1))),
                            "forces": np.zeros((len(z), 3))}

    mods = {"xtb": types.ModuleType("xtb"), "xtb.ase": types.ModuleType("xtb.ase"),
            "xtb.ase.calculator": types.ModuleType("xtb.ase.calculator")}
    mods["xtb.ase.calculator"].XTB = XTB
    sys.modules.update(mods)


async def run_agent(seed: str, iterations: int):
    from mol_design_agents import XTBConfig, XTBSimulationAgent
    agent = XTBSimulationAgent(XTBConfig(), seed, iterations, 99)
    await agent.agent_on_startup()
    try:
        await agent.conduct_simulation_campaign(asyncio.Event())
    finally:
        await agent.agent_on_shutdown()
    return agent


def main() -> int:
    ns = f"loop-{time.time_ns()}"
    os.environ.update({"PGMEM_MODE": "full", "PGMEM_NAMESPACE": ns, "LLM_PROVIDER": "scripted",
                       "OLLAMA_MODEL": "scripted", "N_XTB_WORKERS": "1", "PGMEM_PROVENANCE": "none"})
    install_xtb_stub()
    from pgmem.analyzer import Analyzer
    from pgmem.config import Config
    from pgmem.llm import AnalysisLLM
    from pgmem.store import MemoryStore
    cfg = Config.from_env()
    store = MemoryStore.connect(cfg.redis_host, cfg.redis_port, ns)
    an = Analyzer(store, cfg, llm=AnalysisLLM("scripted"))
    stop = threading.Event()

    def analyze():
        while not stop.is_set():
            an.step(block_ms=200, retry_idle_ms=1000)
    th = threading.Thread(target=analyze, daemon=True)
    th.start()
    t0 = time.time()
    a1 = asyncio.run(run_agent("CNC(N)=O", 3))
    time.sleep(1.0)                                   # the analysis agent works on its own; give it a moment
    a2 = asyncio.run(run_agent("CNC(N)=O", 3))
    time.sleep(1.5)
    stop.set()
    th.join(timeout=5)
    an.step(block_ms=10, retry_idle_ms=0)
    report = an.post_execution(records=[])
    dt = time.time() - t0

    m1 = store.all("M1")
    m2 = store.all("M2")
    reused = [e for _, b in store.r.xrange(store._k("events")) for e in [b] if b.get("event") == "reused"]
    checks = {
        "M1 observations recorded, each linked to a task id": bool(m1) and all(
            any(r.get("task_id") for r in e.prov_refs) for e in m1),
        "conditions with convergence stored in M1": all("converged" in e.data["conditions"] for e in m1),
        "invalid SMILES recorded as failures in M2, not M1": any(e.kind == "failure" for e in m2)
                                                          and not any(e.key == "C1CC" for e in m1),
        "an attempt (M2) per iteration": sum(1 for e in m2 if e.kind == "attempt") == 6,
        "knowledge proposed at conclusion (M3) with evidence": any(
            e.data["supporting_evidence"] for e in store.all("M3")),
        "second agent reused the first agent's results": len(reused) > 0,
        "analysis agent wrote diagnoses": any(e.kind == "diagnosis" for e in m2),
        "recurring failure class produced a procedure (M4)": bool(store.all("M4")),
        "guidance delivered to an agent during execution": any(g.delivered for g in store.all_guidance()),
        "guidance application was assessed": any(g.applied for g in store.all_guidance()),
    }
    print(f"two agents x 3 iterations in {dt:.1f} s; memory {report['memory']}; guidance {report['guidance']}")
    for e in m2:
        if e.kind == "diagnosis":
            print(f"   diagnosis [{e.data['problem_type']} / {e.data['basis']}] {e.text}")
    for g in store.all_guidance():
        print(f"   guidance -> {g.target}: delivered {len(g.delivered)}, applied {g.applied}, "
              f"effective {g.effective}: {g.message[:90]}")
    failed = [k for k, ok in checks.items() if not ok]
    for k, ok in checks.items():
        print("ok  " if ok else "FAIL", k)
    store.reset()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
