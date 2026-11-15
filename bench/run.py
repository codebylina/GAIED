"""
Runs the benchmark: scenarios x memory conditions x environment seeds.

    python -m bench.run bench/configs/<config>.json [--only scenario/condition] [--dry]

Each run writes results/<config name>/runs/<scenario>__<condition>__s<seed>.json with the memory
export, the agents' logs, the environment specification and the oracle optimum. A run is skipped
only if its file exists and says status "ok". Same seed = same environment and same start designs
for every condition (paired design).
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import traceback

from bench.agent import ResearchAgent
from bench.env import EnvSpec, SyntheticAdapter, SyntheticEnv
from memlab import LLM, InMemoryStore, LLMUsage, MemoryConfig, make_memory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_llm(cfg: dict, env: SyntheticEnv, usage: LLMUsage) -> LLM:
    if os.environ.get("LLM_PROVIDER", "").lower() == "scripted" or cfg.get("scripted"):
        from bench.scripted import ScriptedLLM
        return LLM(provider="scripted", scripted=ScriptedLLM(env, seed=env.spec.seed), usage=usage)
    return LLM(usage=usage)


def run_one(cfg: dict, scenario: dict, cond: dict, seed: int) -> dict:
    spec = EnvSpec(**{**cfg["env"], **scenario.get("env", {}), "seed": seed})
    env = SyntheticEnv(spec)
    adapter = SyntheticAdapter(env)
    usage = LLMUsage()
    llm = build_llm(cfg, env, usage)
    mcfg = MemoryConfig(**{**cfg.get("memory_defaults", {}), **cond.get("cfg", {})})
    store = InMemoryStore()
    kw = {"llm": llm}
    if cond["memory"] == "mem0":
        import shutil

        from memlab.mem0_memory import LocalMem0, build_mem0
        run_dir = os.path.join(ROOT, "results", cfg["name"], "mem0_store", f"{scenario['name']}__{cond['name']}__s{seed}")
        shutil.rmtree(run_dir, ignore_errors=True)       # every run starts from an empty memory
        scripted = None
        if llm.provider == "scripted":
            from bench.scripted import mem0_scripted
            scripted = mem0_scripted(int(os.environ.get("EMBED_DIMS", "768")))
        kw["backend"] = LocalMem0(build_mem0(run_dir, "campaign", scripted=scripted), run_id="campaign", usage=usage)
        kw["top_k"] = cond.get("top_k")              # None: Mem0's default
    memory = make_memory(cond["memory"], store, adapter, mcfg, **kw)

    n_agents, iters = cfg["n_agents"], cfg["iterations"]
    seeds = env.seeds(n_agents)
    adversary = scenario.get("adversary")          # index of the lying agent, or None
    false_stmts = env.planted_statements(scenario.get("n_false", 0), truth=False, seed=seed) if adversary is not None else []
    stop = threading.Event()

    def make_agents():
        return [ResearchAgent(f"agent{i}", seeds[i], env, memory, llm, cfg["batch_size"],
                              pull_rounds=cfg.get("agent", {}).get("pull_rounds", 0) if cond.get("pull") else 0,
                              wait_timeout_s=cfg.get("agent", {}).get("wait_timeout_s", 0.0),
                              false_statements=false_stmts if adversary == i else None, stop=stop,
                              refill_rounds=cfg.get("agent", {}).get("refill_rounds", 0),
                              max_conclusions=cfg.get("agent", {}).get("max_conclusions"))
                for i in range(n_agents)]

    agent_done = threading.Event()

    def steward():
        while not agent_done.is_set():
            if memory.maintain() == 0:
                time.sleep(cfg.get("steward_idle_s", 0.5))
        while memory.maintain():
            pass

    t0 = time.time()
    st = threading.Thread(target=steward, daemon=True)
    st.start()
    crash_at = scenario.get("crash_at")
    phases = [(0, crash_at), (crash_at, iters)] if crash_at else [(0, iters)]
    agents_log = []
    for a, b in phases:     # a crash discards every agent's in-RAM state; the memory store persists
        agents = make_agents()
        threads = [threading.Thread(target=ag.run, args=(b, a), daemon=True) for ag in agents]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        agents_log += [{"agent": ag.name, "phase": [a, b], "seed": ag.seed, "log": ag.log,
                        "adversary": ag.false_statements is not None} for ag in agents]
    agent_done.set()
    st.join()
    return {"status": "ok", "config": cfg["name"], "scenario": scenario, "condition": cond, "seed": seed,
            "env": spec.__dict__, "optimum": env.optimum(), "start_region": list(env.region),
            "decoy": list(env.decoy) if env.decoy else None, "false_statements": false_stmts,
            "agents": agents_log, "wall_time_s": time.time() - t0, "memory": memory.export()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--only", default="", help="scenario/condition filter, e.g. deceptive/mem0")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    out = os.path.join(ROOT, "results", cfg["name"], "runs")
    os.makedirs(out, exist_ok=True)
    for scenario in cfg["scenarios"]:
        for cond in cfg["conditions"]:
            if args.only and args.only not in f"{scenario['name']}/{cond['name']}":
                continue
            for seed in cfg["seeds"]:
                path = os.path.join(out, f"{scenario['name']}__{cond['name']}__s{seed}.json")
                if os.path.exists(path) and json.load(open(path)).get("status") == "ok":
                    print("skip", os.path.basename(path), flush=True)
                    continue
                print("run ", os.path.basename(path), flush=True)
                try:
                    res = run_one(cfg, scenario, cond, seed)
                except Exception:
                    res = {"status": "error", "error": traceback.format_exc()}
                    print(res["error"], flush=True)
                with open(path + ".tmp", "w") as fh:
                    json.dump(res, fh, default=str)
                os.replace(path + ".tmp", path)


if __name__ == "__main__":
    main()
