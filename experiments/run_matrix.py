"""
Run the real experiment matrix: every condition x every repetition, each one a full
campaign of run-07.py (Academy + LangGraph + LLM + xTB + Flowcept), then export it.

In Docker, from GAIED/docker (Redis and the LLM must be up):
    docker compose run --rm gaied python experiments/run_matrix.py experiments/configs/pilot.json
    docker compose run --rm gaied python experiments/run_matrix.py experiments/configs/main.json
    docker compose run --rm gaied python experiments/analyze.py experiments/results/main

- Repetitions are interleaved (rep 1 of every condition, then rep 2...) so that a drift of
  the machine or of an online LLM affects all conditions equally.
- Resumable: a run whose export already exists is skipped. Re-run the same command after
  a crash or an interruption.
- A run that exceeds `timeout_min` is killed; what it had done is still exported and
  marked as timed out in its .meta.json.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.conditions import CONDITIONS  # noqa: E402


def export_from_redis(campaign: str, path: str) -> None:
    from evident.memory import EvidentMemory
    import redis
    client = redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"), decode_responses=True)
    mem = EvidentMemory(client, campaign, "cli")
    with open(path, "w") as fh:
        json.dump(mem.export(), fh)


def main(config_path: str) -> None:
    with open(config_path) as fh:
        cfg = json.load(fh)
    name = cfg.get("name") or os.path.splitext(os.path.basename(config_path))[0]
    out = os.path.join(ROOT, "experiments", "results", name)
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    conditions = cfg.get("conditions", list(CONDITIONS))
    reps = int(cfg.get("repetitions", 3))
    timeout = float(cfg.get("timeout_min", 120)) * 60

    plan = [(rep, cond) for rep in range(1, reps + 1) for cond in conditions]
    for i, (rep, cond) in enumerate(plan, 1):
        export = os.path.join(out, "runs", f"{cond}_r{rep}.json")
        if os.path.exists(export):
            print(f"[{i}/{len(plan)}] {cond} rep {rep}: already done, skipped")
            continue
        campaign = f"{name}-{cond}-r{rep}-{int(time.time())}"
        env = dict(os.environ)
        env.update({k: str(v) for k, v in cfg.get("env", {}).items()})
        env.update(CONDITIONS[cond])
        env.update({"MEMORY_CAMPAIGN_ID": campaign, "EVIDENT_EXPORT": export,
                    "EXPERIMENT_CONDITION": cond, "SEEDS": ",".join(cfg["seeds"]),
                    "MAX_ITERATIONS": str(cfg.get("max_iterations", 10))})
        log_path = os.path.join(out, "logs", f"{cond}_r{rep}.log")
        print(f"[{i}/{len(plan)}] {cond} rep {rep}: campaign {campaign} (log {log_path})", flush=True)
        t0 = time.time()
        status = "ok"
        with open(log_path, "w") as log:
            proc = subprocess.Popen([sys.executable, "run-07.py"], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=timeout)
                if code != 0:
                    status = f"exit code {code}"
            except subprocess.TimeoutExpired:
                status = "timeout"
                os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(10)
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if not os.path.exists(export):              # killed or crashed: export what exists
            try:
                export_from_redis(campaign, export)
            except Exception as exc:
                status += f"; export failed: {exc}"
        with open(export.replace(".json", ".meta.json"), "w") as fh:
            json.dump({"campaign": campaign, "condition": cond, "rep": rep, "status": status,
                       "seconds": round(time.time() - t0, 1)}, fh)
        print(f"    -> {status} in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"\nDone. Analyse with:  python experiments/analyze.py {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "experiments", "configs", "pilot.json"))
