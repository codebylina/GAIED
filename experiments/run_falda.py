"""
Run the FALDA baseline campaigns (repetitions x conditions) and export each one in the
EVIDENT export format, so that experiments/analyze.py of the evident-memory branch can
compare them with the EVIDENT conditions.

From GAIED/docker (Redis, the FALDA gateway and the LLM must be up):
    docker compose --profile falda up -d redis falda
    docker compose run --rm --no-deps gaied python experiments/run_falda.py experiments/configs/falda.json

- A run is skipped only if its .meta.json says "ok"; a failed attempt is kept aside as
  <run>.json.failed and redone. A run longer than timeout_min is killed.
- Conditions: "falda" (FALDA shared memory) or "none" (no memory, original prompts).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(config_path: str) -> None:
    with open(config_path) as fh:
        cfg = json.load(fh)
    name = cfg["name"]
    out = os.path.join(ROOT, "experiments", "results", name)
    for sub in ("runs", "logs"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    plan = [(rep, c) for rep in range(1, int(cfg.get("repetitions", 1)) + 1) for c in cfg["conditions"]]
    for i, (rep, cond) in enumerate(plan, 1):
        export = os.path.join(out, "runs", f"{cond}_r{rep}.json")
        meta_path = export.replace(".json", ".meta.json")
        if os.path.exists(export) and os.path.exists(meta_path):
            with open(meta_path) as fh:
                if json.load(fh).get("status") == "ok":
                    print(f"[{i}/{len(plan)}] {cond} rep {rep}: already done, skipped")
                    continue
        for stale in (export, meta_path):
            if os.path.exists(stale):
                os.replace(stale, stale + ".failed")
        campaign = f"{name}-{cond}-r{rep}-{int(time.time())}"
        env = dict(os.environ)
        env.update({k: str(v) for k, v in cfg.get("env", {}).items()})
        env.update({"MEMORY_MODE": cond, "MEMORY_CAMPAIGN_ID": campaign, "BENCH_EXPORT": export,
                    "EXPERIMENT_CONDITION": cond, "SEEDS": ",".join(cfg["seeds"]),
                    "MAX_ITERATIONS": str(cfg.get("max_iterations", 10)), "PYTHONUNBUFFERED": "1"})
        log_path = os.path.join(out, "logs", f"{cond}_r{rep}.log")
        print(f"[{i}/{len(plan)}] {cond} rep {rep}: campaign {campaign} (log {log_path})", flush=True)
        t0, status = time.time(), "ok"
        with open(log_path, "w") as log:
            proc = subprocess.Popen([sys.executable, "run-07.py"], cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=float(cfg.get("timeout_min", 180)) * 60)
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
        if status == "ok" and not os.path.exists(export):
            status = "no export written"
        if status != "ok" and os.path.exists(export):
            os.replace(export, export + ".failed")
        with open(meta_path, "w") as fh:
            json.dump({"campaign": campaign, "condition": cond, "rep": rep, "status": status,
                       "seconds": round(time.time() - t0, 1)}, fh)
        print(f"    -> {status} in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"\nDone. Copy {out}/runs/*.json next to the EVIDENT runs to analyse them together.")


if __name__ == "__main__":
    main(sys.argv[1])
