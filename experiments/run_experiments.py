"""
Run GAIED under the four experimental configurations and measure them.

  1 none                  no shared memory (original GAIED)
  2 basic                 first docker-local shared memory
  3 structured            four-layer memory
  4 structured+analysis   four-layer memory + online provenance & memory analysis service

Each run gets its own working directory (Flowcept buffers, Parsl logs) and its own memory namespace, so
runs are isolated; metrics come from the Flowcept provenance and the memory (experiments/metrics.py).
The LLM is whatever docker/local_llm.py is configured for (Ollama or an API). Nothing is simulated here:
if the LLM is slow, runs are slow.

Usage (repository root; in Docker: docker compose run --rm gaied python experiments/run_experiments.py ...):
  python experiments/run_experiments.py --repeats 3 --iterations 5 --seeds "CNC(N)=O,CNC(N)=O,CC1=C(O)N=C(O)N1"
  python experiments/run_experiments.py --configs none,structured --flowcept offline
Results: experiments/results/<stamp>/<config>-<rep>/{run.log, metrics.json}, summary.md, metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CONFIGS = ["none", "basic", "structured", "structured+analysis"]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seeds", default=os.environ.get("SEEDS", "CNC(N)=O,CNC(N)=O"))
    ap.add_argument("--iterations", type=int, default=int(os.environ.get("MAX_ITERATIONS", "3")))
    ap.add_argument("--flowcept", choices=["online", "offline"], default="online",
                    help="online: Flowcept MQ + MongoDB (needed for live analysis); offline: JSONL buffers")
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("RUN_TIMEOUT", "7200")))
    a = ap.parse_args(argv)

    settings = os.path.join(ROOT, "docker", "flowcept_settings_online.yaml" if a.flowcept == "online"
                            else "flowcept_settings.yaml")
    os.environ["FLOWCEPT_SETTINGS_PATH"] = settings            # before importing flowcept (metrics)
    import redis

    from experiments.metrics import aggregate, collect, offline_detection
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = a.out or os.path.join(ROOT, "experiments", "results", stamp)
    os.makedirs(out, exist_ok=True)
    r = redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"), decode_responses=True)
    rows = []
    for rep in range(a.repeats):
        for config in a.configs.split(","):
            run_dir = os.path.join(out, f"{config.replace('+', '_')}-{rep}")
            os.makedirs(run_dir, exist_ok=True)
            ns = f"exp-{stamp}-{config.replace('+', '_')}-{rep}"
            env = {**os.environ, "MEMORY_MODE": config, "MEMORY_NAMESPACE": ns, "SEEDS": a.seeds,
                   "MAX_ITERATIONS": str(a.iterations), "FLOWCEPT_SETTINGS_PATH": settings,
                   "PYTHONPATH": os.pathsep.join([ROOT, os.path.join(ROOT, "docker"), os.environ.get("PYTHONPATH", "")])}
            analysis = None
            if config == "structured+analysis":
                analysis = subprocess.Popen([sys.executable, "-m", "analysis", "run"], cwd=ROOT, env=env,
                                            stdout=open(os.path.join(run_dir, "analysis.log"), "w"),
                                            stderr=subprocess.STDOUT)
                time.sleep(3)
            print(f"[{rep}] {config}: running in {run_dir}", flush=True)
            t0 = time.time()
            with open(os.path.join(run_dir, "run.log"), "w") as log:
                proc = subprocess.run([sys.executable, os.path.join(ROOT, "run-07.py")], cwd=run_dir, env=env,
                                      stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout)
            wall = time.time() - t0
            if analysis is not None:
                time.sleep(10)                                     # last analysis cycle on the final data
                analysis.send_signal(signal.SIGINT)
                try:
                    analysis.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    analysis.kill()
            text = open(os.path.join(run_dir, "run.log")).read()
            found = re.search(r"Flowcept campaign: (\S+)", text)
            if not found:
                print(f"    no campaign id in run.log (exit {proc.returncode}); skipped")
                continue
            m = collect(run_dir, config, ns, found.group(1), a.flowcept == "online", r)
            m.update(repeat=rep, exit_code=proc.returncode, run_wall_time_s=round(wall, 1))
            if config == "none":   # detection accuracy of the analysis on real redundancy (ground truth: provenance)
                m["offline_detection"] = offline_detection(run_dir, found.group(1), a.flowcept == "online", r,
                                                           ns + "-offline-analysis")
            json.dump(m, open(os.path.join(run_dir, "metrics.json"), "w"), indent=1, default=str)
            rows.append(m)
            print(f"    xTB runs {m['xtb_runs']}, distinct {m['distinct_molecules']}, redundant "
                  f"{m['redundant_computations']}, best {m['best_value']}, LLM calls {m['llm_calls']}", flush=True)
    json.dump(rows, open(os.path.join(out, "metrics.json"), "w"), indent=1, default=str)
    table = aggregate(rows)
    det = [r_["offline_detection"] for r_ in rows if r_.get("offline_detection")]
    if det:
        table += "\n\nDetection of redundant computations by the analysis service on the provenance of the runs " \
                 "without shared memory (ground truth from the same provenance):\n" + \
                 "\n".join(f"- truth {d['truth']}, detected {d['detected']}, precision {d['precision']}, "
                           f"recall {d['recall']}" for d in det)
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write(f"# GAIED memory experiments {stamp}\n\nSeeds: {a.seeds}; iterations: {a.iterations}; "
                 f"Flowcept: {a.flowcept}; LLM: {os.environ.get('LLM_PROVIDER', 'ollama')} "
                 f"{os.environ.get('LLM_MODEL') or os.environ.get('OLLAMA_MODEL', '')}\n\n{table}\n")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
