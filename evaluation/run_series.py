"""
Run an evaluation series: every (repetition, scenario, condition) of a series in config.SERIES, one GAIED campaign
each, one after the other, each in its own memory namespace. Meant to run inside the gaied container:

    python -m evaluation.run_series smoke                      # scripted LLM, a few minutes
    python -m evaluation.run_series pilot                      # the model of OLLAMA_URL / OLLAMA_MODEL
    python -m evaluation.run_series main --only full           # a subset of conditions

For each run, results/eval/<series>/<run id>/ receives: meta.json, campaign.log, analyzer.log, flowcept_settings.yaml,
faults.json and faults.jsonl (every computation and injected fault: the ground truth), provenance.jsonl (every Flowcept record published
during the run), memory.json and events.json (the memory at the end), and the post-execution report when the
analysis agent ran. A run whose meta.json says "ok" is skipped, so an interrupted series can be resumed.

Flowcept runs in online mode for every condition (the same settings, so the same overhead), and a recorder keeps
every published record: in online mode Flowcept's own JSONL file only keeps the last buffer.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import msgpack
import redis
import yaml

from pgmem.store import MemoryStore

from .config import CONDITIONS, SCENARIOS, SERIES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REDIS_HOST, REDIS_PORT = os.environ.get("REDIS_HOST", "localhost"), int(os.environ.get("REDIS_PORT", "6379"))


class ProvenanceRecorder:
    """Writes every record published on Flowcept's Redis channel to a JSONL file."""

    def __init__(self, path: str, channel: str = "interception") -> None:
        self.path, self.n = path, 0
        self.iterations: set[str] = set()            # LangGraph should_continue nodes seen: completed iterations
        self._r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        self._ps = self._r.pubsub(ignore_subscribe_messages=True)
        self._ps.psubscribe(channel)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            while not self._stop.is_set():
                try:
                    m = self._ps.get_message(timeout=1.0)
                except redis.exceptions.ConnectionError:
                    self._stop.wait(2.0)
                    continue
                if not m or m.get("type") != "pmessage":
                    continue
                try:
                    rec = msgpack.loads(m["data"])
                except Exception:  # noqa: BLE001 - not a Flowcept record
                    continue
                for r in rec if isinstance(rec, list) else [rec]:
                    if isinstance(r, dict):
                        fh.write(json.dumps(r, default=str) + "\n")
                        self.n += 1
                        if r.get("subtype") == "langgraph_node" and r.get("activity_id") == "should_continue":
                            self.iterations.add(str(r.get("task_id")))
                fh.flush()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._ps.close()


def flowcept_settings(run_dir: str) -> str:
    """Online Flowcept settings for this run: this Redis server, and Flowcept's own buffer file in the run folder."""
    s = yaml.safe_load(open(os.path.join(ROOT, "docker", "flowcept_settings_online.yaml")))
    for section in ("mq", "kv_db"):
        s[section]["host"], s[section]["port"] = REDIS_HOST, REDIS_PORT
    s["project"]["dump_buffer"]["path"] = os.path.join(run_dir, "flowcept_buffer.jsonl")
    path = os.path.join(run_dir, "flowcept_settings.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(s, fh)
    return path


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def drain(store: MemoryStore, timeout_s: float) -> bool:
    """Wait until the analysis agent has processed every memory event (nothing pending, nothing unread)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            groups = store.r.xinfo_groups(store._k("events"))
        except redis.exceptions.ResponseError:
            return True
        g = next((g for g in groups if g["name"] == "analyzer"), None)
        if g is None or (g["pending"] == 0 and (g.get("lag") in (0, None))):
            return True
        time.sleep(2)
    return False


def watch_switch(recorder: ProvenanceRecorder, faults_path: str, faults_log: str, switch: dict, meta: dict,
                 stop: threading.Event) -> None:
    """Change the injected faults once the agents have completed a given number of iterations."""
    while not stop.wait(1):
        if len(recorder.iterations) >= switch["after_iterations"]:
            with open(faults_path, "w") as fh:
                json.dump(switch["faults"], fh)
            meta["fault_switch_at"] = time.time()
            with open(faults_log, "a") as fh:
                fh.write(json.dumps({"t": meta["fault_switch_at"], "fault": "switch",
                                     "phase": switch["faults"].get("phase"),
                                     "iterations": len(recorder.iterations)}) + "\n")
            return


def run_one(series: str, cfg: dict, scenario: str, condition: str, rep: int, out_root: str) -> dict:
    run_id = f"{scenario}__{condition}__r{rep}"
    run_dir = os.path.join(out_root, run_id)
    meta_path = os.path.join(run_dir, "meta.json")
    if os.path.exists(meta_path) and json.load(open(meta_path)).get("status") == "ok":
        print(f"skip {run_id} (done)")
        return json.load(open(meta_path))
    if os.path.isdir(run_dir):              # an interrupted run is started again from scratch
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    sc, cond = SCENARIOS[scenario], CONDITIONS[condition]
    namespace = f"eval-{series}-{run_id}".replace("__", "-")
    store = MemoryStore.connect(REDIS_HOST, REDIS_PORT, namespace)
    store.reset()
    faults_path = os.path.join(run_dir, "faults.json")
    with open(faults_path, "w") as fh:
        json.dump(sc["faults"], fh)
    faults_log = os.path.join(run_dir, "faults.jsonl")
    open(faults_log, "w").close()

    max_iterations = sc.get("max_iterations", cfg["max_iterations"])
    env = {**os.environ, **cond["env"], "PGMEM_NAMESPACE": namespace, "SEEDS": sc["seeds"],
           "MAX_ITERATIONS": str(max_iterations),
           "LLM_SEED": str(1000 + rep),                     # paired runs: the same sampling seed in every condition
           "FLOWCEPT_SETTINGS_PATH": flowcept_settings(run_dir),
           "PGMEM_FAULTS": faults_path, "PGMEM_FAULTS_LOG": faults_log,
           "PYTHONPATH": os.pathsep.join(x for x in (ROOT, os.path.join(ROOT, "docker"), os.environ.get("PYTHONPATH"))
                                         if x)}
    if not sc["faults"]:
        env.pop("PGMEM_FAULTS")
    if cfg["llm"] == "scripted":
        env.update({"LLM_PROVIDER": "scripted", "OLLAMA_MODEL": "scripted",
                    "SCRIPTED_DELAY_S": env.get("SCRIPTED_DELAY_S", "1")})
    meta = {"series": series, "run_id": run_id, "scenario": scenario, "condition": condition, "repetition": rep,
            "namespace": namespace, "seeds": sc["seeds"], "faults": sc["faults"], "llm": cfg["llm"],
            "model": env.get("OLLAMA_MODEL") or env.get("LLM_MODEL"), "max_iterations": max_iterations,
            "llm_seed": env["LLM_SEED"], "motifs": sc.get("motifs"),
            "git_commit": git_commit(), "started_at": time.time(), "status": "running"}
    json.dump(meta, open(meta_path, "w"), indent=1)
    print(f"run {run_id} (namespace {namespace})", flush=True)

    recorder = ProvenanceRecorder(os.path.join(run_dir, "provenance.jsonl"))
    stop_watch = threading.Event()
    if "switch" in sc:
        threading.Thread(target=watch_switch, daemon=True,
                         args=(recorder, faults_path, faults_log, sc["switch"], meta, stop_watch)).start()
    analyzer = None
    if cond["analyzer"]:
        analyzer = subprocess.Popen([sys.executable, "-m", "pgmem.analyzer"], cwd=ROOT, env=env,
                                    stdout=open(os.path.join(run_dir, "analyzer.log"), "w"), stderr=subprocess.STDOUT)
    timeout = float(os.environ.get("EVAL_CAMPAIGN_TIMEOUT_S", "5400"))
    try:
        with open(os.path.join(run_dir, "campaign.log"), "w") as log:
            campaign = subprocess.run([sys.executable, "run-07.py"], cwd=ROOT, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, timeout=timeout)
        meta["campaign_exit_code"] = campaign.returncode
        meta["status"] = "ok" if campaign.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        meta["status"] = "timeout"
    meta["campaign_ended_at"] = time.time()
    stop_watch.set()
    time.sleep(float(os.environ.get("EVAL_PROVENANCE_GRACE_S", "15")))   # online records arrive a few seconds late
    if analyzer is not None:
        meta["analyzer_drained"] = drain(store, float(os.environ.get("EVAL_ANALYZER_DRAIN_S", "600")))
        analyzer.terminate()
        try:
            analyzer.wait(timeout=30)
        except subprocess.TimeoutExpired:
            analyzer.kill()
    recorder.close()
    meta["provenance_records"] = recorder.n
    if analyzer is not None and meta["status"] == "ok":
        post_env = {**env, "PGMEM_PROVENANCE": "jsonl"}
        post = subprocess.run([sys.executable, "-m", "pgmem.analyzer", "--post", "--jsonl",
                               os.path.join(run_dir, "provenance.jsonl"), "--out", run_dir], cwd=ROOT, env=post_env,
                              capture_output=True, text=True)
        meta["post_analysis_exit_code"] = post.returncode
        with open(os.path.join(run_dir, "analyzer.log"), "a") as fh:
            fh.write(post.stdout + post.stderr)
    with open(os.path.join(run_dir, "memory.json"), "w") as fh:
        json.dump(store.export(), fh, indent=1, default=str)
    with open(os.path.join(run_dir, "events.json"), "w") as fh:
        json.dump([{"id": i, "event": f.get("event"), "t": float(f.get("t", 0)),
                    "payload": json.loads(f.get("payload") or "{}")}
                   for i, f in store.r.xrange(store._k("events"))], fh, indent=1)
    meta["ended_at"] = time.time()
    json.dump(meta, open(meta_path, "w"), indent=1)
    print(f"  {meta['status']} in {meta['ended_at'] - meta['started_at']:.0f} s", flush=True)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("series", choices=sorted(SERIES))
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these conditions")
    ap.add_argument("--scenarios", nargs="*", default=None, help="restrict to these scenarios")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "eval"))
    a = ap.parse_args(argv)
    cfg = SERIES[a.series]
    out_root = os.path.join(a.out, a.series)
    os.makedirs(out_root, exist_ok=True)
    conditions = [c for c in cfg["conditions"] if not a.only or c in a.only]
    scenarios = [s for s in cfg["scenarios"] if not a.scenarios or s in a.scenarios]
    runs = [(s, c, r) for r in range(1, cfg["repetitions"] + 1) for s in scenarios for c in conditions]
    print(f"series {a.series}: {len(runs)} runs ({cfg['llm']} LLM)", flush=True)
    failed = 0
    for i, (s, c, r) in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}]", end=" ")
        failed += run_one(a.series, cfg, s, c, r, out_root)["status"] != "ok"
    print(f"done: {len(runs) - failed} ok, {failed} not ok. Metrics: python -m evaluation.metrics {out_root}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
