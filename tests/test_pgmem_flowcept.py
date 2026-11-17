"""
Integration tests with the real Flowcept package (needs Redis: REDIS_HOST / REDIS_PORT):
  1. online: Flowcept, configured with docker/flowcept_settings_online.yaml, publishes tasks on the Redis channel and
     StreamProvenance receives them while the producer is still running;
  2. post-execution: the analysis agent reads a campaign's JSONL files and runs Flowcept's own analysis functions.
Run: python tests/test_pgmem_flowcept.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]
HOST, PORT = os.environ.get("REDIS_HOST", "localhost"), int(os.environ.get("REDIS_PORT", "6379"))


def online_settings(tmp: str) -> str:
    s = open(os.path.join(ROOT, "docker", "flowcept_settings_online.yaml")).read()
    s = s.replace("host: redis", f"host: {HOST}").replace("port: 6379", f"port: {PORT}")
    s = s.replace("path: output/flowcept_buffer.jsonl", f"path: {tmp}/flowcept_buffer.jsonl")
    path = os.path.join(tmp, "settings.yaml")
    open(path, "w").write(s)
    return path


def test_online_stream_receives_tasks_during_the_run():
    from pgmem.provenance import StreamProvenance, smiles_of
    tmp = tempfile.mkdtemp()
    settings = online_settings(tmp)
    src = StreamProvenance(HOST, PORT)
    producer = textwrap.dedent("""
        import time
        from flowcept import Flowcept
        from flowcept.instrumentation.flowcept_task import flowcept_task
        @flowcept_task(subtype="parsl_task")
        def compute_ionization_energy(smiles):
            return {"ionization_energy": 1.0}
        with Flowcept(start_persistence=False, workflow_name="pgmem-test"):
            for s in ["C", "CC", "CCC"]:
                compute_ionization_energy(smiles=s)
            time.sleep(12)             # still running while the subscriber checks what it received
    """)
    proc = subprocess.Popen([sys.executable, "-c", producer], env={**os.environ, "FLOWCEPT_SETTINGS_PATH": settings},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        got, t0 = [], time.time()
        while time.time() - t0 < 10 and len(got) < 3:
            got += src.poll()
            time.sleep(0.5)
        assert proc.poll() is None, "the producer ended before the records arrived: not an online test"
        assert sorted(smiles_of(r) for r in got) == ["C", "CC", "CCC"], got
        print(f"     received {len(got)} tasks {time.time() - t0:.1f} s after start, producer still running")
    finally:
        proc.wait(timeout=60)
        src.close()


def test_post_execution_report_uses_flowcept_analysis():
    from pgmem.analyzer import Analyzer
    from pgmem.config import Config
    from pgmem.llm import AnalysisLLM
    from pgmem.provenance import JsonlProvenance
    from pgmem.store import MemoryStore
    cfg = Config(mode="full", namespace=f"post-{time.time_ns()}", redis_host=HOST, redis_port=PORT)
    store = MemoryStore.connect(HOST, PORT, cfg.namespace)
    src = JsonlProvenance(os.path.join(ROOT, "output", "flowcept_buffer_SC.jsonl"))
    an = Analyzer(store, cfg, llm=AnalysisLLM("scripted"), source=src)
    rep = an.post_execution()
    assert rep["computations"]["total"] == 16 and "provenance_summary" in rep, rep.keys()
    assert rep["provenance_summary"]["n_tasks"] > 0
    store.reset()


if __name__ == "__main__":
    failed = 0
    for t in [test_online_stream_receives_tasks_during_the_run, test_post_execution_report_uses_flowcept_analysis]:
        try:
            t()
            print("ok  ", t.__name__)
        except Exception:  # noqa: BLE001
            failed += 1
            print("FAIL", t.__name__)
            traceback.print_exc()
    sys.exit(1 if failed else 0)
