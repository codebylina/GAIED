"""
Execution of a test written by the judge, on the observation table, in a separate process.

The judge writes `def test(df): ...` where df is a pandas DataFrame with one row per observation
(columns: id, entity, value, status, agent, t, evidence, plus the domain descriptors). The function
returns a dict:
    {"in_scope": [ids], "support": [ids], "contradict": [ids], "stats": {...}, "note": "..."}
The code runs in a child process (isolated interpreter, restricted builtins and imports, time limit).
This keeps a faulty or slow test from affecting the memory; it is not a security boundary against a
malicious author, which is out of scope for a research prototype.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sandbox_runner.py")


class TestError(Exception):
    pass


def run_test(code: str, rows: list[dict], timeout_s: float) -> dict:
    return validate(_run(code, rows, timeout_s, "test"), {str(r["id"]) for r in rows})


def run_function(code: str, rows: list[dict], fn: str, timeout_s: float):
    """Run any function fn(df) of the code and return its (JSON) result."""
    return _run(code, rows, timeout_s, fn).get("result")


def _run(code: str, rows: list[dict], timeout_s: float, fn: str) -> dict:
    payload = json.dumps({"code": code, "rows": rows, "fn": fn}, default=str)
    try:
        proc = subprocess.run([sys.executable, "-I", RUNNER], input=payload, capture_output=True,
                              text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise TestError(f"test exceeded {timeout_s} s") from exc
    if proc.returncode != 0:
        raise TestError((proc.stderr or proc.stdout or "test failed").strip()[-1500:])
    try:
        out = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        raise TestError(f"unreadable test output: {proc.stdout[-500:]}") from exc
    if "error" in out:
        raise TestError(out["error"])
    return out


def validate(out: dict, known_ids: set[str]) -> dict:
    res = {}
    for key in ("in_scope", "support", "contradict"):
        vals = out.get(key) or []
        if not isinstance(vals, list):
            raise TestError(f"'{key}' must be a list of observation ids")
        ids = [str(v) for v in vals]
        unknown = [i for i in ids if i not in known_ids]
        if unknown:
            raise TestError(f"'{key}' contains unknown observation ids: {unknown[:5]}")
        res[key] = sorted(set(ids))
    overlap = set(res["support"]) & set(res["contradict"])
    if overlap:
        raise TestError(f"observations both support and contradict: {sorted(overlap)[:5]}")
    outside = (set(res["support"]) | set(res["contradict"])) - set(res["in_scope"])
    if outside:
        raise TestError(f"support/contradict ids outside in_scope: {sorted(outside)[:5]}")
    stats = out.get("stats") or {}
    res["stats"] = {str(k): (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                    for k, v in (stats.items() if isinstance(stats, dict) else [])}
    res["note"] = str(out.get("note") or "")[:500]
    return res
