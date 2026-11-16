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


def run_many(codes: dict[str, str], rows: list[dict], timeout_s: float) -> dict[str, dict]:
    """Run several tests on the same table in one child process. Returns, per key, either the
    validated result or {"error": "..."}; one failing test does not affect the others."""
    if not codes:
        return {}
    payload = json.dumps({"codes": codes, "rows": rows, "fn": "test"}, default=str)
    try:
        proc = subprocess.run([sys.executable, "-I", RUNNER], input=payload, capture_output=True,
                              text=True, timeout=timeout_s)
        raw = json.loads(proc.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {k: {"error": f"tests exceeded {timeout_s} s"} for k in codes}
    except Exception as exc:
        return {k: {"error": f"sandbox failure: {exc}"} for k in codes}
    known = {str(r["id"]) for r in rows}
    by_entity: dict[str, list[str]] = {}
    for r in rows:
        by_entity.setdefault(str(r.get("entity")), []).append(str(r["id"]))
    out = {}
    for k in codes:
        r = raw.get(k) or {"error": "no result"}
        if "error" in r:
            out[k] = r
            continue
        try:
            v = validate(r, known, by_entity)
            v["verdict"] = str(r.get("verdict", "")).lower()
            out[k] = v
        except TestError as exc:
            out[k] = {"error": str(exc)}
    return out


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


def validate(out: dict, known_ids: set[str], by_entity: dict[str, list[str]] | None = None) -> dict:
    """Check a test's answer. Two forms keep their meaning and are normalized: a design key in place of its
    measurement ids (all the measurements of that design), and support/contradict ids missing from in_scope
    (added to it). Anything else that is not a measurement id is an error."""
    by_entity = by_entity or {}
    res = {}
    for key in ("in_scope", "support", "contradict"):
        vals = out.get(key) or []
        if not isinstance(vals, list):
            raise TestError(f"'{key}' must be a list of values of df['id']")
        ids = []
        for v in (str(x) for x in vals):
            if v in known_ids:
                ids.append(v)
            elif v in by_entity:
                ids.extend(by_entity[v])
            else:
                raise TestError(f"'{key}' contains {v!r}, which is not a value of df['id']: list the ids of the "
                                f"measurement rows (df['id']), never the memory's id")
        res[key] = sorted(set(ids))
    overlap = set(res["support"]) & set(res["contradict"])
    if overlap:
        raise TestError(f"observations both support and contradict: {sorted(overlap)[:5]}")
    res["in_scope"] = sorted(set(res["in_scope"]) | set(res["support"]) | set(res["contradict"]))
    stats = out.get("stats") or {}
    res["stats"] = {str(k): (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                    for k, v in (stats.items() if isinstance(stats, dict) else [])}
    res["note"] = str(out.get("note") or "")[:500]
    return res
