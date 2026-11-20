"""
Fault injection for the controlled evaluation (scenarios with a known ground truth). Off unless PGMEM_FAULTS is set.

PGMEM_FAULTS is a JSON object, or the path of a JSON file re-read at every computation (the evaluation runner rewrites
it during a run for the "procedure becomes ineffective" scenario). Each fault selects molecules by a SMARTS pattern
and/or a deterministic fraction (a hash of the canonical SMILES and a salt, independent of the LLM's proposals):

  {"fail":     {"smarts": "C#N", "fraction": 0.3, "salt": "a", "message": "SCF not converged"},
   "truncate": {"fraction": 0.2, "salt": "b", "steps": 3},          # incomplete artifact: optimization cut short
   "offset":   {"fraction": 0.2, "salt": "c", "value": -3.0}}       # wrong value (contradictory result), in eV

With both "smarts" and "fraction", a molecule must match the pattern and fall in the fraction; with "smarts" only,
every matching molecule is selected. Faults can change during a run, after a number of computations (the same point
in every condition):

  {"phases": [{"until_computations": 20, "fail": {"smarts": "C#N", "message": "SCF not converged"}},
              {"fail": {"smarts": "[CX4](F)(F)F", "message": "SCF not converged"}}]}

Every computation and every injected fault is appended to PGMEM_FAULTS_LOG (JSONL) as ground truth; nothing about it
is written into the provenance or the memory (the failure message is the ordinary xTB one), so neither the agents nor
the analysis agent can see what was injected.
"""
from __future__ import annotations

import hashlib
import json
import os
import time


def _load() -> dict:
    spec = os.environ.get("PGMEM_FAULTS", "").strip()
    if not spec:
        return {}
    if spec.startswith("{"):
        return json.loads(spec)
    try:
        with open(spec) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _canonical(smiles: str) -> str | None:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def selected(rule: dict | None, smiles: str) -> bool:
    """Whether a fault rule applies to this molecule (never to a SMILES RDKit cannot read)."""
    if not rule:
        return False
    key = _canonical(smiles)
    if key is None:
        return False
    if rule.get("smarts"):
        from rdkit import Chem
        patt = Chem.MolFromSmarts(rule["smarts"])
        if patt is None or not Chem.MolFromSmiles(key).HasSubstructMatch(patt):
            return False
    if "fraction" in rule:
        h = int(hashlib.sha1(f"{rule.get('salt', '')}|{key}".encode()).hexdigest(), 16) % 1000
        return h < float(rule["fraction"]) * 1000
    return bool(rule.get("smarts"))


def _log(kind: str, smiles: str, **kw) -> None:
    path = os.environ.get("PGMEM_FAULTS_LOG")
    if path:
        with open(path, "a") as fh:
            fh.write(json.dumps({"t": time.time(), "fault": kind, "smiles": smiles, "key": _canonical(smiles), **kw})
                     + "\n")


def _computations() -> int:
    path = os.environ.get("PGMEM_FAULTS_LOG")
    try:
        with open(path) as fh:
            return sum(1 for line in fh if '"fault": "computation"' in line)
    except (OSError, TypeError):
        return 0


def _active(spec: dict) -> tuple[dict, int]:
    """The faults in force now, and the phase number (0 without phases)."""
    phases = spec.get("phases")
    if not phases:
        return spec, 0
    n = _computations()
    for i, ph in enumerate(phases):
        if "until_computations" not in ph or n < int(ph["until_computations"]):
            return ph, i
    return phases[-1], len(phases) - 1


def before(smiles: str, steps: int) -> int:
    """Called before the computation: may raise (injected failure) or shorten the optimization."""
    f, phase = _active(_load())
    _log("computation", smiles, phase=phase)
    if selected(f.get("fail"), smiles):
        _log("failure", smiles, phase=phase, rule=f["fail"].get("smarts") or f["fail"].get("salt"))
        raise RuntimeError(f["fail"].get("message", "SCF not converged"))
    if selected(f.get("truncate"), smiles):
        n = int(f["truncate"].get("steps", 3))
        _log("truncated", smiles, steps=n, phase=phase)
        return n
    return steps


def after(smiles: str, value: float) -> float:
    """Called after the computation: may shift the value (wrong result)."""
    rule = _active(_load())[0].get("offset")
    if selected(rule, smiles):
        _log("offset", smiles, delta=float(rule.get("value", -3.0)))
        return value + float(rule.get("value", -3.0))
    return value
