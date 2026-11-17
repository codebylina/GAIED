"""
Fault injection for the controlled evaluation (scenarios with known ground truth). Off unless PGMEM_FAULTS is set.

PGMEM_FAULTS is a JSON object, or the path of a JSON file re-read at every computation (edit the file during a run to
change the faults, e.g. for the "procedure becomes ineffective" scenario):
  {"fail_smarts": "<SMARTS>", "fail_message": "SCF not converged (injected)",   # technical failure
   "truncate_smarts": "<SMARTS>", "truncate_steps": 3,                          # incomplete artifact
   "offset": {"<canonical SMILES>": 5.0}}                                        # wrong (contradictory) value
Every injected fault is appended to PGMEM_FAULTS_LOG (JSONL) as ground truth; nothing about it is written into the
provenance or the memory, so the analysis agent cannot see it.
"""
from __future__ import annotations

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


def _match(smarts: str | None, smiles: str) -> bool:
    if not smarts:
        return False
    from rdkit import Chem
    mol, patt = Chem.MolFromSmiles(smiles), Chem.MolFromSmarts(smarts)
    return mol is not None and patt is not None and mol.HasSubstructMatch(patt)


def _log(kind: str, smiles: str, **kw) -> None:
    path = os.environ.get("PGMEM_FAULTS_LOG")
    if path:
        with open(path, "a") as fh:
            fh.write(json.dumps({"t": time.time(), "fault": kind, "smiles": smiles, **kw}) + "\n")


def before(smiles: str, steps: int) -> int:
    """Called before the computation: may raise (injected failure) or shorten the optimization."""
    f = _load()
    if not f:
        return steps
    if _match(f.get("fail_smarts"), smiles):
        _log("failure", smiles, smarts=f["fail_smarts"])
        raise RuntimeError(f.get("fail_message", "SCF not converged (injected)"))
    if _match(f.get("truncate_smarts"), smiles):
        n = int(f.get("truncate_steps", 3))
        _log("truncated", smiles, steps=n)
        return n
    return steps


def after(smiles: str, value: float) -> float:
    """Called after the computation: may shift the value (wrong result)."""
    f = _load()
    if not f or not f.get("offset"):
        return value
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    key = Chem.MolToSmiles(mol) if mol is not None else smiles
    if key in f["offset"]:
        _log("offset", smiles, delta=f["offset"][key])
        return value + float(f["offset"][key])
    return value
