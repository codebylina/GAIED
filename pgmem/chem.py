"""Domain helpers for the GAIED case study: molecule identity and classification of technical failures."""
from __future__ import annotations

import math
import re

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:  # pragma: no cover
    Chem = None


def canonical(smiles: str) -> str:
    """Canonical SMILES (RDKit); the raw string, stripped, when RDKit cannot read it."""
    s = str(smiles).strip()
    mol = Chem.MolFromSmiles(s) if Chem is not None and s else None
    return Chem.MolToSmiles(mol) if mol is not None else s


def parses(smiles: str) -> bool:
    return Chem is not None and Chem.MolFromSmiles(str(smiles).strip()) is not None


# Deterministic classification of the error messages GAIED's xTB pipeline produces
# (mol_design_agents._generate_initial_xyz, RDKit embedding, xTB / ASE).
_FAILURE_RULES = (
    ("invalid_smiles", re.compile(r"parse failure", re.I)),
    ("embedding_failure", re.compile(r"bad conformer id|embed", re.I)),
    ("scf_not_converged", re.compile(r"scf|self.consistent|not converg", re.I)),
    ("timeout", re.compile(r"timed? ?out", re.I)),
)


def failure_class(message: str) -> str:
    for name, rx in _FAILURE_RULES:
        if rx.search(message or ""):
            return name
    return "other"


def confirm_cause(fclass: str, smiles: str) -> bool:
    """A cause is confirmed only by a deterministic check: here, RDKit itself cannot read the SMILES."""
    return fclass == "invalid_smiles" and not parses(smiles)


def data_quality_issues(value, conditions: dict) -> list[str]:
    """Data-quality problems of a result, read from its own conditions (no scientific threshold)."""
    issues = []
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        issues.append("missing or non-finite value")
    if conditions.get("converged") is False:
        issues.append(f"geometry optimization did not converge ({conditions.get('opt_steps')} steps)")
    return issues
