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


# ---------------------------------------------------------------------------------------------- structure
# A closed vocabulary of functional groups and structural motifs, used to compare molecules (retrieval), to state
# motif effects (claim verification) and to judge retrieval relevance (evaluation). Names are what the LLM may use.
FEATURES: dict[str, str] = {
    "nitrile": "C#N",
    "trifluoromethyl": "[CX4](F)(F)F",
    "difluoromethylene": "[CX4;!$(C(F)(F)F)](F)F",
    "fluorine": "[F]",
    "chlorine": "[Cl]",
    "isocyanate": "N=C=O",
    "carbonyl": "[CX3]=[OX1]",
    "amide": "[NX3][CX3]=[OX1]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "hydroxyl": "[OX2H]",
    "ether": "[OD2]([#6])[#6]",
    "amine": "[NX3;H2,H1,H0;!$(NC=O);!$(N=*);!$(N#*)]",
    "imine": "[CX3]=[NX2]",
    "alkyne": "C#C",
    "alkene": "C=C",
    "aromatic_ring": "a",
    "ring": "[R]",
    "heteroaromatic_nitrogen": "n",
}
_PATTERNS = None


def features(smiles: str) -> set[str]:
    """Names of the FEATURES present in a molecule (empty when RDKit cannot read it)."""
    global _PATTERNS
    if Chem is None:
        return set()
    if _PATTERNS is None:
        _PATTERNS = {n: Chem.MolFromSmarts(s) for n, s in FEATURES.items()}
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return set()
    return {n for n, p in _PATTERNS.items() if p is not None and mol.HasSubstructMatch(p)}


_FP_CACHE: dict[str, object] = {}


def similarity(a: str, b: str) -> float:
    """Tanimoto similarity of Morgan fingerprints (radius 2); 0 when a molecule cannot be read."""
    if Chem is None:
        return 0.0
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    fps = []
    for s in (a, b):
        if s not in _FP_CACHE:
            mol = Chem.MolFromSmiles(str(s).strip())
            _FP_CACHE[s] = (rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024).GetFingerprint(mol)
                            if mol is not None else None)
        fps.append(_FP_CACHE[s])
    if fps[0] is None or fps[1] is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fps[0], fps[1]))


_SMILES_TOKEN = re.compile(r"[A-Za-z0-9@+\-\[\]()=#$/\\%.]{2,}")
_PLAIN_WORD = re.compile(r"^[A-Za-z][a-z]+$")


def molecules_in(text: str, limit: int = 20) -> list[str]:
    """Canonical SMILES of the molecules written in a text (tokens RDKit reads, ordinary words left out)."""
    out: list[str] = []
    if Chem is None:
        return out
    for tok in _SMILES_TOKEN.findall(text or ""):
        tok = tok.strip(".,;:()")
        if len(tok) < 2 or _PLAIN_WORD.match(tok) or tok.isdigit():
            continue
        mol = Chem.MolFromSmiles(tok)
        if mol is None or mol.GetNumHeavyAtoms() < 2:
            continue
        key = Chem.MolToSmiles(mol)
        if key not in out:
            out.append(key)
        if len(out) >= limit:
            break
    return out
