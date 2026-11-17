"""
Chemistry domain for GAIED: molecules as SMILES, an objective computed by xTB.

Entities are canonical SMILES (RDKit). Features are structural counts (heavy atoms, rings, atoms per
element, functional groups by SMARTS); similarity is the Tanimoto similarity of Morgan fingerprints.
"""
from __future__ import annotations

from functools import lru_cache

from ..domain import PERMANENT, SCIENTIFIC, TRANSIENT, Domain

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
except ImportError:  # pragma: no cover
    Chem = None

GROUPS = {
    "nitrile": "[CX2]#[NX1]", "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]", "amide": "[CX3](=[OX1])[NX3]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]", "ester": "[CX3](=O)[OX2][#6]", "carbonyl": "[CX3]=[OX1]",
    "amine": "[NX3;!$(NC=[O,S,N]);!$(N-a);!$(N=*);!$([N+])]", "hydroxyl": "[OX2H][#6;!$(C=O)]",
    "ether": "[OD2;!$(OC=O)]([#6])[#6]", "aromatic_nitrogen": "[n]", "alkyne": "[CX2]#[CX2]",
    "alkene": "[CX3;!a]=[CX3;!a]", "trifluoromethyl": "[CX4](F)(F)F", "halogen": "[F,Cl,Br,I]",
}
ELEMENTS = ("C", "N", "O", "F", "S", "Cl", "Br", "P")
_SCIENTIFIC = ("parse failure", "sanitiz", "kekulize", "valence", "bad conformer", "embed", "not converge",
               "convergence")
_PERMANENT = ("no module named", "not found", "permission denied", "importerror", "modulenotfound")


@lru_cache(maxsize=None)
def _pattern(smarts):
    return Chem.MolFromSmarts(smarts)


@lru_cache(maxsize=100_000)
def _canonical(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else None


@lru_cache(maxsize=100_000)
def _features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ()
    f = {"heavy_atoms": mol.GetNumHeavyAtoms(), "n_rings": rdMolDescriptors.CalcNumRings(mol),
         "n_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol)}
    counts: dict = {}
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
    for e in ELEMENTS:
        f[f"n_{e}"] = counts.get(e, 0)
    for name, smarts in GROUPS.items():
        f[f"n_{name}"] = len(mol.GetSubstructMatches(_pattern(smarts)))
    return tuple(f.items())


_FPGEN = None


@lru_cache(maxsize=100_000)
def _fingerprint(smiles):
    global _FPGEN
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if _FPGEN is None:
        _FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    return _FPGEN.GetFingerprint(mol)


class ChemistryDomain(Domain):
    name = "molecule"
    input_key = "smiles"

    def __init__(self, objective: str = "ionization_energy", unit: str = "Ha", direction: str = "max",
                 method: str = "xtb", activity: str | None = "compute_ionization_energy",
                 params_key: str | None = "xtb_config", default_params: dict | None = None) -> None:
        if Chem is None:
            raise ImportError("ChemistryDomain needs RDKit")
        self.objective, self.output_key, self.unit, self.direction = objective, objective, unit, direction
        self.method, self.activity, self.params_key, self.default_params = method, activity, params_key, default_params

    def canonical(self, raw):
        return _canonical(str(raw).strip())

    def features(self, entity):
        return dict(_features(entity)) if not entity.startswith("invalid:") else {}

    def feature_doc(self):
        return ("heavy_atoms, n_rings, n_aromatic_rings; " + ", ".join(f"n_{e}" for e in ELEMENTS)
                + " (atoms per element); " + ", ".join(f"n_{g}" for g in GROUPS) + " (functional groups)")

    def similarity(self, a, b):
        fa, fb = _fingerprint(a), _fingerprint(b)
        if fa is None or fb is None:
            return 1.0 if a == b else 0.0
        return float(DataStructs.TanimotoSimilarity(fa, fb))

    def classify_failure(self, error):
        e = (error or "").lower()
        if any(s in e for s in _SCIENTIFIC):
            return SCIENTIFIC
        if any(s in e for s in _PERMANENT):
            return PERMANENT
        return TRANSIENT
