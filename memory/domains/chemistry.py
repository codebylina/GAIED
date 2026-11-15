"""
Chemistry domain: molecules given as SMILES, an objective computed by a simulation (e.g. xTB).

Entities are canonical SMILES (RDKit), so two spellings of one molecule are one entity. Features are
standard structural counts (heavy atoms, rings, atoms per element, functional groups by SMARTS); they
are the vocabulary of "feature_effect" lessons, e.g. "molecules with a nitrile group (n_nitrile > 0)
have a higher ionization energy".
"""
from __future__ import annotations

from functools import lru_cache

from ..domain import SCIENTIFIC, TECHNICAL, Domain

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
except ImportError:  # pragma: no cover - RDKit is required for canonical keys and features
    Chem = None

GROUPS = {
    "nitrile": "[CX2]#[NX1]", "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]", "amide": "[CX3](=[OX1])[NX3]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]", "ester": "[CX3](=O)[OX2][#6]", "carbonyl": "[CX3]=[OX1]",
    "amine": "[NX3;!$(NC=[O,S,N]);!$(N-a);!$(N=*);!$([N+])]", "hydroxyl": "[OX2H][#6;!$(C=O)]",
    "ether": "[OD2;!$(OC=O)]([#6])[#6]", "aromatic_nitrogen": "[n]", "alkyne": "[CX2]#[CX2]",
    "alkene": "[CX3;!a]=[CX3;!a]", "trifluoromethyl": "[CX4](F)(F)F", "halogen": "[F,Cl,Br,I]",
}
ELEMENTS = ("C", "N", "O", "F", "S", "Cl", "Br", "P")

# Errors that come from the molecule itself: the same experiment would fail again.
_SCIENTIFIC_ERRORS = ("parse failure", "sanitiz", "kekulize", "valence", "bad conformer", "embed",
                      "not converge", "convergence")


@lru_cache(maxsize=None)
def _pattern(smarts: str):
    return Chem.MolFromSmarts(smarts)


@lru_cache(maxsize=100_000)
def _canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else None


@lru_cache(maxsize=100_000)
def _features(smiles: str) -> tuple:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ()
    f = {"heavy_atoms": mol.GetNumHeavyAtoms(), "n_rings": rdMolDescriptors.CalcNumRings(mol),
         "n_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol)}
    counts: dict[str, int] = {}
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
    for e in ELEMENTS:
        f[f"n_{e}"] = counts.get(e, 0)
    for name, smarts in GROUPS.items():
        f[f"n_{name}"] = len(mol.GetSubstructMatches(_pattern(smarts)))
    return tuple(f.items())


class ChemistryDomain(Domain):
    name = "molecule"
    input_key = "smiles"

    def __init__(self, objective: str = "ionization_energy", direction: str = "max", method: str = "xtb",
                 activity: str | None = "compute_ionization_energy", params_key: str | None = "xtb_config",
                 default_params: dict | None = None) -> None:
        if Chem is None:
            raise ImportError("ChemistryDomain needs RDKit")
        self.objective, self.output_key, self.direction = objective, objective, direction
        self.method, self.activity, self.params_key = method, activity, params_key
        self.default_params = default_params

    def canonical(self, raw: str) -> str | None:
        return _canonical(str(raw).strip())

    def features(self, entity: str) -> dict[str, float]:
        return dict(_features(entity)) if not entity.startswith("invalid:") else {}

    def feature_doc(self) -> str:
        return ("heavy_atoms, n_rings, n_aromatic_rings; "
                + ", ".join(f"n_{e}" for e in ELEMENTS) + " (atoms per element); "
                + ", ".join(f"n_{g}" for g in GROUPS) + " (number of functional groups)")

    def classify_failure(self, error: str) -> str:
        e = (error or "").lower()
        return SCIENTIFIC if any(s in e for s in _SCIENTIFIC_ERRORS) else TECHNICAL
