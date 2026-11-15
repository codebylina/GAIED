"""
Chemistry adapter for GAIED: molecules proposed as SMILES, xTB ionization energy as the objective.

Descriptors are standard structural counts computed with RDKit (atoms per element, rings, aromatic
rings, heavy atoms, functional groups from SMARTS). They are the vocabulary in which the judge writes
the scope and the test of a claim; the judge is free to combine them as the claim requires.
"""
from __future__ import annotations

from functools import lru_cache

from memlab.adapter import Goal, Objective, ProvenanceTaskAdapter

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
except ImportError:  # pragma: no cover - RDKit is in the GAIED image
    Chem = None

GROUPS = {
    "nitrile": "[CX2]#[NX1]", "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]", "urea": "[NX3][CX3](=[OX1])[NX3]",
    "amide": "[CX3](=[OX1])[NX3]", "carboxylic_acid": "[CX3](=O)[OX2H1]", "ester": "[CX3](=O)[OX2][#6]",
    "carbonyl": "[CX3]=[OX1]", "amine": "[NX3;!$(NC=[O,S,N]);!$(N-a);!$(N=*);!$([N+])]",
    "hydroxyl": "[OX2H][#6;!$(C=O)]", "ether": "[OD2;!$(OC=O)]([#6])[#6]", "aromatic_nitrogen": "[n]",
    "alkyne": "[CX2]#[CX2]", "alkene": "[CX3;!a]=[CX3;!a]", "trifluoromethyl": "[CX4](F)(F)F",
    "halogen": "[F,Cl,Br,I]", "methyl": "[CH3]",
}
ELEMENTS = ("C", "N", "O", "F", "S", "Cl", "Br", "P")


@lru_cache(maxsize=None)
def _pat(smarts: str):
    return Chem.MolFromSmarts(smarts)


class ChemistryAdapter(ProvenanceTaskAdapter):
    entity_noun = "molecule"
    input_key = "smiles"
    output_key = "ionization_energy"
    activity = "compute_ionization_energy"

    def __init__(self, goal_text: str, max_heavy_atoms: int | None = None,
                 allowed_elements: list[str] | None = None) -> None:
        self.max_heavy_atoms, self.allowed_elements = max_heavy_atoms, allowed_elements
        constraints = []
        if max_heavy_atoms:
            constraints.append(f"at most {max_heavy_atoms} heavy atoms")
        if allowed_elements:
            constraints.append("only elements " + ", ".join(allowed_elements))
        self.goal = Goal(Objective("ionization_energy", "max", "Ha"), goal_text, constraints)

    @lru_cache(maxsize=100_000)
    def canonical(self, raw: str) -> str | None:
        if Chem is None:
            return raw.strip() or None
        mol = Chem.MolFromSmiles((raw or "").strip())
        return Chem.MolToSmiles(mol) if mol is not None else None

    @lru_cache(maxsize=100_000)
    def _desc(self, smiles: str) -> tuple:
        mol = Chem.MolFromSmiles(smiles)
        d = {"heavy_atoms": mol.GetNumHeavyAtoms(), "n_rings": rdMolDescriptors.CalcNumRings(mol),
             "n_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol)}
        counts = {}
        for a in mol.GetAtoms():
            counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
        for e in ELEMENTS:
            d[f"n_{e}"] = counts.get(e, 0)
        d["n_other_elements"] = sum(v for k, v in counts.items() if k not in ELEMENTS)
        for name, smarts in GROUPS.items():
            d[f"n_{name}"] = len(mol.GetSubstructMatches(_pat(smarts)))
        return tuple(sorted(d.items()))

    def descriptors(self, entity):
        if Chem is None or Chem.MolFromSmiles(entity) is None:
            return {}
        return dict(self._desc(entity))

    def descriptor_doc(self):
        return ("heavy_atoms, n_rings, n_aromatic_rings (int)\n"
                + "".join(f"n_{e} (number of {e} atoms)\n" for e in ELEMENTS)
                + "n_other_elements\n"
                + "".join(f"n_{g} (number of {g} groups, SMARTS {s})\n" for g, s in GROUPS.items()))

    def violations(self, entity):
        d = self.descriptors(entity)
        if not d:
            return ["invalid"]
        out = []
        if self.max_heavy_atoms and d["heavy_atoms"] > self.max_heavy_atoms:
            out.append(f"more than {self.max_heavy_atoms} heavy atoms")
        if self.allowed_elements:
            mol = Chem.MolFromSmiles(entity)
            bad = sorted({a.GetSymbol() for a in mol.GetAtoms()} - set(self.allowed_elements))
            if bad:
                out.append("forbidden elements " + ",".join(bad))
        return out
