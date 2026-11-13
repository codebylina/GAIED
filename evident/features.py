"""
Closed vocabulary of chemical features.

EVIDENT never stores a free-text "lesson" as knowledge. A claim must be about one
feature of this vocabulary, so that it can be checked by code against the measured
facts (a SMARTS pattern decides, for every computed molecule, whether the feature is
present). The keywords are only used to recognize the feature in free text written
by the LLM when it does not return the structured CLAIMS block.
"""
from __future__ import annotations

from functools import lru_cache

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:  # pragma: no cover - RDKit is installed in the Docker image
    Chem = None

# name -> (SMARTS, short description, keywords searched in LLM text)
# Order matters for keyword matching: more specific names come first.
FEATURES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "trifluoromethyl": ("[CX4](F)(F)F", "CF3 group", ("trifluoromethyl", "cf3")),
    "fluorine": ("[F]", "fluorine atom", ("fluorin", "fluoro", "fluoride")),
    "chlorine": ("[Cl]", "chlorine atom", ("chlorin", "chloro")),
    "bromine": ("[Br]", "bromine atom", ("bromin", "bromo")),
    "nitrile": ("[CX2]#[NX1]", "nitrile (C#N)", ("nitrile", "cyano")),
    "nitro": ("[$([NX3](=O)=O),$([NX3+](=O)[O-])]", "nitro group", ("nitro group", "nitro")),
    "urea": ("[NX3][CX3](=[OX1])[NX3]", "urea motif", ("urea",)),
    "amide": ("[CX3](=[OX1])[NX3]", "amide", ("amide",)),
    "carboxylic_acid": ("[CX3](=O)[OX2H1]", "carboxylic acid", ("carboxylic", "carboxyl")),
    "ester": ("[CX3](=O)[OX2][#6]", "ester", ("ester",)),
    "carbonyl": ("[CX3]=[OX1]", "carbonyl (C=O)", ("carbonyl", "ketone", "aldehyde")),
    "primary_amine": ("[NX3;H2;!$(NC=[O,S,N])]", "primary amine (NH2)", ("primary amine", "amino group")),
    "amine": ("[NX3;!$(NC=[O,S,N]);!$(N-a);!$(N=*);!$([N+])]", "aliphatic amine", ("amine",)),
    "hydroxyl": ("[OX2H][#6;!$(C=O)]", "hydroxyl (OH)", ("hydroxyl", "alcohol")),
    "ether": ("[OD2;!$(OC=O)]([#6])[#6]", "ether", ("ether",)),
    "aromatic_nitrogen": ("[n]", "aromatic nitrogen", ("pyridin", "pyrimidin", "azine", "heteroaromatic", "imidazol")),
    "aromatic_ring": ("[a]", "aromatic ring", ("aromatic", "phenyl", "benzene", "aryl")),
    "alkyne": ("[CX2]#[CX2]", "C#C triple bond", ("alkyne", "triple bond", "ethynyl")),
    "alkene": ("[CX3;!a]=[CX3;!a]", "C=C double bond", ("alkene", "vinyl", "olefin")),
    "sulfur": ("[#16]", "sulfur atom", ("sulfur", "sulphur", "thio", "sulfon")),
    "aliphatic_ring": ("[R;!a]", "non-aromatic ring", ("cyclo", "aliphatic ring", "saturated ring")),
    "methyl": ("[CH3]", "methyl group", ("methyl", "alkyl")),
}

FEATURE_NAMES = tuple(FEATURES)


@lru_cache(maxsize=None)
def _pattern(name: str):
    return Chem.MolFromSmarts(FEATURES[name][0])


@lru_cache(maxsize=100_000)
def canonical_smiles(smiles: str) -> str:
    """RDKit canonical SMILES (CCO and OCC give the same key); input unchanged if invalid."""
    smiles = (smiles or "").strip()
    if Chem is None:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else smiles


@lru_cache(maxsize=100_000)
def mol_info(smiles: str) -> dict:
    """Validity, features, heavy-atom count and elements of a molecule (cached)."""
    if Chem is None:
        return {"valid": True, "features": [], "heavy_atoms": None, "elements": []}
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return {"valid": False, "features": [], "heavy_atoms": None, "elements": []}
    feats = [n for n in FEATURE_NAMES if mol.HasSubstructMatch(_pattern(n))]
    elements = sorted({a.GetSymbol() for a in mol.GetAtoms()})
    return {"valid": True, "features": feats, "heavy_atoms": mol.GetNumHeavyAtoms(),
            "elements": elements}


def features_of(smiles: str) -> list[str]:
    return mol_info(smiles)["features"]


def describe(name: str) -> str:
    return FEATURES[name][1] if name in FEATURES else name


def vocabulary_for_prompt() -> str:
    return ", ".join(f"{n} ({FEATURES[n][1]})" for n in FEATURE_NAMES)


def constraint_violations(smiles: str, constraints: dict | None) -> list[str]:
    """Machine-checkable Goal constraints: validity, size and allowed elements."""
    info = mol_info(smiles)
    if not info["valid"]:
        return ["invalid_smiles"]
    out = []
    constraints = constraints or {}
    max_heavy = constraints.get("max_heavy_atoms")
    if max_heavy and info["heavy_atoms"] is not None and info["heavy_atoms"] > int(max_heavy):
        out.append(f"too_large({info['heavy_atoms']}>{max_heavy}_heavy_atoms)")
    allowed = constraints.get("allowed_elements")
    if allowed:
        bad = [e for e in info["elements"] if e not in allowed]
        if bad:
            out.append(f"forbidden_elements({','.join(bad)})")
    return out
