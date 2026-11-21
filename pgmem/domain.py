"""
The domain interface: what the memory needs to know about the scientific entities it stores. Everything else in
pgmem (graph, retrieval, analysis) is written against it, so another domain (materials, proteins...) only needs
another implementation of these few functions.

  canonical(x)     one spelling per entity (the node id of the entity in the graph)
  valid(x)         whether x is a well-formed entity
  entities_in(t)   the entities written in a text
  motifs(x)        the characteristic motifs of an entity (motif nodes of the graph; generic ones left out)
  similarity(a,b)  structural similarity in [0, 1]

The GAIED case study is chemistry: molecules (SMILES), functional groups (SMARTS), Tanimoto similarity.
"""
from __future__ import annotations

from . import chem


class Chemistry:
    name = "chemistry"
    motif_vocabulary = chem.FEATURES               # motif name -> SMARTS

    def canonical(self, smiles: str) -> str:
        return chem.canonical(smiles)

    def valid(self, smiles: str) -> bool:
        return chem.parses(smiles)

    def entities_in(self, text: str) -> list[str]:
        return chem.molecules_in(text)

    def motifs(self, smiles: str) -> set[str]:
        return chem.features(smiles) - chem.GENERIC

    def motifs_named_in(self, text: str) -> set[str]:
        """Motifs a statement names in words ("nitrile", "trifluoromethyl group")."""
        low = (text or "").lower()
        return {m for m in chem.FEATURES if m not in chem.GENERIC and m.replace("_", " ") in low}

    def similarity(self, a: str, b: str) -> float:
        return chem.similarity(a, b)


DOMAIN = Chemistry()
