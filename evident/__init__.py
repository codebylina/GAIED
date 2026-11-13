"""EVIDENT: evidence-grounded shared memory and real-time control for scientific agent teams."""
from .features import canonical_smiles, constraint_violations, features_of, mol_info
from .claims import claims_instruction, extract_claims
from .memory import MODES, EvidentMemory, SimulationFailed, StoppedByDirective
from .verifier import VerifierParams

__all__ = ["MODES", "EvidentMemory", "SimulationFailed", "StoppedByDirective", "VerifierParams",
           "canonical_smiles", "constraint_violations", "features_of", "mol_info",
           "claims_instruction", "extract_claims"]
