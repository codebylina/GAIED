"""Shared memory: coordination, memory and verification for teams of
scientific agents, on one Redis instance, linked to Flowcept provenance by ids."""
from .coordination import Coordinator
from .domain import SCIENTIFIC, TECHNICAL, Domain
from .lessons import LessonMemory
from .core import Memory, KnownFailure
from .observations import ExperimentalMemory, condition_id
from .proposer import LessonProposer
from .verification import Verifier

__all__ = ["Memory", "KnownFailure", "Domain", "SCIENTIFIC", "TECHNICAL", "Coordinator", "ExperimentalMemory",
           "LessonMemory", "Verifier", "LessonProposer", "condition_id"]
