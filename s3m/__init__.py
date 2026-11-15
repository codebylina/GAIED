"""S3M — Simple Shared Scientific Memory: coordination, memory and verification for teams of
scientific agents, on one Redis instance, linked to Flowcept provenance by ids."""
from .coordination import Coordinator
from .domain import SCIENTIFIC, TECHNICAL, Domain
from .lessons import LessonMemory
from .memory import S3M, KnownFailure
from .observations import ExperimentalMemory, condition_id
from .proposer import LessonProposer
from .verification import Verifier

__all__ = ["S3M", "KnownFailure", "Domain", "SCIENTIFIC", "TECHNICAL", "Coordinator", "ExperimentalMemory",
           "LessonMemory", "Verifier", "LessonProposer", "condition_id"]
