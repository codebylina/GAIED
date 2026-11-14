"""FALDA as the shared memory of the GAIED agents (baseline), plus a measurement recorder."""
from .client import FaldaMemory, campaign_pool, falda_name
from .recorder import Recorder

__all__ = ["FaldaMemory", "Recorder", "campaign_pool", "falda_name"]
