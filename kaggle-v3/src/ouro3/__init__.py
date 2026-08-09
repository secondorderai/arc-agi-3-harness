"""ARC-AGI-3 kaggle-v3 hybrid harness.

The package layers deterministic perception, memory, verification, scheduling,
and failure handling over the attributed Duck/TAAF direct-Arcade harness.
"""

from ouro3.config import HarnessConfig, RuntimeProfile
from ouro3.ledger import HypothesisLedger
from ouro3.perception import analyze_frame, analyze_transition

__all__ = [
    "HarnessConfig",
    "HypothesisLedger",
    "RuntimeProfile",
    "analyze_frame",
    "analyze_transition",
]
