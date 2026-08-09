"""Locked Duck reference execution path.

The underlying MIT source is vendored in :mod:`inference`.  These named
reference types make it impossible for the runner to select the hybrid
analyzer or scheduler accidentally.
"""

from duck_reference.agent import DuckReferenceToolAgent
from duck_reference.solver import DuckReferenceHarnessSolver

__all__ = ["DuckReferenceHarnessSolver", "DuckReferenceToolAgent"]
