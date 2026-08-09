"""Sparse self-audit sidecar for the Stock Duck control loop."""

from duck_audit.agent import DuckAuditToolAgent
from duck_audit.solver import DuckAuditHarnessSolver

__all__ = ["DuckAuditHarnessSolver", "DuckAuditToolAgent"]
