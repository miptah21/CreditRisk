"""Fairness module public API."""

from credit_risk.fairness.audit import (
    FairnessReport,
    compute_disparate_impact,
    compute_equal_opportunity_gap,
    run_fairness_audit,
)

__all__ = [
    "FairnessReport",
    "compute_disparate_impact",
    "compute_equal_opportunity_gap",
    "run_fairness_audit",
]
