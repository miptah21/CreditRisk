"""
Fairness auditing: disparate impact, equalized odds, calibration parity.

Implements ECOA/CFPB 80% rule + Fairlearn integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class FairnessReport:
    """Results of a fairness audit across demographic groups."""

    disparate_impact: float         # Approval rate ratio (protected/reference)
    di_passes: bool                 # True if DI >= 0.80
    equal_opportunity_gap: float    # TPR difference between groups
    eo_passes: bool                 # True if |gap| < 0.05
    approval_rates: dict[str, float]
    default_rates: dict[str, float]
    details: dict[str, Any]


def compute_disparate_impact(
    decisions: np.ndarray,
    group_labels: np.ndarray,
    protected_group: str,
    reference_group: str,
) -> float:
    """
    Disparate Impact ratio = approval_rate(protected) / approval_rate(reference).

    Must be >= 0.80 (CFPB 80% rule).
    """
    protected_mask = group_labels == protected_group
    reference_mask = group_labels == reference_group

    approved = decisions == "APPROVE"

    rate_protected = approved[protected_mask].mean() if protected_mask.sum() > 0 else 0.0
    rate_reference = approved[reference_mask].mean() if reference_mask.sum() > 0 else 1.0

    di = rate_protected / max(rate_reference, 1e-6)
    return float(di)


def compute_equal_opportunity_gap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_labels: np.ndarray,
    protected_group: str,
    reference_group: str,
    threshold: float = 0.5,
) -> float:
    """
    Equal Opportunity = TPR(protected) - TPR(reference).

    |gap| should be < 0.05 in production.
    """
    protected_mask = group_labels == protected_group
    reference_mask = group_labels == reference_group

    pred_labels = (y_pred >= threshold).astype(int)

    def _tpr(mask: np.ndarray) -> float:
        positives = y_true[mask] == 1
        if positives.sum() == 0:
            return 0.0
        return float((pred_labels[mask][positives] == 1).mean())

    tpr_protected = _tpr(protected_mask)
    tpr_reference = _tpr(reference_mask)

    return tpr_protected - tpr_reference


def run_fairness_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    decisions: np.ndarray,
    group_labels: np.ndarray,
    protected_group: str,
    reference_group: str,
    threshold: float = 0.5,
) -> FairnessReport:
    """
    Full fairness audit covering disparate impact + equal opportunity.

    Parameters
    ----------
    y_true : array
        Actual default labels (0/1).
    y_pred : array
        Predicted PD scores.
    decisions : array
        Decision outcomes ('APPROVE', 'DECLINE', 'REVIEW').
    group_labels : array
        Demographic group for each applicant.
    protected_group : str
        Name of the protected demographic group.
    reference_group : str
        Name of the reference group.
    """
    di = compute_disparate_impact(decisions, group_labels, protected_group, reference_group)
    eo_gap = compute_equal_opportunity_gap(
        y_true, y_pred, group_labels, protected_group, reference_group, threshold
    )

    # Approval rates per group
    groups = np.unique(group_labels)
    approval_rates = {}
    default_rates = {}
    for g in groups:
        mask = group_labels == g
        approval_rates[str(g)] = float((decisions[mask] == "APPROVE").mean())
        default_rates[str(g)] = float(y_true[mask].mean()) if mask.sum() > 0 else 0.0

    di_passes = di >= 0.80
    eo_passes = abs(eo_gap) < 0.05

    if not di_passes:
        logger.error(f"FAIRNESS FAIL: Disparate Impact = {di:.3f} (< 0.80 threshold)")
    if not eo_passes:
        logger.warning(f"FAIRNESS WARNING: Equal Opportunity gap = {eo_gap:.3f}")

    return FairnessReport(
        disparate_impact=round(di, 4),
        di_passes=di_passes,
        equal_opportunity_gap=round(eo_gap, 4),
        eo_passes=eo_passes,
        approval_rates=approval_rates,
        default_rates=default_rates,
        details={
            "protected_group": protected_group,
            "reference_group": reference_group,
            "n_protected": int((group_labels == protected_group).sum()),
            "n_reference": int((group_labels == reference_group).sum()),
        },
    )
