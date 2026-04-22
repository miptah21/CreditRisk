"""
Model monitoring: PSI drift detection, performance tracking, vintage analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from credit_risk.config import settings


# ---------------------------------------------------------------------------
# PSI — Population Stability Index
# ---------------------------------------------------------------------------

@dataclass
class DriftReport:
    """Result of a drift check."""

    metric: str
    value: float
    status: str  # OK | WARNING | CRITICAL
    details: dict[str, Any]


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Population Stability Index between reference and production distributions.

    PSI < 0.10  → stable
    PSI 0.10–0.25 → moderate drift (investigate)
    PSI > 0.25  → significant drift (retrain)
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    def _bucket_dist(arr: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(arr, bins=breakpoints)
        dist = counts / len(arr)
        return np.clip(dist, 1e-6, None)  # avoid log(0)

    exp_dist = _bucket_dist(expected)
    act_dist = _bucket_dist(actual)

    psi = float(np.sum((act_dist - exp_dist) * np.log(act_dist / exp_dist)))
    return psi


def check_score_drift(
    train_scores: np.ndarray,
    production_scores: np.ndarray,
) -> DriftReport:
    """Check PSI between training score distribution and recent production scores."""
    psi = compute_psi(train_scores, production_scores)
    cfg = settings.monitoring

    if psi >= cfg.psi_critical:
        status = "CRITICAL"
        logger.error(f"PSI = {psi:.4f} — CRITICAL drift detected. Retrain required.")
    elif psi >= cfg.psi_warn:
        status = "WARNING"
        logger.warning(f"PSI = {psi:.4f} — moderate drift. Investigate feature pipelines.")
    else:
        status = "OK"
        logger.info(f"PSI = {psi:.4f} — stable.")

    return DriftReport(
        metric="PSI",
        value=psi,
        status=status,
        details={
            "train_mean": float(train_scores.mean()),
            "prod_mean": float(production_scores.mean()),
            "train_std": float(train_scores.std()),
            "prod_std": float(production_scores.std()),
        },
    )


# ---------------------------------------------------------------------------
# CSI — Characteristic (Feature) Stability Index
# ---------------------------------------------------------------------------

def check_feature_drift(
    train_features: pd.DataFrame,
    production_features: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> list[DriftReport]:
    """Compute PSI per feature to detect upstream data pipeline issues."""
    cols = feature_names or list(train_features.columns)
    reports = []

    for col in cols:
        if col not in train_features.columns or col not in production_features.columns:
            continue
        train_vals = train_features[col].dropna().values
        prod_vals = production_features[col].dropna().values

        if len(train_vals) < 10 or len(prod_vals) < 10:
            continue

        psi = compute_psi(train_vals, prod_vals)
        status = "CRITICAL" if psi > 0.25 else ("WARNING" if psi > 0.10 else "OK")

        reports.append(DriftReport(
            metric=f"CSI_{col}",
            value=psi,
            status=status,
            details={"feature": col},
        ))

    drifted = [r for r in reports if r.status != "OK"]
    if drifted:
        logger.warning(f"{len(drifted)} features drifted: {[r.details['feature'] for r in drifted]}")

    return reports


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_discrimination_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """AUC, Gini, KS statistic."""
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1

    # KS statistic
    ks_stat, p_val = stats.ks_2samp(
        y_prob[y_true == 0], y_prob[y_true == 1]
    )
    # Manual KS: max separation between CDFs
    sorted_probs = np.sort(y_prob)
    n = len(sorted_probs)
    cdf_good = np.array([(y_true[y_prob <= t] == 0).sum() / (y_true == 0).sum() for t in sorted_probs])
    cdf_bad = np.array([(y_true[y_prob <= t] == 1).sum() / (y_true == 1).sum() for t in sorted_probs])
    ks = float(np.max(np.abs(cdf_bad - cdf_good)))

    return {"auc": auc, "gini": gini, "ks": ks}


def compute_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Expected Calibration Error + reliability diagram data."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_means = []
    bin_actuals = []
    bin_counts = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        count = mask.sum()
        if count > 0:
            bin_means.append(float(y_prob[mask].mean()))
            bin_actuals.append(float(y_true[mask].mean()))
            bin_counts.append(int(count))

    # ECE
    total = sum(bin_counts)
    ece = sum(
        (c / total) * abs(pred - actual)
        for c, pred, actual in zip(bin_counts, bin_means, bin_actuals)
    )

    return {
        "ece": ece,
        "bin_predicted": bin_means,
        "bin_actual": bin_actuals,
        "bin_counts": bin_counts,
    }


# ---------------------------------------------------------------------------
# Vintage tracking
# ---------------------------------------------------------------------------

def track_vintage_curve(
    origination_month: str,
    loan_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Track actual vs. predicted default rates for a cohort at 3/6/9/12 months.

    Parameters
    ----------
    origination_month : str
        Format 'YYYY-MM'.
    loan_data : DataFrame
        Must contain: origination_month, first_default_date, pd_at_origination.
    """
    cohort = loan_data[loan_data["origination_month"] == origination_month]
    if cohort.empty:
        return pd.DataFrame()

    results = []
    for mob in [3, 6, 9, 12]:
        cutoff = pd.Timestamp(origination_month) + pd.DateOffset(months=mob)
        defaulted = cohort[
            cohort["first_default_date"].notna()
            & (cohort["first_default_date"] <= cutoff)
        ]
        actual_dr = len(defaulted) / len(cohort) if len(cohort) > 0 else 0
        predicted_pd = cohort["pd_at_origination"].mean()

        results.append({
            "origination_month": origination_month,
            "months_on_book": mob,
            "actual_default_rate": actual_dr,
            "predicted_pd": predicted_pd,
            "overestimate_ratio": predicted_pd / (actual_dr + 1e-6),
        })

    return pd.DataFrame(results)
