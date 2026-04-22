"""
Point-in-time feature extraction from transaction and bureau data.

Every feature is computed using ONLY data available at or before the
application date — the core guard against temporal leakage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Cashflow features (highest alpha for thin-file borrowers)
# ---------------------------------------------------------------------------

def extract_cashflow_features(
    transactions: pd.DataFrame,
    application_date: pd.Timestamp,
    applicant_id: str,
) -> dict[str, Any]:
    """
    Extract cash-flow features from bank transaction history.

    Parameters
    ----------
    transactions : DataFrame
        Must contain columns: applicant_id, date, amount, balance, is_nsf, category.
    application_date : Timestamp
        Strict cutoff — only data BEFORE this date is used.
    applicant_id : str
        Applicant identifier.

    Returns
    -------
    dict of feature_name → value
    """
    history = transactions[
        (transactions["applicant_id"] == applicant_id)
        & (transactions["date"] < application_date)  # strict less-than
    ].copy()

    if history.empty:
        logger.warning(f"No transaction history for {applicant_id} before {application_date}")
        return _cold_start_cashflow_defaults()

    w1m = application_date - pd.DateOffset(months=1)
    w3m = application_date - pd.DateOffset(months=3)
    w6m = application_date - pd.DateOffset(months=6)
    w12m = application_date - pd.DateOffset(months=12)

    def _window(df: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
        return df[df["date"] >= start]

    credits = history[history["amount"] > 0]
    debits = history[history["amount"] < 0]

    # Monthly inflow aggregates
    inflow_3m = _window(credits, w3m)["amount"].sum()
    inflow_6m = _window(credits, w6m)["amount"].sum()
    inflow_12m = _window(credits, w12m)["amount"].sum()

    # Income volatility: coefficient of variation of monthly inflows
    monthly_inflows = (
        _window(credits, w6m)
        .set_index("date")
        .resample("ME")["amount"]
        .sum()
    )
    income_cv = (
        monthly_inflows.std() / (monthly_inflows.mean() + 1e-6)
        if len(monthly_inflows) > 1
        else 0.0
    )

    # NSF (Non-Sufficient Funds) — strongest default predictor for thin-file
    nsf_3m = int(_window(history, w3m)["is_nsf"].sum()) if "is_nsf" in history.columns else 0
    nsf_6m = int(_window(history, w6m)["is_nsf"].sum()) if "is_nsf" in history.columns else 0

    # Balance behaviour
    recent_30d = _window(history, application_date - pd.DateOffset(days=30))
    days_negative = int(recent_30d["balance"].lt(0).sum()) if not recent_30d.empty else 0

    bal_3m = _window(history, w3m)["balance"]

    # Spending patterns
    total_debit_3m = abs(_window(debits, w3m)["amount"].sum())
    total_debit_6m = abs(_window(debits, w6m)["amount"].sum())

    # Rent detection
    rent_count_12m = 0
    if "category" in history.columns:
        rent_count_12m = int(
            _window(history, w12m)
            .loc[history["category"].str.upper() == "RENT"]
            .shape[0]
        )

    return {
        "avg_monthly_inflow_3m": inflow_3m / 3,
        "avg_monthly_inflow_6m": inflow_6m / 6,
        "avg_monthly_inflow_12m": inflow_12m / 12,
        "income_volatility_cv": float(income_cv),
        "nsf_count_3m": nsf_3m,
        "nsf_count_6m": nsf_6m,
        "days_negative_balance_30d": days_negative,
        "max_balance_3m": float(bal_3m.max()) if not bal_3m.empty else 0.0,
        "min_balance_3m": float(bal_3m.min()) if not bal_3m.empty else 0.0,
        "avg_balance_3m": float(bal_3m.mean()) if not bal_3m.empty else 0.0,
        "total_debit_3m": total_debit_3m,
        "total_debit_6m": total_debit_6m,
        "debit_to_income_ratio_3m": total_debit_3m / (inflow_3m + 1e-6),
        "rent_payments_12m": rent_count_12m,
        "months_of_history": _months_of_history(history),
    }


def _months_of_history(history: pd.DataFrame) -> int:
    if history.empty:
        return 0
    span = history["date"].max() - history["date"].min()
    return max(1, int(span.days / 30))


def _cold_start_cashflow_defaults() -> dict[str, Any]:
    """Imputation values for applicants with zero transaction history."""
    return {
        "avg_monthly_inflow_3m": 0.0,
        "avg_monthly_inflow_6m": 0.0,
        "avg_monthly_inflow_12m": 0.0,
        "income_volatility_cv": 1.0,  # high uncertainty
        "nsf_count_3m": 0,
        "nsf_count_6m": 0,
        "days_negative_balance_30d": 0,
        "max_balance_3m": 0.0,
        "min_balance_3m": 0.0,
        "avg_balance_3m": 0.0,
        "total_debit_3m": 0.0,
        "total_debit_6m": 0.0,
        "debit_to_income_ratio_3m": 0.0,
        "rent_payments_12m": 0,
        "months_of_history": 0,
    }


# ---------------------------------------------------------------------------
# Bureau features (use trends, not just levels)
# ---------------------------------------------------------------------------

def extract_bureau_features(bureau_snapshot: dict[str, Any]) -> dict[str, Any]:
    """
    Extract credit bureau features from a point-in-time snapshot.

    Parameters
    ----------
    bureau_snapshot : dict
        Raw bureau data at application date.  Expected keys:
        fico_score, revolving_utilization, num_tradelines,
        months_since_last_delinquency, num_inquiries_90d,
        oldest_account_months, total_balance, num_derogatory.

    Returns
    -------
    dict of feature_name → value
    """
    fico = bureau_snapshot.get("fico_score", np.nan)
    util = bureau_snapshot.get("revolving_utilization", np.nan)
    tradelines = bureau_snapshot.get("num_tradelines", 0)
    months_since_delinq = bureau_snapshot.get("months_since_last_delinquency", 999)
    inquiries_90d = bureau_snapshot.get("num_inquiries_90d", 0)
    oldest_months = bureau_snapshot.get("oldest_account_months", 0)
    total_balance = bureau_snapshot.get("total_balance", 0.0)
    derogatory = bureau_snapshot.get("num_derogatory", 0)

    return {
        "fico_score": fico,
        "revolving_utilization": util,
        "num_tradelines": tradelines,
        "months_since_last_delinquency": months_since_delinq,
        "inquiry_velocity_90d": inquiries_90d,
        "oldest_account_months": oldest_months,
        "total_balance": total_balance,
        "num_derogatory": derogatory,
        # Derived: thin-file indicator
        "is_thin_file": int(tradelines <= 2 or oldest_months < 24),
    }


# ---------------------------------------------------------------------------
# Merge + validation
# ---------------------------------------------------------------------------

FEATURE_ORDER: list[str] = [
    # Bureau
    "fico_score",
    "revolving_utilization",
    "num_tradelines",
    "months_since_last_delinquency",
    "inquiry_velocity_90d",
    "oldest_account_months",
    "total_balance",
    "num_derogatory",
    "is_thin_file",
    # Cashflow
    "avg_monthly_inflow_3m",
    "avg_monthly_inflow_6m",
    "avg_monthly_inflow_12m",
    "income_volatility_cv",
    "nsf_count_3m",
    "nsf_count_6m",
    "days_negative_balance_30d",
    "max_balance_3m",
    "min_balance_3m",
    "avg_balance_3m",
    "total_debit_3m",
    "total_debit_6m",
    "debit_to_income_ratio_3m",
    "rent_payments_12m",
    "months_of_history",
]


def merge_features(
    bureau: dict[str, Any],
    cashflow: dict[str, Any],
) -> np.ndarray:
    """
    Merge bureau + cashflow dicts into a single ordered feature vector.

    Returns
    -------
    np.ndarray of shape (1, n_features)
    """
    combined = {**bureau, **cashflow}
    vec = [combined.get(f, 0.0) for f in FEATURE_ORDER]
    arr = np.array(vec, dtype=np.float64).reshape(1, -1)

    # Sanity check: no NaN except fico_score (can be missing for thin-file)
    nan_mask = np.isnan(arr[0])
    nan_features = [FEATURE_ORDER[i] for i, is_nan in enumerate(nan_mask) if is_nan]
    if nan_features:
        logger.warning(f"NaN features detected (will be imputed by model): {nan_features}")

    return arr
