"""
SHAP-based explainability for regulatory compliance.

Generates adverse action codes (ECOA/FCRA) and audit-trail SHAP records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import shap
from loguru import logger

from credit_risk.features.engineering import FEATURE_ORDER


# ---------------------------------------------------------------------------
# Adverse action code mapping: feature name → human-readable reason
# ---------------------------------------------------------------------------

ADVERSE_CODE_MAP: dict[str, str] = {
    "fico_score": "Credit score too low",
    "revolving_utilization": "High credit utilization",
    "num_tradelines": "Insufficient credit accounts",
    "months_since_last_delinquency": "Recent delinquency on record",
    "inquiry_velocity_90d": "Too many recent credit inquiries",
    "oldest_account_months": "Credit history too short",
    "total_balance": "High outstanding balances",
    "num_derogatory": "Derogatory marks on credit report",
    "is_thin_file": "Insufficient credit history",
    "avg_monthly_inflow_3m": "Insufficient recent income",
    "avg_monthly_inflow_6m": "Insufficient income over 6 months",
    "avg_monthly_inflow_12m": "Insufficient annual income",
    "income_volatility_cv": "Unstable income pattern",
    "nsf_count_3m": "Recent overdraft events",
    "nsf_count_6m": "Frequent overdraft events",
    "days_negative_balance_30d": "Negative account balance",
    "max_balance_3m": "Low peak account balance",
    "min_balance_3m": "Low minimum account balance",
    "avg_balance_3m": "Low average account balance",
    "total_debit_3m": "High recent spending",
    "total_debit_6m": "High spending over 6 months",
    "debit_to_income_ratio_3m": "Spending exceeds income",
    "rent_payments_12m": "Inconsistent rent payments",
    "months_of_history": "Limited transaction history",
}


# ---------------------------------------------------------------------------
# SHAP explainer wrapper
# ---------------------------------------------------------------------------

@dataclass
class ExplanationResult:
    """SHAP explanation for a single application."""

    shap_values: np.ndarray           # Per-feature SHAP values
    base_value: float                  # Expected model output (mean prediction)
    adverse_codes: list[str]           # Top-N human-readable reasons
    feature_contributions: list[dict[str, Any]]  # Ordered list for audit


class CreditExplainer:
    """
    Wraps SHAP TreeExplainer for production use.

    Usage:
        explainer = CreditExplainer(lgbm_booster)
        result = explainer.explain(feature_vector)
        print(result.adverse_codes)  # ['Credit score too low', ...]
    """

    def __init__(self, model: Any) -> None:
        """
        Initialize with a LightGBM Booster.

        Parameters
        ----------
        model : lgb.Booster
            Trained LightGBM model. TreeExplainer is exact and fast for GBDTs.
        """
        self._explainer = shap.TreeExplainer(model)
        logger.info("SHAP TreeExplainer initialized.")

    def explain(
        self,
        X: np.ndarray,
        n_reasons: int = 4,
    ) -> ExplanationResult:
        """
        Generate SHAP explanation for a single application.

        Parameters
        ----------
        X : np.ndarray
            Feature vector of shape (1, n_features).
        n_reasons : int
            Number of adverse action codes to return (ECOA typically requires 2-4).

        Returns
        -------
        ExplanationResult
        """
        shap_values = self._explainer.shap_values(X)

        # Handle multi-output (binary classification can return list)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 = default

        sv = shap_values[0]  # single sample
        base_value = float(self._explainer.expected_value)
        if isinstance(self._explainer.expected_value, (list, np.ndarray)):
            base_value = float(self._explainer.expected_value[1])

        # Adverse codes: features that INCREASE default risk (positive SHAP)
        contributions = []
        for feat, val in zip(FEATURE_ORDER, sv):
            contributions.append({
                "feature": feat,
                "shap_value": float(val),
                "direction": "increases_risk" if val > 0 else "decreases_risk",
                "reason": ADVERSE_CODE_MAP.get(feat, feat),
            })

        # Sort by impact: most risk-increasing first
        risk_increasing = sorted(
            [c for c in contributions if c["shap_value"] > 0],
            key=lambda c: c["shap_value"],
            reverse=True,
        )

        adverse_codes = [c["reason"] for c in risk_increasing[:n_reasons]]

        return ExplanationResult(
            shap_values=sv,
            base_value=base_value,
            adverse_codes=adverse_codes,
            feature_contributions=sorted(
                contributions, key=lambda c: abs(c["shap_value"]), reverse=True
            ),
        )

    def explain_batch(
        self,
        X: np.ndarray,
        n_reasons: int = 4,
    ) -> list[ExplanationResult]:
        """Batch explain multiple applications."""
        results = []
        for i in range(X.shape[0]):
            results.append(self.explain(X[i : i + 1], n_reasons=n_reasons))
        return results
