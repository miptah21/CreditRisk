"""
Credit decision engine: PD → approval/decline + risk-based pricing + credit limit.

Converts calibrated probability of default into actionable business decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from credit_risk.config import settings


# ---------------------------------------------------------------------------
# Decision result
# ---------------------------------------------------------------------------

@dataclass
class CreditDecision:
    """Immutable decision record for audit trail."""

    action: str              # APPROVE | REVIEW | DECLINE
    pd: float                # Calibrated probability of default
    apr: float | None        # Annual percentage rate (None if declined)
    credit_limit: float | None  # Assigned limit (None if declined)
    expected_loss: float     # PD × LGD × EAD
    reason_codes: list[str]  # SHAP-derived adverse action codes
    risk_tier: str           # PRIME | NEAR_PRIME | SUBPRIME
    model_version: str


# ---------------------------------------------------------------------------
# Pricing formulas
# ---------------------------------------------------------------------------

def compute_breakeven_apr(
    pd: float,
    lgd: float = 0.60,
    funding_cost: float | None = None,
    opex_ratio: float | None = None,
    target_roe: float | None = None,
    tier1_ratio: float | None = None,
) -> float:
    """
    Minimum APR to break even on a risk-adjusted basis.

    APR = funding_cost + opex + credit_spread + capital_charge

    Parameters
    ----------
    pd : float
        Calibrated probability of default (0–1).
    lgd : float
        Loss given default (typically 0.40–0.70 for unsecured).
    """
    cfg = settings.pricing
    fc = funding_cost or cfg.funding_cost
    opex = opex_ratio or cfg.opex_ratio
    roe = target_roe or cfg.target_roe
    t1 = tier1_ratio or cfg.tier1_ratio

    # Expected loss spread (breakeven)
    credit_spread = (pd * lgd) / max(1 - pd, 1e-6)

    # Capital charge: simplified unexpected loss × capital × ROE target
    ul = np.sqrt(pd * (1 - pd)) * lgd
    capital_charge = ul * t1 * roe

    return fc + opex + credit_spread + capital_charge


def compute_credit_limit(
    pd: float,
    monthly_income: float,
    lgd: float = 0.60,
    max_dti: float = 0.40,
    max_limit: float = 50_000.0,
) -> float:
    """
    Dynamic credit limit based on income, risk, and DTI constraint.

    limit = min(
        income-based limit,
        risk-adjusted limit,
        absolute maximum
    )
    """
    # Income-based: max X% of annual income
    income_limit = monthly_income * 12 * max_dti

    # Risk-adjusted: scale down by default probability
    risk_factor = max(0.2, 1.0 - (pd * 5))  # linear decay, floor at 20%
    risk_limit = income_limit * risk_factor

    return round(min(risk_limit, max_limit), -2)  # round to nearest $100


def classify_risk_tier(pd: float) -> str:
    """Map PD to business risk tier."""
    if pd < 0.02:
        return "PRIME"
    elif pd < 0.05:
        return "NEAR_PRIME"
    elif pd < 0.10:
        return "SUBPRIME"
    else:
        return "DEEP_SUBPRIME"


# ---------------------------------------------------------------------------
# Core decision logic
# ---------------------------------------------------------------------------

def make_decision(
    pd: float,
    reason_codes: list[str] | None = None,
    monthly_income: float = 5_000.0,
    lgd: float = 0.60,
    ead: float | None = None,
) -> CreditDecision:
    """
    Convert calibrated PD into a full credit decision.

    Parameters
    ----------
    pd : float
        Calibrated probability of default.
    reason_codes : list[str]
        SHAP-derived adverse action codes (top-3 negative contributors).
    monthly_income : float
        Applicant's estimated monthly income.
    lgd : float
        Loss given default assumption.
    ead : float or None
        Exposure at default; if None, derived from credit limit.
    """
    cfg = settings.model
    codes = reason_codes or []
    tier = classify_risk_tier(pd)

    # --- DECLINE ---
    if pd >= cfg.auto_decline_threshold:
        el = pd * lgd * (ead or 0.0)
        logger.info(f"DECLINE | PD={pd:.4f} tier={tier}")
        return CreditDecision(
            action="DECLINE",
            pd=pd,
            apr=None,
            credit_limit=None,
            expected_loss=el,
            reason_codes=codes[:4],  # ECOA requires specific reasons
            risk_tier=tier,
            model_version=cfg.version,
        )

    # --- MANUAL REVIEW ---
    if pd >= cfg.manual_review_threshold:
        limit = compute_credit_limit(pd, monthly_income, lgd)
        apr = max(compute_breakeven_apr(pd, lgd), cfg.floor_apr)
        el = pd * lgd * (ead or limit * 0.75)
        logger.info(f"REVIEW | PD={pd:.4f} tier={tier} APR={apr:.2%} limit=${limit:,.0f}")
        return CreditDecision(
            action="REVIEW",
            pd=pd,
            apr=round(apr, 4),
            credit_limit=limit,
            expected_loss=el,
            reason_codes=codes[:4],
            risk_tier=tier,
            model_version=cfg.version,
        )

    # --- APPROVE ---
    limit = compute_credit_limit(pd, monthly_income, lgd)
    apr = max(compute_breakeven_apr(pd, lgd), cfg.floor_apr)
    el = pd * lgd * (ead or limit * 0.75)
    logger.info(f"APPROVE | PD={pd:.4f} tier={tier} APR={apr:.2%} limit=${limit:,.0f}")
    return CreditDecision(
        action="APPROVE",
        pd=pd,
        apr=round(apr, 4),
        credit_limit=limit,
        expected_loss=el,
        reason_codes=codes[:4],
        risk_tier=tier,
        model_version=cfg.version,
    )
