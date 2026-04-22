"""
FastAPI inference endpoint for real-time credit scoring.

Target: <100ms P99 latency.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from credit_risk.config import settings
from credit_risk.decision.engine import CreditDecision, make_decision
from credit_risk.explainability.shap_explainer import CreditExplainer
from credit_risk.features.engineering import (
    FEATURE_ORDER,
    extract_bureau_features,
    merge_features,
)
from credit_risk.models.trainer import load_model

# ---------------------------------------------------------------------------
# App + global state
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Credit Risk Decision API",
    version="1.0.0",
    description="Real-time PD scoring with SHAP explainability.",
)

# Global model state (loaded on startup)
_model = None
_calibrator = None
_explainer: CreditExplainer | None = None
_MODEL_DIR = Path("models/production")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class BureauData(BaseModel):
    """Credit bureau snapshot at application time."""
    fico_score: float | None = Field(None, ge=300, le=850)
    revolving_utilization: float | None = Field(None, ge=0.0, le=10.0)
    num_tradelines: int = 0
    months_since_last_delinquency: int = 999
    num_inquiries_90d: int = 0
    oldest_account_months: int = 0
    total_balance: float = 0.0
    num_derogatory: int = 0


class CashflowData(BaseModel):
    """Pre-computed cashflow features (from feature store or real-time)."""
    avg_monthly_inflow_3m: float = 0.0
    avg_monthly_inflow_6m: float = 0.0
    avg_monthly_inflow_12m: float = 0.0
    income_volatility_cv: float = 1.0
    nsf_count_3m: int = 0
    nsf_count_6m: int = 0
    days_negative_balance_30d: int = 0
    max_balance_3m: float = 0.0
    min_balance_3m: float = 0.0
    avg_balance_3m: float = 0.0
    total_debit_3m: float = 0.0
    total_debit_6m: float = 0.0
    debit_to_income_ratio_3m: float = 0.0
    rent_payments_12m: int = 0
    months_of_history: int = 0


class ScoreRequest(BaseModel):
    """Full scoring request."""
    applicant_id: str
    bureau: BureauData
    cashflow: CashflowData
    monthly_income: float = Field(5000.0, gt=0)


class ScoreResponse(BaseModel):
    """Scoring result with decision and explanation."""
    applicant_id: str
    pd: float
    decision: str
    apr: float | None
    credit_limit: float | None
    expected_loss: float
    risk_tier: str
    adverse_codes: list[str]
    model_version: str
    latency_ms: float


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def load_models() -> None:
    """Load model + calibrator + SHAP explainer on startup."""
    global _model, _calibrator, _explainer

    if not _MODEL_DIR.exists():
        logger.warning(f"Model dir {_MODEL_DIR} not found. API will return 503 until model is deployed.")
        return

    _model, _calibrator = load_model(_MODEL_DIR)
    _explainer = CreditExplainer(_model)
    logger.info("Model, calibrator, and SHAP explainer loaded.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    """Health check."""
    model_status = "loaded" if _model is not None else "not_loaded"
    return {"status": "ok", "model": model_status, "version": settings.model.version}


@app.post("/score", response_model=ScoreResponse)
async def score_application(req: ScoreRequest) -> ScoreResponse:
    """
    Score a single credit application.

    Returns calibrated PD, decision (APPROVE/REVIEW/DECLINE),
    risk-based pricing, credit limit, and adverse action codes.
    """
    start = time.perf_counter()

    if _model is None or _calibrator is None or _explainer is None:
        raise HTTPException(503, "Model not loaded. Deploy model to models/production/.")

    try:
        # 1. Build feature vector
        bureau = extract_bureau_features(req.bureau.model_dump())
        cashflow = req.cashflow.model_dump()
        X = merge_features(bureau, cashflow)

        # 2. Raw prediction + calibration
        pd_raw = float(_model.predict(X)[0])
        pd_calibrated = float(_calibrator.predict_proba(X)[0, 1])

        # 3. SHAP explanation
        explanation = _explainer.explain(X, n_reasons=4)

        # 4. Business decision
        decision = make_decision(
            pd=pd_calibrated,
            reason_codes=explanation.adverse_codes,
            monthly_income=req.monthly_income,
        )

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"Scored {req.applicant_id} | PD={pd_calibrated:.4f} | "
            f"{decision.action} | {latency_ms:.1f}ms"
        )

        return ScoreResponse(
            applicant_id=req.applicant_id,
            pd=round(pd_calibrated, 6),
            decision=decision.action,
            apr=decision.apr,
            credit_limit=decision.credit_limit,
            expected_loss=round(decision.expected_loss, 2),
            risk_tier=decision.risk_tier,
            adverse_codes=decision.reason_codes,
            model_version=decision.model_version,
            latency_ms=round(latency_ms, 1),
        )

    except Exception as e:
        logger.error(f"Scoring failed for {req.applicant_id}: {e}")
        raise HTTPException(500, f"Scoring error: {str(e)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def serve() -> None:
    """Run the API server."""
    uvicorn.run(
        "credit_risk.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    serve()
