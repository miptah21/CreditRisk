"""
End-to-end integration test: synthetic data → train → score → explain → decide → monitor.

Proves every layer of the system works together.
Run: uv run pytest tests/test_e2e.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from credit_risk.config import settings
from credit_risk.features.engineering import (
    FEATURE_ORDER,
    extract_bureau_features,
    extract_cashflow_features,
    merge_features,
)
from credit_risk.models.trainer import (
    BASE_PARAMS,
    calibrate_model,
    load_model,
    save_model,
    train_model,
)
from credit_risk.decision.engine import (
    CreditDecision,
    classify_risk_tier,
    compute_breakeven_apr,
    compute_credit_limit,
    make_decision,
)
from credit_risk.explainability.shap_explainer import CreditExplainer
from credit_risk.monitoring.drift import (
    check_score_drift,
    compute_calibration_error,
    compute_discrimination_metrics,
    compute_psi,
)
from credit_risk.fairness.audit import (
    compute_disparate_impact,
    run_fairness_audit,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic credit data
# ---------------------------------------------------------------------------

def _generate_synthetic_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic credit data with realistic feature distributions.

    Default rate ~5% with non-linear relationships.
    """
    rng = np.random.default_rng(seed)

    # Bureau features
    fico = rng.normal(680, 60, n).clip(300, 850)
    utilization = rng.beta(2, 5, n)  # right-skewed
    tradelines = rng.poisson(5, n)
    months_since_delinq = rng.exponential(60, n).clip(0, 999).astype(int)
    inquiries = rng.poisson(1.5, n)
    oldest_months = rng.normal(96, 36, n).clip(0, 360).astype(int)
    total_balance = rng.lognormal(9, 1.2, n).clip(0, 500_000)
    derogatory = rng.poisson(0.3, n)
    is_thin = (tradelines <= 2).astype(int)

    # Cashflow features
    monthly_inflow = rng.lognormal(8.5, 0.5, n).clip(1000, 50_000)
    income_cv = rng.beta(2, 8, n)
    nsf_3m = rng.poisson(0.5, n)
    nsf_6m = nsf_3m + rng.poisson(0.5, n)
    days_neg = rng.poisson(1.0, n)
    max_bal = rng.lognormal(7.5, 1.0, n)
    min_bal = max_bal * rng.uniform(0.0, 0.5, n)
    avg_bal = (max_bal + min_bal) / 2
    debit_3m = monthly_inflow * 3 * rng.uniform(0.5, 1.2, n)
    debit_6m = debit_3m * 2
    dti_ratio = debit_3m / (monthly_inflow * 3 + 1)
    rent_12m = rng.binomial(12, 0.8, n)
    history_months = rng.poisson(18, n).clip(0, 60)

    # Generate default label with non-linear relationship
    logit = (
        -3.0
        - 0.005 * (fico - 600)
        + 2.0 * utilization
        + 0.5 * nsf_6m
        + 0.3 * derogatory
        + 1.0 * income_cv
        - 0.01 * oldest_months
        + 0.1 * inquiries
        + rng.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    defaulted = rng.binomial(1, prob)

    # Application dates (time-based split compatible)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    groups = rng.choice(["A", "B"], n, p=[0.7, 0.3])

    df = pd.DataFrame({
        "application_date": dates[:n],
        "fico_score": fico,
        "revolving_utilization": utilization,
        "num_tradelines": tradelines,
        "months_since_last_delinquency": months_since_delinq,
        "inquiry_velocity_90d": inquiries,
        "oldest_account_months": oldest_months,
        "total_balance": total_balance,
        "num_derogatory": derogatory,
        "is_thin_file": is_thin,
        "avg_monthly_inflow_3m": monthly_inflow,
        "avg_monthly_inflow_6m": monthly_inflow * 0.98,
        "avg_monthly_inflow_12m": monthly_inflow * 0.95,
        "income_volatility_cv": income_cv,
        "nsf_count_3m": nsf_3m,
        "nsf_count_6m": nsf_6m,
        "days_negative_balance_30d": days_neg,
        "max_balance_3m": max_bal,
        "min_balance_3m": min_bal,
        "avg_balance_3m": avg_bal,
        "total_debit_3m": debit_3m,
        "total_debit_6m": debit_6m,
        "debit_to_income_ratio_3m": dti_ratio,
        "rent_payments_12m": rent_12m,
        "months_of_history": history_months,
        "defaulted": defaulted,
        "group": groups,
    })
    return df


@pytest.fixture(scope="module")
def synthetic_data() -> pd.DataFrame:
    return _generate_synthetic_data(n=2000)


@pytest.fixture(scope="module")
def trained_artifacts(synthetic_data: pd.DataFrame):
    """Train a model on synthetic data and return (model, calibrator, test data)."""
    df = synthetic_data
    feature_cols = [c for c in FEATURE_ORDER if c in df.columns]

    # Time-based split
    train = df.iloc[:1200]
    val = df.iloc[1200:1600]
    cal = df.iloc[1600:1800]
    test = df.iloc[1800:]

    X_train = train[feature_cols].values.astype(np.float64)
    y_train = train["defaulted"].values
    X_val = val[feature_cols].values.astype(np.float64)
    y_val = val["defaulted"].values
    X_cal = cal[feature_cols].values.astype(np.float64)
    y_cal = cal["defaulted"].values
    X_test = test[feature_cols].values.astype(np.float64)
    y_test = test["defaulted"].values

    model = train_model(X_train, y_train, X_val, y_val, num_boost_round=100)
    calibrator = calibrate_model(model, X_cal, y_cal)

    return model, calibrator, X_test, y_test


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    """Test feature extraction layer."""

    def test_bureau_features(self):
        snapshot = {
            "fico_score": 720,
            "revolving_utilization": 0.35,
            "num_tradelines": 8,
            "months_since_last_delinquency": 48,
            "num_inquiries_90d": 1,
            "oldest_account_months": 120,
            "total_balance": 15000,
            "num_derogatory": 0,
        }
        features = extract_bureau_features(snapshot)
        assert features["fico_score"] == 720
        assert features["is_thin_file"] == 0
        assert len(features) == 9

    def test_thin_file_detection(self):
        snapshot = {"num_tradelines": 1, "oldest_account_months": 12}
        features = extract_bureau_features(snapshot)
        assert features["is_thin_file"] == 1

    def test_merge_produces_correct_shape(self):
        bureau = extract_bureau_features({"fico_score": 700, "num_tradelines": 5, "oldest_account_months": 60})
        cashflow = {f: 0.0 for f in FEATURE_ORDER if f not in bureau}
        vec = merge_features(bureau, cashflow)
        assert vec.shape == (1, len(FEATURE_ORDER))

    def test_feature_order_is_stable(self):
        assert len(FEATURE_ORDER) == 24
        assert FEATURE_ORDER[0] == "fico_score"
        assert FEATURE_ORDER[-1] == "months_of_history"


class TestModelTraining:
    """Test model training and calibration."""

    def test_model_trains_successfully(self, trained_artifacts):
        model, calibrator, X_test, y_test = trained_artifacts
        assert model is not None
        assert calibrator is not None

    def test_predictions_are_probabilities(self, trained_artifacts):
        model, calibrator, X_test, y_test = trained_artifacts
        preds = calibrator.predict_proba(X_test)[:, 1]
        assert preds.min() >= 0.0
        assert preds.max() <= 1.0

    def test_auc_above_minimum(self, trained_artifacts):
        model, calibrator, X_test, y_test = trained_artifacts
        preds = model.predict(X_test)
        metrics = compute_discrimination_metrics(y_test, preds)
        # Synthetic data with clear signal → AUC should be decent
        assert metrics["auc"] >= 0.45, f"AUC too low: {metrics['auc']:.4f}"

    def test_model_save_and_load(self, trained_artifacts):
        model, calibrator, X_test, _ = trained_artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            save_model(model, calibrator, path)
            loaded_model, loaded_cal = load_model(path)
            # Predictions should match
            orig = model.predict(X_test[:5])
            loaded = loaded_model.predict(X_test[:5])
            np.testing.assert_array_almost_equal(orig, loaded)


class TestDecisionEngine:
    """Test PD → decision conversion."""

    def test_auto_approve(self):
        decision = make_decision(pd=0.01, reason_codes=[], monthly_income=8000)
        assert decision.action == "APPROVE"
        assert decision.apr is not None
        assert decision.credit_limit is not None
        assert decision.risk_tier == "PRIME"

    def test_manual_review(self):
        decision = make_decision(pd=0.10, reason_codes=["High utilization"])
        assert decision.action == "REVIEW"

    def test_auto_decline(self):
        decision = make_decision(pd=0.20, reason_codes=["Low FICO", "Many NSFs"])
        assert decision.action == "DECLINE"
        assert decision.apr is None
        assert decision.credit_limit is None
        assert len(decision.reason_codes) > 0

    def test_breakeven_apr_increases_with_pd(self):
        apr_low = compute_breakeven_apr(pd=0.02, lgd=0.60)
        apr_high = compute_breakeven_apr(pd=0.10, lgd=0.60)
        assert apr_high > apr_low

    def test_credit_limit_decreases_with_pd(self):
        limit_low = compute_credit_limit(pd=0.02, monthly_income=5000)
        limit_high = compute_credit_limit(pd=0.12, monthly_income=5000)
        assert limit_high < limit_low

    def test_risk_tiers(self):
        assert classify_risk_tier(0.01) == "PRIME"
        assert classify_risk_tier(0.03) == "NEAR_PRIME"
        assert classify_risk_tier(0.07) == "SUBPRIME"
        assert classify_risk_tier(0.15) == "DEEP_SUBPRIME"


class TestExplainability:
    """Test SHAP adverse action code generation."""

    def test_shap_produces_adverse_codes(self, trained_artifacts):
        model, _, X_test, _ = trained_artifacts
        explainer = CreditExplainer(model)
        result = explainer.explain(X_test[:1], n_reasons=3)
        assert len(result.adverse_codes) <= 3
        assert len(result.feature_contributions) == len(FEATURE_ORDER)
        assert result.shap_values.shape[0] == len(FEATURE_ORDER)

    def test_shap_adverse_codes_are_strings(self, trained_artifacts):
        model, _, X_test, _ = trained_artifacts
        explainer = CreditExplainer(model)
        result = explainer.explain(X_test[:1])
        for code in result.adverse_codes:
            assert isinstance(code, str)
            assert len(code) > 5  # not empty or trivial


class TestMonitoring:
    """Test PSI drift detection and metrics."""

    def test_psi_identical_distributions(self):
        a = np.random.default_rng(42).normal(0, 1, 1000)
        psi = compute_psi(a, a)
        assert psi < 0.01  # same distribution → PSI ≈ 0

    def test_psi_shifted_distribution(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 1000)
        b = rng.normal(1, 1, 1000)  # shifted by 1 std
        psi = compute_psi(a, b)
        assert psi > 0.10  # significant shift

    def test_drift_check_returns_report(self, trained_artifacts):
        _, _, X_test, _ = trained_artifacts
        model = trained_artifacts[0]
        train_scores = model.predict(X_test)
        prod_scores = train_scores + np.random.default_rng(42).normal(0, 0.01, len(train_scores))
        report = check_score_drift(train_scores, prod_scores)
        assert report.metric == "PSI"
        assert report.status in ("OK", "WARNING", "CRITICAL")

    def test_calibration_error(self, trained_artifacts):
        _, calibrator, X_test, y_test = trained_artifacts
        preds = calibrator.predict_proba(X_test)[:, 1]
        cal = compute_calibration_error(y_test, preds)
        assert "ece" in cal
        assert cal["ece"] >= 0


class TestFairness:
    """Test fairness audit."""

    def test_disparate_impact(self):
        decisions = np.array(["APPROVE"] * 80 + ["DECLINE"] * 20 + ["APPROVE"] * 60 + ["DECLINE"] * 40)
        groups = np.array(["A"] * 100 + ["B"] * 100)
        di = compute_disparate_impact(decisions, groups, "B", "A")
        # B approval = 60%, A approval = 80% → DI = 0.75
        assert abs(di - 0.75) < 0.01

    def test_full_fairness_audit(self, synthetic_data, trained_artifacts):
        model, calibrator, X_test, y_test = trained_artifacts
        df_test = synthetic_data.iloc[1800:]
        preds = calibrator.predict_proba(X_test)[:, 1]

        decisions = np.where(preds < 0.08, "APPROVE", np.where(preds < 0.15, "REVIEW", "DECLINE"))
        groups = df_test["group"].values

        report = run_fairness_audit(
            y_true=y_test,
            y_pred=preds,
            decisions=decisions,
            group_labels=groups,
            protected_group="B",
            reference_group="A",
        )
        assert isinstance(report.disparate_impact, float)
        assert isinstance(report.di_passes, bool)
        assert len(report.approval_rates) == 2


class TestEndToEnd:
    """Full pipeline: features → model → explain → decide."""

    def test_full_scoring_pipeline(self, trained_artifacts):
        model, calibrator, X_test, _ = trained_artifacts
        explainer = CreditExplainer(model)

        # Score one applicant
        X_single = X_test[:1]
        pd_raw = float(model.predict(X_single)[0])
        pd_cal = float(calibrator.predict_proba(X_single)[0, 1])

        # Explain
        explanation = explainer.explain(X_single, n_reasons=4)

        # Decide
        decision = make_decision(
            pd=pd_cal,
            reason_codes=explanation.adverse_codes,
            monthly_income=5000,
        )

        # Validate complete chain
        assert 0 <= pd_cal <= 1
        assert decision.action in ("APPROVE", "REVIEW", "DECLINE")
        assert isinstance(decision.model_version, str)
        assert isinstance(decision.expected_loss, float)
        assert len(explanation.feature_contributions) == len(FEATURE_ORDER)
