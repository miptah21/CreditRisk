"""
LightGBM credit risk model: training, calibration, and persistence.

Production pattern: LightGBM + monotonicity constraints + Optuna tuning + isotonic calibration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import optuna
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from credit_risk.features.engineering import FEATURE_ORDER

# ---------------------------------------------------------------------------
# Monotonicity constraints — ensures SHAP directions are logically consistent
# ---------------------------------------------------------------------------
#  +1 = higher value → higher risk (monotone increasing)
#  -1 = higher value → lower risk (monotone decreasing)
#   0 = unconstrained

MONOTONE_CONSTRAINTS = {
    "fico_score": -1,                   # higher FICO → lower risk
    "revolving_utilization": 1,         # higher util → higher risk
    "num_tradelines": 0,                # ambiguous
    "months_since_last_delinquency": -1,  # longer since delinq → lower risk
    "inquiry_velocity_90d": 1,          # more inquiries → higher risk
    "oldest_account_months": -1,        # longer history → lower risk
    "total_balance": 0,                 # context-dependent
    "num_derogatory": 1,                # more derogs → higher risk
    "is_thin_file": 1,                  # thin file → higher risk
    "avg_monthly_inflow_3m": -1,        # more income → lower risk
    "avg_monthly_inflow_6m": -1,
    "avg_monthly_inflow_12m": -1,
    "income_volatility_cv": 1,          # volatile income → higher risk
    "nsf_count_3m": 1,                  # more NSFs → higher risk
    "nsf_count_6m": 1,
    "days_negative_balance_30d": 1,     # overdraft → higher risk
    "max_balance_3m": 0,
    "min_balance_3m": -1,              # higher min balance → lower risk
    "avg_balance_3m": -1,
    "total_debit_3m": 0,
    "total_debit_6m": 0,
    "debit_to_income_ratio_3m": 1,     # high spend ratio → higher risk
    "rent_payments_12m": -1,           # consistent rent → lower risk
    "months_of_history": -1,           # more history → lower risk
}


def get_monotone_constraints_list() -> list[int]:
    """Return monotonicity constraints in FEATURE_ORDER."""
    return [MONOTONE_CONSTRAINTS.get(f, 0) for f in FEATURE_ORDER]


# ---------------------------------------------------------------------------
# Base LightGBM parameters
# ---------------------------------------------------------------------------

BASE_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": 42,
    "is_unbalance": True,  # handles class imbalance (3-5% default rate)
    "monotone_constraints": get_monotone_constraints_list(),
}


# ---------------------------------------------------------------------------
# Optuna hyperparameter search
# ---------------------------------------------------------------------------

def _optuna_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Optuna objective: maximize validation AUC."""
    params = {
        **BASE_PARAMS,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    y_pred = model.predict(X_val)
    return roc_auc_score(y_val, y_pred)


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
) -> dict[str, Any]:
    """Run Optuna hyperparameter search, return best params."""
    logger.info(f"Starting Optuna search with {n_trials} trials...")

    study = optuna.create_study(direction="maximize", study_name="credit_lgbm")
    study.optimize(
        lambda trial: _optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = {**BASE_PARAMS, **study.best_params}
    logger.info(f"Best AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    return best


# ---------------------------------------------------------------------------
# Train + calibrate
# ---------------------------------------------------------------------------

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 1000,
) -> lgb.Booster:
    """Train a LightGBM model with early stopping."""
    if params is None:
        params = BASE_PARAMS

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_ORDER)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    y_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, y_pred)
    logger.info(f"Validation AUC: {auc:.4f} | Gini: {2 * auc - 1:.4f}")

    return model


def calibrate_model(
    model: lgb.Booster,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
) -> "_CalibratedModel":
    """
    Wrap the trained booster with isotonic calibration.

    Uses a separate calibration set (NOT the validation set).
    Returns a wrapper with predict_proba() interface.
    """
    from sklearn.isotonic import IsotonicRegression

    # Get raw predictions on calibration set
    raw_probs = model.predict(X_cal)

    # Fit isotonic regression: maps raw probabilities → calibrated probabilities
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_probs, y_cal)

    logger.info("Isotonic calibration fitted on calibration set.")
    return _CalibratedModel(model, iso)


class _CalibratedModel:
    """Wraps LightGBM Booster + IsotonicRegression for calibrated predictions."""

    def __init__(self, booster: lgb.Booster, isotonic: Any) -> None:
        self.booster = booster
        self.isotonic = isotonic

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities as (n, 2) array [P(good), P(default)]."""
        raw = self.booster.predict(X)
        calibrated = self.isotonic.predict(raw)
        calibrated = np.clip(calibrated, 0.0, 1.0)
        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: lgb.Booster, calibrator: Any, path: Path) -> None:
    """Save booster + calibrator to disk."""
    path.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path / "model.lgb"))
    joblib.dump(calibrator, path / "calibrator.pkl")
    logger.info(f"Model saved to {path}")


def load_model(path: Path) -> tuple[lgb.Booster, Any]:
    """Load booster + calibrator from disk."""
    model = lgb.Booster(model_file=str(path / "model.lgb"))
    calibrator = joblib.load(path / "calibrator.pkl")
    logger.info(f"Model loaded from {path}")
    return model, calibrator
