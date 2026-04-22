"""
Training pipeline: data prep → time-split → train → calibrate → evaluate → save.

Usage:
    uv run credit-train
    uv run credit-train --data-path data/loans.parquet --n-trials 100
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import typer
from loguru import logger

from credit_risk.config import settings
from credit_risk.explainability.shap_explainer import CreditExplainer
from credit_risk.features.engineering import FEATURE_ORDER
from credit_risk.models.trainer import (
    calibrate_model,
    save_model,
    train_model,
    tune_hyperparameters,
)
from credit_risk.monitoring.drift import (
    compute_calibration_error,
    compute_discrimination_metrics,
)

app = typer.Typer(help="Credit Risk Training Pipeline")


def _time_based_split(
    df: pd.DataFrame,
    date_col: str = "application_date",
    train_end: str = "2022-12-31",
    val_end: str = "2023-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strict time-based split — the core guard against temporal leakage.

    Train: <= train_end
    Val:   train_end < date <= val_end
    Cal:   50% of val (for isotonic calibration)
    Test:  > val_end
    """
    df[date_col] = pd.to_datetime(df[date_col])

    train = df[df[date_col] <= train_end]
    val = df[(df[date_col] > train_end) & (df[date_col] <= val_end)]
    test = df[df[date_col] > val_end]

    # Split validation into val + calibration
    val_shuffled = val.sample(frac=1, random_state=42)
    mid = len(val_shuffled) // 2
    val_set = val_shuffled.iloc[:mid]
    cal_set = val_shuffled.iloc[mid:]

    logger.info(
        f"Split: train={len(train)}, val={len(val_set)}, "
        f"cal={len(cal_set)}, test={len(test)}"
    )
    return train, val_set, cal_set, test


@app.command()
def main(
    data_path: str = typer.Option("data/loans.parquet", help="Path to loan data"),
    output_dir: str = typer.Option("models/production", help="Model output directory"),
    n_trials: int = typer.Option(30, help="Optuna trials for hyperparameter search"),
    skip_tuning: bool = typer.Option(False, help="Skip Optuna and use base params"),
) -> None:
    """Run the full training pipeline."""
    output = Path(output_dir)

    # --- MLflow tracking ---
    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    with mlflow.start_run(run_name=f"train_{settings.model.version}"):
        # 1. Load data
        path = Path(data_path)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            raise typer.BadParameter(f"Unsupported format: {path.suffix}")

        logger.info(f"Loaded {len(df)} records from {data_path}")
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("n_records", len(df))

        # 2. Time-based split
        train, val, cal, test = _time_based_split(df)

        # Extract features + labels
        feature_cols = [c for c in FEATURE_ORDER if c in df.columns]
        target_col = "defaulted"

        X_train = train[feature_cols].values.astype(np.float64)
        y_train = train[target_col].values.astype(int)
        X_val = val[feature_cols].values.astype(np.float64)
        y_val = val[target_col].values.astype(int)
        X_cal = cal[feature_cols].values.astype(np.float64)
        y_cal = cal[target_col].values.astype(int)
        X_test = test[feature_cols].values.astype(np.float64)
        y_test = test[target_col].values.astype(int)

        logger.info(f"Features used: {len(feature_cols)}")
        logger.info(f"Default rate: train={y_train.mean():.2%}, test={y_test.mean():.2%}")
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("default_rate_train", round(float(y_train.mean()), 4))

        # 3. Hyperparameter tuning (optional)
        if skip_tuning:
            from credit_risk.models.trainer import BASE_PARAMS
            best_params = BASE_PARAMS
            logger.info("Skipping Optuna — using BASE_PARAMS.")
        else:
            best_params = tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials=n_trials)

        mlflow.log_params({k: str(v) for k, v in best_params.items() if not isinstance(v, list)})

        # 4. Train final model
        model = train_model(X_train, y_train, X_val, y_val, params=best_params)

        # 5. Calibrate on held-out calibration set
        calibrator = calibrate_model(model, X_cal, y_cal)

        # 6. Evaluate on test set
        y_pred_raw = model.predict(X_test)
        y_pred_cal = calibrator.predict_proba(X_test)[:, 1]

        metrics_raw = compute_discrimination_metrics(y_test, y_pred_raw)
        metrics_cal = compute_discrimination_metrics(y_test, y_pred_cal)
        cal_error = compute_calibration_error(y_test, y_pred_cal)

        logger.info(f"Test AUC (raw): {metrics_raw['auc']:.4f}")
        logger.info(f"Test AUC (calibrated): {metrics_cal['auc']:.4f}")
        logger.info(f"Test Gini: {metrics_cal['gini']:.4f}")
        logger.info(f"Test KS: {metrics_cal['ks']:.4f}")
        logger.info(f"ECE: {cal_error['ece']:.4f}")

        mlflow.log_metrics({
            "test_auc_raw": metrics_raw["auc"],
            "test_auc_calibrated": metrics_cal["auc"],
            "test_gini": metrics_cal["gini"],
            "test_ks": metrics_cal["ks"],
            "test_ece": cal_error["ece"],
        })

        # 7. SHAP summary (log top features)
        explainer = CreditExplainer(model)
        sample = X_test[:100]
        import shap
        shap_vals = explainer._explainer.shap_values(sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        top_features = sorted(
            zip(feature_cols, mean_abs_shap), key=lambda x: x[1], reverse=True
        )
        logger.info("Top-10 features by mean |SHAP|:")
        for feat, imp in top_features[:10]:
            logger.info(f"  {feat}: {imp:.4f}")

        # 8. Save model
        save_model(model, calibrator, output)
        mlflow.log_artifacts(str(output), artifact_path="model")

        logger.info(f"Pipeline complete. Model saved to {output}")
        logger.info(f"AUC={metrics_cal['auc']:.4f} Gini={metrics_cal['gini']:.4f} KS={metrics_cal['ks']:.4f}")


if __name__ == "__main__":
    app()
