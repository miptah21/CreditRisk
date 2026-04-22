# Credit Risk

AI Credit Risk Decision System — PD modeling, decision engine, explainability, and monitoring.

## Stack

- **Models:** LightGBM, XGBoost, CatBoost + Optuna tuning
- **Explainability:** SHAP TreeExplainer + LIME
- **Fairness:** Fairlearn (equalized odds, disparate impact)
- **Serving:** FastAPI + Redis feature cache
- **Tracking:** MLflow model registry
- **Feature Store:** Feast (offline Delta Lake + online Redis)
- **Monitoring:** Evidently (PSI, CSI, drift detection)
