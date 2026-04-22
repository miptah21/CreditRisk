"""Centralized configuration loaded from .env with typed defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_float(key: str, default: float = 0.0) -> float:
    return float(os.getenv(key, str(default)))


def _env_int(key: str, default: int = 0) -> int:
    return int(os.getenv(key, str(default)))


@dataclass(frozen=True)
class ModelConfig:
    """Model versioning and decision thresholds."""

    version: str = field(default_factory=lambda: _env("MODEL_VERSION", "credit_v1"))
    auto_decline_threshold: float = field(
        default_factory=lambda: _env_float("AUTO_DECLINE_THRESHOLD", 0.15)
    )
    manual_review_threshold: float = field(
        default_factory=lambda: _env_float("MANUAL_REVIEW_THRESHOLD", 0.08)
    )
    floor_apr: float = field(default_factory=lambda: _env_float("FLOOR_APR", 0.065))


@dataclass(frozen=True)
class PricingConfig:
    """Risk-based pricing parameters."""

    funding_cost: float = field(default_factory=lambda: _env_float("FUNDING_COST", 0.055))
    opex_ratio: float = field(default_factory=lambda: _env_float("OPEX_RATIO", 0.02))
    target_roe: float = field(default_factory=lambda: _env_float("TARGET_ROE", 0.15))
    tier1_ratio: float = field(default_factory=lambda: _env_float("TIER1_RATIO", 0.12))


@dataclass(frozen=True)
class RedisConfig:
    """Online feature store connection."""

    host: str = field(default_factory=lambda: _env("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: _env_int("REDIS_PORT", 6379))
    db: int = field(default_factory=lambda: _env_int("REDIS_DB", 0))


@dataclass(frozen=True)
class MonitoringConfig:
    """Drift and performance monitoring thresholds."""

    psi_warn: float = field(default_factory=lambda: _env_float("PSI_WARN_THRESHOLD", 0.10))
    psi_critical: float = field(
        default_factory=lambda: _env_float("PSI_CRITICAL_THRESHOLD", 0.25)
    )
    auc_min: float = field(default_factory=lambda: _env_float("AUC_MIN_THRESHOLD", 0.72))


@dataclass(frozen=True)
class Settings:
    """Top-level settings aggregating all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    mlflow_uri: str = field(
        default_factory=lambda: _env("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    mlflow_experiment: str = field(
        default_factory=lambda: _env("MLFLOW_EXPERIMENT_NAME", "credit_risk")
    )


# Singleton — import `settings` elsewhere
settings = Settings()
