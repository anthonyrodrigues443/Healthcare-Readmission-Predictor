"""
Phase 1 baseline training: Logistic Regression on raw features vs. domain-engineered features.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.utils import get_project_root, load_config, setup_logger, save_metrics
from src.data_pipeline import run_pipeline
from src.feature_engineering import engineer_features
from src.evaluate import compute_metrics, print_metrics

logger = setup_logger(__name__)


def train_baseline(X_train, y_train, X_test, y_test, config: dict) -> dict:
    """Experiment 1.1: Logistic Regression on raw features."""
    logger.info("Experiment 1.1: Baseline Logistic Regression (raw features)")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=config["model"]["baseline"]["C"],
            max_iter=config["model"]["baseline"]["max_iter"],
            class_weight=config["model"]["baseline"]["class_weight"],
            random_state=config["model"]["baseline"]["random_state"],
            solver="lbfgs",
            n_jobs=-1,
        ))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob, model_name="LR_baseline_raw")
    print_metrics(metrics)

    root = get_project_root()
    model_path = root / config["results"]["models_path"] / "lr_baseline_raw.joblib"
    joblib.dump(pipe, model_path)
    logger.info(f"Model saved to {model_path}")

    return metrics


def train_domain_features(X_train, y_train, X_test, y_test, config: dict) -> dict:
    """Experiment 1.2: Logistic Regression with LACE + Charlson + domain features."""
    logger.info("Experiment 1.2: Logistic Regression with domain-engineered features (LACE + Charlson + polypharmacy)")

    X_train_eng = engineer_features(X_train)
    X_test_eng = engineer_features(X_test)

    X_train_eng = X_train_eng.fillna(0)
    X_test_eng = X_test_eng.fillna(0)

    for col in X_train_eng.select_dtypes(include="object").columns:
        X_train_eng[col] = pd.Categorical(X_train_eng[col]).codes
        X_test_eng[col] = pd.Categorical(X_test_eng[col]).codes

    test_cols = [c for c in X_train_eng.columns if c in X_test_eng.columns]
    X_train_eng = X_train_eng[test_cols]
    X_test_eng = X_test_eng[test_cols]

    new_features = [c for c in X_train_eng.columns if c not in X_train.columns]
    logger.info(f"Domain features added: {new_features}")
    logger.info(f"Total feature count: {X_train_eng.shape[1]} (was {X_train.shape[1]})")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
            n_jobs=-1,
        ))
    ])

    pipe.fit(X_train_eng, y_train)
    y_pred = pipe.predict(X_test_eng)
    y_prob = pipe.predict_proba(X_test_eng)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob, model_name="LR_domain_features")
    print_metrics(metrics)

    root = get_project_root()
    model_path = root / config["results"]["models_path"] / "lr_domain_features.joblib"
    joblib.dump(pipe, model_path)
    logger.info(f"Model saved to {model_path}")

    metrics["feature_count"] = X_train_eng.shape[1]
    metrics["new_domain_features"] = new_features

    return metrics, X_train_eng, X_test_eng


def main():
    config = load_config()
    root = get_project_root()

    processed_path = root / config["data"]["processed_path"]
    if (processed_path / "X_train.parquet").exists():
        logger.info("Loading preprocessed data from parquet files")
        X_train = pd.read_parquet(processed_path / "X_train.parquet")
        X_test = pd.read_parquet(processed_path / "X_test.parquet")
        y_train = pd.read_parquet(processed_path / "y_train.parquet").squeeze()
        y_test = pd.read_parquet(processed_path / "y_test.parquet").squeeze()
    else:
        X_train, X_test, y_train, y_test = run_pipeline(config)

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    logger.info(f"Positive rate — Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")

    metrics_baseline = train_baseline(X_train, y_train, X_test, y_test, config)

    metrics_domain, X_train_eng, X_test_eng = train_domain_features(
        X_train, y_train, X_test, y_test, config
    )

    all_metrics = {
        "experiment_1_1_baseline": metrics_baseline,
        "experiment_1_2_domain": metrics_domain,
        "dataset_info": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "raw_feature_count": X_train.shape[1],
            "domain_feature_count": X_train_eng.shape[1],
        }
    }

    metrics_path = root / config["results"]["metrics_path"]
    save_metrics(all_metrics, str(metrics_path))
    logger.info(f"All metrics saved to {metrics_path}")

    logger.info("\n" + "="*65)
    logger.info("HEAD-TO-HEAD: Raw Features vs Domain-Engineered Features")
    logger.info("="*65)
    for key in ["accuracy", "f1", "precision", "recall_sensitivity", "auc_roc"]:
        b = metrics_baseline.get(key, "N/A")
        d = metrics_domain.get(key, "N/A")
        delta = ""
        if isinstance(b, float) and isinstance(d, float):
            delta = f"  Δ={d-b:+.4f}"
        logger.info(f"  {key:25s} Baseline={b}  Domain={d}{delta}")

    return all_metrics


if __name__ == "__main__":
    main()
