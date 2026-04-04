"""Production training pipeline for Healthcare Readmission Predictor.

Trains the champion CatBoost model with Optuna-tuned hyperparameters,
calibrates probabilities via isotonic regression, and serializes all
artifacts needed for inference.

Usage:
    python -m src.train
    python -m src.train --output-dir models/v2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

from src.production_pipeline import (
    RANDOM_STATE,
    json_safe_float,
    load_modeling_frame,
    make_production_splits,
    optimal_f1_threshold,
    safe_auc,
    split_metadata,
)

DEFAULT_OUTPUT_DIR = Path("models")
TUNING_RESULTS = Path("results/phase4_tuning_results.json")


def load_best_params() -> dict:
    """Load Optuna-tuned hyperparameters from Phase 4."""
    return json.loads(TUNING_RESULTS.read_text(encoding="utf-8"))["best_params"]


def load_and_prepare_data() -> tuple[pd.DataFrame, str]:
    """Load raw data, clean, engineer features, return ready-to-model DataFrame."""
    df, _, target = load_modeling_frame()
    return df, target


def train_champion(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    best_params: dict,
    output_dir: Path,
) -> dict:
    """Train CatBoost champion, calibrate, save artifacts, return metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    X = df[feature_cols]
    y = df[target]

    splits = make_production_splits(X, y)
    X_train, y_train = splits["train"]
    X_early, y_early = splits["early_stop"]
    X_cal, y_cal = splits["calibration"]
    X_test, y_test = splits["test"]

    print(
        f"Train: {len(X_train)} | Early-stop: {len(X_early)} | "
        f"Cal: {len(X_cal)} | Test: {len(X_test)}"
    )
    print(
        "Positive rate — "
        f"Train: {y_train.mean():.3f} | Early-stop: {y_early.mean():.3f} | "
        f"Cal: {y_cal.mean():.3f} | Test: {y_test.mean():.3f}"
    )

    # --- Train CatBoost ---
    model = CatBoostClassifier(
        **best_params,
        iterations=1500,
        auto_class_weights="Balanced",
        eval_metric="AUC",
        loss_function="Logloss",
        od_type="Iter",
        od_wait=50,
        random_seed=RANDOM_STATE,
        verbose=100,
        thread_count=2,
    )

    t0 = time.time()
    model.fit(X_train, y_train, eval_set=(X_early, y_early), early_stopping_rounds=50)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.1f}s ({model.best_iteration_} iterations)")

    # --- Calibrate ---
    calibrator = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrator.fit(X_cal, y_cal)

    # --- Find optimal threshold ---
    cal_probs = calibrator.predict_proba(X_cal)[:, 1]
    optimal_threshold = optimal_f1_threshold(y_cal, cal_probs)

    # --- Evaluate on test set ---
    test_probs = calibrator.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= optimal_threshold).astype(int)

    test_metrics = {
        "accuracy": float(accuracy_score(y_test, test_preds)),
        "f1": float(f1_score(y_test, test_preds)),
        "precision": float(precision_score(y_test, test_preds)),
        "recall": float(recall_score(y_test, test_preds)),
        "auc": json_safe_float(safe_auc(y_test, test_probs)),
        "avg_precision": float(average_precision_score(y_test, test_probs)),
        "brier": float(brier_score_loss(y_test, test_probs)),
        "threshold": optimal_threshold,
    }

    # --- Subgroup metrics ---
    low_util_mask = X_test["prior_utilization"] == 0
    high_util_mask = X_test["prior_utilization"] >= 4

    subgroup_metrics = {}
    for name, mask in [("low_util", low_util_mask), ("high_util", high_util_mask)]:
        if mask.sum() > 0:
            sg_probs = test_probs[mask.values]
            sg_preds = test_preds[mask.values]
            sg_y = y_test[mask.values]
            subgroup_metrics[name] = {
                "n": int(mask.sum()),
                "recall": float(recall_score(sg_y, sg_preds, zero_division=0)),
                "auc": json_safe_float(safe_auc(sg_y, sg_probs)),
                "readmit_rate": float(sg_y.mean()),
            }

    # --- Inference latency benchmark ---
    sample = X_test.iloc[: min(100, len(X_test))]
    t0 = time.perf_counter()
    for _ in range(10):
        calibrator.predict_proba(sample)
    latency_ms = (time.perf_counter() - t0) / 10 / len(sample) * 1000

    # --- Save artifacts ---
    model.save_model(str(output_dir / "catboost_champion.cbm"))
    joblib.dump(calibrator, output_dir / "calibrator.joblib")
    joblib.dump(feature_cols, output_dir / "feature_columns.joblib")

    manifest = {
        "model_type": "CatBoost + Isotonic Calibration",
        "best_params": best_params,
        "n_iterations": model.best_iteration_,
        "feature_count": len(feature_cols),
        "optimal_threshold": optimal_threshold,
        "train_time_seconds": round(train_time, 1),
        "latency_ms_per_sample": round(latency_ms, 4),
        "split_metadata": split_metadata(splits),
        "test_metrics": test_metrics,
        "subgroup_metrics": subgroup_metrics,
        "train_samples": len(X_train),
        "early_stop_samples": len(X_early),
        "cal_samples": len(X_cal),
        "test_samples": len(X_test),
        "positive_rate": float(y.mean()),
        "artifacts": [
            "catboost_champion.cbm",
            "calibrator.joblib",
            "feature_columns.joblib",
            "training_manifest.json",
        ],
    }

    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\n{'='*60}")
    print(f"Champion Model Saved to {output_dir}/")
    print(f"{'='*60}")
    print(f"AUC:        {test_metrics['auc']:.4f}")
    print(f"F1:         {test_metrics['f1']:.4f}")
    print(f"Recall:     {test_metrics['recall']:.4f}")
    print(f"Precision:  {test_metrics['precision']:.4f}")
    print(f"Brier:      {test_metrics['brier']:.4f}")
    print(f"Threshold:  {optimal_threshold:.4f}")
    print(f"Latency:    {latency_ms:.3f} ms/sample")
    print(f"Low-util recall:  {subgroup_metrics.get('low_util', {}).get('recall', 'N/A')}")
    print(f"High-util recall: {subgroup_metrics.get('high_util', {}).get('recall', 'N/A')}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Train healthcare readmission model")
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()

    best_params = load_best_params()
    df, feature_cols, target = load_modeling_frame()
    train_champion(df, target, feature_cols, best_params, Path(args.output_dir))


if __name__ == "__main__":
    main()
