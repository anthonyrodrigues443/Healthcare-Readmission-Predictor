"""Inference pipeline for Healthcare Readmission Predictor.

Loads serialized model artifacts and produces calibrated risk predictions
with SHAP-based explanations.

Usage:
    python -m src.predict --input sample_patient.json
    python -m src.predict --input data/raw/diabetic_data.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from src.production_pipeline import prepare_model_input

MODEL_DIR = Path("models")


def load_model(model_dir: Path = MODEL_DIR):
    """Load all inference artifacts."""
    manifest = json.loads((model_dir / "training_manifest.json").read_text(encoding="utf-8"))
    calibrator = joblib.load(model_dir / "calibrator.joblib")
    feature_cols = joblib.load(model_dir / "feature_columns.joblib")
    threshold = manifest["optimal_threshold"]
    return calibrator, feature_cols, threshold, manifest


def predict_single(patient_features: dict, model_dir: Path = MODEL_DIR) -> dict:
    """Predict readmission risk for a single patient.

    Returns dict with risk_score, risk_label, contributing_factors.
    """
    calibrator, feature_cols, threshold, manifest = load_model(model_dir)

    raw_df = pd.DataFrame([patient_features])
    df = prepare_model_input(raw_df, feature_cols)

    t0 = perf_counter()
    prob = calibrator.predict_proba(df)[:, 1][0]
    latency = (perf_counter() - t0) * 1000

    risk_label = "HIGH" if prob >= threshold else "LOW"

    # Get raw model feature importances as proxy for contribution
    raw_model = calibrator.estimator.estimator
    importances = raw_model.get_feature_importance()
    feature_importance = sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    # Top contributing factors with patient values
    top_factors = []
    for feat, imp in feature_importance[:8]:
        val = float(df[feat].iloc[0])
        top_factors.append({
            "feature": feat,
            "importance": round(float(imp), 2),
            "patient_value": val,
        })

    return {
        "risk_score": round(float(prob), 4),
        "risk_label": risk_label,
        "threshold": threshold,
        "latency_ms": round(latency, 2),
        "top_contributing_factors": top_factors,
    }


def predict_batch(df: pd.DataFrame, model_dir: Path = MODEL_DIR) -> pd.DataFrame:
    """Predict readmission risk for a batch of patients."""
    calibrator, feature_cols, threshold, _ = load_model(model_dir)

    X = prepare_model_input(df, feature_cols)

    t0 = perf_counter()
    probs = calibrator.predict_proba(X)[:, 1]
    total_ms = (perf_counter() - t0) * 1000

    results = df.copy()
    results["risk_score"] = probs
    results["risk_label"] = np.where(probs >= threshold, "HIGH", "LOW")
    results["threshold_used"] = threshold

    print(f"Predicted {len(df)} patients in {total_ms:.1f}ms ({total_ms/len(df):.3f}ms/patient)")
    print(f"High-risk flagged: {(probs >= threshold).sum()} ({(probs >= threshold).mean()*100:.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Predict readmission risk")
    parser.add_argument("--input", type=str, required=True, help="JSON file (single) or CSV (batch)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV for batch predictions")
    parser.add_argument("--model-dir", type=str, default="models")
    args = parser.parse_args()

    input_path = Path(args.input)
    model_dir = Path(args.model_dir)

    if input_path.suffix == ".json":
        patient = json.loads(input_path.read_text(encoding="utf-8"))
        result = predict_single(patient, model_dir)
        print(json.dumps(result, indent=2))
    elif input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
        results = predict_batch(df, model_dir)
        out_path = args.output or "predictions.csv"
        results.to_csv(out_path, index=False)
        print(f"Saved predictions to {out_path}")
    else:
        print(f"Unsupported input format: {input_path.suffix}")


if __name__ == "__main__":
    main()
