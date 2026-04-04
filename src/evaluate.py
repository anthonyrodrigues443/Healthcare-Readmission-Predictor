"""Evaluation suite for Healthcare Readmission Predictor.

Runs comprehensive evaluation on the trained model: overall metrics,
subgroup analysis, calibration check, and generates evaluation plots.

Usage:
    python -m src.evaluate
    python -m src.evaluate --model-dir models/v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.data_pipeline import clean_and_engineer, download_dataset
from src.phase3_feature_engineering import add_phase3_features, get_feature_sets

RANDOM_STATE = 42
RESULTS_DIR = Path("results")


def evaluate_model(model_dir: Path = Path("models")) -> dict:
    """Run full evaluation suite."""
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load artifacts
    manifest = json.loads((model_dir / "training_manifest.json").read_text(encoding="utf-8"))
    calibrator = joblib.load(model_dir / "calibrator.joblib")
    feature_cols = joblib.load(model_dir / "feature_columns.joblib")
    threshold = manifest["optimal_threshold"]

    # Reproduce test set with same split
    raw_df = download_dataset()
    df = clean_and_engineer(raw_df)
    df = add_phase3_features(df)
    target = "readmitted_binary"

    X = df[feature_cols]
    y = df[target]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )

    probs = calibrator.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    # --- Overall metrics ---
    overall = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "auc": float(roc_auc_score(y_test, probs)),
        "avg_precision": float(average_precision_score(y_test, probs)),
        "brier": float(brier_score_loss(y_test, probs)),
        "threshold": threshold,
        "n_test": len(y_test),
        "positive_rate": float(y_test.mean()),
        "flagged_rate": float(preds.mean()),
    }

    # --- Subgroup analysis ---
    subgroups = {}
    subgroup_defs = {
        "prior_util_0": X_test["prior_utilization"] == 0,
        "prior_util_1": X_test["prior_utilization"] == 1,
        "prior_util_4+": X_test["prior_utilization"] >= 4,
        "age_lt45": X_test["age_numeric"] < 45,
        "age_45_64": (X_test["age_numeric"] >= 45) & (X_test["age_numeric"] < 65),
        "age_65_79": (X_test["age_numeric"] >= 65) & (X_test["age_numeric"] < 80),
        "age_80+": X_test["age_numeric"] >= 80,
        "los_1_2d": X_test["time_in_hospital"] <= 2,
        "los_3_5d": (X_test["time_in_hospital"] >= 3) & (X_test["time_in_hospital"] <= 5),
        "los_6+d": X_test["time_in_hospital"] >= 6,
    }
    for name, mask in subgroup_defs.items():
        n = int(mask.sum())
        if n < 50:
            continue
        sg_y = y_test[mask.values]
        sg_probs = probs[mask.values]
        sg_preds = preds[mask.values]
        subgroups[name] = {
            "n": n,
            "readmit_rate": round(float(sg_y.mean()), 4),
            "recall": round(float(recall_score(sg_y, sg_preds, zero_division=0)), 4),
            "auc": round(float(roc_auc_score(sg_y, sg_probs)), 4),
            "flagged_rate": round(float(sg_preds.mean()), 4),
        }

    # --- Calibration check ---
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy="uniform")
    calibration = {
        "bin_true": [round(float(v), 4) for v in prob_true],
        "bin_pred": [round(float(v), 4) for v in prob_pred],
        "brier_score": overall["brier"],
    }

    # --- Plots ---
    _plot_roc(y_test, probs, overall["auc"])
    _plot_precision_recall(y_test, probs)
    _plot_confusion_matrix(y_test, preds)
    _plot_calibration(prob_true, prob_pred)
    _plot_subgroups(subgroups)

    eval_results = {
        "overall": overall,
        "subgroups": subgroups,
        "calibration": calibration,
    }

    (RESULTS_DIR / "evaluation_results.json").write_text(
        json.dumps(eval_results, indent=2), encoding="utf-8"
    )

    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")

    print(f"\n{'Subgroup Analysis':}")
    print(f"  {'Group':<20s} {'N':>6s} {'Rate':>6s} {'Recall':>7s} {'AUC':>6s}")
    for name, sg in subgroups.items():
        print(f"  {name:<20s} {sg['n']:>6d} {sg['readmit_rate']:>6.3f} {sg['recall']:>7.3f} {sg['auc']:>6.3f}")

    return eval_results


def _plot_roc(y_test, probs, auc_score):
    fpr, tpr, _ = roc_curve(y_test, probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"CatBoost (AUC={auc_score:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Production Model")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "eval_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_precision_recall(y_test, probs):
    prec, rec, _ = precision_recall_curve(y_test, probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, lw=2, color="#d95f02")
    ax.axhline(y=y_test.mean(), color="gray", linestyle="--", label=f"Baseline ({y_test.mean():.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Production Model")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "eval_precision_recall.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_confusion_matrix(y_test, preds):
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Readmitted", "Readmitted"],
                yticklabels=["Not Readmitted", "Readmitted"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Production Model")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "eval_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_calibration(prob_true, prob_pred):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_pred, prob_true, "s-", label="Calibrated CatBoost")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve — Isotonic Calibration")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "eval_calibration.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_subgroups(subgroups):
    if not subgroups:
        return
    sg_df = pd.DataFrame(subgroups).T
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sg_df["recall"].sort_values().plot.barh(ax=axes[0], color="#1f77b4")
    axes[0].set_title("Recall by Subgroup")
    axes[0].set_xlim(0, 1)
    sg_df["auc"].sort_values().plot.barh(ax=axes[1], color="#d95f02")
    axes[1].set_title("AUC by Subgroup")
    axes[1].set_xlim(0.5, 0.8)
    plt.suptitle("Subgroup Performance — Production Model", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "eval_subgroups.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate readmission model")
    parser.add_argument("--model-dir", type=str, default="models")
    args = parser.parse_args()
    evaluate_model(Path(args.model_dir))


if __name__ == "__main__":
    main()
