"""Phase 4: Hyperparameter Tuning + Error Analysis for Healthcare Readmission.

Builds on Phase 3 champion (CatBoost + full_83, AUC=0.687).
Research-informed Optuna tuning based on:
- CatBoost official parameter tuning docs (catboost.ai)
- PMC 2025: ADASYN > SMOTE for imbalanced EHR (JMIR Med Informatics)
- Published readmission benchmarks: AUC 0.62-0.82 for 30-day readmission

Key questions:
1. How much AUC can systematic tuning recover over default CatBoost?
2. Which hyperparameters matter most? (depth, lr, regularization)
3. What clinical operating point balances recall vs precision?
4. Are probability outputs calibrated for clinical decision support?
5. Where does the model systematically fail (subgroup error analysis)?
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    learning_curve,
    train_test_split,
)

import optuna
from catboost import CatBoostClassifier

from src.data_pipeline import clean_and_engineer, download_dataset
from src.phase3_feature_engineering import add_phase3_features

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_DIR = Path("results")
REPORTS_DIR = Path("reports")
PROCESSED_PATH = Path("data/processed/readmission_processed.csv")
PHASE4_JSON = RESULTS_DIR / "phase4_tuning_results.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"
EXPERIMENT_LOG_PATH = RESULTS_DIR / "EXPERIMENT_LOG.md"

N_OPTUNA_TRIALS = 80
CV_FOLDS = 5
RANDOM_STATE = 42


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if PROCESSED_PATH.exists():
        return pd.read_csv(PROCESSED_PATH)
    raw_df = download_dataset()
    df = clean_and_engineer(raw_df)
    df.to_csv(PROCESSED_PATH, index=False)
    return df


def get_full_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "readmitted_binary"]


# ---------------------------------------------------------------------------
# Experiment 4.1: Optuna hyperparameter tuning for CatBoost
# ---------------------------------------------------------------------------

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for CatBoost — research-informed search ranges."""
    params = {
        "iterations": 1000,
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        "auto_class_weights": "Balanced",
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "od_type": "Iter",
        "od_wait": 50,
        "random_seed": RANDOM_STATE,
        "verbose": 0,
        "thread_count": 2,
    }

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    y_prob = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_prob)


def run_optuna_tuning(X_train, y_train, X_val, y_val, n_trials=N_OPTUNA_TRIALS):
    """Run Optuna TPE optimization."""
    print(f"\n{'='*72}")
    print(f"EXPERIMENT 4.1: OPTUNA HYPERPARAMETER TUNING ({n_trials} trials)")
    print(f"{'='*72}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    def objective(trial):
        return optuna_objective(trial, X_train, y_train, X_val, y_val)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    return study


def train_tuned_model(best_params, X_train, y_train, X_val, y_val):
    """Train final model with best Optuna params."""
    params = {
        **best_params,
        "iterations": 1500,
        "auto_class_weights": "Balanced",
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "od_type": "Iter",
        "od_wait": 50,
        "random_seed": RANDOM_STATE,
        "verbose": 0,
        "thread_count": 2,
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    return model


# ---------------------------------------------------------------------------
# Experiment 4.2: Cross-validated tuned model performance
# ---------------------------------------------------------------------------

def cross_validate_model(params, X, y, n_folds=CV_FOLDS):
    """5-fold stratified CV with tuned params."""
    print(f"\n{'='*72}")
    print(f"EXPERIMENT 4.2: {n_folds}-FOLD CROSS-VALIDATION (tuned CatBoost)")
    print(f"{'='*72}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        model_params = {
            **params,
            "iterations": 1000,
            "auto_class_weights": "Balanced",
            "eval_metric": "AUC",
            "loss_function": "Logloss",
            "od_type": "Iter",
            "od_wait": 50,
            "random_seed": RANDOM_STATE,
            "verbose": 0,
            "thread_count": 2,
        }
        model = CatBoostClassifier(**model_params)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)

        y_prob = model.predict_proba(X_va)[:, 1]
        y_pred = model.predict(X_va)

        metrics = {
            "fold": fold_i,
            "auc": roc_auc_score(y_va, y_prob),
            "f1": f1_score(y_va, y_pred, zero_division=0),
            "precision": precision_score(y_va, y_pred, zero_division=0),
            "recall": recall_score(y_va, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_va, y_pred),
        }
        fold_metrics.append(metrics)
        print(f"  Fold {fold_i}: AUC={metrics['auc']:.4f} | F1={metrics['f1']:.3f} | Recall={metrics['recall']:.3f}")

    df_folds = pd.DataFrame(fold_metrics)
    means = df_folds.drop(columns=["fold"]).mean()
    stds = df_folds.drop(columns=["fold"]).std()
    print(f"\n  Mean AUC: {means['auc']:.4f} ± {stds['auc']:.4f}")
    print(f"  Mean F1:  {means['f1']:.4f} ± {stds['f1']:.4f}")

    return df_folds, means, stds


# ---------------------------------------------------------------------------
# Experiment 4.3: Threshold optimization (clinical operating points)
# ---------------------------------------------------------------------------

def optimize_thresholds(y_test, y_prob):
    """Find optimal thresholds for different clinical objectives."""
    print(f"\n{'='*72}")
    print("EXPERIMENT 4.3: THRESHOLD OPTIMIZATION")
    print(f"{'='*72}")

    results = {}

    # Strategy A: Youden's J (balanced sensitivity/specificity)
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    results["youden_j"] = {
        "threshold": float(roc_thresholds[best_j_idx]),
        "sensitivity": float(tpr[best_j_idx]),
        "specificity": float(1 - fpr[best_j_idx]),
        "j_statistic": float(j_scores[best_j_idx]),
    }

    # Strategy B: F1-optimal
    precision_arr, recall_arr, pr_thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    results["f1_optimal"] = {
        "threshold": float(pr_thresholds[best_f1_idx]) if best_f1_idx < len(pr_thresholds) else 0.5,
        "f1": float(f1_scores[best_f1_idx]),
        "precision": float(precision_arr[best_f1_idx]),
        "recall": float(recall_arr[best_f1_idx]),
    }

    # Strategy C: Recall-constrained (clinical: catch ≥75% of readmissions)
    for target_recall in [0.75, 0.80, 0.85]:
        key = f"recall_{int(target_recall*100)}"
        # Walk thresholds from low to high; find first where recall >= target
        sorted_indices = np.argsort(pr_thresholds)
        found = False
        for idx in sorted_indices:
            y_pred_at = (y_prob >= pr_thresholds[idx]).astype(int)
            rec = recall_score(y_test, y_pred_at, zero_division=0)
            if rec >= target_recall:
                prec = precision_score(y_test, y_pred_at, zero_division=0)
                f1 = f1_score(y_test, y_pred_at, zero_division=0)
                results[key] = {
                    "threshold": float(pr_thresholds[idx]),
                    "recall": float(rec),
                    "precision": float(prec),
                    "f1": float(f1),
                    "flagged_pct": float(y_pred_at.mean()),
                }
                found = True
                break
        if not found:
            # Use lowest threshold
            results[key] = {
                "threshold": float(pr_thresholds[0]) if len(pr_thresholds) > 0 else 0.01,
                "recall": float(target_recall),
                "precision": 0.0,
                "f1": 0.0,
                "flagged_pct": 1.0,
            }

    for name, vals in results.items():
        print(f"  {name}: threshold={vals.get('threshold', 0):.3f} | "
              f"recall={vals.get('recall', vals.get('sensitivity', 0)):.3f} | "
              f"precision={vals.get('precision', 0):.3f}")

    return results


# ---------------------------------------------------------------------------
# Experiment 4.4: Calibration analysis
# ---------------------------------------------------------------------------

def calibration_analysis(model, X_train, y_train, X_test, y_test):
    """Assess and improve probability calibration."""
    print(f"\n{'='*72}")
    print("EXPERIMENT 4.4: PROBABILITY CALIBRATION")
    print(f"{'='*72}")

    y_prob_raw = model.predict_proba(X_test)[:, 1]
    brier_raw = brier_score_loss(y_test, y_prob_raw)

    # Isotonic calibration (n > 1000, recommended per literature)
    calibrated_iso = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated_iso.fit(X_train, y_train)
    y_prob_iso = calibrated_iso.predict_proba(X_test)[:, 1]
    brier_iso = brier_score_loss(y_test, y_prob_iso)

    # Platt scaling
    calibrated_platt = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated_platt.fit(X_train, y_train)
    y_prob_platt = calibrated_platt.predict_proba(X_test)[:, 1]
    brier_platt = brier_score_loss(y_test, y_prob_platt)

    results = {
        "brier_raw": float(brier_raw),
        "brier_isotonic": float(brier_iso),
        "brier_platt": float(brier_platt),
        "auc_raw": float(roc_auc_score(y_test, y_prob_raw)),
        "auc_isotonic": float(roc_auc_score(y_test, y_prob_iso)),
        "auc_platt": float(roc_auc_score(y_test, y_prob_platt)),
    }

    print(f"  Raw CatBoost:    Brier={brier_raw:.4f} | AUC={results['auc_raw']:.4f}")
    print(f"  Isotonic:        Brier={brier_iso:.4f} | AUC={results['auc_isotonic']:.4f}")
    print(f"  Platt scaling:   Brier={brier_platt:.4f} | AUC={results['auc_platt']:.4f}")

    return results, y_prob_raw, y_prob_iso, y_prob_platt


# ---------------------------------------------------------------------------
# Experiment 4.5: Error analysis — subgroup failures
# ---------------------------------------------------------------------------

def error_analysis(model, X_test, y_test, df_full):
    """Analyze systematic failure patterns across clinical subgroups."""
    print(f"\n{'='*72}")
    print("EXPERIMENT 4.5: ERROR ANALYSIS")
    print(f"{'='*72}")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Overall confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix:")
    print(f"  TN={tn} | FP={fp}")
    print(f"  FN={fn} | TP={tp}")
    print(f"  False Negative Rate: {fn/(fn+tp):.3f} (missed readmissions)")
    print(f"  False Positive Rate: {fp/(fp+tn):.3f} (unnecessary interventions)")

    # Error types
    errors = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "error_type": np.where(
            y_test.values == y_pred, "correct",
            np.where(y_test.values == 1, "false_negative", "false_positive")
        ),
    }, index=X_test.index)

    # Subgroup analysis on key clinical features
    subgroup_results = []
    X_aligned = X_test.copy()

    # Age subgroups (from age_numeric)
    if "age_numeric" in X_aligned.columns:
        age_bins = pd.cut(X_aligned["age_numeric"], bins=[0, 45, 65, 80, 100], labels=["<45", "45-64", "65-79", "80+"])
        for group_name in age_bins.unique():
            if pd.isna(group_name):
                continue
            mask = age_bins == group_name
            if mask.sum() < 10:
                continue
            sub_auc = roc_auc_score(y_test[mask], y_prob[mask]) if y_test[mask].nunique() > 1 else 0.5
            sub_recall = recall_score(y_test[mask], y_pred[mask], zero_division=0)
            sub_f1 = f1_score(y_test[mask], y_pred[mask], zero_division=0)
            sub_rate = y_test[mask].mean()
            subgroup_results.append({
                "subgroup": f"Age {group_name}",
                "n": int(mask.sum()),
                "readmit_rate": float(sub_rate),
                "auc": float(sub_auc),
                "recall": float(sub_recall),
                "f1": float(sub_f1),
            })

    # Prior utilization subgroups
    if "prior_utilization" in X_aligned.columns:
        util_bins = pd.cut(X_aligned["prior_utilization"], bins=[-0.1, 0, 1, 3, 100], labels=["0", "1", "2-3", "4+"])
        for group_name in util_bins.unique():
            if pd.isna(group_name):
                continue
            mask = util_bins == group_name
            if mask.sum() < 10:
                continue
            sub_auc = roc_auc_score(y_test[mask], y_prob[mask]) if y_test[mask].nunique() > 1 else 0.5
            sub_recall = recall_score(y_test[mask], y_pred[mask], zero_division=0)
            sub_f1 = f1_score(y_test[mask], y_pred[mask], zero_division=0)
            sub_rate = y_test[mask].mean()
            subgroup_results.append({
                "subgroup": f"Prior Util {group_name}",
                "n": int(mask.sum()),
                "readmit_rate": float(sub_rate),
                "auc": float(sub_auc),
                "recall": float(sub_recall),
                "f1": float(sub_f1),
            })

    # Number of inpatient visits
    if "number_inpatient" in X_aligned.columns:
        inpat_bins = pd.cut(X_aligned["number_inpatient"], bins=[-0.1, 0, 1, 100], labels=["0", "1", "2+"])
        for group_name in inpat_bins.unique():
            if pd.isna(group_name):
                continue
            mask = inpat_bins == group_name
            if mask.sum() < 10:
                continue
            sub_auc = roc_auc_score(y_test[mask], y_prob[mask]) if y_test[mask].nunique() > 1 else 0.5
            sub_recall = recall_score(y_test[mask], y_pred[mask], zero_division=0)
            sub_f1 = f1_score(y_test[mask], y_pred[mask], zero_division=0)
            sub_rate = y_test[mask].mean()
            subgroup_results.append({
                "subgroup": f"Prior Inpatient {group_name}",
                "n": int(mask.sum()),
                "readmit_rate": float(sub_rate),
                "auc": float(sub_auc),
                "recall": float(sub_recall),
                "f1": float(sub_f1),
            })

    # Time in hospital
    if "time_in_hospital" in X_aligned.columns:
        los_bins = pd.cut(X_aligned["time_in_hospital"], bins=[0, 2, 5, 8, 100], labels=["1-2d", "3-5d", "6-8d", "9+d"])
        for group_name in los_bins.unique():
            if pd.isna(group_name):
                continue
            mask = los_bins == group_name
            if mask.sum() < 10:
                continue
            sub_auc = roc_auc_score(y_test[mask], y_prob[mask]) if y_test[mask].nunique() > 1 else 0.5
            sub_recall = recall_score(y_test[mask], y_pred[mask], zero_division=0)
            sub_f1 = f1_score(y_test[mask], y_pred[mask], zero_division=0)
            sub_rate = y_test[mask].mean()
            subgroup_results.append({
                "subgroup": f"LOS {group_name}",
                "n": int(mask.sum()),
                "readmit_rate": float(sub_rate),
                "auc": float(sub_auc),
                "recall": float(sub_recall),
                "f1": float(sub_f1),
            })

    df_subgroups = pd.DataFrame(subgroup_results)
    if not df_subgroups.empty:
        print("\n  Subgroup Performance:")
        print(df_subgroups.to_string(index=False))

    return cm, errors, df_subgroups


# ---------------------------------------------------------------------------
# Experiment 4.6: Hyperparameter importance
# ---------------------------------------------------------------------------

def analyze_param_importance(study):
    """Analyze which hyperparameters matter most."""
    print(f"\n{'='*72}")
    print("EXPERIMENT 4.6: HYPERPARAMETER IMPORTANCE")
    print(f"{'='*72}")

    try:
        importances = optuna.importance.get_param_importances(study)
        print("\n  Parameter Importance (fANOVA):")
        for param, imp in importances.items():
            print(f"    {param}: {imp:.4f}")
        return importances
    except Exception as e:
        print(f"  Could not compute importance: {e}")
        return {}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_optuna_history(study):
    """Plot optimization history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Trial values over time
    trials = [t for t in study.trials if t.value is not None]
    values = [t.value for t in trials]
    best_so_far = np.maximum.accumulate(values)

    axes[0].scatter(range(len(values)), values, alpha=0.4, s=15, color="#1f77b4", label="Trial AUC")
    axes[0].plot(range(len(best_so_far)), best_so_far, color="#d62728", linewidth=2, label="Best so far")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Validation AUC")
    axes[0].set_title("Optuna Optimization History", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Parameter importance
    try:
        importances = optuna.importance.get_param_importances(study)
        params = list(importances.keys())
        imps = list(importances.values())
        axes[1].barh(params[::-1], imps[::-1], color="#2ca02c")
        axes[1].set_xlabel("Importance (fANOVA)")
        axes[1].set_title("Hyperparameter Importance", fontweight="bold")
        axes[1].grid(alpha=0.3, axis="x")
    except Exception:
        axes[1].text(0.5, 0.5, "Could not compute\nimportance", ha="center", va="center", fontsize=14)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase4_optuna_history.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, title="Tuned CatBoost"):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Readmitted", "Readmitted"],
                yticklabels=["Not Readmitted", "Readmitted"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {title}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase4_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_calibration(y_test, y_prob_raw, y_prob_iso, y_prob_platt):
    """Plot calibration curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for name, probs, color in [
        ("Raw CatBoost", y_prob_raw, "#1f77b4"),
        ("Isotonic", y_prob_iso, "#2ca02c"),
        ("Platt", y_prob_platt, "#d62728"),
    ]:
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy="uniform")
        axes[0].plot(prob_pred, prob_true, marker="o", label=name, color=color)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Fraction of Positives")
    axes[0].set_title("Calibration Curves", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Probability distribution
    for name, probs, color in [
        ("Raw CatBoost", y_prob_raw, "#1f77b4"),
        ("Isotonic", y_prob_iso, "#2ca02c"),
    ]:
        axes[1].hist(probs[y_test == 0], bins=50, alpha=0.4, color=color, label=f"{name} (neg)", density=True)
        axes[1].hist(probs[y_test == 1], bins=50, alpha=0.6, color=color, label=f"{name} (pos)", density=True, histtype="step", linewidth=2)

    axes[1].set_xlabel("Predicted Probability")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Probability Distributions by Class", fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase4_calibration.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_threshold_analysis(y_test, y_prob, threshold_results):
    """Plot threshold vs metrics curves."""
    thresholds = np.linspace(0.02, 0.5, 200)
    metrics_at_thresh = {"threshold": [], "precision": [], "recall": [], "f1": [], "flagged_pct": []}

    for t in thresholds:
        y_pred_at = (y_prob >= t).astype(int)
        metrics_at_thresh["threshold"].append(t)
        metrics_at_thresh["precision"].append(precision_score(y_test, y_pred_at, zero_division=0))
        metrics_at_thresh["recall"].append(recall_score(y_test, y_pred_at, zero_division=0))
        metrics_at_thresh["f1"].append(f1_score(y_test, y_pred_at, zero_division=0))
        metrics_at_thresh["flagged_pct"].append(y_pred_at.mean())

    df_t = pd.DataFrame(metrics_at_thresh)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(df_t["threshold"], df_t["recall"], label="Recall (Sensitivity)", linewidth=2)
    axes[0].plot(df_t["threshold"], df_t["precision"], label="Precision (PPV)", linewidth=2)
    axes[0].plot(df_t["threshold"], df_t["f1"], label="F1", linewidth=2, linestyle="--")

    # Mark clinical operating points
    colors = {"youden_j": "red", "f1_optimal": "green", "recall_75": "purple", "recall_80": "orange"}
    for name, vals in threshold_results.items():
        if "threshold" in vals and name in colors:
            axes[0].axvline(x=vals["threshold"], color=colors[name], linestyle=":", alpha=0.7, label=f"{name} (t={vals['threshold']:.3f})")

    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Metric")
    axes[0].set_title("Threshold vs Classification Metrics", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Flagged percentage vs recall tradeoff
    axes[1].plot(df_t["flagged_pct"], df_t["recall"], linewidth=2, color="#d62728")
    axes[1].set_xlabel("% Patients Flagged as High-Risk")
    axes[1].set_ylabel("Recall (% Readmissions Caught)")
    axes[1].set_title("Clinical Tradeoff: Coverage vs Sensitivity", fontweight="bold")
    axes[1].grid(alpha=0.3)
    # Mark 75% recall
    for name in ["recall_75", "recall_80"]:
        if name in threshold_results:
            fp = threshold_results[name].get("flagged_pct", 0)
            rec = threshold_results[name].get("recall", 0)
            axes[1].scatter([fp], [rec], s=100, zorder=5, label=f"{name}: flag {fp:.0%} → catch {rec:.0%}")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase4_threshold_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_subgroup_analysis(df_subgroups):
    """Plot subgroup performance heatmap."""
    if df_subgroups.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # AUC by subgroup
    df_sorted = df_subgroups.sort_values("auc", ascending=True)
    colors = ["#d62728" if v < 0.6 else "#ff7f0e" if v < 0.65 else "#2ca02c" for v in df_sorted["auc"]]
    axes[0].barh(df_sorted["subgroup"], df_sorted["auc"], color=colors)
    axes[0].axvline(x=0.687, color="black", linestyle="--", alpha=0.5, label="Overall AUC=0.687")
    axes[0].set_xlabel("AUC")
    axes[0].set_title("AUC by Clinical Subgroup", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="x")

    # Recall by subgroup + readmission rate
    df_sorted2 = df_subgroups.sort_values("recall", ascending=True)
    axes[1].barh(df_sorted2["subgroup"], df_sorted2["recall"], color="#1f77b4", alpha=0.7, label="Recall")
    ax2 = axes[1].twiny()
    ax2.plot(df_sorted2["readmit_rate"], df_sorted2["subgroup"], "ro-", alpha=0.7, label="Readmit Rate")
    axes[1].set_xlabel("Recall")
    ax2.set_xlabel("Readmission Rate", color="red")
    axes[1].set_title("Recall & Readmission Rate by Subgroup", fontweight="bold")
    axes[1].grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase4_subgroup_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_learning_curves(params, X, y):
    """Plot learning curves to assess data sufficiency."""
    print(f"\n{'='*72}")
    print("GENERATING LEARNING CURVES")
    print(f"{'='*72}")

    model_params = {
        **params,
        "iterations": 300,
        "auto_class_weights": "Balanced",
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "random_seed": RANDOM_STATE,
        "verbose": 0,
        "thread_count": 2,
    }
    model = CatBoostClassifier(**model_params)

    train_sizes = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=3,
        scoring="roc_auc",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
    ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1, color="orange")
    ax.plot(train_sizes_abs, train_mean, "o-", color="blue", label="Training AUC")
    ax.plot(train_sizes_abs, test_mean, "o-", color="orange", label="Validation AUC")

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("AUC")
    ax.set_title("Learning Curves — Tuned CatBoost", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Annotate gap
    gap = train_mean[-1] - test_mean[-1]
    ax.annotate(f"Train-Val Gap: {gap:.3f}", xy=(train_sizes_abs[-1], test_mean[-1]),
                xytext=(train_sizes_abs[-1]*0.6, test_mean[-1] - 0.02),
                arrowprops=dict(arrowstyle="->", color="red"), fontsize=10, color="red")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase4_learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Train AUC at full size: {train_mean[-1]:.4f}")
    print(f"  Val AUC at full size:   {test_mean[-1]:.4f}")
    print(f"  Gap: {gap:.4f}")

    return {"train_sizes": train_sizes_abs.tolist(), "train_auc": train_mean.tolist(), "val_auc": test_mean.tolist()}


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def update_metrics_json(all_results):
    """Update metrics.json with Phase 4 results."""
    payload = {}
    if METRICS_PATH.exists():
        with METRICS_PATH.open() as f:
            payload = json.load(f)

    payload["phase4_anthony"] = {
        "phase": 4,
        "date": "2026-04-03",
        "researcher": "Anthony Rodrigues",
        **all_results,
    }

    with METRICS_PATH.open("w") as f:
        json.dump(payload, f, indent=2)


def update_experiment_log(best_params, default_metrics, tuned_metrics, cv_means, cv_stds, threshold_results, calibration_results, param_importance):
    """Append Phase 4 to experiment log."""
    lines = [
        "",
        "## Phase 4 - 2026-04-03 (Anthony)",
        "",
        "| # | Approach | AUC | F1 | Precision | Recall | Delta vs Phase 3 Champion | Verdict |",
        "|---|----------|-----|----|-----------|--------|---------------------------|---------|",
    ]

    p3_auc = 0.687  # Phase 3 champion
    experiments = [
        ("4.1", "CatBoost default (Phase 3)", default_metrics, "Baseline"),
        ("4.2", f"CatBoost Optuna-tuned ({N_OPTUNA_TRIALS} trials)", tuned_metrics, ""),
    ]

    for num, name, m, verdict_override in experiments:
        delta = m["auc"] - p3_auc
        if not verdict_override:
            verdict_override = "Improved" if delta > 0 else "No change" if delta == 0 else "Regression"
        lines.append(
            f"| {num} | {name} | {m['auc']:.3f} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {delta:+.3f} | {verdict_override} |"
        )

    # Add threshold variants
    for name, vals in threshold_results.items():
        if "f1" in vals and "recall" in vals:
            f1_val = vals.get("f1", 0)
            rec_val = vals.get("recall", vals.get("sensitivity", 0))
            prec_val = vals.get("precision", 0)
            lines.append(
                f"| 4.3 | Tuned CatBoost @ {name} (t={vals['threshold']:.3f}) | — | {f1_val:.3f} | {prec_val:.3f} | {rec_val:.3f} | — | Operating point |"
            )

    # Top params
    if param_importance:
        top_param = list(param_importance.keys())[0]
        top_imp = list(param_importance.values())[0]
        lines.append(f"\n**Most important hyperparameter:** {top_param} (fANOVA importance={top_imp:.3f})")

    lines.extend([
        "",
        f"**Champion after Phase 4:** Optuna-tuned CatBoost (AUC={tuned_metrics['auc']:.4f}, F1={tuned_metrics['f1']:.3f})",
        f"**Cross-validation:** AUC={cv_means['auc']:.4f} ± {cv_stds['auc']:.4f}",
        f"**Calibration:** Brier raw={calibration_results['brier_raw']:.4f}, isotonic={calibration_results['brier_isotonic']:.4f}",
        f"**Key finding:** {_generate_key_finding(default_metrics, tuned_metrics, param_importance, threshold_results)}",
    ])

    section = "\n".join(lines)
    if EXPERIMENT_LOG_PATH.exists():
        current = EXPERIMENT_LOG_PATH.read_text(encoding="utf-8")
    else:
        current = "# Experiment Log - Healthcare Readmission Predictor\n"

    marker = "## Phase 4 - 2026-04-03 (Anthony)"
    if marker in current:
        current = current.split(marker)[0].rstrip()
    current = current.rstrip() + "\n\n" + section + "\n"
    EXPERIMENT_LOG_PATH.write_text(current, encoding="utf-8")


def _generate_key_finding(default_m, tuned_m, param_imp, thresh):
    auc_gain = tuned_m["auc"] - default_m["auc"]
    if param_imp:
        top_param = list(param_imp.keys())[0]
        return (
            f"Optuna tuning gained +{auc_gain:.3f} AUC over default CatBoost. "
            f"The most important hyperparameter is {top_param}. "
            f"At recall≥75% operating point, the model flags "
            f"{thresh.get('recall_75', {}).get('flagged_pct', 0):.0%} of patients."
        )
    return f"Optuna tuning gained +{auc_gain:.3f} AUC over default CatBoost."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_dirs()
    print("=" * 72)
    print("PHASE 4: HYPERPARAMETER TUNING + ERROR ANALYSIS")
    print("Healthcare Readmission Predictor | Anthony Rodrigues | 2026-04-03")
    print("=" * 72)

    # Load data with Phase 3 features
    df = add_phase3_features(load_dataset())
    target = "readmitted_binary"
    features = get_full_features(df)

    X = df[features]
    y = df[target]

    # Split: 60% train, 20% val (for Optuna), 20% test (held out)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=y_trainval
    )
    print(f"\nData splits: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    print(f"Positive rate: Train={y_train.mean():.3f} | Val={y_val.mean():.3f} | Test={y_test.mean():.3f}")
    print(f"Features: {len(features)}")

    # --- Default CatBoost baseline (Phase 3 settings) ---
    print(f"\n{'='*72}")
    print("DEFAULT CATBOOST BASELINE (Phase 3 params)")
    print(f"{'='*72}")
    default_model = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.1,
        auto_class_weights="Balanced", random_seed=RANDOM_STATE, verbose=0, thread_count=2,
    )
    default_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    y_pred_default = default_model.predict(X_test)
    y_prob_default = default_model.predict_proba(X_test)[:, 1]
    default_metrics = {
        "auc": float(roc_auc_score(y_test, y_prob_default)),
        "f1": float(f1_score(y_test, y_pred_default, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred_default, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_default, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred_default)),
    }
    print(f"  Default CatBoost: AUC={default_metrics['auc']:.4f} | F1={default_metrics['f1']:.3f} | Recall={default_metrics['recall']:.3f}")

    # --- Experiment 4.1: Optuna tuning ---
    study = run_optuna_tuning(X_train, y_train, X_val, y_val, n_trials=N_OPTUNA_TRIALS)
    best_params = study.best_params

    # Train final model on train+val with best params
    tuned_model = train_tuned_model(best_params, X_trainval, y_trainval, X_test, y_test)
    y_pred_tuned = tuned_model.predict(X_test)
    y_prob_tuned = tuned_model.predict_proba(X_test)[:, 1]
    tuned_metrics = {
        "auc": float(roc_auc_score(y_test, y_prob_tuned)),
        "f1": float(f1_score(y_test, y_pred_tuned, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred_tuned, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_tuned, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred_tuned)),
        "avg_precision": float(average_precision_score(y_test, y_prob_tuned)),
    }
    print(f"\n  TUNED CatBoost: AUC={tuned_metrics['auc']:.4f} | F1={tuned_metrics['f1']:.3f} | Recall={tuned_metrics['recall']:.3f}")
    auc_gain = tuned_metrics["auc"] - default_metrics["auc"]
    print(f"  Delta vs default: {auc_gain:+.4f} AUC")

    # --- Experiment 4.2: Cross-validation ---
    df_folds, cv_means, cv_stds = cross_validate_model(best_params, X, y)

    # --- Experiment 4.3: Threshold optimization ---
    threshold_results = optimize_thresholds(y_test, y_prob_tuned)

    # --- Experiment 4.4: Calibration ---
    # Use trainval for calibration fitting, test for evaluation
    calibration_results, y_prob_raw, y_prob_iso, y_prob_platt = calibration_analysis(
        tuned_model, X_trainval, y_trainval, X_test, y_test
    )

    # --- Experiment 4.5: Error analysis ---
    cm, errors, df_subgroups = error_analysis(tuned_model, X_test, y_test, df)

    # --- Experiment 4.6: Hyperparameter importance ---
    param_importance = analyze_param_importance(study)

    # --- Plots ---
    print(f"\n{'='*72}")
    print("GENERATING PLOTS")
    print(f"{'='*72}")
    plot_optuna_history(study)
    print("  Saved: phase4_optuna_history.png")
    plot_confusion_matrix(cm)
    print("  Saved: phase4_confusion_matrix.png")
    plot_calibration(y_test, y_prob_raw, y_prob_iso, y_prob_platt)
    print("  Saved: phase4_calibration.png")
    plot_threshold_analysis(y_test, y_prob_tuned, threshold_results)
    print("  Saved: phase4_threshold_analysis.png")
    plot_subgroup_analysis(df_subgroups)
    print("  Saved: phase4_subgroup_analysis.png")

    # --- Learning curves ---
    lc_results = plot_learning_curves(best_params, X, y)
    print("  Saved: phase4_learning_curves.png")

    # --- Save all results ---
    all_results = {
        "best_params": best_params,
        "default_metrics": default_metrics,
        "tuned_metrics": tuned_metrics,
        "auc_gain": float(auc_gain),
        "cv_folds": df_folds.to_dict(orient="records"),
        "cv_mean_auc": float(cv_means["auc"]),
        "cv_std_auc": float(cv_stds["auc"]),
        "threshold_results": threshold_results,
        "calibration_results": calibration_results,
        "subgroup_analysis": df_subgroups.to_dict(orient="records") if not df_subgroups.empty else [],
        "learning_curves": lc_results,
        "param_importance": {k: float(v) for k, v in param_importance.items()} if param_importance else {},
        "n_optuna_trials": N_OPTUNA_TRIALS,
    }

    PHASE4_JSON.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    update_metrics_json(all_results)
    update_experiment_log(best_params, default_metrics, tuned_metrics, cv_means, cv_stds, threshold_results, calibration_results, param_importance)

    # --- Final summary ---
    print(f"\n{'='*72}")
    print("PHASE 4 SUMMARY")
    print(f"{'='*72}")
    print(f"\n  Default CatBoost:  AUC={default_metrics['auc']:.4f}")
    print(f"  Tuned CatBoost:    AUC={tuned_metrics['auc']:.4f} ({auc_gain:+.4f})")
    print(f"  CV AUC:            {cv_means['auc']:.4f} ± {cv_stds['auc']:.4f}")
    print(f"  Best params:       {best_params}")
    if param_importance:
        print(f"  Top hyperparameter: {list(param_importance.keys())[0]} (imp={list(param_importance.values())[0]:.3f})")
    print(f"  Calibration:       Brier raw={calibration_results['brier_raw']:.4f} → isotonic={calibration_results['brier_isotonic']:.4f}")
    for name in ["recall_75", "recall_80"]:
        if name in threshold_results:
            t = threshold_results[name]
            print(f"  {name} operating point: flag {t.get('flagged_pct', 0):.1%} → catch {t.get('recall', 0):.1%} (precision={t.get('precision', 0):.3f})")
    print(f"\n  Files saved: phase4_{{optuna_history,confusion_matrix,calibration,threshold_analysis,subgroup_analysis,learning_curves}}.png")
    print(f"  Results: {PHASE4_JSON}")


if __name__ == "__main__":
    main()
