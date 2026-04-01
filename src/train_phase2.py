"""
Phase 2: Multi-Model Experiment — Healthcare Readmission Predictor
6 models across 3 paradigms, with domain-feature ablation on top-3.

Research question: Do non-linear models finally unlock the domain features
(LACE, Charlson, polypharmacy) that added ZERO value to logistic regression?

Models:
  2.1  Decision Tree (tree baseline)
  2.2  Random Forest
  2.3  XGBoost
  2.4  LightGBM
  2.5  CatBoost
  2.6  SVM (linear kernel — alternative linear paradigm vs LR)

For each model, run TWO variants:
  A) Raw features only (43 features from Phase 1)
  B) Raw + domain-engineered features (LACE, Charlson, polypharmacy, etc.)

PRIMARY metric: Sensitivity (recall for readmitted class).
Catching missed readmissions costs lives and dollars in real hospitals.
Secondary: AUC-ROC, F1, Precision.
"""
import os
import sys
import json
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import get_project_root, load_config, setup_logger
from src.data_pipeline import run_pipeline
from src.feature_engineering import engineer_features
from src.evaluate import compute_metrics, print_metrics

logger = setup_logger(__name__)

RANDOM_STATE = 42
SCALE_POS_WEIGHT = 5.7   # ~85/15 imbalance ratio


def prep_domain_features(X_train, X_test):
    """Apply feature engineering and align columns."""
    X_tr = engineer_features(X_train).fillna(0)
    X_te = engineer_features(X_test).fillna(0)
    for col in X_tr.select_dtypes("object").columns:
        X_tr[col] = pd.Categorical(X_tr[col]).codes
    for col in X_te.select_dtypes("object").columns:
        X_te[col] = pd.Categorical(X_te[col]).codes
    shared = [c for c in X_tr.columns if c in X_te.columns]
    return X_tr[shared], X_te[shared]


def run_experiment(model, X_train, y_train, X_test, y_test, name, train_timer=True):
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t1 = time.time()
    y_pred = model.predict(X_test)
    infer_time_ms = (time.time() - t1) / len(X_test) * 1000

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    else:
        y_prob = None

    metrics = compute_metrics(y_test, y_pred, y_prob, model_name=name)
    metrics["train_time_s"] = round(train_time, 2)
    metrics["infer_ms_per_sample"] = round(infer_time_ms, 4)
    metrics["n_features"] = X_train.shape[1]
    return metrics, model


def build_models():
    """Return list of (id, label, model) tuples."""
    return [
        (
            "2.1_dt",
            "Decision Tree",
            DecisionTreeClassifier(
                max_depth=10, min_samples_leaf=20,
                class_weight="balanced", random_state=RANDOM_STATE
            ),
        ),
        (
            "2.2_rf",
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_leaf=10,
                class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
            ),
        ),
        (
            "2.3_xgb",
            "XGBoost",
            XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=SCALE_POS_WEIGHT,
                eval_metric="logloss", use_label_encoder=False,
                n_jobs=-1, random_state=RANDOM_STATE, verbosity=0
            ),
        ),
        (
            "2.4_lgbm",
            "LightGBM",
            LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                class_weight="balanced",
                n_jobs=-1, random_state=RANDOM_STATE, verbose=-1
            ),
        ),
        (
            "2.5_catboost",
            "CatBoost",
            CatBoostClassifier(
                iterations=300, depth=6, learning_rate=0.05,
                auto_class_weights="Balanced",
                random_seed=RANDOM_STATE, verbose=0
            ),
        ),
        (
            "2.6_svm",
            "SVM (linear)",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", CalibratedClassifierCV(
                    LinearSVC(C=0.1, class_weight="balanced",
                              max_iter=2000, random_state=RANDOM_STATE),
                    cv=3
                )),
            ]),
        ),
    ]


def plot_model_comparison(results_raw, results_dom, save_path):
    labels = [r["model"].replace("_raw", "").replace("_dom", "") for r in results_raw]
    labels_clean = [
        l.replace("2.1_dt", "Decision Tree")
         .replace("2.2_rf", "Random Forest")
         .replace("2.3_xgb", "XGBoost")
         .replace("2.4_lgbm", "LightGBM")
         .replace("2.5_catboost", "CatBoost")
         .replace("2.6_svm", "SVM")
        for l in labels
    ]

    metrics_to_plot = ["recall_sensitivity", "auc_roc", "f1", "precision"]
    titles = ["Sensitivity (PRIMARY)", "AUC-ROC", "F1 Score", "Precision"]
    colors_raw = "#4C72B0"
    colors_dom = "#DD8452"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phase 2: 6-Model Comparison — Raw vs Domain Features\n(Healthcare Readmission Predictor)", fontsize=13, fontweight="bold")

    for ax, metric, title in zip(axes.flat, metrics_to_plot, titles):
        vals_raw = [r.get(metric, 0) or 0 for r in results_raw]
        vals_dom = [r.get(metric, 0) or 0 for r in results_dom]
        x = np.arange(len(labels_clean))
        width = 0.35
        bars_r = ax.bar(x - width/2, vals_raw, width, label="Raw features", color=colors_raw, alpha=0.85)
        bars_d = ax.bar(x + width/2, vals_dom, width, label="+ Domain features", color=colors_dom, alpha=0.85)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_clean, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        for bar in bars_r:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=7)
        for bar in bars_d:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparison chart saved to {save_path}")


def plot_sensitivity_ranking(results_raw, results_dom, lace_sensitivity, lr_sensitivity, save_path):
    """Bar chart ranking models by sensitivity vs LACE vs LR baselines."""
    entries = []
    for r, d in zip(results_raw, results_dom):
        label = r["model"].replace("_raw", "")
        entries.append((label + " (raw)", r.get("recall_sensitivity", 0) or 0, "#4C72B0"))
        entries.append((label + " (+domain)", d.get("recall_sensitivity", 0) or 0, "#DD8452"))

    entries.append(("LACE ≥10 (clinical std)", lace_sensitivity, "#2ca02c"))
    entries.append(("LR baseline (Phase 1)", lr_sensitivity, "#9467bd"))

    entries.sort(key=lambda x: x[1], reverse=True)

    labels = [e[0] for e in entries]
    vals = [e[1] for e in entries]
    colors = [e[2] for e in entries]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(labels, vals, color=colors, alpha=0.85)
    ax.axvline(x=lace_sensitivity, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"LACE benchmark ({lace_sensitivity:.3f})")
    ax.axvline(x=lr_sensitivity, color="#9467bd", linestyle=":", linewidth=1.5, label=f"LR Phase 1 ({lr_sensitivity:.3f})")
    ax.set_xlabel("Sensitivity (Recall for Readmitted Class)", fontsize=11)
    ax.set_title("Phase 2: Sensitivity Ranking — Can We Beat 0.51?\n(Primary metric: catching readmissions before discharge)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Sensitivity ranking saved to {save_path}")


def plot_domain_feature_delta(results_raw, results_dom, save_path):
    """Show Δ sensitivity from adding domain features per model."""
    labels = [r["model"].replace("_raw", "") for r in results_raw]
    labels_clean = [
        l.replace("2.1_dt", "DT").replace("2.2_rf", "RF")
         .replace("2.3_xgb", "XGB").replace("2.4_lgbm", "LGBM")
         .replace("2.5_catboost", "CatBoost").replace("2.6_svm", "SVM")
        for l in labels
    ]
    deltas_sens = [(d.get("recall_sensitivity") or 0) - (r.get("recall_sensitivity") or 0) for r, d in zip(results_raw, results_dom)]
    deltas_auc  = [(d.get("auc_roc") or 0) - (r.get("auc_roc") or 0) for r, d in zip(results_raw, results_dom)]

    x = np.arange(len(labels_clean))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    c_pos = "#2ca02c"
    c_neg = "#d62728"
    bar_colors_s = [c_pos if v >= 0 else c_neg for v in deltas_sens]
    bar_colors_a = [c_pos if v >= 0 else c_neg for v in deltas_auc]
    ax.bar(x - width/2, deltas_sens, width, color=bar_colors_s, alpha=0.85, label="Δ Sensitivity")
    ax.bar(x + width/2, deltas_auc, width, color=bar_colors_a, alpha=0.5, label="Δ AUC-ROC")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_clean, fontsize=10)
    ax.set_ylabel("Δ Metric (domain − raw)")
    ax.set_title("Do Domain Features Help Non-Linear Models?\n(green = yes, red = no)", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (ds, da) in enumerate(zip(deltas_sens, deltas_auc)):
        ax.text(i - width/2, ds + (0.002 if ds >= 0 else -0.006), f"{ds:+.3f}", ha="center", fontsize=8)
        ax.text(i + width/2, da + (0.002 if da >= 0 else -0.006), f"{da:+.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Domain feature delta plot saved to {save_path}")


def print_leaderboard(results_raw, results_dom):
    print("\n" + "=" * 105)
    print(f"{'PHASE 2 HEAD-TO-HEAD LEADERBOARD':^105}")
    print("=" * 105)
    fmt = "{:<30s} {:>8s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}"
    print(fmt.format("Model", "Features", "Sensitivity", "AUC-ROC", "F1", "Precision", "AUC-PR", "Train(s)"))
    print("-" * 105)

    all_results = []
    for r, d in zip(results_raw, results_dom):
        all_results.append((r, "raw"))
        all_results.append((d, "domain"))
    all_results.sort(key=lambda x: x[0].get("recall_sensitivity") or 0, reverse=True)

    for res, feat_type in all_results:
        name = res["model"].replace("_raw", "").replace("_dom", "")
        print(fmt.format(
            name[:30], feat_type,
            f"{res.get('recall_sensitivity', 0) or 0:.4f}",
            f"{res.get('auc_roc', 0) or 0:.4f}",
            f"{res.get('f1', 0) or 0:.4f}",
            f"{res.get('precision', 0) or 0:.4f}",
            f"{res.get('auc_pr', 0) or 0:.4f}",
            f"{res.get('train_time_s', 0):.1f}",
        ))
    print("=" * 105)


def main():
    root = get_project_root()
    config = load_config()
    plots_path = root / config["results"]["plots_path"]
    models_path = root / config["results"]["models_path"]
    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    processed_path = root / config["data"]["processed_path"]
    if (processed_path / "X_train.parquet").exists():
        logger.info("Loading preprocessed splits from parquet")
        X_train = pd.read_parquet(processed_path / "X_train.parquet")
        X_test  = pd.read_parquet(processed_path / "X_test.parquet")
        y_train = pd.read_parquet(processed_path / "y_train.parquet").squeeze()
        y_test  = pd.read_parquet(processed_path / "y_test.parquet").squeeze()
    else:
        X_train, X_test, y_train, y_test = run_pipeline(config)

    logger.info(f"Raw feature set: {X_train.shape[1]} features, {len(X_train)} train / {len(X_test)} test samples")
    logger.info(f"Positive rate — Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")

    X_train_dom, X_test_dom = prep_domain_features(X_train, X_test)
    logger.info(f"Domain feature set: {X_train_dom.shape[1]} features")

    models = build_models()
    results_raw = []
    results_dom = []

    for exp_id, label, model_template in models:
        logger.info(f"\n{'='*55}")
        logger.info(f"  Experiment {exp_id}: {label} — raw features")
        import copy
        m_raw = copy.deepcopy(model_template)
        m_dom = copy.deepcopy(model_template)

        metrics_r, fitted_raw = run_experiment(m_raw, X_train, y_train, X_test, y_test,
                                               name=f"{exp_id}_raw")
        print_metrics(metrics_r)
        results_raw.append(metrics_r)

        logger.info(f"  Experiment {exp_id}: {label} — +domain features")
        metrics_d, fitted_dom = run_experiment(m_dom, X_train_dom, y_train, X_test_dom, y_test,
                                               name=f"{exp_id}_dom")
        print_metrics(metrics_d)
        results_dom.append(metrics_d)

        model_file = models_path / f"{exp_id}_best.joblib"
        best_model = fitted_dom if (metrics_d.get("recall_sensitivity") or 0) >= (metrics_r.get("recall_sensitivity") or 0) else fitted_raw
        joblib.dump(best_model, model_file)

    print_leaderboard(results_raw, results_dom)

    # === PLOTS ===
    plot_model_comparison(results_raw, results_dom, plots_path / "phase2_model_comparison.png")
    plot_sensitivity_ranking(results_raw, results_dom,
                             lace_sensitivity=0.146, lr_sensitivity=0.512,
                             save_path=plots_path / "phase2_sensitivity_ranking.png")
    plot_domain_feature_delta(results_raw, results_dom, plots_path / "phase2_domain_feature_delta.png")

    # === SAVE METRICS ===
    phase2_metrics = {
        "phase": 2,
        "date": "2026-04-01",
        "raw_feature_count": X_train.shape[1],
        "domain_feature_count": X_train_dom.shape[1],
        "results_raw_features": results_raw,
        "results_domain_features": results_dom,
        "baselines": {
            "lace_sensitivity": 0.146,
            "lr_raw_sensitivity": 0.512,
            "lr_raw_auc": 0.569,
        }
    }

    metrics_path = root / "results" / "phase2_model_comparison.json"
    with open(metrics_path, "w") as f:
        json.dump(phase2_metrics, f, indent=2)
    logger.info(f"Phase 2 metrics saved to {metrics_path}")

    # === DOMAIN FEATURE IMPACT SUMMARY ===
    print("\n" + "=" * 65)
    print("DOMAIN FEATURE IMPACT: Does engineering help non-linear models?")
    print("=" * 65)
    for r, d in zip(results_raw, results_dom):
        name = r["model"].replace("_raw", "")
        ds = (d.get("recall_sensitivity") or 0) - (r.get("recall_sensitivity") or 0)
        da = (d.get("auc_roc") or 0) - (r.get("auc_roc") or 0)
        verdict = "YES (+)" if ds > 0.005 else ("HURTS" if ds < -0.005 else "NEUTRAL")
        print(f"  {name:<25s}  ΔSensitivity={ds:+.4f}  ΔAUC={da:+.4f}  → {verdict}")

    # Winner summary
    best_raw = max(results_raw, key=lambda x: x.get("recall_sensitivity") or 0)
    best_dom = max(results_dom, key=lambda x: x.get("recall_sensitivity") or 0)
    overall_best = best_raw if (best_raw.get("recall_sensitivity") or 0) >= (best_dom.get("recall_sensitivity") or 0) else best_dom

    print(f"\n{'='*65}")
    print(f"PHASE 2 CHAMPION: {overall_best['model']}")
    print(f"  Sensitivity: {overall_best.get('recall_sensitivity'):.4f}")
    print(f"  AUC-ROC:     {overall_best.get('auc_roc'):.4f}")
    print(f"  F1:          {overall_best.get('f1'):.4f}")
    print(f"  vs LACE:     +{overall_best.get('recall_sensitivity', 0) - 0.146:.4f} sensitivity")
    print(f"  vs LR Phase1:+{overall_best.get('recall_sensitivity', 0) - 0.512:.4f} sensitivity")
    print(f"{'='*65}")

    return phase2_metrics


if __name__ == "__main__":
    main()
