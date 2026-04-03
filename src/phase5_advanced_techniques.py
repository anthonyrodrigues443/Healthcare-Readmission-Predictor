"""Phase 5 advanced techniques for healthcare readmission."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

from src.data_pipeline import clean_and_engineer, download_dataset
from src.phase3_feature_engineering import add_phase3_features, get_feature_sets

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
RESULTS_DIR = Path("results")
PROCESSED_PATH = Path("data/processed/readmission_processed.csv")
PHASE4_JSON = RESULTS_DIR / "phase4_tuning_results.json"
PHASE5_JSON = RESULTS_DIR / "phase5_advanced_techniques.json"
RULES_PATH = RESULTS_DIR / "phase5_extracted_rules.txt"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_df() -> pd.DataFrame:
    if PROCESSED_PATH.exists():
        df = pd.read_csv(PROCESSED_PATH)
    else:
        df = clean_and_engineer(download_dataset())
        df.to_csv(PROCESSED_PATH, index=False)
    return add_phase3_features(df)


def load_best_params() -> dict[str, float]:
    return json.loads(PHASE4_JSON.read_text(encoding="utf-8"))["best_params"]


def build_model(best_params: dict[str, float]) -> CatBoostClassifier:
    return CatBoostClassifier(
        **best_params,
        iterations=1500,
        auto_class_weights="Balanced",
        eval_metric="AUC",
        loss_function="Logloss",
        od_type="Iter",
        od_wait=50,
        random_seed=RANDOM_STATE,
        verbose=0,
        thread_count=2,
    )


def calibrate(model: CatBoostClassifier, X_cal: pd.DataFrame, y_cal: pd.Series) -> CalibratedClassifierCV:
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    calibrated.fit(X_cal, y_cal)
    return calibrated


def best_threshold(y_true: pd.Series, y_prob: np.ndarray) -> float:
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-8)
    best_idx = int(np.nanargmax(f1_scores))
    return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5


def metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return metrics_from_pred(y_true, y_prob, y_pred)


def metrics_from_pred(y_true: pd.Series, y_prob: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "flagged_pct": float(y_pred.mean()),
    }


def latency_ms(predict_fn, frame: pd.DataFrame) -> float:
    sample = frame.iloc[:100]
    start = perf_counter()
    for _ in range(5):
        predict_fn(sample)
    return float((perf_counter() - start) / 5 / len(sample) * 1000)


def fit_calibrated(
    best_params: dict[str, float],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    sample_weight: np.ndarray | None = None,
) -> tuple[CatBoostClassifier, CalibratedClassifierCV]:
    model = build_model(best_params)
    fit_kwargs = {"eval_set": (X_cal, y_cal), "early_stopping_rounds": 50}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(X_train, y_train, **fit_kwargs)
    return model, calibrate(model, X_cal, y_cal)


def summarize(name: str, X_all: pd.DataFrame, y_all: pd.Series, y_cal_true: pd.Series, cal_prob: np.ndarray, test_prob: np.ndarray, threshold: float, note: str, lat_ms: float) -> dict[str, object]:
    low_mask = X_all["prior_utilization"] == 0
    hi_mask = X_all["prior_utilization"] >= 4
    return {
        "model": name,
        "threshold": threshold,
        "latency_ms": lat_ms,
        "notes": note,
        "overall": metrics(y_all, test_prob, threshold),
        "low_util": metrics(y_all[low_mask], test_prob[low_mask], threshold),
        "high_util": metrics(y_all[hi_mask], test_prob[hi_mask], threshold),
        "calibration_auc": float(roc_auc_score(y_cal_true, cal_prob)),
        "calibration_brier": float(brier_score_loss(y_cal_true, cal_prob)),
    }


def summarize_group_threshold(
    name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_cal_true: pd.Series,
    cal_prob: np.ndarray,
    test_prob: np.ndarray,
    default_threshold: float,
    low_util_threshold: float,
    note: str,
    lat_ms: float,
) -> dict[str, object]:
    low_mask = X_test["prior_utilization"] == 0
    hi_mask = X_test["prior_utilization"] >= 4
    y_pred = (test_prob >= default_threshold).astype(int)
    y_pred[low_mask.to_numpy()] = (test_prob[low_mask.to_numpy()] >= low_util_threshold).astype(int)
    return {
        "model": name,
        "threshold": default_threshold,
        "low_util_threshold": low_util_threshold,
        "latency_ms": lat_ms,
        "notes": note,
        "overall": metrics_from_pred(y_test, test_prob, y_pred),
        "low_util": metrics(y_test[low_mask], test_prob[low_mask], low_util_threshold),
        "high_util": metrics(y_test[hi_mask], test_prob[hi_mask], default_threshold),
        "calibration_auc": float(roc_auc_score(y_cal_true, cal_prob)),
        "calibration_brier": float(brier_score_loss(y_cal_true, cal_prob)),
    }


def hybrid_probs(frame: pd.DataFrame, base_model: CalibratedClassifierCV, specialist: CalibratedClassifierCV, base_cols: list[str], spec_cols: list[str]) -> np.ndarray:
    probs = base_model.predict_proba(frame[base_cols])[:, 1]
    low_mask = frame["prior_utilization"] == 0
    if low_mask.any():
        probs[low_mask.to_numpy()] = specialist.predict_proba(frame.loc[low_mask, spec_cols])[:, 1]
    return probs


def make_plots(results_df: pd.DataFrame, base_model: CatBoostClassifier, full_cols: list[str], spec_model: CatBoostClassifier, spec_cols: list[str], X_low_test: pd.DataFrame, y_low_test: pd.Series, X_cal: pd.DataFrame, hybrid_cal_prob: np.ndarray, X_test: pd.DataFrame, hybrid_test_prob: np.ndarray) -> dict[str, object]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=results_df, x="auc", y="model", ax=axes[0], color="#1f77b4")
    axes[0].set_title("Overall AUC")
    sns.barplot(data=results_df, x="low_util_recall", y="model", ax=axes[1], color="#d95f02")
    axes[1].set_title("Recall on prior_utilization = 0")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase5_model_tradeoffs.png", dpi=150, bbox_inches="tight")
    plt.close()

    base_imp = pd.DataFrame({"feature": full_cols, "importance": base_model.get_feature_importance()}).sort_values("importance", ascending=False).head(12)
    spec_imp = pd.DataFrame({"feature": spec_cols, "importance": spec_model.get_feature_importance()}).sort_values("importance", ascending=False).head(12)
    perm = permutation_importance(spec_model, X_low_test[spec_cols], y_low_test, scoring="roc_auc", n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
    perm_imp = pd.DataFrame({"feature": spec_cols, "importance": perm.importances_mean}).sort_values("importance", ascending=False).head(12)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.barplot(data=base_imp, x="importance", y="feature", ax=axes[0], color="#1f77b4")
    axes[0].set_title("Base model importance")
    sns.barplot(data=spec_imp, x="importance", y="feature", ax=axes[1], color="#d95f02")
    axes[1].set_title("Specialist importance")
    sns.barplot(data=perm_imp, x="importance", y="feature", ax=axes[2], color="#66a61e")
    axes[2].set_title("Specialist permutation importance")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase5_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    surrogate = DecisionTreeRegressor(max_depth=3, min_samples_leaf=400, random_state=RANDOM_STATE)
    surrogate.fit(X_cal[spec_cols], hybrid_cal_prob)
    rules = export_text(surrogate, feature_names=spec_cols, decimals=3)
    RULES_PATH.write_text(rules, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(surrogate, feature_names=spec_cols, filled=True, rounded=True, impurity=False, precision=3, fontsize=9, ax=ax)
    ax.set_title("Surrogate tree for best hybrid")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase5_surrogate_tree.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "base_top_features": base_imp.to_dict(orient="records"),
        "specialist_top_features": spec_imp.to_dict(orient="records"),
        "specialist_permutation": perm_imp.to_dict(orient="records"),
        "surrogate_fidelity_r2": float(surrogate.score(X_test[spec_cols], hybrid_test_prob)),
    }


if __name__ == "__main__":
    ensure_dirs()
    best_params = load_best_params()
    df = load_df()
    feature_sets = get_feature_sets(df)
    full_cols = feature_sets["full_83"]
    clinical_cols = feature_sets["clinical_23"]
    transition_cols = feature_sets["clinical_plus_transition_29"]

    X = df[full_cols]
    y = df["readmitted_binary"]
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    X_train, X_cal, y_train, y_cal = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=y_trainval)

    base_raw, base_cal = fit_calibrated(best_params, X_train[full_cols], y_train, X_cal[full_cols], y_cal)
    base_cal_prob = base_cal.predict_proba(X_cal[full_cols])[:, 1]
    base_test_prob = base_cal.predict_proba(X_test[full_cols])[:, 1]
    base_thr = best_threshold(y_cal, base_cal_prob)

    weights = np.ones(len(X_train))
    weights[(X_train["prior_utilization"].to_numpy() == 0) & (y_train.to_numpy() == 1)] = 3.0
    weighted_raw, weighted_cal = fit_calibrated(best_params, X_train[full_cols], y_train, X_cal[full_cols], y_cal, sample_weight=weights)
    weighted_cal_prob = weighted_cal.predict_proba(X_cal[full_cols])[:, 1]
    weighted_test_prob = weighted_cal.predict_proba(X_test[full_cols])[:, 1]
    weighted_thr = best_threshold(y_cal, weighted_cal_prob)

    low_train = X_train["prior_utilization"] == 0
    low_cal = X_cal["prior_utilization"] == 0
    low_test = X_test["prior_utilization"] == 0

    clin_raw, clin_cal = fit_calibrated(best_params, X_train.loc[low_train, clinical_cols], y_train.loc[low_train], X_cal.loc[low_cal, clinical_cols], y_cal.loc[low_cal])
    clin_cal_prob = clin_cal.predict_proba(X_cal.loc[low_cal, clinical_cols])[:, 1]
    clin_test_prob = clin_cal.predict_proba(X_test.loc[low_test, clinical_cols])[:, 1]
    clin_thr = best_threshold(y_cal.loc[low_cal], clin_cal_prob)

    trans_raw, trans_cal = fit_calibrated(best_params, X_train.loc[low_train, transition_cols], y_train.loc[low_train], X_cal.loc[low_cal, transition_cols], y_cal.loc[low_cal])
    trans_cal_prob = trans_cal.predict_proba(X_cal.loc[low_cal, transition_cols])[:, 1]
    trans_test_prob = trans_cal.predict_proba(X_test.loc[low_test, transition_cols])[:, 1]
    trans_thr = best_threshold(y_cal.loc[low_cal], trans_cal_prob)

    hybrid_clin_cal_prob = hybrid_probs(X_cal, base_cal, clin_cal, full_cols, clinical_cols)
    hybrid_clin_test_prob = hybrid_probs(X_test, base_cal, clin_cal, full_cols, clinical_cols)
    hybrid_clin_thr = best_threshold(y_cal, hybrid_clin_cal_prob)

    hybrid_trans_cal_prob = hybrid_probs(X_cal, base_cal, trans_cal, full_cols, transition_cols)
    hybrid_trans_test_prob = hybrid_probs(X_test, base_cal, trans_cal, full_cols, transition_cols)
    hybrid_trans_thr = best_threshold(y_cal, hybrid_trans_cal_prob)

    experiments = [
        summarize("Tuned CatBoost (full_83)", X_test, y_test, y_cal, base_cal_prob, base_test_prob, base_thr, "Anthony Phase 4 backbone.", latency_ms(lambda f: base_cal.predict_proba(f[full_cols])[:, 1], X_test)),
        summarize("Weighted CatBoost (full_83, low-util positives x3)", X_test, y_test, y_cal, weighted_cal_prob, weighted_test_prob, weighted_thr, "Extra weight on low-util positives.", latency_ms(lambda f: weighted_cal.predict_proba(f[full_cols])[:, 1], X_test)),
        summarize("Hybrid router (clinical_23 specialist)", X_test, y_test, y_cal, hybrid_clin_cal_prob, hybrid_clin_test_prob, hybrid_clin_thr, "Route first-timers to clinical specialist.", latency_ms(lambda f: hybrid_probs(f, base_cal, clin_cal, full_cols, clinical_cols), X_test)),
        summarize("Hybrid router (transition_29 specialist)", X_test, y_test, y_cal, hybrid_trans_cal_prob, hybrid_trans_test_prob, hybrid_trans_thr, "Route first-timers to transition-aware specialist.", latency_ms(lambda f: hybrid_probs(f, base_cal, trans_cal, full_cols, transition_cols), X_test)),
        summarize_group_threshold("Hybrid router + subgroup thresholds (clinical_23)", X_test, y_test, y_cal, hybrid_clin_cal_prob, hybrid_clin_test_prob, base_thr, clin_thr, "Use specialist threshold only for first-timers.", latency_ms(lambda f: hybrid_probs(f, base_cal, clin_cal, full_cols, clinical_cols), X_test)),
        summarize_group_threshold("Hybrid router + subgroup thresholds (transition_29)", X_test, y_test, y_cal, hybrid_trans_cal_prob, hybrid_trans_test_prob, base_thr, trans_thr, "Use transition specialist threshold only for first-timers.", latency_ms(lambda f: hybrid_probs(f, base_cal, trans_cal, full_cols, transition_cols), X_test)),
    ]
    slice_models = [
        {"model": "Clinical specialist only (low-util slice)", "threshold": clin_thr, **metrics(y_test.loc[low_test], clin_test_prob, clin_thr)},
        {"model": "Transition specialist only (low-util slice)", "threshold": trans_thr, **metrics(y_test.loc[low_test], trans_test_prob, trans_thr)},
    ]

    rows = []
    for exp in experiments:
        rows.append(
            {
                "model": exp["model"],
                "accuracy": exp["overall"]["accuracy"],
                "f1": exp["overall"]["f1"],
                "precision": exp["overall"]["precision"],
                "recall": exp["overall"]["recall"],
                "auc": exp["overall"]["auc"],
                "brier": exp["overall"]["brier"],
                "latency_ms": exp["latency_ms"],
                "low_util_recall": exp["low_util"]["recall"],
                "high_util_recall": exp["high_util"]["recall"],
                "notes": exp["notes"],
            }
        )
    results_df = pd.DataFrame(rows)
    explainability = make_plots(results_df, base_raw, full_cols, trans_raw, transition_cols, X_test.loc[low_test], y_test.loc[low_test], X_cal.loc[low_cal], trans_cal_prob, X_test.loc[low_test], trans_test_prob)

    payload = {
        "phase": 5,
        "date": "2026-04-03",
        "researcher": "Mark Rodrigues",
        "dataset_rows": int(len(df)),
        "feature_sets": {"full_83": len(full_cols), "clinical_23": len(clinical_cols), "transition_29": len(transition_cols)},
        "experiments": experiments,
        "overall_table": results_df.to_dict(orient="records"),
        "specialist_slice_models": slice_models,
        "explainability": explainability,
        "frontier_baseline_status": {
            "gpt_5_4": "Blocked: codex exec returned Access is denied",
            "opus_4_6": "Blocked: Claude CLI requires Git Bash on Windows",
        },
    }
    PHASE5_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(results_df.to_string(index=False))
    print(pd.DataFrame(slice_models).to_string(index=False))
    print(f"Saved {PHASE5_JSON}")
