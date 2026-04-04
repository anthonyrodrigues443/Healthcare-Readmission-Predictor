"""Phase 7: native categorical CatBoost experiment for readmission prediction."""

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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_pipeline import classify_icd9, compute_lace_score, download_dataset
from src.production_pipeline import (
    MEDICATION_COLUMNS,
    RANDOM_STATE,
    json_safe_float,
    load_modeling_frame,
    make_production_splits,
    optimal_f1_threshold,
    safe_auc,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results")
REPORTS_DIR = Path("reports")
PHASE4_JSON = RESULTS_DIR / "phase4_tuning_results.json"
PHASE7_JSON = RESULTS_DIR / "phase7_native_categorical.json"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)


def load_best_params() -> dict[str, float]:
    return json.loads(PHASE4_JSON.read_text(encoding="utf-8"))["best_params"]


def build_native_categorical_frame() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Keep raw categorical structure and add the strongest engineered features."""
    raw = download_dataset().copy()
    raw = raw.drop(
        columns=[c for c in ["weight", "payer_code", "medical_specialty", "encounter_id", "patient_nbr"] if c in raw.columns]
    )
    raw = raw.replace("?", np.nan)
    raw["readmitted_binary"] = (raw["readmitted"] == "<30").astype(int)
    raw = raw.drop(columns=["readmitted"])

    for col in ["diag_1", "diag_2", "diag_3"]:
        raw[f"{col}_group"] = raw[col].apply(classify_icd9)

    med_cols = [c for c in MEDICATION_COLUMNS if c in raw.columns]
    raw["n_medications_changed"] = raw[med_cols].apply(
        lambda row: sum(1 for value in row if value in {"Up", "Down"}),
        axis=1,
    )
    raw["n_medications_active"] = raw[med_cols].apply(
        lambda row: sum(1 for value in row if pd.notna(value) and value != "No"),
        axis=1,
    )
    raw["polypharmacy"] = (raw["num_medications"] > 5).astype(int)
    raw["prior_utilization"] = raw["number_outpatient"] + raw["number_emergency"] + raw["number_inpatient"]
    raw["lab_procedure_ratio"] = raw["num_lab_procedures"] / (raw["time_in_hospital"] + 1)
    raw["procedure_ratio"] = raw["num_procedures"] / (raw["time_in_hospital"] + 1)
    raw["A1C_high"] = raw["A1Cresult"].isin([">7", ">8"]).astype(int)
    raw["A1C_tested"] = raw["A1Cresult"].fillna("None").ne("None").astype(int)
    raw["glucose_high"] = raw["max_glu_serum"].isin([">200", ">300"]).astype(int)
    raw["glucose_tested"] = raw["max_glu_serum"].fillna("None").ne("None").astype(int)
    raw["med_changed"] = raw["change"].eq("Ch").astype(int)
    raw["diabetes_med"] = raw["diabetesMed"].eq("Yes").astype(int)
    raw["diabetes_primary"] = raw["diag_1_group"].eq("Diabetes").astype(int)
    raw["lace_score"] = compute_lace_score(raw)
    raw["discharge_post_acute"] = raw["discharge_disposition_id"].isin([2, 3, 4, 5, 6, 22, 23, 24]).astype(int)
    raw["discharge_ama_or_psych"] = raw["discharge_disposition_id"].isin([7, 28]).astype(int)
    raw["discharge_home"] = raw["discharge_disposition_id"].eq(1).astype(int)
    raw["admission_emergency"] = raw["admission_type_id"].eq(1).astype(int)
    raw["admission_transfer_source"] = raw["admission_source_id"].isin([4, 5, 6, 20, 22, 25]).astype(int)
    raw["admission_ed_source"] = raw["admission_source_id"].eq(7).astype(int)
    raw["utilization_band"] = pd.cut(raw["prior_utilization"], bins=[-0.1, 0.5, 1.5, 3.5, np.inf], labels=[0, 1, 2, 3]).astype("Int64")
    raw["acute_prior_load"] = raw["number_inpatient"] * 2 + raw["number_emergency"]
    raw["meds_per_day"] = raw["num_medications"] / (raw["time_in_hospital"] + 1)
    raw["diagnoses_per_day"] = raw["number_diagnoses"] / (raw["time_in_hospital"] + 1)
    raw["glycemic_instability"] = ((raw["A1C_high"] == 1) | (raw["glucose_high"] == 1)).astype(int)
    raw["utilization_x_polypharmacy"] = raw["prior_utilization"] * raw["polypharmacy"]
    raw["utilization_x_transition"] = raw["prior_utilization"] * raw["discharge_post_acute"]
    raw["los_x_med_burden"] = raw["time_in_hospital"] * raw["num_medications"]
    raw["instability_x_utilization"] = raw["glycemic_instability"] * raw["prior_utilization"]

    for col in ["change", "diabetesMed", "A1Cresult", "max_glu_serum"]:
        raw[col] = raw[col].fillna("None").astype(str)
    for col in ["race", "gender", "age", "diag_1", "diag_2", "diag_3", "diag_1_group", "diag_2_group", "diag_3_group"] + med_cols:
        raw[col] = raw[col].fillna("Unknown").astype(str)
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        raw[col] = raw[col].astype(str)

    feature_cols = [c for c in raw.columns if c != "readmitted_binary"]
    categorical_cols = [c for c in feature_cols if str(raw[c].dtype) in {"object", "str", "string"}]
    return raw, feature_cols, categorical_cols


def train_calibrated_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_early: pd.DataFrame,
    y_early: pd.Series,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    best_params: dict[str, float],
    cat_feature_indices: list[int] | None = None,
) -> tuple[CatBoostClassifier, CalibratedClassifierCV, float, float]:
    model = CatBoostClassifier(
        **best_params,
        iterations=1500,
        auto_class_weights="Balanced",
        eval_metric="AUC",
        loss_function="Logloss",
        od_type="Iter",
        od_wait=50,
        random_seed=RANDOM_STATE,
        verbose=0,
        thread_count=4,
    )

    fit_kwargs = {"eval_set": (X_early, y_early), "early_stopping_rounds": 50}
    if cat_feature_indices:
        fit_kwargs["cat_features"] = cat_feature_indices

    t0 = perf_counter()
    model.fit(X_train, y_train, **fit_kwargs)
    train_time = perf_counter() - t0

    calibrator = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrator.fit(X_cal, y_cal)
    cal_prob = calibrator.predict_proba(X_cal)[:, 1]
    threshold = optimal_f1_threshold(y_cal, cal_prob)
    return model, calibrator, threshold, train_time


def evaluate_model(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    probs: np.ndarray,
    threshold: float,
    train_time: float,
) -> dict[str, float | None]:
    preds = (probs >= threshold).astype(int)
    low_mask = X_test["prior_utilization"] == 0
    high_mask = X_test["prior_utilization"] >= 4

    def subgroup_recall(mask: pd.Series) -> float:
        if mask.sum() == 0:
            return 0.0
        return float(recall_score(y_test[mask], preds[mask], zero_division=0))

    sample = X_test.iloc[: min(100, len(X_test))]
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "auc": json_safe_float(safe_auc(y_test, probs)),
        "avg_precision": float(average_precision_score(y_test, probs)),
        "brier": float(brier_score_loss(y_test, probs)),
        "threshold": float(threshold),
        "low_util_recall": subgroup_recall(low_mask),
        "high_util_recall": subgroup_recall(high_mask),
        "flagged_rate": float(preds.mean()),
        "train_time_seconds": round(train_time, 2),
    }


def bootstrap_auc_delta(
    y_true: pd.Series,
    baseline_prob: np.ndarray,
    candidate_prob: np.ndarray,
    n_bootstrap: int = 400,
) -> dict[str, float | None]:
    rng = np.random.default_rng(RANDOM_STATE)
    y_arr = np.asarray(y_true)
    deltas = []
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, len(y_arr), len(y_arr))
        sample_y = y_arr[sample_idx]
        if len(np.unique(sample_y)) < 2:
            continue
        base_auc = safe_auc(sample_y, baseline_prob[sample_idx])
        cand_auc = safe_auc(sample_y, candidate_prob[sample_idx])
        if base_auc is not None and cand_auc is not None:
            deltas.append(cand_auc - base_auc)

    delta_arr = np.asarray(deltas, dtype=float)
    return {
        "n_bootstrap": int(len(delta_arr)),
        "mean_delta_auc": json_safe_float(float(delta_arr.mean()) if len(delta_arr) else None),
        "ci_lower": json_safe_float(float(np.quantile(delta_arr, 0.025)) if len(delta_arr) else None),
        "ci_upper": json_safe_float(float(np.quantile(delta_arr, 0.975)) if len(delta_arr) else None),
        "probability_candidate_beats_baseline": json_safe_float(float((delta_arr > 0).mean()) if len(delta_arr) else None),
    }


def plot_comparison(results_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sns.barplot(data=results_df, x="auc", y="model", ax=axes[0], color="#1f77b4")
    axes[0].set_title("Held-out AUC")
    axes[0].set_xlabel("AUC")
    axes[0].set_ylabel("")

    sns.barplot(data=results_df, x="f1", y="model", ax=axes[1], color="#d95f02")
    axes[1].set_title("Held-out F1")
    axes[1].set_xlabel("F1")
    axes[1].set_ylabel("")

    sns.barplot(data=results_df, x="low_util_recall", y="model", ax=axes[2], color="#66a61e")
    axes[2].set_title("Recall for prior_utilization = 0")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase7_native_categorical_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_native_top_features(model: CatBoostClassifier, feature_cols: list[str]) -> list[dict[str, float]]:
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.get_feature_importance(),
        }
    ).sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=importance, x="importance", y="feature", color="#7570b3")
    plt.title("Phase 7 Native-Categorical CatBoost - Top Features", fontweight="bold")
    plt.xlabel("CatBoost Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase7_native_categorical_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    return [
        {"feature": row.feature, "importance": round(float(row.importance), 4)}
        for row in importance.itertuples()
    ]


if __name__ == "__main__":
    ensure_dirs()
    best_params = load_best_params()

    baseline_df, baseline_cols, _ = load_modeling_frame()
    X_base = baseline_df[baseline_cols]
    y = baseline_df["readmitted_binary"]
    base_splits = make_production_splits(X_base, y)
    Xb_train, yb_train = base_splits["train"]
    Xb_early, yb_early = base_splits["early_stop"]
    Xb_cal, yb_cal = base_splits["calibration"]
    Xb_test, yb_test = base_splits["test"]

    base_model, base_calibrator, base_threshold, base_train_time = train_calibrated_model(
        Xb_train,
        yb_train,
        Xb_early,
        yb_early,
        Xb_cal,
        yb_cal,
        best_params,
    )
    base_test_prob = base_calibrator.predict_proba(Xb_test)[:, 1]
    baseline_metrics = evaluate_model(Xb_test, yb_test, base_test_prob, base_threshold, base_train_time)

    native_df, native_cols, native_cat_cols = build_native_categorical_frame()
    X_native = native_df[native_cols]
    y_native = native_df["readmitted_binary"]
    native_splits = make_production_splits(X_native, y_native)
    Xn_train, yn_train = native_splits["train"]
    Xn_early, yn_early = native_splits["early_stop"]
    Xn_cal, yn_cal = native_splits["calibration"]
    Xn_test, yn_test = native_splits["test"]
    native_cat_indices = [X_native.columns.get_loc(col) for col in native_cat_cols]

    native_model, native_calibrator, native_threshold, native_train_time = train_calibrated_model(
        Xn_train,
        yn_train,
        Xn_early,
        yn_early,
        Xn_cal,
        yn_cal,
        best_params,
        cat_feature_indices=native_cat_indices,
    )
    native_test_prob = native_calibrator.predict_proba(Xn_test)[:, 1]
    native_metrics = evaluate_model(Xn_test, yn_test, native_test_prob, native_threshold, native_train_time)

    bootstrap = bootstrap_auc_delta(yn_test, base_test_prob, native_test_prob)
    top_native_features = plot_native_top_features(native_model, native_cols)

    results_df = pd.DataFrame(
        [
            {"model": "One-hot CatBoost (full_83)", **baseline_metrics},
            {"model": "Native-cat CatBoost (raw+engineered)", **native_metrics},
        ]
    )
    plot_comparison(results_df)

    payload = {
        "phase": 7,
        "date": "2026-04-05",
        "objective": "Test whether native categorical handling beats the one-hot full_83 champion on the same deterministic split.",
        "baseline": {"model": "One-hot CatBoost (full_83)", **baseline_metrics},
        "candidate": {"model": "Native-cat CatBoost (raw+engineered)", **native_metrics},
        "bootstrap_auc_delta": bootstrap,
        "native_feature_count": len(native_cols),
        "native_categorical_count": len(native_cat_cols),
        "native_top_features": top_native_features,
        "comparison_table": results_df.to_dict(orient="records"),
    }
    PHASE7_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(results_df.to_string(index=False))
    print(json.dumps(bootstrap, indent=2))
    print(f"Saved {PHASE7_JSON}")
