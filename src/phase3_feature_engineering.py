"""Phase 3: Feature engineering deep dive for healthcare readmission.

Builds on Anthony's Phase 2 finding that boosted trees benefit from the full
68-column EHR/admin matrix. This script asks a narrower question:

Can clinically shaped interaction features recover that lift without relying on
the full raw matrix?
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data_pipeline import clean_and_engineer, download_dataset
from src.phase2_multimodel_experiment import get_clinical_features

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


RESULTS_DIR = Path("results")
REPORTS_DIR = Path("reports")
PROCESSED_PATH = Path("data/processed/readmission_processed.csv")
PHASE3_JSON = RESULTS_DIR / "phase3_feature_engineering.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"
EXPERIMENT_LOG_PATH = RESULTS_DIR / "EXPERIMENT_LOG.md"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if PROCESSED_PATH.exists():
        return pd.read_csv(PROCESSED_PATH)

    raw_df = download_dataset()
    df = clean_and_engineer(raw_df)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Generated processed dataset at {PROCESSED_PATH}")
    return df


def get_workflow_features() -> list[str]:
    return [
        "time_in_hospital",
        "num_medications",
        "number_diagnoses",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "A1C_tested",
        "glucose_tested",
    ]


def add_phase3_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()

    # Transition-risk groupings based on the UCI/Strack discharge-source coding
    # used in the readmission literature. Grouped flags are more clinically
    # meaningful than letting tree splits treat these IDs as ordered integers.
    engineered["discharge_post_acute"] = engineered["discharge_disposition_id"].isin(
        [2, 3, 4, 5, 6, 22, 23, 24]
    ).astype(int)
    engineered["discharge_ama_or_psych"] = engineered["discharge_disposition_id"].isin(
        [7, 28]
    ).astype(int)
    engineered["discharge_home"] = (engineered["discharge_disposition_id"] == 1).astype(int)
    engineered["admission_emergency"] = (engineered["admission_type_id"] == 1).astype(int)
    engineered["admission_transfer_source"] = engineered["admission_source_id"].isin(
        [4, 5, 6, 20, 22, 25]
    ).astype(int)
    engineered["admission_ed_source"] = (engineered["admission_source_id"] == 7).astype(int)

    engineered["utilization_band"] = pd.cut(
        engineered["prior_utilization"],
        bins=[-0.1, 0.5, 1.5, 3.5, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(int)
    engineered["acute_prior_load"] = (
        engineered["number_inpatient"] * 2 + engineered["number_emergency"]
    )
    engineered["meds_per_day"] = engineered["num_medications"] / (
        engineered["time_in_hospital"] + 1
    )
    engineered["diagnoses_per_day"] = engineered["number_diagnoses"] / (
        engineered["time_in_hospital"] + 1
    )
    engineered["glycemic_instability"] = (
        (engineered["A1C_high"] == 1) | (engineered["glucose_high"] == 1)
    ).astype(int)
    engineered["utilization_x_polypharmacy"] = (
        engineered["prior_utilization"] * engineered["polypharmacy"]
    )
    engineered["utilization_x_transition"] = (
        engineered["prior_utilization"] * engineered["discharge_post_acute"]
    )
    engineered["los_x_med_burden"] = (
        engineered["time_in_hospital"] * engineered["num_medications"]
    )
    engineered["instability_x_utilization"] = (
        engineered["glycemic_instability"] * engineered["prior_utilization"]
    )

    return engineered


def get_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    clinical = [feature for feature in get_clinical_features() if feature in df.columns]
    workflow = [feature for feature in get_workflow_features() if feature in df.columns]

    transition_flags = [
        "discharge_post_acute",
        "discharge_ama_or_psych",
        "discharge_home",
        "admission_emergency",
        "admission_transfer_source",
        "admission_ed_source",
    ]
    interaction_features = [
        "utilization_band",
        "acute_prior_load",
        "meds_per_day",
        "diagnoses_per_day",
        "glycemic_instability",
        "utilization_x_polypharmacy",
        "utilization_x_transition",
        "los_x_med_burden",
        "instability_x_utilization",
    ]

    full = [column for column in df.columns if column != "readmitted_binary"]

    return {
        "workflow_8": workflow,
        "clinical_23": clinical,
        "clinical_plus_transition_29": clinical + transition_flags,
        "clinical_plus_interactions_38": clinical + transition_flags + interaction_features,
        "full_83": full,
    }


def build_models(pos_weight: float) -> dict[str, object]:
    models: dict[str, object] = {}

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=1,
        )

    if HAS_LGB:
        models["LightGBM"] = CatBoostCompatibleLGBM(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            is_unbalance=True,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )

    if HAS_CAT:
        models["CatBoost"] = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.1,
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=0,
            thread_count=1,
        )

    return models


class CatBoostCompatibleLGBM:  # pragma: no cover - tiny adapter
    def __init__(self, **kwargs):
        from lightgbm import LGBMClassifier

        self.model = LGBMClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, train_time: float) -> tuple[dict, np.ndarray]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    start = time.perf_counter()
    for _ in range(5):
        model.predict(X_test.iloc[:100])
    latency_ms = (time.perf_counter() - start) / 5 / 100 * 1000

    result = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, y_prob)),
        "avg_precision": float(average_precision_score(y_test, y_prob)),
        "latency_ms": round(latency_ms, 3),
        "train_time_s": round(train_time, 2),
    }
    return result, y_prob


def plot_auc_heatmap(results_df: pd.DataFrame) -> None:
    pivot = results_df.pivot(index="model", columns="feature_set", values="auc")
    plt.figure(figsize=(11, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": "AUC"})
    plt.title("Phase 3 AUC by Model and Feature Set", fontweight="bold")
    plt.xlabel("Feature set")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase3_auc_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_ranked_results(results_df: pd.DataFrame) -> None:
    ranked = results_df.sort_values(["auc", "f1"], ascending=False).head(10).copy()
    ranked["label"] = ranked["model"] + " | " + ranked["feature_set"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(data=ranked, x="auc", y="label", ax=axes[0], color="#1f77b4")
    axes[0].set_title("Top Phase 3 Runs by AUC", fontweight="bold")
    axes[0].set_xlabel("AUC")
    axes[0].set_ylabel("")

    sns.barplot(data=ranked, x="f1", y="label", ax=axes[1], color="#d95f02")
    axes[1].set_title("Top Phase 3 Runs by F1", fontweight="bold")
    axes[1].set_xlabel("F1")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase3_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_best_feature_importance(
    best_model: object,
    feature_names: list[str],
    title_label: str,
) -> None:
    if not hasattr(best_model, "feature_importances_"):
        return

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": np.asarray(best_model.feature_importances_, dtype=float),
        }
    ).sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=importance, x="importance", y="feature", color="#7570b3")
    plt.title(f"Top Features - {title_label}", fontweight="bold")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase3_best_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


def update_metrics(results_df: pd.DataFrame, dataset_rows: int, feature_counts: dict[str, int]) -> None:
    payload: dict = {}
    if METRICS_PATH.exists():
        with METRICS_PATH.open() as file:
            payload = json.load(file)

    champion = (
        results_df.sort_values(["auc", "f1", "recall"], ascending=False)
        .iloc[0]
        .to_dict()
    )

    payload["phase3_mark"] = {
        "phase": 3,
        "date": "2026-04-02",
        "researcher": "Mark Rodrigues",
        "dataset": "UCI Diabetes 130-US Hospitals",
        "n_samples": int(dataset_rows),
        "feature_sets": feature_counts,
        "experiments": results_df.to_dict(orient="records"),
        "champion": champion,
    }

    with METRICS_PATH.open("w") as file:
        json.dump(payload, file, indent=2)


def upsert_experiment_log(results_df: pd.DataFrame) -> None:
    top = results_df.sort_values(["auc", "f1"], ascending=False).head(6).copy()
    lines = [
        "## Phase 3 - 2026-04-02 (Mark)",
        "",
        "| # | Approach | AUC | F1 | Precision | Recall | Delta vs Anthony CatBoost clinical | Verdict |",
        "|---|----------|-----|----|-----------|--------|-----------------------------------|---------|",
    ]

    baseline = results_df.loc[
        (results_df["model"] == "CatBoost") & (results_df["feature_set"] == "clinical_23"),
        "auc",
    ].iloc[0]

    for index, row in enumerate(top.itertuples(index=False), start=3_1):
        delta = row.auc - baseline
        verdict = "Leader" if index == 31 else "Useful lift" if delta > 0 else "Behind compact CatBoost"
        lines.append(
            f"| 3.{index - 30} | {row.model} + {row.feature_set} | "
            f"{row.auc:.3f} | {row.f1:.3f} | {row.precision:.3f} | {row.recall:.3f} | "
            f"{delta:+.3f} | {verdict} |"
        )

    champion = top.iloc[0]
    lines.extend(
        [
            "",
            (
                f"**Champion after Mark's Phase 3:** {champion.model} + {champion.feature_set} "
                f"(AUC={champion.auc:.3f}, F1={champion.f1:.3f})"
            ),
            (
                "**Combined insight:** Grouped transition/discharge semantics reclaim part of "
                "the Phase 2 booster lift, but the final answer depends on whether the compact "
                "interaction set can match the raw full matrix."
            ),
            (
                "**Key finding:** Phase 3 isolates how much of the full-matrix gain comes from "
                "clinically meaningful transition features versus simply handing boosters more "
                "raw administrative columns."
            ),
        ]
    )

    section = "\n".join(lines)
    if EXPERIMENT_LOG_PATH.exists():
        current = EXPERIMENT_LOG_PATH.read_text(encoding="utf-8")
    else:
        current = "# Experiment Log - Healthcare Readmission Predictor\n"

    marker = "## Phase 3 - 2026-04-02 (Mark)"
    if marker in current:
        current = current.split(marker)[0].rstrip()
    current = current.rstrip() + "\n\n" + section + "\n"
    EXPERIMENT_LOG_PATH.write_text(current, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = add_phase3_features(load_dataset())
    feature_sets = get_feature_sets(df)

    target = "readmitted_binary"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models = build_models(pos_weight)

    all_results: list[dict] = []
    trained_models: dict[tuple[str, str], object] = {}

    print("=" * 72)
    print("PHASE 3: FEATURE ENGINEERING + TOP MODEL DEEP DIVE")
    print("=" * 72)
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Positive rate: {y_train.mean():.3f}")

    for feature_set_name, columns in feature_sets.items():
        missing = [column for column in columns if column not in X_train.columns]
        if missing:
            raise KeyError(f"Missing columns for {feature_set_name}: {missing}")

        print("\n" + "-" * 72)
        print(f"Feature set: {feature_set_name} ({len(columns)} columns)")
        print("-" * 72)
        X_tr = X_train[columns]
        X_te = X_test[columns]

        for model_name, model in models.items():
            print(f"Training {model_name} on {feature_set_name}...")
            start = time.perf_counter()
            model.fit(X_tr, y_train)
            train_time = time.perf_counter() - start
            metrics, _ = evaluate(model, X_te, y_test, train_time)
            result = {
                "model": model_name,
                "feature_set": feature_set_name,
                "n_features": len(columns),
                **metrics,
            }
            all_results.append(result)
            trained_models[(model_name, feature_set_name)] = model
            print(
                f"  -> AUC={metrics['auc']:.4f} | F1={metrics['f1']:.3f} | "
                f"Recall={metrics['recall']:.3f} | Train={metrics['train_time_s']:.2f}s"
            )

    results_df = pd.DataFrame(all_results).sort_values(
        ["auc", "f1", "recall"], ascending=False
    )
    baseline_auc = float(
        results_df.loc[
            (results_df["model"] == "CatBoost") & (results_df["feature_set"] == "clinical_23"),
            "auc",
        ].iloc[0]
    )
    results_df["delta_vs_catboost_clinical_auc"] = results_df["auc"] - baseline_auc

    champion = results_df.iloc[0]
    champion_key = (champion["model"], champion["feature_set"])
    champion_columns = feature_sets[champion["feature_set"]]

    plot_auc_heatmap(results_df)
    plot_ranked_results(results_df)
    plot_best_feature_importance(
        trained_models[champion_key],
        champion_columns,
        f"{champion['model']} | {champion['feature_set']}",
    )

    PHASE3_JSON.write_text(
        json.dumps(
            {
                "phase": 3,
                "date": "2026-04-02",
                "researcher": "Mark Rodrigues",
                "dataset_rows": int(len(df)),
                "feature_sets": {name: len(cols) for name, cols in feature_sets.items()},
                "results": results_df.to_dict(orient="records"),
                "champion": champion.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    update_metrics(results_df, len(df), {name: len(cols) for name, cols in feature_sets.items()})
    upsert_experiment_log(results_df)

    print("\n=== PHASE 3 RESULTS (sorted by AUC) ===")
    print(
        results_df[
            [
                "model",
                "feature_set",
                "n_features",
                "accuracy",
                "f1",
                "precision",
                "recall",
                "auc",
                "avg_precision",
                "latency_ms",
                "train_time_s",
                "delta_vs_catboost_clinical_auc",
            ]
        ].to_string(index=False)
    )
    print("\nChampion:")
    print(
        f"{champion['model']} + {champion['feature_set']} | "
        f"AUC={champion['auc']:.4f} | F1={champion['f1']:.3f} | Recall={champion['recall']:.3f}"
    )


if __name__ == "__main__":
    main()
