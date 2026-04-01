"""
Phase 3: Mark's complementary feature-engineering deep dive.

Anthony's Phase 2 finding:
- CatBoost was the best balanced model.
- LightGBM was the only model clearly helped by domain features.
- Random Forest collapsed badly on minority recall.

Mark's Phase 3 angle:
- Keep the strongest families in play.
- Add interaction-heavy utilization/complexity features.
- Probe whether BalancedRandomForest rescues the RF collapse when the sampling
  strategy matches the imbalance better than plain majority-vote bagging.
"""
import copy
import json
import os
import sys
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_pipeline import run_pipeline
from src.evaluate import compute_metrics, print_metrics
from src.feature_engineering import engineer_features, engineer_phase3_features
from src.utils import get_project_root, load_config, setup_logger


warnings.filterwarnings("ignore")
logger = setup_logger(__name__)
RANDOM_STATE = 42


def align_feature_frames(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy().replace([np.inf, -np.inf], 0).fillna(0)
    test_df = test_df.copy().replace([np.inf, -np.inf], 0).fillna(0)

    bool_train = train_df.select_dtypes(include="bool").columns
    bool_test = test_df.select_dtypes(include="bool").columns
    if len(bool_train) > 0:
        train_df[bool_train] = train_df[bool_train].astype(int)
    if len(bool_test) > 0:
        test_df[bool_test] = test_df[bool_test].astype(int)

    train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)
    return train_df, test_df


def run_experiment(model, X_train, y_train, X_test, y_test, name: str) -> tuple[dict, object, np.ndarray]:
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob, model_name=name)
    metrics["train_time_s"] = round(train_time, 2)
    metrics["n_features"] = int(X_train.shape[1])
    return metrics, model, y_prob


def build_experiments():
    return [
        (
            "3.1_catboost_domain",
            "CatBoost + Anthony domain features",
            CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                auto_class_weights="Balanced",
                random_seed=RANDOM_STATE,
                verbose=0,
            ),
            "domain",
        ),
        (
            "3.2_catboost_interactions",
            "CatBoost + Mark interaction features",
            CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                auto_class_weights="Balanced",
                random_seed=RANDOM_STATE,
                verbose=0,
            ),
            "phase3",
        ),
        (
            "3.3_lgbm_domain",
            "LightGBM + Anthony domain features",
            LGBMClassifier(
                n_estimators=350,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                verbose=-1,
            ),
            "domain",
        ),
        (
            "3.4_lgbm_interactions",
            "LightGBM + Mark interaction features",
            LGBMClassifier(
                n_estimators=350,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                verbose=-1,
            ),
            "phase3",
        ),
        (
            "3.5_brf_domain",
            "BalancedRandomForest + Anthony domain features",
            BalancedRandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                sampling_strategy="all",
                replacement=False,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "domain",
        ),
        (
            "3.6_brf_interactions",
            "BalancedRandomForest + Mark interaction features",
            BalancedRandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                sampling_strategy="all",
                replacement=False,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "phase3",
        ),
    ]


def plot_model_comparison(results_df: pd.DataFrame, save_path) -> None:
    plot_df = results_df.melt(
        id_vars=["label", "feature_family"],
        value_vars=["recall_sensitivity", "f1", "precision", "auc_roc"],
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(14, 6))
    sns.barplot(data=plot_df, x="label", y="score", hue="metric")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1.0)
    plt.title("Phase 3 Mark Deep Dive: Top Models + Interaction Features")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_delta(results_df: pd.DataFrame, save_path) -> None:
    records = []
    for model_family in ["CatBoost", "LightGBM", "BalancedRandomForest"]:
        family_rows = results_df[results_df["model_family"] == model_family]
        if len(family_rows) != 2:
            continue
        domain = family_rows[family_rows["feature_family"] == "domain"].iloc[0]
        phase3 = family_rows[family_rows["feature_family"] == "phase3"].iloc[0]
        records.append(
            {
                "model_family": model_family,
                "delta_sensitivity": phase3["recall_sensitivity"] - domain["recall_sensitivity"],
                "delta_f1": phase3["f1"] - domain["f1"],
                "delta_auc": phase3["auc_roc"] - domain["auc_roc"],
            }
        )

    delta_df = pd.DataFrame(records).melt(
        id_vars=["model_family"],
        value_vars=["delta_sensitivity", "delta_f1", "delta_auc"],
        var_name="metric",
        value_name="delta",
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(data=delta_df, x="model_family", y="delta", hue="metric")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Effect of Mark's Interaction Features vs Anthony's Domain Features")
    plt.xlabel("")
    plt.ylabel("Delta score")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_top_features(best_model, feature_names: list[str], base_feature_names: set[str], save_path) -> pd.DataFrame:
    if hasattr(best_model, "get_feature_importance"):
        importances = best_model.get_feature_importance()
    elif hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    else:
        importances = np.zeros(len(feature_names))

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)
    importance_df["feature_origin"] = np.where(
        importance_df["feature"].isin(base_feature_names),
        "Anthony/base domain",
        "Mark interaction",
    )

    top_df = importance_df.head(12)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_df, x="importance", y="feature", hue="feature_origin")
    plt.title("Top Features in the Best Phase 3 Model")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return importance_df


def main():
    root = get_project_root()
    config = load_config()
    plots_path = root / config["results"]["plots_path"]
    models_path = root / config["results"]["models_path"]
    results_path = root / "results"
    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    processed_path = root / config["data"]["processed_path"]
    if (processed_path / "X_train.parquet").exists():
        logger.info("Loading preprocessed splits from parquet")
        X_train = pd.read_parquet(processed_path / "X_train.parquet")
        X_test = pd.read_parquet(processed_path / "X_test.parquet")
        y_train = pd.read_parquet(processed_path / "y_train.parquet").squeeze()
        y_test = pd.read_parquet(processed_path / "y_test.parquet").squeeze()
    else:
        X_train, X_test, y_train, y_test = run_pipeline(config)

    X_train_domain, X_test_domain = align_feature_frames(engineer_features(X_train), engineer_features(X_test))
    X_train_phase3, X_test_phase3 = align_feature_frames(engineer_phase3_features(X_train), engineer_phase3_features(X_test))
    base_feature_names = set(X_train_domain.columns)

    logger.info(f"Domain feature set: {X_train_domain.shape[1]} columns")
    logger.info(f"Phase 3 interaction feature set: {X_train_phase3.shape[1]} columns")

    experiment_outputs = []
    best_bundle = None

    for exp_id, label, template, feature_family in build_experiments():
        model = copy.deepcopy(template)
        if feature_family == "domain":
            X_tr, X_te = X_train_domain, X_test_domain
        else:
            X_tr, X_te = X_train_phase3, X_test_phase3

        logger.info(f"Running {exp_id}: {label}")
        metrics, fitted_model, y_prob = run_experiment(model, X_tr, y_train, X_te, y_test, name=exp_id)
        metrics["label"] = label
        metrics["model_family"] = label.split(" + ")[0]
        metrics["feature_family"] = feature_family
        print_metrics(metrics)

        experiment_outputs.append(
            {
                "metrics": metrics,
                "feature_names": list(X_tr.columns),
                "fitted_model": fitted_model,
                "y_prob": y_prob,
            }
        )

        joblib.dump(fitted_model, models_path / f"{exp_id}.joblib")

        if best_bundle is None or metrics["recall_sensitivity"] > best_bundle["metrics"]["recall_sensitivity"]:
            best_bundle = experiment_outputs[-1]

    results_df = pd.DataFrame([item["metrics"] for item in experiment_outputs]).sort_values(
        ["recall_sensitivity", "f1", "auc_roc"], ascending=False
    )

    plot_model_comparison(results_df, plots_path / "phase3_mark_model_comparison.png")
    plot_feature_delta(results_df, plots_path / "phase3_mark_feature_delta.png")
    importance_df = plot_top_features(
        best_bundle["fitted_model"],
        best_bundle["feature_names"],
        base_feature_names,
        plots_path / "phase3_mark_top_features.png",
    )

    summary = {
        "phase": 3,
        "date": "2026-04-01",
        "anthony_reference": {
            "phase2_pr": 1,
            "phase2_champion": "CatBoost + domain features",
            "phase2_champion_sensitivity": 0.4469,
        },
        "dataset": {
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "train_positive_rate": round(float(y_train.mean()), 4),
            "test_positive_rate": round(float(y_test.mean()), 4),
            "domain_feature_count": int(X_train_domain.shape[1]),
            "phase3_feature_count": int(X_train_phase3.shape[1]),
        },
        "results": results_df.to_dict(orient="records"),
        "best_model": best_bundle["metrics"],
        "top_features": importance_df.head(12).to_dict(orient="records"),
    }

    with open(results_path / "phase3_mark_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    importance_df.to_csv(results_path / "phase3_mark_feature_importance.csv", index=False)

    logger.info("Phase 3 Mark experiment complete")
    logger.info(
        f"Best model: {best_bundle['metrics']['label']} | "
        f"Sensitivity={best_bundle['metrics']['recall_sensitivity']:.4f} | "
        f"F1={best_bundle['metrics']['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
