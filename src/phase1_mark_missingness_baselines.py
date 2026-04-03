"""Mark's complementary Phase 1 experiment for healthcare readmission.

Focus:
- Missingness and test-ordering signals
- Prior-utilization heuristics
- Lightweight baseline families that complement Anthony's logistic baselines
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
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.data_pipeline import download_dataset, clean_and_engineer

warnings.filterwarnings("ignore")


RESULTS_DIR = Path("results")
PROCESSED_DIR = Path("data/processed")
MARK_RESULTS_PATH = RESULTS_DIR / "mark_phase1_missingness_baselines.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def prior_utilization_bucket(series: pd.Series) -> pd.Series:
    return pd.cut(
        series,
        bins=[-0.1, 0.5, 1.5, 3.5, np.inf],
        labels=["0", "1", "2-3", "4+"],
    )


def load_anthony_baselines() -> list[dict]:
    if not METRICS_PATH.exists():
        return []
    with METRICS_PATH.open() as f:
        payload = json.load(f)
    return payload.get("baselines", []) if isinstance(payload, dict) else []


def plot_missingness_signal(raw_df: pd.DataFrame, features_df: pd.DataFrame) -> dict:
    analysis = {}

    raw_clean = raw_df.replace("?", np.nan).copy()
    raw_clean["readmitted_binary"] = (raw_clean["readmitted"] == "<30").astype(int)
    raw_clean["A1C_tested_raw"] = raw_clean["A1Cresult"].fillna("None") != "None"
    raw_clean["glucose_tested_raw"] = raw_clean["max_glu_serum"].fillna("None") != "None"
    raw_clean["prior_utilization"] = (
        raw_clean["number_outpatient"] +
        raw_clean["number_emergency"] +
        raw_clean["number_inpatient"]
    )

    a1c_rates = raw_clean.groupby("A1C_tested_raw")["readmitted_binary"].mean()
    glucose_rates = raw_clean.groupby("glucose_tested_raw")["readmitted_binary"].mean()
    utilization_rates = raw_clean.groupby(
        prior_utilization_bucket(raw_clean["prior_utilization"])
    )["readmitted_binary"].mean()

    analysis["a1c_tested_readmit_rate"] = {
        str(k): round(float(v), 4) for k, v in a1c_rates.items()
    }
    analysis["glucose_tested_readmit_rate"] = {
        str(k): round(float(v), 4) for k, v in glucose_rates.items()
    }
    analysis["prior_utilization_readmit_rate"] = {
        str(k): round(float(v), 4) for k, v in utilization_rates.items()
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    sns.barplot(
        x=a1c_rates.index.map({False: "Not tested", True: "Test ordered"}),
        y=a1c_rates.values,
        ax=axes[0],
        color="#d95f02",
    )
    axes[0].set_title("A1C Test Ordering Signal", fontweight="bold")
    axes[0].set_ylabel("30-day readmission rate")
    axes[0].set_xlabel("")

    sns.barplot(
        x=glucose_rates.index.map({False: "Not tested", True: "Test ordered"}),
        y=glucose_rates.values,
        ax=axes[1],
        color="#1b9e77",
    )
    axes[1].set_title("Glucose Test Ordering Signal", fontweight="bold")
    axes[1].set_ylabel("30-day readmission rate")
    axes[1].set_xlabel("")

    utilization_rates.plot(kind="bar", ax=axes[2], color="#7570b3", edgecolor="black")
    axes[2].set_title("Prior Utilization vs Readmission", fontweight="bold")
    axes[2].set_ylabel("30-day readmission rate")
    axes[2].set_xlabel("Prior utilization bucket")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mark_missingness_and_utilization.png", dpi=150, bbox_inches="tight")
    plt.close()

    return analysis


def score_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray, latency_ms: float) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_score)),
        "latency_ms_per_sample": round(latency_ms, 4),
    }


def measure_latency(fn, X, repeats: int = 10) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        fn(X)
    elapsed = time.perf_counter() - start
    return elapsed * 1000 / (len(X) * repeats)


def run_mark_experiments(features_df: pd.DataFrame) -> list[dict]:
    target = "readmitted_binary"
    y = features_df[target]
    X = features_df.drop(columns=[target])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = []

    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    dummy_score = dummy.predict_proba(X_test)[:, 1]
    dummy_latency = measure_latency(dummy.predict_proba, X_test)
    results.append({
        "model": "DummyClassifier (stratified)",
        "feature_set": "None",
        **score_metrics(y_test, dummy_pred, dummy_score, dummy_latency),
        "notes": "Sanity-check baseline preserving class imbalance."
    })

    rule_features = [
        "number_inpatient",
        "number_emergency",
        "prior_utilization",
        "polypharmacy",
        "time_in_hospital",
        "A1C_tested",
        "glucose_tested",
    ]
    rule_train = X_train[rule_features]
    rule_test = X_test[rule_features]

    def utilization_rule_score(df: pd.DataFrame) -> np.ndarray:
        score = np.zeros(len(df), dtype=int)
        score += (df["number_inpatient"] >= 1).astype(int) * 3
        score += (df["number_emergency"] >= 1).astype(int) * 2
        score += (df["prior_utilization"] >= 2).astype(int)
        score += (df["polypharmacy"] == 1).astype(int)
        score += (df["time_in_hospital"] >= 5).astype(int)
        score += (df["A1C_tested"] == 1).astype(int)
        score += (df["glucose_tested"] == 1).astype(int)
        return score

    rule_train_score = utilization_rule_score(rule_train)
    rule_test_score = utilization_rule_score(rule_test)
    best_threshold = 2
    best_f1 = -1.0
    for threshold in range(1, 8):
        train_pred = (rule_train_score >= threshold).astype(int)
        train_f1 = f1_score(y_train, train_pred, zero_division=0)
        if train_f1 > best_f1:
            best_f1 = train_f1
            best_threshold = threshold
    rule_pred = (rule_test_score >= best_threshold).astype(int)
    rule_latency = measure_latency(utilization_rule_score, rule_test)
    results.append({
        "model": f"Workflow Triage Rule (threshold={best_threshold})",
        "feature_set": "7 workflow features",
        **score_metrics(y_test, rule_pred, rule_test_score, rule_latency),
        "notes": "Hand-built utilization + test-ordering heuristic."
    })

    nb_features = [
        "A1C_tested",
        "A1C_high",
        "glucose_tested",
        "glucose_high",
        "med_changed",
        "diabetes_med",
        "polypharmacy",
        "diabetes_primary",
    ]
    nb = BernoulliNB()
    nb.fit(X_train[nb_features], y_train)
    nb_pred = nb.predict(X_test[nb_features])
    nb_score = nb.predict_proba(X_test[nb_features])[:, 1]
    nb_latency = measure_latency(nb.predict_proba, X_test[nb_features])
    results.append({
        "model": "BernoulliNB (test-ordering only)",
        "feature_set": "8 binary workflow features",
        **score_metrics(y_test, nb_pred, nb_score, nb_latency),
        "notes": "Asks whether missingness and medication-state signals carry standalone signal."
    })

    svm_features = [
        "time_in_hospital",
        "num_medications",
        "number_diagnoses",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "A1C_tested",
        "glucose_tested",
    ]
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(class_weight="balanced", random_state=42, max_iter=8000)),
    ])
    svm.fit(X_train[svm_features], y_train)
    svm_pred = svm.predict(X_test[svm_features])
    svm_score = svm.decision_function(X_test[svm_features])
    svm_latency = measure_latency(svm.decision_function, X_test[svm_features])
    results.append({
        "model": "Linear SVM (workflow compact set)",
        "feature_set": "8 workflow features",
        **score_metrics(y_test, svm_pred, svm_score, svm_latency),
        "notes": "Compact non-probabilistic linear margin model."
    })

    return sorted(results, key=lambda item: item["auc"], reverse=True)


def save_metrics(mark_results: list[dict], signal_analysis: dict, dataset_rows: int, feature_count: int) -> None:
    payload = {
        "phase": 1,
        "date": "2026-04-01",
        "researcher": "Mark Rodrigues",
        "dataset": "UCI Diabetes 130-US Hospitals",
        "n_samples": dataset_rows,
        "n_features": feature_count,
        "signal_analysis": signal_analysis,
        "models": mark_results,
    }

    with MARK_RESULTS_PATH.open("w") as f:
        json.dump(payload, f, indent=2)

    if METRICS_PATH.exists():
        with METRICS_PATH.open() as f:
            aggregate = json.load(f)
    else:
        aggregate = {}

    if not isinstance(aggregate, dict):
        aggregate = {"history": aggregate}

    aggregate["mark_phase1_complementary"] = payload

    with METRICS_PATH.open("w") as f:
        json.dump(aggregate, f, indent=2)


def plot_comparison(mark_results: list[dict], anthony_baselines: list[dict]) -> None:
    comparison_rows = []
    for baseline in anthony_baselines:
        if baseline["model"] in {
            "LACE Index (threshold=6)",
            "LogReg (clinical features only, n=23)",
            "Logistic Regression (balanced)",
        }:
            comparison_rows.append({
                "model": baseline["model"],
                "auc": baseline["auc"],
                "f1": baseline["f1"],
                "owner": "Anthony",
            })

    for row in mark_results:
        comparison_rows.append({
            "model": row["model"],
            "auc": row["auc"],
            "f1": row["f1"],
            "owner": "Mark",
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.sort_values("auc", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.barplot(data=comparison_df, x="auc", y="model", hue="owner", ax=axes[0], dodge=False)
    axes[0].set_title("AUC Comparison", fontweight="bold")
    axes[0].set_xlim(0.45, 0.70)
    axes[0].set_xlabel("AUC")
    axes[0].set_ylabel("")

    sns.barplot(data=comparison_df, x="f1", y="model", hue="owner", ax=axes[1], dodge=False)
    axes[1].set_title("F1 Comparison", fontweight="bold")
    axes[1].set_xlim(0.0, 0.30)
    axes[1].set_xlabel("F1")
    axes[1].set_ylabel("")

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[:2], labels[:2], loc="lower right")
    axes[1].legend([], [], frameon=False)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mark_phase1_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def print_results(mark_results: list[dict], signal_analysis: dict) -> None:
    print("=" * 72)
    print("MARK PHASE 1 COMPLEMENTARY BASELINES")
    print("=" * 72)
    print("Signal analysis:")
    print(json.dumps(signal_analysis, indent=2))
    print("\nModel comparison:")
    print(
        f"{'Rank':<5} {'Model':<38} {'AUC':<8} {'F1':<8} {'Recall':<8} {'Latency(ms)':<12}"
    )
    print("-" * 84)
    for rank, row in enumerate(mark_results, start=1):
        print(
            f"{rank:<5} {row['model']:<38} "
            f"{row['auc']:.3f}{'':>3} {row['f1']:.3f}{'':>3} "
            f"{row['recall']:.3f}{'':>3} {row['latency_ms_per_sample']:.4f}"
        )


def main() -> None:
    ensure_dirs()
    raw_df = download_dataset()
    features_df = clean_and_engineer(raw_df)
    features_df.to_csv(PROCESSED_DIR / "readmission_processed.csv", index=False)

    anthony_baselines = load_anthony_baselines()
    signal_analysis = plot_missingness_signal(raw_df, features_df)
    mark_results = run_mark_experiments(features_df)
    save_metrics(mark_results, signal_analysis, len(features_df), features_df.shape[1] - 1)
    plot_comparison(mark_results, anthony_baselines)
    print_results(mark_results, signal_analysis)


if __name__ == "__main__":
    main()
