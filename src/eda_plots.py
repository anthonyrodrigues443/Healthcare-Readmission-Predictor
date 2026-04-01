"""
EDA plots for Phase 1 report.
Generates: class_distribution.png, feature_correlation.png
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils import get_project_root, load_config, setup_logger

logger = setup_logger(__name__)
sns.set_theme(style="whitegrid", palette="muted")


def plot_class_distribution(df: pd.DataFrame, plots_dir: str) -> None:
    target_col = "readmitted_binary" if "readmitted_binary" in df.columns else "readmitted"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Readmission Class Distribution — UCI Diabetic Readmission Dataset", fontsize=14, fontweight="bold")

    counts = df[target_col].value_counts().sort_index()
    labels = ["Not Readmitted\n(<30 days = 0)", "Readmitted\n(<30 days = 1)"]
    colors = ["#4C9BE8", "#E85C4C"]

    axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
    axes[0].set_title("Absolute Counts", fontsize=12)
    axes[0].set_ylabel("Number of Patients")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 200, f"{v:,}", ha="center", fontweight="bold", fontsize=11)
    axes[0].set_ylim(0, max(counts.values) * 1.15)

    pct = counts / counts.sum() * 100
    wedge_props = {"edgecolor": "white", "linewidth": 2}
    axes[1].pie(
        pct.values, labels=[f"Not Readmitted\n{pct[0]:.1f}%", f"Readmitted <30d\n{pct[1]:.1f}%"],
        colors=colors, autopct=None, wedgeprops=wedge_props, startangle=90
    )
    axes[1].set_title("Class Split", fontsize=12)

    total = len(df)
    pos = counts.get(1, 0)
    fig.text(0.5, 0.01,
             f"Total: {total:,} patients  |  Positive (readmitted <30d): {pos:,} ({pos/total*100:.1f}%)  |  Class imbalance ratio ~{counts[0]/counts[1]:.0f}:1",
             ha="center", fontsize=10, style="italic", color="#555555")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = os.path.join(plots_dir, "class_distribution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


def plot_feature_correlation(df: pd.DataFrame, plots_dir: str, top_n: int = 20) -> None:
    target_col = "readmitted_binary" if "readmitted_binary" in df.columns else "readmitted"

    numeric_df = df.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns:
        logger.warning("Target column not numeric — skipping correlation plot")
        return

    corr = numeric_df.corr()[target_col].drop(target_col)
    corr_abs_sorted = corr.abs().sort_values(ascending=False).head(top_n)
    corr_sorted = corr[corr_abs_sorted.index]

    colors = ["#E85C4C" if v > 0 else "#4C9BE8" for v in corr_sorted.values]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(corr_sorted)), corr_sorted.values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(range(len(corr_sorted)))
    ax.set_yticklabels(corr_sorted.index, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Pearson Correlation with Readmission (<30d)", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Correlations with 30-Day Readmission\n(Red = positive, Blue = negative)",
                 fontsize=13, fontweight="bold")

    for i, (val, bar) in enumerate(zip(corr_sorted.values, bars)):
        xpos = val + 0.001 if val >= 0 else val - 0.001
        ha = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", ha=ha, fontsize=8.5)

    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(plots_dir, "feature_correlation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


def plot_lace_distribution(df: pd.DataFrame, plots_dir: str) -> None:
    """Show LACE score distribution by readmission outcome."""
    if "lace_score" not in df.columns or "readmitted_binary" not in df.columns:
        logger.info("LACE score not in dataframe — skipping LACE plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LACE Score Distribution by Readmission Outcome", fontsize=13, fontweight="bold")

    for val, label, color, ax in zip([0, 1], ["Not Readmitted", "Readmitted <30d"],
                                      ["#4C9BE8", "#E85C4C"], axes):
        subset = df[df["readmitted_binary"] == val]["lace_score"].dropna()
        ax.hist(subset, bins=20, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.axvline(subset.mean(), color="black", linestyle="--", linewidth=1.5, label=f"Mean={subset.mean():.1f}")
        ax.axvline(10, color="orange", linestyle=":", linewidth=2, label="LACE>=10 (high risk)")
        ax.set_title(f"{label} (n={len(subset):,})")
        ax.set_xlabel("LACE Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(plots_dir, "lace_distribution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


def run_all_plots():
    config = load_config()
    root = get_project_root()
    processed_path = root / config["data"]["processed_path"]
    plots_dir = root / config["results"]["plots_path"]
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_parquet(processed_path / "full_processed.parquet")
    logger.info(f"Loaded dataset: {df.shape}")

    plot_class_distribution(df, str(plots_dir))
    plot_feature_correlation(df, str(plots_dir))

    from src.feature_engineering import engineer_features
    df_eng = engineer_features(df)
    df_eng["readmitted_binary"] = df["readmitted_binary"]
    df_eng = df_eng.fillna(0)
    for col in df_eng.select_dtypes(include="object").columns:
        df_eng[col] = pd.Categorical(df_eng[col]).codes
    plot_lace_distribution(df_eng, str(plots_dir))

    logger.info("All plots generated.")


if __name__ == "__main__":
    run_all_plots()
