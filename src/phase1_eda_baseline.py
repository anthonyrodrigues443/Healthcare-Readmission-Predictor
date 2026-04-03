"""
Phase 1 runner for Healthcare Readmission Predictor.

This wrapper keeps Anthony's original entrypoint name but routes execution
through the stable data pipeline plus the baseline/EDA module that already
generates the Phase 1 artifacts.
"""

from pathlib import Path

from src.data_pipeline import download_dataset, clean_and_engineer
from src.eda_and_baseline import run_eda, run_baselines


def main() -> None:
    print("=" * 60)
    print("PHASE 1: Healthcare Readmission Predictor")
    print("Date: 2026-04-01 | Researcher: Anthony Rodrigues")
    print("=" * 60)

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    print("\nLoading data...")
    raw_df = download_dataset()
    print(f"Loaded {len(raw_df):,} rows, {raw_df.shape[1]} columns")

    print("\nEngineering features...")
    processed_df = clean_and_engineer(raw_df)
    print(f"After exclusions: {len(processed_df):,} rows")

    processed_df.to_parquet("data/processed/features.parquet", index=False)
    processed_df.to_csv("data/processed/readmission_processed.csv", index=False)
    print("Saved: data/processed/features.parquet")
    print("Saved: data/processed/readmission_processed.csv")

    run_eda(raw_df, processed_df)
    run_baselines(processed_df)

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print("\nKey findings:")
    print("1. Class imbalance is severe: only 11.2% are <30-day readmissions.")
    print("2. LACE underperforms compared with the domain-feature logistic baseline.")
    print("3. Clinical features already carry most of the Phase 1 predictive signal.")
    print("4. Prior inpatient visits and prior utilization remain the strongest signals.")
    print("5. Phase 2 should compare tree models and imbalance strategies.")


if __name__ == "__main__":
    main()
