# Healthcare Readmission Predictor

Predicting 30-day hospital readmission risk for diabetic patients using ML on 101,766 real encounters from 130 US hospitals (UCI Diabetes dataset).

## Problem

Hospital readmissions cost the US healthcare system over $26 billion annually. The Hospital Readmissions Reduction Program (HRRP) penalizes hospitals with excess readmission rates. Current clinical tools like the LACE index have limited predictive power (AUC ~0.55-0.66), flagging too many patients as high-risk for meaningful intervention.

This project builds a research-grade ML pipeline that:
- Compares multiple model architectures head-to-head
- Engineers features informed by clinical literature
- Benchmarks against industry-standard clinical scoring tools
- Tests against frontier LLMs (GPT-5.4, Claude Opus 4.6) on the same prediction task

## Dataset

| Metric | Value |
|--------|-------|
| Source | UCI ML Repository — Diabetes 130-US Hospitals (1999-2008) |
| Total encounters | 101,766 |
| Readmission rate | 11.2% (30-day) |
| Features (raw) | 50 |
| Features (engineered) | 69 |
| Train/Test split | 80/20 stratified |

## Current Status

**Phase:** 3 of 7 completed
**Best Model:** CatBoost on full 83-feature matrix (AUC 0.687, F1 0.282)
**Target:** AUC > 0.70 (published SOTA: 0.78-0.87)
**Models Compared:** 26 configurations across 3 phases

## Key Findings

1. CatBoost on full 83-feature matrix is champion — AUC 0.687, F1 0.282, +0.042 over Phase 1 LogReg baseline
2. SMOTE destroys performance on this dataset — F1 drops 0.277→0.104; cost-sensitive class weighting is strictly superior
3. Tree models extract +0.037-0.043 AUC from one-hot features that LogReg misses — nonlinear interactions in admin/demographic categoricals carry real signal
4. Grouped discharge/admission transition semantics improve compact CatBoost recall 0.544→0.598, but raw `discharge_disposition_id` dominates the full-matrix model
5. LACE index flags 61% of all patients for 74% recall — clinically unusable alert volume

## Iteration Summary

<!-- Iteration summaries appended daily by readme-updater cron -->
<!-- Format: Phase N: Title — Date | Author -->

### Phase 2: Multi-Model Experiment — 2026-04-02 | Anthony Rodrigues
**Question:** Which model family best captures 30-day readmission risk, and does imbalance handling strategy matter more than model choice?
**Approach:** Compared 6 models (XGBoost, LightGBM, CatBoost, RF, GBM, SVM-RBF) across 68 and 23 features, then tested 5 class-imbalance strategies on XGBoost. Based on Rajkomar et al. 2018 (gradient-boosted trees dominate structured EHR tasks), we focused on tree ensembles.
**Key Result:** CatBoost wins — AUC 0.686, F1 0.283, Recall 0.585. +0.041 AUC over Phase 1 LogReg baseline.
**Surprise:** SMOTE was catastrophically bad — F1 dropped from 0.277 to 0.104. Synthetic oversampling in a 68-dimensional sparse binary space creates unrealistic fractional patients the model learns as noise. Cost-sensitive weighting is strictly superior.
**Research:** Based on Strack et al. 2014 (LogReg baseline AUC ~0.64 on this data), Phase 1 reproducibility was confirmed. Kaggle community findings consistently warned SMOTE hurts at moderate imbalance (8:1) with categorical EHR data — this held up exactly.
**Key Visual:** <img src="results/phase2_imbalance_strategies.png" width="400">
**Best Model So Far:** CatBoost (68 features, class_weight) — AUC 0.686, F1 0.283, Recall 0.585

---

### Phase 3: Feature Engineering — 2026-04-02 | Mark Rodrigues
**Question:** Can clinically grouped transition features and utilization interactions recover Anthony's Phase 2 full-matrix booster lift without the full raw administrative feature set?
**Approach:** Held the top 3 boosters fixed and varied feature sets: workflow-only (8), clinical (23), clinical + transition flags (29), clinical + interactions (38), full engineered matrix (83). Based on Woudneh et al. 2025 and Hohl et al. 2021 (prior hospitalizations, LOS, discharge transitions, and polypharmacy drive readmission risk), we built grouped semantic flags and interaction ratios.
**Key Result:** CatBoost on full 83-feature matrix remains champion at AUC 0.687, F1 0.282. The compact 29-feature transition set beat the full model on recall (0.598 vs 0.576) while still losing AUC.
**Surprise:** The larger 9-interaction bundle (38 features) did not beat the simpler 6-flag transition set (29 features) — hand-built interactions mostly added noise. Grouped transitions recovered only ~1/3 of the full-matrix lift; `discharge_disposition_id` alone accounts for the rest.
**Research:** Based on Woudneh et al. 2025 who found discharge transitions are significant readmission drivers, we grouped admission/discharge codes into semantic flags — this gave +0.010 AUC and +0.054 recall on the compact set, confirming the signal, but the raw disaggregated feature still dominated.
**Key Visual:** <img src="results/phase3_auc_heatmap.png" width="400">
**Best Model So Far:** CatBoost (full 83 features) — AUC 0.687, F1 0.282, Recall 0.576

---

## Project Structure

```
Healthcare-Readmission-Predictor/
├── src/                  # Source code (data pipeline, training, evaluation)
├── data/                 # Datasets (raw + processed)
├── models/               # Saved model artifacts
├── results/              # Metrics, plots, experiment logs
├── reports/              # Daily detailed research reports
├── tests/                # Unit and integration tests
├── config/               # Configuration files
├── notebooks/            # EDA notebooks
└── app.py                # Streamlit/Gradio UI (Phase 6)
```

## References

1. [Effective hospital readmission prediction models using machine-learned features, BMC Health Services Research 2022](https://link.springer.com/article/10.1186/s12913-022-08748-y)
2. [ML-based prediction model for 30-day readmission risk in elderly patients, PMC 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12819643/)
3. [Heart failure readmission — optimal feature set, Frontiers in AI 2024](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1363226/full)
4. [UCI ML Repository — Diabetes 130-US Hospitals Dataset](https://archive.ics.uci.edu/dataset/296)
