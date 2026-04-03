# Experiment Log - Healthcare Readmission Predictor
Master record of all experiments. Updated every session.

## Phase 1 - 2026-04-01 (Anthony)

| # | Approach | AUC | F1 | Precision | Recall | Delta vs Naive | Verdict |
|---|----------|-----|----|-----------|--------|----------------|---------|
| 1.1 | Majority Class (Naive) | 0.500 | 0.000 | 0.000 | 0.000 | baseline | Floor |
| 1.2 | LACE Index (Industry Standard) | 0.565 | 0.216 | 0.130 | 0.635 | +0.065 | Weak but clinically standard |
| 1.3 | LogReg + 26 domain features | 0.653 | 0.264 | 0.173 | 0.556 | +0.153 | Beats LACE; Phase 2 floor |

**Champion after Anthony's Phase 1:** Logistic Regression (AUC=0.653)
**Target for Phase 2:** AUC >= 0.72 (published RF benchmark)
**Key finding:** LACE underperformed its published validation on this diabetic readmission cohort.

## Phase 1 - 2026-04-01 (Mark)

| # | Approach | AUC | F1 | Precision | Recall | Delta vs Anthony clinical LogReg | Verdict |
|---|----------|-----|----|-----------|--------|----------------------------------|---------|
| 1.4 | Linear SVM + 8 workflow features | 0.633 | 0.252 | 0.172 | 0.473 | -0.012 | Best compact complementary baseline |
| 1.3 | Workflow triage rule | 0.621 | 0.248 | 0.176 | 0.418 | -0.024 | Strong interpretable fallback |
| 1.2 | BernoulliNB + test-ordering only | 0.539 | 0.000 | 0.000 | 0.000 | -0.106 | Important negative result |
| 1.1 | DummyClassifier (stratified) | 0.497 | 0.106 | 0.106 | 0.106 | -0.148 | Sanity floor |

**Champion after combined Phase 1:** Anthony's Logistic Regression (AUC=0.653) still leads overall, but Mark's workflow SVM is the strongest low-feature alternative (AUC=0.633).
**Combined insight:** 23 clinically derived features are enough, and performance only starts to degrade meaningfully when compressed all the way down to 7-8 workflow proxies.
**Key finding:** Prior utilization is monotonic (8.2% readmission at zero prior utilization vs 21.1% at `4+`), but missingness/test-ordering alone is too weak to stand on its own.

---
Append new phases below as research progresses.

## Phase 3 - 2026-04-02 (Mark)

| # | Approach | AUC | F1 | Precision | Recall | Delta vs Anthony CatBoost clinical | Verdict |
|---|----------|-----|----|-----------|--------|-----------------------------------|---------|
| 3.1 | CatBoost + full_83 | 0.687 | 0.282 | 0.187 | 0.576 | +0.038 | Leader |
| 3.2 | LightGBM + full_83 | 0.674 | 0.277 | 0.188 | 0.527 | +0.025 | Useful lift |
| 3.3 | XGBoost + full_83 | 0.674 | 0.274 | 0.189 | 0.499 | +0.025 | Useful lift |
| 3.4 | CatBoost + clinical_plus_interactions_38 | 0.661 | 0.265 | 0.171 | 0.581 | +0.012 | Useful lift |
| 3.5 | CatBoost + clinical_plus_transition_29 | 0.659 | 0.268 | 0.173 | 0.598 | +0.010 | Useful lift |
| 3.6 | LightGBM + clinical_plus_transition_29 | 0.652 | 0.260 | 0.171 | 0.540 | +0.003 | Useful lift |

**Champion after Mark's Phase 3:** CatBoost + full_83 (AUC=0.687, F1=0.282)
**Combined insight:** Grouped transition/discharge semantics recover roughly one-third of the full-matrix CatBoost lift, and they actually produce the highest recall among the compact feature sets, but raw discharge detail still wins on overall discrimination.
**Key finding:** The best manual feature engineering move was six semantic transition flags. The larger interaction bundle mostly added complexity without adding performance.


## Phase 4 - 2026-04-03 (Anthony)

| # | Approach | AUC | F1 | Precision | Recall | Delta vs Phase 3 Champion | Verdict |
|---|----------|-----|----|-----------|--------|---------------------------|---------|
| 4.1 | CatBoost default (Phase 3) | 0.687 | 0.285 | 0.186 | 0.604 | -0.000 | Baseline |
| 4.2 | CatBoost Optuna-tuned (80 trials) | 0.691 | 0.287 | 0.188 | 0.600 | +0.004 | Improved |
| 4.3 | Tuned CatBoost @ f1_optimal (t=0.520) | — | 0.291 | 0.199 | 0.539 | — | Operating point |
| 4.3 | Tuned CatBoost @ recall_75 (t=0.020) | — | 0.201 | 0.112 | 1.000 | — | Operating point |
| 4.3 | Tuned CatBoost @ recall_80 (t=0.020) | — | 0.201 | 0.112 | 1.000 | — | Operating point |
| 4.3 | Tuned CatBoost @ recall_85 (t=0.020) | — | 0.201 | 0.112 | 1.000 | — | Operating point |

**Most important hyperparameter:** random_strength (fANOVA importance=0.874)

**Champion after Phase 4:** Optuna-tuned CatBoost (AUC=0.6912, F1=0.287)
**Cross-validation:** AUC=0.6766 ± 0.0058
**Calibration:** Brier raw=0.2112, isotonic=0.0948
**Key finding:** Optuna tuning gained +0.004 AUC over default CatBoost. The most important hyperparameter is random_strength. At recall≥75% operating point, the model flags 100% of patients.
