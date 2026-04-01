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
