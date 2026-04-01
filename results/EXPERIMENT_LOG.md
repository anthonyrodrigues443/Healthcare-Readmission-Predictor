# Experiment Log — Healthcare Readmission Predictor
Master record of all experiments. Updated every session.

## Phase 1 — 2026-04-01 (Anthony)

| # | Approach | AUC | F1 | Precision | Recall | Δ vs Naive | Verdict |
|---|----------|-----|-----|-----------|--------|------------|---------|
| 1.1 | Majority Class (Naive) | 0.500 | 0.000 | 0.000 | 0.000 | baseline | Floor |
| 1.2 | LACE Index (Industry Standard) | 0.565 | 0.216 | 0.130 | 0.635 | +0.065 | Weak but clinically standard |
| 1.3 | LogReg + 26 domain features | 0.653 | 0.264 | 0.173 | 0.556 | **+0.153** | Beats LACE; Phase 2 floor |

**Champion after Phase 1:** Logistic Regression (AUC=0.653)
**Target for Phase 2:** AUC ≥ 0.72 (published RF benchmark)
**Key finding:** LACE index (industry tool) AUC=0.565 — significantly underperforms on this US diabetic population vs its published 0.69 validation

---
*Append new phases below as research progresses*
