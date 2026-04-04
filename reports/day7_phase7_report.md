# Phase 7: Native Categorical CatBoost + Validation Tightening — Healthcare Readmission Predictor
**Date:** 2026-04-05
**Session:** 7 of 7
**Researcher:** Anthony Rodrigues

## Objective
Answer two questions cleanly:

1. Was the production pipeline overstating performance by using the same holdout slice for both early stopping and calibration?
2. If the ceiling is still in representation rather than tuning, does preserving raw categorical structure inside CatBoost beat the compact one-hot `full_83` champion on the same deterministic split?

## Research & References
1. CatBoost documentation, "Transforming categorical features to numerical features" — CatBoost computes category statistics and category combinations rather than forcing everything through manual one-hot encoding - https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic
2. Prokhorenkova et al. (NeurIPS 2018), *CatBoost: unbiased boosting with categorical features* — shows why one-hot encoding breaks down on high-cardinality categories and why ordered target statistics reduce leakage/prediction shift - https://papers.nips.cc/paper/7898-catboost-unbiased-boosting-with-categorical-features.pdf
3. Sarthak et al. (2020), *EmbPred30: Assessing 30-days Readmission for Diabetic Patients using Categorical Embeddings* — on this same UCI diabetic readmission cohort, the authors treated categorical variables as first-class inputs rather than flattening them away - https://arxiv.org/abs/2002.11215
4. Zarghani (2024), *Comparative Analysis of LSTM Neural Networks and Traditional Machine Learning Models for Predicting Diabetes Patient Readmission* — on this dataset, discharge disposition and lab procedures again surface as critical predictors - https://arxiv.org/abs/2406.19980

How research influenced today's work: instead of adding another round of hand-built ratios, I tested whether the repo's own strongest signals were being diluted by one-hot preprocessing. The CatBoost paper and docs made that the obvious next experiment. The diabetic readmission papers were useful as a check on whether the recovered native-cat features looked clinically plausible rather than just numerically convenient.

## Dataset
| Metric | Value |
|--------|-------|
| Total samples | 101,766 |
| Production split | 61,059 / 10,176 / 10,177 / 20,354 (train / early-stop / calibration / test) |
| Production feature set | 83 one-hot engineered features |
| Native-cat feature set | 76 features total, including 39 categorical |
| Target variable | `readmitted_binary` |
| Class distribution | 11.2% positive |

## Experiments

### Experiment 7.1: Correct the production validation protocol
**Hypothesis:** Separating early stopping from calibration will lower the reported production score a bit, but the result will be more defensible.
**Method:** Re-trained the `full_83` Optuna-tuned CatBoost on a `60/10/10/20` split: train, early-stop, calibration, test.
**Result:**

| Model | Accuracy | F1 | Precision | Recall | AUC | Brier | Low-util Recall |
|------|----------|----|-----------|--------|-----|-------|-----------------|
| One-hot CatBoost (`full_83`) | 0.709 | 0.286 | 0.197 | 0.522 | 0.683 | 0.094 | 0.186 |

**Interpretation:** The original Phase 6 production number (`AUC 0.686`) was not wildly wrong, but it was a little optimistic because the same slice helped both early stopping and calibration. The corrected split lowers AUC by about `0.003` and gives the repo a cleaner reference point.

### Experiment 7.2: Preserve raw categorical structure inside CatBoost
**Hypothesis:** The repo's current preprocessing is flattening away useful diagnosis, medication-state, and discharge-pathway information.
**Method:** Kept CatBoost, the same tuned hyperparameters, the same deterministic split, and the same calibration recipe, but replaced the one-hot `full_83` matrix with a `raw+engineered` frame that preserves:

- Raw diagnosis codes (`diag_1`, `diag_2`, `diag_3`) plus grouped diagnosis families
- Raw medication states (`Up`, `Down`, `Steady`, `No`) instead of only aggregates
- Raw discharge/admission IDs as categorical values rather than ordered integers
- Race, gender, and age bins as native categories

**Result:**

| Model | Accuracy | F1 | Precision | Recall | AUC | Avg Precision | Brier | Low-util Recall |
|------|----------|----|-----------|--------|-----|---------------|-------|-----------------|
| One-hot CatBoost (`full_83`) | 0.709 | 0.286 | 0.197 | 0.522 | 0.683 | 0.219 | 0.094 | 0.186 |
| Native-cat CatBoost (`raw+engineered`) | 0.762 | 0.293 | 0.219 | 0.442 | 0.694 | 0.234 | 0.093 | 0.154 |

**Interpretation:** The native-cat version is the new best global ranker. It improves AUC by `+0.011`, improves F1 by `+0.007`, improves precision by `+0.023`, lowers the flagged-rate from `29.6%` to `22.5%`, and slightly improves Brier score. That is a real representation lift, not another tuning artifact.

### Experiment 7.3: Paired bootstrap on the test set
**Hypothesis:** If the native-cat lift is real, it should survive paired resampling rather than depend on one lucky split.
**Method:** Bootstrapped the shared held-out test set `400` times and compared candidate AUC minus baseline AUC on each resample.
**Result:**

| Statistic | Value |
|-----------|-------|
| Mean `ΔAUC` | `+0.0110` |
| 95% CI | `[+0.0070, +0.0155]` |
| `P(candidate > baseline)` | `1.00` |

**Interpretation:** The lift is stable. This is enough evidence to treat native categorical handling as the strongest remaining modeling direction in the repo.

## Feature-Level Readout
Top native-cat features:

1. `discharge_disposition_id`
2. `acute_prior_load`
3. `diag_1`
4. `diag_1_group`
5. `diag_3`
6. `diag_2`
7. `number_inpatient`
8. `insulin`

Why this matters: the lift did not come from obscure artifacts. It came from exactly the kind of raw categorical structure the one-hot production path had compressed away, and those features are clinically plausible.

## Key Findings
1. The repo's best remaining improvement was not another hyperparameter sweep. It was preserving raw categorical structure inside CatBoost.
2. The stricter split fixed an evaluation weakness without changing the qualitative story: the production model is still useful, just slightly weaker than the older report suggested.
3. The first-timer blind spot survived and even worsened under the better global ranker. That means Phase 5's subgroup-routing idea is still relevant.

## Recommendation
Do not immediately replace the production artifacts with the native-cat model. It is better, but promotion requires a richer serving surface:

1. The current Streamlit app and CLI only collect the compact 83-feature path.
2. The native-cat model needs raw diagnosis and medication-state inputs that the current UI does not expose.
3. The best next experiment is a Phase 5 + Phase 7 combination: native-cat backbone plus subgroup-aware routing or thresholds for first-timers.

## Code Changes
- Added `src/production_pipeline.py` to centralize split logic and serving-time feature construction
- Updated `src/train.py`, `src/evaluate.py`, `src/predict.py`, and `app.py` to use the shared production helpers
- Added `src/phase7_native_categorical_experiment.py`
- Generated `results/phase7_native_categorical.json`
- Generated `results/phase7_native_categorical_comparison.png`
- Generated `results/phase7_native_categorical_importance.png`
