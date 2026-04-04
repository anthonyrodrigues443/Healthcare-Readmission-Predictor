# Phase 6: Explainability & Model Understanding -- Healthcare Readmission Predictor
**Date:** 2026-04-04
**Session:** 6 of 7
**Researcher:** Mark Rodrigues

## Objective
Do the CatBoost champion's predictions rest on clinically defensible features, or on data artifacts? How do explanations differ for the low-utilization blind spot vs the high-utilization patients the model excels on?

## Building on Anthony's Work
**Anthony found:** Production pipeline achieves 0.01 ms/sample inference with calibrated CatBoost (AUC 0.686, F1 0.290, Brier 0.094). LACE and ML disagree on ~15% of patients. Streamlit UI with single-patient + batch prediction + LACE comparison.
**My approach:** Deep explainability -- SHAP global/local analysis, LIME individual explanations, partial dependence, subgroup-specific feature importance, and clinical domain taxonomy mapping. While Anthony productionized the model, I'm asking whether a clinician should TRUST it.
**Combined insight:** Anthony proved the model CAN be deployed; Mark's analysis shows WHERE it should (and shouldn't) be trusted.

## Research & References
1. [Scientific Reports, 2024] SHAP-based predictive modeling for 1-year readmission in elderly heart failure -- showed human-machine feature collaboration improves clinical acceptance
2. [Frontiers in Cardiovascular Medicine, 2025] ML prediction for 30-day readmission in T2DM+HF -- SHAP revealed creatinine, ER visits, HbA1c as top drivers; ensemble methods beat conventional statistics
3. [PMC, 2022] ML for predicting readmission among the frail -- emphasized that SHAP explanations are necessary for clinical adoption, not just performance metrics

How research influenced today's experiments: Published literature consistently shows prior utilization and discharge pathway as top readmission predictors. I built a clinical taxonomy mapping all 83 features to 11 clinical domains, then verified whether SHAP rankings match the literature.

## Dataset
| Metric | Value |
|--------|-------|
| Total samples | 101,766 |
| Test set | 20,354 |
| Features | 83 (full matrix) |
| Target variable | readmitted_binary (30-day) |
| Positive rate | 11.16% |
| SHAP subsample | 2,000 |

## Experiments

### Experiment 6.1: SHAP Global Feature Importance vs CatBoost Native
**Hypothesis:** SHAP and native CatBoost importance should roughly agree, but SHAP may reveal different ordering for correlated features.
**Method:** TreeExplainer on 2,000 test samples. Compared mean |SHAP| ranking against CatBoost native feature importance. Mapped all features to 11 clinical domains.
**Result:**

| Rank | SHAP Feature | |SHAP| | Native Feature | Native |
|------|-------------|-------|----------------|--------|
| 1 | acute_prior_load | 0.1594 | discharge_disposition_id | 17.4 |
| 2 | discharge_post_acute | 0.1495 | acute_prior_load | 6.5 |
| 3 | number_inpatient | 0.0907 | discharge_post_acute | 5.6 |
| 4 | discharge_disposition_id | 0.0767 | age_numeric | 4.3 |
| 5 | number_diagnoses | 0.0585 | number_inpatient | 4.0 |

Spearman rank correlation: rho = 0.971 (p = 2.65e-52)

**Domain breakdown:**
| Clinical Domain | SHAP Share |
|----------------|-----------|
| Prior Utilization | 30.2% |
| Discharge Pathway | 22.8% |
| Administrative/Other | 12.3% |
| Clinical Complexity | 12.0% |
| Medication | 11.1% |
| Interaction features | 3.5% |
| Demographics (age) | 2.1% |
| Glycemic Control | 1.3% |
| LACE | 1.2% |

**Interpretation:** The model is clinically defensible. Prior Utilization + Discharge Pathway account for 53% of total SHAP importance, consistent with AHRQ readmission literature. The concerning finding: Administrative/Other features (one-hot encoded diagnosis groups, gender, etc.) account for 12.3% -- these may be proxies for unmeasured confounders rather than causal signals. Glycemic control contributes only 1.3%, which is surprisingly low for a diabetic cohort -- but consistent with Phase 1 finding that A1C/glucose features were weak predictors.

### Experiment 6.2: Subgroup-Specific SHAP (Low-Util vs High-Util)
**Hypothesis:** The model uses fundamentally different features for low-utilization (the blind spot) vs high-utilization patients.
**Method:** Separate SHAP analysis on 1,000 samples from each subgroup.
**Result:**

| Feature | Low-Util |SHAP| | High-Util |SHAP| | Delta |
|---------|---------|---------|-------|
| acute_prior_load | 0.160 | 0.317 | +0.158 (high-util) |
| number_inpatient | 0.083 | 0.204 | +0.121 (high-util) |
| discharge_post_acute | 0.166 | 0.113 | -0.053 (low-util) |
| number_diagnoses | 0.069 | 0.026 | -0.044 (low-util) |

**Interpretation:** This explains the blind spot. For high-utilization patients, the model has a strong signal (acute_prior_load dominates at |SHAP| 0.317). For low-utilization patients, the model must fall back on weaker discharge pathway signals (discharge_post_acute becomes #1) and diagnosis count. The utilization features that drive the model's best predictions are structurally zero for first-timers. This is not a model bug -- it's a fundamental data limitation.

### Experiment 6.3: LIME Individual Explanations
**Hypothesis:** LIME can identify cases where the model's reasoning is clinically suspect.
**Method:** Selected 4 representative patients: true positive, false negative, false positive, low-utilization readmission.
**Result:**

| Case | Actual | Predicted | Prob | Top LIME Factor |
|------|--------|-----------|------|-----------------|
| True positive (high-risk) | Readmitted | HIGH | 0.625 | acute_prior_load > 2 (+0.058) |
| False negative (missed) | Readmitted | LOW | 0.000 | discharge_post_acute = 0 (-0.039) |
| False positive (over-flag) | Not readmitted | HIGH | 0.600 | acute_prior_load > 2 (+0.061) |
| Low-util readmission | Readmitted | HIGH | 0.184 | discharge_post_acute = 1 (+0.040) |

**Interpretation:** The false negative case is the most revealing. The model gave it probability 0.000 because ALL utilization/discharge signals point to "safe" -- zero prior admissions, no post-acute discharge, no AMA. LIME confirms the model has NO positive signal to work with. This patient was readmitted despite having none of the model's learned risk factors. For the false positive, the model is clinically reasonable -- high prior utilization IS a risk factor, it just happened to not result in readmission this time. The model's errors are understandable, not arbitrary.

### Experiment 6.4: Partial Dependence Plots
**Hypothesis:** Top features show monotonic relationships consistent with clinical intuition.
**Method:** PDP for top 6 SHAP features on 2,000 test samples.
**Result:** All 6 features show clinically expected relationships:
- acute_prior_load: monotonically increasing (more prior acute visits = higher risk)
- discharge_post_acute: step function increase (discharged to SNF/rehab = higher risk)
- number_inpatient: monotonically increasing
- discharge_disposition_id: non-monotonic (expected -- different IDs represent qualitatively different destinations)
- number_diagnoses: increasing then plateau around 9+
- n_medications_active: mild U-shape (both zero meds and many meds are risk flags)

**Interpretation:** No red flags. The model's learned relationships match clinical expectations, which supports trustworthiness for deployment.

### Experiment 6.5: SHAP Dependence Pairs
**Hypothesis:** Key clinical feature pairs show interaction effects that the model captures.
**Method:** SHAP dependence plots for 4 clinically meaningful pairs.
**Result:**
- number_inpatient x discharge_disposition_id: Strong interaction -- patients with high prior admissions discharged to SNF show the highest SHAP values
- number_inpatient x num_medications: Moderate interaction -- polypharmacy amplifies the inpatient history signal
- time_in_hospital x number_diagnoses: Weak interaction -- mostly additive
- age_numeric x prior_utilization: Clear stratification -- prior utilization matters more for younger patients (counterintuitive but interesting)

## Head-to-Head Comparison
| Experiment | Method | Key Finding | Clinical Validity |
|-----------|--------|------------|-------------------|
| 6.1 | SHAP Global | Prior Utilization = 30.2% of importance | HIGH -- matches literature |
| 6.2 | Subgroup SHAP | Low-util patients lack discriminative signal | EXPLAINS blind spot |
| 6.3 | LIME Cases | False negatives have zero positive signals | Model errors are understandable |
| 6.4 | PDP | All top features show expected relationships | HIGH -- no red flags |
| 6.5 | SHAP Pairs | Discharge x utilization interaction is real | Clinically plausible |

## Key Findings
1. **The model is clinically defensible.** 53% of total SHAP importance comes from Prior Utilization + Discharge Pathway -- the two domains that AHRQ and published literature consistently identify as top readmission predictors.
2. **The low-utilization blind spot is a data problem, not a model problem.** SHAP confirms the model literally has no discriminative signal for first-time patients: the features that matter most (acute_prior_load, number_inpatient) are structurally zero.
3. **Glycemic control contributes only 1.3% of importance** despite this being a diabetic cohort. This aligns with Phase 1 findings but challenges the clinical intuition that diabetes management quality should predict readmission.
4. **SHAP and native importance agree strongly (rho=0.971)** but differ on #1: SHAP says acute_prior_load, native says discharge_disposition_id. The native importance overweights features with many split points (discharge_disposition_id has more unique values).

## Error Analysis
- False negatives are predominantly low-utilization first-timers with home discharge -- the model has no positive signal for these patients
- False positives are high-utilization patients who happened to not be readmitted -- clinically reasonable flags
- The model's errors are interpretable and non-arbitrary, supporting clinical deployment with appropriate caveats

## Next Steps
- Phase 7: Testing + README + polish
- Add SHAP waterfall plots to the Streamlit UI for individual patient explanations
- Consider a "confidence" flag: when model relies mostly on utilization features (>80% of SHAP), flag prediction as "high confidence"; when no utilization signal available, flag as "low confidence -- consider additional clinical review"

## References Used Today
- [1] SHAP-based predictive modeling for readmission risk (Scientific Reports, 2024) -- https://www.nature.com/articles/s41598-024-67844-7
- [2] ML prediction for 30-day readmission with SHAP (Frontiers in Cardiovascular Medicine, 2025) -- https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2025.1673159/full
- [3] Explainable ML for ICU readmission prediction (arXiv, 2024) -- https://arxiv.org/html/2309.13781v4

## Code Changes
- Created: `src/phase6_explainability.py` -- 5 experiments (SHAP global, subgroup SHAP, LIME, PDP, dependence pairs)
- Created: `reports/day6_phase6_mark_report.md` -- this report
- Modified: `requirements.txt` -- added lime>=0.2
- Generated: 7 PNG plots + 4 LIME HTML files + `results/phase6_explainability.json`
