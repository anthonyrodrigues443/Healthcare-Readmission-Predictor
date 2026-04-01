# Phase 1: Domain Research + Dataset + Baseline — Healthcare Readmission Predictor
**Date:** 2026-04-01
**Session:** 1 of 7
**Researcher:** Anthony Rodrigues

## Objective
Establish a credible foundation for predicting 30-day hospital readmission in diabetic patients. Answer: How do simple baselines (LACE index, LogReg) perform, and what's the gap to published SOTA?

## Research & References
1. [Springer, 2022] — ML models with machine-learned features achieved AUC 0.83, vs LACE's 0.66, using gradient boosting on administrative claims data — [https://link.springer.com/article/10.1186/s12913-022-08748-y](https://link.springer.com/article/10.1186/s12913-022-08748-y)
2. [PMC, 2025] — Stacking ensemble achieved AUC 0.867 for elderly diabetes + HF readmission, outperforming LACE by 0.09-0.26 — [https://pmc.ncbi.nlm.nih.gov/articles/PMC12819643/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12819643/)
3. [Frontiers in AI, 2024] — Heart failure readmission models report AUC 0.72-0.79 with XGBoost; key features are prior inpatient visits, diagnosis codes, and length of stay — [https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1363226/full](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1363226/full)

How research influenced today's experiments: Published benchmarks show LACE AUC ~0.66 and ML SOTA ~0.78-0.87. I used the same UCI Diabetes 130 Hospitals dataset commonly used in the literature. I engineered clinical features based on the LACE index components (Length of stay, Acuity, Comorbidities, ER visits) and added domain features cited in literature: polypharmacy risk, prior utilization intensity, lab procedure ratios.

## Dataset

| Metric | Value |
|--------|-------|
| Total samples | 101,766 |
| Raw features | 50 |
| Engineered features | 69 |
| Target variable | readmitted_binary (1 = readmitted <30 days) |
| Class distribution | 11.2% readmitted, 88.8% not readmitted |
| Train/Test split | 81,412 / 20,354 (stratified 80/20) |
| Source | UCI ML Repository — Diabetes 130-US Hospitals (1999-2008) |
| Missing data | weight (96.9%), max_glu_serum (94.7%), A1Cresult (83.3%), medical_specialty (49.1%) |

### Data Quality Findings
- **Severe class imbalance**: 11.2% positive class — will need class weighting, SMOTE, or threshold tuning
- **Missing data patterns**: weight is almost entirely missing (96.9%) — likely not recorded systematically (MNAR). A1C and glucose labs are missing 83-95% — tested vs not-tested is more informative than the value itself
- **Key clinical correlations**: number_inpatient (ρ=0.165), prior_utilization (ρ=0.126), lace_score (ρ=0.061)
- **No obvious data leakage**: discharge_disposition_id is somewhat correlated (ρ=0.051) but is known before the prediction point

## Experiments

### Experiment 1.1: Majority Class Baseline
**Hypothesis:** Establishes the floor — a model that always predicts "no readmission"
**Method:** Predict class 0 for all samples
**Result:** Accuracy 88.8%, F1 0.000, AUC 0.500
**Interpretation:** High accuracy is deceptive due to class imbalance. This model catches zero readmissions. Accuracy is a misleading metric for this problem — F1 and AUC are what matter.

### Experiment 1.2: LACE Index (Industry Standard Baseline)
**Hypothesis:** LACE should achieve AUC ~0.66 as reported in literature
**Method:** Computed LACE proxy using: Length of stay (0-7 pts), Acuity from admission_type_id (0/3 pts), Comorbidity from diagnosis group count (0-5 pts), ER visits (0-4 pts). Optimized threshold on train set.
**Result:** AUC 0.558, F1 0.213, Recall 0.742
**Interpretation:** Our LACE implementation underperforms published benchmarks (0.558 vs 0.66). This is expected: the UCI dataset lacks full comorbidity data (Charlson/Elixhauser indices) and the ER visit count is a rough proxy. LACE's recall is high (74%) but precision is abysmal (12.4%) — it flags too many patients.

### Experiment 1.3: Logistic Regression (All Features, Balanced Weights)
**Hypothesis:** LogReg with balanced class weights should outperform LACE
**Method:** StandardScaler + LogReg with class_weight='balanced', all 68 features
**Result:** AUC 0.648, F1 0.260, Recall 0.537
**Interpretation:** Beats LACE on AUC (+0.09) and F1 (+0.047). Recall drops to 54% but precision doubles to 17%. The all-features model is our strongest baseline. AUC 0.648 is in line with lower-end published ML results — significant room for improvement with better models.

### Experiment 1.4: Logistic Regression (Clinical Features Only, n=23)
**Hypothesis:** A curated set of clinically-motivated features should match or beat the all-features model
**Method:** 23 features selected based on clinical literature: LACE components, polypharmacy, prior utilization, lab/procedure ratios, diabetes-specific indicators
**Result:** AUC 0.645, F1 0.255, Recall 0.504
**Interpretation:** Nearly identical to the all-features model (AUC 0.645 vs 0.648) with 23 features vs 68. This means ~45 one-hot encoded features from race, gender, admission codes add almost no predictive value. The clinical features capture nearly all the signal. This is a genuine finding — domain knowledge compresses the feature space by 66% with negligible loss.

## Head-to-Head Comparison

| Rank | Model | Accuracy | F1 | Precision | Recall | AUC | Notes |
|------|-------|----------|-----|-----------|--------|-----|-------|
| 1 | LogReg (all, balanced) | 0.659 | 0.260 | 0.171 | 0.537 | **0.648** | Best overall |
| 2 | LogReg (clinical, n=23) | 0.672 | 0.255 | 0.171 | 0.504 | 0.645 | Nearly matches with 66% fewer features |
| 3 | LACE Index (thresh=6) | 0.387 | 0.213 | 0.124 | 0.742 | 0.558 | High recall, terrible precision |
| 4 | Majority Class | 0.888 | 0.000 | 0.000 | 0.000 | 0.500 | Useless — catches no readmissions |

## Key Findings
1. **Clinical features carry 99% of the signal.** 23 domain features match 68 total features (AUC 0.645 vs 0.648). Demographic and admission-code one-hots add noise, not signal.
2. **number_inpatient is the single strongest predictor** (coefficient 0.30, correlation 0.165). Patients with prior hospitalizations are most likely to be readmitted — consistent with all published literature.
3. **LACE index has high recall (74%) but is a fire hose** — it flags 12,459 out of 20,354 patients as high-risk. Hospitals can't follow up on 61% of discharges. ML models trade recall for much better precision.
4. **11.2% base rate makes this genuinely hard.** Published SOTA is AUC 0.78-0.87. Our LogReg baseline at 0.648 has clear room for improvement with gradient boosting, feature engineering, and ensemble methods.
5. **Missing A1C/glucose labs are informative.** Whether a test was ordered (A1C_tested, glucose_tested) is more predictive than the result — testing indicates clinical concern about glycemic control.

## Error Analysis
- LogReg misses 46% of readmissions (recall = 0.537) — the false negative rate is concerning for clinical use
- Most false positives are patients with high prior utilization who were NOT readmitted — suggests protective factors (good discharge planning?) not captured in the data
- The model struggles most with patients who have moderate risk profiles (LACE 5-8 range)

## Next Steps
- **Phase 2**: Try XGBoost, LightGBM, CatBoost, Random Forest — tree models handle imbalanced data and non-linear interactions better than LogReg
- Experiment with SMOTE vs class weights vs threshold tuning for the imbalance problem
- Create interaction features (e.g., age × number_inpatient, polypharmacy × number_diagnoses)
- Target: AUC > 0.70 (breaking into published ML performance range)

## References Used Today
- [1] [Effective hospital readmission prediction models using machine-learned features, BMC Health Services Research 2022](https://link.springer.com/article/10.1186/s12913-022-08748-y)
- [2] [ML-based prediction model for 30-day readmission risk in elderly patients, PMC 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12819643/)
- [3] [Heart failure readmission — optimal feature set, Frontiers in AI 2024](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1363226/full)
- [4] [UCI ML Repository — Diabetes 130-US Hospitals Dataset](https://archive.ics.uci.edu/dataset/296)
- [5] [Machine learning and LACE index for predicting 30-day readmissions, PubMed 2022](https://pubmed.ncbi.nlm.nih.gov/35661313/)

## Code Changes
- `src/data_pipeline.py` — Data download, cleaning, LACE computation, feature engineering (23 clinical features)
- `src/eda_and_baseline.py` — EDA analysis, 4 baseline models, comparison plots
- `config/config.yaml` — Project configuration
- `requirements.txt` — All dependencies
- `results/eda_overview.png` — 6-panel EDA visualization
- `results/correlation_heatmap.png` — Top 15 feature correlations
- `results/confusion_matrices_baseline.png` — Confusion matrices for 3 models
- `results/roc_curves_baseline.png` — ROC curves comparison
- `results/feature_importance_baseline.png` — Top 10 LogReg coefficients
- `results/metrics.json` — All metrics in JSON format
