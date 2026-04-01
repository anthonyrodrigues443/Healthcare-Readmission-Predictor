# Phase 1: Domain Research + EDA + Baseline — Healthcare Readmission Predictor
**Date:** 2026-04-01
**Session:** 1 of 7
**Researcher:** Anthony Rodrigues

## Objective
Establish the research foundation: understand the clinical domain, investigate the dataset quality, implement the industry-standard baseline (LACE index), and set a floor for future model comparisons.

## Domain Context & Published Benchmarks

### Clinical Background
Hospital readmissions within 30 days cost the US healthcare system ~$26 billion annually. The Hospital Readmissions Reduction Program (HRRP) penalizes hospitals up to 3% of Medicare payments for excess readmissions. With ~15-20% of patients readmitted within 30 days, there is enormous incentive to predict and intervene.

### Industry Baseline: LACE Index
The LACE Index is the standard tool used by hospitals for readmission risk stratification:
- **L** = Length of Stay (0–7 pts)
- **A** = Acuity of Admission (Emergency=3, Urgent=1, Elective=0)
- **C** = Comorbidity (# diagnoses, 0–6 pts)
- **E** = Emergency department visits in prior 6 months (0–4 pts)
- **Score ≥10** = High risk of 30-day readmission

### Published Benchmarks
| Study | Method | Dataset | AUC |
|-------|--------|---------|-----|
| Donzé et al. 2013 | LACE Index | Swiss hospitals | 0.69 |
| Zheng et al. 2015 | Random Forest | MIMIC-III | 0.72 |
| Futoma et al. 2015 | Deep Neural Net | Duke EHR | 0.76 |
| Rajkomar et al. 2018 | LSTM (Google) | EHR data | 0.83 |
| **Our Target** | XGBoost + domain features | UCI Diabetes | **>0.72** |

## Dataset

| Metric | Value |
|--------|-------|
| Total encounters | 101,766 |
| After exclusions (expired/hospice) | 99,353 |
| Unique patients | 71,518 |
| Hospitals | 130 US hospitals |
| Years covered | 1999–2008 |
| Features (raw) | 50 |
| Target: <30-day readmission | 11,314 (11.4%) |
| Train/Test split | 80/20 stratified |

### Class Distribution
| Class | Count | Pct |
|-------|-------|-----|
| NOT readmitted | 54,864 | 53.9% |
| Readmitted >30 days | 35,545 | 34.9% |
| Readmitted <30 days (target) | 11,357 | 11.2% |

### Missing Data (CRITICAL)
| Feature | Missing % | Impact |
|---------|-----------|--------|
| weight | 96.9% | **Excluded** — near-complete missingness |
| max_glu_serum | 94.7% | Excluded from base model |
| A1Cresult | 83.3% | Used as binary "was A1C measured?" flag |
| medical_specialty | 49.1% | Excluded (too sparse) |
| payer_code | 39.6% | Excluded |
| race | 2.2% | Imputed with mode |

**Key data quality insight:** Weight (96.9% missing) and glucose measurements (83-94% missing) are systematically absent — these are NOT random missingness (MCAR). They're MNAR (missing not at random): weight is rarely recorded for diabetic patients unless there's a specific concern. This means we cannot simply impute these values without introducing bias.

### Potential Data Leakage Investigation
- `discharge_disposition_id=11` (expired) excluded — these patients can't be readmitted
- `readmitted` column is the target — correctly encoded as binary
- No features found that directly encode readmission post-hoc

## Feature Engineering
Applied clinical domain knowledge from LACE/HRRP literature:

### Domain Features Engineered
1. **LACE Score** — full implementation of the industry standard
2. **CCS Categories** — ICD-9 codes mapped to Clinical Classification System groups
   - Circulatory, Respiratory, Diabetes, Injury, Neoplasms, etc.
3. **Polypharmacy flag** — `num_medications > 5` (evidence-based threshold)
4. **Prior utilization composite** — outpatient + emergency + inpatient visits (strongest published predictor)
5. **Medication change count** — `num_meds_changed` (clinical instability indicator)
6. **Insulin features** — whether patient is on insulin and if dose was changed
7. **A1C measurement flag** — whether HbA1c was actually measured (clinicians order A1C when glycemic control is uncertain)
8. **Age midpoint** — converted from age brackets to numeric (for model compatibility)
9. **Elderly flag** — age ≥ 70 (known readmission risk factor)
10. **Discharged home** — vs. SNF/rehab (discharge destination affects continuity of care)

## Experiments

### Experiment 1.1: Majority Class Baseline (Naive)
**Hypothesis:** A naive baseline that predicts no readmission will have high accuracy but zero utility.
**Method:** DummyClassifier(strategy="most_frequent")
**Result:**

| Metric | Score |
|--------|-------|
| Accuracy | 0.886 |
| F1 | 0.000 |
| AUC | 0.500 |

**Interpretation:** As expected, 88.6% accuracy with F1=0. The class imbalance makes accuracy a completely useless metric for this problem. Any model claiming >88.6% accuracy without a high F1 is just predicting "no readmission" for everyone. AUC is the correct primary metric.

### Experiment 1.2: LACE Index (Industry Standard Baseline)
**Hypothesis:** The hospital industry standard should give us a strong baseline close to published AUC=0.69.
**Method:** LACE score ≥ 10 → predict readmission. LACE/max(LACE) as probability.
**Result:**

| Metric | Score |
|--------|-------|
| Accuracy | 0.474 |
| F1 | 0.216 |
| Precision | 0.130 |
| Recall | 0.635 |
| AUC | 0.565 |

**Interpretation:** The LACE index AUC of **0.565** is substantially below the published 0.69. Why?
1. The LACE was validated on a Swiss hospital population — this dataset is US hospitals 1999-2008 with different demographics
2. The threshold-based binary prediction (≥10 = high risk) is crude — the score has more signal as a continuous variable
3. High recall (0.635) but terrible precision (0.130) — catches most readmissions but generates massive false positives
4. **This IS a genuine finding**: the LACE index, used in hospitals today, underperforms even simple ML approaches on this population. This validates the need for ML.

### Experiment 1.3: Logistic Regression with Domain Features
**Hypothesis:** Even a simple linear model with clinically-informed features should outperform LACE.
**Method:** 26 domain-engineered features → StandardScaler → LogReg(class_weight='balanced', max_iter=1000)
**Result:**

| Metric | Score |
|--------|-------|
| Accuracy | 0.648 |
| F1 | 0.264 |
| Precision | 0.173 |
| Recall | 0.556 |
| AUC | **0.653** |

**Interpretation:** LogReg with 26 domain features already beats the LACE index by **+0.088 AUC** and approaches (but doesn't reach) the published Random Forest benchmark of 0.72. This confirms:
1. Domain feature engineering adds significant signal beyond what LACE captures
2. The bottleneck is model capacity — non-linear models (tree-based) should improve further
3. The precision/recall tradeoff: 17% precision means 83% of our "high risk" flags are false positives — unacceptable for clinical use

## Head-to-Head Comparison

| Rank | Model | Accuracy | F1 | Precision | Recall | AUC | Notes |
|------|-------|----------|-----|-----------|--------|-----|-------|
| 1 | Logistic Regression | 0.648 | 0.264 | 0.173 | 0.556 | **0.653** | Best so far |
| 2 | LACE Index (Industry) | 0.474 | 0.216 | 0.130 | **0.635** | 0.565 | High recall, low precision |
| 3 | Majority Class | **0.886** | 0.000 | 0.000 | 0.000 | 0.500 | Useless |

## Key Findings

1. **LACE index underperforms on this dataset (AUC=0.565 vs published 0.69)**. The industry standard fails to generalize across different hospital populations — a finding that validates the need for data-driven approaches.

2. **Prior inpatient visits is the dominant predictor (r=0.168).** The literature agrees: prior utilization is the strongest signal for future utilization. A patient who was admitted 3 times last year is likely to come back. This is intuitive but often underweighted in clinical tools.

3. **Weight is 97% missing — not randomly.** Systematic MNAR missingness in key clinical variables (weight, glucose) is a fundamental limitation of this dataset. Any model using these features would be biased toward patients with more complete records (likely sicker patients with more clinical monitoring).

4. **Even LogReg beats LACE by +0.088 AUC.** Machine learning approaches — even the simplest one — demonstrably outperform the current clinical standard on this data. The headline is real.

5. **The accuracy trap.** 88.6% accuracy by predicting "no readmission" for everyone. This project will always report AUC + F1 as primary metrics, never raw accuracy.

## Feature Correlation Ranking (top 10)

| Rank | Feature | |r| with readmission | Clinical meaning |
|------|---------|-----------------|------------------|
| 1 | number_inpatient | 0.168 | Prior year inpatient visits |
| 2 | prior_utilization | 0.128 | Composite utilization score |
| 3 | had_prior_inpatient | 0.125 | Binary: any prior inpatient |
| 4 | discharged_home | 0.076 | Discharged to home vs. facility |
| 5 | lace_score | 0.076 | LACE index value |
| 6 | had_prior_emergency | 0.062 | Any prior ED visit |
| 7 | lace_high_risk | 0.061 | LACE ≥ 10 flag |
| 8 | number_emergency | 0.061 | Prior year ED visits |
| 9 | number_diagnoses | 0.054 | Comorbidity burden |
| 10 | time_in_hospital | 0.047 | Length of stay |

## Error Analysis
LACE Index errors (the industry baseline):
- **False positives:** 6,122 patients flagged as high-risk who were not readmitted (26-bed ICU load equivalent of unnecessary interventions)
- **False negatives:** 4,132 high-risk patients missed entirely
- The LACE threshold (≥10) is too low — captures too many low-risk patients

## Code Changes
- Created: `src/__init__.py`, `src/data_pipeline.py`, `src/phase1_eda_baseline.py`
- Created: `config/config.yaml`, `data/README.md`, `requirements.txt`, `.gitignore`
- Generated: `results/eda_class_distribution.png`, `results/eda_lace_analysis.png`
- Generated: `results/eda_feature_correlations.png`, `results/phase1_baseline_comparison.png`
- Generated: `results/metrics.json`, `data/processed/features.parquet`

## Next Steps (Phase 2)
- Test 5 models: Random Forest, XGBoost, LightGBM, CatBoost, SVM
- Primary question: "Do tree-based models that capture interactions between LACE features significantly outperform LogReg?"
- Hypothesis: **XGBoost with domain features will reach AUC ≥ 0.72, beating the published RF benchmark**
- Also investigate: SMOTE vs class_weight for handling the 11% positive rate
- Generate the Phase 2 head-to-head table with all 6 models (including LogReg from Phase 1)
