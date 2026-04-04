# Model Card: 30-Day Diabetic Readmission Risk Predictor

## Model Details

- **Model Type:** CatBoost gradient-boosted trees with isotonic calibration
- **Version:** 1.0.0
- **Training Date:** 2026-04-05
- **Developed By:** Anthony Rodrigues
- **Model Architecture:** CatBoost (788 iterations) → Isotonic CalibratedClassifierCV
- **Hyperparameters:** Optuna-tuned (80 trials) — depth=8, learning_rate=0.006, random_strength=0.185
- **Promotion Note:** A Phase 7 native-categorical CatBoost beat this model on AUC (`0.694` vs `0.683`), but it has not been promoted because the current app/CLI intake only supports the compact 83-feature production path.

## Intended Use

- **Primary Use:** Clinical decision-support for identifying diabetic patients at risk of 30-day hospital readmission
- **Intended Users:** Healthcare researchers, hospital informatics teams, data scientists
- **Out-of-Scope Use:** Direct clinical decision-making without physician oversight; non-diabetic populations; real-time triage without further validation

## Dataset

- **Source:** UCI Diabetes 130-US Hospitals for Years 1999-2008 (Strack et al., 2014)
- **Size:** 101,766 encounters from 130 US hospitals
- **Target:** 30-day readmission (binary: <30 days = positive)
- **Class Distribution:** 11.2% positive (readmitted within 30 days)
- **Split:** 60% train / 10% early-stop / 10% calibration / 20% test (stratified)

## Features

- **Total Features:** 83 (clinical + transition + interaction)
- **Feature Categories:**
  - Demographics: age, gender
  - Utilization: prior inpatient/outpatient/ER visits
  - Stay: length of stay, procedures, lab work, medications
  - Diagnosis: ICD-9 groups, diabetes flags
  - Discharge: disposition, admission source/type
  - Engineered: LACE score, polypharmacy, acute_prior_load, utilization bands

## Performance Metrics

| Metric | Test Set |
|--------|----------|
| AUC | 0.683 |
| F1 | 0.286 |
| Precision | 0.197 |
| Recall | 0.522 |
| Accuracy | 0.709 |
| Brier Score | 0.094 |
| Decision Threshold | 0.148 |
| Inference Latency | ~0.01 ms/sample |

### Subgroup Performance

| Subgroup | N | Readmit Rate | Recall | AUC |
|----------|---|-------------|--------|-----|
| Prior Util = 0 | 11,108 | 8.1% | 0.187 | 0.658 |
| Prior Util = 1 | 4,008 | 12.0% | 0.536 | 0.643 |
| Prior Util 4+ | 1,983 | 21.8% | 0.912 | 0.673 |
| Age < 45 | 1,213 | 10.5% | 0.567 | 0.771 |
| Age 80+ | 3,990 | 12.5% | 0.520 | 0.648 |

## Known Limitations

1. **First-timer blind spot:** Only 18.6% recall on patients with zero prior utilization. The stronger Phase 7 native-categorical model improved global AUC but pushed this slice even lower (`15.4%`), so the problem is cohort-specific rather than just representational.
2. **Dataset age:** Trained on 1999-2008 data; clinical practice and patient demographics have evolved significantly.
3. **Missing comorbidity detail:** UCI dataset lacks full Charlson/Elixhauser comorbidity indices, limiting LACE comparison accuracy.
4. **Low base rate:** 11.2% readmission rate is below the national average (15-20%), which constrains precision at any recall level.

## Ethical Considerations

- **Bias Risk:** Model performance varies by age and utilization history. First-time patients are systematically underserved.
- **Fairness:** Race/ethnicity features were not used. However, structural healthcare disparities may be encoded in utilization patterns.
- **Clinical Impact:** False negatives (missed readmissions) can lead to patient harm. False positives waste follow-up resources. The current threshold balances these tradeoffs but should be tuned per institution.
- **Transparency:** SHAP explanations are available for all predictions. The top feature (`discharge_disposition_id`) remains clinically interpretable and dominates both the production and native-categorical experiments.

## Calibration

- Raw CatBoost probabilities are materially miscalibrated on this cohort
- Isotonic calibration reduces held-out Brier to 0.094 and should be treated as part of the production model
- Clinical deployment MUST use the calibrated model (calibrator.joblib), not raw CatBoost predictions

## Training & Inference

```bash
# Train
python -m src.train

# Evaluate
python -m src.evaluate

# Predict (single patient)
python -m src.predict --input patient.json

# Predict (batch)
python -m src.predict --input patients.csv --output predictions.csv

# Interactive UI
streamlit run app.py
```

## References

- Strack et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates. BioMed Research International.
- van Walraven et al. (2010). Derivation and validation of the LACE index. CMAJ.
- AHRQ (2023). Statistical Brief on 30-Day Hospital Readmissions.
