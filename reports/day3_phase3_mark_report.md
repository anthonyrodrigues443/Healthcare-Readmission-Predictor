# Phase 3: Deep Dive on Top Models + Feature Engineering - Healthcare Readmission Predictor
**Date:** 2026-04-02
**Session:** 3 of 7
**Researcher:** Mark Rodrigues

## Objective
Can clinically grouped transition features and readmission-risk interactions recover Anthony's Phase 2 full-matrix booster lift without needing the full raw administrative feature set?

## Building on Anthony's Work
**Anthony found:** CatBoost on the full 68-feature matrix was the Phase 2 champion (`AUC 0.686 / F1 0.283`), and the extra administrative / categorical detail added roughly `+0.04 AUC` over the compact 23-feature clinical subset.
**My approach:** Instead of repeating the model sweep, I kept the same top three boosters and changed only the feature sets. I regrouped admission/discharge transitions into semantic risk flags, then added utilization and medication-burden interactions inspired by readmission literature.
**Combined insight:** Anthony showed that raw admin detail matters for boosters. Phase 3 shows why: grouped transition semantics recover only about one-third of that lift, and the rest still lives in finer-grained discharge/admission detail. The surprise is that the 29-feature transition set beats the full matrix on recall (`0.598` vs `0.576`) even while losing on AUC.

## Research & References
1. Woudneh et al., 2025 - previous hospitalizations, longer length of stay, and discharge protocols were significant readmission drivers in elderly inpatients - https://pmc.ncbi.nlm.nih.gov/articles/PMC12738708/
2. Hohl et al., 2021 - polypharmacy, multimorbidity, and longer stay increased 30-day readmission risk in polymedicated older adults - https://pmc.ncbi.nlm.nih.gov/articles/PMC8281082/
3. Strack et al. / UCI Diabetes 130 Hospitals dataset - HbA1c management, utilization history, and discharge context define the diabetic readmission benchmark we are building on - https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999-2008

How research influenced today's experiments: the literature repeatedly pointed to prior hospital use, LOS, discharge transitions, and medication burden as risk signals. That led me to test grouped discharge/admission flags first, then interaction features like medication burden per day and utilization x transition risk.

## Dataset
| Metric | Value |
|--------|-------|
| Total samples | 101,766 |
| Features | 83 maximum after Phase 3 engineering |
| Target variable | `readmitted_binary` |
| Class distribution | 11.2% positive / 88.8% negative |
| Train/Test split | 81,412 / 20,354 |

## Experiments

### Experiment 3.1: Workflow-only Booster Ceiling
**Hypothesis:** If model family was the main bottleneck, the top Phase 2 boosters should lift Mark's 8-feature workflow baseline close to Anthony's clinical/full models.
**Method:** Reused XGBoost, LightGBM, and CatBoost Phase 2 settings on the compact workflow feature set (`time_in_hospital`, medication count, diagnosis count, outpatient / ED / inpatient history, A1C tested, glucose tested).
**Result:**

| Model | Features | Accuracy | F1 | Precision | Recall | AUC |
|------|----------|----------|----|-----------|--------|-----|
| CatBoost | workflow_8 | 0.652 | 0.250 | 0.165 | 0.520 | 0.634 |
| LightGBM | workflow_8 | 0.657 | 0.245 | 0.162 | 0.497 | 0.624 |
| XGBoost | workflow_8 | 0.656 | 0.244 | 0.161 | 0.497 | 0.616 |

**Interpretation:** The ceiling stayed low. Better models could not rescue an underspecified workflow-only feature set; the bottleneck was the features, not the algorithm.

### Experiment 3.2: Transition-aware Clinical Set
**Hypothesis:** Grouping admission/discharge codes into clinically meaningful transition flags will recover much of Anthony's full-matrix lift without needing the whole raw feature matrix.
**Method:** Added six semantic transition flags to the 23-feature clinical base: `discharge_post_acute`, `discharge_ama_or_psych`, `discharge_home`, `admission_emergency`, `admission_transfer_source`, and `admission_ed_source`.
**Result:**

| Model | Features | Accuracy | F1 | Precision | Recall | AUC |
|------|----------|----------|----|-----------|--------|-----|
| CatBoost | clinical_plus_transition_29 | 0.636 | 0.268 | 0.173 | 0.598 | 0.659 |
| LightGBM | clinical_plus_transition_29 | 0.657 | 0.260 | 0.171 | 0.540 | 0.652 |
| XGBoost | clinical_plus_transition_29 | 0.675 | 0.260 | 0.175 | 0.513 | 0.646 |

**Interpretation:** This was the cleanest Phase 3 gain. CatBoost improved from `AUC 0.649 -> 0.659` on only six extra grouped variables, and recall climbed from `0.544 -> 0.598`. Transition semantics matter, but they still did not close the full-matrix gap.

### Experiment 3.3: Interaction-heavy Clinical Set
**Hypothesis:** Utilization x medication-burden x instability interactions should add another meaningful layer beyond grouped transitions.
**Method:** Added nine engineered interactions and ratios on top of the 29-feature transition set, including `meds_per_day`, `diagnoses_per_day`, `acute_prior_load`, `utilization_x_polypharmacy`, `utilization_x_transition`, `los_x_med_burden`, and `instability_x_utilization`.
**Result:**

| Model | Features | Accuracy | F1 | Precision | Recall | AUC |
|------|----------|----------|----|-----------|--------|-----|
| CatBoost | clinical_plus_interactions_38 | 0.640 | 0.265 | 0.171 | 0.581 | 0.661 |
| LightGBM | clinical_plus_interactions_38 | 0.659 | 0.259 | 0.171 | 0.534 | 0.647 |
| XGBoost | clinical_plus_interactions_38 | 0.677 | 0.254 | 0.171 | 0.492 | 0.645 |

**Interpretation:** More hand-built interactions did not translate into a better model. CatBoost gained only `+0.002 AUC` over the simpler transition set and actually lost recall (`0.598 -> 0.581`). The added complexity mostly created noise.

### Experiment 3.4: Full engineered control
**Hypothesis:** If the raw administrative detail still carries unique signal, the expanded full matrix will remain the champion even after better feature engineering on the compact sets.
**Method:** Re-ran the same top three boosters on the full 83-column matrix: Anthony's merged 68 features plus the Phase 3 transition and interaction columns.
**Result:**

| Model | Features | Accuracy | F1 | Precision | Recall | AUC |
|------|----------|----------|----|-----------|--------|-----|
| CatBoost | full_83 | 0.673 | 0.282 | 0.187 | 0.576 | 0.687 |
| LightGBM | full_83 | 0.693 | 0.277 | 0.188 | 0.527 | 0.674 |
| XGBoost | full_83 | 0.705 | 0.274 | 0.189 | 0.499 | 0.674 |

**Interpretation:** Anthony's overall story survives. The full matrix still wins on AUC and F1, and CatBoost stays the best family. The most informative raw feature remained `discharge_disposition_id`, which dominated the champion feature-importance ranking.

## Head-to-Head Comparison
| Rank | Model | Accuracy | F1 | Precision | Recall | AUC | Latency | Notes |
|------|-------|----------|----|-----------|--------|-----|---------|-------|
| 1 | CatBoost + full_83 | 0.673 | 0.282 | 0.187 | 0.576 | 0.687 | 0.026 ms | Overall champion |
| 2 | LightGBM + full_83 | 0.693 | 0.277 | 0.188 | 0.527 | 0.674 | 0.029 ms | Fastest strong model |
| 3 | XGBoost + full_83 | 0.705 | 0.274 | 0.189 | 0.499 | 0.674 | 0.084 ms | Similar AUC, lower recall |
| 4 | CatBoost + clinical_plus_interactions_38 | 0.640 | 0.265 | 0.171 | 0.581 | 0.661 | 0.014 ms | Interactions add little |
| 5 | CatBoost + clinical_plus_transition_29 | 0.636 | 0.268 | 0.173 | 0.598 | 0.659 | 0.013 ms | Best compact recall |
| 6 | CatBoost + clinical_23 | 0.650 | 0.257 | 0.169 | 0.544 | 0.649 | 0.011 ms | Phase 3 compact baseline |

## Key Findings
1. The full CatBoost model stayed champion (`AUC 0.687 / F1 0.282`), confirming Anthony's Phase 2 result from merged `main`.
2. Grouped transition semantics were the only compact feature engineering move that clearly helped: `CatBoost clinical_23 -> clinical_plus_transition_29` delivered `+0.010 AUC` and `+0.054 recall`.
3. The larger interaction bundle was mostly a negative result. It added nine extra features but did not beat the simpler 29-feature transition set on F1 or recall.
4. `discharge_disposition_id` remained the dominant raw feature in the full model, which explains why grouped flags recovered only about one-third of the full-matrix lift.

## Frontier Model Comparison (when applicable)
Deferred to Phase 5. Phase 3 focused on isolating the effect of feature engineering while holding the model family fixed.

## Error Analysis
- The full champion still misses about 42.4% of true readmissions (`recall 0.576`), so there is still substantial false-negative risk.
- The transition-aware CatBoost model trades calibration for sensitivity: it catches more readmissions than the full model (`0.598` recall) but loses AUC and precision.
- XGBoost benefited least from the grouped transition features, suggesting CatBoost is better at mixing continuous burden signals with sparse clinical flags on this dataset.

## Next Steps
- Phase 4 should tune CatBoost on both `full_83` and `clinical_plus_transition_29` so we can map the AUC vs recall tradeoff instead of assuming the single default threshold is optimal.
- Error analysis should focus on patients readmitted after "home" discharge versus post-acute transfer, because discharge disposition is now the clearest unresolved source of signal.

## References Used Today
- [1] Woudneh A.F. et al. (2025). Prediction of 30-day unplanned hospital readmission among elderly patients using patient and ward level factors. Scientific Reports. https://pmc.ncbi.nlm.nih.gov/articles/PMC12738708/
- [2] Hohl C.M. et al. (2021). Risk of 30-day hospital readmission associated with medical conditions and drug regimens of polymedicated, older inpatients discharged home. BMJ Open. https://pmc.ncbi.nlm.nih.gov/articles/PMC8281082/
- [3] Strack B. et al. / UCI Machine Learning Repository. Diabetes 130-US hospitals for years 1999-2008. https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999-2008

## Code Changes
- Created `src/phase3_feature_engineering.py` to run the Phase 3 feature-set matrix across XGBoost, LightGBM, and CatBoost.
- Generated `results/phase3_feature_engineering.json`, `results/phase3_auc_heatmap.png`, `results/phase3_model_comparison.png`, and `results/phase3_best_feature_importance.png`.
- Updated `results/metrics.json` and `results/EXPERIMENT_LOG.md` with Phase 3 results from the merged-main starting point.
