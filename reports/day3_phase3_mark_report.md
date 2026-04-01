# Phase 3: Feature Engineering + Deep Dive on Top Models
**Date:** 2026-04-01
**Session:** 3 of 7
**Researcher:** Mark Rodrigues

## Objective
Build directly on Anthony's Phase 2 finding that CatBoost was the best balanced model and LightGBM was the only architecture clearly helped by domain features. Instead of repeating another broad model sweep, test whether interaction-heavy utilization and complexity features can:

1. Improve the top Phase 2 families further.
2. Rescue the Random Forest failure mode with a better imbalance-aware variant.

## Building on Anthony's Work
- Anthony found: plain Random Forest collapsed to **0.098 sensitivity**, CatBoost + domain features led Phase 2 at **0.447 sensitivity / 0.254 F1**, and LightGBM was the only model clearly helped by the original domain feature bundle.
- My complementary approach: keep CatBoost and LightGBM in play, add a new interaction-focused feature family, and test `BalancedRandomForest` as a direct response to the minority-recall collapse Anthony uncovered in standard RF.
- Combined insight: the RF failure was not "tree ensembles do not work" in general. It was a **sampling/aggregation problem**. Once the ensemble is balanced correctly and fed interaction features, it becomes the strongest sensitivity model in the project so far.

## Dataset
| Metric | Value |
|--------|-------|
| Source | Synthetic faithful replica used by existing pipeline |
| Train samples | 40,695 |
| Test samples | 10,174 |
| Positive rate | 14.9% |
| Anthony domain features | 65 |
| Mark interaction feature set | 75 |

## Mark Interaction Features Added
- `acute_utilization_burden`
- `chronic_complexity_index`
- `medication_complexity_index`
- `service_intensity`
- `ed_inpatient_mix`
- `polypharmacy_utilization_overlap`
- `age_comorbidity_pressure`
- `lace_diagnosis_pressure`
- `lab_medication_pressure`
- `outpatient_inpatient_gap`

These target the non-linear structure Anthony kept seeing in the data: utilization x acuity, burden x complexity, and service intensity rather than isolated raw counts.

## Experiments Run
| # | Approach | Sensitivity | AUC-ROC | F1 | Precision | Verdict |
|---|----------|-------------|---------|----|-----------|---------|
| 3.1 | CatBoost + Anthony domain features | 0.4370 | 0.5533 | 0.2482 | 0.1733 | Control |
| 3.2 | CatBoost + Mark interaction features | 0.4383 | 0.5528 | 0.2472 | 0.1721 | Neutral |
| 3.3 | LightGBM + Anthony domain features | 0.3584 | 0.5365 | 0.2278 | 0.1670 | Control |
| 3.4 | LightGBM + Mark interaction features | 0.3604 | 0.5468 | 0.2363 | 0.1758 | Small gain |
| 3.5 | BalancedRandomForest + Anthony domain features | 0.5287 | 0.5600 | 0.2603 | 0.1726 | Major rescue |
| 3.6 | BalancedRandomForest + Mark interaction features | **0.5380** | **0.5627** | **0.2647** | **0.1756** | Phase 3 winner |

## Key Findings

### 1. BalancedRandomForest rescues Anthony's RF collapse
Anthony's standard Random Forest with 300 trees achieved **0.0977 sensitivity** in Phase 2. BalancedRandomForest jumps to **0.5287 sensitivity** on the same general feature family, and **0.5380** with Mark's interaction features.

That is the central Phase 3 result. The issue was not "forest models cannot work here." The issue was that vanilla bootstrap voting drowned the minority class signal.

### 2. Mark's interaction features help the right models
- CatBoost: essentially neutral on sensitivity (+0.0013), slightly down on AUC/F1.
- LightGBM: modest but real gains across sensitivity, AUC, F1, and precision.
- BalancedRandomForest: clearest lift, with +0.0093 sensitivity and +0.0044 F1 over the domain-only BRF.

The feature bundle is not universally good. It helps when the learner benefits from explicit interaction structure and balanced sampling.

### 3. The strongest new signals are interaction terms, not new raw variables
Top features in the winning model included:
- `service_intensity`
- `medication_complexity_index`
- `lace_diagnosis_pressure`
- `acute_utilization_burden`

This is strong support for Anthony's Phase 1 hypothesis that the problem is fundamentally interaction-driven.

## Plots Generated
- `results/plots/phase3_mark_model_comparison.png`
- `results/plots/phase3_mark_feature_delta.png`
- `results/plots/phase3_mark_top_features.png`

## What Didn't Work
- CatBoost did not meaningfully improve from the new interaction bundle. Its Phase 2 domain setup was already close to saturating the useful signal.
- The new features improved LightGBM more on ranking quality and precision than on raw sensitivity, so they are not a universal recall unlock by themselves.

## Combined Anthony + Mark Insight
Anthony showed that:
- linear models are weak,
- LightGBM can use domain features,
- and standard RF collapses badly on recall.

This run adds the missing explanation:
- **balanced bagging plus interaction features restores the tree-ensemble path**,
- and the best current sensitivity model is now a forest again, just not the vanilla one Anthony disproved.

## Next Steps
1. Threshold-tune BalancedRandomForest to see whether sensitivity can cross 0.60 without precision collapsing below the LACE baseline.
2. Run feature ablation on the four winning interaction terms to see which carry most of the BRF lift.
3. Compare BRF vs CatBoost at matched sensitivity bands rather than only default threshold.
