# Mark Experiment Log

## 2026-04-01 | Phase 3 | Feature Engineering + Deep Dive

- Anthony reference point: CatBoost + domain features was Phase 2 champion at sensitivity 0.4469.
- Mark best model: BalancedRandomForest + interaction features
- Sensitivity: 0.5380
- F1: 0.2647
- AUC-ROC: 0.5627
- Highest-value new features: service_intensity, medication_complexity_index, lace_diagnosis_pressure, acute_utilization_burden
- Plots:
  - results/plots/phase3_mark_model_comparison.png
  - results/plots/phase3_mark_feature_delta.png
  - results/plots/phase3_mark_top_features.png
