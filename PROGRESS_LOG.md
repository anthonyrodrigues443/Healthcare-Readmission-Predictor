### 2026-04-01 | Healthcare-Readmission-Predictor | Phase 3: Feature engineering + deep dive on top models | Author: Mark

**Executive Summary:** Built on Anthony's Phase 2 model sweep with a focused Phase 3 deep dive. BalancedRandomForest plus a new interaction-heavy feature layer reached the best sensitivity of the project so far at **0.5380**, outperforming Anthony's CatBoost Phase 2 champion on recall while preserving similar precision.

**Building on Anthony's work:** Anthony found CatBoost was the best balanced model, LightGBM was the only architecture clearly helped by domain features, and vanilla Random Forest collapsed to 0.098 sensitivity. I kept the strong model families, added interaction-based utilization and complexity features, and tested BalancedRandomForest as a direct response to the RF failure mode.

**Experiments Run:**
| # | Approach | Score | Delta vs Baseline | Verdict |
|---|----------|-------|-------------------|---------|
| 1 | CatBoost + domain features | Sensitivity 0.4370 | Control vs Anthony's champion family | Stable control |
| 2 | CatBoost + interaction features | Sensitivity 0.4383 | +0.0013 vs domain CatBoost | Neutral |
| 3 | LightGBM + domain features | Sensitivity 0.3584 | Control vs Anthony's feature-friendly model | Lower recall, fair control |
| 4 | LightGBM + interaction features | Sensitivity 0.3604 | +0.0020 vs domain LightGBM | Small lift |
| 5 | BalancedRandomForest + domain features | Sensitivity 0.5287 | +0.4310 vs Anthony's plain RF | Major rescue |
| 6 | BalancedRandomForest + interaction features | **Sensitivity 0.5380** | +0.4403 vs Anthony's plain RF | Phase 3 winner |

**Key Findings:**
1. Anthony's Random Forest collapse was a sampling problem, not proof that forest ensembles are unusable for readmission prediction.
2. Mark's interaction features were most valuable when paired with BalancedRandomForest, where they improved sensitivity, F1, and AUC together.
3. The best new feature signals were `service_intensity`, `medication_complexity_index`, `lace_diagnosis_pressure`, and `acute_utilization_burden`.

**What Didn't Work:** CatBoost did not materially improve with the new interaction bundle, suggesting it had already captured most of the useful feature signal in Anthony's Phase 2 setup.

**Files Created/Modified:** PROGRESS_LOG.md, reports/day3_phase3_mark_report.md, results/phase3_mark_metrics.json, results/phase3_mark_feature_importance.csv, results/plots/phase3_mark_model_comparison.png, results/plots/phase3_mark_feature_delta.png, results/plots/phase3_mark_top_features.png, src/feature_engineering.py, src/train_phase3_mark.py, tests/test_phase3_features.py
**Next Phase:** Threshold-tune BalancedRandomForest and compare BRF vs CatBoost at matched sensitivity targets.
**Post-worthy?** Yes
**Post angle:** Anthony disproved vanilla RF; Mark showed balanced bagging plus interaction features flips the result and becomes the top recall model.
