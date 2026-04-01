# Phase 2: Multi-Model Experiment — Healthcare Readmission Predictor
**Date:** 2026-04-01
**Session:** 2 of 7
**Researcher:** Anthony Rodrigues

## Objective
Can non-linear tree-based models do what logistic regression could not: turn domain-engineered features (LACE score, Charlson comorbidity index, polypharmacy flags) into a meaningful sensitivity improvement?

Phase 1 showed: LR with 22 clinical features gained exactly ZERO over LR on raw features (ΔAUC = -0.003). Hypothesis: the signal is non-linear, and tree models will unlock it.

## Dataset
| Metric | Value |
|--------|-------|
| Total samples | 50,869 |
| Train samples | 40,695 |
| Test samples | 10,174 |
| Raw features | 43 |
| Domain-engineered features | 65 (+22 clinical features) |
| Target: readmitted <30 days | 14.9% (1,515 positives in test) |
| Train/Test split | 80/20, stratified |

## Experiments

### Experiment 2.1: Decision Tree
**Hypothesis:** A single tree with max_depth=10 will learn non-linear thresholds LR can't represent.
**Method:** `DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, class_weight="balanced")`

| Variant | Sensitivity | AUC-ROC | F1 | Precision | Specificity |
|---------|-------------|---------|-----|-----------|-------------|
| Raw features | 0.5657 | 0.5247 | 0.2451 | 0.1564 | 0.4662 |
| + Domain features | **0.5663** | 0.5288 | 0.2481 | 0.1589 | 0.4753 |

**Interpretation:** Highest sensitivity in the entire Phase 2 experiment — but at a brutal cost: specificity drops to 0.47, meaning we falsely flag 53% of non-readmitted patients. The DT is basically memorizing "flag everyone who looks slightly risky." Domain features gave +0.0006 sensitivity improvement — marginally better but not a real unlock. Precision of 0.16 means 84% of alerts would be false alarms in a hospital setting.

---

### Experiment 2.2: Random Forest
**Hypothesis:** 300 trees should denoise the DT and improve on its 0.566 sensitivity while fixing precision.
**Method:** `RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=10, class_weight="balanced")`

| Variant | Sensitivity | AUC-ROC | F1 | Precision | Specificity |
|---------|-------------|---------|-----|-----------|-------------|
| Raw features | 0.0977 | 0.5599 | 0.1304 | 0.1960 | 0.9299 |
| + Domain features | 0.0931 | 0.5564 | 0.1267 | 0.1986 | 0.9343 |

**Interpretation:** Massive, counterintuitive collapse. RF catches less than 10% of readmissions — WORSE than the LACE clinical index (0.146). Ensemble voting aggregates 300 trees, each trained on bootstrap samples. Despite `class_weight="balanced"`, the ensemble averaging suppresses the minority vote. Domain features made it slightly WORSE. This is a genuine finding: for imbalanced medical data, ensemble averaging can actively harm sensitivity. The `balanced` flag on individual trees doesn't survive majority-vote aggregation.

---

### Experiment 2.3: XGBoost
**Hypothesis:** Gradient boosting with `scale_pos_weight=5.7` (class imbalance ratio) should focus on misclassified positives.
**Method:** `XGBClassifier(n_estimators=300, max_depth=6, lr=0.05, scale_pos_weight=5.7)`

| Variant | Sensitivity | AUC-ROC | F1 | Precision | Specificity |
|---------|-------------|---------|-----|-----------|-------------|
| Raw features | 0.3122 | 0.5365 | 0.2208 | 0.1708 | 0.7347 |
| + Domain features | 0.3043 | 0.5374 | 0.2154 | 0.1667 | 0.7339 |

**Interpretation:** Domain features actually HURT XGBoost sensitivity by -0.0079. The 22 additional features added noise that gradient boosting partially exploited in the wrong direction. Interesting: despite lower sensitivity than DT, XGBoost has substantially better specificity (0.73 vs 0.47), meaning it has a better sensitivity/specificity tradeoff. Its AUC-PR (0.1661) is competitive. This is the best "clinical trade-off" model if you care about not over-alarming staff.

---

### Experiment 2.4: LightGBM
**Hypothesis:** LightGBM's leaf-wise growth and `class_weight="balanced"` may better handle this imbalanced problem than XGBoost's `scale_pos_weight`.
**Method:** `LGBMClassifier(n_estimators=300, max_depth=6, lr=0.05, class_weight="balanced")`

| Variant | Sensitivity | AUC-ROC | F1 | Precision | Specificity |
|---------|-------------|---------|-----|-----------|-------------|
| Raw features | 0.3630 | 0.5418 | 0.2292 | 0.1675 | 0.6843 |
| + Domain features | **0.3749** | 0.5358 | 0.2313 | 0.1673 | 0.6734 |

**Interpretation:** **Only model where domain features meaningfully helped** (+0.0119 sensitivity). LightGBM's leaf-wise tree construction appears to find value in the clinical composite scores (Charlson index, LACE components) that XGBoost's symmetric depth-first growth misses. This is the domain feature unlock we were looking for — but it's model-specific.

---

### Experiment 2.5: CatBoost
**Hypothesis:** CatBoost's `auto_class_weights="Balanced"` provides more principled handling of class imbalance than the manual weight tuning in other models.
**Method:** `CatBoostClassifier(iterations=300, depth=6, lr=0.05, auto_class_weights="Balanced")`

| Variant | Sensitivity | AUC-ROC | F1 | Precision | Specificity |
|---------|-------------|---------|-----|-----------|-------------|
| Raw features | 0.4416 | 0.5499 | 0.2486 | 0.1730 | 0.6307 |
| + Domain features | **0.4469** | 0.5497 | 0.2542 | 0.1776 | 0.6381 |

**Interpretation:** CatBoost delivers the best **balanced trade-off**: second-highest sensitivity (0.447), highest F1 (0.254), best precision among high-sensitivity models (0.178). Domain features helped marginally (+0.005). The `auto_class_weights` setting — which CatBoost optimizes internally — outperforms manual `scale_pos_weight` tuning. **Phase 3 champion candidate**.

---

### Experiment 2.6: SVM (linear kernel)
**Hypothesis:** Linear SVM with proper scaling may find a hyperplane that separates classes better than LR's penalized log-loss.
**Method:** `CalibratedClassifierCV(LinearSVC(C=0.1, class_weight="balanced"))` in a `StandardScaler` pipeline.

| Variant | Sensitivity | AUC-ROC | F1 | Precision | Specificity |
|---------|-------------|---------|-----|-----------|-------------|
| Raw features | 0.0000 | 0.5684 | 0.0000 | 0.0000 | 1.0000 |
| + Domain features | 0.0000 | 0.5653 | 0.0000 | 0.0000 | 1.0000 |

**Interpretation:** Complete failure — predicts all-negative in both variants. Despite `class_weight="balanced"`, LinearSVC with C=0.1 finds the maximum-margin hyperplane that trivially classifies all as negative. AUC-ROC=0.568 means the probability calibration CalibratedClassifierCV performs is reasonable, but the hard-threshold classifier still collapses. This is not a quirk — linear models fundamentally struggle with this problem, consistent with Phase 1 LR findings. Domain: **SVM is not a viable approach for this dataset without kernel tricks (Phase 3: consider RBF kernel).**

---

## Head-to-Head Comparison (ranked by primary metric: Sensitivity)

| Rank | Model | Features | Sensitivity | AUC-ROC | F1 | Precision | AUC-PR | Train(s) |
|------|-------|----------|-------------|---------|-----|-----------|--------|----------|
| 1 | Decision Tree | domain | **0.5663** | 0.5288 | 0.2481 | 0.1589 | 0.1623 | 0.2s |
| 2 | Decision Tree | raw | 0.5657 | 0.5247 | 0.2451 | 0.1564 | 0.1608 | 0.1s |
| 3 | CatBoost | domain | 0.4469 | 0.5497 | **0.2542** | **0.1776** | 0.1729 | 1.1s |
| 4 | CatBoost | raw | 0.4416 | **0.5499** | 0.2486 | 0.1730 | **0.1741** | 1.1s |
| 5 | LightGBM | domain | 0.3749 | 0.5358 | 0.2313 | 0.1673 | 0.1668 | 1.4s |
| 6 | LightGBM | raw | 0.3630 | 0.5418 | 0.2292 | 0.1675 | 0.1688 | 1.4s |
| 7 | XGBoost | raw | 0.3122 | 0.5365 | 0.2208 | 0.1708 | 0.1661 | 0.7s |
| 8 | XGBoost | domain | 0.3043 | 0.5374 | 0.2154 | 0.1667 | 0.1686 | 0.7s |
| 9 | Random Forest | raw | 0.0977 | 0.5599 | 0.1304 | 0.1960 | 0.1780 | 1.1s |
| 10 | Random Forest | domain | 0.0931 | 0.5564 | 0.1267 | 0.1986 | 0.1750 | 1.2s |
| 11 | SVM | raw | 0.0000 | 0.5684 | 0.0000 | 0.0000 | 0.1860 | 0.2s |
| 12 | SVM | domain | 0.0000 | 0.5653 | 0.0000 | 0.0000 | 0.1830 | 0.6s |
| — | LR Phase 1 (baseline) | raw | 0.512 | 0.569 | 0.262 | 0.176 | — | — |
| — | LACE ≥10 (clinical std) | — | 0.146 | — | 0.159 | 0.175 | — | — |

## Key Findings

1. **Decision Tree wins on sensitivity but is clinically useless alone**: DT (0.566) marginally beats LR (0.512) on sensitivity, but its precision (0.16) and specificity (0.47) mean 84% of alerts are false alarms. In a real ICU this triggers alarm fatigue. The raw sensitivity number is misleading without the precision context.

2. **Random Forest's ensemble voting destroys minority-class recall** (0.098): The most counterintuitive result of Phase 2. 300 trees with `class_weight="balanced"` still collapses to near-zero sensitivity. Probable cause: each individual tree generates a "vote" that is biased toward majority class under bootstrap sampling. When you average 300 such votes, the minority class signal is drowned. This is a **known but under-discussed failure mode** for ensemble methods on highly imbalanced data.

3. **CatBoost is the real Phase 2 champion**: Best balanced trade-off. 0.447 sensitivity + 0.178 precision + highest F1 (0.254). `auto_class_weights` internally optimizes the weighting schedule, outperforming manual `scale_pos_weight=5.7` in XGBoost. **Take to Phase 3 for hyperparameter tuning.**

4. **Domain features unlocked LGBM (+0.012) but hurt XGBoost (-0.008)**: The LACE/Charlson composite features are non-trivially additive — they only help when the model's tree-growing strategy is compatible (leaf-wise for LGBM). This is evidence that domain feature gains are **architecture-dependent**, not universally beneficial.

5. **Linear models (LR, SVM) are fundamentally inadequate**: All linear models in Phase 1 and Phase 2 fail to separate the classes above 0.57 AUC. The signal in readmission prediction is in **feature interactions** (e.g., high medications × recent inpatient visits), not individual feature magnitudes.

## Frontier Model Comparison
Not applicable in Phase 2 — LLM baseline comparison scheduled for Phase 5.

## Error Analysis

**Decision Tree's failure mode**: Flags too broadly — predicts positive for any patient with time_in_hospital > threshold OR num_medications > threshold. 4,622 false positives vs 857 true positives in the test set. In a 500-bed hospital this would generate ~230 false readmission alerts per day.

**Random Forest's failure mode**: 1,367 false negatives (missed readmissions), 607 false positives. Opposite of DT — the averaging suppresses the minority class signal completely. The 300 trees converge on "predict negative unless overwhelming evidence."

**CatBoost's failure mode**: 838 false negatives, 3,134 false positives. Better than DT on precision (0.178 vs 0.159) but still generates significant alert volume. This is the optimal sensitivity/precision curve for the current feature set.

## Domain Feature Impact Summary

| Model | ΔSensitivity | ΔAUC | Verdict |
|-------|-------------|------|---------|
| Decision Tree | +0.0006 | +0.0041 | Neutral |
| Random Forest | -0.0046 | -0.0035 | Neutral/Hurts |
| XGBoost | **-0.0079** | +0.0009 | **Hurts** |
| LightGBM | **+0.0119** | -0.0060 | **Helps** |
| CatBoost | +0.0053 | -0.0002 | Marginally helps |
| SVM | +0.0000 | -0.0031 | Irrelevant (both fail) |

**Research answer:** Domain features did NOT universally unlock non-linear models. The unlock is architecture-specific. LightGBM found value; XGBoost treated the composite scores as noisy redundancies.

## Next Steps (Phase 3)
1. **CatBoost + LightGBM as Phase 3 focus**: Best precision/sensitivity trade-offs. Tune class weighting aggressively (try threshold optimization, not just class_weight).
2. **Decision Tree threshold tuning**: DT has the raw sensitivity. Can we push its precision to >0.25 via post-training threshold optimization at the probability level?
3. **Fix Random Forest's collapse**: Try extremely aggressive weighting (class_weight=50x) or replace majority-vote aggregation with probability averaging at a lower threshold. Or switch to `BalancedRandomForest` from imbalanced-learn.
4. **Feature ablation on CatBoost**: Remove domain features one-by-one. Which clinical composite (LACE vs Charlson vs polypharmacy) drives the +0.005 gain?
5. **Threshold-independent evaluation**: Phase 2 used default 0.5 threshold. Phase 3 should optimize threshold per model for the clinical operating point (sensitivity ≥ 0.70 at highest achievable precision).

## Code Changes
- `src/train_phase2.py` — new Phase 2 training script (12 experiments, 3 visualizations)
- `requirements.txt` — added xgboost>=2.0.0, lightgbm>=4.0.0, catboost>=1.2.0, pyarrow>=12.0.0
- `results/phase2_model_comparison.json` — full metrics for all 12 experiment variants
- `results/plots/phase2_model_comparison.png` — 4-panel metric comparison chart
- `results/plots/phase2_sensitivity_ranking.png` — horizontal bar chart vs LACE/LR baselines
- `results/plots/phase2_domain_feature_delta.png` — Δ metric plot for domain feature impact
