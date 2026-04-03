# Phase 5: Advanced techniques + explainability - Healthcare Readmission Predictor
**Date:** 2026-04-03
**Session:** 5 of 7
**Researcher:** Mark Rodrigues

## Objective
Can a subgroup-aware strategy fix Anthony's first-timer blind spot without destroying the overall ranking quality of the tuned CatBoost champion?

GitHub gate status: no Anthony Phase 5 PR was visible from local refs, and `gh` auth remained invalid, so this session proceeded from merged `origin/main` commit `58078b9` only.

## Building on Anthony's Work
**Anthony found:** The tuned `full_83` CatBoost reached `AUC 0.691 / F1 0.287`, but recall collapsed to `0.296` for patients with `prior_utilization == 0`, turning the model into a "frequent-flyer detector."
**My approach:** I kept Anthony's tuned CatBoost as the backbone and tested two complementary fixes: blunt subgroup reweighting and a routed first-timer specialist trained on compact clinical plus transition features.
**Combined insight:** Anthony's global CatBoost remains the best raw ranker, but Phase 5 shows the blind spot is a cohort-heterogeneity problem, not just a tuning problem. The specialist only helps if the deployment threshold is subgroup-aware too.

## Research & References
1. Selmer et al., 2025 - a systematic review/meta-analysis found pooled AUC around `0.71` and highlighted age, comorbidity burden, length of stay, ED visits, prior admissions, and polypharmacy as common predictors - https://doi.org/10.1016/j.ijnurstu.2025.105188
2. Luo et al., 2024 - a CatBoost human-machine collaboration model beat human-only and machine-only feature selection, and SHAP was used to surface clinically meaningful risk drivers - https://pmc.ncbi.nlm.nih.gov/articles/PMC11291677/
3. Gao et al., 2023 - a two-step extracted regression tree preserved much of black-box readmission performance while making the decision logic interpretable - https://pmc.ncbi.nlm.nih.gov/articles/PMC10243084/

How research influenced today's experiments: Selmer et al. reinforced that prior utilization and polypharmacy dominate readmission models, which matched Anthony's Phase 4 blind spot. Luo et al. motivated the compact specialist, and Gao et al. motivated extracting a shallow surrogate tree instead of stopping with another opaque CatBoost score.

## Dataset
| Metric | Value |
|--------|-------|
| Total samples | 101,766 |
| Features | 83 full-model features; 29 specialist features |
| Target variable | `readmitted_binary` |
| Class distribution | 11.2% positive |
| Train/Test split | 61,059 / 20,353 / 20,354 (train / calibration / test) |

## Experiments

### Experiment 5.1: Weighted CatBoost on the full cohort
**Hypothesis:** If the blind spot is mostly a weighting problem, giving extra weight to first-time readmissions should recover low-utilization recall.
**Method:** Re-trained Anthony's tuned `full_83` CatBoost with 3x sample weight for positive training rows where `prior_utilization == 0`, then calibrated with Platt scaling and tuned the threshold on the calibration split.
**Result:**

| Model | Accuracy | F1 | Precision | Recall | AUC | Low-util Recall |
|--------|----------|----|-----------|--------|-----|-----------------|
| Weighted CatBoost | 0.676 | 0.225 | 0.154 | 0.422 | 0.574 | 0.561 |

**Interpretation:** Reweighting recovered first-timer recall, but it wrecked discrimination. This is the same pattern Anthony saw with SMOTE in Phase 2: brute-force imbalance fixes create recall, but the ranking quality collapses.

### Experiment 5.2: First-timer specialist on compact transition features
**Hypothesis:** A model trained only on `prior_utilization == 0` patients will lean on discharge context, diagnosis burden, and medication load instead of utilization history.
**Method:** Trained a CatBoost specialist only on the low-utilization subset using the `clinical_plus_transition_29` feature set, then calibrated it on the low-utilization calibration slice.
**Result:**

| Slice model | F1 | Precision | Recall | AUC | Brier |
|------------|----|-----------|--------|-----|-------|
| Transition specialist only | 0.196 | 0.121 | 0.509 | 0.629 | 0.073 |

**Interpretation:** The specialist can recover roughly half the missed first-time readmissions, but only when evaluated with its own much lower threshold. That turned out to be the central Phase 5 insight.

### Experiment 5.3: Routed hybrids
**Hypothesis:** Keep Anthony's tuned CatBoost for the overall cohort and route only first-timers to a compact specialist.
**Method:** Combined the calibrated `full_83` CatBoost with either a `clinical_23` or `transition_29` specialist. I evaluated both probability routing under a single global threshold and subgroup-aware routing where first-timers use the specialist threshold while the rest use Anthony's tuned threshold.
**Result:**

| Hybrid | Accuracy | F1 | Precision | Recall | AUC | Low-util Recall | Notes |
|--------|----------|----|-----------|--------|-----|-----------------|-------|
| Hybrid router (`clinical_23`) | 0.730 | 0.273 | 0.195 | 0.454 | 0.664 | 0.000 | Global threshold kills specialist lift |
| Hybrid router (`transition_29`) | 0.691 | 0.275 | 0.186 | 0.524 | 0.673 | 0.205 | Better than clinical router, still threshold-limited |
| Hybrid + subgroup thresholds (`clinical_23`) | 0.569 | 0.255 | 0.158 | 0.660 | 0.664 | 0.520 | First-timer lift returns, but precision collapses |
| Hybrid + subgroup thresholds (`transition_29`) | 0.590 | 0.263 | 0.164 | 0.656 | 0.673 | 0.509 | Best Phase 5 tradeoff |

**Interpretation:** The routed model only works when the threshold is routed too. The `transition_29` version is the best compromise: it keeps AUC near Anthony's original CatBoost band while recovering roughly half of first-time readmissions, and it beats brute-force weighting by `+0.100` AUC and `+0.038` F1.

## Head-to-Head Comparison
| Rank | Model | Accuracy | F1 | Precision | Recall | AUC | Latency | Notes |
|------|-------|----------|-----|-----------|--------|-----|---------|-------|
| 1 | Hybrid + subgroup thresholds (`transition_29`) | 0.590 | 0.263 | 0.164 | 0.656 | 0.673 | 0.068 ms | Best tradeoff on the first-timer question |
| 2 | Tuned CatBoost (`full_83`) | 0.703 | 0.289 | 0.197 | 0.539 | 0.687 | 0.039 ms | Best raw AUC, but misses most first-timers |
| 3 | Hybrid router (`transition_29`) | 0.691 | 0.275 | 0.186 | 0.524 | 0.673 | 0.070 ms | Better probabilities than the clinical router |
| 4 | Hybrid + subgroup thresholds (`clinical_23`) | 0.569 | 0.255 | 0.158 | 0.660 | 0.664 | 0.066 ms | Threshold routing helps, but transition flags matter |
| 5 | Hybrid router (`clinical_23`) | 0.730 | 0.273 | 0.195 | 0.454 | 0.664 | 0.067 ms | Looks clean overall, but gives zero first-timer recall |
| 6 | Weighted CatBoost (`full_83`, low-util positives x3) | 0.676 | 0.225 | 0.154 | 0.422 | 0.574 | 0.047 ms | Blunt weighting is too destructive |

## Key Findings
1. The strongest Phase 5 fix was not generic reweighting. It was subgroup-aware routing with subgroup-aware thresholds.
2. Reweighting nearly tripled first-timer recall (`0.215 -> 0.561`), but AUC cratered from `0.687` to `0.574`. That is too expensive a trade.
3. Transition features matter specifically for first-timers. `discharge_post_acute` was the top specialist feature by both CatBoost importance and permutation importance.

## Frontier Model Comparison (when applicable)
| Model | Metric | Your Model | GPT-5.4 | Opus 4.6 | Winner |
|-------|--------|------------|---------|----------|--------|
| Environment status | Availability | Hybrid + subgroup thresholds (`transition_29`) | Blocked: `codex exec` returned `Access is denied` | Blocked: Claude CLI requires Git Bash on Windows | N/A |

## Error Analysis
- The clinical-only router looked fine on overall accuracy, but it predicted zero first-timer positives at the global threshold. That was a false sense of quality.
- The transition specialist concentrated risk on clinically interpretable features: `discharge_post_acute`, `number_diagnoses`, `time_in_hospital`, `age_numeric`, and `num_medications`.
- The extracted regression tree reached fidelity `R2 = 0.830`, and its top split was `discharge_post_acute`, followed by diagnosis count, diabetes medication status, and length of stay.

## Next Steps
- Phase 6 should package the routed `transition_29` path as the production candidate and expose subgroup-specific explanations in inference/UI.
- If frontier-model access is restored, run the blocked GPT-5.4 and Opus baselines on the same held-out patient slice used here.

## References Used Today
- [1] Selmer et al. (2025), *Identifying risk prediction models and predictors for hospital readmission in patients with medical conditions: A systematic review and meta-analysis* - https://doi.org/10.1016/j.ijnurstu.2025.105188
- [2] Luo et al. (2024), *SHAP based predictive modeling for 1 year all-cause readmission risk in elderly heart failure patients* - https://pmc.ncbi.nlm.nih.gov/articles/PMC11291677/
- [3] Gao et al. (2023), *Interpretable machine learning models for hospital readmission prediction: a two-step extracted regression tree approach* - https://pmc.ncbi.nlm.nih.gov/articles/PMC10243084/

## Code Changes
- Created `src/phase5_advanced_techniques.py`
- Generated `results/phase5_advanced_techniques.json`
- Generated `results/phase5_model_tradeoffs.png`
- Generated `results/phase5_feature_importance.png`
- Generated `results/phase5_surrogate_tree.png`
- Generated `results/phase5_extracted_rules.txt`
