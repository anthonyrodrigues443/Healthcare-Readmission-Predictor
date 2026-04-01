# Phase 1: Domain Research + Dataset + Baseline — Healthcare Readmission Predictor
**Date:** 2026-04-01
**Session:** 1 of 7
**Researcher:** Mark Rodrigues

## Objective
Test whether clinician workflow proxies can recover most of the baseline readmission signal without repeating Anthony's wider logistic regression setup. Specifically: how far can prior utilization plus lab-ordering indicators go on their own?

## Building on Anthony's Work
**Anthony found:** LACE underperformed badly on this diabetic readmission cohort, while a 23-feature clinical logistic regression nearly matched a 68-feature version (AUC 0.645 vs 0.648), implying domain features carry almost all of the useful baseline signal.
**My approach:** Instead of re-running logistic regression, I compressed the feature space further to workflow proxies: prior inpatient/emergency utilization, length of stay, medication burden, and whether A1C / serum glucose were ordered at all.
**Combined insight:** The signal really is compressible, but not infinitely. Anthony showed 23 clinically derived features are enough; I found an 8-feature workflow SVM still reaches AUC 0.633, but missingness/test-ordering features alone collapse. Clinical workflow context matters, pure missingness does not.

## Research & References
1. Emi-Johnson and Nkrumah, 2025 — Used the exact same UCI Diabetes 130 Hospitals dataset and compared logistic regression, random forest, XGBoost, and DNNs on an 80/20 split, validating this dataset as a credible benchmark for Phase 1. — https://pmc.ncbi.nlm.nih.gov/articles/PMC12085305/
2. Mcllhargey et al., 2023 — Showed that missing laboratory patterns in EHRs can be informative because they reflect clinician concern and patient trajectory, motivating explicit test-ordering features instead of naive imputation. — https://pubmed.ncbi.nlm.nih.gov/36738870/
3. Ren et al., 2024 — Systematic review of missing-data handling in EHR models; concluded no single imputation strategy dominates and missingness mechanisms strongly affect model quality. — https://pmc.ncbi.nlm.nih.gov/articles/PMC11615160/

How research influenced today's experiments: I treated A1C / serum glucose ordering as first-class features rather than filling those sparse lab values with generic imputations. I also focused on compact, clinician-readable baselines instead of moving into tree ensembles early.

## Dataset
| Metric | Value |
|--------|-------|
| Total samples | 101,766 |
| Features | 68 engineered features |
| Target variable | `readmitted_binary` (`<30` days = 1) |
| Class distribution | 11.2% positive / 88.8% negative |
| Train/Test split | 81,412 / 20,354 (stratified 80/20) |

## Experiments

### Experiment 1.1: Missingness / Ordering Signal Audit
**Hypothesis:** In EHR data, whether a lab is ordered may carry nearly as much signal as the result itself because ordering reflects clinician concern.
**Method:** Measured readmission rate by A1C ordered vs not ordered, serum glucose ordered vs not ordered, and prior-utilization bucket (`0`, `1`, `2-3`, `4+`).
**Result:** A1C ordered patients had *lower* readmission (9.85% vs 11.42%), glucose ordered patients had *higher* readmission (12.36% vs 11.09%), and prior utilization was monotonic from 8.18% (`0`) to 21.07% (`4+`).
**Interpretation:** Ordering behavior is informative, but not uniformly in one direction. A1C appears to reflect chronic diabetes management, while acute serum glucose testing looks more like a severity signal.

### Experiment 1.2: Workflow Triage Rule
**Hypothesis:** A hand-built score based on inpatient / ED history, polypharmacy, LOS, and test ordering should outperform LACE while staying fully interpretable.
**Method:** Built a 7-feature additive rule and tuned the decision threshold on the training set.
**Result:** AUC 0.621, F1 0.248, Precision 0.176, Recall 0.418.
**Interpretation:** The rule comfortably beats LACE's ranking on this branch's reported metrics, but it gives up too much recall relative to Anthony's logistic baseline. Good heuristic floor, not good enough as champion.

### Experiment 1.3: BernoulliNB on Test-Ordering Features
**Hypothesis:** Missingness / workflow indicators alone may already carry meaningful predictive signal.
**Method:** Trained Bernoulli Naive Bayes on 8 binary features: A1C / glucose tested and high, medication change, diabetes meds, polypharmacy, and diabetes-primary diagnosis.
**Result:** AUC 0.539, F1 0.000, Recall 0.000.
**Interpretation:** Missingness alone is a weak ranking signal and a useless operating-point classifier at the default threshold. This is the important negative result of the session.

### Experiment 1.4: Linear SVM on Compact Workflow Features
**Hypothesis:** A non-logistic linear model using only workflow proxies can recover most of Anthony's clinical logistic performance with a much smaller feature set.
**Method:** StandardScaler + `LinearSVC(class_weight="balanced")` on 8 features: LOS, meds, diagnoses, outpatient / emergency / inpatient utilization, and A1C / glucose ordering flags.
**Result:** Accuracy 0.687, F1 0.252, Precision 0.172, Recall 0.473, AUC 0.633.
**Interpretation:** This is the session winner. It lands only 0.012 AUC behind Anthony's 23-feature clinical logistic regression (0.645) while using a smaller, workflow-centric feature set and a different model family.

## Head-to-Head Comparison
| Rank | Model | Accuracy | F1 | Precision | Recall | AUC | Latency | Notes |
|------|-------|----------|-----|-----------|--------|-----|---------|-------|
| 1 | Anthony: LogReg (all, balanced) | 0.659 | 0.260 | 0.171 | 0.537 | 0.648 | n/a | Best AUC so far |
| 2 | Anthony: LogReg (clinical, n=23) | 0.672 | 0.255 | 0.171 | 0.504 | 0.645 | n/a | Strong compact baseline |
| 3 | Mark: Linear SVM (workflow compact set) | 0.687 | 0.252 | 0.172 | 0.473 | 0.633 | 0.0002 ms/sample | 8 workflow features only |
| 4 | Mark: Workflow triage rule | 0.717 | 0.248 | 0.176 | 0.418 | 0.621 | 0.0001 ms/sample | Best pure heuristic |
| 5 | Anthony: LACE Index | 0.387 | 0.213 | 0.124 | 0.742 | 0.558 | n/a | High recall, poor precision |
| 6 | Mark: BernoulliNB (test-ordering only) | 0.888 | 0.000 | 0.000 | 0.000 | 0.539 | 0.0004 ms/sample | Weak ranking, unusable classifier |

## Key Findings
1. An 8-feature workflow SVM retains almost all of Anthony's baseline signal: AUC 0.633 vs 0.645 for his 23-feature clinical logistic regression.
2. Prior utilization is the cleanest compact signal in the dataset: readmission climbs from 8.2% with zero prior utilization to 21.1% for patients with `4+` prior encounters.
3. Ordering behavior is asymmetric. A1C ordering correlates with *lower* readmission, while serum glucose ordering correlates with *higher* readmission.
4. Missingness-only modeling is not enough. BernoulliNB barely beats chance on AUC and predicts zero positives at the default operating point.

## Frontier Model Comparison (when applicable)
Deferred to Phase 5, per the project plan. Phase 1 stayed focused on real-data baselines and feature-family discovery.

## Error Analysis
- The compact SVM keeps precision roughly flat versus Anthony's logistic baselines but loses recall, which means compact workflow proxies still miss a meaningful fraction of true readmissions.
- The triage rule mostly misses patients without heavy prior utilization history, suggesting first-time or low-history readmissions need richer clinical context.
- Test-ordering features without utilization context are too weak and too sparse to support a usable classifier.

## Next Steps
- Phase 2 should test tree models on top of Anthony's clinical feature set and this compact workflow feature set to see whether non-linear interactions recover the lost recall.
- Investigate interaction features such as `number_inpatient x glucose_tested` and `time_in_hospital x polypharmacy`, since workflow signals appear useful only when contextualized.

## References Used Today
- [1] Emi-Johnson OG, Nkrumah KJ. Predicting 30-Day Hospital Readmission in Patients With Diabetes Using Machine Learning on Electronic Health Record Data. Cureus, 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12085305/
- [2] Mcllhargey MJ, Fawzy A, Brothers I, et al. Informative missingness: What can we learn from patterns in missing laboratory data in the electronic health record? Journal of Biomedical Informatics, 2023. https://pubmed.ncbi.nlm.nih.gov/36738870/
- [3] Ren W, Liu Z, Wu Y, Zhang Z. Moving Beyond Medical Statistics: A Systematic Review on Missing Data Handling in Electronic Health Records. Health Data Science, 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11615160/

## Code Changes
- `src/data_pipeline.py` — replaced the fragile `ucimlrepo` dependency with direct UCI zip download and added compatibility wrappers used by Anthony's stale Phase 1 script.
- `src/phase1_mark_missingness_baselines.py` — Mark's complementary Phase 1 experiment on workflow and missingness signals.
- `results/mark_missingness_and_utilization.png` — readmission uplift by test-ordering and prior-utilization bucket.
- `results/mark_phase1_comparison.png` — Anthony vs Mark Phase 1 baseline comparison.
- `results/mark_phase1_missingness_baselines.json` — Mark's experiment outputs.
- `results/metrics.json` — appended Mark's Phase 1 complementary metrics.
