"""Phase 6 (Mark): Deep explainability for healthcare readmission champion model.

Complements Anthony's Phase 6 production pipeline with SHAP global/local
analysis, LIME individual explanations, partial dependence, and subgroup-
specific feature importance — all connected to clinical domain literature.

Key questions:
1. Do SHAP rankings match traditional CatBoost feature importance?
2. Does the model rely on clinically defensible features or data artifacts?
3. How do explanations differ for low-utilization vs high-utilization patients?
4. Can LIME identify cases where the model's reasoning is clinically suspect?

Usage:
    python -m src.phase6_explainability
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.data_pipeline import clean_and_engineer, download_dataset
from src.phase3_feature_engineering import add_phase3_features, get_feature_sets

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
RESULTS_DIR = Path("results")
REPORTS_DIR = Path("reports")
MODEL_DIR = Path("models")


def load_data_and_model():
    """Load processed data and trained champion model."""
    raw_df = download_dataset()
    df = clean_and_engineer(raw_df)
    df = add_phase3_features(df)
    target = "readmitted_binary"

    feature_sets = get_feature_sets(df)
    feature_cols = feature_sets["full_83"]

    X = df[feature_cols]
    y = df[target]

    # Same split as production training
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )

    # Load the calibrated model
    calibrator = joblib.load(MODEL_DIR / "calibrator.joblib")
    manifest = json.loads((MODEL_DIR / "training_manifest.json").read_text(encoding="utf-8"))
    threshold = manifest["optimal_threshold"]

    # Get the raw CatBoost model for SHAP
    raw_model = calibrator.estimator.estimator

    return X_test, y_test, raw_model, calibrator, feature_cols, threshold


# ─── Clinical feature taxonomy ──────────────────────────────────────────────

CLINICAL_TAXONOMY = {
    # Prior utilization — strongest signal per Phase 1-5
    "number_inpatient": ("Prior Utilization", "Prior inpatient admissions — strongest readmission predictor per AHRQ"),
    "number_emergency": ("Prior Utilization", "Prior ER visits — acute care dependency marker"),
    "number_outpatient": ("Prior Utilization", "Prior outpatient visits — healthcare engagement"),
    "prior_utilization": ("Prior Utilization", "Sum of all prior encounters — composite utilization"),
    "acute_prior_load": ("Prior Utilization", "Weighted acute load (2×inpatient + ER)"),
    "utilization_band": ("Prior Utilization", "Binned utilization intensity"),
    "utilization_x_polypharmacy": ("Interaction", "Utilization × polypharmacy flag"),
    "utilization_x_transition": ("Interaction", "Utilization × post-acute discharge"),
    "instability_x_utilization": ("Interaction", "Glycemic instability × utilization"),

    # Discharge/admission pathway
    "discharge_disposition_id": ("Discharge Pathway", "Raw discharge destination — finer signal than grouped flags"),
    "discharge_post_acute": ("Discharge Pathway", "Discharged to SNF/rehab/home-health"),
    "discharge_ama_or_psych": ("Discharge Pathway", "Left AMA or psych transfer — high-risk"),
    "discharge_home": ("Discharge Pathway", "Discharged home — generally protective"),
    "admission_type_id": ("Admission Pathway", "Emergency vs urgent vs elective"),
    "admission_emergency": ("Admission Pathway", "Emergency admission flag"),
    "admission_source_id": ("Admission Pathway", "Referral, transfer, or ER origin"),
    "admission_transfer_source": ("Admission Pathway", "Transferred from another facility"),
    "admission_ed_source": ("Admission Pathway", "Admitted through ER"),

    # Medication burden
    "num_medications": ("Medication", "Total active medications — polypharmacy proxy"),
    "n_medications_changed": ("Medication", "Medications changed during stay"),
    "n_medications_active": ("Medication", "Active medication count (capped at 5)"),
    "med_changed": ("Medication", "Any medication change flag"),
    "diabetes_med": ("Medication", "Diabetic medication prescribed"),
    "polypharmacy": ("Medication", "Polypharmacy flag (>5 meds)"),
    "meds_per_day": ("Medication", "Medication intensity per hospital day"),
    "los_x_med_burden": ("Interaction", "LOS × medication count"),

    # Clinical complexity
    "number_diagnoses": ("Clinical Complexity", "Number of ICD diagnoses — comorbidity proxy"),
    "num_lab_procedures": ("Clinical Complexity", "Lab tests ordered — acuity signal"),
    "num_procedures": ("Clinical Complexity", "Procedures performed"),
    "diagnoses_per_day": ("Clinical Complexity", "Diagnostic density per hospital day"),
    "lab_procedure_ratio": ("Clinical Complexity", "Labs per hospital day"),
    "procedure_ratio": ("Clinical Complexity", "Procedures per hospital day"),

    # Glycemic control
    "A1C_high": ("Glycemic Control", "HbA1c > 7% — poor long-term control"),
    "A1C_tested": ("Glycemic Control", "A1C tested during stay — quality marker"),
    "glucose_high": ("Glycemic Control", "Max glucose > 200 mg/dL"),
    "glucose_tested": ("Glycemic Control", "Glucose tested during stay"),
    "glycemic_instability": ("Glycemic Control", "A1C or glucose elevated"),
    "diabetes_primary": ("Glycemic Control", "Diabetes as primary diagnosis"),

    # Stay characteristics
    "time_in_hospital": ("Stay", "Length of stay in days"),
    "age_numeric": ("Demographics", "Patient age (midpoint of bin)"),
    "lace_score": ("LACE", "Industry-standard LACE readmission index"),
}


def get_domain_label(feature: str) -> tuple[str, str]:
    """Return (domain_category, clinical_description) for a feature."""
    if feature in CLINICAL_TAXONOMY:
        return CLINICAL_TAXONOMY[feature]
    return ("Administrative/Other", f"One-hot encoded: {feature}")


# ─── Experiment 1: SHAP Global Analysis ─────────────────────────────────────

def run_shap_global(X_test, raw_model, feature_cols):
    """Compute SHAP values and compare with native CatBoost importance."""
    import shap

    print("\n" + "=" * 60)
    print("EXPERIMENT 6.1: SHAP Global Feature Importance")
    print("=" * 60)

    # Use a subsample for SHAP (computational cost)
    n_shap = min(2000, len(X_test))
    X_shap = X_test.sample(n=n_shap, random_state=RANDOM_STATE)

    t0 = time.time()
    explainer = shap.TreeExplainer(raw_model)
    shap_values = explainer.shap_values(X_shap)
    shap_time = time.time() - t0
    print(f"SHAP computed on {n_shap} samples in {shap_time:.1f}s")

    # Global mean |SHAP| importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_ranking = sorted(
        zip(feature_cols, mean_abs_shap),
        key=lambda x: x[1],
        reverse=True,
    )

    # Native CatBoost importance
    native_imp = raw_model.get_feature_importance()
    native_ranking = sorted(
        zip(feature_cols, native_imp),
        key=lambda x: x[1],
        reverse=True,
    )

    print("\n--- Top 20 Features: SHAP vs CatBoost Native ---")
    print(f"{'Rank':>4} | {'SHAP Feature':<35} {'|SHAP|':>8} | {'Native Feature':<35} {'Native':>8}")
    print("-" * 100)
    for i in range(20):
        sf, sv = shap_ranking[i]
        nf, nv = native_ranking[i]
        cat, _ = get_domain_label(sf)
        print(f"{i+1:>4} | {sf:<35} {sv:>8.4f} | {nf:<35} {nv:>8.1f}")

    # Rank correlation between methods
    shap_order = {f: i for i, (f, _) in enumerate(shap_ranking)}
    native_order = {f: i for i, (f, _) in enumerate(native_ranking)}
    common = set(shap_order) & set(native_order)
    r1 = [shap_order[f] for f in common]
    r2 = [native_order[f] for f in common]
    from scipy.stats import spearmanr
    rho, pval = spearmanr(r1, r2)
    print(f"\nSpearman rank correlation (SHAP vs Native): rho = {rho:.3f} (p = {pval:.2e})")

    # Domain category breakdown
    print("\n--- SHAP Importance by Clinical Domain ---")
    domain_shap = {}
    for feat, val in shap_ranking:
        cat, _ = get_domain_label(feat)
        domain_shap.setdefault(cat, 0.0)
        domain_shap[cat] += val
    total = sum(domain_shap.values())
    for cat, val in sorted(domain_shap.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:<30s} {val:>8.4f} ({val/total*100:>5.1f}%)")

    # --- SHAP summary plot ---
    fig, ax = plt.subplots(figsize=(10, 10))
    shap.summary_plot(shap_values, X_shap, feature_names=feature_cols,
                      max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase6_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/phase6_shap_summary.png")

    # --- SHAP bar plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, feature_names=feature_cols,
                      plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase6_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/phase6_shap_bar.png")

    # --- Domain category pie chart ---
    fig, ax = plt.subplots(figsize=(8, 8))
    cats = sorted(domain_shap.items(), key=lambda x: x[1], reverse=True)
    labels = [c for c, _ in cats]
    vals = [v for _, v in cats]
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    wedges, texts, autotexts = ax.pie(vals, labels=labels, autopct="%1.1f%%",
                                       colors=colors, textprops={"fontsize": 9})
    ax.set_title("SHAP Importance by Clinical Domain", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase6_domain_shap_pie.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/phase6_domain_shap_pie.png")

    return shap_values, X_shap, shap_ranking, native_ranking, rho, domain_shap


# ─── Experiment 2: Subgroup-Specific SHAP ───────────────────────────────────

def run_shap_subgroups(X_test, y_test, raw_model, feature_cols, threshold, calibrator):
    """Compare SHAP explanations for low-util vs high-util patients."""
    import shap

    print("\n" + "=" * 60)
    print("EXPERIMENT 6.2: Subgroup-Specific SHAP Analysis")
    print("=" * 60)

    low_mask = X_test["prior_utilization"] == 0
    high_mask = X_test["prior_utilization"] >= 4

    subgroups = {
        "low_utilization": X_test[low_mask],
        "high_utilization": X_test[high_mask],
    }

    subgroup_results = {}
    explainer = shap.TreeExplainer(raw_model)

    for name, X_sg in subgroups.items():
        n_sg = min(1000, len(X_sg))
        X_sample = X_sg.sample(n=n_sg, random_state=RANDOM_STATE)
        sv = explainer.shap_values(X_sample)
        mean_abs = np.abs(sv).mean(axis=0)
        ranking = sorted(zip(feature_cols, mean_abs), key=lambda x: x[1], reverse=True)

        print(f"\n--- {name.replace('_', ' ').title()} (n={n_sg}) ---")
        print(f"{'Rank':>4} | {'Feature':<35} {'|SHAP|':>8} | {'Domain':<25}")
        print("-" * 80)
        for i, (f, v) in enumerate(ranking[:15]):
            cat, _ = get_domain_label(f)
            print(f"{i+1:>4} | {f:<35} {v:>8.4f} | {cat:<25}")

        subgroup_results[name] = {
            "n": n_sg,
            "top_10": [(f, float(v)) for f, v in ranking[:10]],
            "shap_values": sv,
            "X_sample": X_sample,
        }

    # --- Compare: what features differ most between subgroups ---
    low_dict = dict(subgroup_results["low_utilization"]["top_10"])
    high_dict = dict(subgroup_results["high_utilization"]["top_10"])
    all_feats = set(list(low_dict.keys()) + list(high_dict.keys()))

    print("\n--- Feature Importance Delta: High-Util minus Low-Util ---")
    deltas = []
    for f in all_feats:
        lo = low_dict.get(f, 0.0)
        hi = high_dict.get(f, 0.0)
        deltas.append((f, hi - lo, hi, lo))
    deltas.sort(key=lambda x: abs(x[1]), reverse=True)
    for f, d, hi, lo in deltas[:10]:
        direction = "+high-util" if d > 0 else "-low-util"
        print(f"  {f:<35} delta = {d:>+.4f} ({direction})")

    # --- Subgroup SHAP bar comparison plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for idx, (name, data) in enumerate(subgroup_results.items()):
        ax = axes[idx]
        top = data["top_10"]
        feats = [f for f, _ in top][::-1]
        vals = [v for _, v in top][::-1]
        colors = ["#d95f02" if f in ("number_inpatient", "prior_utilization",
                                      "number_emergency", "acute_prior_load")
                  else "#1b9e77" for f in feats]
        ax.barh(feats, vals, color=colors)
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title(f"{name.replace('_', ' ').title()} Patients\n(n={data['n']})")
    plt.suptitle("SHAP Feature Importance: Low vs High Utilization", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase6_subgroup_shap_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/phase6_subgroup_shap_comparison.png")

    return subgroup_results, deltas


# ─── Experiment 3: LIME Individual Explanations ─────────────────────────────

def run_lime_analysis(X_test, y_test, calibrator, feature_cols, threshold):
    """LIME explanations for representative patients across risk spectrum."""
    from lime.lime_tabular import LimeTabularExplainer

    print("\n" + "=" * 60)
    print("EXPERIMENT 6.3: LIME Individual Explanations")
    print("=" * 60)

    # Get predictions
    probs = calibrator.predict_proba(X_test)[:, 1]

    # Select representative patients
    cases = {}

    # True positive high-risk
    tp_mask = (probs >= threshold) & (y_test == 1)
    if tp_mask.sum() > 0:
        tp_idx = X_test[tp_mask].index
        # Pick the highest-risk true positive
        best_tp = tp_idx[np.argmax(probs[tp_mask.values])]
        cases["true_positive_high_risk"] = best_tp

    # False negative (missed readmission)
    fn_mask = (probs < threshold) & (y_test == 1)
    if fn_mask.sum() > 0:
        fn_idx = X_test[fn_mask].index
        # Pick the most confident false negative (lowest prob among actual readmissions)
        worst_fn = fn_idx[np.argmin(probs[fn_mask.values])]
        cases["false_negative_missed"] = worst_fn

    # False positive (unnecessary flag)
    fp_mask = (probs >= threshold) & (y_test == 0)
    if fp_mask.sum() > 0:
        fp_idx = X_test[fp_mask].index
        best_fp = fp_idx[np.argmax(probs[fp_mask.values])]
        cases["false_positive_over_flag"] = best_fp

    # Low-utilization readmission (the blind spot)
    low_util_readmit = (X_test["prior_utilization"] == 0) & (y_test == 1)
    if low_util_readmit.sum() > 0:
        lu_idx = X_test[low_util_readmit].index
        cases["low_util_readmission"] = lu_idx[0]

    # Build LIME explainer
    explainer = LimeTabularExplainer(
        X_test.values,
        feature_names=feature_cols,
        class_names=["Not Readmitted", "Readmitted"],
        mode="classification",
        random_state=RANDOM_STATE,
    )

    lime_results = {}
    for case_name, idx in cases.items():
        patient = X_test.loc[idx]
        actual = int(y_test.loc[idx])
        prob = float(probs[X_test.index.get_loc(idx)])
        pred_label = "HIGH" if prob >= threshold else "LOW"

        t0 = time.time()
        exp = explainer.explain_instance(
            patient.values,
            calibrator.predict_proba,
            num_features=10,
            num_samples=2000,
        )
        lime_time = time.time() - t0

        print(f"\n--- Case: {case_name} ---")
        print(f"  Actual: {'Readmitted' if actual else 'Not Readmitted'} | "
              f"Predicted: {pred_label} (prob={prob:.3f}) | LIME time: {lime_time:.1f}s")

        top_factors = exp.as_list()
        print(f"  Top LIME factors:")
        for factor, weight in top_factors[:8]:
            direction = "+readmit" if weight > 0 else "-no readmit"
            print(f"    {factor:<50s} weight={weight:>+.4f} ({direction})")

        # Check clinical plausibility
        clinically_sensible = []
        for factor, weight in top_factors[:5]:
            # Parse feature name from LIME's condition format
            feat_name = factor.split(" ")[0].strip("<>=!").strip()
            cat, desc = get_domain_label(feat_name)
            clinically_sensible.append({
                "factor": factor,
                "weight": float(weight),
                "domain": cat,
                "description": desc,
            })

        lime_results[case_name] = {
            "actual": actual,
            "predicted_prob": prob,
            "predicted_label": pred_label,
            "lime_time_s": round(lime_time, 2),
            "top_factors": [(f, float(w)) for f, w in top_factors[:10]],
            "clinically_sensible": clinically_sensible,
        }

        # Save LIME explanation as HTML
        exp.save_to_file(str(RESULTS_DIR / f"phase6_lime_{case_name}.html"))

    # --- LIME comparison plot ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for idx, (case_name, data) in enumerate(lime_results.items()):
        ax = axes[idx // 2][idx % 2]
        factors = data["top_factors"][:8]
        labels = [f[:40] for f, _ in factors][::-1]
        weights = [w for _, w in factors][::-1]
        colors = ["#d95f02" if w > 0 else "#1b9e77" for w in weights]
        ax.barh(labels, weights, color=colors)
        ax.axvline(0, color="black", linewidth=0.5)
        actual_str = "Readmitted" if data["actual"] else "Not Readmitted"
        ax.set_title(f"{case_name.replace('_', ' ').title()}\n"
                     f"Actual: {actual_str} | Prob: {data['predicted_prob']:.3f}",
                     fontsize=10)
        ax.set_xlabel("LIME Weight")
    plt.suptitle("LIME Explanations: 4 Representative Patients", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase6_lime_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: results/phase6_lime_comparison.png")

    return lime_results


# ─── Experiment 4: Partial Dependence for Top Features ──────────────────────

def run_partial_dependence(X_test, raw_model, feature_cols, shap_ranking):
    """Partial dependence plots for top 6 features."""
    from sklearn.inspection import PartialDependenceDisplay

    print("\n" + "=" * 60)
    print("EXPERIMENT 6.4: Partial Dependence Plots")
    print("=" * 60)

    top_features = [f for f, _ in shap_ranking[:6]]
    top_indices = [feature_cols.index(f) for f in top_features]

    X_pdp = X_test.sample(n=min(2000, len(X_test)), random_state=RANDOM_STATE)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, (feat, feat_idx) in enumerate(zip(top_features, top_indices)):
        ax = axes[i // 3][i % 3]
        cat, desc = get_domain_label(feat)

        try:
            PartialDependenceDisplay.from_estimator(
                raw_model, X_pdp, [feat_idx],
                feature_names=feature_cols,
                ax=ax,
                kind="average",
            )
            ax.set_title(f"{feat}\n({cat})", fontsize=10)
            print(f"  PDP for {feat} ({cat}): computed")
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes, ha="center")
            print(f"  PDP for {feat}: failed - {e}")

    plt.suptitle("Partial Dependence: Top 6 SHAP Features", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase6_partial_dependence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/phase6_partial_dependence.png")


# ─── Experiment 5: SHAP Interaction Effects ─────────────────────────────────

def run_shap_dependence_pairs(shap_values, X_shap, feature_cols, shap_ranking):
    """SHAP dependence plots for the top feature pairs."""
    import shap

    print("\n" + "=" * 60)
    print("EXPERIMENT 6.5: SHAP Dependence Pairs")
    print("=" * 60)

    # Test key clinical pairs
    pairs = [
        ("number_inpatient", "discharge_disposition_id"),
        ("number_inpatient", "num_medications"),
        ("time_in_hospital", "number_diagnoses"),
        ("age_numeric", "prior_utilization"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for idx, (feat, interact) in enumerate(pairs):
        ax = axes[idx // 2][idx % 2]
        if feat in feature_cols and interact in feature_cols:
            feat_idx = feature_cols.index(feat)
            plt.sca(ax)
            shap.dependence_plot(
                feat_idx, shap_values, X_shap.values,
                feature_names=feature_cols,
                interaction_index=feature_cols.index(interact),
                ax=ax, show=False,
            )
            cat1, _ = get_domain_label(feat)
            cat2, _ = get_domain_label(interact)
            ax.set_title(f"{feat} x {interact}\n({cat1} x {cat2})", fontsize=10)
            print(f"  Dependence: {feat} x {interact}")
        else:
            ax.text(0.5, 0.5, "Feature not found", transform=ax.transAxes, ha="center")

    plt.suptitle("SHAP Dependence: Key Clinical Feature Pairs", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase6_shap_dependence_pairs.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/phase6_shap_dependence_pairs.png")


# ─── Consolidation ──────────────────────────────────────────────────────────

def consolidate_results(shap_ranking, native_ranking, rho, domain_shap,
                        subgroup_results, subgroup_deltas,
                        lime_results):
    """Save all results to JSON."""
    results = {
        "phase": "6_explainability",
        "author": "Mark Rodrigues",
        "date": "2026-04-04",
        "shap_vs_native_spearman_rho": rho,
        "top_20_shap": [(f, float(v)) for f, v in shap_ranking[:20]],
        "top_20_native": [(f, float(v)) for f, v in native_ranking[:20]],
        "domain_shap_breakdown": {k: float(v) for k, v in domain_shap.items()},
        "subgroup_top_features": {
            name: data["top_10"] for name, data in subgroup_results.items()
        },
        "subgroup_deltas": [(f, float(d)) for f, d, _, _ in subgroup_deltas[:10]],
        "lime_cases": {
            name: {k: v for k, v in data.items() if k != "clinically_sensible"}
            for name, data in lime_results.items()
        },
    }

    out_path = RESULTS_DIR / "phase6_explainability.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved consolidated results to {out_path}")
    return results


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    print("Loading data and model...")
    X_test, y_test, raw_model, calibrator, feature_cols, threshold = load_data_and_model()
    feature_cols = list(feature_cols)
    print(f"Test set: {len(X_test)} samples, {len(feature_cols)} features")

    # Experiment 6.1: SHAP Global
    shap_values, X_shap, shap_ranking, native_ranking, rho, domain_shap = \
        run_shap_global(X_test, raw_model, feature_cols)

    # Experiment 6.2: Subgroup SHAP
    subgroup_results, subgroup_deltas = \
        run_shap_subgroups(X_test, y_test, raw_model, feature_cols, threshold, calibrator)

    # Experiment 6.3: LIME
    lime_results = run_lime_analysis(X_test, y_test, calibrator, feature_cols, threshold)

    # Experiment 6.4: Partial Dependence
    run_partial_dependence(X_test, raw_model, feature_cols, shap_ranking)

    # Experiment 6.5: SHAP Dependence Pairs
    run_shap_dependence_pairs(shap_values, X_shap, feature_cols, shap_ranking)

    # Consolidate
    results = consolidate_results(
        shap_ranking, native_ranking, rho, domain_shap,
        subgroup_results, subgroup_deltas, lime_results,
    )

    # --- Final summary ---
    print("\n" + "=" * 60)
    print("PHASE 6 EXPLAINABILITY SUMMARY")
    print("=" * 60)
    total_domain = sum(domain_shap.values())
    top_domain = max(domain_shap.items(), key=lambda x: x[1])
    print(f"SHAP vs Native rank correlation: rho = {rho:.3f}")
    print(f"Dominant clinical domain: {top_domain[0]} ({top_domain[1]/total_domain*100:.1f}% of total SHAP)")
    print(f"Top SHAP feature: {shap_ranking[0][0]} (|SHAP| = {shap_ranking[0][1]:.4f})")
    print(f"LIME cases analyzed: {len(lime_results)}")
    print(f"Plots saved: 7 PNG + {len(lime_results)} LIME HTML files")


if __name__ == "__main__":
    main()
