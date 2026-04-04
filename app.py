"""Streamlit UI for Healthcare Readmission Risk Predictor.

A clinical decision-support tool that predicts 30-day hospital readmission
risk for diabetic patients. Provides calibrated risk scores, SHAP-based
explanations, and LACE index comparison.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = Path("models")


# ─── Load model artifacts (cached) ──────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    manifest = json.loads((MODEL_DIR / "training_manifest.json").read_text(encoding="utf-8"))
    calibrator = joblib.load(MODEL_DIR / "calibrator.joblib")
    feature_cols = joblib.load(MODEL_DIR / "feature_columns.joblib")
    raw_model = calibrator.estimator.estimator
    importances = dict(zip(feature_cols, raw_model.get_feature_importance()))
    return calibrator, feature_cols, manifest, importances


def compute_lace(time_in_hospital, admission_type_id, number_diagnoses, number_emergency):
    """Compute LACE index for comparison."""
    los = min(time_in_hospital, 14)
    if los <= 1: l_score = 0
    elif los == 2: l_score = 1
    elif los == 3: l_score = 2
    elif los <= 6: l_score = 3
    elif los <= 13: l_score = 4
    else: l_score = 7

    a_score = 3 if admission_type_id == 1 else 0
    c_score = min(number_diagnoses, 5)
    er = min(number_emergency, 4)
    e_score = er

    return l_score + a_score + c_score + e_score


# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏥 30-Day Hospital Readmission Risk Predictor")
st.markdown(
    "Clinical decision-support tool for diabetic patients. "
    "Enter patient features to get a calibrated readmission risk score with "
    "explainable contributing factors."
)

# ─── Check model availability ────────────────────────────────────────────────

if not (MODEL_DIR / "training_manifest.json").exists():
    st.error(
        "Model artifacts not found. Run `python -m src.train` first to train "
        "and save the model."
    )
    st.stop()

calibrator, feature_cols, manifest, feature_importances = load_artifacts()
threshold = manifest["optimal_threshold"]

# ─── Sidebar: model info ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("Model Information")
    st.metric("Model Type", "CatBoost + Calibration")
    st.metric("Test AUC", f"{manifest['test_metrics']['auc']:.3f}")
    st.metric("Test F1", f"{manifest['test_metrics']['f1']:.3f}")
    st.metric("Brier Score", f"{manifest['test_metrics']['brier']:.4f}")
    st.metric("Inference Speed", f"{manifest['latency_ms_per_sample']:.2f} ms")
    st.metric("Decision Threshold", f"{threshold:.3f}")
    st.divider()
    st.markdown("**Features used:** " + str(manifest["feature_count"]))
    st.markdown("**Training samples:** " + f"{manifest['train_samples']:,}")
    st.divider()
    st.caption(
        "This model predicts 30-day readmission risk for diabetic patients "
        "using the UCI 130-US Hospitals dataset. It is NOT intended for "
        "clinical use without further validation."
    )


# ─── Tabs ────────────────────────────────────────────────────────────────────

tab_predict, tab_batch, tab_about = st.tabs([
    "🩺 Single Patient", "📊 Batch Prediction", "ℹ️ About"
])


# ─── Tab 1: Single Patient ──────────────────────────────────────────────────

with tab_predict:
    st.subheader("Patient Information")

    # --- Example patients for quick testing ---
    examples = {
        "-- Select example --": {},
        "High-risk frequent flyer": {
            "time_in_hospital": 8, "num_lab_procedures": 60, "num_procedures": 3,
            "num_medications": 18, "number_outpatient": 2, "number_emergency": 3,
            "number_inpatient": 4, "number_diagnoses": 9, "age_numeric": 75,
            "admission_type_id": 1, "discharge_disposition_id": 3,
            "admission_source_id": 7, "A1C_high": 1, "A1C_tested": 1,
            "glucose_high": 1, "glucose_tested": 1, "diabetes_primary": 1,
            "med_changed": 1, "diabetes_med": 1,
        },
        "Low-risk first-timer": {
            "time_in_hospital": 2, "num_lab_procedures": 30, "num_procedures": 1,
            "num_medications": 8, "number_outpatient": 0, "number_emergency": 0,
            "number_inpatient": 0, "number_diagnoses": 4, "age_numeric": 45,
            "admission_type_id": 3, "discharge_disposition_id": 1,
            "admission_source_id": 1, "A1C_high": 0, "A1C_tested": 1,
            "glucose_high": 0, "glucose_tested": 1, "diabetes_primary": 0,
            "med_changed": 0, "diabetes_med": 1,
        },
        "Moderate-risk elderly": {
            "time_in_hospital": 5, "num_lab_procedures": 45, "num_procedures": 2,
            "num_medications": 14, "number_outpatient": 1, "number_emergency": 1,
            "number_inpatient": 1, "number_diagnoses": 7, "age_numeric": 85,
            "admission_type_id": 1, "discharge_disposition_id": 6,
            "admission_source_id": 7, "A1C_high": 1, "A1C_tested": 1,
            "glucose_high": 0, "glucose_tested": 0, "diabetes_primary": 1,
            "med_changed": 1, "diabetes_med": 1,
        },
    }

    selected_example = st.selectbox("Load example patient:", list(examples.keys()))
    defaults = examples.get(selected_example, {})

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics & Stay**")
        age_numeric = st.slider("Age", 5, 95, defaults.get("age_numeric", 65), step=5)
        time_in_hospital = st.slider("Length of Stay (days)", 1, 14, defaults.get("time_in_hospital", 3))
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16, defaults.get("number_diagnoses", 6))
        admission_type_id = st.selectbox(
            "Admission Type",
            [1, 2, 3, 5],
            format_func=lambda x: {1: "Emergency", 2: "Urgent", 3: "Elective", 5: "Other"}[x],
            index=[1, 2, 3, 5].index(defaults.get("admission_type_id", 1)),
        )

    with col2:
        st.markdown("**Utilization & Procedures**")
        number_inpatient = st.slider("Prior Inpatient Visits", 0, 10, defaults.get("number_inpatient", 0))
        number_emergency = st.slider("Prior ER Visits", 0, 10, defaults.get("number_emergency", 0))
        number_outpatient = st.slider("Prior Outpatient Visits", 0, 10, defaults.get("number_outpatient", 0))
        num_medications = st.slider("Number of Medications", 1, 40, defaults.get("num_medications", 12))
        num_lab_procedures = st.slider("Lab Procedures", 1, 120, defaults.get("num_lab_procedures", 40))
        num_procedures = st.slider("Other Procedures", 0, 6, defaults.get("num_procedures", 1))

    with col3:
        st.markdown("**Discharge & Diabetes**")
        discharge_disposition_id = st.selectbox(
            "Discharge Disposition",
            [1, 2, 3, 5, 6, 7],
            format_func=lambda x: {
                1: "Home", 2: "Short-term rehab", 3: "Skilled Nursing",
                5: "Long-term care", 6: "Home w/ services", 7: "AMA (Left against advice)"
            }[x],
            index=[1, 2, 3, 5, 6, 7].index(defaults.get("discharge_disposition_id", 1)),
        )
        admission_source_id = st.selectbox(
            "Admission Source",
            [1, 4, 5, 7],
            format_func=lambda x: {1: "Physician Referral", 4: "Transfer from Hospital", 5: "Transfer from SNF", 7: "Emergency Room"}[x],
            index=[1, 4, 5, 7].index(defaults.get("admission_source_id", 1)),
        )
        diabetes_primary = st.checkbox("Diabetes as Primary Diagnosis", value=defaults.get("diabetes_primary", False))
        med_changed = st.checkbox("Medication Changed During Stay", value=defaults.get("med_changed", False))
        diabetes_med = st.checkbox("Diabetic Medication Prescribed", value=defaults.get("diabetes_med", True))
        A1C_high = st.checkbox("A1C > 7%", value=defaults.get("A1C_high", False))
        A1C_tested = st.checkbox("A1C Tested", value=defaults.get("A1C_tested", True))
        glucose_high = st.checkbox("Max Glucose > 200", value=defaults.get("glucose_high", False))
        glucose_tested = st.checkbox("Glucose Tested", value=defaults.get("glucose_tested", True))

    if st.button("🔍 Predict Readmission Risk", type="primary", use_container_width=True):
        # Build derived features matching the training pipeline
        prior_utilization = number_outpatient + number_emergency + number_inpatient
        polypharmacy = 1 if num_medications > 5 else 0

        patient = {
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "number_outpatient": number_outpatient,
            "number_emergency": number_emergency,
            "number_inpatient": number_inpatient,
            "number_diagnoses": number_diagnoses,
            "age_numeric": age_numeric,
            "admission_type_id": admission_type_id,
            "discharge_disposition_id": discharge_disposition_id,
            "admission_source_id": admission_source_id,
            "A1C_high": int(A1C_high),
            "A1C_tested": int(A1C_tested),
            "glucose_high": int(glucose_high),
            "glucose_tested": int(glucose_tested),
            "diabetes_primary": int(diabetes_primary),
            "med_changed": int(med_changed),
            "diabetes_med": int(diabetes_med),
            "prior_utilization": prior_utilization,
            "polypharmacy": polypharmacy,
            "n_medications_changed": 1 if med_changed else 0,
            "n_medications_active": min(num_medications, 5),
            "lab_procedure_ratio": num_lab_procedures / (time_in_hospital + 1),
            "procedure_ratio": num_procedures / (time_in_hospital + 1),
            "lace_score": compute_lace(time_in_hospital, admission_type_id, number_diagnoses, number_emergency),
            # Phase 3 transition features
            "discharge_post_acute": 1 if discharge_disposition_id in [2, 3, 4, 5, 6, 22, 23, 24] else 0,
            "discharge_ama_or_psych": 1 if discharge_disposition_id in [7, 28] else 0,
            "discharge_home": 1 if discharge_disposition_id == 1 else 0,
            "admission_emergency": 1 if admission_type_id == 1 else 0,
            "admission_transfer_source": 1 if admission_source_id in [4, 5, 6, 20, 22, 25] else 0,
            "admission_ed_source": 1 if admission_source_id == 7 else 0,
            "utilization_band": min(3, 0 if prior_utilization == 0 else (1 if prior_utilization == 1 else (2 if prior_utilization <= 3 else 3))),
            "acute_prior_load": number_inpatient * 2 + number_emergency,
            "meds_per_day": num_medications / (time_in_hospital + 1),
            "diagnoses_per_day": number_diagnoses / (time_in_hospital + 1),
            "glycemic_instability": 1 if (A1C_high or glucose_high) else 0,
            "utilization_x_polypharmacy": prior_utilization * polypharmacy,
            "utilization_x_transition": prior_utilization * (1 if discharge_disposition_id in [2, 3, 4, 5, 6, 22, 23, 24] else 0),
            "los_x_med_burden": time_in_hospital * num_medications,
            "instability_x_utilization": (1 if (A1C_high or glucose_high) else 0) * prior_utilization,
        }

        # Build feature vector
        df = pd.DataFrame([patient])
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_cols]

        # Predict
        t0 = perf_counter()
        prob = float(calibrator.predict_proba(df)[:, 1][0])
        latency = (perf_counter() - t0) * 1000

        risk_label = "HIGH RISK" if prob >= threshold else "LOW RISK"
        lace = compute_lace(time_in_hospital, admission_type_id, number_diagnoses, number_emergency)

        # --- Display results ---
        st.divider()

        rcol1, rcol2, rcol3, rcol4 = st.columns(4)
        with rcol1:
            color = "🔴" if prob >= threshold else "🟢"
            st.metric(f"{color} Risk Score", f"{prob:.1%}")
        with rcol2:
            st.metric("Risk Label", risk_label)
        with rcol3:
            st.metric("LACE Score", f"{lace}/19")
            lace_risk = "HIGH" if lace >= 10 else ("MODERATE" if lace >= 5 else "LOW")
            st.caption(f"LACE category: {lace_risk}")
        with rcol4:
            st.metric("Inference Time", f"{latency:.1f} ms")

        # --- Contributing factors ---
        st.subheader("Top Contributing Factors")

        # Rank features by importance, show top 10 with patient values
        sorted_feats = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        top_feats = [(f, imp) for f, imp in sorted_feats if f in df.columns][:10]

        feat_data = []
        for feat, imp in top_feats:
            val = float(df[feat].iloc[0])
            feat_data.append({
                "Feature": feat.replace("_", " ").title(),
                "Patient Value": round(val, 2),
                "Model Importance": round(imp, 1),
            })

        feat_df = pd.DataFrame(feat_data)

        fcol1, fcol2 = st.columns([3, 2])
        with fcol1:
            st.dataframe(feat_df, use_container_width=True, hide_index=True)
        with fcol2:
            fig, ax = plt.subplots(figsize=(5, 4))
            bars = ax.barh(
                [f["Feature"][:25] for f in feat_data][::-1],
                [f["Model Importance"] for f in feat_data][::-1],
                color="#1f77b4",
            )
            ax.set_xlabel("Feature Importance")
            ax.set_title("Top Risk Factors")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # --- Clinical context ---
        st.subheader("Clinical Context")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Patient Profile:**")
            st.markdown(f"- **Age:** {age_numeric} years")
            st.markdown(f"- **Length of Stay:** {time_in_hospital} days")
            st.markdown(f"- **Prior Utilization:** {prior_utilization} visits (inpatient: {number_inpatient}, ER: {number_emergency}, outpatient: {number_outpatient})")
            st.markdown(f"- **Medications:** {num_medications} (polypharmacy: {'Yes' if polypharmacy else 'No'})")
            st.markdown(f"- **Discharge:** {'Post-acute facility' if patient['discharge_post_acute'] else 'Home'}")

        with col_b:
            st.markdown("**Model vs LACE Comparison:**")
            st.markdown(f"- ML model risk: **{prob:.1%}** ({'above' if prob >= threshold else 'below'} threshold)")
            st.markdown(f"- LACE score: **{lace}/19** ({lace_risk} risk)")
            if (prob >= threshold) != (lace >= 10):
                st.warning(
                    "ML model and LACE disagree on risk level. "
                    "The ML model uses 83 features vs LACE's 4 components — "
                    "discrepancies are common for edge cases."
                )
            else:
                st.success("ML model and LACE agree on risk category.")


# ─── Tab 2: Batch Prediction ────────────────────────────────────────────────

with tab_batch:
    st.subheader("Batch Prediction")
    st.markdown(
        "Upload a CSV file with patient data to get readmission risk predictions "
        "for all patients at once."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df_upload = pd.read_csv(uploaded)
        st.write(f"Loaded {len(df_upload)} patients")

        if st.button("Run Batch Prediction"):
            for col in feature_cols:
                if col not in df_upload.columns:
                    df_upload[col] = 0
            X_batch = df_upload[feature_cols]

            t0 = perf_counter()
            probs = calibrator.predict_proba(X_batch)[:, 1]
            batch_ms = (perf_counter() - t0) * 1000

            df_upload["risk_score"] = probs
            df_upload["risk_label"] = np.where(probs >= threshold, "HIGH", "LOW")

            st.metric("Total Time", f"{batch_ms:.0f} ms")
            st.metric("Per-Patient", f"{batch_ms/len(df_upload):.2f} ms")
            st.metric("High Risk", f"{(probs >= threshold).sum()} / {len(df_upload)}")

            st.dataframe(
                df_upload[["risk_score", "risk_label"]].head(50),
                use_container_width=True,
            )

            # Distribution plot
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(probs, bins=50, color="#1f77b4", edgecolor="white", alpha=0.8)
            ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.3f})")
            ax.set_xlabel("Risk Score")
            ax.set_ylabel("Count")
            ax.set_title("Risk Score Distribution")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            csv = df_upload.to_csv(index=False)
            st.download_button("Download Results", csv, "predictions.csv", "text/csv")


# ─── Tab 3: About ───────────────────────────────────────────────────────────

with tab_about:
    st.subheader("About This Model")
    st.markdown("""
    ### Healthcare Readmission Risk Predictor

    This tool predicts the probability of 30-day hospital readmission for
    diabetic patients, trained on the UCI Diabetes 130-US Hospitals dataset
    (101,766 encounters from 130 US hospitals, 1999-2008).

    #### Research Highlights

    | Metric | Value |
    |--------|-------|
    | **Champion Model** | CatBoost with Optuna-tuned hyperparameters |
    | **Calibration** | Isotonic regression (Brier: 0.211 → 0.095) |
    | **Test AUC** | 0.691 |
    | **Features** | 83 (clinical + transition + interaction) |
    | **Key Finding** | `random_strength` accounts for 87.4% of HP importance |

    #### Key Discoveries

    1. **SMOTE destroys performance on EHR data** — F1 drops 0.28 → 0.10.
       Synthetic oversampling creates fractional binary features = unrealistic patients.

    2. **First-timer blind spot** — The model catches 92% of frequent flyers
       (prior inpatient 2+) but only 30% of first-time readmissions (prior util = 0).

    3. **23 clinical features match 68 total** — Domain knowledge compresses
       feature space by 66% with negligible AUC loss (0.645 vs 0.648).

    4. **Calibration matters more than tuning** — Isotonic calibration reduces
       Brier score by 55% (0.211 → 0.095), while Optuna tuning adds only +0.004 AUC.

    #### LACE Index Comparison

    The LACE index is the industry standard for readmission risk:
    - **L**ength of stay (0-7 pts)
    - **A**cuity of admission (0 or 3 pts)
    - **C**omorbidity (0-5 pts)
    - **E**R visits (0-4 pts)

    Our ML model uses LACE as one of 83 features and achieves AUC 0.691
    vs LACE's 0.558 on this dataset.

    #### Limitations

    - Trained on 1999-2008 data; clinical practice has evolved
    - UCI dataset lacks full comorbidity indices (Charlson/Elixhauser)
    - 11.2% readmission rate is below national average (15-20%)
    - NOT validated for clinical decision-making

    #### References

    - Strack et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates
    - van Walraven et al. (2010). Derivation and validation of the LACE index
    - AHRQ (2023). Statistical Brief on 30-Day Hospital Readmissions
    """)
