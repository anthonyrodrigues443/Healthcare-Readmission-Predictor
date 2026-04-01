"""
Data loading, cleaning, and preprocessing pipeline.
Healthcare Readmission Predictor — ML-1
"""

import pandas as pd
import numpy as np
from pathlib import Path


RAW_PATH = "data/raw/diabetic_data.csv"

# ICD-9 CCS groupings (clinical classification system)
# Maps ICD-9 ranges to clinical categories used in published research
DIAG_CATEGORIES = {
    "circulatory": (390, 459),
    "respiratory": (460, 519),
    "digestive": (520, 579),
    "diabetes": (250, 251),
    "injury": (800, 999),
    "musculoskeletal": (710, 739),
    "genitourinary": (580, 629),
    "neoplasms": (140, 239),
}


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["?", "None", "Unknown/Invalid"])
    return df


def _icd9_category(code_str) -> str:
    """Map ICD-9 string to CCS category."""
    if pd.isna(code_str):
        return "other"
    code_str = str(code_str).strip()
    # V codes, E codes
    if code_str.startswith("V") or code_str.startswith("E"):
        return "external"
    try:
        code = float(code_str)
        for cat, (lo, hi) in DIAG_CATEGORIES.items():
            if lo <= code <= hi:
                return cat
        return "other"
    except ValueError:
        return "other"


def _compute_lace(df: pd.DataFrame) -> pd.Series:
    """
    LACE Index — the industry-standard 30-day readmission risk tool.
    L = Length of stay (0-7 pts)
    A = Acuity of admission (emergency=3, urgent=1, elective=0)
    C = Comorbidity (Charlson-proxy: number of diagnoses)
    E = Emergency department visits in prior 6 months (0-4 pts)
    """
    # L — Length of stay scoring
    l_score = df["time_in_hospital"].clip(1, 14).apply(
        lambda x: 0 if x == 1 else 1 if x == 2 else 2 if x == 3 else
                  3 if x in [4, 5] else 4 if x in [6, 7, 8, 9] else 5 if x in [10, 11, 12, 13] else 7
    )
    # A — Acuity
    a_score = df["admission_type_id"].apply(
        lambda x: 3 if x == 1 else 1 if x in [2, 7] else 0
    )
    # C — Comorbidity (simplified: number of diagnoses as proxy)
    c_score = df["number_diagnoses"].clip(0, 7).apply(
        lambda x: 0 if x <= 1 else 1 if x == 2 else 2 if x == 3 else
                  3 if x == 4 else 4 if x == 5 else 5 if x == 6 else 6
    )
    # E — ED visits
    e_score = df["number_emergency"].apply(
        lambda x: 0 if x == 0 else 1 if x == 1 else 2 if x == 2 else
                  3 if x in [3, 4] else 4
    )
    return l_score + a_score + c_score + e_score


def _encode_medications(df: pd.DataFrame) -> pd.DataFrame:
    """
    Medications: each drug col has values 'No', 'Steady', 'Up', 'Down'.
    Encode change count (number of meds changed) and whether on insulin.
    """
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide",
        "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
        "tolazamide", "examide", "citoglipton", "insulin",
        "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
        "metformin-rosiglitazone", "metformin-pioglitazone",
    ]
    med_cols = [c for c in med_cols if c in df.columns]

    # Number of medications being changed (Up or Down)
    df["num_meds_changed"] = (df[med_cols]
                               .isin(["Up", "Down"])
                               .sum(axis=1))
    # Whether patient is on insulin and whether it was changed
    if "insulin" in df.columns:
        df["on_insulin"] = (df["insulin"] != "No").astype(int)
        df["insulin_changed"] = df["insulin"].isin(["Up", "Down"]).astype(int)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clinical domain feature engineering.
    Based on published readmission literature (LACE, Elixhauser, HRRP).
    """
    df = df.copy()

    # ---- Target: <30 day readmission ----
    df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)

    # ---- LACE Index (industry baseline) ----
    df["lace_score"] = _compute_lace(df)
    df["lace_high_risk"] = (df["lace_score"] >= 10).astype(int)

    # ---- ICD-9 CCS Category features ----
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[f"{col}_cat"] = df[col].apply(_icd9_category)
    df["has_diabetes_diag"] = (
        (df.get("diag_1_cat") == "diabetes") |
        (df.get("diag_2_cat") == "diabetes") |
        (df.get("diag_3_cat") == "diabetes")
    ).astype(int)
    df["has_circulatory_diag"] = (
        (df.get("diag_1_cat") == "circulatory") |
        (df.get("diag_2_cat") == "circulatory") |
        (df.get("diag_3_cat") == "circulatory")
    ).astype(int)

    # ---- Medication features ----
    df = _encode_medications(df)

    # ---- Polypharmacy risk (>5 medications) ----
    df["polypharmacy"] = (df["num_medications"] > 5).astype(int)

    # ---- Prior utilization (strongest predictor per literature) ----
    df["prior_utilization"] = (
        df["number_outpatient"].fillna(0) +
        df["number_emergency"].fillna(0) +
        df["number_inpatient"].fillna(0)
    )
    df["had_prior_inpatient"] = (df["number_inpatient"].fillna(0) > 0).astype(int)
    df["had_prior_emergency"] = (df["number_emergency"].fillna(0) > 0).astype(int)

    # ---- HbA1c result (glucose control) ----
    if "A1Cresult" in df.columns:
        df["a1c_measured"] = (df["A1Cresult"] != "None").astype(int)
        df["a1c_high"] = df["A1Cresult"].isin([">7", ">8"]).astype(int)

    # ---- Age: convert age bucket to midpoint ----
    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95
    }
    df["age_numeric"] = df["age"].map(age_map)
    df["elderly"] = (df["age_numeric"] >= 70).astype(int)

    # ---- Gender encode ----
    df["gender_male"] = (df["gender"] == "Male").astype(int)

    # ---- Discharge disposition: home vs facility ----
    if "discharge_disposition_id" in df.columns:
        # Disposition 1 = Home, 2 = Home w/ services, 3 = SNF, etc.
        df["discharged_home"] = df["discharge_disposition_id"].isin([1, 2]).astype(int)
        # Disposition 11 = Expired — exclude from analysis
        df = df[df["discharge_disposition_id"] != 11]
        # Dispositions 13, 14 = Hospice — exclude
        df = df[~df["discharge_disposition_id"].isin([13, 14])]

    return df


def get_feature_matrix(df: pd.DataFrame):
    """Return X, y with selected features."""
    feature_cols = [
        # LACE components
        "time_in_hospital", "number_emergency", "number_diagnoses", "lace_score",
        "lace_high_risk",
        # Utilization
        "number_outpatient", "number_inpatient", "prior_utilization",
        "had_prior_inpatient", "had_prior_emergency",
        # Clinical
        "num_lab_procedures", "num_procedures", "num_medications",
        "num_meds_changed", "polypharmacy",
        # Medication-specific
        "on_insulin", "insulin_changed",
        # Diagnoses
        "has_diabetes_diag", "has_circulatory_diag",
        # A1C
        "a1c_measured", "a1c_high",
        # Demographics
        "age_numeric", "elderly", "gender_male",
        # Admission type (emergency=1, urgent=2, elective=3)
        "admission_type_id",
        # Discharge
        "discharged_home",
    ]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0)
    y = df["readmitted_binary"]
    return X, y
