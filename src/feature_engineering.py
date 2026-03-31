"""
Domain-driven feature engineering for readmission prediction.

Clinical features engineered based on:
- LACE index (Length of stay, Acuity, Comorbidities, Emergency dept visits)
- Charlson Comorbidity Index (approximated from ICD-9 codes)
- Elixhauser comorbidity groupings
- Polypharmacy risk (>5 medications)
- Prior utilization patterns
- CCS (Clinical Classifications Software) diagnostic groupings
"""
import numpy as np
import pandas as pd

from src.utils import setup_logger

logger = setup_logger(__name__)


CIRCULATORY_ICD9 = set(range(390, 460)) | {785}
RESPIRATORY_ICD9 = set(range(460, 520)) | {786}
DIABETES_ICD9 = set(range(249, 251))
RENAL_ICD9 = set(range(580, 630)) | {788}
NEOPLASM_ICD9 = set(range(140, 240))
MUSCULOSKELETAL_ICD9 = set(range(710, 740))
INJURY_ICD9 = set(range(800, 1000))
MENTAL_ICD9 = set(range(290, 320))
GENITOURINARY_ICD9 = set(range(580, 630))
DIGESTIVE_ICD9 = set(range(520, 580))

CHARLSON_MAP = {
    "MI": (set(range(410, 412)),),
    "CHF": (set(range(428, 429)),),
    "PVD": (set(range(440, 449)),),
    "CVD": (set(range(430, 439)),),
    "DEMENTIA": (set(range(290, 295)),),
    "COPD": (set(range(490, 497)) | set(range(500, 509)),),
    "RHEUMATIC": (set(range(714, 716)) | {725},),
    "PEPTIC_ULCER": (set(range(531, 535)),),
    "MILD_LIVER": ({571},),
    "DIABETES_UNCOMPLICATED": (set(range(250, 251)),),
    "DIABETES_COMPLICATED": ({2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509},),
    "HEMIPLEGIA": (set(range(342, 344)) | {3341},),
    "RENAL": (set(range(582, 588)) | {585, 586, 588},),
    "MALIGNANCY": (set(range(140, 173)) | set(range(174, 209)),),
    "SEVERE_LIVER": ({4560, 4561, 4562, 5722, 5723, 5724, 5728},),
    "METASTATIC": (set(range(196, 200)),),
    "AIDS": ({42},),
}

CHARLSON_WEIGHTS = {
    "MI": 1, "CHF": 1, "PVD": 1, "CVD": 1, "DEMENTIA": 1, "COPD": 1,
    "RHEUMATIC": 1, "PEPTIC_ULCER": 1, "MILD_LIVER": 1,
    "DIABETES_UNCOMPLICATED": 1, "DIABETES_COMPLICATED": 2,
    "HEMIPLEGIA": 2, "RENAL": 2, "MALIGNANCY": 2, "SEVERE_LIVER": 3,
    "METASTATIC": 6, "AIDS": 6,
}


def _icd9_to_int(code_str: str) -> int:
    """Convert ICD-9 string to integer prefix for range lookups."""
    if pd.isna(code_str) or str(code_str).strip() in ("?", "nan", ""):
        return -1
    code = str(code_str).strip().upper()
    if code.startswith("V") or code.startswith("E"):
        return -1
    try:
        return int(float(code.split(".")[0]))
    except (ValueError, AttributeError):
        return -1


def compute_charlson_index(diag_1, diag_2, diag_3) -> int:
    codes = {_icd9_to_int(diag_1), _icd9_to_int(diag_2), _icd9_to_int(diag_3)}
    codes.discard(-1)
    score = 0
    for condition, (code_set,) in CHARLSON_MAP.items():
        if codes & code_set:
            score += CHARLSON_WEIGHTS[condition]
    return score


def classify_primary_diagnosis(diag_str: str) -> str:
    """Map primary ICD-9 to CCS-style category."""
    code = _icd9_to_int(diag_str)
    if code == -1:
        return "other"
    if code in CIRCULATORY_ICD9:
        return "circulatory"
    if code in DIABETES_ICD9:
        return "diabetes"
    if code in RESPIRATORY_ICD9:
        return "respiratory"
    if code in RENAL_ICD9:
        return "renal"
    if code in NEOPLASM_ICD9:
        return "neoplasm"
    if code in MUSCULOSKELETAL_ICD9:
        return "musculoskeletal"
    if code in INJURY_ICD9:
        return "injury"
    if code in MENTAL_ICD9:
        return "mental"
    if code in DIGESTIVE_ICD9:
        return "digestive"
    return "other"


def _age_bucket_to_midpoint(age_str: str) -> float:
    """Convert [70-80) -> 75.0"""
    try:
        low = int(str(age_str).replace("[", "").replace(")", "").replace("(", "").split("-")[0])
        return float(low + 5)
    except (ValueError, AttributeError, IndexError):
        return 65.0


def engineer_features(df: pd.DataFrame, age_col_raw: pd.Series = None) -> pd.DataFrame:
    """
    Add domain-specific features. Expects encoded df but also accepts age as raw Series.
    Because age is label-encoded before we get here, we compute LACE's age component
    from the encoded bucket index (0-9 -> 5, 15, ..., 95 midpoints).
    """
    df = df.copy()

    if "time_in_hospital" in df.columns:
        df["lace_length"] = pd.cut(
            df["time_in_hospital"],
            bins=[-1, 0, 2, 4, 6, 7, 13, 99],
            labels=[0, 1, 2, 3, 4, 5, 7]
        ).astype(float)

    if "admission_type_id" in df.columns:
        df["lace_acuity"] = (df["admission_type_id"] == 1).astype(int) * 3

    diag_cols_present = [c for c in ["diag_1", "diag_2", "diag_3"] if c in df.columns]

    if len(diag_cols_present) == 3:
        df["charlson_score"] = df.apply(
            lambda row: compute_charlson_index(row["diag_1"], row["diag_2"], row["diag_3"]),
            axis=1
        )
    else:
        df["charlson_score"] = 0

    df["lace_comorbidity"] = pd.cut(
        df.get("charlson_score", pd.Series(np.zeros(len(df)))),
        bins=[-1, 0, 1, 2, 3, 99],
        labels=[0, 1, 2, 3, 5]
    ).astype(float)

    if "number_emergency" in df.columns:
        df["lace_ed_visits"] = pd.cut(
            df["number_emergency"],
            bins=[-1, 0, 1, 2, 3, 99],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)

    lace_components = ["lace_length", "lace_acuity", "lace_comorbidity", "lace_ed_visits"]
    available_lace = [c for c in lace_components if c in df.columns]
    if available_lace:
        df["lace_score"] = df[available_lace].sum(axis=1)
        df["lace_high_risk"] = (df["lace_score"] >= 10).astype(int)

    if "num_medications" in df.columns:
        df["polypharmacy"] = (df["num_medications"] > 5).astype(int)
        df["high_polypharmacy"] = (df["num_medications"] > 10).astype(int)
        df["extreme_polypharmacy"] = (df["num_medications"] > 15).astype(int)

    if "number_inpatient" in df.columns:
        df["prior_inpatient_flag"] = (df["number_inpatient"] > 0).astype(int)
        df["high_utilizer"] = (df["number_inpatient"] >= 2).astype(int)

    if "number_outpatient" in df.columns and "number_emergency" in df.columns and "number_inpatient" in df.columns:
        df["total_prior_visits"] = (
            df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
        )
        df["prior_visit_intensity"] = (
            df["number_emergency"] * 3 + df["number_inpatient"] * 2 + df["number_outpatient"]
        )

    if "diag_1" in df.columns:
        df["primary_diag_category"] = df["diag_1"].apply(
            lambda x: classify_primary_diagnosis(str(x))
        )
        diag_dummies = pd.get_dummies(df["primary_diag_category"], prefix="diag_cat")
        df = pd.concat([df, diag_dummies], axis=1)
        df = df.drop(columns=["primary_diag_category"])

    if "num_lab_procedures" in df.columns:
        df["lab_intensity"] = pd.cut(
            df["num_lab_procedures"],
            bins=[-1, 20, 40, 60, 80, 999],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)
        df["high_lab_burden"] = (df["num_lab_procedures"] > 60).astype(int)

    if "number_diagnoses" in df.columns:
        df["complex_patient"] = (df["number_diagnoses"] >= 7).astype(int)

    if "time_in_hospital" in df.columns and "num_medications" in df.columns:
        df["med_per_day"] = df["num_medications"] / df["time_in_hospital"].replace(0, 1)

    if "A1Cresult" in df.columns:
        df["a1c_tested"] = (df["A1Cresult"] != 0).astype(int)

    if "num_procedures" in df.columns and "time_in_hospital" in df.columns:
        df["procedure_density"] = df["num_procedures"] / df["time_in_hospital"].replace(0, 1)

    logger.info(f"Engineered feature set: {df.shape[1]} columns (was before engineering)")
    return df


if __name__ == "__main__":
    from src.data_pipeline import run_pipeline
    X_train, X_test, y_train, y_test = run_pipeline()
    X_train_eng = engineer_features(X_train)
    print(f"Feature count after engineering: {X_train_eng.shape[1]}")
    new_cols = [c for c in X_train_eng.columns if c not in X_train.columns]
    print(f"New clinical features: {new_cols}")
