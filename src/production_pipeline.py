"""Shared helpers for production training, evaluation, and inference."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.data_pipeline import clean_and_engineer, compute_lace_score, download_dataset
from src.phase3_feature_engineering import add_phase3_features, get_feature_sets

RANDOM_STATE = 42
MODEL_TARGET = "readmitted_binary"
TEST_SIZE = 0.2
TRAIN_SIZE_WITHIN_TRAINVAL = 0.75
CALIBRATION_SIZE_WITHIN_HOLDOUT = 0.5
MEDICATION_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]
AGE_MAP = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}


def load_modeling_frame() -> tuple[pd.DataFrame, list[str], str]:
    """Load the engineered modeling frame used by the production model."""
    raw_df = download_dataset()
    df = add_phase3_features(clean_and_engineer(raw_df))
    feature_cols = get_feature_sets(df)["full_83"]
    return df, feature_cols, MODEL_TARGET


def make_production_splits(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = RANDOM_STATE,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Build a leak-resistant 60/10/10/20 split for train/early-stop/cal/test."""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_trainval,
        y_trainval,
        test_size=1 - TRAIN_SIZE_WITHIN_TRAINVAL,
        random_state=random_state,
        stratify=y_trainval,
    )
    X_early, X_cal, y_early, y_cal = train_test_split(
        X_holdout,
        y_holdout,
        test_size=CALIBRATION_SIZE_WITHIN_HOLDOUT,
        random_state=random_state,
        stratify=y_holdout,
    )
    return {
        "train": (X_train, y_train),
        "early_stop": (X_early, y_early),
        "calibration": (X_cal, y_cal),
        "test": (X_test, y_test),
    }


def split_metadata(splits: dict[str, tuple[pd.DataFrame, pd.Series]]) -> dict[str, Any]:
    """Return compact metadata for a split dictionary."""
    return {
        "random_state": RANDOM_STATE,
        "schema": {
            "train": 0.6,
            "early_stop": 0.1,
            "calibration": 0.1,
            "test": 0.2,
        },
        "counts": {
            name: len(values[0]) for name, values in splits.items()
        },
        "positive_rate": {
            name: round(float(values[1].mean()), 6) for name, values in splits.items()
        },
    }


def safe_auc(y_true: pd.Series | np.ndarray, y_prob: np.ndarray) -> float | None:
    """Return ROC AUC when both classes are present."""
    unique = np.unique(np.asarray(y_true))
    if len(unique) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def optimal_f1_threshold(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Find the threshold that maximizes F1 on a calibration slice."""
    from sklearn.metrics import precision_recall_curve

    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-8)
    best_idx = int(np.nanargmax(f1_scores))
    return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5


def compute_lace_proxy(frame: pd.DataFrame) -> pd.Series:
    """Compute LACE from either grouped diagnoses or diagnosis count."""
    if {"diag_1_group", "diag_2_group", "diag_3_group"} & set(frame.columns):
        return compute_lace_score(frame)

    lace = pd.Series(0.0, index=frame.index, dtype=float)
    los = _numeric_series(frame, "time_in_hospital").clip(upper=14)
    lace += np.where(
        los <= 1,
        0,
        np.where(los == 2, 1, np.where(los == 3, 2, np.where(los <= 6, 3, np.where(los <= 13, 4, 7)))),
    )
    lace += np.where(_numeric_series(frame, "admission_type_id") == 1, 3, 0)
    lace += _numeric_series(frame, "number_diagnoses").clip(lower=0, upper=5)
    er = _numeric_series(frame, "number_emergency").clip(lower=0, upper=4)
    lace += np.where(er == 0, 0, np.where(er == 1, 1, np.where(er == 2, 2, np.where(er == 3, 3, 4))))
    return lace


def prepare_model_input(input_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Prepare a single or batch input frame for the production model."""
    if set(feature_cols).issubset(input_df.columns):
        return _align_feature_frame(input_df.copy(), feature_cols)

    if _looks_like_raw_dataset(input_df):
        prepared = add_phase3_features(clean_and_engineer(input_df.copy()))
        return _align_feature_frame(prepared, feature_cols)

    prepared = _engineer_serving_features(input_df.copy())
    return _align_feature_frame(prepared, feature_cols)


def _looks_like_raw_dataset(input_df: pd.DataFrame) -> bool:
    markers = {
        "readmitted",
        "diag_1",
        "diag_2",
        "diag_3",
        "race",
        "gender",
        "max_glu_serum",
        "A1Cresult",
    }
    return bool(markers & set(input_df.columns))


def _engineer_serving_features(frame: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
    ]:
        frame[col] = _numeric_series(frame, col)

    if "age_numeric" not in frame.columns:
        if "age" in frame.columns:
            frame["age_numeric"] = frame["age"].map(AGE_MAP).fillna(0)
        else:
            frame["age_numeric"] = 0
    frame["age_numeric"] = _numeric_series(frame, "age_numeric")

    frame["A1C_high"] = _binary_series(
        frame,
        primary="A1C_high",
        fallback="A1Cresult",
        truthy={1, True, "1", "true", "True", ">7", ">8"},
    )
    frame["A1C_tested"] = _binary_series(
        frame,
        primary="A1C_tested",
        fallback="A1Cresult",
        truthy=lambda value: str(value) not in {"None", "nan", "NaN", ""},
    )
    frame["glucose_high"] = _binary_series(
        frame,
        primary="glucose_high",
        fallback="max_glu_serum",
        truthy={1, True, "1", "true", "True", ">200", ">300"},
    )
    frame["glucose_tested"] = _binary_series(
        frame,
        primary="glucose_tested",
        fallback="max_glu_serum",
        truthy=lambda value: str(value) not in {"None", "nan", "NaN", ""},
    )
    frame["med_changed"] = _binary_series(
        frame,
        primary="med_changed",
        fallback="change",
        truthy={1, True, "1", "true", "True", "Ch"},
    )
    frame["diabetes_med"] = _binary_series(
        frame,
        primary="diabetes_med",
        fallback="diabetesMed",
        truthy={1, True, "1", "true", "True", "Yes"},
    )
    frame["diabetes_primary"] = _binary_series(
        frame,
        primary="diabetes_primary",
        fallback="diag_1_group",
        truthy={1, True, "1", "true", "True", "Diabetes"},
    )

    frame["prior_utilization"] = (
        _numeric_series(frame, "prior_utilization")
        if "prior_utilization" in frame.columns
        else frame["number_outpatient"] + frame["number_emergency"] + frame["number_inpatient"]
    )
    frame["polypharmacy"] = (
        _numeric_series(frame, "polypharmacy")
        if "polypharmacy" in frame.columns
        else (frame["num_medications"] > 5).astype(int)
    )
    frame["n_medications_changed"] = _medication_change_count(frame)
    frame["n_medications_active"] = _medication_active_count(frame)
    frame["lab_procedure_ratio"] = frame["num_lab_procedures"] / (frame["time_in_hospital"] + 1)
    frame["procedure_ratio"] = frame["num_procedures"] / (frame["time_in_hospital"] + 1)
    frame["lace_score"] = (
        _numeric_series(frame, "lace_score")
        if "lace_score" in frame.columns
        else compute_lace_proxy(frame)
    )

    frame["discharge_post_acute"] = (
        _numeric_series(frame, "discharge_post_acute")
        if "discharge_post_acute" in frame.columns
        else frame["discharge_disposition_id"].isin([2, 3, 4, 5, 6, 22, 23, 24]).astype(int)
    )
    frame["discharge_ama_or_psych"] = (
        _numeric_series(frame, "discharge_ama_or_psych")
        if "discharge_ama_or_psych" in frame.columns
        else frame["discharge_disposition_id"].isin([7, 28]).astype(int)
    )
    frame["discharge_home"] = (
        _numeric_series(frame, "discharge_home")
        if "discharge_home" in frame.columns
        else (frame["discharge_disposition_id"] == 1).astype(int)
    )
    frame["admission_emergency"] = (
        _numeric_series(frame, "admission_emergency")
        if "admission_emergency" in frame.columns
        else (frame["admission_type_id"] == 1).astype(int)
    )
    frame["admission_transfer_source"] = (
        _numeric_series(frame, "admission_transfer_source")
        if "admission_transfer_source" in frame.columns
        else frame["admission_source_id"].isin([4, 5, 6, 20, 22, 25]).astype(int)
    )
    frame["admission_ed_source"] = (
        _numeric_series(frame, "admission_ed_source")
        if "admission_ed_source" in frame.columns
        else (frame["admission_source_id"] == 7).astype(int)
    )
    frame["utilization_band"] = (
        _numeric_series(frame, "utilization_band")
        if "utilization_band" in frame.columns
        else pd.cut(frame["prior_utilization"], bins=[-0.1, 0.5, 1.5, 3.5, np.inf], labels=[0, 1, 2, 3]).astype(int)
    )
    frame["acute_prior_load"] = (
        _numeric_series(frame, "acute_prior_load")
        if "acute_prior_load" in frame.columns
        else frame["number_inpatient"] * 2 + frame["number_emergency"]
    )
    frame["meds_per_day"] = (
        _numeric_series(frame, "meds_per_day")
        if "meds_per_day" in frame.columns
        else frame["num_medications"] / (frame["time_in_hospital"] + 1)
    )
    frame["diagnoses_per_day"] = (
        _numeric_series(frame, "diagnoses_per_day")
        if "diagnoses_per_day" in frame.columns
        else frame["number_diagnoses"] / (frame["time_in_hospital"] + 1)
    )
    frame["glycemic_instability"] = (
        _numeric_series(frame, "glycemic_instability")
        if "glycemic_instability" in frame.columns
        else ((frame["A1C_high"] == 1) | (frame["glucose_high"] == 1)).astype(int)
    )
    frame["utilization_x_polypharmacy"] = (
        _numeric_series(frame, "utilization_x_polypharmacy")
        if "utilization_x_polypharmacy" in frame.columns
        else frame["prior_utilization"] * frame["polypharmacy"]
    )
    frame["utilization_x_transition"] = (
        _numeric_series(frame, "utilization_x_transition")
        if "utilization_x_transition" in frame.columns
        else frame["prior_utilization"] * frame["discharge_post_acute"]
    )
    frame["los_x_med_burden"] = (
        _numeric_series(frame, "los_x_med_burden")
        if "los_x_med_burden" in frame.columns
        else frame["time_in_hospital"] * frame["num_medications"]
    )
    frame["instability_x_utilization"] = (
        _numeric_series(frame, "instability_x_utilization")
        if "instability_x_utilization" in frame.columns
        else frame["glycemic_instability"] * frame["prior_utilization"]
    )

    return frame


def _align_feature_frame(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    for col in feature_cols:
        if col not in frame.columns:
            frame[col] = 0
    aligned = frame[feature_cols].copy()
    for col in aligned.columns:
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(0)
    return aligned


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0)


def _binary_series(
    frame: pd.DataFrame,
    primary: str,
    fallback: str | None,
    truthy: set[Any] | None = None,
) -> pd.Series:
    if primary in frame.columns:
        return _numeric_series(frame, primary).clip(lower=0, upper=1).astype(int)

    if fallback is None or fallback not in frame.columns:
        return pd.Series(0, index=frame.index, dtype=int)

    fallback_series = frame[fallback].fillna("None")
    if callable(truthy):
        return fallback_series.map(lambda value: int(bool(truthy(value)))).astype(int)
    return fallback_series.map(lambda value: int(value in truthy)).astype(int)


def _medication_change_count(frame: pd.DataFrame) -> pd.Series:
    med_cols = [col for col in MEDICATION_COLUMNS if col in frame.columns]
    if med_cols:
        return frame[med_cols].apply(
            lambda row: sum(1 for value in row if value in {"Up", "Down"}),
            axis=1,
        )
    if "n_medications_changed" in frame.columns:
        return _numeric_series(frame, "n_medications_changed")
    return _binary_series(frame, primary="med_changed", fallback="change", truthy={1, True, "1", "true", "True", "Ch"})


def _medication_active_count(frame: pd.DataFrame) -> pd.Series:
    med_cols = [col for col in MEDICATION_COLUMNS if col in frame.columns]
    if med_cols:
        return frame[med_cols].apply(
            lambda row: sum(1 for value in row if pd.notna(value) and value != "No"),
            axis=1,
        )
    if "n_medications_active" in frame.columns:
        return _numeric_series(frame, "n_medications_active")
    return _numeric_series(frame, "num_medications").clip(lower=0, upper=len(MEDICATION_COLUMNS))


def json_safe_float(value: float | None) -> float | None:
    """Convert NaN to None so metrics survive JSON serialization cleanly."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)
