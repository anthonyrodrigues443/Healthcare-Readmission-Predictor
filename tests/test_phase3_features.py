import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.feature_engineering import engineer_phase3_features


def test_phase3_features_add_expected_columns():
    df = pd.DataFrame(
        {
            "age": [6, 7],
            "time_in_hospital": [4, 7],
            "num_lab_procedures": [30, 70],
            "num_procedures": [1, 3],
            "num_medications": [8, 14],
            "number_outpatient": [1, 0],
            "number_emergency": [0, 2],
            "number_inpatient": [1, 3],
            "number_diagnoses": [5, 9],
            "admission_type_id": [1, 3],
            "diag_1": [401, 250],
            "diag_2": [428, 786],
            "diag_3": [585, 250],
            "A1Cresult": [1, 0],
        }
    )

    out = engineer_phase3_features(df)

    expected_cols = {
        "acute_utilization_burden",
        "chronic_complexity_index",
        "medication_complexity_index",
        "service_intensity",
        "ed_inpatient_mix",
        "polypharmacy_utilization_overlap",
        "age_comorbidity_pressure",
        "lace_diagnosis_pressure",
        "lab_medication_pressure",
        "outpatient_inpatient_gap",
    }
    assert expected_cols.issubset(out.columns)


def test_phase3_features_do_not_emit_inf():
    df = pd.DataFrame(
        {
            "age": [5],
            "time_in_hospital": [0],
            "num_lab_procedures": [0],
            "num_procedures": [0],
            "num_medications": [0],
            "number_outpatient": [0],
            "number_emergency": [0],
            "number_inpatient": [0],
            "number_diagnoses": [0],
            "admission_type_id": [2],
            "diag_1": [250],
            "diag_2": [250],
            "diag_3": [250],
            "A1Cresult": [0],
        }
    )

    out = engineer_phase3_features(df)
    assert out.replace([float("inf"), float("-inf")], pd.NA).isna().sum().sum() == 0
