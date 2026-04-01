"""Data pipeline for Healthcare Readmission Predictor.

Downloads UCI Diabetes 130-US Hospitals dataset, cleans and engineers features
based on clinical literature (LACE index, Elixhauser comorbidities, etc.)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from urllib.request import urlopen
import io
import zipfile


# ICD-9 groupings based on CCS (Clinical Classifications Software)
ICD9_GROUPS = {
    'Circulatory': range(390, 460),
    'Respiratory': range(460, 520),
    'Digestive': range(520, 580),
    'Diabetes': range(250, 251),
    'Injury': range(800, 1000),
    'Musculoskeletal': range(710, 740),
    'Genitourinary': range(580, 630),
    'Neoplasms': range(140, 240),
    'Mental': range(290, 320),
}


def classify_icd9(code):
    """Map ICD-9 code to clinical category."""
    if pd.isna(code) or code == '?':
        return 'Unknown'
    try:
        code_str = str(code).strip()
        if code_str.startswith('V'):
            return 'Supplementary'
        if code_str.startswith('E'):
            return 'External'
        num = int(float(code_str.split('.')[0]))
        for group, rng in ICD9_GROUPS.items():
            if num in rng:
                return group
        return 'Other'
    except (ValueError, IndexError):
        return 'Unknown'


def download_dataset(raw_path: str = 'data/raw/diabetic_data.csv') -> pd.DataFrame:
    """Download UCI Diabetes 130-US Hospitals dataset."""
    raw_path = Path(raw_path)
    if raw_path.exists():
        print(f"Dataset already exists at {raw_path}")
        return pd.read_csv(raw_path)

    print("Downloading UCI Diabetes 130-US Hospitals dataset...")
    dataset_url = (
        "https://archive.ics.uci.edu/static/public/296/"
        "diabetes+130-us+hospitals+for+years+1999-2008.zip"
    )
    with urlopen(dataset_url, timeout=120) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        with archive.open("diabetic_data.csv") as dataset_file:
            df = pd.read_csv(dataset_file)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
    print(f"Saved {len(df)} records to {raw_path}")
    return df


def compute_lace_score(df: pd.DataFrame) -> pd.Series:
    """Compute LACE index: Length of stay + Acuity + Comorbidities + ER visits.

    LACE is the industry-standard tool hospitals use for readmission risk.
    - L: Length of stay (0-7 points)
    - A: Acuity of admission (0 or 3 points for emergency)
    - C: Charlson Comorbidity Index proxy (0-5 points based on # diagnoses)
    - E: ER visits in past 6 months (0-4 points)
    """
    lace = pd.Series(0, index=df.index, dtype=float)

    # L: Length of stay scoring
    los = df['time_in_hospital'].clip(upper=14)
    lace += np.where(los <= 1, 0,
            np.where(los == 2, 1,
            np.where(los == 3, 2,
            np.where(los <= 6, 3,
            np.where(los <= 13, 4, 7)))))

    # A: Acuity — emergency admission = 3 points
    if 'admission_type_id' in df.columns:
        lace += np.where(df['admission_type_id'] == 1, 3, 0)  # 1 = Emergency

    # C: Comorbidity proxy — count of distinct diagnosis categories
    diag_cols = ['diag_1_group', 'diag_2_group', 'diag_3_group']
    existing_diag_cols = [c for c in diag_cols if c in df.columns]
    if existing_diag_cols:
        n_diag_groups = df[existing_diag_cols].nunique(axis=1)
        lace += n_diag_groups.clip(upper=5)

    # E: ER visits (number_emergency in past year as proxy)
    if 'number_emergency' in df.columns:
        er = df['number_emergency'].clip(upper=4)
        lace += np.where(er == 0, 0,
                np.where(er == 1, 1,
                np.where(er == 2, 2,
                np.where(er == 3, 3, 4))))

    return lace


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data and engineer clinically-motivated features."""
    df = df.copy()

    # Drop columns with >40% missing or near-zero variance
    drop_cols = ['weight', 'payer_code', 'medical_specialty', 'encounter_id', 'patient_nbr']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Target: binary readmission (<30 days = 1, else = 0)
    if 'readmitted' in df.columns:
        df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)
        df.drop(columns=['readmitted'], inplace=True)

    # ICD-9 diagnosis grouping
    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in df.columns:
            df[f'{col}_group'] = df[col].apply(classify_icd9)
            df.drop(columns=[col], inplace=True)

    # Clinical feature engineering
    # 1. Polypharmacy: number of medications changed (clinical risk factor)
    med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                'miglitol', 'troglitazone', 'tolazamide', 'insulin',
                'glyburide-metformin', 'glipizide-metformin',
                'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                'metformin-pioglitazone']
    existing_med_cols = [c for c in med_cols if c in df.columns]
    df['n_medications_changed'] = df[existing_med_cols].apply(
        lambda row: sum(1 for v in row if v in ['Up', 'Down']), axis=1
    )
    df['n_medications_active'] = df[existing_med_cols].apply(
        lambda row: sum(1 for v in row if v != 'No'), axis=1
    )
    df['polypharmacy'] = (df['num_medications'] > 5).astype(int)

    # 2. Prior utilization intensity
    df['prior_utilization'] = (
        df.get('number_outpatient', 0) +
        df.get('number_emergency', 0) +
        df.get('number_inpatient', 0)
    )

    # 3. Lab work intensity
    df['lab_procedure_ratio'] = df['num_lab_procedures'] / (df['time_in_hospital'] + 1)

    # 4. Procedure intensity
    df['procedure_ratio'] = df['num_procedures'] / (df['time_in_hospital'] + 1)

    # 5. Age encoding (midpoint of ranges)
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    if 'age' in df.columns:
        df['age_numeric'] = df['age'].map(age_map)
        df.drop(columns=['age'], inplace=True)

    # 6. A1C result encoding
    if 'A1Cresult' in df.columns:
        df['A1C_high'] = (df['A1Cresult'].isin(['>7', '>8'])).astype(int)
        df['A1C_tested'] = (df['A1Cresult'] != 'None').astype(int)
        df.drop(columns=['A1Cresult'], inplace=True)

    # 7. Max glucose encoding
    if 'max_glu_serum' in df.columns:
        df['glucose_high'] = (df['max_glu_serum'].isin(['>200', '>300'])).astype(int)
        df['glucose_tested'] = (df['max_glu_serum'] != 'None').astype(int)
        df.drop(columns=['max_glu_serum'], inplace=True)

    # 8. Diabetes as primary diagnosis
    df['diabetes_primary'] = (df.get('diag_1_group', '') == 'Diabetes').astype(int)

    # 9. Change in medication flag
    if 'change' in df.columns:
        df['med_changed'] = (df['change'] == 'Ch').astype(int)
        df.drop(columns=['change'], inplace=True)

    # 10. Diabetic medication prescribed flag
    if 'diabetesMed' in df.columns:
        df['diabetes_med'] = (df['diabetesMed'] == 'Yes').astype(int)
        df.drop(columns=['diabetesMed'], inplace=True)

    # LACE score
    df['lace_score'] = compute_lace_score(df)

    # Encode remaining categoricals
    # Drop individual medication columns (captured by aggregates above)
    df.drop(columns=[c for c in existing_med_cols if c in df.columns], inplace=True)

    # One-hot encode remaining categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'readmitted_binary' in cat_cols:
        cat_cols.remove('readmitted_binary')

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # Drop any remaining NaN rows
    df.dropna(inplace=True)

    return df


def prepare_data(df: pd.DataFrame, target: str = 'readmitted_binary', test_size: float = 0.2):
    """Split into train/test with stratification."""
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Target distribution - Train: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test


def load_raw(raw_path: str = 'data/raw/diabetic_data.csv') -> pd.DataFrame:
    """Compatibility wrapper for legacy scripts."""
    return download_dataset(raw_path=raw_path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper for legacy scripts."""
    return clean_and_engineer(df)


def get_feature_matrix(df: pd.DataFrame, target: str = 'readmitted_binary'):
    """Return model matrix and target vector."""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


if __name__ == '__main__':
    df = download_dataset()
    print(f"\nRaw dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target distribution:\n{df['readmitted'].value_counts(normalize=True)}")

    df_clean = clean_and_engineer(df)
    print(f"\nCleaned dataset: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    print(f"Target distribution:\n{df_clean['readmitted_binary'].value_counts(normalize=True)}")

    df_clean.to_csv('data/processed/readmission_processed.csv', index=False)
    print("\nSaved processed data to data/processed/readmission_processed.csv")

    X_train, X_test, y_train, y_test = prepare_data(df_clean)
