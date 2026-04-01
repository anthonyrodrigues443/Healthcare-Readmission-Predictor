# Data

## Source
**Dataset:** Diabetes 130-US Hospitals (1999-2008)
**Origin:** UCI Machine Learning Repository
**URL:** https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
**License:** CC BY 4.0

## Download Instructions
```bash
# Option 1: Direct download
wget -O data/raw/diabetic_data.csv https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip
unzip data/raw/dataset_diabetes.zip -d data/raw/

# Option 2: Via Kaggle
kaggle datasets download -d brandao/diabetes -p data/raw/ --unzip
```

## Dataset Description
- **101,766 hospital encounters** for diabetic patients across 130 US hospitals
- **50 features** per encounter: demographics, diagnoses, medications, lab results
- **Target:** readmission within 30 days (binary classification)
- **Years covered:** 1999–2008

## Key Columns
| Column | Description |
|--------|-------------|
| time_in_hospital | Days of hospital stay (1-14) |
| num_lab_procedures | # lab tests performed |
| num_procedures | # non-lab procedures |
| num_medications | # distinct medications |
| number_outpatient | # outpatient visits in prior year |
| number_emergency | # emergency visits in prior year |
| number_inpatient | # inpatient visits in prior year |
| number_diagnoses | # diagnoses entered |
| diag_1/2/3 | Primary/secondary/tertiary ICD-9 diagnosis codes |
| readmitted | <30, >30, NO |
