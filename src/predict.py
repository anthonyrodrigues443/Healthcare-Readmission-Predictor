"""
Inference module — load a trained model and predict readmission risk.
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils import get_project_root, load_config, setup_logger
from src.feature_engineering import engineer_features

logger = setup_logger(__name__)


def load_model(model_name: str = "lr_domain_features"):
    root = get_project_root()
    config = load_config()
    model_path = root / config["results"]["models_path"] / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def predict(X: pd.DataFrame, model_name: str = "lr_domain_features", use_domain_features: bool = True) -> np.ndarray:
    model = load_model(model_name)
    if use_domain_features:
        X = engineer_features(X)
        X = X.fillna(0)
        for col in X.select_dtypes(include="object").columns:
            X[col] = pd.Categorical(X[col]).codes
    probs = model.predict_proba(X)[:, 1]
    return probs


def predict_single(patient_dict: dict, model_name: str = "lr_domain_features") -> dict:
    df = pd.DataFrame([patient_dict])
    prob = predict(df, model_name=model_name)[0]
    risk_level = "HIGH" if prob >= 0.3 else ("MODERATE" if prob >= 0.15 else "LOW")
    return {
        "readmission_probability": round(float(prob), 4),
        "risk_level": risk_level,
        "model": model_name,
    }
