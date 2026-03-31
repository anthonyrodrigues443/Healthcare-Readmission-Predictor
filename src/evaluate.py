"""
Evaluation metrics for readmission prediction.
Sensitivity (recall for positive class) is the PRIMARY metric in clinical settings.
A false negative (missing a readmission) is far more costly than a false positive.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score
)
from src.utils import setup_logger

logger = setup_logger(__name__)


def compute_metrics(y_true, y_pred, y_prob=None, model_name: str = "model") -> dict:
    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall_sensitivity": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "specificity": None,
        "auc_roc": None,
        "auc_pr": None,
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["specificity"] = round(tn / (tn + fp) if (tn + fp) > 0 else 0.0, 4)
    metrics["tp"] = int(tp)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tn"] = int(tn)

    if y_prob is not None:
        metrics["auc_roc"] = round(roc_auc_score(y_true, y_prob), 4)
        metrics["auc_pr"] = round(average_precision_score(y_true, y_prob), 4)

    return metrics


def print_metrics(metrics: dict) -> None:
    logger.info(f"\n{'='*55}")
    logger.info(f"  Model: {metrics['model']}")
    logger.info(f"{'='*55}")
    logger.info(f"  Accuracy:            {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Score:            {metrics['f1']:.4f}")
    logger.info(f"  Precision:           {metrics['precision']:.4f}")
    logger.info(f"  Recall/Sensitivity:  {metrics['recall_sensitivity']:.4f}  <-- PRIMARY")
    logger.info(f"  Specificity:         {metrics['specificity']:.4f}")
    if metrics.get("auc_roc"):
        logger.info(f"  AUC-ROC:             {metrics['auc_roc']:.4f}")
    if metrics.get("auc_pr"):
        logger.info(f"  AUC-PR:              {metrics['auc_pr']:.4f}")
    logger.info(f"  Confusion: TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")
