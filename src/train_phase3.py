"""
Phase 3: Feature Engineering Deep Dive + Threshold Optimization
Healthcare Readmission Predictor

Research question: Is the performance bottleneck the MODEL or the FEATURES?

Phase 2 found:
  - CatBoost champion: sensitivity=0.447, F1=0.254 (with domain features)
  - LightGBM: sensitivity=0.375, domain features helped +0.012
  - Random Forest: collapsed to 0.098 — ensemble averaging kills minority recall
  - Decision Tree: sensitivity=0.566 but precision=0.159 (84% false alarms)

Phase 3 experiments:
  3.1  Threshold optimization on Phase 2 champions — can we hit sensitivity>=0.70?
  3.2  BalancedRandomForest (imbalanced-learn) — fix RF collapse
  3.3  Feature ablation on CatBoost: LACE vs Charlson vs polypharmacy vs CCS vs utilization
  3.4  Interaction features: LACE×Charlson, polypharmacy×inpatient, med_per_day×complex
  3.5  K-fold CV (5-fold stratified) on Phase 2 champion vs Phase 3 best

PRIMARY metric: Sensitivity (clinical requirement). SECONDARY: F1, Precision, AUC-PR.
Clinical target: sensitivity >= 0.70 at precision >= 0.15 (hospital alert management threshold).
"""
import os
import sys
import json
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score
)
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import get_project_root, load_config, setup_logger
from src.data_pipeline import run_pipeline
from src.feature_engineering import engineer_features
from src.evaluate import compute_metrics, print_metrics

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

logger = setup_logger(__name__)
RANDOM_STATE = 42
ROOT = get_project_root()
PLOTS_DIR = ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prep_domain_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    X_tr = engineer_features(X_train).fillna(0)
    X_te = engineer_features(X_test).fillna(0)
    for col in X_tr.select_dtypes("object").columns:
        X_tr[col] = pd.Categorical(X_tr[col]).codes
    for col in X_te.select_dtypes("object").columns:
        X_te[col] = pd.Categorical(X_te[col]).codes
    shared = [c for c in X_tr.columns if c in X_te.columns]
    return X_tr[shared], X_te[shared]


def add_interaction_features(X: pd.DataFrame) -> pd.DataFrame:
    """New Phase 3 feature set: interaction terms."""
    X = X.copy()
    # Interaction 1: Clinical severity × comorbidity burden
    if "lace_score" in X.columns and "charlson_score" in X.columns:
        X["lace_x_charlson"] = X["lace_score"] * X["charlson_score"]

    # Interaction 2: Polypharmacy × prior inpatient (high-risk overlap)
    if "num_medications" in X.columns and "number_inpatient" in X.columns:
        X["polypharm_x_inpatient"] = X["num_medications"] * (X["number_inpatient"] + 1)

    # Interaction 3: Medication burden per day × diagnosis complexity
    if "med_per_day" in X.columns and "number_diagnoses" in X.columns:
        X["medperday_x_diag"] = X["med_per_day"] * X["number_diagnoses"]

    # Interaction 4: Length of stay × number of procedures (intensity score)
    if "time_in_hospital" in X.columns and "num_procedures" in X.columns:
        X["los_x_procedures"] = X["time_in_hospital"] * (X["num_procedures"] + 1)

    # Interaction 5: Emergency visits × lab burden (acute complexity)
    if "number_emergency" in X.columns and "num_lab_procedures" in X.columns:
        X["ed_x_labs"] = X["number_emergency"] * X["num_lab_procedures"]

    # Interaction 6: High utilizer + polypharmacy (double high-risk flag)
    if "high_utilizer" in X.columns and "high_polypharmacy" in X.columns:
        X["high_util_and_polypharm"] = X["high_utilizer"] * X["high_polypharmacy"]

    return X


def threshold_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    return dict(threshold=threshold, sensitivity=sens, specificity=spec,
                precision=prec, f1=f1, tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def find_clinical_threshold(y_true, y_prob, target_sensitivity=0.70):
    """Find the threshold giving highest precision at sensitivity >= target."""
    thresholds = np.linspace(0.0, 1.0, 201)
    best = None
    for t in thresholds:
        m = threshold_metrics(y_true, y_prob, t)
        if m["sensitivity"] >= target_sensitivity:
            if best is None or m["precision"] > best["precision"]:
                best = m
    return best


# ---------------------------------------------------------------------------
# Experiment 3.1: Threshold Optimization on Phase 2 Champions
# ---------------------------------------------------------------------------

def experiment_3_1(X_train, y_train, X_test, y_test,
                   X_train_dom, X_test_dom):
    print("\n" + "="*70)
    print("EXPERIMENT 3.1: Threshold Optimization on Phase 2 Champions")
    print("="*70)

    results = {}

    models = {
        "CatBoost_domain": CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            auto_class_weights="Balanced", random_seed=RANDOM_STATE,
            verbose=0
        ),
        "LightGBM_domain": LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            class_weight="balanced", random_state=RANDOM_STATE,
            verbose=-1
        ),
    }
    data_map = {
        "CatBoost_domain": (X_train_dom, X_test_dom),
        "LightGBM_domain": (X_train_dom, X_test_dom),
    }

    for name, model in models.items():
        Xtr, Xte = data_map[name]
        t0 = time.time()
        model.fit(Xtr, y_train)
        train_sec = time.time() - t0

        y_prob = model.predict_proba(Xte)[:, 1]
        auc_roc = roc_auc_score(y_test, y_prob)
        auc_pr = average_precision_score(y_test, y_prob)

        # Default threshold (0.5)
        default_m = threshold_metrics(y_test, y_prob, 0.50)

        # Clinical target: sensitivity >= 0.70
        clinical_m = find_clinical_threshold(y_test, y_prob, target_sensitivity=0.70)

        # Optimal F1 threshold
        prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_test, y_prob)
        f1_arr = 2 * prec_arr * rec_arr / np.where(prec_arr + rec_arr > 0, prec_arr + rec_arr, 1)
        best_f1_idx = np.argmax(f1_arr[:-1])
        optimal_f1_t = float(thresh_arr[best_f1_idx]) if len(thresh_arr) > best_f1_idx else 0.5
        opt_f1_m = threshold_metrics(y_test, y_prob, optimal_f1_t)

        results[name] = {
            "train_sec": round(train_sec, 2),
            "auc_roc": round(auc_roc, 4),
            "auc_pr": round(auc_pr, 4),
            "default_threshold": default_m,
            "clinical_threshold_070": clinical_m,
            "optimal_f1_threshold": opt_f1_m,
        }

        print(f"\n{name}")
        print(f"  AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  Train={train_sec:.1f}s")
        print(f"  Default t=0.50:  sens={default_m['sensitivity']:.3f}  prec={default_m['precision']:.3f}  "
              f"f1={default_m['f1']:.3f}  spec={default_m['specificity']:.3f}")
        if clinical_m:
            print(f"  Clinical t={clinical_m['threshold']:.2f}: sens={clinical_m['sensitivity']:.3f}  "
                  f"prec={clinical_m['precision']:.3f}  f1={clinical_m['f1']:.3f}  "
                  f"spec={clinical_m['specificity']:.3f}")
        else:
            print(f"  Clinical t=0.70: TARGET NOT ACHIEVABLE")
        print(f"  Optimal F1 t={optimal_f1_t:.2f}: sens={opt_f1_m['sensitivity']:.3f}  "
              f"prec={opt_f1_m['precision']:.3f}  f1={opt_f1_m['f1']:.3f}")

    # Plot PR curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 3.1 — Precision-Recall Curves (CatBoost vs LightGBM)", fontsize=13)
    colors = ["#2196F3", "#FF9800"]
    for ax, (name, color) in zip(axes, zip(models.keys(), colors)):
        Xtr, Xte = data_map[name]
        model = models[name]
        y_prob = model.predict_proba(Xte)[:, 1]
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(rec_arr, prec_arr, color=color, lw=2, label=f"AP={ap:.3f}")
        ax.axhline(0.15, color="red", linestyle="--", alpha=0.5, label="Prec=0.15 (min hospital)")
        ax.axvline(0.70, color="green", linestyle="--", alpha=0.5, label="Sens=0.70 (clinical target)")
        ax.set_xlabel("Recall (Sensitivity)")
        ax.set_ylabel("Precision")
        ax.set_title(name)
        ax.legend(fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "phase3_pr_curves.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  [Saved] results/plots/phase3_pr_curves.png")

    return results


# ---------------------------------------------------------------------------
# Experiment 3.2: BalancedRandomForest (fix RF collapse)
# ---------------------------------------------------------------------------

def experiment_3_2(X_train_dom, y_train, X_test_dom, y_test):
    print("\n" + "="*70)
    print("EXPERIMENT 3.2: BalancedRandomForest — Fixing RF Collapse")
    print("="*70)

    results = {}

    # Vanilla RF (Phase 2 baseline for comparison)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=10,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
    )
    t0 = time.time()
    rf.fit(X_train_dom, y_train)
    t_rf = time.time() - t0
    y_prob_rf = rf.predict_proba(X_test_dom)[:, 1]
    m_rf_default = threshold_metrics(y_test, y_prob_rf, 0.5)
    m_rf_clinical = find_clinical_threshold(y_test, y_prob_rf, 0.70)
    print(f"\nVanilla RF (baseline): sens={m_rf_default['sensitivity']:.3f}  "
          f"prec={m_rf_default['precision']:.3f}  f1={m_rf_default['f1']:.3f}  "
          f"AUC-ROC={roc_auc_score(y_test, y_prob_rf):.4f}")
    results["VanillaRF"] = {
        "sensitivity": m_rf_default["sensitivity"],
        "precision": m_rf_default["precision"],
        "f1": m_rf_default["f1"],
        "auc_roc": round(roc_auc_score(y_test, y_prob_rf), 4),
        "auc_pr": round(average_precision_score(y_test, y_prob_rf), 4),
        "clinical_threshold_070": m_rf_clinical,
        "train_sec": round(t_rf, 2),
    }

    if HAS_IMBLEARN:
        # BalancedRandomForest: uses balanced bootstrap (undersamples majority each tree)
        brf = BalancedRandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_STATE
        )
        t0 = time.time()
        brf.fit(X_train_dom, y_train)
        t_brf = time.time() - t0
        y_prob_brf = brf.predict_proba(X_test_dom)[:, 1]
        m_brf_default = threshold_metrics(y_test, y_prob_brf, 0.5)
        m_brf_clinical = find_clinical_threshold(y_test, y_prob_brf, 0.70)

        print(f"\nBalancedRandomForest: sens={m_brf_default['sensitivity']:.3f}  "
              f"prec={m_brf_default['precision']:.3f}  f1={m_brf_default['f1']:.3f}  "
              f"AUC-ROC={roc_auc_score(y_test, y_prob_brf):.4f}")
        if m_brf_clinical:
            print(f"  → Clinical target (sens>=0.70): t={m_brf_clinical['threshold']:.2f}  "
                  f"prec={m_brf_clinical['precision']:.3f}  f1={m_brf_clinical['f1']:.3f}")
        results["BalancedRF"] = {
            "sensitivity": m_brf_default["sensitivity"],
            "precision": m_brf_default["precision"],
            "f1": m_brf_default["f1"],
            "auc_roc": round(roc_auc_score(y_test, y_prob_brf), 4),
            "auc_pr": round(average_precision_score(y_test, y_prob_brf), 4),
            "clinical_threshold_070": m_brf_clinical,
            "train_sec": round(t_brf, 2),
        }

        delta_sens = m_brf_default["sensitivity"] - m_rf_default["sensitivity"]
        print(f"\n  ΔSensitivity (BRF - RF): {delta_sens:+.3f}")
    else:
        print("  [SKIP] imbalanced-learn not available")

    return results


# ---------------------------------------------------------------------------
# Experiment 3.3: Feature Ablation on CatBoost
# ---------------------------------------------------------------------------

def experiment_3_3(X_train_dom: pd.DataFrame, y_train,
                   X_test_dom: pd.DataFrame, y_test):
    print("\n" + "="*70)
    print("EXPERIMENT 3.3: Feature Ablation on CatBoost (domain feature groups)")
    print("="*70)

    # Define feature groups
    feature_groups = {
        "LACE_components": ["lace_length", "lace_acuity", "lace_comorbidity",
                            "lace_ed_visits", "lace_score", "lace_high_risk"],
        "Charlson_index": ["charlson_score"],
        "Polypharmacy": ["polypharmacy", "high_polypharmacy", "extreme_polypharmacy",
                         "med_per_day"],
        "Prior_utilization": ["prior_inpatient_flag", "high_utilizer",
                              "total_prior_visits", "prior_visit_intensity"],
        "CCS_diagnosis": [c for c in X_train_dom.columns if c.startswith("diag_cat_")],
        "Lab_complexity": ["lab_intensity", "high_lab_burden", "complex_patient",
                           "procedure_density"],
    }

    # Full model (all domain features) — baseline for ablation
    def train_catboost(X_tr, X_te):
        m = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            auto_class_weights="Balanced", random_seed=RANDOM_STATE, verbose=0
        )
        m.fit(X_tr, y_train)
        y_prob = m.predict_proba(X_te)[:, 1]
        sens = threshold_metrics(y_test, y_prob, 0.5)["sensitivity"]
        f1 = threshold_metrics(y_test, y_prob, 0.5)["f1"]
        auc = roc_auc_score(y_test, y_prob)
        return sens, f1, auc

    print("\nTraining full CatBoost (all domain features) ...")
    full_sens, full_f1, full_auc = train_catboost(X_train_dom, X_test_dom)
    print(f"  Full model: sens={full_sens:.3f}  f1={full_f1:.3f}  AUC={full_auc:.4f}")

    results = {"full_model": {"sensitivity": full_sens, "f1": full_f1, "auc_roc": full_auc}}

    # Remove one group at a time
    ablation_results = []
    for group_name, group_cols in feature_groups.items():
        present_cols = [c for c in group_cols if c in X_train_dom.columns]
        if not present_cols:
            continue
        X_tr_ablated = X_train_dom.drop(columns=present_cols, errors="ignore")
        X_te_ablated = X_test_dom.drop(columns=present_cols, errors="ignore")
        sens, f1, auc = train_catboost(X_tr_ablated, X_te_ablated)
        delta_sens = sens - full_sens
        delta_f1 = f1 - full_f1
        print(f"  Remove {group_name}: sens={sens:.3f} ({delta_sens:+.3f})  "
              f"f1={f1:.3f} ({delta_f1:+.3f})  AUC={auc:.4f}")
        ablation_results.append({
            "removed_group": group_name,
            "n_features_removed": len(present_cols),
            "sensitivity": round(sens, 4),
            "delta_sensitivity": round(delta_sens, 4),
            "f1": round(f1, 4),
            "delta_f1": round(delta_f1, 4),
            "auc_roc": round(auc, 4),
        })
        results[f"no_{group_name}"] = ablation_results[-1]

    # Raw features only (no domain engineering)
    raw_cols = [c for c in X_train_dom.columns
                if c not in sum(feature_groups.values(), [])]
    X_tr_raw = X_train_dom[raw_cols]
    X_te_raw = X_test_dom[raw_cols]
    sens_raw, f1_raw, auc_raw = train_catboost(X_tr_raw, X_te_raw)
    delta_sens_raw = sens_raw - full_sens
    print(f"  Raw features only: sens={sens_raw:.3f} ({delta_sens_raw:+.3f})  "
          f"f1={f1_raw:.3f}  AUC={auc_raw:.4f}")
    ablation_results.append({
        "removed_group": "ALL_domain_features",
        "n_features_removed": X_train_dom.shape[1] - len(raw_cols),
        "sensitivity": round(sens_raw, 4),
        "delta_sensitivity": round(delta_sens_raw, 4),
        "f1": round(f1_raw, 4),
        "delta_f1": round(f1_raw - full_f1, 4),
        "auc_roc": round(auc_raw, 4),
    })
    results["raw_features_only"] = ablation_results[-1]

    # Plot ablation waterfall
    sorted_res = sorted(ablation_results, key=lambda x: x["delta_sensitivity"])
    labels = [r["removed_group"].replace("_", "\n") for r in sorted_res]
    deltas = [r["delta_sensitivity"] for r in sorted_res]
    colors = ["#F44336" if d > 0 else "#4CAF50" for d in deltas]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(labels, deltas, color=colors, edgecolor="white", height=0.7)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("ΔSensitivity vs Full Model (positive = HURTS when removed)")
    ax.set_title("Phase 3.3 — CatBoost Feature Ablation\n(Which domain feature group drives performance?)")
    for bar, val in zip(bars, deltas):
        ax.text(val + (0.001 if val >= 0 else -0.001),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "phase3_ablation_catboost.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  [Saved] results/plots/phase3_ablation_catboost.png")

    return results, ablation_results


# ---------------------------------------------------------------------------
# Experiment 3.4: Interaction Features
# ---------------------------------------------------------------------------

def experiment_3_4(X_train_dom, y_train, X_test_dom, y_test):
    print("\n" + "="*70)
    print("EXPERIMENT 3.4: Interaction Features (clinical compound signals)")
    print("="*70)

    results = {}

    configs = {
        "domain_only": (X_train_dom, X_test_dom),
    }

    X_tr_int = add_interaction_features(X_train_dom)
    X_te_int = add_interaction_features(X_test_dom)
    for col in X_tr_int.select_dtypes("object").columns:
        X_tr_int[col] = pd.Categorical(X_tr_int[col]).codes
    for col in X_te_int.select_dtypes("object").columns:
        X_te_int[col] = pd.Categorical(X_te_int[col]).codes
    shared = [c for c in X_tr_int.columns if c in X_te_int.columns]
    X_tr_int, X_te_int = X_tr_int[shared], X_te_int[shared]
    configs["domain_plus_interactions"] = (X_tr_int, X_te_int)

    new_features = [c for c in X_tr_int.columns if c not in X_train_dom.columns]
    print(f"\nNew interaction features ({len(new_features)}): {new_features}")

    for config_name, (Xtr, Xte) in configs.items():
        for model_name, model_cls, model_kwargs in [
            ("CatBoost", CatBoostClassifier, dict(
                iterations=300, depth=6, learning_rate=0.05,
                auto_class_weights="Balanced", random_seed=RANDOM_STATE, verbose=0
            )),
            ("LightGBM", LGBMClassifier, dict(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                class_weight="balanced", random_state=RANDOM_STATE, verbose=-1
            )),
        ]:
            model = model_cls(**model_kwargs)
            t0 = time.time()
            model.fit(Xtr, y_train)
            train_sec = time.time() - t0
            y_prob = model.predict_proba(Xte)[:, 1]
            m = threshold_metrics(y_test, y_prob, 0.5)
            auc = roc_auc_score(y_test, y_prob)
            ap = average_precision_score(y_test, y_prob)
            key = f"{model_name}_{config_name}"
            results[key] = {
                "sensitivity": round(m["sensitivity"], 4),
                "precision": round(m["precision"], 4),
                "f1": round(m["f1"], 4),
                "auc_roc": round(auc, 4),
                "auc_pr": round(ap, 4),
                "n_features": Xtr.shape[1],
                "train_sec": round(train_sec, 2),
            }
            print(f"  {key}: sens={m['sensitivity']:.3f}  f1={m['f1']:.3f}  "
                  f"AUC={auc:.4f}  features={Xtr.shape[1]}")

    # Delta analysis
    for mname in ["CatBoost", "LightGBM"]:
        base = results.get(f"{mname}_domain_only", {})
        inter = results.get(f"{mname}_domain_plus_interactions", {})
        if base and inter:
            ds = inter["sensitivity"] - base["sensitivity"]
            df1 = inter["f1"] - base["f1"]
            print(f"\n  {mname} interaction effect: ΔSens={ds:+.3f}  ΔF1={df1:+.3f}")

    return results


# ---------------------------------------------------------------------------
# Experiment 3.5: K-Fold Cross Validation on Champion
# ---------------------------------------------------------------------------

def experiment_3_5(X_train_dom, y_train):
    print("\n" + "="*70)
    print("EXPERIMENT 3.5: 5-Fold CV Validation of Phase 3 Champion")
    print("="*70)

    X_combined = pd.concat([X_train_dom], axis=0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    catboost = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        auto_class_weights="Balanced", random_seed=RANDOM_STATE, verbose=0
    )

    fold_results = []
    fold_probs = []
    fold_labels = []

    print("\nRunning 5-fold CV on CatBoost (domain features) ...")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_train)):
        Xtr_fold = X_combined.iloc[train_idx]
        ytr_fold = y_train.iloc[train_idx]
        Xval_fold = X_combined.iloc[val_idx]
        yval_fold = y_train.iloc[val_idx]

        model = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            auto_class_weights="Balanced", random_seed=RANDOM_STATE, verbose=0
        )
        model.fit(Xtr_fold, ytr_fold)
        y_prob = model.predict_proba(Xval_fold)[:, 1]
        m = threshold_metrics(yval_fold, y_prob, 0.5)
        auc = roc_auc_score(yval_fold, y_prob)
        ap = average_precision_score(yval_fold, y_prob)

        fold_results.append({
            "fold": fold_idx + 1,
            "sensitivity": round(m["sensitivity"], 4),
            "precision": round(m["precision"], 4),
            "f1": round(m["f1"], 4),
            "auc_roc": round(auc, 4),
            "auc_pr": round(ap, 4),
        })
        fold_probs.extend(y_prob.tolist())
        fold_labels.extend(yval_fold.tolist())
        print(f"  Fold {fold_idx+1}: sens={m['sensitivity']:.3f}  "
              f"f1={m['f1']:.3f}  AUC={auc:.4f}")

    sens_arr = [r["sensitivity"] for r in fold_results]
    f1_arr = [r["f1"] for r in fold_results]
    auc_arr = [r["auc_roc"] for r in fold_results]

    print(f"\n  CV Summary:")
    print(f"  Sensitivity: {np.mean(sens_arr):.3f} ± {np.std(sens_arr):.3f}")
    print(f"  F1:          {np.mean(f1_arr):.3f} ± {np.std(f1_arr):.3f}")
    print(f"  AUC-ROC:     {np.mean(auc_arr):.3f} ± {np.std(auc_arr):.3f}")

    return {
        "folds": fold_results,
        "cv_sensitivity_mean": round(float(np.mean(sens_arr)), 4),
        "cv_sensitivity_std": round(float(np.std(sens_arr)), 4),
        "cv_f1_mean": round(float(np.mean(f1_arr)), 4),
        "cv_f1_std": round(float(np.std(f1_arr)), 4),
        "cv_auc_mean": round(float(np.mean(auc_arr)), 4),
        "cv_auc_std": round(float(np.std(auc_arr)), 4),
    }


# ---------------------------------------------------------------------------
# Summary Plots
# ---------------------------------------------------------------------------

def make_summary_plot(threshold_results, ablation_results, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Phase 3 — Feature Engineering & Threshold Optimization", fontsize=13)

    # Panel 1: Phase 2 champion metrics at 3 thresholds
    ax = axes[0]
    models = list(threshold_results.keys())
    metrics = ["default_threshold", "clinical_threshold_070", "optimal_f1_threshold"]
    labels = ["Default (t=0.50)", "Clinical (t≥sens0.70)", "Optimal F1"]
    x = np.arange(len(models))
    width = 0.25
    colors = ["#2196F3", "#F44336", "#4CAF50"]
    for i, (m_key, label, color) in enumerate(zip(metrics, labels, colors)):
        sens_vals = []
        for mn in models:
            m_data = threshold_results[mn].get(m_key)
            if m_data and isinstance(m_data, dict):
                sens_vals.append(m_data.get("sensitivity", 0))
            else:
                sens_vals.append(0)
        ax.bar(x + i * width, sens_vals, width, label=label, color=color, alpha=0.85)
    ax.set_xlabel("Model")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Threshold Optimization\n(Sensitivity at Different Thresholds)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_domain", "") for m in models], rotation=15)
    ax.axhline(0.70, color="red", linestyle="--", alpha=0.5, label="Clinical target")
    ax.legend(fontsize=8)

    # Panel 2: Ablation — ΔSensitivity when each group removed
    ax = axes[1]
    if ablation_results:
        sorted_res = sorted(ablation_results, key=lambda x: x["delta_sensitivity"])
        labels_abl = [r["removed_group"].replace("_components", "").replace("_", " ")
                      for r in sorted_res]
        deltas = [r["delta_sensitivity"] for r in sorted_res]
        bar_colors = ["#F44336" if d > 0 else "#4CAF50" for d in deltas]
        bars = ax.barh(labels_abl, deltas, color=bar_colors, edgecolor="white")
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("ΔSensitivity when group removed")
        ax.set_title("Feature Ablation on CatBoost\n(Red = hurts, Green = neutral/helps)")
        for bar, val in zip(bars, deltas):
            ax.text(val + (0.0005 if val >= 0 else -0.0005),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center",
                    ha="left" if val >= 0 else "right", fontsize=8)

    # Panel 3: Sens/Prec trade-off at clinical threshold
    ax = axes[2]
    all_configs = []
    for mn, data in threshold_results.items():
        label = mn.replace("_domain", "")
        for t_key, t_label in [("default_threshold", "default"), ("clinical_threshold_070", "t≥0.70")]:
            m = data.get(t_key)
            if m and isinstance(m, dict) and m.get("sensitivity", 0) > 0:
                all_configs.append({
                    "label": f"{label}\n({t_label})",
                    "sensitivity": m["sensitivity"],
                    "precision": m["precision"],
                    "f1": m["f1"],
                })
    for cfg in all_configs:
        ax.scatter(cfg["sensitivity"], cfg["precision"], s=120, zorder=5)
        ax.annotate(cfg["label"], (cfg["sensitivity"], cfg["precision"]),
                    fontsize=7, textcoords="offset points", xytext=(4, 4))
    ax.axhline(0.15, color="red", linestyle="--", alpha=0.5, label="Min precision (hospital)")
    ax.axvline(0.70, color="green", linestyle="--", alpha=0.5, label="Clinical target")
    ax.set_xlabel("Sensitivity")
    ax.set_ylabel("Precision")
    ax.set_title("Sensitivity vs Precision Trade-off\n(Clinical Operating Points)")
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 0.4])

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved] {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*70)
    print("PHASE 3: Feature Engineering Deep Dive + Threshold Optimization")
    print("Healthcare Readmission Predictor")
    print("="*70)

    print("\nLoading data pipeline ...")
    X_train, X_test, y_train, y_test = run_pipeline()
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}  "
          f"Positive rate (test): {y_test.mean():.1%}")

    print("\nBuilding domain feature sets ...")
    X_train_dom, X_test_dom = prep_domain_features(X_train, X_test)
    print(f"  Domain features: {X_train_dom.shape[1]} columns")

    # Run experiments
    results_3_1 = experiment_3_1(X_train, y_train, X_test, y_test,
                                  X_train_dom, X_test_dom)
    results_3_2 = experiment_3_2(X_train_dom, y_train, X_test_dom, y_test)
    results_3_3, ablation_results = experiment_3_3(X_train_dom, y_train,
                                                    X_test_dom, y_test)
    results_3_4 = experiment_3_4(X_train_dom, y_train, X_test_dom, y_test)
    results_3_5 = experiment_3_5(X_train_dom, y_train)

    # Compile full results
    all_results = {
        "phase": 3,
        "date": "2026-04-01",
        "experiment_3_1_threshold_optimization": results_3_1,
        "experiment_3_2_balanced_rf": results_3_2,
        "experiment_3_3_feature_ablation": results_3_3,
        "experiment_3_4_interaction_features": results_3_4,
        "experiment_3_5_kfold_cv": results_3_5,
    }

    out_path = ROOT / "results" / "phase3_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[Saved] results/phase3_results.json")

    # Summary plot
    make_summary_plot(
        results_3_1,
        ablation_results,
        PLOTS_DIR / "phase3_summary.png"
    )

    # Print Phase 3 master comparison
    print("\n" + "="*70)
    print("PHASE 3 MASTER COMPARISON (all experiments)")
    print("="*70)
    print(f"\n{'Model/Config':<42} {'Sens':>6} {'Prec':>6} {'F1':>6} {'AUC':>7}")
    print("-"*72)

    # Phase 2 baselines
    phase2_baselines = [
        ("LR baseline (Phase 1)",      0.512, 0.176, 0.262, 0.569),
        ("LACE clinical index",         0.146, 0.175, 0.159, None),
        ("CatBoost domain (Phase 2)",   0.447, 0.178, 0.254, 0.550),
        ("LightGBM domain (Phase 2)",   0.375, 0.167, 0.231, 0.536),
    ]
    for row in phase2_baselines:
        auc_str = f"{row[4]:.4f}" if row[4] else "  N/A "
        print(f"  {row[0]:<40} {row[1]:>6.3f} {row[2]:>6.3f} {row[3]:>6.3f} {auc_str:>7}")

    print("-"*72)
    # Phase 3 new results
    for mname in ["CatBoost", "LightGBM"]:
        for config in ["domain_only", "domain_plus_interactions"]:
            key = f"{mname}_{config}"
            r = results_3_4.get(key)
            if r:
                print(f"  Phase3.4 {key:<33} {r['sensitivity']:>6.3f} {r['precision']:>6.3f} "
                      f"{r['f1']:>6.3f} {r['auc_roc']:>7.4f}")

    for mname in ["VanillaRF", "BalancedRF"]:
        r = results_3_2.get(mname)
        if r:
            print(f"  Phase3.2 {mname:<33} {r['sensitivity']:>6.3f} {r['precision']:>6.3f} "
                  f"{r['f1']:>6.3f} {r['auc_roc']:>7.4f}")

    cv = results_3_5
    print(f"\n  CatBoost 5-Fold CV: sens={cv['cv_sensitivity_mean']:.3f}±{cv['cv_sensitivity_std']:.3f}  "
          f"f1={cv['cv_f1_mean']:.3f}±{cv['cv_f1_std']:.3f}  "
          f"AUC={cv['cv_auc_mean']:.3f}±{cv['cv_auc_std']:.3f}")

    print("\n" + "="*70)
    print("Phase 3 complete.")
    print("="*70)

    return all_results


if __name__ == "__main__":
    main()
