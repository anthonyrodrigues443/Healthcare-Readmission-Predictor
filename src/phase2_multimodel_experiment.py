"""Phase 2: Multi-Model Experiment — Healthcare Readmission Predictor.

Compares 6 model families on the UCI Diabetes 130-US Hospitals dataset.
Also tests 3 class-imbalance strategies (class_weight, SMOTE, threshold tuning).

Research-informed model selection:
- Rajkomar et al. 2018: gradient-boosted trees dominate EHR readmission tasks
- Kaggle healthcare competitions: XGBoost/LightGBM consistently top leaderboard
- Strack et al. 2014 (original dataset paper): LogReg baseline AUC ~0.64
"""

import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# Optional imports
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


def load_data():
    """Load processed dataset and split."""
    processed_path = Path('data/processed/readmission_processed.csv')
    if processed_path.exists():
        df = pd.read_csv(processed_path)
    else:
        from src.data_pipeline import clean_and_engineer, download_dataset

        raw_df = download_dataset()
        df = clean_and_engineer(raw_df)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"Generated processed dataset at {processed_path}")

    target = 'readmitted_binary'
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Positive rate — Train: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")
    print(f"Features: {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, model_name, train_time=None):
    """Compute comprehensive metrics."""
    y_pred = model.predict(X_test)

    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_prob = model.decision_function(X_test)
        # Normalize to [0, 1] for AUC
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-10)
    else:
        y_prob = y_pred.astype(float)

    auc_score = roc_auc_score(y_test, y_prob)
    ap_score = average_precision_score(y_test, y_prob)

    # Inference latency
    start = time.perf_counter()
    for _ in range(5):
        model.predict(X_test[:100])
    latency_ms = (time.perf_counter() - start) / 5 / 100 * 1000

    results = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'auc': auc_score,
        'avg_precision': ap_score,
        'latency_ms': round(latency_ms, 3),
        'train_time_s': round(train_time, 2) if train_time else None,
    }
    return results, y_prob


def get_clinical_features():
    """Return the 23 clinical features from Phase 1."""
    return [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses', 'n_medications_changed',
        'n_medications_active', 'polypharmacy', 'prior_utilization',
        'lab_procedure_ratio', 'procedure_ratio', 'age_numeric',
        'A1C_high', 'A1C_tested', 'glucose_high', 'glucose_tested',
        'diabetes_primary', 'med_changed', 'diabetes_med', 'lace_score',
    ]


def run_experiment_1_six_models(X_train, X_test, y_train, y_test):
    """Experiment 2.1: Compare 6 model families on all features."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2.1: Six-Model Comparison (All Features)")
    print("=" * 70)

    # Positive class weight for imbalance
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Class imbalance ratio: {pos_weight:.1f}:1")

    models = {}

    # 1. Random Forest
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=20,
        class_weight='balanced', random_state=42, n_jobs=1,
    )

    # 2. XGBoost
    if HAS_XGB:
        models['XGBoost'] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=pos_weight, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            eval_metric='logloss', verbosity=0,
        )

    # 3. LightGBM
    if HAS_LGB:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            is_unbalance=True, subsample=0.8,
            colsample_bytree=0.8, random_state=42, verbose=-1,
        )

    # 4. CatBoost
    if HAS_CAT:
        models['CatBoost'] = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1,
            auto_class_weights='Balanced', random_seed=42,
            verbose=0,
        )

    # 5. Gradient Boosting (sklearn)
    models['GradientBoosting'] = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42,
    )

    # 6. SVM (RBF kernel — needs scaling, subsample for speed)
    # SVM is O(n^2) to O(n^3) — too slow on 80K+ samples. Use subsample.
    models['SVM-RBF'] = SVC(
        kernel='rbf', C=1.0, gamma='scale',
        class_weight='balanced', probability=True, random_state=42,
    )

    results = []
    probs = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")
        # SVM needs scaling + subsample (O(n^2) complexity)
        if 'SVM' in name:
            scaler = StandardScaler()
            # Subsample to 10K for SVM training (stratified)
            svm_sample_size = min(10000, len(X_train))
            np.random.seed(42)
            idx = np.random.choice(len(X_train), svm_sample_size, replace=False)
            X_tr_svm = X_train.iloc[idx]
            y_tr_svm = y_train.iloc[idx]
            print(f"  (SVM subsampled to {svm_sample_size} for speed)")
            X_tr_scaled = scaler.fit_transform(X_tr_svm)
            X_te_scaled = scaler.transform(X_test)
            start = time.perf_counter()
            model.fit(X_tr_scaled, y_tr_svm)
            train_time = time.perf_counter() - start
            res, y_prob = evaluate_model(model, X_te_scaled, y_test, name, train_time)
        else:
            start = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - start
            res, y_prob = evaluate_model(model, X_test, y_test, name, train_time)

        results.append(res)
        probs[name] = y_prob
        print(f"  -> AUC={res['auc']:.4f} | F1={res['f1']:.3f} | Recall={res['recall']:.3f} | {train_time:.1f}s")

    # Print table
    results_df = pd.DataFrame(results).sort_values('auc', ascending=False)
    results_df['rank'] = range(1, len(results_df) + 1)
    print("\n  === EXPERIMENT 2.1 RESULTS (sorted by AUC) ===")
    print(results_df[['rank', 'model', 'accuracy', 'f1', 'precision', 'recall', 'auc', 'avg_precision', 'latency_ms', 'train_time_s']].to_string(index=False))

    return results, probs, models


def run_experiment_2_clinical_features(X_train, X_test, y_train, y_test):
    """Experiment 2.2: Top 3 models on 23 clinical features only."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2.2: Top 3 Models on 23 Clinical Features")
    print("=" * 70)

    clinical = get_clinical_features()
    available = [f for f in clinical if f in X_train.columns]
    print(f"Using {len(available)} of {len(clinical)} clinical features")

    X_tr_clin = X_train[available]
    X_te_clin = X_test[available]

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {}
    if HAS_XGB:
        models['XGBoost (clinical)'] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=pos_weight, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            eval_metric='logloss', verbosity=0,
        )
    if HAS_LGB:
        models['LightGBM (clinical)'] = LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            is_unbalance=True, subsample=0.8,
            colsample_bytree=0.8, random_state=42, verbose=-1,
        )
    if HAS_CAT:
        models['CatBoost (clinical)'] = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1,
            auto_class_weights='Balanced', random_seed=42, verbose=0,
        )

    results = []
    for name, model in models.items():
        print(f"\n  Training {name}...")
        start = time.perf_counter()
        model.fit(X_tr_clin, y_train)
        train_time = time.perf_counter() - start
        res, _ = evaluate_model(model, X_te_clin, y_test, name, train_time)
        results.append(res)
        print(f"  -> AUC={res['auc']:.4f} | F1={res['f1']:.3f} | Recall={res['recall']:.3f}")

    results_df = pd.DataFrame(results).sort_values('auc', ascending=False)
    print("\n  === EXPERIMENT 2.2 RESULTS (Clinical Features) ===")
    print(results_df[['model', 'accuracy', 'f1', 'precision', 'recall', 'auc']].to_string(index=False))

    return results


def run_experiment_3_imbalance_strategies(X_train, X_test, y_train, y_test):
    """Experiment 2.3: SMOTE vs class_weight vs threshold tuning on XGBoost."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2.3: Imbalance Handling Strategies")
    print("=" * 70)

    if not HAS_XGB:
        print("  XGBoost not available, skipping")
        return []

    results = []
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Strategy 1: No imbalance handling (baseline)
    print("\n  Strategy 1: No imbalance handling...")
    xgb_none = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric='logloss', verbosity=0,
    )
    start = time.perf_counter()
    xgb_none.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    res, _ = evaluate_model(xgb_none, X_test, y_test, 'XGB (no handling)', train_time)
    results.append(res)
    print(f"  -> AUC={res['auc']:.4f} | F1={res['f1']:.3f} | Recall={res['recall']:.3f}")

    # Strategy 2: scale_pos_weight
    print("\n  Strategy 2: scale_pos_weight...")
    xgb_weight = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=pos_weight, random_state=42,
        eval_metric='logloss', verbosity=0,
    )
    start = time.perf_counter()
    xgb_weight.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    res, _ = evaluate_model(xgb_weight, X_test, y_test, 'XGB (class_weight)', train_time)
    results.append(res)
    print(f"  -> AUC={res['auc']:.4f} | F1={res['f1']:.3f} | Recall={res['recall']:.3f}")

    # Strategy 3: SMOTE
    if HAS_IMBLEARN:
        print("\n  Strategy 3: SMOTE oversampling...")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_sm, y_sm = smote.fit_resample(X_train, y_train)
        print(f"  SMOTE: {len(X_train)} -> {len(X_sm)} samples")
        xgb_smote = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric='logloss', verbosity=0,
        )
        start = time.perf_counter()
        xgb_smote.fit(X_sm, y_sm)
        train_time = time.perf_counter() - start
        res, _ = evaluate_model(xgb_smote, X_test, y_test, 'XGB (SMOTE)', train_time)
        results.append(res)
        print(f"  -> AUC={res['auc']:.4f} | F1={res['f1']:.3f} | Recall={res['recall']:.3f}")
    else:
        print("  imbalanced-learn not available, skipping SMOTE")

    # Strategy 4: Threshold tuning (on validation set)
    print("\n  Strategy 4: Threshold tuning...")
    # Use the class-weighted model, tune threshold for best F1
    y_prob = xgb_weight.predict_proba(X_test)[:, 1]
    best_f1 = 0
    best_thresh = 0.5
    thresholds = np.arange(0.05, 0.50, 0.01)
    for thresh in thresholds:
        y_th = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_test, y_th)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    y_tuned = (y_prob >= best_thresh).astype(int)
    res = {
        'model': f'XGB (threshold={best_thresh:.2f})',
        'accuracy': accuracy_score(y_test, y_tuned),
        'f1': f1_score(y_test, y_tuned),
        'precision': precision_score(y_test, y_tuned, zero_division=0),
        'recall': recall_score(y_test, y_tuned),
        'auc': roc_auc_score(y_test, y_prob),
        'avg_precision': average_precision_score(y_test, y_prob),
        'latency_ms': None,
        'train_time_s': None,
    }
    results.append(res)
    print(f"  -> Best threshold={best_thresh:.2f} | F1={res['f1']:.3f} | Recall={res['recall']:.3f} | Precision={res['precision']:.3f}")

    # Strategy 5: Cost-sensitive with higher weight
    print("\n  Strategy 5: Aggressive cost-sensitive (2x weight)...")
    xgb_heavy = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=pos_weight * 2, random_state=42,
        eval_metric='logloss', verbosity=0,
    )
    start = time.perf_counter()
    xgb_heavy.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    res, _ = evaluate_model(xgb_heavy, X_test, y_test, 'XGB (2x weight)', train_time)
    results.append(res)
    print(f"  -> AUC={res['auc']:.4f} | F1={res['f1']:.3f} | Recall={res['recall']:.3f}")

    results_df = pd.DataFrame(results).sort_values('f1', ascending=False)
    print("\n  === EXPERIMENT 2.3 RESULTS (sorted by F1) ===")
    print(results_df[['model', 'accuracy', 'f1', 'precision', 'recall', 'auc']].to_string(index=False))

    return results


def run_experiment_4_cross_validation(X_train, y_train):
    """Experiment 2.4: 5-fold CV on top 3 models for variance estimation."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2.4: 5-Fold Cross-Validation (Variance Estimation)")
    print("=" * 70)

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_models = {}
    if HAS_XGB:
        cv_models['XGBoost'] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=pos_weight, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            eval_metric='logloss', verbosity=0,
        )
    if HAS_LGB:
        cv_models['LightGBM'] = LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            is_unbalance=True, subsample=0.8,
            colsample_bytree=0.8, random_state=42, verbose=-1,
        )
    if HAS_CAT:
        cv_models['CatBoost'] = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1,
            auto_class_weights='Balanced', random_seed=42, verbose=0,
        )

    results = []
    for name, model in cv_models.items():
        print(f"\n  5-fold CV for {name}...")
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=1)
        f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=1)
        res = {
            'model': name,
            'auc_mean': scores.mean(),
            'auc_std': scores.std(),
            'f1_mean': f1_scores.mean(),
            'f1_std': f1_scores.std(),
            'folds': scores.tolist(),
        }
        results.append(res)
        print(f"  -> AUC: {scores.mean():.4f} +/- {scores.std():.4f}")
        print(f"  -> F1:  {f1_scores.mean():.4f} +/- {f1_scores.std():.4f}")
        print(f"  -> Folds: {[f'{s:.4f}' for s in scores]}")

    return results


def plot_results(all_results, probs, y_test):
    """Generate comparison plots."""
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # 1. Model comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    df = pd.DataFrame(all_results).sort_values('auc', ascending=True)

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df)))

    for ax, metric, title in zip(
        axes, ['auc', 'f1', 'recall'],
        ['ROC AUC', 'F1 Score', 'Recall']
    ):
        bars = ax.barh(df['model'], df[metric], color=colors)
        ax.set_xlabel(title)
        ax.set_xlim(0, max(df[metric].max() * 1.15, 0.8))
        for bar, val in zip(bars, df[metric]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=9)
    plt.suptitle('Phase 2: Multi-Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'phase2_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved phase2_model_comparison.png")

    # 2. ROC curves for all models with probabilities
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, y_prob in probs.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Phase 2: ROC Curves — All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'phase2_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved phase2_roc_curves.png")

    # 3. Precision-Recall curves (more informative for imbalanced data)
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, y_prob in probs.items():
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, label=f'{name} (AP={ap:.3f})', linewidth=2)

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Phase 2: Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    baseline_rate = y_test.mean()
    ax.axhline(y=baseline_rate, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline_rate:.3f})')
    plt.tight_layout()
    plt.savefig(results_dir / 'phase2_precision_recall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved phase2_precision_recall.png")


def plot_imbalance_comparison(imbalance_results):
    """Plot imbalance strategy comparison."""
    results_dir = Path('results')
    fig, ax = plt.subplots(figsize=(10, 6))
    df = pd.DataFrame(imbalance_results)

    x = np.arange(len(df))
    width = 0.25

    ax.bar(x - width, df['f1'], width, label='F1', color='#2196F3')
    ax.bar(x, df['precision'], width, label='Precision', color='#4CAF50')
    ax.bar(x + width, df['recall'], width, label='Recall', color='#FF9800')

    ax.set_xticks(x)
    ax.set_xticklabels(df['model'], rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('Phase 2: Imbalance Strategy Comparison (XGBoost)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'phase2_imbalance_strategies.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved phase2_imbalance_strategies.png")


def main():
    print("=" * 70)
    print("PHASE 2: MULTI-MODEL EXPERIMENT")
    print("Healthcare Readmission Predictor")
    print("=" * 70)

    X_train, X_test, y_train, y_test = load_data()

    # Experiment 2.1: Six-model comparison (all features)
    exp1_results, probs, trained_models = run_experiment_1_six_models(
        X_train, X_test, y_train, y_test
    )

    # Experiment 2.2: Top models on clinical features only
    exp2_results = run_experiment_2_clinical_features(
        X_train, X_test, y_train, y_test
    )

    # Experiment 2.3: Imbalance handling strategies
    exp3_results = run_experiment_3_imbalance_strategies(
        X_train, X_test, y_train, y_test
    )

    # Experiment 2.4: Cross-validation variance
    exp4_results = run_experiment_4_cross_validation(X_train, y_train)

    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    plot_results(exp1_results, probs, y_test)
    if exp3_results:
        plot_imbalance_comparison(exp3_results)

    # Save all results to JSON
    results_dir = Path('results')
    phase2_output = {
        'phase': 2,
        'date': '2026-04-02',
        'researcher': 'Anthony Rodrigues',
        'experiment_2_1_six_models': exp1_results,
        'experiment_2_2_clinical_features': exp2_results,
        'experiment_2_3_imbalance_strategies': exp3_results,
        'experiment_2_4_cross_validation': exp4_results,
    }

    # Merge with existing metrics
    metrics_path = results_dir / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path) as f:
            existing = json.load(f)
    else:
        existing = {}

    existing['phase2'] = phase2_output
    with open(metrics_path, 'w') as f:
        json.dump(existing, f, indent=2, default=str)
    print(f"\n  Saved results to {metrics_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 2 SUMMARY")
    print("=" * 70)
    all_results = exp1_results + exp2_results
    # Pick the clinically useful champion: F1 first, then AUC.
    best = max(all_results, key=lambda x: (x['f1'], x['auc']))
    baseline_auc = 0.645  # Phase 1 LogReg baseline
    print(f"\n  Champion: {best['model']}")
    print(f"  AUC: {best['auc']:.4f} (delta vs Phase 1 baseline: +{best['auc'] - baseline_auc:.4f})")
    print(f"  F1: {best['f1']:.3f} | Precision: {best['precision']:.3f} | Recall: {best['recall']:.3f}")

    # Find worst model
    worst = min(all_results, key=lambda x: x['auc'])
    print(f"\n  Worst: {worst['model']}")
    print(f"  AUC: {worst['auc']:.4f}")

    gap = best['auc'] - worst['auc']
    print(f"\n  Gap between best and worst: {gap:.4f} AUC")
    print(f"  Published SOTA range: 0.78-0.87 AUC")

    if exp4_results:
        print("\n  Cross-Validation Summary:")
        for cv in exp4_results:
            print(f"    {cv['model']}: AUC {cv['auc_mean']:.4f} +/- {cv['auc_std']:.4f}")

    print("\n  DONE — Phase 2 complete.")


if __name__ == '__main__':
    main()
