"""
Path A Baseline: Chi-square + Random Forest
=============================================
Reproduction of the Random Forest pipeline from:
  Aqilla Khairunnisya et al. (2025)

Can be run standalone or imported by adaptive_hybrid.py.
Preprocessing is handled by preprocess.py; hyperparameters are in config.yaml.
"""

import warnings
import time
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import tempfile
import os

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_validate
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, classification_report, confusion_matrix
)

from preprocess import load_config, aqilla_preprocess

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# TRAIN FUNCTION (callable by adaptive_hybrid)
# ──────────────────────────────────────────────

def train_path_a(cfg):
    """Full Path A pipeline. Returns dict with model artefacts + metrics.

    Can be called standalone or by adaptive_hybrid.py.
    Outputs P_A(x) probability scores and validation log-loss L_A.
    """
    SEED = cfg["random_seed"]
    acfg = cfg["aqilla"]
    np.random.seed(SEED)

    # ── 1. Load & preprocess ──
    data = aqilla_preprocess(cfg)
    X_final = data["X_final"]
    y = data["y"]
    le = data["le"]
    chi2_df = data["chi2_df"]
    feature_names = data["feature_names"]

    # ── 2. Train / Val / Test split ──
    print("\n" + "=" * 60)
    print("CHI-SQUARE + RANDOM FOREST TRAINING & EVALUATION")
    print("=" * 60)

    X_data = X_final.values
    y_data = y.values
    print(f"\n  Features after Chi-Square selection: {X_data.shape[1]}")

    test_size = acfg["test_size"]
    val_size = acfg.get("val_size", 0.15)

    # First split: (train+val) vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_data, y_data, test_size=test_size, random_state=SEED, stratify=y_data
    )
    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=SEED, stratify=y_trainval
    )

    print(f"  Train: {X_train.shape[0]}  |  Val: {X_val.shape[0]}  |  Test: {X_test.shape[0]}")

    # ── 3. GridSearchCV ──
    cv = StratifiedKFold(n_splits=acfg["cv_folds"], shuffle=True, random_state=SEED)

    # Convert null → None for max_depth
    rf_max_depth = [None if v is None else v for v in acfg["rf"]["max_depth"]]

    print(f"\n  >>> Random Forest")
    print(f"      GridSearchCV with {acfg['cv_folds']}-Fold CV ...")

    train_start = time.time()

    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=SEED, n_jobs=-1),
        param_grid={
            "n_estimators": acfg["rf"]["n_estimators"],
            "max_depth": rf_max_depth,
            "min_samples_split": acfg["rf"]["min_samples_split"],
        },
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=2,
    )
    grid.fit(X_train, y_train)
    train_time = time.time() - train_start

    best = grid.best_estimator_
    print(f"      Best params: {grid.best_params_}")

    cv_results = cross_validate(
        best, X_train, y_train, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        n_jobs=-1,
        verbose=1,
    )
    print(f"      CV Accuracy:  {cv_results['test_accuracy'].mean():.4f} "
          f"± {cv_results['test_accuracy'].std():.4f}")

    # ── 4. Validation metrics (for adaptive weighting: L_A) ──
    val_proba = best.predict_proba(X_val)[:, 1]
    val_loss_value = log_loss(y_val, val_proba)
    print(f"\n      Validation log-loss (L_A): {val_loss_value:.6f}")

    # ── 5. Test-set evaluation ──
    y_pred = best.predict(X_test)
    test_proba = best.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1v  = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, test_proba)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n      ── Test Set Results ──")
    print(f"      Accuracy : {acc:.4f}")
    print(f"      Precision: {prec:.4f}")
    print(f"      Recall   : {rec:.4f}")
    print(f"      F1-score : {f1v:.4f}")
    print(f"      ROC AUC  : {auc:.4f}")
    print(f"      Confusion Matrix:\n{cm}")
    print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

    # ── 6. Computational metrics ──
    inf_start = time.time()
    for _ in range(3):
        _ = best.predict_proba(X_test)
    inference_time = (time.time() - inf_start) / 3
    inference_per_sample_ms = (inference_time / len(X_test)) * 1000

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "model.joblib")
        joblib.dump(best, tmp_path)
        model_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)

    print(f"\n      ── Computational Metrics ──")
    print(f"      Training time   : {train_time:.2f}s")
    print(f"      Inference/sample: {inference_per_sample_ms:.4f}ms")
    print(f"      Model size      : {model_size_mb:.2f}MB")

    imp = pd.Series(best.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=False)
    print(f"      Top-10 Feature Importances:")
    for fname, fval in imp.head(10).items():
        print(f"        {fname:30s} {fval:.4f}")

    # ── 7. Save results ──
    OUT_DIR = Path(f"results/path_a_baseline/dataset_{acfg['dataset_id']}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results_all = [{
        "model": "Chi-Square + Random Forest",
        "best_params": grid.best_params_,
        "cv_accuracy_mean": round(cv_results["test_accuracy"].mean(), 4),
        "cv_accuracy_std": round(cv_results["test_accuracy"].std(), 4),
        "test_accuracy": round(acc, 4),
        "test_precision": round(prec, 4),
        "test_recall": round(rec, 4),
        "test_f1": round(f1v, 4),
        "test_roc_auc": round(auc, 4),
        "val_log_loss": round(val_loss_value, 6),
        "train_time_s": round(train_time, 2),
        "inference_ms_per_sample": round(inference_per_sample_ms, 4),
        "model_size_mb": round(model_size_mb, 2),
    }]
    results_df = pd.DataFrame(results_all)
    results_df.to_csv(OUT_DIR / "metrics_summary.csv", index=False)
    chi2_df.to_csv(OUT_DIR / "chi2_feature_scores.csv", index=False)

    print(f"\nResults saved to {OUT_DIR}/")

    # ── 8. Return dict for adaptive hybrid ──
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1v,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "inference_ms_per_sample": inference_per_sample_ms,
        "model_size_mb": model_size_mb,
        "train_time_s": train_time,
    }

    return {
        "model": best,
        "selected_features": feature_names,
        "optimal_k": len(feature_names),
        "val_proba": val_proba,
        "val_loss": val_loss_value,
        "test_proba": test_proba,
        "y_val": y_val,
        "y_test": y_test,
        "train_time": train_time,
        "metrics": metrics,
    }


# ──────────────────────────────────────────────
# CLI ENTRY-POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    cfg = load_config("config.yaml")
    result = train_path_a(cfg)
    print("Done.")
