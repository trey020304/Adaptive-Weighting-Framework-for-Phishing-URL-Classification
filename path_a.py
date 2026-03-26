"""
Path A: Chi-square Feature Selection + Random Forest Classifier
================================================================
1. Determines optimal k via stratified 5-fold CV on training set (max F1).
2. Trains Random Forest on the top-k features.
3. Outputs probability P_A(x) and validation log-loss L_A.
"""

import os
import sys
import time
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, log_loss, accuracy_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_split(split_dir, feature_cols, label_col, positive_label):
    """Load train / val / test CSVs and return X, y arrays."""
    data = {}
    for role in ("train", "val", "test"):
        fp = os.path.join(split_dir, f"{role}.csv")
        df = pd.read_csv(fp)
        X = df[feature_cols].values.astype(np.float32)
        y = (df[label_col] == positive_label).astype(np.int32).values
        data[role] = (X, y)
    return data


# ------------------------------------------------------------------ #
#  Chi-square Feature Selection                                       #
# ------------------------------------------------------------------ #

def find_optimal_k(X_train, y_train, k_range, n_folds, seed):
    """Find the k that maximises mean F1 via stratified CV."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    best_k, best_f1 = k_range[0], -1.0

    print(f"{'k':>4}  {'Mean F1':>8}")
    print("-" * 16)

    for k in k_range:
        fold_f1s = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            Xtr, ytr = X_train[train_idx], y_train[train_idx]
            Xvl, yvl = X_train[val_idx], y_train[val_idx]

            selector = SelectKBest(chi2, k=k).fit(Xtr, ytr)
            Xtr_k = selector.transform(Xtr)
            Xvl_k = selector.transform(Xvl)

            rf = RandomForestClassifier(
                n_estimators=100, random_state=seed, class_weight="balanced", n_jobs=-1
            )
            rf.fit(Xtr_k, ytr)
            preds = rf.predict(Xvl_k)
            fold_f1s.append(f1_score(yvl, preds))

        mean_f1 = np.mean(fold_f1s)
        print(f"{k:4d}  {mean_f1:.6f}")
        if mean_f1 > best_f1:
            best_f1, best_k = mean_f1, k

    print(f"\n=> Optimal k = {best_k}  (F1 = {best_f1:.6f})")
    return best_k


# ------------------------------------------------------------------ #
#  Train & Evaluate                                                   #
# ------------------------------------------------------------------ #

def train_path_a(cfg):
    """Full Path A pipeline.  Returns dict with model artefacts + metrics."""
    seed = cfg["random_seed"]
    np.random.seed(seed)

    split_dir = os.path.join("processed_datasets", cfg["split_dir"])
    feature_cols = cfg["feature_columns"]
    label_col = cfg["label_column"]
    pos_label = cfg["positive_label"]
    pa = cfg["path_a"]

    # 1. Load data
    print("Loading data …")
    data = load_split(split_dir, feature_cols, label_col, pos_label)
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # 2. Optimal k
    print("\n--- Chi-square k search ---")
    k = find_optimal_k(
        X_train, y_train,
        k_range=pa["chi2_k_range"],
        n_folds=pa["chi2_cv_folds"],
        seed=seed,
    )

    # 3. Fit selector on full training set
    selector = SelectKBest(chi2, k=k)
    X_train_k = selector.fit_transform(X_train, y_train)
    X_val_k = selector.transform(X_val)
    X_test_k = selector.transform(X_test)

    selected_mask = selector.get_support()
    selected_features = [f for f, m in zip(feature_cols, selected_mask) if m]
    print(f"\nSelected features ({k}): {selected_features}")

    # 4. Train Random Forest
    print("\nTraining Random Forest …")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=pa["rf_n_estimators"],
        random_state=seed,
        class_weight=pa["rf_class_weight"],
        n_jobs=-1,
    )
    rf.fit(X_train_k, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.2f}s")

    # 5. Validation probabilities & loss
    val_proba = rf.predict_proba(X_val_k)[:, 1]
    val_loss = log_loss(y_val, val_proba)
    print(f"  Validation log-loss (L_A): {val_loss:.6f}")

    # 6. Test-set evaluation
    test_proba = rf.predict_proba(X_test_k)[:, 1]
    test_preds = (test_proba >= 0.5).astype(int)

    t0 = time.time()
    _ = rf.predict_proba(X_test_k)
    inference_time_total = time.time() - t0
    inference_time_per_sample = (inference_time_total / len(X_test_k)) * 1000  # ms

    metrics = {
        "accuracy": accuracy_score(y_test, test_preds),
        "precision": precision_score(y_test, test_preds),
        "recall": recall_score(y_test, test_preds),
        "f1": f1_score(y_test, test_preds),
        "roc_auc": roc_auc_score(y_test, test_proba),
        "confusion_matrix": confusion_matrix(y_test, test_preds).tolist(),
        "train_time_s": train_time,
        "inference_ms_per_sample": inference_time_per_sample,
    }

    print("\n--- Path A Test Metrics ---")
    for k_name, v in metrics.items():
        if k_name != "confusion_matrix":
            print(f"  {k_name}: {v:.6f}" if isinstance(v, float) else f"  {k_name}: {v}")
    print(f"  confusion_matrix:\n    {metrics['confusion_matrix']}")

    # 7. Save artefacts
    out_dir = os.path.join(cfg["output_dir"], cfg["split_dir"])
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(selector, os.path.join(out_dir, "chi2_selector.joblib"))
    joblib.dump(rf, os.path.join(out_dir, "random_forest.joblib"))
    joblib.dump(selected_features, os.path.join(out_dir, "selected_features.joblib"))
    print(f"\nArtefacts saved to {out_dir}")

    return {
        "model": rf,
        "selector": selector,
        "selected_features": selected_features,
        "optimal_k": len(selected_features),
        "val_proba": val_proba,
        "val_loss": val_loss,
        "test_proba": test_proba,
        "test_preds": test_preds,
        "y_val": y_val,
        "y_test": y_test,
        "X_val_k": X_val_k,
        "X_test_k": X_test_k,
        "metrics": metrics,
        "train_time": train_time,
    }


# ------------------------------------------------------------------ #
#  CLI entry-point                                                    #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(cfg_path)
    results = train_path_a(cfg)
    print("\nPath A complete.")
