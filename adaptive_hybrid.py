"""
Adaptive Hybrid Phishing URL Classification Framework
======================================================
Orchestrates:
  1. Path A (Chi-square + Random Forest)
  2. Path B (CNN-BiLSTM)
  3. Exponential loss-weighted adaptive gating
  4. Full evaluation and comparison
"""

import os
import sys
import json
import time
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss, confusion_matrix
)
from scipy import stats

from path_a import load_config, train_path_a
from path_b import train_path_b


# ------------------------------------------------------------------ #
#  Adaptive Weighting                                                 #
# ------------------------------------------------------------------ #

def compute_weights(L_A, L_B, gamma):
    """
    Exponential loss-weighted gating:
        alpha = exp(-gamma * L_A) / (exp(-gamma * L_A) + exp(-gamma * L_B))
        beta  = 1 - alpha
    """
    exp_a = np.exp(-gamma * L_A)
    exp_b = np.exp(-gamma * L_B)
    denom = exp_a + exp_b
    alpha = exp_a / denom
    beta = 1.0 - alpha
    return alpha, beta


def adaptive_combine(P_A, P_B, L_A, epoch_L_B, gamma):
    """
    Compute per-epoch weights and return the final combined probability
    using the weights from the best epoch (lowest combined val loss).
    """
    weight_history = []
    best_combined_loss = float("inf")
    best_alpha, best_beta, best_epoch = 0.5, 0.5, 0

    for t, L_B_t in enumerate(epoch_L_B):
        alpha_t, beta_t = compute_weights(L_A, L_B_t, gamma)
        weight_history.append({"epoch": t + 1, "alpha": alpha_t, "beta": beta_t,
                               "L_A": L_A, "L_B": L_B_t})

        # Combined val loss (weighted)
        combined_loss = alpha_t * L_A + beta_t * L_B_t
        if combined_loss < best_combined_loss:
            best_combined_loss = combined_loss
            best_alpha, best_beta = alpha_t, beta_t
            best_epoch = t + 1

    P_final = best_alpha * P_A + best_beta * P_B
    return P_final, best_alpha, best_beta, best_epoch, weight_history


# ------------------------------------------------------------------ #
#  Evaluation helpers                                                 #
# ------------------------------------------------------------------ #

def evaluate(y_true, proba, name="Model"):
    preds = (proba >= 0.5).astype(int)
    m = {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds),
        "recall": recall_score(y_true, preds),
        "f1": f1_score(y_true, preds),
        "roc_auc": roc_auc_score(y_true, proba),
        "log_loss": log_loss(y_true, proba),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
    }
    print(f"\n--- {name} Test Metrics ---")
    for k, v in m.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.6f}")
    print(f"  confusion_matrix:\n    {m['confusion_matrix']}")
    return m


def select_gamma(P_A_val, P_B_val, y_val, L_A, epoch_L_B, candidates):
    """Pick gamma that gives the best combined F1 on validation set."""
    best_gamma, best_f1 = candidates[0], -1.0
    for g in candidates:
        P_final, *_ = adaptive_combine(P_A_val, P_B_val, L_A, epoch_L_B, g)
        preds = (P_final >= 0.5).astype(int)
        f = f1_score(y_val, preds)
        print(f"  gamma={g:.1f}  val F1={f:.6f}")
        if f > best_f1:
            best_f1, best_gamma = f, g
    print(f"  => Selected gamma = {best_gamma}")
    return best_gamma


# ------------------------------------------------------------------ #
#  Main pipeline                                                      #
# ------------------------------------------------------------------ #

def run_hybrid(cfg):
    out_dir = os.path.join(cfg["output_dir"], cfg["split_dir"])
    os.makedirs(out_dir, exist_ok=True)

    # ---- Path A ----
    print("=" * 60)
    print("  PATH A: Chi-square + Random Forest")
    print("=" * 60)
    res_a = train_path_a(cfg)

    # ---- Path B ----
    print("\n" + "=" * 60)
    print("  PATH B: CNN-BiLSTM")
    print("=" * 60)
    res_b = train_path_b(cfg)

    # ---- Adaptive Weighting ----
    print("\n" + "=" * 60)
    print("  ADAPTIVE WEIGHTING")
    print("=" * 60)

    L_A = res_a["val_loss"]
    epoch_L_B = res_b["epoch_val_losses"]

    gamma_candidates = cfg["adaptive"]["gamma_candidates"]
    print("\nSelecting gamma on validation set:")
    gamma = select_gamma(
        res_a["val_proba"], res_b["val_proba"], res_a["y_val"],
        L_A, epoch_L_B, gamma_candidates,
    )

    # Combined test predictions
    P_final_test, alpha, beta, best_ep, weight_hist = adaptive_combine(
        res_a["test_proba"], res_b["test_proba"],
        L_A, epoch_L_B, gamma,
    )
    print(f"\nBest-epoch weights: alpha(A)={alpha:.4f}  beta(B)={beta:.4f}  (epoch {best_ep})")

    # ---- Static equal-weight baseline ----
    P_static = 0.5 * res_a["test_proba"] + 0.5 * res_b["test_proba"]

    # ---- Evaluation ----
    print("\n" + "=" * 60)
    print("  EVALUATION (Test Set)")
    print("=" * 60)

    y_test = res_a["y_test"]
    m_a = evaluate(y_test, res_a["test_proba"], "Path A (RF)")
    m_b = evaluate(y_test, res_b["test_proba"], "Path B (CNN-BiLSTM)")
    m_static = evaluate(y_test, P_static, "Static Ensemble (0.5/0.5)")
    m_hybrid = evaluate(y_test, P_final_test, "Adaptive Hybrid")

    # ---- Computational Metrics ----
    print("\n--- Computational Metrics ---")
    comp = {
        "path_a_train_time_s": res_a["train_time"],
        "path_b_train_time_s": res_b["train_time"],
        "path_b_best_epoch": res_b["metrics"].get("best_epoch"),
        "path_b_total_epochs": res_b["metrics"].get("total_epochs"),
        "path_a_inference_ms": res_a["metrics"]["inference_ms_per_sample"],
        "path_b_inference_ms": res_b["metrics"]["inference_ms_per_sample"],
    }
    if "model_size_mb" in res_b["metrics"]:
        comp["path_b_model_size_mb"] = res_b["metrics"]["model_size_mb"]

    # Path A model size
    rf_path = os.path.join(out_dir, "random_forest.joblib")
    if os.path.exists(rf_path):
        comp["path_a_model_size_mb"] = os.path.getsize(rf_path) / (1024 * 1024)

    for k, v in comp.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ---- Comparison Table ----
    print("\n" + "=" * 60)
    print("  COMPARISON TABLE")
    print("=" * 60)
    header = f"{'Metric':<18} {'Path A':>10} {'Path B':>10} {'Static':>10} {'Adaptive':>10}"
    print(header)
    print("-" * len(header))
    for metric_name in ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"):
        row = (
            f"{metric_name:<18} "
            f"{m_a[metric_name]:>10.4f} "
            f"{m_b[metric_name]:>10.4f} "
            f"{m_static[metric_name]:>10.4f} "
            f"{m_hybrid[metric_name]:>10.4f}"
        )
        print(row)

    # ---- Statistical Significance (bootstrap) ----
    print("\n--- Statistical Significance (bootstrap paired test) ---")
    n_bootstrap = 1000
    rng = np.random.RandomState(cfg["random_seed"])
    n = len(y_test)
    hybrid_preds = (P_final_test >= 0.5).astype(int)

    for name, other_proba in [("Path A", res_a["test_proba"]),
                               ("Path B", res_b["test_proba"]),
                               ("Static", P_static)]:
        other_preds = (other_proba >= 0.5).astype(int)
        hybrid_f1s, other_f1s = [], []
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            hybrid_f1s.append(f1_score(y_test[idx], hybrid_preds[idx]))
            other_f1s.append(f1_score(y_test[idx], other_preds[idx]))
        diff = np.array(hybrid_f1s) - np.array(other_f1s)
        mean_diff = np.mean(diff)
        p_value = np.mean(diff <= 0) * 2  # two-sided
        p_value = min(p_value, 1.0)
        sig = "YES" if p_value < 0.05 else "NO"
        print(f"  Adaptive vs {name}: mean F1 diff = {mean_diff:+.4f}, p = {p_value:.4f} (significant: {sig})")

    # ---- Save results ----
    report = {
        "config": {
            "split_dir": cfg["split_dir"],
            "gamma": gamma,
            "optimal_k": res_a["optimal_k"],
            "selected_features": res_a["selected_features"],
            "best_epoch_weights": {"alpha": alpha, "beta": beta, "epoch": best_ep},
        },
        "metrics": {
            "path_a": m_a,
            "path_b": m_b,
            "static_ensemble": m_static,
            "adaptive_hybrid": m_hybrid,
        },
        "computational": comp,
        "weight_history": weight_hist,
    }
    report_path = os.path.join(out_dir, "results.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull results saved to {report_path}")

    return report


# ------------------------------------------------------------------ #
#  CLI entry-point                                                    #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(cfg_path)
    run_hybrid(cfg)
    print("\n✓ Adaptive Hybrid pipeline complete.")
