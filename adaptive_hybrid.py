"""
Adaptive Hybrid Phishing URL Classification Framework
======================================================
Orchestrates:
  1. Path A (Chi-square + Random Forest)  — from path_a_baseline
  2. Path B (Hybrid CNN-BiLSTM)           — from path_b_baseline
  3. Exponential loss-weighted adaptive gating
  4. Full evaluation and comparison

Weighting equation (per epoch t):
    alpha(t) = exp(-gamma * L_A) / (exp(-gamma * L_A) + exp(-gamma * L_B(t)))
    beta(t)  = 1 - alpha(t)
    P_final  = alpha(t) * P_A(x) + beta(t) * P_B(x)
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss, confusion_matrix
)

from preprocess import load_config
from path_a_baseline import train_path_a
from path_b_baseline import train_path_b


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
    # ── Override both pipeline dataset_ids with the adaptive dataset_id ──
    adaptive_cfg = cfg["adaptive"]
    adaptive_ds = adaptive_cfg.get("dataset_id")

    if adaptive_ds:
        cfg["aqilla"]["dataset_id"] = adaptive_ds
        cfg["princeton_improved"]["dataset_id"] = adaptive_ds
        # Ensure both paths use identical split ratios
        if "test_size" in adaptive_cfg:
            cfg["aqilla"]["test_size"] = adaptive_cfg["test_size"]
            cfg["princeton_improved"]["test_size"] = adaptive_cfg["test_size"]
        if "val_size" in adaptive_cfg:
            cfg["aqilla"]["val_size"] = adaptive_cfg["val_size"]
            cfg["princeton_improved"]["val_size"] = adaptive_cfg["val_size"]

    out_dir = os.path.join("results", "adaptive_hybrid", f"dataset_{adaptive_ds or 'default'}")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Path A: Chi-square + Random Forest ----
    print("=" * 60)
    print("  PATH A: Chi-square + Random Forest")
    print("=" * 60)
    res_a = train_path_a(cfg)

    # ---- Path B: Hybrid CNN-BiLSTM ----
    print("\n" + "=" * 60)
    print("  PATH B: Hybrid CNN-BiLSTM")
    print("=" * 60)
    res_b = train_path_b(cfg)

    # ---- Adaptive Weighting ----
    print("\n" + "=" * 60)
    print("  ADAPTIVE WEIGHTING")
    print("=" * 60)

    L_A = res_a["val_loss"]
    epoch_L_B = res_b["epoch_val_losses"]

    gamma_candidates = adaptive_cfg["gamma_candidates"]
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
    path_b_best = res_b["metrics"].get("best_epoch")
    path_b_total = res_b["metrics"].get("total_epochs")

    # Adaptive hybrid's own inference time: time to combine P_A + P_B
    n_test = len(y_test)
    inf_start = time.time()
    for _ in range(100):
        _ = alpha * res_a["test_proba"] + beta * res_b["test_proba"]
    hybrid_inference_per_run = (time.time() - inf_start) / 100
    # Total hybrid inference = Path A inference + Path B inference + weighting
    hybrid_inference_ms = (
        res_a["metrics"]["inference_ms_per_sample"]
        + res_b["metrics"]["inference_ms_per_sample"]
        + (hybrid_inference_per_run / n_test) * 1000
    )

    # Adaptive hybrid total training time = Path A + Path B training
    hybrid_train_time = res_a["train_time"] + res_b["train_time"]

    # Adaptive hybrid model size = Path A + Path B models combined
    path_a_size = res_a["metrics"].get("model_size_mb", 0) or 0
    path_b_size = res_b["metrics"].get("model_size_mb", 0) or 0
    hybrid_model_size = path_a_size + path_b_size

    comp = {
        "path_a_train_time_s": res_a["train_time"],
        "path_b_train_time_s": res_b["train_time"],
        "adaptive_train_time_s": hybrid_train_time,
        "path_a_inference_ms": res_a["metrics"]["inference_ms_per_sample"],
        "path_b_inference_ms": res_b["metrics"]["inference_ms_per_sample"],
        "adaptive_inference_ms": hybrid_inference_ms,
        "path_a_model_size_mb": path_a_size,
        "path_b_model_size_mb": path_b_size,
        "adaptive_model_size_mb": hybrid_model_size,
        "path_b_best_epoch": path_b_best,
        "path_b_total_epochs": path_b_total,
        "path_b_convergence_rate": round(path_b_best / path_b_total, 4) if path_b_best and path_b_total else None,
        "adaptive_best_epoch": best_ep,
        "adaptive_convergence_rate": round(best_ep / path_b_total, 4) if path_b_total else None,
    }

    for k, v in comp.items():
        if v is None:
            print(f"  {k}: N/A")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

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

    # ---- Statistical Significance (bootstrap paired test) ----
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
            "dataset_id": adaptive_ds,
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

    # ---- Save checkpoint (validated classifier) ----
    ckpt_dir = os.path.join(out_dir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Path A model + preprocessing
    joblib.dump(res_a["model"], os.path.join(ckpt_dir, "path_a_rf.joblib"))
    joblib.dump(res_a["selected_features"], os.path.join(ckpt_dir, "path_a_features.joblib"))
    joblib.dump(res_a["mm_scaler"], os.path.join(ckpt_dir, "path_a_mm_scaler.joblib"))
    joblib.dump(res_a["z_scaler"], os.path.join(ckpt_dir, "path_a_z_scaler.joblib"))
    joblib.dump(res_a["le"], os.path.join(ckpt_dir, "label_encoder.joblib"))

    # Path B model + preprocessing
    res_b["model"].save(os.path.join(ckpt_dir, "path_b_cnn_bilstm.keras"))
    joblib.dump(res_b["char_to_idx"], os.path.join(ckpt_dir, "path_b_char_to_idx.joblib"))
    joblib.dump(res_b["token_vocab"], os.path.join(ckpt_dir, "path_b_token_vocab.joblib"))
    joblib.dump(res_b["tab_scaler"], os.path.join(ckpt_dir, "path_b_tab_scaler.joblib"))

    # Adaptive weights
    adaptive_state = {
        "gamma": gamma,
        "alpha": alpha,
        "beta": beta,
        "best_epoch": best_ep,
    }
    joblib.dump(adaptive_state, os.path.join(ckpt_dir, "adaptive_weights.joblib"))

    print(f"Checkpoint saved to {ckpt_dir}/")
    print(f"  Path A: path_a_rf.joblib, path_a_features.joblib, path_a_mm_scaler.joblib, path_a_z_scaler.joblib")
    print(f"  Path B: path_b_cnn_bilstm.keras, path_b_char_to_idx.joblib, path_b_token_vocab.joblib, path_b_tab_scaler.joblib")
    print(f"  Adaptive: adaptive_weights.joblib (gamma={gamma}, alpha={alpha:.4f}, beta={beta:.4f})")
    print(f"  Labels: label_encoder.joblib")

    return report


# ------------------------------------------------------------------ #
#  CLI entry-point                                                    #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(cfg_path)
    run_hybrid(cfg)
    print("\n✓ Adaptive Hybrid pipeline complete.")
