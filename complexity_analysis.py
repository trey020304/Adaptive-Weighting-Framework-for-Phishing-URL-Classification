"""
Computational Complexity (Big O) Analysis & Visualization
==========================================================
Computes and visualizes the theoretical Big O complexity of each pipeline
using the actual hyperparameters from config.yaml and real dataset sizes.

Output: results/charts/complexity/ directory with charts + console summary.

Usage:
    python complexity_analysis.py
"""

import os
import json
import math
import yaml
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = "results"
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts", "complexity")
DPI = 300
FIG_FORMAT = "png"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "path_a": "#2196F3",
    "path_b": "#FF9800",
    "adaptive_hybrid": "#4CAF50",
    "odae_wpdc": "#E91E63",
    "pso_xgboost": "#9C27B0",
    "bigru_attention": "#00BCD4",
}

MODEL_LABELS = {
    "path_a": "Path A\n(Chi² + RF)",
    "path_b": "Path B\n(CNN-BiLSTM)",
    "adaptive_hybrid": "Adaptive Hybrid\n(Proposed)",
    "odae_wpdc": "ODAE-WPDC",
    "pso_xgboost": "PSO-XGBoost",
    "bigru_attention": "BiGRU-Attention",
}

MODEL_LABELS_SHORT = {
    "path_a": "Path A",
    "path_b": "Path B",
    "adaptive_hybrid": "Adaptive",
    "odae_wpdc": "ODAE-WPDC",
    "pso_xgboost": "PSO-XGB",
    "bigru_attention": "BiGRU",
}

MODEL_ORDER = [
    "path_a", "path_b", "adaptive_hybrid",
    "odae_wpdc", "pso_xgboost", "bigru_attention",
]


# ══════════════════════════════════════════════════════════════════════════════
#  Load Config
# ══════════════════════════════════════════════════════════════════════════════

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def get_dataset_size(cfg, dataset_id):
    """Return (n_total, n_features) for a given dataset by reading the CSV header + row count."""
    ds = cfg["datasets"].get(str(dataset_id))
    if ds is None:
        return None, None
    path = ds["path"]
    if os.path.isfile(path):
        df = pd.read_csv(path, nrows=0)
        n_rows = sum(1 for _ in open(path, encoding="utf-8", errors="ignore")) - 1
        return n_rows, len(cfg["feature_columns"])
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
#  Big O Complexity Formulas (as operation counts)
# ══════════════════════════════════════════════════════════════════════════════

def complexity_path_a(n, d, cfg):
    """Chi-Square + Random Forest with GridSearchCV."""
    pa = cfg["path_a"]
    aq = cfg.get("aqilla", {})

    T = pa.get("rf_n_estimators", 300)
    D = pa.get("rf_max_depth", 30) or 30
    k_range = pa.get("chi2_k_range", list(range(1, d + 1)))
    d_prime = max(k_range) if k_range else d  # worst-case selected features

    # GridSearchCV parameters (from aqilla config if present)
    rf_grid = aq.get("rf", {})
    n_est_list = rf_grid.get("n_estimators", [T])
    max_d_list = rf_grid.get("max_depth", [D])
    min_s_list = rf_grid.get("min_samples_split", [2])
    G = len(n_est_list) * len(max_d_list) * len(min_s_list)
    cv_folds = aq.get("cv_folds", 5)

    # Chi-square scoring
    chi2_ops = n * d

    # GridSearchCV: G combos × k folds × (train RF with T trees)
    # Each tree: O(n * sqrt(d') * D)
    grid_ops = G * cv_folds * T * n * math.sqrt(d_prime) * D

    train_ops = chi2_ops + grid_ops
    infer_ops = T * D  # per sample

    return {
        "train_ops": train_ops,
        "infer_ops_per_sample": infer_ops,
        "formula_train": f"O(G·k·T·n·√d'·D) = O({G}·{cv_folds}·{T}·n·√{d_prime}·{D})",
        "formula_infer": f"O(T·D) = O({T}·{D})",
        "big_o_train": "O(G·k·T·n·√d'·D)",
        "big_o_infer": "O(T·D)",
        "dominant_factor": "Grid search × RF ensemble",
        "params": {
            "G": G, "k": cv_folds, "T": T, "D": D,
            "d'": d_prime, "n": n, "d": d,
        },
    }


def complexity_path_b(n, d, cfg):
    """Hybrid CNN-BiLSTM (tri-branch: char + token + tabular)."""
    pb = cfg["princeton_improved"]

    L_c = pb["char_max_len"]          # 300
    e_c = pb["char_embedding_dim"]    # 256
    f1 = pb["char_conv1_filters"]     # 128
    k1 = pb["char_conv1_kernel"]      # 3
    f2 = pb["char_conv2_filters"]     # 64
    k2 = pb["char_conv2_kernel"]      # 3
    h_c = pb["char_bilstm_units"]     # 64

    L_t = pb["token_max_len"]         # 30
    e_t = pb["token_embedding_dim"]   # 64
    h_t = pb["token_bilstm_units"]    # 32

    u1 = pb["tab_dense1_units"]       # 128
    u2 = pb["tab_dense2_units"]       # 64
    m1 = pb["merge_dense1_units"]     # 128
    m2 = pb["merge_dense2_units"]     # 64

    E = pb["max_epochs"]              # 50
    B = pb["batch_size"]              # 32

    # Char branch: Conv1D #1 + Pool + Conv1D #2 + Pool + BiLSTM + Attention
    conv1_ops = L_c * f1 * k1 * e_c             # ~29.5M
    L_after_pool1 = L_c // 2
    conv2_ops = L_after_pool1 * f2 * k2 * f1    # ~3.7M
    L_after_pool2 = L_after_pool1 // 2
    bilstm_char_ops = L_after_pool2 * 4 * h_c * (h_c + f2) * 2  # 2 directions, 4 gates
    attn_char_ops = L_after_pool2 * 2 * h_c

    # Token branch: BiLSTM + Attention
    bilstm_token_ops = L_t * 4 * h_t * (h_t + e_t) * 2
    attn_token_ops = L_t * 2 * h_t

    # Tabular branch
    tab_ops = d * u1 + u1 * u2

    # Merge head
    concat_dim = 2 * h_c + 2 * h_t + u2
    merge_ops = concat_dim * m1 + m1 * m2 + m2

    per_sample = (conv1_ops + conv2_ops + bilstm_char_ops + attn_char_ops +
                  bilstm_token_ops + attn_token_ops + tab_ops + merge_ops)

    train_ops = E * (n / B) * B * per_sample  # = E * n * per_sample
    infer_ops = per_sample

    return {
        "train_ops": train_ops,
        "infer_ops_per_sample": infer_ops,
        "formula_train": f"O(E·n·(L_c·f1·e_c + ...)) ≈ O({E}·n·{per_sample:.0f})",
        "formula_infer": f"O({per_sample:.0f}) per sample",
        "big_o_train": "O(E·n·L_c·f₁·e_c)",
        "big_o_infer": "O(L_c·f₁·e_c)",
        "dominant_factor": "Char-branch Conv1D",
        "params": {
            "E": E, "B": B, "L_c": L_c, "e_c": e_c,
            "f1": f1, "f2": f2, "h_c": h_c,
            "L_t": L_t, "e_t": e_t, "h_t": h_t,
            "n": n, "d": d,
        },
        "per_sample_breakdown": {
            "char_conv1": conv1_ops,
            "char_conv2": conv2_ops,
            "char_bilstm": bilstm_char_ops,
            "char_attention": attn_char_ops,
            "token_bilstm": bilstm_token_ops,
            "token_attention": attn_token_ops,
            "tabular_dense": tab_ops,
            "merge_dense": merge_ops,
        },
    }


def complexity_adaptive(n, d, cfg):
    """Adaptive Hybrid = Path A + Path B + lightweight weighting."""
    pa = complexity_path_a(n, d, cfg)
    pb = complexity_path_b(n, d, cfg)

    ad = cfg["adaptive"]
    n_gamma = len(ad.get("gamma_candidates", [1.0, 2.0, 5.0]))
    E_b = cfg["princeton_improved"]["max_epochs"]

    # Adaptive overhead: gamma search over epoch losses + bootstrap test
    n_val = int(n * 0.15)
    n_test = int(n * 0.20)
    adaptive_overhead = n_gamma * E_b * n_val + 1000 * n_test  # gamma search + bootstrap

    train_ops = pa["train_ops"] + pb["train_ops"] + adaptive_overhead
    infer_ops = pa["infer_ops_per_sample"] + pb["infer_ops_per_sample"] + 1  # weighted avg

    return {
        "train_ops": train_ops,
        "infer_ops_per_sample": infer_ops,
        "formula_train": "O(Path_A + Path_B + |Γ|·E_B·n_val)",
        "formula_infer": "O(Path_A_infer + Path_B_infer)",
        "big_o_train": "O(Path_A + Path_B)",
        "big_o_infer": "O(Path_A + Path_B)",
        "dominant_factor": "Sum of Path A + Path B (adaptive overhead negligible)",
        "params": {
            "n_gamma": n_gamma, "E_B": E_b,
            "n_val": n_val, "n_test": n_test,
            "path_a_ops": pa["train_ops"],
            "path_b_ops": pb["train_ops"],
            "adaptive_overhead": adaptive_overhead,
            "n": n, "d": d,
        },
    }


def complexity_odae_wpdc(n, d, cfg):
    """ODAE-WPDC: AAA feature selection + IWO hyperparameter opt + DAE + k-fold CV."""
    oc = cfg["odae_wpdc"]
    aaa = oc["aaa"]
    dae = oc["dae"]
    iwo = oc["iwo"]
    K = oc["cv_folds"]

    # AAA parameters
    I_A = aaa["max_iterations"]
    P_A = aaa["population_size"]
    knn_k = aaa["knn_neighbors"]

    # IWO parameters
    I_W = iwo["max_iterations"]
    P_W = iwo["population_size"]
    s_max = iwo["smax"]

    # DAE parameters
    enc_layers = dae["encoder_layers"]
    h1 = enc_layers[0] if len(enc_layers) > 0 else 64
    h2 = enc_layers[1] if len(enc_layers) > 1 else 32
    E_pre = dae["pretrain_epochs"]
    E_ft = dae["finetune_epochs"]
    B = dae["batch_size"]

    n_fold = int(n * (1 - oc.get("test_size", 0.20)))
    n_train = int(n_fold * (1 - oc.get("val_size", 0.15)))
    n_val = n_fold - n_train

    # AAA: each colony evaluation = KNN prediction = O(n_train * n_val * d)
    aaa_ops = I_A * P_A * n_train * n_val * d

    # IWO: each weed evaluation = train DAE + evaluate
    dae_train_ops = (E_pre + E_ft) * (n_train / B) * (d * h1 + h1 * h2)
    iwo_ops = I_W * (P_W + P_W * s_max) * dae_train_ops

    # Final DAE training
    final_dae = dae_train_ops

    # Per fold total
    per_fold = aaa_ops + iwo_ops + final_dae

    train_ops = K * per_fold
    infer_ops = d * h1 + h1 * h2 + h2 * 64 + 64  # encoder + classifier head

    return {
        "train_ops": train_ops,
        "infer_ops_per_sample": infer_ops,
        "formula_train": f"O(K·(I_A·P_A·n²·d + I_W·P_W·s_max·E·(n/B)·d·h1))",
        "formula_infer": f"O(d·h1 + h1·h2)",
        "big_o_train": "O(K·I_A·P_A·n²·d)",
        "big_o_infer": "O(d·h₁ + h₁·h₂)",
        "dominant_factor": "AAA's KNN evaluation (n² term)",
        "params": {
            "K": K, "I_A": I_A, "P_A": P_A,
            "I_W": I_W, "P_W": P_W, "s_max": s_max,
            "h1": h1, "h2": h2, "E_pre": E_pre, "E_ft": E_ft,
            "B": B, "n": n, "d": d, "knn_k": knn_k,
        },
        "ops_breakdown": {
            "aaa_feature_selection": K * aaa_ops,
            "iwo_optimization": K * iwo_ops,
            "final_dae_training": K * final_dae,
        },
    }


def complexity_pso_xgboost(n, d, cfg):
    """PSO-XGBoost: PSO hyperparameter search + XGBoost linear booster."""
    pc = cfg["pso_xgboost"]
    pso = pc["pso"]

    I_P = pso["max_iterations"]
    P = pso["num_particles"]
    k = pc["cv_folds"]
    R_max = pso["search_space"]["nrounds"][1]  # worst-case boosting rounds

    # PSO: each particle evaluation = k-fold CV with XGBoost
    # Each XGBoost training: R rounds × O(n × d) for linear booster
    pso_ops = I_P * P * k * R_max * n * d

    # Final model training
    final_train = R_max * n * d

    # Final CV evaluation
    final_cv = k * R_max * n * d

    train_ops = pso_ops + final_train + final_cv
    infer_ops = d  # linear model: dot product

    return {
        "train_ops": train_ops,
        "infer_ops_per_sample": infer_ops,
        "formula_train": f"O(I_P·P·k·R·n·d) = O({I_P}·{P}·{k}·{R_max}·n·{d})",
        "formula_infer": f"O(d) = O({d})",
        "big_o_train": "O(I_P·P·k·R·n·d)",
        "big_o_infer": "O(d)",
        "dominant_factor": "PSO iterations × CV folds × boosting rounds",
        "params": {
            "I_P": I_P, "P": P, "k": k, "R_max": R_max,
            "n": n, "d": d,
        },
    }


def complexity_bigru(n, d, cfg):
    """BiGRU-Attention: character-level BiGRU with attention."""
    bc = cfg["bigru"]

    L = bc["char_max_len"]           # 150
    e = bc["embedding_dim"]          # 128
    h = bc["gru_units"]              # 60
    a = bc["attention_size"]         # 80
    E = bc["max_epochs"]             # 20
    B = bc["batch_size"]             # 256

    # BiGRU: 2 directions × 3 gates × L timesteps × h × (h + e)
    bigru_ops = L * 3 * h * (h + e) * 2  # GRU has 3 gates (vs LSTM's 4)
    attn_ops = L * h * a + L * a          # tanh projection + score computation
    dense_ops = 2 * h                     # output layer

    per_sample = bigru_ops + attn_ops + dense_ops

    train_ops = E * n * per_sample
    infer_ops = per_sample

    return {
        "train_ops": train_ops,
        "infer_ops_per_sample": infer_ops,
        "formula_train": f"O(E·n·L·h·(h+e)) = O({E}·n·{L}·{h}·{h + e})",
        "formula_infer": f"O(L·h·(h+e)) = O({L}·{h}·{h + e})",
        "big_o_train": "O(E·n·L·h·(h+e))",
        "big_o_infer": "O(L·h·(h+e))",
        "dominant_factor": "GRU sequential computation over L timesteps",
        "params": {
            "E": E, "B": B, "L": L, "e": e, "h": h, "a": a,
            "n": n, "d": d,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Run All Complexity Analyses
# ══════════════════════════════════════════════════════════════════════════════

def compute_all(cfg, n, d):
    """Compute complexity for all pipelines given sample count n and feature dim d."""
    return {
        "path_a": complexity_path_a(n, d, cfg),
        "path_b": complexity_path_b(n, d, cfg),
        "adaptive_hybrid": complexity_adaptive(n, d, cfg),
        "odae_wpdc": complexity_odae_wpdc(n, d, cfg),
        "pso_xgboost": complexity_pso_xgboost(n, d, cfg),
        "bigru_attention": complexity_bigru(n, d, cfg),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Console Summary
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results, n, d):
    """Print a formatted table of Big O results to the console."""
    sep = "=" * 100
    print(f"\n{sep}")
    print(f"  COMPUTATIONAL COMPLEXITY ANALYSIS  (n = {n:,}  samples,  d = {d}  features)")
    print(sep)

    for key in MODEL_ORDER:
        r = results[key]
        label = MODEL_LABELS[key].replace("\n", " ")
        print(f"\n  ┌─ {label}")
        print(f"  │  Training:   {r['big_o_train']}")
        print(f"  │  Inference:  {r['big_o_infer']}  per sample")
        print(f"  │  Dominant:   {r['dominant_factor']}")
        print(f"  │  Concrete:   {r['formula_train']}")
        print(f"  │  Train ops:  {r['train_ops']:,.0f}")
        print(f"  │  Infer ops:  {r['infer_ops_per_sample']:,.0f}  per sample")
        print(f"  └{'─' * 70}")

    print(f"\n{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart Helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_fig(fig, name):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    path = os.path.join(CHARTS_DIR, f"{name}.{FIG_FORMAT}")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def format_ops(val):
    """Human-readable operation count."""
    if val >= 1e15:
        return f"{val:.2e}"
    if val >= 1e12:
        return f"{val / 1e12:.1f}T"
    if val >= 1e9:
        return f"{val / 1e9:.1f}B"
    if val >= 1e6:
        return f"{val / 1e6:.1f}M"
    if val >= 1e3:
        return f"{val / 1e3:.1f}K"
    return f"{val:.0f}"


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 1: Training Complexity Bar Chart (log scale)
# ══════════════════════════════════════════════════════════════════════════════

def chart_training_complexity_bar(results, n):
    fig, ax = plt.subplots(figsize=(10, 6))

    keys = MODEL_ORDER
    labels = [MODEL_LABELS_SHORT[k] for k in keys]
    ops = [results[k]["train_ops"] for k in keys]
    colors = [COLORS[k] for k in keys]

    bars = ax.bar(labels, ops, color=colors, edgecolor="white", linewidth=0.8, width=0.6)

    ax.set_yscale("log")
    ax.set_ylabel("Training Operations (log scale)")
    ax.set_title(f"Training Computational Complexity (n = {n:,})")

    # Add value labels on bars
    for bar, val in zip(bars, ops):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.5,
                format_ops(val), ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(bottom=min(ops) * 0.1, top=max(ops) * 20)
    fig.tight_layout()
    save_fig(fig, "01_training_complexity_bar")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 2: Inference Complexity Bar Chart
# ══════════════════════════════════════════════════════════════════════════════

def chart_inference_complexity_bar(results):
    fig, ax = plt.subplots(figsize=(10, 6))

    keys = MODEL_ORDER
    labels = [MODEL_LABELS_SHORT[k] for k in keys]
    ops = [results[k]["infer_ops_per_sample"] for k in keys]
    colors = [COLORS[k] for k in keys]

    bars = ax.bar(labels, ops, color=colors, edgecolor="white", linewidth=0.8, width=0.6)

    ax.set_yscale("log")
    ax.set_ylabel("Operations per Sample (log scale)")
    ax.set_title("Inference Computational Complexity (per sample)")

    for bar, val in zip(bars, ops):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.5,
                format_ops(val), ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(bottom=min(ops) * 0.1, top=max(ops) * 20)
    fig.tight_layout()
    save_fig(fig, "02_inference_complexity_bar")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 3: Scalability Curves — Training ops vs n
# ══════════════════════════════════════════════════════════════════════════════

def chart_scalability_curves(cfg, d, n_ref):
    n_values = np.linspace(1000, n_ref * 3, 200).astype(int)

    fig, ax = plt.subplots(figsize=(11, 7))

    for key in MODEL_ORDER:
        ops_list = []
        for n_i in n_values:
            r = compute_all(cfg, int(n_i), d)
            ops_list.append(r[key]["train_ops"])
        ax.plot(n_values, ops_list, label=MODEL_LABELS_SHORT[key],
                color=COLORS[key], linewidth=2)

    ax.set_yscale("log")
    ax.set_xlabel("Number of Training Samples (n)")
    ax.set_ylabel("Training Operations (log scale)")
    ax.set_title(f"Training Complexity Scalability (d = {d} features)")

    # Mark reference n
    ax.axvline(x=n_ref, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(n_ref, ax.get_ylim()[0], f"  n={n_ref:,}", va="bottom", fontsize=9, color="gray")

    ax.legend(loc="upper left", framealpha=0.9)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}k"))
    fig.tight_layout()
    save_fig(fig, "03_scalability_curves")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 4: Complexity Breakdown — Stacked bar for multi-phase pipelines
# ══════════════════════════════════════════════════════════════════════════════

def chart_complexity_breakdown(results):
    """Stacked bar showing phase breakdown for ODAE-WPDC and Adaptive Hybrid."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # ── Adaptive Hybrid breakdown ──
    ax = axes[0]
    ad = results["adaptive_hybrid"]["params"]
    phases = ["Path A\nTraining", "Path B\nTraining", "Adaptive\nOverhead"]
    values = [ad["path_a_ops"], ad["path_b_ops"], ad["adaptive_overhead"]]
    cols = [COLORS["path_a"], COLORS["path_b"], COLORS["adaptive_hybrid"]]

    bars = ax.bar(phases, values, color=cols, edgecolor="white", width=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Operations (log scale)")
    ax.set_title("Adaptive Hybrid — Phase Breakdown")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.8,
                format_ops(v), ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(bottom=min(values) * 0.05, top=max(values) * 50)

    # ── ODAE-WPDC breakdown ──
    ax = axes[1]
    od = results["odae_wpdc"].get("ops_breakdown", {})
    if od:
        phases = ["AAA Feature\nSelection", "IWO\nOptimization", "Final DAE\nTraining"]
        values = [od["aaa_feature_selection"], od["iwo_optimization"], od["final_dae_training"]]
    else:
        phases = ["Total"]
        values = [results["odae_wpdc"]["train_ops"]]
    cols = ["#E91E63", "#F48FB1", "#FCE4EC"]

    bars = ax.bar(phases, values, color=cols[:len(phases)], edgecolor="white", width=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Operations (log scale)")
    ax.set_title("ODAE-WPDC — Phase Breakdown")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.8,
                format_ops(v), ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(bottom=min(values) * 0.05, top=max(values) * 50)

    fig.tight_layout()
    save_fig(fig, "04_complexity_breakdown")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 5: Path B Per-Sample Operation Breakdown (stacked horizontal bar)
# ══════════════════════════════════════════════════════════════════════════════

def chart_path_b_breakdown(results):
    bd = results["path_b"].get("per_sample_breakdown")
    if not bd:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    component_labels = {
        "char_conv1": "Char Conv1D #1",
        "char_conv2": "Char Conv1D #2",
        "char_bilstm": "Char BiLSTM",
        "char_attention": "Char Attention",
        "token_bilstm": "Token BiLSTM",
        "token_attention": "Token Attention",
        "tabular_dense": "Tabular Dense",
        "merge_dense": "Merge Dense",
    }

    ordered = ["char_conv1", "char_conv2", "char_bilstm", "char_attention",
               "token_bilstm", "token_attention", "tabular_dense", "merge_dense"]
    labels = [component_labels[k] for k in ordered]
    vals = [bd[k] for k in ordered]

    cmap = plt.cm.YlOrBr(np.linspace(0.25, 0.85, len(vals)))

    left = 0
    for i, (label, val) in enumerate(zip(labels, vals)):
        ax.barh(0, val, left=left, color=cmap[i], edgecolor="white", height=0.5, label=label)
        if val / sum(vals) > 0.05:
            ax.text(left + val / 2, 0, f"{val / sum(vals) * 100:.0f}%",
                    ha="center", va="center", fontsize=8, fontweight="bold")
        left += val

    ax.set_yticks([])
    ax.set_xlabel("Operations per Sample")
    ax.set_title("Path B (CNN-BiLSTM) — Per-Sample Operation Breakdown")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: format_ops(x)))
    fig.tight_layout()
    save_fig(fig, "05_path_b_breakdown")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 6: Summary Table as Figure
# ══════════════════════════════════════════════════════════════════════════════

def chart_summary_table(results, n, d):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    headers = ["Pipeline", "Training Big O", "Inference Big O",
               "Train Ops (concrete)", "Infer Ops/Sample", "Dominant Factor"]

    rows = []
    for key in MODEL_ORDER:
        r = results[key]
        rows.append([
            MODEL_LABELS_SHORT[key],
            r["big_o_train"],
            r["big_o_infer"],
            format_ops(r["train_ops"]),
            format_ops(r["infer_ops_per_sample"]),
            r["dominant_factor"],
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="center", colColours=["#E0E0E0"] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Color the pipeline name cells
    for i, key in enumerate(MODEL_ORDER):
        cell = table[i + 1, 0]
        cell.set_facecolor(COLORS[key])
        cell.set_text_props(color="white", fontweight="bold")

    ax.set_title(f"Computational Complexity Summary  (n = {n:,},  d = {d})", fontsize=13, pad=20)
    fig.tight_layout()
    save_fig(fig, "06_summary_table")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 7: Training vs Inference Scatter Plot
# ══════════════════════════════════════════════════════════════════════════════

def chart_train_vs_inference_scatter(results, n):
    fig, ax = plt.subplots(figsize=(9, 7))

    for key in MODEL_ORDER:
        r = results[key]
        ax.scatter(r["train_ops"], r["infer_ops_per_sample"],
                   color=COLORS[key], s=200, zorder=5, edgecolors="black", linewidth=0.5)
        ax.annotate(MODEL_LABELS_SHORT[key],
                    (r["train_ops"], r["infer_ops_per_sample"]),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=10, fontweight="bold", color=COLORS[key])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training Operations (log scale)")
    ax.set_ylabel("Inference Operations per Sample (log scale)")
    ax.set_title(f"Training Cost vs. Inference Cost (n = {n:,})")
    fig.tight_layout()
    save_fig(fig, "07_train_vs_inference_scatter")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 8: Relative Complexity Radar Chart
# ══════════════════════════════════════════════════════════════════════════════

def chart_radar(results):
    """Radar chart comparing normalized training & inference complexity."""
    categories = ["Training\nComplexity", "Inference\nComplexity",
                   "Scalability\n(1/n² term)", "Param Count\nProxy"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    # Normalize all on 0-1 scale (1 = worst/most expensive)
    train_ops = {k: results[k]["train_ops"] for k in MODEL_ORDER}
    infer_ops = {k: results[k]["infer_ops_per_sample"] for k in MODEL_ORDER}
    max_train = max(train_ops.values())
    max_infer = max(infer_ops.values())

    # Scalability: 1.0 if has n² term, else proportional
    scalability_score = {
        "path_a": 0.3,
        "path_b": 0.5,
        "adaptive_hybrid": 0.5,
        "odae_wpdc": 1.0,   # n² from KNN
        "pso_xgboost": 0.4,
        "bigru_attention": 0.4,
    }

    # Parameter count proxy (from architecture complexity)
    param_proxy = {
        "path_a": 0.6,
        "path_b": 0.9,
        "adaptive_hybrid": 1.0,
        "odae_wpdc": 0.4,
        "pso_xgboost": 0.2,
        "bigru_attention": 0.5,
    }

    for key in MODEL_ORDER:
        values = [
            train_ops[key] / max_train,
            infer_ops[key] / max_infer,
            scalability_score[key],
            param_proxy[key],
        ]
        values += values[:1]
        ax.plot(angles, values, color=COLORS[key], linewidth=2, label=MODEL_LABELS_SHORT[key])
        ax.fill(angles, values, color=COLORS[key], alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("Complexity Profile Comparison (Normalized)", y=1.08, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    save_fig(fig, "08_complexity_radar")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  Computational Complexity (Big O) Analysis")
    print("=" * 60)

    cfg = load_config()
    d = len(cfg["feature_columns"])  # 53

    # Determine n from the primary dataset
    dataset_id = cfg["adaptive"]["dataset_id"]
    n, _ = get_dataset_size(cfg, dataset_id)
    if n is None:
        print(f"  Could not read dataset {dataset_id}, using default n=10,000")
        n = 10_000

    print(f"\n  Dataset:  {dataset_id}")
    print(f"  Samples:  {n:,}")
    print(f"  Features: {d}")

    # Compute complexities
    results = compute_all(cfg, n, d)

    # Console output
    print_summary(results, n, d)

    # Generate all charts
    os.makedirs(CHARTS_DIR, exist_ok=True)
    print("  Generating charts...")
    chart_training_complexity_bar(results, n)
    chart_inference_complexity_bar(results)
    chart_scalability_curves(cfg, d, n)
    chart_complexity_breakdown(results)
    chart_path_b_breakdown(results)
    chart_summary_table(results, n, d)
    chart_train_vs_inference_scatter(results, n)
    chart_radar(results)

    # Save raw data as JSON
    export = {}
    for key in MODEL_ORDER:
        r = results[key]
        export[key] = {
            "big_o_train": r["big_o_train"],
            "big_o_infer": r["big_o_infer"],
            "train_ops": r["train_ops"],
            "infer_ops_per_sample": r["infer_ops_per_sample"],
            "dominant_factor": r["dominant_factor"],
            "formula_train": r["formula_train"],
            "formula_infer": r["formula_infer"],
            "params": r["params"],
        }
    json_path = os.path.join(CHARTS_DIR, "complexity_data.json")
    with open(json_path, "w") as f:
        json.dump({"n": n, "d": d, "dataset_id": dataset_id, "results": export}, f, indent=2)
    print(f"  Saved: {json_path}")

    print(f"\n  All charts saved to: {CHARTS_DIR}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
