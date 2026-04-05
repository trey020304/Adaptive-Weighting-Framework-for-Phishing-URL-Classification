"""
Thesis Results Visualization Generator
======================================
Generates all charts from adaptive_hybrid results:
  - Path A (Chi-Square + Random Forest)
  - Path B (CNN-BiLSTM)
  - Static Ensemble (0.5/0.5)
  - Adaptive Hybrid Ensemble (proposed framework)

Output: results/charts/ directory with publication-ready PNG figures.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = "results"
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")
DPI = 300
FIG_FORMAT = "png"

# Thesis-quality style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Color palette
COLORS = {
    "path_a": "#2196F3",         # Blue
    "path_b": "#FF9800",         # Orange
    "static_ensemble": "#9E9E9E", # Gray
    "adaptive_hybrid": "#4CAF50", # Green
    "odae_wpdc": "#E91E63",      # Pink
    "pso_xgboost": "#9C27B0",    # Purple
    "bigru_attention": "#00BCD4", # Cyan
}

MODEL_LABELS = {
    "path_a": "Path A (Chi² + RF)",
    "path_b": "Path B (CNN-BiLSTM)",
    "static_ensemble": "Static Ensemble (0.5/0.5)",
    "adaptive_hybrid": "Adaptive Hybrid (Proposed)",
    "odae_wpdc": "ODAE-WPDC",
    "pso_xgboost": "PSO-XGBoost",
    "bigru_attention": "BiGRU-Attention",
}

DATASET_LABELS = {
    "3": "Dataset 3 (Kaggle)",
    "7": "Dataset 7 (Kaggle)",
    "8": "Dataset 8 (Kaggle)",
    "9": "Dataset 9 (Mendeley)",
    "11": "Dataset 11 (Kaggle)",
    "12": "Dataset 12 (Kaggle)",
    "c1": "Dataset C1 (Mendeley)",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_adaptive_results():
    """Load all adaptive_hybrid results.json files."""
    results = {}
    base = os.path.join(RESULTS_DIR, "adaptive_hybrid")
    if not os.path.isdir(base):
        return results
    for d in os.listdir(base):
        rpath = os.path.join(base, d, "results.json")
        if os.path.isfile(rpath):
            with open(rpath) as f:
                data = json.load(f)
            dataset_id = data.get("config", {}).get("dataset_id", d.replace("dataset_", ""))
            results[dataset_id] = data
    return results


def load_baseline_results():
    """Load all baseline model results (ODAE-WPDC, PSO-XGBoost, BiGRU-Attention)."""
    baselines = {}  # {dataset_id: {model_key: {metric_name: value}}}

    # ── ODAE-WPDC ──
    odae_base = os.path.join(RESULTS_DIR, "odae_wpdc")
    if os.path.isdir(odae_base):
        for d in os.listdir(odae_base):
            rpath = os.path.join(odae_base, d, "results.json")
            if os.path.isfile(rpath):
                with open(rpath) as f:
                    data = json.load(f)
                did = data.get("dataset_id", d.replace("dataset_", ""))
                pred = data.get("predictive_metrics", {})
                comp = data.get("computational_metrics", {})
                baselines.setdefault(did, {})["odae_wpdc"] = {
                    "accuracy": pred.get("accuracy", {}).get("mean"),
                    "precision": pred.get("precision", {}).get("mean"),
                    "recall": pred.get("recall", {}).get("mean"),
                    "f1": pred.get("f1", {}).get("mean"),
                    "roc_auc": pred.get("roc_auc", {}).get("mean"),
                    "confusion_matrix_per_fold": pred.get("confusion_matrix_per_fold"),
                    "training_time_s": comp.get("training_time_s", {}).get("mean"),
                    "inference_ms_per_sample": comp.get("inference_ms_per_sample", {}).get("mean"),
                    "model_size_mb": comp.get("model_size_mb", {}).get("mean"),
                    "total_pipeline_time_s": data.get("total_pipeline_time_s"),
                    "_raw": data,
                }

    # ── PSO-XGBoost ──
    pso_base = os.path.join(RESULTS_DIR, "pso_xgboost")
    if os.path.isdir(pso_base):
        for d in os.listdir(pso_base):
            rpath = os.path.join(pso_base, d, "results.json")
            if os.path.isfile(rpath):
                with open(rpath) as f:
                    data = json.load(f)
                did = data.get("dataset_id", d.replace("dataset_", ""))
                test = data.get("test_metrics", {})
                comp = data.get("computational_metrics", {})
                baselines.setdefault(did, {})["pso_xgboost"] = {
                    "accuracy": test.get("accuracy"),
                    "precision": test.get("precision"),
                    "recall": test.get("recall"),
                    "f1": test.get("f1"),
                    "roc_auc": test.get("roc_auc"),
                    "confusion_matrix": test.get("confusion_matrix"),
                    "training_time_s": comp.get("train_time_s"),
                    "inference_ms_per_sample": comp.get("inference_ms_per_sample"),
                    "model_size_mb": comp.get("model_size_mb"),
                    "total_pipeline_time_s": data.get("total_pipeline_time_s"),
                    "pso_time_s": comp.get("pso_time_s"),
                    "_raw": data,
                }

    # ── BiGRU-Attention ──
    bigru_base = os.path.join(RESULTS_DIR, "bigru")
    if os.path.isdir(bigru_base):
        for variant in os.listdir(bigru_base):
            variant_path = os.path.join(bigru_base, variant)
            if not os.path.isdir(variant_path):
                continue
            for d in os.listdir(variant_path):
                rpath = os.path.join(variant_path, d, "results.json")
                if os.path.isfile(rpath):
                    with open(rpath) as f:
                        data = json.load(f)
                    did = data.get("dataset_id", d.replace("dataset_", ""))
                    test = data.get("test_metrics", {})
                    comp = data.get("computational_metrics", {})
                    baselines.setdefault(did, {})["bigru_attention"] = {
                        "accuracy": test.get("accuracy"),
                        "precision": test.get("precision"),
                        "recall": test.get("recall"),
                        "f1": test.get("f1"),
                        "roc_auc": test.get("roc_auc"),
                        "confusion_matrix": test.get("confusion_matrix"),
                        "training_time_s": comp.get("training_time_s"),
                        "inference_ms_per_sample": comp.get("inference_ms_per_sample"),
                        "model_size_mb": comp.get("model_size_mb"),
                        "total_pipeline_time_s": data.get("total_pipeline_time_s"),
                        "training_history": data.get("training_history"),
                        "_raw": data,
                    }

    return baselines


def save_fig(fig, name):
    """Save figure to charts directory."""
    path = os.path.join(CHARTS_DIR, f"{name}.{FIG_FORMAT}")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 1: Model Performance Comparison (Grouped Bar Chart)
# ══════════════════════════════════════════════════════════════════════════════

def chart_model_comparison(adaptive_results):
    """Grouped bar chart comparing all models across key metrics."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

    for dataset_id, data in adaptive_results.items():
        models = {}
        for model_key in ["path_a", "path_b", "static_ensemble", "adaptive_hybrid"]:
            if model_key in data["metrics"]:
                models[model_key] = data["metrics"][model_key]

        n_metrics = len(metrics)
        n_models = len(models)
        x = np.arange(n_metrics)
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, (model_key, model_data) in enumerate(models.items()):
            vals = [model_data.get(m, 0) * 100 for m in metrics]
            bars = ax.bar(
                x + i * width - (n_models - 1) * width / 2,
                vals, width,
                label=MODEL_LABELS.get(model_key, model_key),
                color=COLORS.get(model_key, "#666"),
                edgecolor="white", linewidth=0.5,
            )
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{val:.2f}%", ha="center", va="bottom", fontsize=7, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel("Score (%)")
        ax.set_title(f"Model Performance Comparison — {DATASET_LABELS.get(dataset_id, dataset_id)}")
        ax.set_ylim(bottom=min(90, min(v for m in models.values() for v in [m.get(met, 1) * 100 for met in metrics]) - 2))
        ax.legend(loc="lower right", framealpha=0.9)
        fig.tight_layout()
        save_fig(fig, f"01_model_comparison_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 2: Log-Loss Comparison
# ══════════════════════════════════════════════════════════════════════════════

def chart_log_loss_comparison(adaptive_results):
    """Bar chart of log-loss across models — lower is better."""
    for dataset_id, data in adaptive_results.items():
        models = {}
        for model_key in ["path_a", "path_b", "static_ensemble", "adaptive_hybrid"]:
            if model_key in data["metrics"] and "log_loss" in data["metrics"][model_key]:
                models[model_key] = data["metrics"][model_key]["log_loss"]

        if not models:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        names = [MODEL_LABELS.get(k, k) for k in models]
        vals = list(models.values())
        colors = [COLORS.get(k, "#666") for k in models]

        bars = ax.barh(names, vals, color=colors, edgecolor="white", height=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=10)

        ax.set_xlabel("Log-Loss (lower is better)")
        ax.set_title(f"Log-Loss Comparison — {DATASET_LABELS.get(dataset_id, dataset_id)}")
        ax.invert_yaxis()
        fig.tight_layout()
        save_fig(fig, f"02_log_loss_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 3: Adaptive Weight Evolution (α/β over epochs)
# ══════════════════════════════════════════════════════════════════════════════

def chart_weight_evolution(adaptive_results):
    """Shows how α (Path A) and β (Path B) evolve during training."""
    for dataset_id, data in adaptive_results.items():
        wh = data.get("weight_history")
        if not wh:
            continue

        epochs = [w["epoch"] for w in wh]
        alphas = [w["alpha"] for w in wh]
        betas = [w["beta"] for w in wh]
        L_A = [w["L_A"] for w in wh]
        L_B = [w["L_B"] for w in wh]

        best_cfg = data.get("config", {}).get("best_epoch_weights", {})
        best_epoch = best_cfg.get("epoch")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Weights
        ax1.fill_between(epochs, 0, alphas, alpha=0.3, color=COLORS["path_a"])
        ax1.fill_between(epochs, alphas, 1, alpha=0.3, color=COLORS["path_b"])
        ax1.plot(epochs, alphas, "o-", color=COLORS["path_a"], label="α (Path A weight)", markersize=5)
        ax1.plot(epochs, betas, "s-", color=COLORS["path_b"], label="β (Path B weight)", markersize=5)
        if best_epoch:
            ax1.axvline(best_epoch, color=COLORS["adaptive_hybrid"], linestyle="--",
                        linewidth=2, label=f"Best Epoch ({best_epoch})")
        ax1.set_ylabel("Weight")
        ax1.set_title("Adaptive Gating Weights Over Epochs")
        ax1.set_ylim(0, 1)
        ax1.legend(loc="upper right")
        ax1.axhline(0.5, color="gray", linestyle=":", alpha=0.5)

        # Validation losses
        ax2.plot(epochs, L_A, "o-", color=COLORS["path_a"], label="$L_A$ (Path A val loss)", markersize=5)
        ax2.plot(epochs, L_B, "s-", color=COLORS["path_b"], label="$L_B$ (Path B val loss)", markersize=5)
        combined = [a * la + b * lb for a, la, b, lb in zip(alphas, L_A, betas, L_B)]
        ax2.plot(epochs, combined, "D--", color=COLORS["adaptive_hybrid"],
                 label="Combined loss ($\\alpha L_A + \\beta L_B$)", markersize=5)
        if best_epoch:
            ax2.axvline(best_epoch, color=COLORS["adaptive_hybrid"], linestyle="--", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation Loss")
        ax2.set_title("Validation Losses Driving the Gating Mechanism")
        ax2.legend()

        fig.suptitle(
            f"Adaptive Weight Evolution — {DATASET_LABELS.get(dataset_id, dataset_id)}\n"
            f"γ = {data['config'].get('gamma', '?')}  |  "
            f"Final: α = {best_cfg.get('alpha', 0):.4f}, β = {best_cfg.get('beta', 0):.4f}",
            fontsize=13, y=1.02,
        )
        fig.tight_layout()
        save_fig(fig, f"03_weight_evolution_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 4: Confusion Matrices (Side-by-Side Heatmaps)
# ══════════════════════════════════════════════════════════════════════════════

def chart_confusion_matrices(adaptive_results):
    """Side-by-side normalized confusion matrix heatmaps for all models."""
    labels = ["Non-Phishing", "Phishing"]

    for dataset_id, data in adaptive_results.items():
        model_keys = [k for k in ["path_a", "path_b", "static_ensemble", "adaptive_hybrid"]
                      if k in data["metrics"] and "confusion_matrix" in data["metrics"][k]]

        n_models = len(model_keys)
        if n_models == 0:
            continue

        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
        if n_models == 1:
            axes = [axes]

        for i, model_key in enumerate(model_keys):
            cm = np.array(data["metrics"][model_key]["confusion_matrix"])
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

            sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                        xticklabels=labels, yticklabels=labels,
                        ax=axes[i], vmin=0, vmax=1, cbar=False)
            # Add raw counts as secondary annotation
            for r in range(2):
                for c in range(2):
                    axes[i].text(c + 0.5, r + 0.72, f"(n={cm[r, c]})",
                                 ha="center", va="center", fontsize=8, color="gray")
            axes[i].set_title(MODEL_LABELS.get(model_key, model_key), fontsize=10)
            axes[i].set_ylabel("True Label" if i == 0 else "")
            axes[i].set_xlabel("Predicted Label")

        fig.suptitle(f"Confusion Matrices (Normalized) — {DATASET_LABELS.get(dataset_id, dataset_id)}",
                     fontsize=14, y=1.05)
        fig.tight_layout()
        save_fig(fig, f"04_confusion_matrices_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 5: Computational Efficiency Comparison
# ══════════════════════════════════════════════════════════════════════════════

def chart_computational_efficiency(adaptive_results):
    """3-panel: training time, inference latency, model size."""
    for dataset_id, data in adaptive_results.items():
        comp = data.get("computational", {})
        if not comp:
            continue

        models = {
            "path_a": {
                "train_time": comp.get("path_a_train_time_s", 0),
                "inference_ms": comp.get("path_a_inference_ms", 0),
                "model_size": comp.get("path_a_model_size_mb", 0),
            },
            "path_b": {
                "train_time": comp.get("path_b_train_time_s", 0),
                "inference_ms": comp.get("path_b_inference_ms", 0),
                "model_size": comp.get("path_b_model_size_mb", 0),
            },
            "adaptive_hybrid": {
                "train_time": comp.get("adaptive_train_time_s", 0),
                "inference_ms": comp.get("adaptive_inference_ms", 0),
                "model_size": comp.get("adaptive_model_size_mb", 0),
            },
        }

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

        names = [MODEL_LABELS.get(k, k) for k in models]
        colors = [COLORS.get(k, "#666") for k in models]

        # Training time
        vals = [m["train_time"] for m in models.values()]
        bars = ax1.bar(range(len(names)), vals, color=colors, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                     f"{val:.1f}s", ha="center", va="bottom", fontsize=9)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax1.set_ylabel("Seconds")
        ax1.set_title("Training Time")

        # Inference latency
        vals = [m["inference_ms"] for m in models.values()]
        bars = ax2.bar(range(len(names)), vals, color=colors, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax2.set_ylabel("ms / sample")
        ax2.set_title("Inference Latency")

        # Model size
        vals = [m["model_size"] for m in models.values()]
        bars = ax3.bar(range(len(names)), vals, color=colors, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=9)
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax3.set_ylabel("MB")
        ax3.set_title("Model Size")

        fig.suptitle(f"Computational Efficiency — {DATASET_LABELS.get(dataset_id, dataset_id)}",
                     fontsize=14, y=1.02)
        fig.tight_layout()
        save_fig(fig, f"05_computational_efficiency_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 6: Efficiency-Accuracy Tradeoff Scatter
# ══════════════════════════════════════════════════════════════════════════════

def chart_efficiency_accuracy_tradeoff(adaptive_results):
    """Scatter: x = training time (log scale), y = F1 score."""
    for dataset_id, data in adaptive_results.items():
        comp = data.get("computational", {})
        met = data.get("metrics", {})

        points = []
        for model_key, prefix in [("path_a", "path_a"), ("path_b", "path_b"),
                                   ("adaptive_hybrid", "adaptive")]:
            f1 = met.get(model_key, {}).get("f1", 0)
            tt = comp.get(f"{prefix}_train_time_s", 0)
            inf = comp.get(f"{prefix}_inference_ms", 0)
            if f1 and tt:
                points.append((model_key, tt, f1, inf))

        # Static ensemble has same training time as adaptive
        if "static_ensemble" in met:
            f1 = met["static_ensemble"]["f1"]
            tt = comp.get("adaptive_train_time_s", 0)
            inf = comp.get("adaptive_inference_ms", 0)
            points.append(("static_ensemble", tt, f1, inf))

        if not points:
            continue

        fig, ax = plt.subplots(figsize=(9, 6))
        for model_key, tt, f1, inf in points:
            ax.scatter(tt, f1,
                       s=max(80, inf * 200),  # Bubble size = inference time
                       c=COLORS.get(model_key, "#666"),
                       label=MODEL_LABELS.get(model_key, model_key),
                       edgecolors="black", linewidth=0.5, alpha=0.8, zorder=5)
            ax.annotate(f"F1={f1:.4f}\n{tt:.1f}s",
                        (tt, f1), textcoords="offset points",
                        xytext=(10, 5), fontsize=8)

        ax.set_xscale("log")
        ax.set_xlabel("Training Time (seconds, log scale)")
        ax.set_ylabel("F1 Score")
        ax.set_title(f"Efficiency–Accuracy Tradeoff — {DATASET_LABELS.get(dataset_id, dataset_id)}\n"
                     "(bubble size ∝ inference latency)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        save_fig(fig, f"06_efficiency_accuracy_tradeoff_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 7: Multi-Dataset Performance Summary (Radar Chart)
# ══════════════════════════════════════════════════════════════════════════════

def chart_radar_comparison(adaptive_results):
    """Radar/spider chart comparing models across metrics for each dataset."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

    for dataset_id, data in adaptive_results.items():
        models_data = {}
        for model_key in ["path_a", "path_b", "adaptive_hybrid"]:
            if model_key in data["metrics"]:
                models_data[model_key] = [data["metrics"][model_key].get(m, 0) * 100 for m in metrics]

        if not models_data:
            continue

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for model_key, vals in models_data.items():
            vals_closed = vals + vals[:1]
            ax.fill(angles, vals_closed, alpha=0.15, color=COLORS.get(model_key, "#666"))
            ax.plot(angles, vals_closed, "o-", color=COLORS.get(model_key, "#666"),
                    label=MODEL_LABELS.get(model_key, model_key), linewidth=2, markersize=5)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        # Adjust radial range for detail
        all_vals = [v for vals in models_data.values() for v in vals]
        min_val = max(90, min(all_vals) - 2)
        ax.set_ylim(min_val, 100.0)
        ax.set_title(f"Model Comparison Radar — {DATASET_LABELS.get(dataset_id, dataset_id)}",
                     y=1.08, fontsize=13)
        ax.legend(loc="lower right", bbox_to_anchor=(1.2, 0))
        fig.tight_layout()
        save_fig(fig, f"07_radar_comparison_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 8: Cross-Dataset Performance Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def chart_cross_dataset_heatmap(adaptive_results):
    """Heatmap showing performance across all datasets."""
    metrics = ["accuracy", "f1", "roc_auc", "log_loss"]
    metric_labels = ["Accuracy (%)", "F1 (%)", "ROC-AUC (%)", "Log-Loss"]
    pct_metrics = {"accuracy", "f1", "roc_auc"}
    model_keys = ["path_a", "path_b", "adaptive_hybrid"]

    dataset_ids = sorted(adaptive_results.keys())
    if len(dataset_ids) < 2:
        return

    for model_key in model_keys:
        matrix = []
        annot_matrix = []
        for dataset_id in dataset_ids:
            row = []
            annot_row = []
            for m in metrics:
                val = adaptive_results[dataset_id].get("metrics", {}).get(model_key, {}).get(m, np.nan)
                if m in pct_metrics:
                    row.append(val * 100 if not np.isnan(val) else val)
                    annot_row.append(f"{val * 100:.2f}%" if not np.isnan(val) else "N/A")
                else:
                    row.append(val)
                    annot_row.append(f"{val:.4f}" if not np.isnan(val) else "N/A")
            matrix.append(row)
            annot_matrix.append(annot_row)

        df = pd.DataFrame(matrix,
                          index=[DATASET_LABELS.get(d, d) for d in dataset_ids],
                          columns=metric_labels)

        fig, ax = plt.subplots(figsize=(8, max(3, len(dataset_ids) * 1.2)))
        sns.heatmap(df, annot=annot_matrix, fmt="", cmap="YlGnBu", ax=ax,
                    linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title(f"{MODEL_LABELS.get(model_key, model_key)} — Performance Across Datasets")
        fig.tight_layout()
        save_fig(fig, f"08_cross_dataset_heatmap_{model_key}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 9: Adaptive vs Static Ensemble Improvement
# ══════════════════════════════════════════════════════════════════════════════

def chart_adaptive_vs_static(adaptive_results):
    """Show improvement of adaptive over static ensemble and individual paths."""
    metrics = ["accuracy", "f1", "roc_auc", "log_loss"]
    metric_labels = ["Accuracy", "F1", "ROC-AUC", "Log-Loss"]

    datasets_with_both = [d for d in adaptive_results
                          if "static_ensemble" in adaptive_results[d].get("metrics", {})
                          and "adaptive_hybrid" in adaptive_results[d].get("metrics", {})]

    if not datasets_with_both:
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for i, (m, label) in enumerate(zip(metrics, metric_labels)):
        for dataset_id in datasets_with_both:
            data = adaptive_results[dataset_id]["metrics"]
            models_ordered = ["path_a", "path_b", "static_ensemble", "adaptive_hybrid"]
            scale = 1 if m == "log_loss" else 100
            vals = [data.get(mk, {}).get(m, 0) * scale for mk in models_ordered]
            x = np.arange(len(models_ordered))

            axes[i].plot(x, vals, "o-", label=DATASET_LABELS.get(dataset_id, dataset_id),
                         markersize=8, linewidth=2)

        axes[i].set_xticks(x)
        axes[i].set_xticklabels(
            ["Path A", "Path B", "Static\n(0.5/0.5)", "Adaptive\n(Proposed)"],
            fontsize=8,
        )
        axes[i].set_title(label)
        if m == "log_loss":
            axes[i].set_ylabel("Log-Loss (lower = better)")
        else:
            axes[i].set_ylabel("Score (%) (higher = better)")

    axes[0].legend(fontsize=8)
    fig.suptitle("Progression: Individual Paths → Static → Adaptive Ensemble", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "09_adaptive_vs_static_progression")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 10: Gamma Sensitivity Analysis
# ══════════════════════════════════════════════════════════════════════════════

def chart_gamma_comparison(adaptive_results):
    """Show which gamma was selected per dataset."""
    if len(adaptive_results) < 1:
        return

    gammas = {}
    f1s = {}
    for dataset_id, data in adaptive_results.items():
        g = data.get("config", {}).get("gamma")
        f1 = data.get("metrics", {}).get("adaptive_hybrid", {}).get("f1", 0)
        if g is not None:
            gammas[dataset_id] = g
            f1s[dataset_id] = f1

    if not gammas:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    dataset_ids = sorted(gammas.keys())
    x = np.arange(len(dataset_ids))
    labels = [DATASET_LABELS.get(d, d) for d in dataset_ids]

    bars = ax.bar(x, [gammas[d] for d in dataset_ids], color=COLORS["adaptive_hybrid"],
                  edgecolor="white", width=0.5)
    for bar, d in zip(bars, dataset_ids):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"γ={gammas[d]}\nF1={f1s[d]:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Selected Gamma (γ)")
    ax.set_title("Gamma Selection Per Dataset\n(selected via validation F1)")
    ax.set_ylim(0, max(gammas.values()) + 1.5)
    fig.tight_layout()
    save_fig(fig, "10_gamma_sensitivity")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 11: Weight Distribution (α/β final values across datasets)
# ══════════════════════════════════════════════════════════════════════════════

def chart_weight_distribution(adaptive_results):
    """Stacked bar showing final α/β per dataset."""
    dataset_ids = sorted(adaptive_results.keys())
    if not dataset_ids:
        return

    alphas = []
    betas = []
    for d in dataset_ids:
        cfg = adaptive_results[d].get("config", {}).get("best_epoch_weights", {})
        alphas.append(cfg.get("alpha", 0.5))
        betas.append(cfg.get("beta", 0.5))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(dataset_ids))
    labels = [DATASET_LABELS.get(d, d) for d in dataset_ids]

    ax.bar(x, alphas, label="α (Path A: Chi-Square + Random Forest)", color=COLORS["path_a"], edgecolor="white", width=0.5)
    ax.bar(x, betas, bottom=alphas, label="β (Path B: CNN-BiLSTM", color=COLORS["path_b"],
           edgecolor="white", width=0.5)

    for i, (a, b) in enumerate(zip(alphas, betas)):
        ax.text(i, a / 2, f"α={a:.3f}", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        ax.text(i, a + b / 2, f"β={b:.3f}", ha="center", va="center", fontsize=9, fontweight="bold", color="white")

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Equal weighting (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1.05)
    ax.set_title("Final Adaptive Weights Per Dataset")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_fig(fig, "11_weight_distribution")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 12: Convergence Rate Comparison
# ══════════════════════════════════════════════════════════════════════════════

def chart_convergence_rate(adaptive_results):
    """Compare convergence rate (best_epoch / total_epochs) across datasets."""
    records = []
    for dataset_id, data in adaptive_results.items():
        comp = data.get("computational", {})
        best_ep = comp.get("path_b_best_epoch")
        total_ep = comp.get("path_b_total_epochs")
        conv = comp.get("path_b_convergence_rate")
        if best_ep and total_ep:
            records.append({
                "dataset": DATASET_LABELS.get(dataset_id, dataset_id),
                "best_epoch": best_ep,
                "total_epochs": total_ep,
                "convergence_rate": conv or (best_ep / total_ep),
            })

    if len(records) < 1:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    df = pd.DataFrame(records)

    # Best epoch vs total epochs
    x = np.arange(len(df))
    ax1.bar(x, df["total_epochs"], color="lightgray", edgecolor="white", label="Total Epochs", width=0.5)
    ax1.bar(x, df["best_epoch"], color=COLORS["adaptive_hybrid"], edgecolor="white", label="Best Epoch", width=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["dataset"], rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("Epochs")
    ax1.set_title("Early Stopping: Best vs Total Epochs")
    ax1.legend()

    # Convergence rate
    bars = ax2.bar(x, df["convergence_rate"], color=COLORS["path_b"], edgecolor="white", width=0.5)
    for bar, val in zip(bars, df["convergence_rate"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.2%}", ha="center", va="bottom", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["dataset"], rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("Convergence Rate")
    ax2.set_title("Convergence Rate (best_epoch / total_epochs)")
    ax2.set_ylim(0, 1)

    fig.suptitle("Path B Convergence Analysis", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "12_convergence_rate")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 13: Summary Comparison Table (as figure for thesis)
# ══════════════════════════════════════════════════════════════════════════════

def chart_summary_table(adaptive_results):
    """Generate a publication-ready table as a figure."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"]
    metric_labels = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)", "ROC-AUC (%)", "Log-Loss"]
    pct_metrics = {"accuracy", "precision", "recall", "f1", "roc_auc"}

    for dataset_id, data in adaptive_results.items():
        model_keys = ["path_a", "path_b", "static_ensemble", "adaptive_hybrid"]
        model_names = [MODEL_LABELS.get(k, k) for k in model_keys]

        rows = []
        for mk in model_keys:
            row = []
            for m in metrics:
                val = data["metrics"].get(mk, {}).get(m, "N/A")
                if isinstance(val, float):
                    row.append(f"{val * 100:.2f}%" if m in pct_metrics else f"{val:.4f}")
                else:
                    row.append(str(val))
            rows.append(row)

        fig, ax = plt.subplots(figsize=(14, 1.5 + 0.5 * len(model_keys)))
        ax.axis("off")

        table = ax.table(
            cellText=rows,
            rowLabels=model_names,
            colLabels=metric_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.8)

        # Style header
        for j in range(len(metric_labels)):
            table[0, j].set_facecolor("#4CAF50")
            table[0, j].set_text_props(color="white", fontweight="bold")

        # Highlight best values per column
        for j, m in enumerate(metrics):
            col_vals = []
            for i, mk in enumerate(model_keys):
                try:
                    col_vals.append(float(rows[i][j].rstrip("%")))
                except (ValueError, IndexError):
                    col_vals.append(None)

            if any(v is not None for v in col_vals):
                valid_vals = [v for v in col_vals if v is not None]
                if m == "log_loss":
                    best_val = min(valid_vals)
                else:
                    best_val = max(valid_vals)
                for i, v in enumerate(col_vals):
                    if v is not None and abs(v - best_val) < 1e-8:
                        table[i + 1, j].set_text_props(fontweight="bold")
                        table[i + 1, j].set_facecolor("#E8F5E9")

        # Row colors
        row_colors = [COLORS.get(mk, "#FFFFFF") for mk in model_keys]
        for i, color in enumerate(row_colors):
            table[i + 1, -1].set_facecolor(color + "30")  # Very light tint

        ax.set_title(f"Performance Summary — {DATASET_LABELS.get(dataset_id, dataset_id)}",
                     fontsize=14, pad=20)
        fig.tight_layout()
        save_fig(fig, f"13_summary_table_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 14: Combined Loss Curve (Path A constant + Path B training)
# ══════════════════════════════════════════════════════════════════════════════

def chart_combined_loss_analysis(adaptive_results):
    """Overlay Path A's constant val loss with Path B's training val loss using weight_history."""
    for dataset_id, data in adaptive_results.items():
        wh = data.get("weight_history")
        if not wh:
            continue

        epochs = [w["epoch"] for w in wh]
        L_A_vals = [w["L_A"] for w in wh]
        L_B_vals = [w["L_B"] for w in wh]
        combined = [w["alpha"] * w["L_A"] + w["beta"] * w["L_B"] for w in wh]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.axhline(L_A_vals[0], color=COLORS["path_a"], linestyle="--", linewidth=2,
                    label=f"Path A Val Loss (constant: {L_A_vals[0]:.4f})")
        ax.plot(epochs, L_B_vals, "s-", color=COLORS["path_b"],
                label="Path B Val Loss", markersize=5, linewidth=2)
        ax.plot(epochs, combined, "D-", color=COLORS["adaptive_hybrid"],
                label="Weighted Combined Loss", markersize=6, linewidth=2)

        best_epoch = data.get("config", {}).get("best_epoch_weights", {}).get("epoch")
        if best_epoch:
            ax.axvline(best_epoch, color="red", linestyle=":", linewidth=1.5,
                       label=f"Best Combined Loss (epoch {best_epoch})")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Loss")
        ax.set_title(f"Loss Landscape Analysis — {DATASET_LABELS.get(dataset_id, dataset_id)}")
        ax.legend()
        fig.tight_layout()
        save_fig(fig, f"14_combined_loss_analysis_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 15: All Models Performance Comparison (Framework + Baselines)
# ══════════════════════════════════════════════════════════════════════════════

def chart_all_models_comparison(adaptive_results, baselines):
    """Grouped bar chart: framework components + all baselines on shared datasets."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

    # Find datasets that have at least one baseline
    common_datasets = sorted(set(adaptive_results.keys()) & set(baselines.keys()))
    if not common_datasets:
        print("  (skipped — no overlapping datasets between adaptive & baselines)")
        return

    for dataset_id in common_datasets:
        ad = adaptive_results[dataset_id]
        bl = baselines[dataset_id]

        # Collect all models: framework first, then baselines
        models = {}
        for mk in ["path_a", "path_b", "adaptive_hybrid"]:
            if mk in ad["metrics"]:
                models[mk] = {m: ad["metrics"][mk].get(m, 0) for m in metrics}
        for mk in ["odae_wpdc", "pso_xgboost", "bigru_attention"]:
            if mk in bl:
                models[mk] = {m: bl[mk].get(m, 0) for m in metrics}

        n_metrics = len(metrics)
        n_models = len(models)
        x = np.arange(n_metrics)
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(14, 6.5))

        for i, (mk, md) in enumerate(models.items()):
            vals = [md.get(m, 0) for m in metrics]
            bars = ax.bar(
                x + i * width - (n_models - 1) * width / 2,
                vals, width,
                label=MODEL_LABELS.get(mk, mk),
                color=COLORS.get(mk, "#666"),
                edgecolor="white", linewidth=0.5,
            )
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=6.5, rotation=55)

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel("Score")
        ax.set_title(f"All Models Performance Comparison — {DATASET_LABELS.get(dataset_id, dataset_id)}")
        all_vals = [v for md in models.values() for v in md.values() if v]
        ax.set_ylim(bottom=max(0.85, min(all_vals) - 0.02))
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        fig.tight_layout()
        save_fig(fig, f"15_all_models_comparison_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 16: Framework vs Baselines — F1 Score Comparison
# ══════════════════════════════════════════════════════════════════════════════

def chart_f1_comparison(adaptive_results, baselines):
    """Horizontal bar chart ranking all models by F1 score."""
    common_datasets = sorted(set(adaptive_results.keys()) & set(baselines.keys()))

    for dataset_id in common_datasets:
        ad = adaptive_results[dataset_id]
        bl = baselines[dataset_id]

        # Gather F1 scores
        model_f1 = []
        for mk in ["path_a", "path_b", "static_ensemble", "adaptive_hybrid"]:
            f1 = ad.get("metrics", {}).get(mk, {}).get("f1")
            if f1 is not None:
                model_f1.append((mk, f1))
        for mk in ["odae_wpdc", "pso_xgboost", "bigru_attention"]:
            f1 = bl.get(mk, {}).get("f1")
            if f1 is not None:
                model_f1.append((mk, f1))

        # Sort by F1 ascending (for horizontal bar)
        model_f1.sort(key=lambda x: x[1])

        fig, ax = plt.subplots(figsize=(10, max(4, len(model_f1) * 0.8)))
        names = [MODEL_LABELS.get(mk, mk) for mk, _ in model_f1]
        vals = [f1 for _, f1 in model_f1]
        colors = [COLORS.get(mk, "#666") for mk, _ in model_f1]

        bars = ax.barh(range(len(names)), vals, color=colors, edgecolor="white", height=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=10)

        # Highlight the proposed method
        for i, (mk, _) in enumerate(model_f1):
            if mk == "adaptive_hybrid":
                bars[i].set_edgecolor("#333")
                bars[i].set_linewidth(2)

        ax.set_xlabel("F1 Score")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        low = max(0.85, min(vals) - 0.02)
        ax.set_xlim(left=low)
        ax.set_title(f"F1 Score Ranking — {DATASET_LABELS.get(dataset_id, dataset_id)}")
        fig.tight_layout()
        save_fig(fig, f"16_f1_ranking_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 17B: Confusion Matrices — All Models Side-by-Side
# ══════════════════════════════════════════════════════════════════════════════

def chart_all_confusion_matrices(adaptive_results, baselines):
    """Side-by-side normalized confusion matrix heatmaps for all models."""
    labels = ["Non-Phishing", "Phishing"]
    common_datasets = sorted(set(adaptive_results.keys()) & set(baselines.keys()))

    for dataset_id in common_datasets:
        ad = adaptive_results[dataset_id]
        bl = baselines[dataset_id]

        # Collect all confusion matrices
        cm_data = []  # list of (model_key, cm_array)
        for mk in ["path_a", "path_b", "adaptive_hybrid"]:
            cm = ad.get("metrics", {}).get(mk, {}).get("confusion_matrix")
            if cm:
                cm_data.append((mk, np.array(cm)))

        for mk in ["odae_wpdc", "pso_xgboost", "bigru_attention"]:
            if mk not in bl:
                continue
            cm = bl[mk].get("confusion_matrix")
            if cm:
                cm_data.append((mk, np.array(cm)))
            elif bl[mk].get("confusion_matrix_per_fold"):
                # Average across folds for ODAE-WPDC
                cms = bl[mk]["confusion_matrix_per_fold"]
                cm_avg = np.mean([np.array(c) for c in cms], axis=0).astype(int)
                cm_data.append((mk, cm_avg))

        if not cm_data:
            continue

        n = len(cm_data)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows), squeeze=False)

        for idx, (mk, cm) in enumerate(cm_data):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues" if mk != "adaptive_hybrid" else "Greens",
                        xticklabels=labels, yticklabels=labels,
                        ax=ax, vmin=0, vmax=1, cbar=False)
            for ri in range(2):
                for ci in range(2):
                    ax.text(ci + 0.5, ri + 0.72, f"(n={cm[ri, ci]})",
                            ha="center", va="center", fontsize=8, color="gray")
            ax.set_title(MODEL_LABELS.get(mk, mk), fontsize=10)
            ax.set_ylabel("True" if c == 0 else "")
            ax.set_xlabel("Predicted")

        # Hide unused subplots
        for idx in range(n, rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].set_visible(False)

        fig.suptitle(f"Confusion Matrices — All Models — {DATASET_LABELS.get(dataset_id, dataset_id)}",
                     fontsize=14, y=1.02)
        fig.tight_layout()
        save_fig(fig, f"17_all_confusion_matrices_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 18B: Computational Efficiency — All Models
# ══════════════════════════════════════════════════════════════════════════════

def chart_all_computational_efficiency(adaptive_results, baselines):
    """3-panel comparison of training time, inference latency, model size for all models."""
    common_datasets = sorted(set(adaptive_results.keys()) & set(baselines.keys()))

    for dataset_id in common_datasets:
        ad = adaptive_results[dataset_id]
        bl = baselines[dataset_id]
        comp = ad.get("computational", {})

        models = {}
        # Framework models
        prefix_map = {"path_a": "path_a", "path_b": "path_b", "adaptive_hybrid": "adaptive"}
        for mk, prefix in prefix_map.items():
            tt = comp.get(f"{prefix}_train_time_s", 0)
            inf = comp.get(f"{prefix}_inference_ms", 0)
            sz = comp.get(f"{prefix}_model_size_mb", 0)
            if tt or inf or sz:
                models[mk] = {"train_time": tt, "inference_ms": inf, "model_size": sz}

        # Baselines
        for mk in ["odae_wpdc", "pso_xgboost", "bigru_attention"]:
            if mk not in bl:
                continue
            models[mk] = {
                "train_time": bl[mk].get("training_time_s", 0) or 0,
                "inference_ms": bl[mk].get("inference_ms_per_sample", 0) or 0,
                "model_size": bl[mk].get("model_size_mb", 0) or 0,
            }

        if len(models) < 2:
            continue

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
        model_keys = list(models.keys())
        names = [MODEL_LABELS.get(k, k) for k in model_keys]
        colors = [COLORS.get(k, "#666") for k in model_keys]

        # Training time
        vals = [models[k]["train_time"] for k in model_keys]
        bars = ax1.bar(range(len(names)), vals, color=colors, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                     f"{val:.1f}s", ha="center", va="bottom", fontsize=8)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax1.set_ylabel("Seconds")
        ax1.set_title("Training Time")

        # Inference latency (log scale if wide range)
        vals = [models[k]["inference_ms"] for k in model_keys]
        bars = ax2.bar(range(len(names)), vals, color=colors, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=8)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax2.set_ylabel("ms / sample")
        ax2.set_title("Inference Latency")

        # Model size
        vals = [models[k]["model_size"] for k in model_keys]
        bars = ax3.bar(range(len(names)), vals, color=colors, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax3.set_ylabel("MB")
        ax3.set_title("Model Size")

        fig.suptitle(f"Computational Efficiency — All Models — {DATASET_LABELS.get(dataset_id, dataset_id)}",
                     fontsize=14, y=1.02)
        fig.tight_layout()
        save_fig(fig, f"18_all_computational_efficiency_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 19B: Efficiency-Accuracy Tradeoff — All Models
# ══════════════════════════════════════════════════════════════════════════════

def chart_all_efficiency_tradeoff(adaptive_results, baselines):
    """Scatter: x = training time (log), y = F1 — all models."""
    common_datasets = sorted(set(adaptive_results.keys()) & set(baselines.keys()))

    for dataset_id in common_datasets:
        ad = adaptive_results[dataset_id]
        bl = baselines[dataset_id]
        comp = ad.get("computational", {})

        points = []  # (model_key, train_time, f1, inference_ms)

        # Framework
        for mk, prefix in [("path_a", "path_a"), ("path_b", "path_b"), ("adaptive_hybrid", "adaptive")]:
            f1 = ad.get("metrics", {}).get(mk, {}).get("f1", 0)
            tt = comp.get(f"{prefix}_train_time_s", 0)
            inf = comp.get(f"{prefix}_inference_ms", 0)
            if f1 and tt:
                points.append((mk, tt, f1, inf))

        # Baselines
        for mk in ["odae_wpdc", "pso_xgboost", "bigru_attention"]:
            if mk not in bl:
                continue
            f1 = bl[mk].get("f1", 0)
            tt = bl[mk].get("training_time_s", 0) or 0
            inf = bl[mk].get("inference_ms_per_sample", 0) or 0
            if f1 and tt:
                points.append((mk, tt, f1, inf))

        if len(points) < 2:
            continue

        fig, ax = plt.subplots(figsize=(10, 7))
        for mk, tt, f1, inf in points:
            size = max(100, inf * 200)
            ax.scatter(tt, f1, s=size, c=COLORS.get(mk, "#666"),
                       label=MODEL_LABELS.get(mk, mk),
                       edgecolors="black", linewidth=0.8, alpha=0.85, zorder=5)
            ax.annotate(f"F1={f1:.4f}\n{tt:.1f}s",
                        (tt, f1), textcoords="offset points",
                        xytext=(12, 5), fontsize=8,
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))

        ax.set_xscale("log")
        ax.set_xlabel("Training Time (seconds, log scale)")
        ax.set_ylabel("F1 Score")
        ax.set_title(f"Efficiency–Accuracy Tradeoff — All Models — {DATASET_LABELS.get(dataset_id, dataset_id)}\n"
                     "(bubble size ∝ inference latency)")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        save_fig(fig, f"19_all_efficiency_tradeoff_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 20B: Radar — Framework vs All Baselines
# ══════════════════════════════════════════════════════════════════════════════

def chart_all_radar(adaptive_results, baselines):
    """Radar chart comparing proposed framework against all baselines."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    common_datasets = sorted(set(adaptive_results.keys()) & set(baselines.keys()))

    for dataset_id in common_datasets:
        ad = adaptive_results[dataset_id]
        bl = baselines[dataset_id]

        models_data = {}
        for mk in ["path_a", "path_b", "adaptive_hybrid"]:
            if mk in ad["metrics"]:
                models_data[mk] = [ad["metrics"][mk].get(m, 0) for m in metrics]
        for mk in ["odae_wpdc", "pso_xgboost", "bigru_attention"]:
            if mk in bl:
                models_data[mk] = [bl[mk].get(m, 0) or 0 for m in metrics]

        if len(models_data) < 2:
            continue

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

        for mk, vals in models_data.items():
            vals_closed = vals + vals[:1]
            ax.fill(angles, vals_closed, alpha=0.1, color=COLORS.get(mk, "#666"))
            ax.plot(angles, vals_closed, "o-", color=COLORS.get(mk, "#666"),
                    label=MODEL_LABELS.get(mk, mk), linewidth=2, markersize=5)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        all_vals = [v for vals in models_data.values() for v in vals if v]
        min_val = max(0.85, min(all_vals) - 0.02) if all_vals else 0.85
        ax.set_ylim(min_val, 1.0)
        ax.set_title(f"All Models Radar — {DATASET_LABELS.get(dataset_id, dataset_id)}",
                     y=1.08, fontsize=13)
        ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0), fontsize=8)
        fig.tight_layout()
        save_fig(fig, f"20_all_radar_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 21: Comprehensive Summary Table — All Models
# ══════════════════════════════════════════════════════════════════════════════

def chart_all_summary_table(adaptive_results, baselines):
    """Publication-ready table figure with all models and all metrics."""
    pred_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    pred_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    common_datasets = sorted(set(adaptive_results.keys()) & set(baselines.keys()))

    for dataset_id in common_datasets:
        ad = adaptive_results[dataset_id]
        bl = baselines[dataset_id]
        comp = ad.get("computational", {})

        model_order = ["odae_wpdc", "pso_xgboost", "bigru_attention",
                       "path_a", "path_b", "static_ensemble", "adaptive_hybrid"]
        all_cols = pred_labels + ["Train (s)", "Inf (ms)", "Size (MB)"]

        rows = []
        row_labels = []
        present_models = []

        for mk in model_order:
            # Get predictive metrics
            if mk in ["odae_wpdc", "pso_xgboost", "bigru_attention"]:
                if mk not in bl:
                    continue
                pred_vals = [bl[mk].get(m) for m in pred_metrics]
                tt = bl[mk].get("training_time_s", 0) or 0
                inf = bl[mk].get("inference_ms_per_sample", 0) or 0
                sz = bl[mk].get("model_size_mb", 0) or 0
            else:
                m_data = ad.get("metrics", {}).get(mk)
                if not m_data:
                    continue
                pred_vals = [m_data.get(m) for m in pred_metrics]
                prefix = {"path_a": "path_a", "path_b": "path_b",
                          "static_ensemble": "adaptive", "adaptive_hybrid": "adaptive"}.get(mk, mk)
                tt = comp.get(f"{prefix}_train_time_s", 0)
                inf = comp.get(f"{prefix}_inference_ms", 0)
                sz = comp.get(f"{prefix}_model_size_mb", 0)

            row = []
            for v in pred_vals:
                row.append(f"{v:.4f}" if v is not None else "N/A")
            row.append(f"{tt:.2f}" if tt else "N/A")
            row.append(f"{inf:.4f}" if inf else "N/A")
            row.append(f"{sz:.2f}" if sz else "N/A")
            rows.append(row)
            row_labels.append(MODEL_LABELS.get(mk, mk))
            present_models.append(mk)

        if not rows:
            continue

        fig, ax = plt.subplots(figsize=(16, 1.5 + 0.55 * len(rows)))
        ax.axis("off")

        table = ax.table(
            cellText=rows,
            rowLabels=row_labels,
            colLabels=all_cols,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.8)

        # Style header
        for j in range(len(all_cols)):
            table[0, j].set_facecolor("#37474F")
            table[0, j].set_text_props(color="white", fontweight="bold")

        # Highlight best predictive values per column
        for j in range(len(pred_metrics)):
            col_vals = []
            for i in range(len(rows)):
                try:
                    col_vals.append(float(rows[i][j]))
                except (ValueError, IndexError):
                    col_vals.append(None)
            valid = [v for v in col_vals if v is not None]
            if valid:
                best = max(valid)
                for i, v in enumerate(col_vals):
                    if v is not None and abs(v - best) < 1e-8:
                        table[i + 1, j].set_text_props(fontweight="bold")
                        table[i + 1, j].set_facecolor("#C8E6C9")

        # Color row labels by model color
        for i, mk in enumerate(present_models):
            color = COLORS.get(mk, "#FFFFFF")
            table[i + 1, -1].set_facecolor(color + "20")
            # Highlight proposed method row
            if mk == "adaptive_hybrid":
                for j in range(len(all_cols)):
                    cell = table[i + 1, j]
                    if cell.get_facecolor()[:3] != (0.784, 0.902, 0.784):  # not already green
                        current = cell.get_facecolor()
                        cell.set_edgecolor("#4CAF50")

        ax.set_title(f"Comprehensive Performance Summary — {DATASET_LABELS.get(dataset_id, dataset_id)}",
                     fontsize=14, pad=20)
        fig.tight_layout()
        save_fig(fig, f"21_all_summary_table_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)

    print("=" * 60)
    print("THESIS RESULTS VISUALIZATION GENERATOR")
    print("=" * 60)

    # Load data
    print("\nLoading results...")
    adaptive_results = load_adaptive_results()
    baselines = load_baseline_results()
    print(f"  Adaptive Hybrid: {len(adaptive_results)} dataset(s): {sorted(adaptive_results.keys())}")
    print(f"  Baselines: {len(baselines)} dataset(s): {sorted(baselines.keys())}")
    for did, bl in sorted(baselines.items()):
        print(f"    Dataset {did}: {', '.join(sorted(bl.keys()))}")

    if not adaptive_results:
        print("No adaptive_hybrid results found. Exiting.")
        return

    # Generate all charts
    print(f"\nGenerating charts to: {CHARTS_DIR}/")
    print("-" * 60)

    # ── Part I: Framework-only charts (1-14) ──
    print("\n── Part I: Framework Analysis ──")

    print("\n[1/20] Model Performance Comparison (framework)...")
    chart_model_comparison(adaptive_results)

    print("[2/20] Log-Loss Comparison...")
    chart_log_loss_comparison(adaptive_results)

    print("[3/20] Adaptive Weight Evolution...")
    chart_weight_evolution(adaptive_results)

    print("[4/20] Confusion Matrices (framework)...")
    chart_confusion_matrices(adaptive_results)

    print("[5/20] Computational Efficiency (framework)...")
    chart_computational_efficiency(adaptive_results)

    print("[6/20] Efficiency-Accuracy Tradeoff (framework)...")
    chart_efficiency_accuracy_tradeoff(adaptive_results)

    print("[7/20] Radar Comparison (framework)...")
    chart_radar_comparison(adaptive_results)

    print("[8/20] Cross-Dataset Heatmap...")
    chart_cross_dataset_heatmap(adaptive_results)

    print("[9/20] Adaptive vs Static Progression...")
    chart_adaptive_vs_static(adaptive_results)

    print("[10/20] Gamma Sensitivity...")
    chart_gamma_comparison(adaptive_results)

    print("[11/20] Weight Distribution...")
    chart_weight_distribution(adaptive_results)

    print("[12/20] Convergence Rate...")
    chart_convergence_rate(adaptive_results)

    print("[13/20] Summary Tables (framework)...")
    chart_summary_table(adaptive_results)

    print("[14/20] Combined Loss Analysis...")
    chart_combined_loss_analysis(adaptive_results)

    # ── Part II: Framework vs Baselines (15-21) ──
    print("\n── Part II: Framework vs Baselines ──")

    if baselines:
        print("\n[15/20] All Models Performance Comparison...")
        chart_all_models_comparison(adaptive_results, baselines)

        print("[16/20] F1 Score Ranking...")
        chart_f1_comparison(adaptive_results, baselines)

        print("[17/20] All Confusion Matrices...")
        chart_all_confusion_matrices(adaptive_results, baselines)

        print("[18/20] All Computational Efficiency...")
        chart_all_computational_efficiency(adaptive_results, baselines)

        print("[19/20] All Efficiency-Accuracy Tradeoff...")
        chart_all_efficiency_tradeoff(adaptive_results, baselines)

        print("[20/20] All Radar Comparison...")
        chart_all_radar(adaptive_results, baselines)

        print("[21] Comprehensive Summary Table...")
        chart_all_summary_table(adaptive_results, baselines)
    else:
        print("\n  (No baseline results found — skipping charts 15-21)")

    # Summary
    chart_files = [f for f in os.listdir(CHARTS_DIR) if f.endswith(f".{FIG_FORMAT}")]
    print("\n" + "=" * 60)
    print(f"DONE — Generated {len(chart_files)} charts in {CHARTS_DIR}/")
    print("=" * 60)
    for f in sorted(chart_files):
        print(f"  • {f}")


if __name__ == "__main__":
    main()
