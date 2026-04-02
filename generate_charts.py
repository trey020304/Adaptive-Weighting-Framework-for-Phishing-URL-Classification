"""
Thesis Results Visualization Generator
======================================
Generates all charts, graphs, and visual insights from:
  - Path A baseline (Chi-Square + Random Forest)
  - Path B baseline (Hybrid CNN-BiLSTM)
  - Adaptive Hybrid Ensemble (proposed framework)
  - ODAE-WPDC baseline (if results exist)
  - Inference / cross-dataset generalization results

Output: results/charts/ directory with publication-ready PNG figures.
"""

import os
import json
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
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
}

MODEL_LABELS = {
    "path_a": "Path A (Chi² + RF)",
    "path_b": "Path B (CNN-BiLSTM)",
    "static_ensemble": "Static Ensemble (0.5/0.5)",
    "adaptive_hybrid": "Adaptive Hybrid (Proposed)",
    "odae_wpdc": "ODAE-WPDC Baseline",
}

DATASET_LABELS = {
    "3": "Dataset 3 (Kaggle)",
    "9": "Dataset 9 (Mendeley)",
    "c1": "Dataset C1 (Mendeley)",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading Utilities
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


def load_path_a_metrics():
    """Load all path_a_baseline metrics_summary.csv files."""
    results = {}
    base = os.path.join(RESULTS_DIR, "path_a_baseline")
    if not os.path.isdir(base):
        return results
    for d in os.listdir(base):
        mpath = os.path.join(base, d, "metrics_summary.csv")
        if os.path.isfile(mpath):
            df = pd.read_csv(mpath)
            dataset_id = d.replace("dataset_", "")
            results[dataset_id] = df.iloc[0].to_dict()
    return results


def load_path_b_metrics():
    """Load all path_b_baseline metrics_summary.csv files."""
    results = {}
    base = os.path.join(RESULTS_DIR, "path_b_baseline")
    if not os.path.isdir(base):
        return results
    for d in os.listdir(base):
        mpath = os.path.join(base, d, "metrics_summary.csv")
        if os.path.isfile(mpath):
            df = pd.read_csv(mpath)
            dataset_id = d.replace("dataset_", "")
            results[dataset_id] = df.iloc[0].to_dict()
    return results


def load_training_histories():
    """Load all path_b training_history.csv files."""
    histories = {}
    base = os.path.join(RESULTS_DIR, "path_b_baseline")
    if not os.path.isdir(base):
        return histories
    for d in os.listdir(base):
        hpath = os.path.join(base, d, "training_history.csv")
        if os.path.isfile(hpath):
            dataset_id = d.replace("dataset_", "")
            histories[dataset_id] = pd.read_csv(hpath)
    return histories


def load_chi2_scores():
    """Load all chi2_feature_scores.csv files."""
    scores = {}
    base = os.path.join(RESULTS_DIR, "path_a_baseline")
    if not os.path.isdir(base):
        return scores
    for d in os.listdir(base):
        cpath = os.path.join(base, d, "chi2_feature_scores.csv")
        if os.path.isfile(cpath):
            dataset_id = d.replace("dataset_", "")
            scores[dataset_id] = pd.read_csv(cpath)
    return scores


def load_odae_wpdc_results():
    """Load ODAE-WPDC results if they exist."""
    results = {}
    # Check common locations
    for pattern in [
        os.path.join(RESULTS_DIR, "odae_wpdc", "**", "results.json"),
        os.path.join(RESULTS_DIR, "**", "odae_wpdc", "results.json"),
    ]:
        for rpath in glob.glob(pattern, recursive=True):
            with open(rpath) as f:
                data = json.load(f)
            dataset_id = data.get("dataset_id", "unknown")
            results[dataset_id] = data
    return results


def load_inference_results():
    """Load all inference results."""
    results = {}
    base = os.path.join(RESULTS_DIR, "inference")
    if not os.path.isdir(base):
        return results
    for trained_dir in os.listdir(base):
        trained_path = os.path.join(base, trained_dir)
        if not os.path.isdir(trained_path):
            continue
        for tested_dir in os.listdir(trained_path):
            rpath = os.path.join(trained_path, tested_dir, "inference_results.json")
            if os.path.isfile(rpath):
                with open(rpath) as f:
                    data = json.load(f)
                key = f"{trained_dir} → {tested_dir}"
                results[key] = data
    return results


def load_cross_dataset_results():
    """Load results from cross-dataset split folders (train[*]_val[*]_test[*])."""
    results = {}
    for entry in os.listdir(RESULTS_DIR):
        if entry.startswith("train["):
            rpath = os.path.join(RESULTS_DIR, entry, "results.json")
            if os.path.isfile(rpath):
                with open(rpath) as f:
                    results[entry] = json.load(f)
    return results


def save_fig(fig, name):
    """Save figure to charts directory."""
    path = os.path.join(CHARTS_DIR, f"{name}.{FIG_FORMAT}")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 1: Model Performance Comparison (Grouped Bar Chart)
# ══════════════════════════════════════════════════════════════════════════════

def chart_model_comparison(adaptive_results, odae_results):
    """Grouped bar chart comparing all models across key metrics."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

    for dataset_id, data in adaptive_results.items():
        models = {}
        for model_key in ["path_a", "path_b", "static_ensemble", "adaptive_hybrid"]:
            if model_key in data["metrics"]:
                models[model_key] = data["metrics"][model_key]

        # Add ODAE-WPDC if available for this dataset
        if dataset_id in odae_results:
            odae = odae_results[dataset_id]
            pred = odae.get("predictive_metrics", {})
            models["odae_wpdc"] = {
                m: pred[m]["mean"] for m in metrics if m in pred
            }

        n_metrics = len(metrics)
        n_models = len(models)
        x = np.arange(n_metrics)
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, (model_key, model_data) in enumerate(models.items()):
            vals = [model_data.get(m, 0) for m in metrics]
            bars = ax.bar(
                x + i * width - (n_models - 1) * width / 2,
                vals, width,
                label=MODEL_LABELS.get(model_key, model_key),
                color=COLORS.get(model_key, "#666"),
                edgecolor="white", linewidth=0.5,
            )
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=7, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel("Score")
        ax.set_title(f"Model Performance Comparison — {DATASET_LABELS.get(dataset_id, dataset_id)}")
        ax.set_ylim(bottom=min(0.90, min(v for m in models.values() for v in [m.get(met, 1) for met in metrics]) - 0.02))
        ax.legend(loc="lower right", framealpha=0.9)
        fig.tight_layout()
        save_fig(fig, f"01_model_comparison_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 2: Log-Loss Comparison
# ══════════════════════════════════════════════════════════════════════════════

def chart_log_loss_comparison(adaptive_results, odae_results):
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
#  Chart 3: Training Loss & Accuracy Curves (Path B)
# ══════════════════════════════════════════════════════════════════════════════

def chart_training_curves(histories, adaptive_results):
    """Dual-panel: loss and accuracy over epochs for Path B."""
    for dataset_id, df in histories.items():
        epochs = np.arange(1, len(df) + 1)

        # Find best epoch from adaptive results
        best_epoch = None
        if dataset_id in adaptive_results:
            best_epoch = adaptive_results[dataset_id].get("computational", {}).get("path_b_best_epoch")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        ax1.plot(epochs, df["loss"], "o-", color=COLORS["path_b"], label="Train Loss", markersize=4)
        ax1.plot(epochs, df["val_loss"], "s--", color=COLORS["path_a"], label="Val Loss", markersize=4)
        if best_epoch:
            ax1.axvline(best_epoch, color=COLORS["adaptive_hybrid"], linestyle=":",
                        linewidth=2, label=f"Best Epoch ({best_epoch})")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()

        # Accuracy
        ax2.plot(epochs, df["accuracy"], "o-", color=COLORS["path_b"], label="Train Acc", markersize=4)
        ax2.plot(epochs, df["val_accuracy"], "s--", color=COLORS["path_a"], label="Val Acc", markersize=4)
        if best_epoch:
            ax2.axvline(best_epoch, color=COLORS["adaptive_hybrid"], linestyle=":",
                        linewidth=2, label=f"Best Epoch ({best_epoch})")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training & Validation Accuracy")
        ax2.legend()

        fig.suptitle(f"Path B (CNN-BiLSTM) Training Curves — {DATASET_LABELS.get(dataset_id, dataset_id)}",
                     fontsize=14, y=1.02)
        fig.tight_layout()
        save_fig(fig, f"03_training_curves_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 4: Adaptive Weight Evolution (α/β over epochs)
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
        save_fig(fig, f"04_weight_evolution_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 5: Confusion Matrices (Side-by-Side Heatmaps)
# ══════════════════════════════════════════════════════════════════════════════

def chart_confusion_matrices(adaptive_results, odae_results):
    """Side-by-side normalized confusion matrix heatmaps for all models."""
    labels = ["Non-Phishing", "Phishing"]

    for dataset_id, data in adaptive_results.items():
        model_keys = [k for k in ["path_a", "path_b", "static_ensemble", "adaptive_hybrid"]
                      if k in data["metrics"] and "confusion_matrix" in data["metrics"][k]]

        # Add ODAE if available
        has_odae = (dataset_id in odae_results and
                    "predictive_metrics" in odae_results[dataset_id] and
                    "confusion_matrix_per_fold" in odae_results[dataset_id]["predictive_metrics"])

        n_models = len(model_keys) + (1 if has_odae else 0)
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

        if has_odae:
            # Average confusion matrix across folds
            cms = odae_results[dataset_id]["predictive_metrics"]["confusion_matrix_per_fold"]
            cm_avg = np.mean([np.array(c) for c in cms], axis=0)
            cm_norm = cm_avg / cm_avg.sum(axis=1, keepdims=True)
            ax = axes[-1]
            sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Reds",
                        xticklabels=labels, yticklabels=labels,
                        ax=ax, vmin=0, vmax=1, cbar=False)
            ax.set_title(MODEL_LABELS["odae_wpdc"], fontsize=10)
            ax.set_xlabel("Predicted Label")

        fig.suptitle(f"Confusion Matrices (Normalized) — {DATASET_LABELS.get(dataset_id, dataset_id)}",
                     fontsize=14, y=1.05)
        fig.tight_layout()
        save_fig(fig, f"05_confusion_matrices_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 6: Computational Efficiency Comparison
# ══════════════════════════════════════════════════════════════════════════════

def chart_computational_efficiency(adaptive_results, odae_results):
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

        # Add ODAE-WPDC if available
        if dataset_id in odae_results:
            odae_comp = odae_results[dataset_id].get("computational_metrics", {})
            models["odae_wpdc"] = {
                "train_time": odae_comp.get("training_time_s", {}).get("mean", 0),
                "inference_ms": odae_comp.get("inference_ms_per_sample", {}).get("mean", 0),
                "model_size": odae_comp.get("model_size_mb", {}).get("mean", 0),
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
        save_fig(fig, f"06_computational_efficiency_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 7: Efficiency-Accuracy Tradeoff Scatter
# ══════════════════════════════════════════════════════════════════════════════

def chart_efficiency_accuracy_tradeoff(adaptive_results, odae_results):
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

        # ODAE-WPDC
        if dataset_id in odae_results:
            odae = odae_results[dataset_id]
            f1 = odae.get("predictive_metrics", {}).get("f1", {}).get("mean", 0)
            tt = odae.get("total_pipeline_time_s", 0)
            inf = odae.get("computational_metrics", {}).get("inference_ms_per_sample", {}).get("mean", 0)
            if f1 and tt:
                points.append(("odae_wpdc", tt, f1, inf))

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
        save_fig(fig, f"07_efficiency_accuracy_tradeoff_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 8: Feature Importance (Chi-Square Scores)
# ══════════════════════════════════════════════════════════════════════════════

def chart_feature_importance(chi2_scores):
    """Horizontal bar chart of top features by chi-square score."""
    for dataset_id, df in chi2_scores.items():
        df = df.dropna(subset=["chi2"]).sort_values("chi2", ascending=True)

        # Show top 25 features
        top_n = 25
        df_top = df.tail(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(df_top)))
        ax.barh(range(len(df_top)), df_top["chi2"].values, color=colors, edgecolor="white")
        ax.set_yticks(range(len(df_top)))
        ax.set_yticklabels(df_top["feature"].values, fontsize=9)
        ax.set_xlabel("Chi-Square Score")
        ax.set_title(f"Top {top_n} Features by Chi-Square Score — {DATASET_LABELS.get(dataset_id, dataset_id)}")

        # Add score annotations
        for i, (_, row) in enumerate(df_top.iterrows()):
            ax.text(row["chi2"] + max(df_top["chi2"]) * 0.01, i,
                    f"{row['chi2']:.1f}", va="center", fontsize=8)

        fig.tight_layout()
        save_fig(fig, f"08_feature_importance_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 9: Multi-Dataset Performance Summary (Radar Chart)
# ══════════════════════════════════════════════════════════════════════════════

def chart_radar_comparison(adaptive_results):
    """Radar/spider chart comparing models across metrics for each dataset."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

    for dataset_id, data in adaptive_results.items():
        models_data = {}
        for model_key in ["path_a", "path_b", "adaptive_hybrid"]:
            if model_key in data["metrics"]:
                models_data[model_key] = [data["metrics"][model_key].get(m, 0) for m in metrics]

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
        min_val = max(0.90, min(all_vals) - 0.02)
        ax.set_ylim(min_val, 1.0)
        ax.set_title(f"Model Comparison Radar — {DATASET_LABELS.get(dataset_id, dataset_id)}",
                     y=1.08, fontsize=13)
        ax.legend(loc="lower right", bbox_to_anchor=(1.2, 0))
        fig.tight_layout()
        save_fig(fig, f"09_radar_comparison_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 10: Cross-Dataset Performance Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def chart_cross_dataset_heatmap(adaptive_results):
    """Heatmap showing performance across all datasets."""
    metrics = ["accuracy", "f1", "roc_auc", "log_loss"]
    metric_labels = ["Accuracy", "F1", "ROC-AUC", "Log-Loss"]
    model_keys = ["path_a", "path_b", "adaptive_hybrid"]

    dataset_ids = sorted(adaptive_results.keys())
    if len(dataset_ids) < 2:
        return

    for model_key in model_keys:
        matrix = []
        for dataset_id in dataset_ids:
            row = []
            for m in metrics:
                val = adaptive_results[dataset_id].get("metrics", {}).get(model_key, {}).get(m, np.nan)
                row.append(val)
            matrix.append(row)

        df = pd.DataFrame(matrix,
                          index=[DATASET_LABELS.get(d, d) for d in dataset_ids],
                          columns=metric_labels)

        fig, ax = plt.subplots(figsize=(8, max(3, len(dataset_ids) * 1.2)))
        sns.heatmap(df, annot=True, fmt=".4f", cmap="YlGnBu", ax=ax,
                    linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title(f"{MODEL_LABELS.get(model_key, model_key)} — Performance Across Datasets")
        fig.tight_layout()
        save_fig(fig, f"10_cross_dataset_heatmap_{model_key}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 11: Adaptive vs Static Ensemble Improvement
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
            vals = [data.get(mk, {}).get(m, 0) for mk in models_ordered]
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
            axes[i].set_ylabel("Score (higher = better)")

    axes[0].legend(fontsize=8)
    fig.suptitle("Progression: Individual Paths → Static → Adaptive Ensemble", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "11_adaptive_vs_static_progression")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 12: Gamma Sensitivity Analysis
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
    save_fig(fig, "12_gamma_sensitivity")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 13: Weight Distribution (α/β final values across datasets)
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

    ax.bar(x, alphas, label="α (Path A: Chi² + RF)", color=COLORS["path_a"], edgecolor="white", width=0.5)
    ax.bar(x, betas, bottom=alphas, label="β (Path B: CNN-BiLSTM)", color=COLORS["path_b"],
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
    save_fig(fig, "13_weight_distribution")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 14: Inference / Cross-Dataset Generalization
# ══════════════════════════════════════════════════════════════════════════════

def chart_inference_generalization(inference_results):
    """Bar chart showing how models trained on one dataset perform on others."""
    if not inference_results:
        return

    # Group by trained dataset
    trained_groups = {}
    for key, data in inference_results.items():
        parts = key.split(" → ")
        trained = parts[0].replace("trained_", "")
        tested = parts[1].replace("tested_", "") if len(parts) > 1 else "?"
        trained_groups.setdefault(trained, []).append((tested, data))

    for trained, tests in trained_groups.items():
        if len(tests) < 2:
            continue

        tested_ids = [t[0].replace("dataset_", "") for t in tests]
        metrics_data = {
            "adaptive_hybrid": [t[1].get("metrics", {}).get("adaptive_hybrid", {}).get("f1", 0) for t in tests],
            "path_a": [t[1].get("metrics", {}).get("path_a", {}).get("f1", 0) for t in tests],
            "path_b": [t[1].get("metrics", {}).get("path_b", {}).get("f1", 0) for t in tests],
        }

        fig, ax = plt.subplots(figsize=(max(8, len(tested_ids) * 1.5), 6))
        x = np.arange(len(tested_ids))
        width = 0.25

        for i, (model_key, vals) in enumerate(metrics_data.items()):
            ax.bar(x + i * width - width, vals, width,
                   label=MODEL_LABELS.get(model_key, model_key),
                   color=COLORS.get(model_key, "#666"), edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels([f"Test: {t}" for t in tested_ids], rotation=30, ha="right")
        ax.set_ylabel("F1 Score")
        ax.set_title(f"Cross-Dataset Generalization (Trained on {trained})")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        save_fig(fig, f"14_inference_generalization_{trained}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 15: Cross-Dataset Split Results Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def chart_cross_split_heatmap(cross_results):
    """Heatmap of F1 scores across all cross-dataset split configs."""
    if not cross_results:
        return

    model_keys = ["path_a", "path_b", "adaptive_hybrid"]
    rows = []
    split_names = []

    for split_name in sorted(cross_results.keys()):
        data = cross_results[split_name]
        met = data.get("metrics", {})
        row = []
        for mk in model_keys:
            f1 = met.get(mk, {}).get("f1", np.nan)
            row.append(f1)
        rows.append(row)
        # Shorten name for display
        short = split_name.replace("train", "Tr").replace("val", "V").replace("test", "Te")
        split_names.append(short)

    df = pd.DataFrame(rows, index=split_names,
                      columns=[MODEL_LABELS.get(mk, mk) for mk in model_keys])

    fig, ax = plt.subplots(figsize=(10, max(4, len(split_names) * 0.7)))
    sns.heatmap(df, annot=True, fmt=".4f", cmap="RdYlGn", ax=ax,
                linewidths=0.5, vmin=0, vmax=1, cbar_kws={"label": "F1 Score"})
    ax.set_title("Cross-Dataset Split Performance (F1 Score)")
    ax.set_ylabel("Split Configuration")
    fig.tight_layout()
    save_fig(fig, "15_cross_split_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 16: Cross-dataset Weight Collapse Analysis
# ══════════════════════════════════════════════════════════════════════════════

def chart_cross_dataset_weight_evolution(cross_results):
    """Show weight evolution for cross-dataset configs (often shows weight collapse)."""
    if not cross_results:
        return

    configs_with_history = {k: v for k, v in cross_results.items() if v.get("weight_history")}
    if not configs_with_history:
        return

    n = len(configs_with_history)
    cols = min(3, n)
    rows_count = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_count, cols, figsize=(6 * cols, 4 * rows_count), squeeze=False)

    for idx, (split_name, data) in enumerate(sorted(configs_with_history.items())):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        wh = data["weight_history"]
        epochs = [w["epoch"] for w in wh]
        alphas = [w["alpha"] for w in wh]
        betas = [w["beta"] for w in wh]

        ax.plot(epochs, alphas, "-", color=COLORS["path_a"], label="α (Path A)", linewidth=1.5)
        ax.plot(epochs, betas, "-", color=COLORS["path_b"], label="β (Path B)", linewidth=1.5)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Weight", fontsize=8)
        short = split_name.replace("train", "Tr").replace("val", "V").replace("test", "Te")
        ax.set_title(short, fontsize=9)
        if idx == 0:
            ax.legend(fontsize=7)

    # Hide unused subplots
    for idx in range(n, rows_count * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle("Adaptive Weight Evolution — Cross-Dataset Splits\n"
                 "(weight collapse indicates dataset distribution mismatch)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "16_cross_dataset_weight_evolution")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 17: Convergence Rate Comparison
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
    save_fig(fig, "17_convergence_rate")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 18: Summary Comparison Table (as figure for thesis)
# ══════════════════════════════════════════════════════════════════════════════

def chart_summary_table(adaptive_results, odae_results):
    """Generate a publication-ready table as a figure."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Log-Loss"]

    for dataset_id, data in adaptive_results.items():
        model_keys = ["path_a", "path_b", "static_ensemble", "adaptive_hybrid"]
        model_names = [MODEL_LABELS.get(k, k) for k in model_keys]

        # Check for ODAE-WPDC
        if dataset_id in odae_results:
            model_keys.append("odae_wpdc")
            model_names.append(MODEL_LABELS["odae_wpdc"])

        rows = []
        for mk in model_keys:
            row = []
            if mk == "odae_wpdc":
                pred = odae_results[dataset_id].get("predictive_metrics", {})
                for m in metrics:
                    val = pred.get(m, {}).get("mean", "N/A")
                    row.append(f"{val:.4f}" if isinstance(val, float) else str(val))
            else:
                for m in metrics:
                    val = data["metrics"].get(mk, {}).get(m, "N/A")
                    row.append(f"{val:.4f}" if isinstance(val, float) else str(val))
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
                    col_vals.append(float(rows[i][j]))
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
        save_fig(fig, f"18_summary_table_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 19: Feature Overlap Across Datasets
# ══════════════════════════════════════════════════════════════════════════════

def chart_feature_overlap(chi2_scores, adaptive_results):
    """Show which features are consistently important across datasets."""
    if len(chi2_scores) < 2:
        return

    # Get top 15 features per dataset
    top_n = 15
    all_features = set()
    dataset_tops = {}
    for dataset_id, df in chi2_scores.items():
        df = df.dropna(subset=["chi2"]).sort_values("chi2", ascending=False)
        top_feats = df.head(top_n)["feature"].tolist()
        dataset_tops[dataset_id] = set(top_feats)
        all_features.update(top_feats)

    # Also check adaptive selected features
    for dataset_id, data in adaptive_results.items():
        selected = data.get("config", {}).get("selected_features", [])
        if selected:
            dataset_tops.setdefault(f"{dataset_id}_selected", set(selected[:top_n]))

    # Build presence matrix
    all_features = sorted(all_features)
    dataset_ids = sorted(dataset_tops.keys())
    matrix = np.zeros((len(all_features), len(dataset_ids)))

    for j, did in enumerate(dataset_ids):
        for i, feat in enumerate(all_features):
            if feat in dataset_tops[did]:
                matrix[i, j] = 1

    fig, ax = plt.subplots(figsize=(max(6, len(dataset_ids) * 2), max(6, len(all_features) * 0.35)))
    sns.heatmap(matrix, xticklabels=[DATASET_LABELS.get(d, d) for d in dataset_ids],
                yticklabels=all_features, cmap="YlGn", cbar=False, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_title(f"Top {top_n} Feature Presence Across Datasets\n(green = in top {top_n})")
    fig.tight_layout()
    save_fig(fig, "19_feature_overlap")


# ══════════════════════════════════════════════════════════════════════════════
#  Chart 20: Combined Loss Curve (Path A constant + Path B training)
# ══════════════════════════════════════════════════════════════════════════════

def chart_combined_loss_analysis(adaptive_results, histories):
    """Overlay Path A's constant val loss with Path B's training val loss."""
    for dataset_id, data in adaptive_results.items():
        wh = data.get("weight_history")
        if not wh or dataset_id not in histories:
            continue

        df_hist = histories[dataset_id]
        epochs = np.arange(1, len(df_hist) + 1)

        # Path A loss is constant across epochs
        L_A = wh[0]["L_A"]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.axhline(L_A, color=COLORS["path_a"], linestyle="--", linewidth=2,
                    label=f"Path A Val Loss (constant: {L_A:.4f})")
        ax.plot(epochs, df_hist["val_loss"], "s-", color=COLORS["path_b"],
                label="Path B Val Loss", markersize=5, linewidth=2)

        # Combined loss from weight history
        wh_epochs = [w["epoch"] for w in wh]
        combined = [w["alpha"] * w["L_A"] + w["beta"] * w["L_B"] for w in wh]
        ax.plot(wh_epochs, combined, "D-", color=COLORS["adaptive_hybrid"],
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
        save_fig(fig, f"20_combined_loss_analysis_dataset_{dataset_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)

    print("=" * 60)
    print("THESIS RESULTS VISUALIZATION GENERATOR")
    print("=" * 60)

    # Load all data
    print("\nLoading results...")
    adaptive_results = load_adaptive_results()
    path_a_metrics = load_path_a_metrics()
    path_b_metrics = load_path_b_metrics()
    histories = load_training_histories()
    chi2_scores = load_chi2_scores()
    odae_results = load_odae_wpdc_results()
    inference_results = load_inference_results()
    cross_results = load_cross_dataset_results()

    print(f"  Adaptive Hybrid: {len(adaptive_results)} dataset(s)")
    print(f"  Path A Baseline: {len(path_a_metrics)} dataset(s)")
    print(f"  Path B Baseline: {len(path_b_metrics)} dataset(s)")
    print(f"  Training Histories: {len(histories)} dataset(s)")
    print(f"  Chi² Feature Scores: {len(chi2_scores)} dataset(s)")
    print(f"  ODAE-WPDC: {len(odae_results)} dataset(s)")
    print(f"  Inference Results: {len(inference_results)} config(s)")
    print(f"  Cross-Dataset Splits: {len(cross_results)} config(s)")

    # Generate all charts
    print(f"\nGenerating charts to: {CHARTS_DIR}/")
    print("-" * 60)

    if adaptive_results:
        print("\n[1/20] Model Performance Comparison...")
        chart_model_comparison(adaptive_results, odae_results)

        print("[2/20] Log-Loss Comparison...")
        chart_log_loss_comparison(adaptive_results, odae_results)

    if histories:
        print("[3/20] Training Curves (Path B)...")
        chart_training_curves(histories, adaptive_results)

    if adaptive_results:
        print("[4/20] Adaptive Weight Evolution...")
        chart_weight_evolution(adaptive_results)

        print("[5/20] Confusion Matrices...")
        chart_confusion_matrices(adaptive_results, odae_results)

        print("[6/20] Computational Efficiency...")
        chart_computational_efficiency(adaptive_results, odae_results)

        print("[7/20] Efficiency-Accuracy Tradeoff...")
        chart_efficiency_accuracy_tradeoff(adaptive_results, odae_results)

    if chi2_scores:
        print("[8/20] Feature Importance (Chi²)...")
        chart_feature_importance(chi2_scores)

    if adaptive_results:
        print("[9/20] Radar Comparison...")
        chart_radar_comparison(adaptive_results)

        print("[10/20] Cross-Dataset Heatmap...")
        chart_cross_dataset_heatmap(adaptive_results)

        print("[11/20] Adaptive vs Static Progression...")
        chart_adaptive_vs_static(adaptive_results)

        print("[12/20] Gamma Sensitivity...")
        chart_gamma_comparison(adaptive_results)

        print("[13/20] Weight Distribution...")
        chart_weight_distribution(adaptive_results)

    if inference_results:
        print("[14/20] Inference Generalization...")
        chart_inference_generalization(inference_results)

    if cross_results:
        print("[15/20] Cross-Split Heatmap...")
        chart_cross_split_heatmap(cross_results)

        print("[16/20] Cross-Dataset Weight Evolution...")
        chart_cross_dataset_weight_evolution(cross_results)

    if adaptive_results:
        print("[17/20] Convergence Rate...")
        chart_convergence_rate(adaptive_results)

        print("[18/20] Summary Tables...")
        chart_summary_table(adaptive_results, odae_results)

    if chi2_scores and len(chi2_scores) >= 2:
        print("[19/20] Feature Overlap...")
        chart_feature_overlap(chi2_scores, adaptive_results)

    if adaptive_results and histories:
        print("[20/20] Combined Loss Analysis...")
        chart_combined_loss_analysis(adaptive_results, histories)

    # Summary
    chart_files = [f for f in os.listdir(CHARTS_DIR) if f.endswith(f".{FIG_FORMAT}")]
    print("\n" + "=" * 60)
    print(f"DONE — Generated {len(chart_files)} charts in {CHARTS_DIR}/")
    print("=" * 60)
    for f in sorted(chart_files):
        print(f"  • {f}")


if __name__ == "__main__":
    main()
