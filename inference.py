"""
Adaptive Hybrid Inference Script
=================================
Load a saved checkpoint and run inference on a new dataset
without retraining. Produces the same metrics (accuracy, precision,
recall, F1, ROC-AUC, log-loss, confusion matrix) using the frozen
adaptive weights from training.

Usage:
    python inference.py --checkpoint results/adaptive_hybrid/dataset_9/checkpoint \
                        --dataset_id 10

    The --dataset_id must be registered in config.yaml under `datasets:`.
"""

import os
import re
import sys
import json
import time
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss, confusion_matrix,
)
from keras.models import load_model
from keras import layers
import tensorflow as tf

from preprocess import load_config, load_any_dataset, extract_features_from_url
from gpu_setup import configure_gpu


# ── Keras cross-version compatibility ──────────────────────────────────────────
# Models saved with newer Keras (>=3.8) include 'quantization_config' in layer
# configs.  Older Keras versions reject the unknown kwarg.  We patch the saved
# config.json inside the .keras zip to strip unsupported keys before loading.
import zipfile
import tempfile
import shutil

_STRIP_KEYS = {"quantization_config"}

def _strip_config_keys(obj):
    """Recursively remove unsupported keys from a nested config dict/list."""
    if isinstance(obj, dict):
        return {k: _strip_config_keys(v) for k, v in obj.items()
                if k not in _STRIP_KEYS}
    if isinstance(obj, list):
        return [_strip_config_keys(item) for item in obj]
    return obj

def _load_keras_model_compat(path, custom_objects=None):
    """Load a .keras model, stripping unsupported config keys for compatibility."""
    tmp_dir = tempfile.mkdtemp()
    try:
        patched_path = os.path.join(tmp_dir, "model_patched.keras")
        with zipfile.ZipFile(path, "r") as zin, \
             zipfile.ZipFile(patched_path, "w") as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "config.json":
                    cfg = json.loads(data.decode("utf-8"))
                    cfg = _strip_config_keys(cfg)
                    data = json.dumps(cfg).encode("utf-8")
                zout.writestr(item, data)
        return load_model(patched_path, custom_objects=custom_objects)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Custom Keras layer (must be defined for model loading) ─────────────────────
class AttentionLayer(layers.Layer):
    """Simple additive attention over the temporal dimension."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight", shape=(input_shape[-1], 1),
            initializer="glorot_uniform", trainable=True,
        )
        self.b = self.add_weight(
            name="att_bias", shape=(input_shape[1], 1),
            initializer="zeros", trainable=True,
        )

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)


# ── Load checkpoint ────────────────────────────────────────────────────────────
def load_checkpoint(ckpt_dir):
    """Load all saved artefacts from a checkpoint directory."""
    print(f"Loading checkpoint from {ckpt_dir} ...")

    ckpt = {
        # Path A
        "rf_model":           joblib.load(os.path.join(ckpt_dir, "path_a_rf.joblib")),
        "selected_features":  joblib.load(os.path.join(ckpt_dir, "path_a_features.joblib")),
        "mm_scaler":          joblib.load(os.path.join(ckpt_dir, "path_a_mm_scaler.joblib")),
        "z_scaler":           joblib.load(os.path.join(ckpt_dir, "path_a_z_scaler.joblib")),
        "le":                 joblib.load(os.path.join(ckpt_dir, "label_encoder.joblib")),
        # Path B
        "keras_model":        _load_keras_model_compat(
            os.path.join(ckpt_dir, "path_b_cnn_bilstm.keras"),
            custom_objects={"AttentionLayer": AttentionLayer},
        ),
        "char_to_idx":        joblib.load(os.path.join(ckpt_dir, "path_b_char_to_idx.joblib")),
        "token_vocab":        joblib.load(os.path.join(ckpt_dir, "path_b_token_vocab.joblib")),
        "tab_scaler":         joblib.load(os.path.join(ckpt_dir, "path_b_tab_scaler.joblib")),
        # Adaptive weights
        "adaptive":           joblib.load(os.path.join(ckpt_dir, "adaptive_weights.joblib")),
    }
    # tab_features may not exist in older checkpoints — reconstruct from
    # the training dataset if needed (requires cfg + training dataset_id)
    tab_feat_path = os.path.join(ckpt_dir, "path_b_tab_features.joblib")
    if os.path.exists(tab_feat_path):
        ckpt["tab_features"] = joblib.load(tab_feat_path)
    else:
        # Will be resolved later in run_inference using training dataset
        ckpt["tab_features"] = None
    print(f"  Path A RF model loaded  ({len(ckpt['selected_features'])} features)")
    print(f"  Path B CNN-BiLSTM loaded ({len(ckpt['char_to_idx'])} chars, "
          f"{len(ckpt['token_vocab'])} tokens)")
    print(f"  Adaptive: gamma={ckpt['adaptive']['gamma']}, "
          f"alpha={ckpt['adaptive']['alpha']:.4f}, "
          f"beta={ckpt['adaptive']['beta']:.4f}")
    return ckpt


# ── Preprocessing for a new dataset ───────────────────────────────────────────
def prepare_path_a(df, ckpt):
    """Prepare features for Path A (RF) from a DataFrame with STANDARD_FEATURES."""
    selected = ckpt["selected_features"]
    # Ensure all selected features exist; fill missing with 0
    for col in selected:
        if col not in df.columns:
            df[col] = 0
    X = df[selected].values.astype(np.float64)
    X = ckpt["mm_scaler"].transform(X)
    X = ckpt["z_scaler"].transform(X)
    return X


def _encode_url_chars(url, mapping, maxlen):
    """Encode a single URL to fixed-length char-index sequence."""
    encoded = [mapping.get(c, 0) for c in url[:maxlen]]
    if len(encoded) < maxlen:
        encoded += [0] * (maxlen - len(encoded))
    return encoded


def _encode_url_tokens(url, token_vocab, maxlen):
    """Tokenise and encode a single URL to fixed-length token-index sequence."""
    tokens = re.split(r'[/:?=&@.\-_~#%]+', url)
    encoded = [token_vocab.get(t, 0) for t in tokens[:maxlen] if t]
    if len(encoded) < maxlen:
        encoded += [0] * (maxlen - len(encoded))
    return encoded[:maxlen]


def prepare_path_b(df, urls, ckpt, char_max_len=300, token_max_len=30):
    """Prepare three input branches for Path B (CNN-BiLSTM)."""
    urls_clean = [u.lower().strip() for u in urls]
    char_to_idx = ckpt["char_to_idx"]
    token_vocab = ckpt["token_vocab"]
    tab_features = ckpt["tab_features"]

    # Char branch
    X_char = np.array([_encode_url_chars(u, char_to_idx, char_max_len)
                       for u in urls_clean])
    # Token branch
    X_token = np.array([_encode_url_tokens(u, token_vocab, token_max_len)
                        for u in urls_clean])
    # Tabular branch
    for col in tab_features:
        if col not in df.columns:
            df[col] = 0
    X_tab = df[tab_features].values.astype(np.float32)
    X_tab = np.nan_to_num(X_tab, nan=0.0)
    X_tab = ckpt["tab_scaler"].transform(X_tab)

    return X_char, X_token, X_tab


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(y_true, y_proba, label=""):
    """Compute classification metrics from probabilities."""
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_proba),
        "log_loss":  log_loss(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if label:
        print(f"\n{'─'*50}")
        print(f"  {label}")
        print(f"{'─'*50}")
        print(f"  Accuracy  : {metrics['accuracy']:.6f}")
        print(f"  Precision : {metrics['precision']:.6f}")
        print(f"  Recall    : {metrics['recall']:.6f}")
        print(f"  F1        : {metrics['f1']:.6f}")
        print(f"  ROC-AUC   : {metrics['roc_auc']:.6f}")
        print(f"  Log-loss  : {metrics['log_loss']:.6f}")
        print(f"  Confusion : {metrics['confusion_matrix']}")
    return metrics


# ── Main inference ────────────────────────────────────────────────────────────
def run_inference(ckpt_dir, dataset_id, cfg_path="config.yaml",
                  train_dataset_id=None):
    configure_gpu()
    cfg = load_config(cfg_path)
    ckpt = load_checkpoint(ckpt_dir)

    # Resolve tab_features for older checkpoints that didn't save them
    if ckpt["tab_features"] is None:
        if train_dataset_id is None:
            # Try to infer from checkpoint path  (e.g. .../dataset_3/checkpoint)
            parent = os.path.basename(os.path.dirname(ckpt_dir))
            if parent.startswith("dataset_"):
                train_dataset_id = parent.replace("dataset_", "")
        if train_dataset_id is None:
            sys.exit("ERROR: checkpoint has no saved tab_features and "
                     "--train_dataset_id was not provided.")
        print(f"  Reconstructing tab_features from training dataset [{train_dataset_id}] ...")
        train_df = load_any_dataset(train_dataset_id, cfg)
        ckpt["tab_features"] = [c for c in train_df.columns
                                if c not in ["url", "status"]]
        print(f"  Reconstructed {len(ckpt['tab_features'])} tabular features")

    # Load and extract features from the new dataset
    df = load_any_dataset(dataset_id, cfg)
    urls = df["url"].astype(str).values

    # Ground-truth labels
    le = ckpt["le"]
    y_true = le.transform(df["status"])

    # ── Path A inference ──
    print("\n── Path A (Random Forest) inference ──")
    t0 = time.perf_counter()
    X_a = prepare_path_a(df.copy(), ckpt)
    proba_a = ckpt["rf_model"].predict_proba(X_a)[:, 1]
    t_a = time.perf_counter() - t0
    m_a = evaluate(y_true, proba_a, "Path A  —  Chi-square + Random Forest")

    # ── Path B inference ──
    print("\n── Path B (CNN-BiLSTM) inference ──")
    t0 = time.perf_counter()
    X_char, X_token, X_tab = prepare_path_b(df.copy(), urls, ckpt)
    proba_b = ckpt["keras_model"].predict(
        [X_char, X_token, X_tab], verbose=0
    ).ravel()
    t_b = time.perf_counter() - t0
    m_b = evaluate(y_true, proba_b, "Path B  —  Hybrid CNN-BiLSTM")

    # ── Adaptive hybrid combination ──
    alpha = ckpt["adaptive"]["alpha"]
    beta = ckpt["adaptive"]["beta"]
    proba_hybrid = alpha * proba_a + beta * proba_b
    m_hybrid = evaluate(y_true, proba_hybrid,
                        f"Adaptive Hybrid  (alpha={alpha:.4f}, beta={beta:.4f})")

    # ── Static 50/50 ensemble (for comparison) ──
    proba_static = 0.5 * proba_a + 0.5 * proba_b
    m_static = evaluate(y_true, proba_static, "Static Ensemble (50/50)")

    # ── Save results ──
    out_dir = os.path.join(
        "results", "inference",
        f"trained_{os.path.basename(os.path.dirname(ckpt_dir))}",
        f"tested_{dataset_id}",
    )
    os.makedirs(out_dir, exist_ok=True)

    report = {
        "checkpoint": ckpt_dir,
        "test_dataset_id": str(dataset_id),
        "num_samples": len(y_true),
        "adaptive_weights": {
            "gamma": ckpt["adaptive"]["gamma"],
            "alpha": alpha,
            "beta": beta,
        },
        "metrics": {
            "path_a": m_a,
            "path_b": m_b,
            "static_ensemble": m_static,
            "adaptive_hybrid": m_hybrid,
        },
        "inference_time": {
            "path_a_s": t_a,
            "path_b_s": t_b,
            "total_s": t_a + t_b,
        },
    }
    report_path = os.path.join(out_dir, "inference_results.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nResults saved to {report_path}")

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run adaptive hybrid inference on a new dataset using a saved checkpoint."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the checkpoint directory (e.g. results/adaptive_hybrid/dataset_9/checkpoint)"
    )
    parser.add_argument(
        "--dataset_id", required=True,
        help="Dataset ID from config.yaml to test on (e.g. 10, 11, 12)"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    parser.add_argument(
        "--train_dataset_id", default=None,
        help="Training dataset ID (only needed for older checkpoints without "
             "saved tab_features; auto-inferred from checkpoint path if possible)"
    )
    args = parser.parse_args()
    run_inference(args.checkpoint, args.dataset_id, args.config,
                  args.train_dataset_id)
