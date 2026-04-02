"""
BiGRU-Attention Pipeline: Character-Level BiGRU-Attention for Phishing Classification
======================================================================================
Reproduction of: Yuan et al. (2019), ICICS 2019, LNCS 11999, pp. 746-762.

Pipeline stages:
  1. Data pre-processing (preprocess.py → bigru_preprocess)
     - Character-level URL encoding (first 150 chars, front zero-padding)
  2. Build BiGRU-Attention model
     - Embedding → BiGRU → Attention → Sum → Sigmoid
  3. Train model
  4. Evaluate on held-out test set
  5. Save results

All hyperparameters are read from config.yaml → bigru section.
"""

import os
import sys
import json
import time
import tempfile
import warnings

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, callbacks

from preprocess import load_config, bigru_preprocess


# ══════════════════════════════════════════════════════════════════════════════
#  Attention Layer (Eq. 3 in the paper)
# ══════════════════════════════════════════════════════════════════════════════

class BiGRUAttentionLayer(layers.Layer):
    """Attention mechanism as described in Yuan et al. (2019), Eq. 3.

    Given BiGRU output h_t at each timestep:
      u_t = tanh(W_w * h_t + b_w)
      a_t = softmax(u_t^T * u_w)
      s_t = sum(a_t * h_t)

    Where u_w is a learnable context vector.
    """
    def __init__(self, attention_size, **kwargs):
        self.attention_size = attention_size
        super().__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch, timesteps, features)
        hidden_dim = input_shape[-1]

        self.W_w = self.add_weight(
            name="W_w", shape=(hidden_dim, self.attention_size),
            initializer="glorot_uniform", trainable=True,
        )
        self.b_w = self.add_weight(
            name="b_w", shape=(self.attention_size,),
            initializer="zeros", trainable=True,
        )
        self.u_w = self.add_weight(
            name="u_w", shape=(self.attention_size,),
            initializer="glorot_uniform", trainable=True,
        )

    def call(self, h):
        # h: (batch, timesteps, 2*gru_units)
        # u_t = tanh(W_w * h_t + b_w)  →  (batch, timesteps, attention_size)
        u = tf.nn.tanh(tf.tensordot(h, self.W_w, axes=[[2], [0]]) + self.b_w)

        # a_t = softmax(u_t^T * u_w)  →  (batch, timesteps)
        score = tf.tensordot(u, self.u_w, axes=[[2], [0]])
        a = tf.nn.softmax(score, axis=1)

        # s = sum(a_t * h_t)  →  (batch, 2*gru_units)
        # Paper Eq. 4: y* = sum_t(s_t), where s_t = a_t * h_t
        # We sum over timesteps with attention weights
        a_expanded = tf.expand_dims(a, axis=-1)  # (batch, timesteps, 1)
        s = tf.reduce_sum(h * a_expanded, axis=1)  # (batch, 2*gru_units)
        return s

    def get_config(self):
        config = super().get_config()
        config.update({"attention_size": self.attention_size})
        return config


# ══════════════════════════════════════════════════════════════════════════════
#  Build BiGRU-Attention Model (Section 3 of the paper)
# ══════════════════════════════════════════════════════════════════════════════

def build_bigru_attention_model(vocab_size, max_len, embedding_dim, gru_units,
                                 attention_size, learning_rate):
    """Build the BiGRU-Attention model as described in Yuan et al. (2019).

    Architecture (Fig. 1 in the paper):
      Input (char sequence) → Embedding → BiGRU → Attention → Sum → Sigmoid

    Key design choice from the paper: no fully connected layer before output.
    The attention output is directly passed through sigmoid for classification.
    """
    # Input: character indices of shape (max_len,)
    inp = layers.Input(shape=(max_len,), name="char_input")

    # Embedding layer (Section 3.1, step 2): map each char to embedding_dim vector
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_len,
        name="char_embedding",
    )(inp)

    # BiGRU layer (Section 3.2): captures forward and backward character info
    x = layers.Bidirectional(
        layers.GRU(gru_units, return_sequences=True, name="gru"),
        merge_mode="concat",
        name="bigru",
    )(x)
    # Output shape: (batch, max_len, 2 * gru_units)

    # Attention layer (Section 3.2, Eq. 3): assigns importance weights to chars
    x = BiGRUAttentionLayer(attention_size, name="attention")(x)
    # Output shape: (batch, 2 * gru_units)

    # Classification (Eq. 5-6): sigmoid on summed attention scores
    # Paper: "directly summed" — the attention already sums over timesteps.
    # A single output neuron with sigmoid produces probability.
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inp, outputs=out, name="BiGRU_Attention")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Training Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def train_bigru(cfg):
    """Full BiGRU-Attention pipeline.

    1. Preprocess data (character-level encoding)
    2. Build BiGRU-Attention model
    3. Train model
    4. Evaluate on held-out test set
    5. Save results
    """
    SEED = cfg["random_seed"]
    bcfg = cfg["bigru"]

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print("\n" + "=" * 60)
    print("BiGRU-ATTENTION PIPELINE (Yuan et al. 2019)")
    print("=" * 60)

    # ── 1. Preprocess ──
    print("\n── Step 1: Preprocessing ──")
    data = bigru_preprocess(cfg)

    X_char_train = data["X_char_train"]
    X_char_val = data["X_char_val"]
    X_char_test = data["X_char_test"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]
    char_vocab_size = data["char_vocab_size"]
    le = data["le"]

    print(f"\n  Vocab size: {char_vocab_size}")
    print(f"  Max URL length: {bcfg['char_max_len']}")
    print(f"  Train: {X_char_train.shape[0]} samples")
    print(f"  Val:   {X_char_val.shape[0]} samples")
    print(f"  Test:  {X_char_test.shape[0]} samples")

    # ── 2. Build Model ──
    print("\n── Step 2: Building BiGRU-Attention Model ──")
    model = build_bigru_attention_model(
        vocab_size=char_vocab_size,
        max_len=bcfg["char_max_len"],
        embedding_dim=bcfg["embedding_dim"],
        gru_units=bcfg["gru_units"],
        attention_size=bcfg["attention_size"],
        learning_rate=bcfg["learning_rate"],
    )
    model.summary()

    # ── 3. Train Model ──
    print("\n── Step 3: Training ──")
    print(f"  Batch size: {bcfg['batch_size']}")
    print(f"  Max epochs: {bcfg['max_epochs']}")
    print(f"  Learning rate: {bcfg['learning_rate']}")
    print(f"  Optimizer: {bcfg['optimizer']}")

    cb_list = []

    # Early stopping (safeguard)
    if bcfg.get("early_stopping_patience"):
        cb_list.append(callbacks.EarlyStopping(
            monitor="val_loss",
            patience=bcfg["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ))

    # Learning rate reduction
    if bcfg.get("reduce_lr_patience"):
        cb_list.append(callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=bcfg["reduce_lr_factor"],
            patience=bcfg["reduce_lr_patience"],
            min_lr=bcfg["reduce_lr_min"],
            verbose=1,
        ))

    train_start = time.time()
    history = model.fit(
        X_char_train, y_train,
        validation_data=(X_char_val, y_val),
        batch_size=bcfg["batch_size"],
        epochs=bcfg["max_epochs"],
        callbacks=cb_list,
        verbose=1,
    )
    train_time = time.time() - train_start

    epochs_trained = len(history.history["loss"])
    print(f"\n  Training completed in {train_time:.2f}s ({epochs_trained} epochs)")

    # ── 4. Evaluate on Test Set ──
    print("\n── Step 4: Evaluation on Test Set ──")

    inf_start = time.time()
    y_test_proba = model.predict(X_char_test, batch_size=bcfg["batch_size"]).flatten()
    inference_time = time.time() - inf_start

    y_test_pred = (y_test_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    f1v = f1_score(y_test, y_test_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_test_proba)
    cm = confusion_matrix(y_test, y_test_pred)

    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"\n  ── Predictive Metrics ──")
    print(f"  Accuracy  : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1v:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  TPR       : {tpr:.4f}")
    print(f"  FPR       : {fpr:.4f}")
    print(f"  FNR       : {fnr:.4f}")

    print(f"\n  Confusion Matrix:")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")

    print(f"\n  Classification Report:")
    report = classification_report(
        y_test, y_test_pred,
        target_names=le.classes_,
        digits=4,
    )
    print(report)

    # ── Computational Metrics ──
    inference_per_sample_ms = (inference_time / len(y_test)) * 1000

    # Model size on disk
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "model.keras")
        model.save(tmp_path)
        model_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)

    # Memory usage (model parameters)
    total_params = model.count_params()

    print(f"\n  ── Computational Metrics ──")
    print(f"  Training time         : {train_time:.2f}s")
    print(f"  Epochs trained        : {epochs_trained}")
    print(f"  Inference time (total): {inference_time:.4f}s")
    print(f"  Inference/sample      : {inference_per_sample_ms:.4f}ms")
    print(f"  Model size            : {model_size_mb:.4f}MB")
    print(f"  Total parameters      : {total_params:,}")

    # ── 5. Save Results ──
    print("\n── Step 5: Saving Results ──")
    out_dir = os.path.join(cfg["output_dir"], "bigru_attention",
                           f"dataset_{bcfg['dataset_id']}")
    os.makedirs(out_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(out_dir, "model.keras")
    model.save(model_path)
    print(f"  Model saved to {model_path}")

    # Save char_to_idx mapping
    char_to_idx_path = os.path.join(out_dir, "char_to_idx.json")
    with open(char_to_idx_path, "w") as f:
        json.dump(data["char_to_idx"], f, indent=2)

    # Save label encoder classes
    le_path = os.path.join(out_dir, "label_encoder_classes.json")
    with open(le_path, "w") as f:
        json.dump(le.classes_.tolist(), f, indent=2)

    # Build training history for convergence tracking
    train_history = {
        "loss": [float(v) for v in history.history["loss"]],
        "accuracy": [float(v) for v in history.history["accuracy"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
        "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
    }

    # Build results dict
    results = {
        "dataset_id": bcfg["dataset_id"],
        "model": "BiGRU-Attention (Yuan et al. 2019)",
        "architecture": {
            "char_max_len": bcfg["char_max_len"],
            "char_vocab_size": char_vocab_size,
            "embedding_dim": bcfg["embedding_dim"],
            "gru_units": bcfg["gru_units"],
            "attention_size": bcfg["attention_size"],
            "total_parameters": total_params,
        },
        "training_config": {
            "batch_size": bcfg["batch_size"],
            "max_epochs": bcfg["max_epochs"],
            "epochs_trained": epochs_trained,
            "learning_rate": bcfg["learning_rate"],
            "optimizer": bcfg["optimizer"],
        },
        "test_metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1v, 4),
            "roc_auc": round(auc, 4),
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "confusion_matrix": cm.tolist(),
        },
        "computational_metrics": {
            "training_time_s": round(train_time, 2),
            "epochs_trained": epochs_trained,
            "inference_time_total_s": round(inference_time, 4),
            "inference_ms_per_sample": round(inference_per_sample_ms, 4),
            "model_size_mb": round(model_size_mb, 4),
            "total_parameters": total_params,
        },
        "training_history": train_history,
    }

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {results_path}")

    # Save CSV summary (consistent with other pipelines)
    summary_df = pd.DataFrame([{
        "model": "BiGRU-Attention",
        "dataset_id": bcfg["dataset_id"],
        "char_max_len": bcfg["char_max_len"],
        "embedding_dim": bcfg["embedding_dim"],
        "gru_units": bcfg["gru_units"],
        "attention_size": bcfg["attention_size"],
        "batch_size": bcfg["batch_size"],
        "epochs_trained": epochs_trained,
        "learning_rate": bcfg["learning_rate"],
        "test_accuracy": round(acc, 4),
        "test_precision": round(prec, 4),
        "test_recall": round(rec, 4),
        "test_f1": round(f1v, 4),
        "test_roc_auc": round(auc, 4),
        "training_time_s": round(train_time, 2),
        "inference_ms_per_sample": round(inference_per_sample_ms, 4),
        "model_size_mb": round(model_size_mb, 4),
        "total_parameters": total_params,
    }])
    summary_df.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)

    # Save training history separately for plotting
    history_path = os.path.join(out_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(train_history, f, indent=2)
    print(f"  Training history saved to {history_path}")

    print(f"\n  All outputs saved to {out_dir}/")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Standalone entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = load_config()

    # Allow overriding dataset_id from CLI:
    #   python bigru_pipeline.py 7
    if len(sys.argv) > 1:
        cfg["bigru"]["dataset_id"] = sys.argv[1]
        print(f"Dataset override: {sys.argv[1]}")

    train_bigru(cfg)
