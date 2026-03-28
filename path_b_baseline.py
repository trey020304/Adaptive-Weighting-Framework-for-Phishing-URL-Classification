"""
Path B Baseline: Hybrid CNN-BiLSTM
====================================
Based on Princeton Vishal J et al. (2025) with enhancements:
  1. Dual-input hybrid model: char-level CNN-BiLSTM + tabular numeric features
  2. Token-level branch alongside char-level
  3. Attention mechanism

Can be run standalone or imported by adaptive_hybrid.py.
Preprocessing is handled by preprocess.py; hyperparameters are in config.yaml.
"""

import os, warnings, time, tempfile
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, classification_report, confusion_matrix
)

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, callbacks

from preprocess import load_config, princeton_improved_preprocess


# ──────────────────────────────────────────────
# ATTENTION LAYER
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# VALIDATION LOSS LOGGER (for adaptive weighting)
# ──────────────────────────────────────────────
class ValidationLossLogger(callbacks.Callback):
    """Records per-epoch validation loss L_B(t) for the adaptive mechanism."""
    def __init__(self):
        super().__init__()
        self.epoch_val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss is not None:
            self.epoch_val_losses.append(float(val_loss))


# ──────────────────────────────────────────────
# TRAIN FUNCTION (callable by adaptive_hybrid)
# ──────────────────────────────────────────────

def train_path_b(cfg):
    """Full Path B pipeline. Returns dict with model artefacts + metrics.

    Can be called standalone or by adaptive_hybrid.py.
    Outputs P_B(x) probability scores, validation log-loss L_B,
    and per-epoch validation losses L_B(t) for adaptive weighting.
    """
    SEED = cfg["random_seed"]
    icfg = cfg["princeton_improved"]
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # ── 1. Load & preprocess ──
    data = princeton_improved_preprocess(cfg)

    X_char_train = data["X_char_train"]
    X_char_val   = data["X_char_val"]
    X_char_test  = data["X_char_test"]

    X_token_train = data["X_token_train"]
    X_token_val   = data["X_token_val"]
    X_token_test  = data["X_token_test"]

    X_tab_train = data["X_tab_train"]
    X_tab_val   = data["X_tab_val"]
    X_tab_test  = data["X_tab_test"]

    y_train = data["y_train"]
    y_val   = data["y_val"]
    y_test  = data["y_test"]

    le = data["le"]
    CHAR_VOCAB_SIZE  = data["char_vocab_size"]
    TOKEN_VOCAB_SIZE = data["token_vocab_size"]
    NUM_FEATURES     = data["num_tab_features"]

    char_to_idx  = data["char_to_idx"]
    token_vocab  = data["token_vocab"]
    tab_scaler   = data["tab_scaler"]
    tab_feature_cols = data["tab_feature_cols"]

    MAX_LEN    = icfg["char_max_len"]
    MAX_TOKENS = icfg["token_max_len"]
    DROPOUT    = icfg["dropout_rate"]

    # ── 2. Build hybrid dual-input model ──
    #    Branch A: Char-level CNN-BiLSTM + Attention
    #    Branch B: Token-level Embedding + BiLSTM
    #    Branch C: Tabular numeric features (Dense)
    #    → Concatenate → Dense → Sigmoid

    def build_hybrid_model():
        # ── Branch A: Char-level CNN-BiLSTM ──
        char_inp = layers.Input(shape=(MAX_LEN,), name="char_input")
        cx = layers.Embedding(
            CHAR_VOCAB_SIZE, icfg["char_embedding_dim"],
            input_length=MAX_LEN, name="char_embed",
        )(char_inp)

        cx = layers.Conv1D(icfg["char_conv1_filters"], icfg["char_conv1_kernel"],
                           padding="same", activation="relu", name="char_conv1")(cx)
        cx = layers.BatchNormalization()(cx)
        cx = layers.MaxPooling1D(2)(cx)
        cx = layers.Dropout(DROPOUT)(cx)

        cx = layers.Conv1D(icfg["char_conv2_filters"], icfg["char_conv2_kernel"],
                           padding="same", activation="relu", name="char_conv2")(cx)
        cx = layers.BatchNormalization()(cx)
        cx = layers.MaxPooling1D(2)(cx)
        cx = layers.Dropout(DROPOUT)(cx)

        cx = layers.Bidirectional(
            layers.LSTM(icfg["char_bilstm_units"], return_sequences=True),
            name="char_bilstm",
        )(cx)
        cx = layers.Dropout(DROPOUT)(cx)
        cx = AttentionLayer(name="char_attention")(cx)

        # ── Branch B: Token-level BiLSTM ──
        token_inp = layers.Input(shape=(MAX_TOKENS,), name="token_input")
        tx = layers.Embedding(
            TOKEN_VOCAB_SIZE, icfg["token_embedding_dim"],
            input_length=MAX_TOKENS, name="token_embed",
        )(token_inp)
        tx = layers.Bidirectional(
            layers.LSTM(icfg["token_bilstm_units"], return_sequences=True),
            name="token_bilstm",
        )(tx)
        tx = layers.Dropout(DROPOUT)(tx)
        tx = AttentionLayer(name="token_attention")(tx)

        # ── Branch C: Tabular features ──
        tab_inp = layers.Input(shape=(NUM_FEATURES,), name="tab_input")
        fx = layers.Dense(icfg["tab_dense1_units"], activation="relu", name="tab_dense1")(tab_inp)
        fx = layers.BatchNormalization()(fx)
        fx = layers.Dropout(DROPOUT)(fx)
        fx = layers.Dense(icfg["tab_dense2_units"], activation="relu", name="tab_dense2")(fx)
        fx = layers.Dropout(DROPOUT)(fx)

        # ── Merge all branches ──
        merged = layers.Concatenate(name="merge")([cx, tx, fx])
        mx = layers.Dense(icfg["merge_dense1_units"], activation="relu")(merged)
        mx = layers.Dropout(DROPOUT)(mx)
        mx = layers.Dense(icfg["merge_dense2_units"], activation="relu")(mx)
        mx = layers.Dropout(DROPOUT)(mx)
        out = layers.Dense(1, activation="sigmoid", name="output")(mx)

        model = Model(inputs=[char_inp, token_inp, tab_inp], outputs=out,
                      name="Hybrid_CNN_BiLSTM")
        return model

    model = build_hybrid_model()
    model.summary()

    # ── 3. Compile & train ──
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=icfg["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print("\n" + "=" * 60)
    print("TRAINING — Hybrid CNN-BiLSTM")
    print("=" * 60)

    val_logger = ValidationLossLogger()

    cbs = [
        val_logger,
        callbacks.EarlyStopping(
            monitor="val_loss", patience=icfg["early_stopping_patience"],
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=icfg["reduce_lr_factor"],
            patience=icfg["reduce_lr_patience"],
            min_lr=icfg["reduce_lr_min"], verbose=1,
        ),
    ]

    train_inputs = [X_char_train, X_token_train, X_tab_train]
    val_inputs   = [X_char_val,   X_token_val,   X_tab_val]
    test_inputs  = [X_char_test,  X_token_test,  X_tab_test]

    train_start = time.time()
    history = model.fit(
        train_inputs, y_train,
        validation_data=(val_inputs, y_val),
        epochs=icfg["max_epochs"],
        batch_size=icfg["batch_size"],
        callbacks=cbs,
        verbose=1,
    )
    train_time = time.time() - train_start

    # ── 4. Validation metrics (for adaptive weighting: L_B) ──
    val_proba = model.predict(val_inputs, batch_size=64, verbose=0).flatten()
    val_loss_value = log_loss(y_val, val_proba)
    print(f"\n  Validation log-loss (L_B): {val_loss_value:.6f}")

    # ── 5. Evaluate on test set ──
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET — Hybrid CNN-BiLSTM")
    print("=" * 60)

    test_proba = model.predict(test_inputs, batch_size=64).flatten()
    y_pred = (test_proba >= 0.5).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1v  = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, test_proba)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1v:.4f}")
    print(f"  ROC AUC  : {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")
    print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

    # ── 6. Computational metrics ──
    inf_start = time.time()
    for _ in range(3):
        _ = model.predict(test_inputs, batch_size=64, verbose=0)
    inference_time = (time.time() - inf_start) / 3
    inference_per_sample_ms = (inference_time / len(y_test)) * 1000

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "model.keras")
        model.save(tmp_path)
        model_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)

    print(f"\n  ── Computational Metrics ──")
    print(f"  Training time   : {train_time:.2f}s")
    print(f"  Inference/sample: {inference_per_sample_ms:.4f}ms")
    print(f"  Model size      : {model_size_mb:.2f}MB")

    # ── 7. Training history ──
    best_epoch = np.argmin(history.history["val_loss"])
    total_epochs = len(history.history["loss"])
    convergence_rate = (best_epoch + 1) / total_epochs

    print(f"\n  Best epoch: {best_epoch + 1} / {total_epochs}")
    print(f"  Convergence rate: {convergence_rate:.4f} (lower = faster)")
    print(f"  Train loss: {history.history['loss'][best_epoch]:.4f}  |  "
          f"Train acc: {history.history['accuracy'][best_epoch]:.4f}")
    print(f"  Val   loss: {history.history['val_loss'][best_epoch]:.4f}  |  "
          f"Val   acc: {history.history['val_accuracy'][best_epoch]:.4f}")

    # ── 8. Save results ──
    OUT_DIR = Path(f"results/path_b_baseline/dataset_{icfg['dataset_id']}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "model": "Hybrid CNN-BiLSTM (improved)",
        "dataset": f"[{icfg['dataset_id']}]",
        "test_accuracy": round(acc, 4),
        "test_precision": round(prec, 4),
        "test_recall": round(rec, 4),
        "test_f1": round(f1v, 4),
        "test_roc_auc": round(auc, 4),
        "val_log_loss": round(val_loss_value, 6),
        "best_epoch": int(best_epoch + 1),
        "total_epochs": total_epochs,
        "convergence_rate": round(convergence_rate, 4),
        "train_time_s": round(train_time, 2),
        "inference_ms_per_sample": round(inference_per_sample_ms, 4),
        "model_size_mb": round(model_size_mb, 2),
    }

    pd.DataFrame([results]).to_csv(OUT_DIR / "metrics_summary.csv", index=False)
    pd.DataFrame(history.history).to_csv(OUT_DIR / "training_history.csv", index=False)

    print(f"\nResults saved to {OUT_DIR}/")

    # ── 9. Return dict for adaptive hybrid ──
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1v,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "inference_ms_per_sample": inference_per_sample_ms,
        "model_size_mb": model_size_mb,
        "best_epoch": int(best_epoch + 1),
        "total_epochs": total_epochs,
        "train_time_s": train_time,
    }

    return {
        "model": model,
        "val_proba": val_proba,
        "val_loss": val_loss_value,
        "epoch_val_losses": val_logger.epoch_val_losses,
        "test_proba": test_proba,
        "y_val": y_val,
        "y_test": y_test,
        "train_time": train_time,
        "metrics": metrics,
        "history": history.history,
        "le": le,
        "char_to_idx": char_to_idx,
        "token_vocab": token_vocab,
        "tab_scaler": tab_scaler,
        "tab_feature_cols": tab_feature_cols,
    }


# ──────────────────────────────────────────────
# CLI ENTRY-POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    cfg = load_config("config.yaml")
    result = train_path_b(cfg)
    print("Done.")
