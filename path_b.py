"""
Path B: CNN-BiLSTM Neural Network
==================================
Processes the full 53-feature vector as a 1-D sequence:
  Input(53,1) -> Conv1D(64,3) -> Conv1D(128,5) -> BiLSTM(128) ->
  GlobalMaxPool -> Dense(1, sigmoid)

Tracks per-epoch validation loss for the adaptive weighting mechanism.
"""

import os
import sys
import time
import yaml
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seeds(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_split(split_dir, feature_cols, label_col, positive_label):
    """Load train / val / test CSVs and return X, y arrays."""
    data = {}
    for role in ("train", "val", "test"):
        fp = os.path.join(split_dir, f"{role}.csv")
        df = pd.read_csv(fp)
        X = df[feature_cols].values.astype(np.float32)
        y = (df[label_col] == positive_label).astype(np.float32).values
        data[role] = (X, y)
    return data


# ------------------------------------------------------------------ #
#  Model Definition                                                   #
# ------------------------------------------------------------------ #

def build_cnn_bilstm(n_features, pb_cfg):
    """Build the CNN-BiLSTM model following the thesis architecture."""
    inp = keras.Input(shape=(n_features, 1), name="url_features")

    # Conv1D block 1
    x = layers.Conv1D(
        filters=pb_cfg["conv1_filters"],
        kernel_size=pb_cfg["conv1_kernel"],
        activation="relu",
        padding="same",
        name="conv1",
    )(inp)

    # Conv1D block 2
    x = layers.Conv1D(
        filters=pb_cfg["conv2_filters"],
        kernel_size=pb_cfg["conv2_kernel"],
        activation="relu",
        padding="same",
        name="conv2",
    )(x)

    # BiLSTM
    x = layers.Bidirectional(
        layers.LSTM(pb_cfg["bilstm_units"], return_sequences=True, name="lstm"),
        name="bilstm",
    )(x)

    # Global max pooling
    x = layers.GlobalMaxPooling1D(name="global_max_pool")(x)

    # Output
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="CNN_BiLSTM")
    return model


# ------------------------------------------------------------------ #
#  Per-epoch Validation Loss Logger (for adaptive weighting)          #
# ------------------------------------------------------------------ #

class ValidationLossLogger(callbacks.Callback):
    """Records per-epoch validation log-loss for the adaptive mechanism."""

    def __init__(self):
        super().__init__()
        self.epoch_val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss is not None:
            self.epoch_val_losses.append(float(val_loss))


# ------------------------------------------------------------------ #
#  Training                                                           #
# ------------------------------------------------------------------ #

def train_path_b(cfg, return_model=True):
    """Full Path B pipeline.  Returns dict with model artefacts + metrics."""
    seed = cfg["random_seed"]
    set_seeds(seed)

    split_dir = os.path.join("processed_datasets", cfg["split_dir"])
    feature_cols = cfg["feature_columns"]
    label_col = cfg["label_column"]
    pos_label = cfg["positive_label"]
    pb = cfg["path_b"]

    # 1. Load data
    print("Loading data …")
    data = load_split(split_dir, feature_cols, label_col, pos_label)
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    n_features = X_train.shape[1]

    # Reshape for Conv1D: (samples, timesteps, channels=1)
    X_train_3d = X_train[..., np.newaxis]
    X_val_3d = X_val[..., np.newaxis]
    X_test_3d = X_test[..., np.newaxis]

    # 2. Build model
    model = build_cnn_bilstm(n_features, pb)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=pb["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # 3. Callbacks
    out_dir = os.path.join(cfg["output_dir"], cfg["split_dir"])
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "cnn_bilstm_best.keras")

    val_logger = ValidationLossLogger()

    cb_list = [
        val_logger,
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=pb["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # 4. Train
    print("\nTraining CNN-BiLSTM …")
    t0 = time.time()
    history = model.fit(
        X_train_3d, y_train,
        validation_data=(X_val_3d, y_val),
        epochs=pb["max_epochs"],
        batch_size=pb["batch_size"],
        callbacks=cb_list,
        verbose=1,
    )
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.2f}s")

    # Convergence: epoch where val_loss reached its best
    val_losses = val_logger.epoch_val_losses
    best_epoch = int(np.argmin(val_losses)) + 1  # 1-indexed
    print(f"  Best epoch (convergence): {best_epoch}")

    # 5. Validation probabilities & loss
    val_proba = model.predict(X_val_3d, batch_size=pb["batch_size"]).flatten()
    from sklearn.metrics import log_loss as sk_log_loss
    val_loss = sk_log_loss(y_val, val_proba)
    print(f"  Validation log-loss (L_B): {val_loss:.6f}")

    # 6. Test evaluation
    test_proba = model.predict(X_test_3d, batch_size=pb["batch_size"]).flatten()
    test_preds = (test_proba >= 0.5).astype(int)

    t0 = time.time()
    _ = model.predict(X_test_3d, batch_size=pb["batch_size"])
    inference_total = time.time() - t0
    inference_per_sample = (inference_total / len(X_test)) * 1000  # ms

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )
    metrics = {
        "accuracy": accuracy_score(y_test, test_preds),
        "precision": precision_score(y_test, test_preds),
        "recall": recall_score(y_test, test_preds),
        "f1": f1_score(y_test, test_preds),
        "roc_auc": roc_auc_score(y_test, test_proba),
        "confusion_matrix": confusion_matrix(y_test, test_preds).tolist(),
        "train_time_s": train_time,
        "inference_ms_per_sample": inference_per_sample,
        "best_epoch": best_epoch,
        "total_epochs": len(val_losses),
    }

    # Model size
    model_size_bytes = os.path.getsize(ckpt_path) if os.path.exists(ckpt_path) else 0
    metrics["model_size_mb"] = model_size_bytes / (1024 * 1024)

    print("\n--- Path B Test Metrics ---")
    for k_name, v in metrics.items():
        if k_name != "confusion_matrix":
            fmt = f"  {k_name}: {v:.6f}" if isinstance(v, float) else f"  {k_name}: {v}"
            print(fmt)
    print(f"  confusion_matrix:\n    {metrics['confusion_matrix']}")

    result = {
        "val_proba": val_proba,
        "val_loss": val_loss,
        "epoch_val_losses": val_losses,
        "test_proba": test_proba,
        "test_preds": test_preds,
        "y_val": y_val,
        "y_test": y_test,
        "metrics": metrics,
        "train_time": train_time,
        "history": history.history,
    }
    if return_model:
        result["model"] = model

    print(f"\nArtefacts saved to {out_dir}")
    return result


# ------------------------------------------------------------------ #
#  CLI entry-point                                                    #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(cfg_path)
    results = train_path_b(cfg)
    print("\nPath B complete.")
