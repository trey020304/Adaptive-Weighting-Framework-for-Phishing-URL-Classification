"""
ODAE-WPDC Pipeline: Optimal Deep Autoencoder for Website Phishing
Detection and Classification
=================================================================
Reproduction of: Alshehri et al. (2022), Applied Sciences, 12, 7441.

Pipeline stages:
  1. Data pre-processing (preprocess.py)
  2. AAA (Artificial Algae Algorithm) feature selection
  3. DAE (Deep Autoencoder) classification with IWO hyperparameter tuning
  4. K-fold cross-validation evaluation

All hyperparameters are read from config.yaml → odae_wpdc section.
"""

import os
import sys
import json
import time
import tempfile
import warnings
import copy
import tracemalloc

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, callbacks

from preprocess import load_config, odae_wpdc_preprocess


# ══════════════════════════════════════════════════════════════════════════════
#  AAA — Artificial Algae Algorithm for Feature Selection
# ══════════════════════════════════════════════════════════════════════════════

class ArtificialAlgaeAlgorithm:
    """Binary AAA for feature subset selection (Uymaz et al. 2015).

    Each algae colony is a binary vector of length D (number of features).
    Fitness = alpha * classifier_error_rate + (1 - alpha) * |R| / |C|
    where |R| = selected count, |C| = total features.
    """

    def __init__(self, X_train, y_train, X_val, y_val, aaa_cfg, seed=42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.N = aaa_cfg["population_size"]
        self.max_iter = aaa_cfg["max_iterations"]
        self.energy_budget = aaa_cfg["energy"]
        self.alpha = aaa_cfg["alpha"]
        self.beta = 1.0 - self.alpha
        self.k_neighbors = aaa_cfg["knn_neighbors"]
        self.adaptation_rate = aaa_cfg["adaptation_rate"]
        self.min_features = aaa_cfg["min_features"]

        self.D = X_train.shape[1]
        self.rng = np.random.RandomState(seed)

    def _ensure_min_features(self, colony):
        """Ensure at least min_features are selected in a colony."""
        if colony.sum() < self.min_features:
            off_indices = np.where(colony == 0)[0]
            n_needed = self.min_features - int(colony.sum())
            chosen = self.rng.choice(off_indices, size=min(n_needed, len(off_indices)),
                                     replace=False)
            colony[chosen] = 1
        return colony

    def _fitness(self, colony):
        """Evaluate fitness of a single colony (lower is better).

        Fitness = alpha * error_rate + beta * (|R| / |C|)
        """
        selected = np.where(colony == 1)[0]
        if len(selected) == 0:
            return 1.0  # worst fitness

        X_tr = self.X_train[:, selected]
        X_va = self.X_val[:, selected]

        knn = KNeighborsClassifier(n_neighbors=self.k_neighbors)
        knn.fit(X_tr, self.y_train)
        preds = knn.predict(X_va)
        error_rate = 1.0 - accuracy_score(self.y_val, preds)

        subset_ratio = len(selected) / self.D
        return self.alpha * error_rate + self.beta * subset_ratio

    def _helical_movement(self, colony, best_colony):
        """Helical movement phase: move colony toward the best solution."""
        new_colony = colony.copy()
        for d in range(self.D):
            if self.rng.random() < 0.5:
                new_colony[d] = best_colony[d]
            else:
                if self.rng.random() < 0.3:
                    new_colony[d] = 1 - new_colony[d]
        return self._ensure_min_features(new_colony)

    def _reproduction(self, population, fitnesses):
        """Evolutionary / reproduction stage: best colony replaces worst."""
        best_idx = np.argmin(fitnesses)
        worst_idx = np.argmax(fitnesses)
        if best_idx != worst_idx:
            child = population[best_idx].copy()
            # Introduce small mutation
            mutation_mask = self.rng.random(self.D) < (1.0 / self.D)
            child[mutation_mask] = 1 - child[mutation_mask]
            child = self._ensure_min_features(child)
            population[worst_idx] = child
        return population

    def _adaptation(self, population, fitnesses):
        """Adaptation stage: probabilistically replace worst colony."""
        if self.rng.random() < self.adaptation_rate:
            worst_idx = np.argmax(fitnesses)
            new_colony = self.rng.randint(0, 2, size=self.D).astype(np.int32)
            new_colony = self._ensure_min_features(new_colony)
            population[worst_idx] = new_colony
        return population

    def run(self):
        """Execute AAA feature selection. Returns (best_mask, best_fitness, history)."""
        print("\n" + "=" * 60)
        print("AAA FEATURE SELECTION")
        print("=" * 60)
        print(f"  Population: {self.N}, Iterations: {self.max_iter}, "
              f"Features: {self.D}")

        # Initialize population
        population = np.array([
            self._ensure_min_features(
                self.rng.randint(0, 2, size=self.D).astype(np.int32)
            ) for _ in range(self.N)
        ])

        fitnesses = np.array([self._fitness(c) for c in population])
        best_idx = np.argmin(fitnesses)
        best_colony = population[best_idx].copy()
        best_fitness = fitnesses[best_idx]

        history = []

        for iteration in range(self.max_iter):
            # Helical movement phase
            for i in range(self.N):
                energy = self.energy_budget
                while energy > 0:
                    new_colony = self._helical_movement(population[i], best_colony)
                    new_fit = self._fitness(new_colony)
                    if new_fit < fitnesses[i]:
                        population[i] = new_colony
                        fitnesses[i] = new_fit
                    energy -= 1

            # Reproduction phase
            population = self._reproduction(population, fitnesses)
            fitnesses = np.array([self._fitness(c) for c in population])

            # Adaptation phase
            population = self._adaptation(population, fitnesses)
            fitnesses = np.array([self._fitness(c) for c in population])

            # Track best
            gen_best_idx = np.argmin(fitnesses)
            if fitnesses[gen_best_idx] < best_fitness:
                best_colony = population[gen_best_idx].copy()
                best_fitness = fitnesses[gen_best_idx]

            n_selected = int(best_colony.sum())
            history.append({
                "iteration": iteration + 1,
                "best_fitness": float(best_fitness),
                "n_features": n_selected,
            })

            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"  Iter {iteration + 1:3d}/{self.max_iter} — "
                      f"fitness: {best_fitness:.6f}, features: {n_selected}/{self.D}")

        selected_idx = np.where(best_colony == 1)[0]
        print(f"\n  AAA complete — selected {len(selected_idx)}/{self.D} features")
        print(f"  Best fitness: {best_fitness:.6f}")

        return best_colony, best_fitness, history


# ══════════════════════════════════════════════════════════════════════════════
#  DAE — Deep Autoencoder Classifier
# ══════════════════════════════════════════════════════════════════════════════

def build_dae_classifier(input_dim, encoder_layers, activation, dropout_rate,
                         learning_rate):
    """Build a Deep Autoencoder with a classification head.

    Architecture (paper Section 3.3):
      Encoder: input → h1 → h2 → bottleneck
      Decoder: bottleneck → h2' → h1' → reconstruction
      Classifier head: bottleneck → softmax(2)

    Returns (full_model, encoder_model) where:
      - full_model: end-to-end classifier (input → softmax)
      - encoder_model: encoder only (input → bottleneck)
    """
    inp = layers.Input(shape=(input_dim,), name="dae_input")

    # ── Encoder ──
    x = inp
    for i, units in enumerate(encoder_layers):
        x = layers.Dense(units, activation=activation,
                         name=f"encoder_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"encoder_drop_{i}")(x)

    bottleneck = x

    # ── Decoder (mirror of encoder, for pretraining) ──
    dec = bottleneck
    for i, units in enumerate(reversed(encoder_layers[:-1])):
        dec = layers.Dense(units, activation=activation,
                           name=f"decoder_{i}")(dec)
        dec = layers.Dropout(dropout_rate, name=f"decoder_drop_{i}")(dec)
    reconstruction = layers.Dense(input_dim, activation="sigmoid",
                                  name="reconstruction")(dec)

    # ── Classifier head (from bottleneck) ──
    clf = layers.Dense(64, activation=activation, name="clf_dense")(bottleneck)
    clf = layers.Dropout(dropout_rate, name="clf_drop")(clf)
    clf_out = layers.Dense(1, activation="sigmoid", name="clf_output")(clf)

    # Full model: dual output (reconstruction + classification)
    pretrain_model = Model(inputs=inp, outputs=reconstruction,
                           name="DAE_pretrain")
    pretrain_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )

    # Classification model
    classifier_model = Model(inputs=inp, outputs=clf_out,
                             name="DAE_classifier")
    classifier_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return pretrain_model, classifier_model


def pretrain_dae(pretrain_model, X_train, dae_cfg):
    """Greedy layer-wise unsupervised pretraining (paper Section 3.3)."""
    print("\n  Pretraining DAE (unsupervised reconstruction) ...")
    pretrain_model.fit(
        X_train, X_train,
        epochs=dae_cfg["pretrain_epochs"],
        batch_size=dae_cfg["batch_size"],
        validation_split=0.1,
        verbose=0,
    )
    print("  Pretraining complete.")
    return pretrain_model


def finetune_dae(classifier_model, pretrain_model, X_train, y_train,
                 X_val, y_val, dae_cfg):
    """Supervised fine-tuning of the DAE classifier (paper Section 3.3).

    Transfers encoder weights from pretrained model, then trains
    end-to-end with classification loss.
    """
    # Transfer encoder weights from pretrained model
    for layer in classifier_model.layers:
        if layer.name.startswith("encoder_"):
            try:
                pretrained_layer = pretrain_model.get_layer(layer.name)
                layer.set_weights(pretrained_layer.get_weights())
            except ValueError:
                pass

    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=dae_cfg["early_stopping_patience"],
        restore_best_weights=True,
    )

    print("  Fine-tuning DAE classifier ...")
    history = classifier_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=dae_cfg["finetune_epochs"],
        batch_size=dae_cfg["batch_size"],
        callbacks=[es],
        verbose=0,
    )

    best_val_loss = min(history.history["val_loss"])
    best_epoch = np.argmin(history.history["val_loss"]) + 1
    print(f"  Fine-tuning complete — best val_loss: {best_val_loss:.6f} "
          f"(epoch {best_epoch})")

    return classifier_model, history


# ══════════════════════════════════════════════════════════════════════════════
#  IWO — Invasive Weed Optimization for DAE Hyperparameters
# ══════════════════════════════════════════════════════════════════════════════

class InvasiveWeedOptimization:
    """IWO algorithm for tuning DAE hyperparameters (paper Section 3.4).

    Each weed represents a set of DAE hyperparameters:
      [encoder_layer1, encoder_layer2, encoder_layer3, learning_rate, dropout_rate]

    Fitness = classification error rate on validation set.
    """

    def __init__(self, X_train, y_train, X_val, y_val, iwo_cfg, dae_cfg, seed=42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.iwo_cfg = iwo_cfg
        self.dae_cfg = dae_cfg
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Search space bounds
        ss = iwo_cfg["search_space"]
        self.bounds = [
            ss["encoder_layer1"],
            ss["encoder_layer2"],
            ss["encoder_layer3"],
            ss["learning_rate"],
            ss["dropout_rate"],
        ]
        self.dim = len(self.bounds)

        # IWO parameters
        self.pop_size = iwo_cfg["population_size"]
        self.max_iter = iwo_cfg["max_iterations"]
        self.max_pop = iwo_cfg["max_population"]
        self.smin = iwo_cfg["smin"]
        self.smax = iwo_cfg["smax"]
        self.sigma_init = iwo_cfg["sigma_init"]
        self.sigma_final = iwo_cfg["sigma_final"]
        self.mod_index = iwo_cfg["modulation_index"]

    def _decode_weed(self, weed):
        """Decode continuous weed vector into DAE hyperparameters."""
        encoder_layers = [
            max(8, int(round(weed[0]))),
            max(4, int(round(weed[1]))),
            max(4, int(round(weed[2]))),
        ]
        # Ensure descending layer sizes
        encoder_layers[1] = min(encoder_layers[1], encoder_layers[0])
        encoder_layers[2] = min(encoder_layers[2], encoder_layers[1])

        lr = float(np.clip(weed[3], self.bounds[3][0], self.bounds[3][1]))
        dropout = float(np.clip(weed[4], self.bounds[4][0], self.bounds[4][1]))

        return encoder_layers, lr, dropout

    def _fitness(self, weed):
        """Evaluate a weed's fitness: train a small DAE and return error rate."""
        encoder_layers, lr, dropout = self._decode_weed(weed)

        input_dim = self.X_train.shape[1]
        activation = self.dae_cfg["activation"]

        # Build and train a quick DAE with reduced epochs for speed
        pretrain_model, classifier_model = build_dae_classifier(
            input_dim, encoder_layers, activation, dropout, lr
        )

        # Quick pretrain (reduced epochs for optimization speed)
        pretrain_model.fit(
            self.X_train, self.X_train,
            epochs=max(5, self.dae_cfg["pretrain_epochs"] // 5),
            batch_size=self.dae_cfg["batch_size"],
            verbose=0,
        )

        # Transfer weights
        for layer in classifier_model.layers:
            if layer.name.startswith("encoder_"):
                try:
                    pretrained_layer = pretrain_model.get_layer(layer.name)
                    layer.set_weights(pretrained_layer.get_weights())
                except ValueError:
                    pass

        # Quick fine-tune
        es = callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        classifier_model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=max(10, self.dae_cfg["finetune_epochs"] // 5),
            batch_size=self.dae_cfg["batch_size"],
            callbacks=[es],
            verbose=0,
        )

        # Evaluate
        proba = classifier_model.predict(self.X_val, verbose=0).flatten()
        preds = (proba >= 0.5).astype(int)
        error_rate = 1.0 - accuracy_score(self.y_val, preds)

        # Clean up
        del pretrain_model, classifier_model
        tf.keras.backend.clear_session()

        return error_rate

    def _clip_weed(self, weed):
        """Clip weed values to search space bounds."""
        for d in range(self.dim):
            weed[d] = np.clip(weed[d], self.bounds[d][0], self.bounds[d][1])
        return weed

    def run(self):
        """Execute IWO optimization. Returns (best_params, best_fitness, history)."""
        print("\n" + "=" * 60)
        print("IWO HYPERPARAMETER OPTIMIZATION")
        print("=" * 60)
        print(f"  Population: {self.pop_size}, Max iterations: {self.max_iter}, "
              f"Max pop: {self.max_pop}")

        # Initialize population randomly
        population = []
        for _ in range(self.pop_size):
            weed = np.array([
                self.rng.uniform(b[0], b[1]) for b in self.bounds
            ])
            population.append(weed)

        fitnesses = [self._fitness(w) for w in population]
        best_idx = np.argmin(fitnesses)
        best_weed = population[best_idx].copy()
        best_fitness = fitnesses[best_idx]

        history = []

        for iteration in range(self.max_iter):
            # Compute sigma for this iteration (Eq. 9)
            sigma_cur = (
                ((self.max_iter - iteration) ** self.mod_index
                 / self.max_iter ** self.mod_index)
                * (self.sigma_init - self.sigma_final)
                + self.sigma_final
            )

            f_best = min(fitnesses)
            f_worst = max(fitnesses)
            f_range = f_worst - f_best if f_worst > f_best else 1e-10

            new_weeds = []
            new_fitnesses = []

            for i, weed in enumerate(population):
                # Eq. 8: number of seeds based on fitness
                n_seeds = int(round(
                    (f_worst - fitnesses[i]) / f_range
                    * (self.smax - self.smin) + self.smin
                ))
                n_seeds = max(self.smin, min(self.smax, n_seeds))

                for _ in range(n_seeds):
                    # Spatial dispersal: seed = parent + N(0, sigma)
                    seed_weed = weed + self.rng.normal(0, sigma_cur, size=self.dim)
                    seed_weed = self._clip_weed(seed_weed)
                    new_weeds.append(seed_weed)

            # Evaluate new seeds
            new_fitnesses = [self._fitness(w) for w in new_weeds]

            # Combine parent and offspring populations
            all_weeds = population + new_weeds
            all_fitnesses = fitnesses + new_fitnesses

            # Competitive exclusion: keep only max_population best
            if len(all_weeds) > self.max_pop:
                sorted_indices = np.argsort(all_fitnesses)[:self.max_pop]
                population = [all_weeds[i] for i in sorted_indices]
                fitnesses = [all_fitnesses[i] for i in sorted_indices]
            else:
                population = all_weeds
                fitnesses = all_fitnesses

            # Track best
            gen_best_idx = np.argmin(fitnesses)
            if fitnesses[gen_best_idx] < best_fitness:
                best_weed = population[gen_best_idx].copy()
                best_fitness = fitnesses[gen_best_idx]

            encoder_layers, lr, dropout = self._decode_weed(best_weed)
            history.append({
                "iteration": iteration + 1,
                "best_fitness": float(best_fitness),
                "encoder_layers": encoder_layers,
                "learning_rate": lr,
                "dropout_rate": dropout,
                "sigma": float(sigma_cur),
                "population_size": len(population),
            })

            print(f"  Iter {iteration + 1:3d}/{self.max_iter} — "
                  f"error: {best_fitness:.6f}, layers: {encoder_layers}, "
                  f"lr: {lr:.6f}, drop: {dropout:.3f}, σ: {sigma_cur:.4f}")

        encoder_layers, lr, dropout = self._decode_weed(best_weed)
        print(f"\n  IWO complete — best error rate: {best_fitness:.6f}")
        print(f"  Optimal DAE config: layers={encoder_layers}, lr={lr:.6f}, "
              f"dropout={dropout:.3f}")

        return {
            "encoder_layers": encoder_layers,
            "learning_rate": lr,
            "dropout_rate": dropout,
            "fitness": best_fitness,
        }, history


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(y_true, proba, name="Model"):
    """Compute and print predictive classification metrics."""
    preds = (proba >= 0.5).astype(int)
    m = {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
    }
    print(f"\n--- {name} Predictive Metrics ---")
    for k, v in m.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.6f}")
    print(f"  confusion_matrix:\n    {m['confusion_matrix']}")
    return m


# ══════════════════════════════════════════════════════════════════════════════
#  Full ODAE-WPDC Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def train_odae_wpdc(cfg):
    """Full ODAE-WPDC pipeline with k-fold cross-validation.

    Steps:
      1. Preprocess data (remove nulls, scale)
      2. K-fold CV loop:
         a. AAA feature selection on fold's train/val
         b. IWO hyperparameter optimization for DAE
         c. Train final DAE with optimal features + hyperparams
         d. Evaluate on fold's test set
      3. Aggregate cross-validation results
    """
    SEED = cfg["random_seed"]
    ocfg = cfg["odae_wpdc"]
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # ── 1. Preprocess ──
    print("\n" + "=" * 60)
    print("ODAE-WPDC PIPELINE")
    print("=" * 60)

    data = odae_wpdc_preprocess(cfg)
    X = data["X"]
    y = data["y"]
    feature_names = data["feature_names"]
    le = data["le"]
    scaler = data["scaler"]

    n_folds = ocfg["cv_folds"]
    aaa_cfg = ocfg["aaa"]
    dae_cfg = ocfg["dae"]
    iwo_cfg = ocfg["iwo"]

    print(f"\n  Total samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  Cross-validation folds: {n_folds}")

    # ── 2. K-fold cross-validation ──
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_results = []
    all_aaa_histories = []
    all_iwo_histories = []

    overall_start = time.time()

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        fold_start = time.time()
        print(f"\n{'─' * 60}")
        print(f"  FOLD {fold}/{n_folds}")
        print(f"{'─' * 60}")

        X_fold_train, X_fold_test = X[train_idx], X[test_idx]
        y_fold_train, y_fold_test = y[train_idx], y[test_idx]

        # Split fold training into train/val for AAA and IWO
        from sklearn.model_selection import train_test_split
        val_ratio = ocfg["val_size"]
        X_train, X_val, y_train, y_val = train_test_split(
            X_fold_train, y_fold_train,
            test_size=val_ratio, random_state=SEED, stratify=y_fold_train
        )

        print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, "
              f"Test: {X_fold_test.shape[0]}")

        # ── 2a. AAA Feature Selection ──
        aaa = ArtificialAlgaeAlgorithm(
            X_train, y_train, X_val, y_val, aaa_cfg, seed=SEED + fold
        )
        best_mask, aaa_fitness, aaa_history = aaa.run()
        all_aaa_histories.append(aaa_history)

        selected_idx = np.where(best_mask == 1)[0]
        selected_names = [feature_names[i] for i in selected_idx]
        print(f"  Selected features ({len(selected_idx)}): {selected_names}")

        # Apply feature selection
        X_train_sel = X_train[:, selected_idx]
        X_val_sel = X_val[:, selected_idx]
        X_test_sel = X_fold_test[:, selected_idx]

        # ── 2b. IWO Hyperparameter Optimization ──
        iwo = InvasiveWeedOptimization(
            X_train_sel, y_train, X_val_sel, y_val,
            iwo_cfg, dae_cfg, seed=SEED + fold
        )
        best_params, iwo_history = iwo.run()
        all_iwo_histories.append(iwo_history)

        # ── 2c. Train final DAE with optimal parameters ──
        print(f"\n  Training final DAE classifier (fold {fold}) ...")
        input_dim = X_train_sel.shape[1]
        optimal_encoder = best_params["encoder_layers"]
        optimal_lr = best_params["learning_rate"]
        optimal_dropout = best_params["dropout_rate"]

        pretrain_model, classifier_model = build_dae_classifier(
            input_dim=input_dim,
            encoder_layers=optimal_encoder,
            activation=dae_cfg["activation"],
            dropout_rate=optimal_dropout,
            learning_rate=dae_cfg["pretrain_learning_rate"],
        )

        # Track memory during training
        tracemalloc.start()
        train_start = time.time()

        # Pretrain
        pretrain_model = pretrain_dae(pretrain_model, X_train_sel, dae_cfg)

        # Fine-tune with optimal learning rate
        classifier_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=optimal_lr),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        classifier_model, train_history = finetune_dae(
            classifier_model, pretrain_model,
            X_train_sel, y_train,
            X_val_sel, y_val,
            dae_cfg,
        )

        train_time = time.time() - train_start
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)

        # ── 2d. Evaluate on fold test set ──
        # Predictive metrics
        proba = classifier_model.predict(X_test_sel, verbose=0).flatten()
        fold_metrics = evaluate(y_fold_test, proba, name=f"Fold {fold}")

        # Computational metrics — inference time (avg over 3 runs)
        inf_start = time.time()
        for _ in range(3):
            _ = classifier_model.predict(X_test_sel, verbose=0)
        inference_time = (time.time() - inf_start) / 3
        inference_per_sample_ms = (inference_time / len(y_fold_test)) * 1000

        # Computational metrics — model memory size on disk
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "model.keras")
            classifier_model.save(tmp_path)
            model_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)

        print(f"\n  ── Fold {fold} Computational Metrics ──")
        print(f"  Training time       : {train_time:.2f}s")
        print(f"  Inference/sample    : {inference_per_sample_ms:.4f}ms")
        print(f"  Peak training memory: {peak_memory_mb:.2f}MB")
        print(f"  Model size on disk  : {model_size_mb:.2f}MB")
        print(f"  Convergence rate    : N/A (not an ensemble model)")

        fold_metrics["selected_features"] = selected_names
        fold_metrics["n_features"] = len(selected_idx)
        fold_metrics["iwo_best_params"] = best_params
        fold_metrics["fold"] = fold
        fold_metrics["training_time_s"] = round(train_time, 2)
        fold_metrics["inference_ms_per_sample"] = round(inference_per_sample_ms, 4)
        fold_metrics["peak_training_memory_mb"] = round(peak_memory_mb, 2)
        fold_metrics["model_size_mb"] = round(model_size_mb, 2)
        fold_metrics["convergence_rate"] = "N/A (not an ensemble model)"
        fold_results.append(fold_metrics)

        # Clean up GPU memory between folds
        del pretrain_model, classifier_model
        tf.keras.backend.clear_session()

    total_time = time.time() - overall_start

    # ── 3. Aggregate results ──
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)

    # Predictive metrics
    predictive_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    predictive_summary = {}
    print("\n  ── Predictive Metrics ──")
    for key in predictive_keys:
        values = [r[key] for r in fold_results]
        predictive_summary[key] = {
            "mean": round(float(np.mean(values)), 6),
            "std": round(float(np.std(values)), 6),
            "per_fold": [round(v, 6) for v in values],
        }
        print(f"  {key}: {np.mean(values):.6f} ± {np.std(values):.6f}")

    # Confusion matrices per fold
    predictive_summary["confusion_matrix_per_fold"] = [
        r["confusion_matrix"] for r in fold_results
    ]

    # Computational metrics
    computational_summary = {}
    comp_keys = [
        ("training_time_s", "Training time (s)"),
        ("inference_ms_per_sample", "Inference/sample (ms)"),
        ("peak_training_memory_mb", "Peak training memory (MB)"),
        ("model_size_mb", "Model size (MB)"),
    ]
    print("\n  ── Computational Metrics ──")
    for key, label in comp_keys:
        values = [r[key] for r in fold_results]
        computational_summary[key] = {
            "mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values)), 4),
            "per_fold": [round(v, 4) for v in values],
        }
        print(f"  {label}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    computational_summary["convergence_rate"] = "N/A (not an ensemble model)"
    print(f"  Convergence rate: N/A (not an ensemble model)")

    print(f"\n  Total pipeline time: {total_time:.1f}s")

    # ── Save results ──
    out_dir = os.path.join(cfg["output_dir"], "odae_wpdc")
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "dataset_id": ocfg["dataset_id"],
        "n_folds": n_folds,
        "total_pipeline_time_s": round(total_time, 2),
        "predictive_metrics": predictive_summary,
        "computational_metrics": computational_summary,
        "fold_results": fold_results,
        "aaa_config": aaa_cfg,
        "dae_config": {k: v for k, v in dae_cfg.items()},
        "iwo_config": {k: v for k, v in iwo_cfg.items() if k != "search_space"},
        "iwo_search_space": {k: v for k, v in iwo_cfg["search_space"].items()},
    }

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    # Save AAA history
    aaa_path = os.path.join(out_dir, "aaa_history.json")
    with open(aaa_path, "w") as f:
        json.dump(all_aaa_histories, f, indent=2, default=str)

    # Save IWO history
    iwo_path = os.path.join(out_dir, "iwo_history.json")
    with open(iwo_path, "w") as f:
        json.dump(all_iwo_histories, f, indent=2, default=str)

    print(f"  AAA history saved to {aaa_path}")
    print(f"  IWO history saved to {iwo_path}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Standalone entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = load_config()

    # Allow overriding dataset_id from CLI:
    #   python odae_wpdc_pipeline.py 7
    if len(sys.argv) > 1:
        cfg["odae_wpdc"]["dataset_id"] = sys.argv[1]
        print(f"Dataset override: {sys.argv[1]}")

    train_odae_wpdc(cfg)
