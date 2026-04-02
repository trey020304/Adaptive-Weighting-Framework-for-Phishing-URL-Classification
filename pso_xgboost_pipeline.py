"""
PSO-XGBoost Pipeline: Particle Swarm Optimization + XGBoost Linear Booster
===========================================================================
Reproduction of: Sheikhi & Kostakos (2024), Computers & Security, 142, 103885.

Pipeline stages:
  1. Data pre-processing (preprocess.py → pso_xgboost_preprocess)
  2. PSO hyperparameter optimization for XGBoost (XgbLinear)
  3. Final model training with optimized hyperparameters
  4. K-fold cross-validation evaluation on training set
  5. Final evaluation on held-out test set

All hyperparameters are read from config.yaml → pso_xgboost section.
"""

import os
import sys
import json
import time
import tempfile
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    cohen_kappa_score,
)
import xgboost as xgb
import joblib

from preprocess import load_config, pso_xgboost_preprocess

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  PSO — Particle Swarm Optimization for XGBoost Hyperparameters
# ══════════════════════════════════════════════════════════════════════════════

class ParticleSwarmOptimizer:
    """PSO for optimizing XGBoost linear booster hyperparameters.

    Particles represent candidate hyperparameter configurations:
      [nrounds, eta, lambda_l2, alpha_l1]

    Fitness = weighted composite of accuracy, precision, recall, F1
    evaluated via k-fold cross-validation (higher is better).

    Velocity/position update follows Eqs. (1)-(3) from the paper:
      V_id = ω * V_id + C1 * r1 * (P_id - X_id) + C2 * r2 * (P_gd - X_id)
      X_id = X_id + V_id
      ω = ω_max + (iter - iter_i) * (ω_max - ω_min) / iter
    """

    def __init__(self, X_train, y_train, pso_cfg, xgb_cfg, cv_folds, seed=42):
        self.X_train = X_train
        self.y_train = y_train
        self.pso_cfg = pso_cfg
        self.xgb_cfg = xgb_cfg
        self.cv_folds = cv_folds
        self.seed = seed

        self.num_particles = pso_cfg["num_particles"]
        self.max_iter = pso_cfg["max_iterations"]
        self.w_max = pso_cfg["inertia_weight_max"]
        self.w_min = pso_cfg["inertia_weight_min"]
        self.c1 = pso_cfg["c1"]
        self.c2 = pso_cfg["c2"]

        self.search_space = pso_cfg["search_space"]
        self.fitness_weights = pso_cfg["fitness_weights"]

        # Dimension order: [nrounds, eta, lambda_l2, alpha_l1]
        self.dim_names = ["nrounds", "eta", "lambda_l2", "alpha_l1"]
        self.D = len(self.dim_names)

        self.bounds_low = np.array([
            self.search_space["nrounds"][0],
            self.search_space["eta"][0],
            self.search_space["lambda_l2"][0],
            self.search_space["alpha_l1"][0],
        ], dtype=np.float64)

        self.bounds_high = np.array([
            self.search_space["nrounds"][1],
            self.search_space["eta"][1],
            self.search_space["lambda_l2"][1],
            self.search_space["alpha_l1"][1],
        ], dtype=np.float64)

        self.rng = np.random.RandomState(seed)

    def _clip_position(self, position):
        """Clip particle position to search bounds."""
        return np.clip(position, self.bounds_low, self.bounds_high)

    def _decode_particle(self, position):
        """Convert continuous position vector to XGBoost hyperparameters."""
        return {
            "nrounds": max(10, int(round(position[0]))),
            "eta": float(position[1]),
            "lambda_l2": float(position[2]),
            "alpha_l1": float(position[3]),
        }

    def _fitness(self, position):
        """Evaluate fitness of a particle via k-fold cross-validation.

        Returns composite fitness score (higher is better).
        """
        params = self._decode_particle(position)

        xgb_params = {
            "booster": self.xgb_cfg["booster"],
            "objective": self.xgb_cfg["objective"],
            "eval_metric": self.xgb_cfg["eval_metric"],
            "eta": params["eta"],
            "lambda": params["lambda_l2"],
            "alpha": params["alpha_l1"],
            "verbosity": 0,
            "nthread": -1,
        }

        skf = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.seed
        )

        acc_scores = []
        prec_scores = []
        rec_scores = []
        f1_scores = []

        for train_idx, val_idx in skf.split(self.X_train, self.y_train):
            X_tr = self.X_train[train_idx]
            y_tr = self.y_train[train_idx]
            X_va = self.X_train[val_idx]
            y_va = self.y_train[val_idx]

            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_va, label=y_va)

            model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=params["nrounds"],
                evals=[(dval, "val")],
                verbose_eval=False,
            )

            y_pred_proba = model.predict(dval)
            y_pred = (y_pred_proba > 0.5).astype(int)

            acc_scores.append(accuracy_score(y_va, y_pred))
            prec_scores.append(precision_score(y_va, y_pred, zero_division=0))
            rec_scores.append(recall_score(y_va, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_va, y_pred, zero_division=0))

        w = self.fitness_weights
        fitness = (
            w["accuracy"] * np.mean(acc_scores)
            + w["precision"] * np.mean(prec_scores)
            + w["recall"] * np.mean(rec_scores)
            + w["f1"] * np.mean(f1_scores)
        )

        return fitness

    def run(self):
        """Execute PSO optimization. Returns (best_params, best_fitness, history)."""
        print("\n" + "=" * 60)
        print("PSO HYPERPARAMETER OPTIMIZATION")
        print("=" * 60)
        print(f"  Particles: {self.num_particles}, Iterations: {self.max_iter}, "
              f"Dimensions: {self.D}")
        print(f"  Inertia weight: [{self.w_min}, {self.w_max}], "
              f"C1: {self.c1}, C2: {self.c2}")
        print(f"  Search space:")
        for name in self.dim_names:
            print(f"    {name}: {self.search_space[name]}")

        # Initialize positions and velocities
        positions = np.array([
            self.rng.uniform(self.bounds_low, self.bounds_high)
            for _ in range(self.num_particles)
        ])
        velocities = np.zeros_like(positions)

        # Initialize velocity bounds (fraction of search range)
        vel_max = (self.bounds_high - self.bounds_low) * 0.2
        vel_min = -vel_max

        # Evaluate initial fitness
        fitnesses = np.array([self._fitness(p) for p in positions])

        # Personal best
        p_best_positions = positions.copy()
        p_best_fitnesses = fitnesses.copy()

        # Global best
        g_best_idx = np.argmax(fitnesses)
        g_best_position = positions[g_best_idx].copy()
        g_best_fitness = fitnesses[g_best_idx]

        history = []

        print(f"\n  Initial global best fitness: {g_best_fitness:.6f}")
        init_params = self._decode_particle(g_best_position)
        print(f"    nrounds={init_params['nrounds']}, eta={init_params['eta']:.6f}, "
              f"lambda={init_params['lambda_l2']:.6f}, alpha={init_params['alpha_l1']:.6f}")

        for iteration in range(self.max_iter):
            # Linearly decreasing inertia weight (Eq. 3)
            w = self.w_max - (iteration * (self.w_max - self.w_min) / self.max_iter)

            for i in range(self.num_particles):
                r1 = self.rng.random(self.D)
                r2 = self.rng.random(self.D)

                # Update velocity (Eq. 1)
                velocities[i] = (
                    w * velocities[i]
                    + self.c1 * r1 * (p_best_positions[i] - positions[i])
                    + self.c2 * r2 * (g_best_position - positions[i])
                )

                # Clamp velocity
                velocities[i] = np.clip(velocities[i], vel_min, vel_max)

                # Update position (Eq. 2)
                positions[i] = positions[i] + velocities[i]

                # Clip to bounds
                positions[i] = self._clip_position(positions[i])

                # Evaluate fitness
                fit = self._fitness(positions[i])
                fitnesses[i] = fit

                # Update personal best
                if fit > p_best_fitnesses[i]:
                    p_best_fitnesses[i] = fit
                    p_best_positions[i] = positions[i].copy()

                    # Update global best
                    if fit > g_best_fitness:
                        g_best_fitness = fit
                        g_best_position = positions[i].copy()

            best_params = self._decode_particle(g_best_position)
            history.append({
                "iteration": iteration + 1,
                "best_fitness": float(g_best_fitness),
                "best_params": best_params,
                "inertia_weight": float(w),
            })

            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"  Iter {iteration + 1:3d}/{self.max_iter} — "
                      f"fitness: {g_best_fitness:.6f} | "
                      f"nrounds={best_params['nrounds']}, "
                      f"eta={best_params['eta']:.6f}, "
                      f"λ={best_params['lambda_l2']:.6f}, "
                      f"α={best_params['alpha_l1']:.6f}")

        final_params = self._decode_particle(g_best_position)
        print(f"\n  PSO complete — best fitness: {g_best_fitness:.6f}")
        print(f"  Optimal hyperparameters:")
        print(f"    nrounds  : {final_params['nrounds']}")
        print(f"    eta      : {final_params['eta']:.6f}")
        print(f"    lambda   : {final_params['lambda_l2']:.6f}")
        print(f"    alpha    : {final_params['alpha_l1']:.6f}")

        return final_params, g_best_fitness, history


# ══════════════════════════════════════════════════════════════════════════════
#  Training Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def train_pso_xgboost(cfg):
    """Full PSO-XGBoost pipeline.

    1. Preprocess data (load, scale, split)
    2. Run PSO to find optimal XGBoost hyperparameters
    3. Train final model with optimized hyperparameters
    4. Cross-validate on training set
    5. Evaluate on held-out test set
    6. Save results
    """
    SEED = cfg["random_seed"]
    xcfg = cfg["pso_xgboost"]
    xgb_cfg = xcfg["xgb"]
    pso_cfg = xcfg["pso"]
    cv_folds = xcfg["cv_folds"]

    np.random.seed(SEED)

    print("\n" + "=" * 60)
    print("PSO-XGBoost PIPELINE (Sheikhi & Kostakos 2024)")
    print("=" * 60)

    # ── 1. Preprocess ──
    print("\n── Step 1: Preprocessing ──")
    data = pso_xgboost_preprocess(cfg)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]
    le = data["le"]
    scaler = data["scaler"]

    print(f"\n  Features: {len(feature_names)}")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    print(f"  Train class distribution: "
          f"legitimate={np.sum(y_train == 0)}, phishing={np.sum(y_train == 1)}")

    # ── 2. PSO Hyperparameter Optimization ──
    print("\n── Step 2: PSO Hyperparameter Optimization ──")
    pso_start = time.time()

    pso = ParticleSwarmOptimizer(
        X_train, y_train, pso_cfg, xgb_cfg, cv_folds, seed=SEED
    )
    optimal_params, best_fitness, pso_history = pso.run()

    pso_time = time.time() - pso_start
    print(f"\n  PSO optimization time: {pso_time:.1f}s")

    # ── 3. Train Final Model with Optimized Hyperparameters ──
    print("\n── Step 3: Training Final Model ──")
    print(f"  Using optimized parameters:")
    print(f"    booster  : {xgb_cfg['booster']}")
    print(f"    nrounds  : {optimal_params['nrounds']}")
    print(f"    eta      : {optimal_params['eta']:.6f}")
    print(f"    lambda   : {optimal_params['lambda_l2']:.6f}")
    print(f"    alpha    : {optimal_params['alpha_l1']:.6f}")

    final_xgb_params = {
        "booster": xgb_cfg["booster"],
        "objective": xgb_cfg["objective"],
        "eval_metric": xgb_cfg["eval_metric"],
        "eta": optimal_params["eta"],
        "lambda": optimal_params["lambda_l2"],
        "alpha": optimal_params["alpha_l1"],
        "verbosity": xgb_cfg["verbosity"],
        "nthread": -1,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    train_start = time.time()
    final_model = xgb.train(
        final_xgb_params,
        dtrain,
        num_boost_round=optimal_params["nrounds"],
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=False,
    )
    train_time = time.time() - train_start
    print(f"  Training time: {train_time:.2f}s")

    # ── 4. Cross-Validation on Training Set ──
    print("\n── Step 4: Cross-Validation on Training Set ──")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)

    cv_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [],
                  "roc_auc": [], "kappa": []}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_va, y_va = X_train[va_idx], y_train[va_idx]

        d_tr = xgb.DMatrix(X_tr, label=y_tr)
        d_va = xgb.DMatrix(X_va, label=y_va)

        fold_model = xgb.train(
            final_xgb_params, d_tr,
            num_boost_round=optimal_params["nrounds"],
            verbose_eval=False,
        )

        y_proba = fold_model.predict(d_va)
        y_pred = (y_proba > 0.5).astype(int)

        cv_metrics["accuracy"].append(accuracy_score(y_va, y_pred))
        cv_metrics["precision"].append(precision_score(y_va, y_pred, zero_division=0))
        cv_metrics["recall"].append(recall_score(y_va, y_pred, zero_division=0))
        cv_metrics["f1"].append(f1_score(y_va, y_pred, zero_division=0))
        cv_metrics["roc_auc"].append(roc_auc_score(y_va, y_proba))
        cv_metrics["kappa"].append(cohen_kappa_score(y_va, y_pred))

    print(f"  {cv_folds}-Fold Cross-Validation Results:")
    for metric_name, scores in cv_metrics.items():
        mean_val = np.mean(scores)
        std_val = np.std(scores)
        print(f"    {metric_name:12s}: {mean_val:.4f} ± {std_val:.4f}")

    # ── 5. Final Evaluation on Test Set ──
    print("\n── Step 5: Final Evaluation on Test Set ──")

    inf_start = time.time()
    y_test_proba = final_model.predict(dtest)
    inference_time = time.time() - inf_start

    y_test_pred = (y_test_proba > 0.5).astype(int)

    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    f1v = f1_score(y_test, y_test_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_test_proba)
    kappa = cohen_kappa_score(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)

    # TPR and FPR (per the paper's evaluation metrics)
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
    print(f"  Kappa     : {kappa:.4f} ({kappa * 100:.2f}%)")
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

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "model.json")
        final_model.save_model(tmp_path)
        model_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)

    total_time = pso_time + train_time

    print(f"\n  ── Computational Metrics ──")
    print(f"  PSO optimization time : {pso_time:.2f}s")
    print(f"  Final training time   : {train_time:.2f}s")
    print(f"  Total pipeline time   : {total_time:.2f}s")
    print(f"  Inference/sample      : {inference_per_sample_ms:.4f}ms")
    print(f"  Model size            : {model_size_mb:.4f}MB")

    # ── 6. Save Results ──
    out_dir = os.path.join(cfg["output_dir"], "pso_xgboost",
                           f"dataset_{xcfg['dataset_id']}")
    os.makedirs(out_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(out_dir, "model.json")
    final_model.save_model(model_path)
    print(f"\n  Model saved to {model_path}")

    # Save scaler
    scaler_path = os.path.join(out_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    # Save label encoder
    le_path = os.path.join(out_dir, "label_encoder.joblib")
    joblib.dump(le, le_path)

    # Build results dict
    results = {
        "dataset_id": xcfg["dataset_id"],
        "model": "PSO-XGBoost (XgbLinear)",
        "total_pipeline_time_s": round(total_time, 2),
        "pso_optimization_time_s": round(pso_time, 2),
        "training_time_s": round(train_time, 2),
        "optimal_hyperparameters": optimal_params,
        "pso_best_fitness": round(best_fitness, 6),
        "cv_results": {
            metric: {
                "mean": round(np.mean(scores), 4),
                "std": round(np.std(scores), 4),
                "per_fold": [round(s, 4) for s in scores],
            }
            for metric, scores in cv_metrics.items()
        },
        "test_metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1v, 4),
            "roc_auc": round(auc, 4),
            "kappa": round(kappa, 4),
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "confusion_matrix": cm.tolist(),
        },
        "computational_metrics": {
            "pso_time_s": round(pso_time, 2),
            "train_time_s": round(train_time, 2),
            "inference_ms_per_sample": round(inference_per_sample_ms, 4),
            "model_size_mb": round(model_size_mb, 4),
        },
        "pso_config": {
            "num_particles": pso_cfg["num_particles"],
            "max_iterations": pso_cfg["max_iterations"],
            "inertia_weight": [pso_cfg["inertia_weight_min"],
                               pso_cfg["inertia_weight_max"]],
            "c1": pso_cfg["c1"],
            "c2": pso_cfg["c2"],
        },
        "xgb_config": {
            "booster": xgb_cfg["booster"],
            "objective": xgb_cfg["objective"],
        },
    }

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {results_path}")

    # Save PSO convergence history
    pso_history_path = os.path.join(out_dir, "pso_history.json")
    with open(pso_history_path, "w") as f:
        json.dump(pso_history, f, indent=2, default=str)
    print(f"  PSO history saved to {pso_history_path}")

    # Save CSV summary (consistent with other pipelines)
    summary_df = pd.DataFrame([{
        "model": "PSO-XGBoost (XgbLinear)",
        "dataset_id": xcfg["dataset_id"],
        "nrounds": optimal_params["nrounds"],
        "eta": round(optimal_params["eta"], 6),
        "lambda_l2": round(optimal_params["lambda_l2"], 6),
        "alpha_l1": round(optimal_params["alpha_l1"], 6),
        "pso_fitness": round(best_fitness, 6),
        "cv_accuracy_mean": round(np.mean(cv_metrics["accuracy"]), 4),
        "cv_accuracy_std": round(np.std(cv_metrics["accuracy"]), 4),
        "test_accuracy": round(acc, 4),
        "test_precision": round(prec, 4),
        "test_recall": round(rec, 4),
        "test_f1": round(f1v, 4),
        "test_roc_auc": round(auc, 4),
        "test_kappa": round(kappa, 4),
        "test_tpr": round(tpr, 4),
        "test_fpr": round(fpr, 4),
        "train_time_s": round(train_time, 2),
        "pso_time_s": round(pso_time, 2),
        "inference_ms_per_sample": round(inference_per_sample_ms, 4),
        "model_size_mb": round(model_size_mb, 4),
    }])
    summary_df.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)

    print(f"\n  All outputs saved to {out_dir}/")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Standalone entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = load_config()

    # Allow overriding dataset_id from CLI:
    #   python pso_xgboost_pipeline.py 7
    if len(sys.argv) > 1:
        cfg["pso_xgboost"]["dataset_id"] = sys.argv[1]
        print(f"Dataset override: {sys.argv[1]}")

    train_pso_xgboost(cfg)
