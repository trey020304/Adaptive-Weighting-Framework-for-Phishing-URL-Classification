# Adaptive Hybrid Phishing URL Classification Framework

An adaptive ensemble framework for phishing URL detection that combines **Chi-square + Random Forest** (Path A) with a **Hybrid CNN-BiLSTM** neural network (Path B) using an exponential loss-weighted gating mechanism. The framework benchmarks this adaptive hybrid against four baseline reproductions from the literature.

## Overview

Phishing remains one of the most prevalent cybersecurity threats. This project proposes an **Adaptive Hybrid** classification approach that dynamically weights the contributions of a traditional machine learning model and a deep learning model based on their validation losses during training. The key insight is that instead of a fixed 50/50 ensemble, the gating weights evolve as the CNN-BiLSTM trains, allowing the system to rely more heavily on whichever model is performing better at each epoch.

### Models

| # | Model | Type | Reference |
|---|-------|------|-----------|
| 1 | **Path A** — Chi-square + Random Forest | ML baseline | Proposed |
| 2 | **Path B** — Hybrid CNN-BiLSTM (char + token + tabular) | DL baseline | Proposed |
| 3 | **Adaptive Hybrid** — Loss-weighted ensemble of Path A & B | Proposed framework | — |
| 4 | **ODAE-WPDC** — Optimal Deep Autoencoder | DL baseline | Alshehri et al. (2022) |
| 5 | **PSO-XGBoost** — PSO-optimized XGBoost Linear | ML baseline | Sheikhi & Kostakos (2024) |
| 6 | **BiGRU-Attention** — Character-level BiGRU | DL baseline | Yuan et al. (2019) |

### Adaptive Weighting Mechanism

The ensemble weights are computed per training epoch $t$ as:

$$\alpha(t) = \frac{e^{-\gamma L_A}}{e^{-\gamma L_A} + e^{-\gamma L_B(t)}}, \quad \beta(t) = 1 - \alpha(t)$$

$$P_{\text{final}}(x) = \alpha(t) \cdot P_A(x) + \beta(t) \cdot P_B(x)$$

Since Random Forest is non-iterative, $L_A$ is computed once and held constant. The weights evolve as $L_B(t)$ changes during CNN-BiLSTM training. The final prediction uses weights from the epoch with the lowest combined validation loss. $\gamma$ is a sensitivity coefficient selected via grid search.

---

## Project Structure

| File | Purpose |
|---|---|
| `config.yaml` | Central configuration — all hyperparameters, dataset registry, pipeline settings |
| `preprocess.py` | Dataset loading, 53-feature extraction, MinMax scaling, SMOTE, train/val/test splitting |
| `path_a_baseline.py` | Path A standalone — Chi-square feature selection + Random Forest (GridSearchCV) |
| `path_b_baseline.py` | Path B standalone — Hybrid CNN-BiLSTM (char + token + tabular inputs) |
| `adaptive_hybrid.py` | Main framework — runs Path A & B, adaptive weighting, full evaluation |
| `odae_wpdc_pipeline.py` | ODAE-WPDC baseline — AAA feature selection + Deep Autoencoder + IWO tuning |
| `pso_xgboost_pipeline.py` | PSO-XGBoost baseline — Particle Swarm Optimization + XGBoost Linear |
| `bigru_pipeline.py` | BiGRU-Attention baseline — character-level BiGRU with additive attention |
| `inference.py` | Run a trained adaptive hybrid checkpoint on new datasets without retraining |
| `simulate.py` | Interactive CLI — enter URLs one at a time for real-time phishing prediction |
| `generate_charts.py` | Generate 21 publication-ready PNG charts at 300 DPI from results |
| `complexity_analysis.py` | Theoretical Big O complexity analysis with visualization |
| `gpu_setup.py` | TensorFlow GPU configuration utility (memory growth / VRAM capping) |

---

## Environment Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU recommended (CUDA 11.8+ / cuDNN 8.6+ for TensorFlow GPU)

### Installation

```bash
# Clone or download the project
cd "THESIS II"

# Create and activate a virtual environment
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install tensorflow>=2.15 scikit-learn>=1.3 xgboost pandas numpy scipy pyyaml joblib imbalanced-learn matplotlib seaborn
```

> **Note:** On CPU-only systems, all pipelines still work — `gpu_setup.py` handles this gracefully.

---

## Datasets

The project uses a registry of 9 phishing URL datasets defined in `config.yaml` under the `datasets` section. Each dataset entry specifies its file path, column mappings, and which label values indicate phishing.

| ID | Source | Type | Balance |
|----|--------|------|---------|
| `3` | Kaggle (Shashwat Tiwari) | Defined (pre-extracted features) | 50/50 |
| `6` | Mendeley (Hannousse) | Defined (pre-extracted features) | 50/50 |
| `7`–`12`, `c1` | Various (Kaggle, Mendeley, PhiUSIIL) | Raw (URL + label only) | Imbalanced |

- **Defined** datasets already contain the 53 lexical features.
- **Raw** datasets contain only URLs and labels; `preprocess.py` extracts the 53 features automatically.

---

## Running Pipelines

### 1. Preprocess Datasets

Two modes are available:

```bash
# Cross-dataset split — assign specific datasets to train/val/test
python preprocess.py 1,3,6 7,8 5,9

# Homogeneous split — pool datasets then stratified split
python preprocess.py --homogeneous 3,5,7,8,9
```

This creates a directory under `processed_datasets/` (e.g., `train[1,3,6]_val[7,8]_test[5,9]/`) containing `train.csv`, `val.csv`, `test.csv`, and `scaler.joblib`.

If no arguments are provided, the script prompts interactively.

### 2. Adaptive Hybrid (Main Framework)

```bash
python adaptive_hybrid.py
```

This will:
1. Train **Path A** (stability-weighted Chi-square k search + Random Forest with GridSearchCV).
2. Train **Path B** (Hybrid CNN-BiLSTM with early stopping and LR reduction).
3. Grid-search the best $\gamma$ on the validation set.
4. Combine predictions using exponential loss-weighted gating.
5. Evaluate all four variants (Path A, Path B, static ensemble, adaptive hybrid) on the test set.
6. Run bootstrap significance tests.
7. Save results to `results/adaptive_hybrid/dataset_<id>/`.

### 3. Individual Baselines

```bash
python path_a_baseline.py          # Chi-square + Random Forest only
python path_b_baseline.py          # Hybrid CNN-BiLSTM only
python odae_wpdc_pipeline.py       # ODAE-WPDC (Alshehri et al.)
python pso_xgboost_pipeline.py     # PSO-XGBoost (Sheikhi & Kostakos)
python bigru_pipeline.py           # BiGRU-Attention (Yuan et al.)
```

Each baseline reads its corresponding section from `config.yaml` and saves results under `results/<pipeline_name>/`.

### 4. Inference on New Data

Run a trained adaptive hybrid checkpoint on a different dataset:

```bash
python inference.py --checkpoint results/adaptive_hybrid/dataset_3/checkpoint --dataset_id 10
```

| Argument | Description |
|----------|-------------|
| `--checkpoint` | Path to a saved checkpoint directory |
| `--dataset_id` | Dataset ID from the registry to evaluate on |
| `--config` | Config file path (default: `config.yaml`) |
| `--train_dataset_id` | Training dataset ID (for older checkpoints without saved metadata) |

### 5. Interactive URL Simulator

```bash
python simulate.py
```

Prompts you to select a checkpoint, then enter URLs one at a time for real-time phishing/legitimate classification. Includes auto-calibration that detects and corrects label inversion.

### 6. Generate Charts

```bash
python generate_charts.py
```

Produces 21 publication-ready PNG charts at 300 DPI in `results/charts/`, including model comparisons, confusion matrices, weight evolution, radar plots, and computational efficiency analyses.

### 7. Complexity Analysis

```bash
python complexity_analysis.py
```

Computes theoretical Big O complexity for all 6 models using actual hyperparameters and dataset sizes. Outputs charts to `results/charts/complexity/`.

---

## Configuration (`config.yaml`)

All hyperparameters and settings are centralized in `config.yaml`. Key sections:

### Global

| Key | Description | Default |
|-----|-------------|---------|
| `random_seed` | Reproducibility seed for NumPy, TensorFlow, scikit-learn | `42` |
| `feature_columns` | List of 53 lexical URL features used across all pipelines | — |
| `label_column` | Target column name after preprocessing | `"label"` |
| `positive_label` | Value representing the phishing class | `"Phishing"` |

### `path_a` — Chi-square + Random Forest

| Key | Description | Default |
|-----|-------------|---------|
| `shift_penalty` | Penalty for distribution shift in stability-weighted chi-square | `10.0` |
| `chi2_k_range` | Candidate k values for feature selection | `[1, 2, ..., 53]` |
| `rf_n_estimators` | Number of trees | `300` |
| `rf_max_depth` | Max tree depth | `30` |
| `rf_min_samples_leaf` | Min samples per leaf | `5` |
| `rf_class_weight` | Class weighting strategy | `"balanced"` |

### `path_b` — CNN-BiLSTM Architecture

| Key | Description | Default |
|-----|-------------|---------|
| `conv1_filters` / `conv1_kernel` | First Conv1D layer | `64` / `3` |
| `conv2_filters` / `conv2_kernel` | Second Conv1D layer | `128` / `5` |
| `bilstm_units` | BiLSTM units per direction | `128` |
| `learning_rate` | Optimizer learning rate | `0.001` |
| `batch_size` | Training batch size | `256` |
| `max_epochs` | Max training epochs | `500` |
| `early_stopping_patience` | Epochs without improvement before stopping | `50` |

### `adaptive` — Ensemble Weighting

| Key | Description | Default |
|-----|-------------|---------|
| `dataset_id` | Dataset to use | `"3"` |
| `gamma_candidates` | Sensitivity coefficients to search | `[1.0, 2.0, 5.0]` |
| `gamma` | Default gamma if not tuning | `2.0` |
| `test_size` / `val_size` | Split ratios | `0.20` / `0.15` |

### `datasets` — Dataset Registry

Each entry maps a dataset ID to its file path, URL/label column names, and which label values indicate phishing. Example:

```yaml
datasets:
  "3":
    path: "datasets/defined/[3] dataset_phishing_kaggle_Shashwat Tiwari (50-50).csv"
    type: "defined"
    url_col: "url"
    label_col: "status"
    phishing_values: ["phishing"]
```

### Baseline Sections

| Section | Pipeline | Key Settings |
|---------|----------|--------------|
| `aqilla` | Random Forest baseline (Khairunnisya 2025) | Correlation threshold, chi2 p-value, RF grid |
| `princeton_improved` | CNN-BiLSTM baseline (Vishal J et al. 2025) | Char/token encoding, architecture dims, training |
| `odae_wpdc` | ODAE-WPDC (Alshehri et al. 2022) | AAA population/iterations, DAE layers, IWO search space |
| `pso_xgboost` | PSO-XGBoost (Sheikhi & Kostakos 2024) | PSO particles/iterations/inertia, XGBoost search space |
| `bigru` | BiGRU-Attention (Yuan et al. 2019) | Char encoding, GRU units, attention size |

---

## Outputs

### Results Directory

Each pipeline saves results under `results/<pipeline_name>/dataset_<id>/`:

| Pipeline | Key Output Files |
|----------|-----------------|
| `adaptive_hybrid` | `results.json` (full metrics, weight history, config), `checkpoint/` (11 model artifacts) |
| `path_a_baseline` | `metrics_summary.csv`, `chi2_feature_scores.csv` |
| `path_b_baseline` | `metrics_summary.csv`, `training_history.csv` |
| `odae_wpdc` | `results.json`, AAA fitness history |
| `pso_xgboost` | `results.json`, PSO convergence |
| `bigru` | `metrics_summary.csv`, `training_history.csv` |

### Checkpoint Contents (`adaptive_hybrid/.../checkpoint/`)

| File | Description |
|------|-------------|
| `path_a_rf.joblib` | Trained Random Forest model |
| `path_a_features.joblib` | Selected feature names |
| `path_a_mm_scaler.joblib` | MinMax scaler |
| `path_a_z_scaler.joblib` | Standard scaler |
| `label_encoder.joblib` | Label encoder |
| `path_b_cnn_bilstm.keras` | Trained CNN-BiLSTM model |
| `path_b_char_to_idx.joblib` | Character-to-index mapping |
| `path_b_token_vocab.joblib` | Token vocabulary |
| `path_b_tab_scaler.joblib` | Tabular feature scaler |
| `path_b_tab_features.joblib` | Tabular feature column names |
| `adaptive_weights.joblib` | `{gamma, alpha, beta, best_epoch}` |

### Metrics

All pipelines compute: **Accuracy, Precision, Recall, F1, ROC-AUC, Log-Loss**, confusion matrix, and classification report. Computational metrics include training time, inference latency per sample, and model size.

---

## Reproducibility

- Random seed is fixed at **42** across NumPy, TensorFlow, and scikit-learn.
- The MinMax scaler is fit only on training data (no data leakage).
- All model artifacts are saved for later inference.
- Stratified splits ensure consistent class distribution across train/val/test.
