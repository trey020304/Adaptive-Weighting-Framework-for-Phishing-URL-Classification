# Adaptive Hybrid Phishing URL Classification Framework

An adaptive ensemble that combines **Chi-square + Random Forest** (Path A) with a **CNN-BiLSTM** neural network (Path B) using an exponential loss-weighted gating mechanism.

## Project Structure

| File | Purpose |
|---|---|
| `preprocess.py` | Dataset alignment, zero-filling, MinMax scaling, train/val/test splitting |
| `path_a.py` | Chi-square feature selection + Random Forest training & evaluation |
| `path_b.py` | CNN-BiLSTM model definition and training with per-epoch loss tracking |
| `adaptive_hybrid.py` | Orchestrator — runs both paths, adaptive weighting, full evaluation |
| `config.yaml` | All hyperparameters (k range, γ, learning rate, architecture, etc.) |

## Prerequisites

```
Python 3.10+
scikit-learn >= 1.3
tensorflow >= 2.15
pandas, numpy, scipy, pyyaml, joblib
```

## Reproducing the Experiment

### 1. Preprocess datasets (already done)

```bash
python preprocess.py 1,2 9 3
```

This creates `processed_datasets/train[1,2]_val[9]_test[3]/` with `train.csv`, `val.csv`, `test.csv`, and `scaler.joblib`.

### 2. Run the full adaptive hybrid pipeline

```bash
python adaptive_hybrid.py
```

This will:
1. Train Path A (Chi-square k search via 5-fold CV, then Random Forest).
2. Train Path B (CNN-BiLSTM with early stopping).
3. Select the best γ on the validation set.
4. Combine predictions using exponential loss-weighted gating.
5. Evaluate all models (Path A, Path B, static ensemble, adaptive hybrid) on the test set.
6. Run bootstrap significance tests.
7. Save results to `results/<split_dir>/results.json`.

### 3. Run individual paths (optional)

```bash
python path_a.py          # Chi-square + Random Forest only
python path_b.py          # CNN-BiLSTM only
```

## Configuration

Edit `config.yaml` to change:

- **`split_dir`**: which processed split to use
- **`path_a.chi2_k_range`**: candidate k values for feature selection
- **`path_b.learning_rate`**, **`batch_size`**, **`max_epochs`**: neural network training
- **`adaptive.gamma_candidates`**: sensitivity coefficients for the gating mechanism

## Adaptive Weighting Mechanism

The weights are computed per training epoch as:

$$\alpha(t) = \frac{e^{-\gamma L_A}}{e^{-\gamma L_A} + e^{-\gamma L_B(t)}}, \quad \beta(t) = 1 - \alpha(t)$$

$$P_{\text{final}}(x) = \alpha(t) \cdot P_A(x) + \beta(t) \cdot P_B(x)$$

Since Random Forest is non-iterative, $L_A$ is computed once and held constant. The weights evolve as $L_B(t)$ changes during CNN-BiLSTM training. The final prediction uses weights from the epoch with the lowest combined validation loss.

## Outputs

After running, `results/<split_dir>/` contains:

| File | Description |
|---|---|
| `chi2_selector.joblib` | Fitted Chi-square feature selector |
| `random_forest.joblib` | Trained Random Forest model |
| `selected_features.joblib` | List of selected feature names |
| `cnn_bilstm_best.keras` | Best CNN-BiLSTM checkpoint |
| `results.json` | Full metrics, weight history, configuration |

## Reproducibility

- Random seed is fixed at **42** across NumPy, TensorFlow, and scikit-learn.
- The MinMax scaler is fit only on training data (no data leakage).
- All model artefacts are saved for later inference.
