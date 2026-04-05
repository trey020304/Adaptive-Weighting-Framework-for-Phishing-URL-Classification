"""
Adaptive Hybrid URL Phishing Simulator
========================================
Interactive tool that lets the user:
  1. Choose a trained checkpoint (by dataset ID)
  2. Enter URLs one at a time for phishing prediction

Usage:
    python simulate.py
"""

import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd

from preprocess import extract_features_from_url, STANDARD_FEATURES
from inference import (
    load_checkpoint,
    prepare_path_a,
    prepare_path_b,
    AttentionLayer,
)
from gpu_setup import configure_gpu


# ── Discover available checkpoints ────────────────────────────────────────────
CHECKPOINT_ROOT = os.path.join("results", "adaptive_hybrid")


def discover_checkpoints():
    """Return a sorted list of (dataset_id, checkpoint_dir) tuples."""
    checkpoints = []
    if not os.path.isdir(CHECKPOINT_ROOT):
        return checkpoints
    for entry in sorted(os.listdir(CHECKPOINT_ROOT)):
        ckpt_dir = os.path.join(CHECKPOINT_ROOT, entry, "checkpoint")
        if os.path.isdir(ckpt_dir) and entry.startswith("dataset_"):
            dataset_id = entry.replace("dataset_", "")
            checkpoints.append((dataset_id, ckpt_dir))
    return checkpoints


# ── Auto-calibration: detect label inversion ──────────────────────────────────
# Reference URLs with obvious phishing / legitimate characteristics.
_PHISH_REF = [
    "http://192.168.1.1/login/password/bank/verify/update/secure?account=123",
    "http://paypal-login-secure-verify.suspicious-site.tk/account/confirm",
    "http://1.2.3.4/signin/ebay/password/free.exe",
]
_LEGIT_REF = [
    "www.google.com",
    "www.wikipedia.org",
    "www.github.com",
]


def _raw_hybrid_score(url, ckpt):
    """Compute raw hybrid probability for a URL (before any inversion fix)."""
    features = extract_features_from_url(url)
    df = pd.DataFrame([features])
    for col in STANDARD_FEATURES:
        if col not in df.columns:
            df[col] = 0
    X_a = prepare_path_a(df.copy(), ckpt)
    pa = ckpt["rf_model"].predict_proba(X_a)[:, 1][0]
    X_char, X_token, X_tab = prepare_path_b(df.copy(), [url], ckpt)
    pb = ckpt["keras_model"].predict(
        [X_char, X_token, X_tab], verbose=0
    ).ravel()[0]
    alpha = ckpt["adaptive"]["alpha"]
    beta = ckpt["adaptive"]["beta"]
    return alpha * pa + beta * pb


def calibrate(ckpt):
    """Detect if the model has inverted labels.

    Returns True if probabilities should be flipped (1 - p) before use.
    Also prints a warning if the model cannot distinguish phishing from legit.
    """
    phish_scores = [_raw_hybrid_score(u, ckpt) for u in _PHISH_REF]
    legit_scores = [_raw_hybrid_score(u, ckpt) for u in _LEGIT_REF]
    avg_phish = float(np.mean(phish_scores))
    avg_legit = float(np.mean(legit_scores))

    # If phishing reference scores lower than legit, labels are inverted
    inverted = avg_phish < avg_legit

    # Measure discrimination ability
    gap = abs(avg_phish - avg_legit)
    if gap < 0.10:
        print(f"  ⚠  Warning: this model has weak discrimination on reference URLs")
        print(f"     (phish avg={avg_phish:.4f}, legit avg={avg_legit:.4f}, gap={gap:.4f})")
        print(f"     Predictions may be unreliable for out-of-distribution URLs.")

    if inverted:
        print(f"  (auto-calibration: label inversion detected — probabilities will be corrected)")

    return inverted


# ── Predict a single URL ─────────────────────────────────────────────────────
def predict_url(url, ckpt, invert=False):
    """Return (label, confidence, proba_a, proba_b, proba_hybrid) for a URL."""
    # Extract 53 lexical features
    features = extract_features_from_url(url)
    df = pd.DataFrame([features])

    # Ensure all standard features exist
    for col in STANDARD_FEATURES:
        if col not in df.columns:
            df[col] = 0

    # ── Path A ──
    X_a = prepare_path_a(df.copy(), ckpt)
    proba_a = ckpt["rf_model"].predict_proba(X_a)[:, 1][0]

    # ── Path B ──
    X_char, X_token, X_tab = prepare_path_b(df.copy(), [url], ckpt)
    proba_b = ckpt["keras_model"].predict(
        [X_char, X_token, X_tab], verbose=0
    ).ravel()[0]

    # ── Adaptive hybrid ──
    alpha = ckpt["adaptive"]["alpha"]
    beta = ckpt["adaptive"]["beta"]
    proba_hybrid = alpha * proba_a + beta * proba_b

    # Correct for label inversion
    if invert:
        proba_a = 1.0 - proba_a
        proba_b = 1.0 - proba_b
        proba_hybrid = 1.0 - proba_hybrid

    # Map back to label
    label = "Phishing" if proba_hybrid >= 0.5 else "Not Phishing"
    confidence = proba_hybrid if proba_hybrid >= 0.5 else 1.0 - proba_hybrid

    return label, confidence, proba_a, proba_b, proba_hybrid


# ── Interactive loop ──────────────────────────────────────────────────────────
def main():
    configure_gpu()

    # ── Step 1: Discover and display checkpoints ──
    checkpoints = discover_checkpoints()
    if not checkpoints:
        print("No trained checkpoints found under", CHECKPOINT_ROOT)
        sys.exit(1)

    print("=" * 60)
    print("  Adaptive Hybrid — URL Phishing Simulator")
    print("=" * 60)
    print("\nAvailable trained models:\n")
    for i, (ds_id, ckpt_dir) in enumerate(checkpoints, 1):
        print(f"  [{i}] Dataset {ds_id}  —  {ckpt_dir}")

    # ── Step 2: User selects a checkpoint ──
    print()
    while True:
        choice = input("Select a model (number): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(checkpoints):
            break
        print(f"  Please enter a number between 1 and {len(checkpoints)}.")

    ds_id, ckpt_dir = checkpoints[int(choice) - 1]
    print(f"\nLoading model trained on Dataset {ds_id} ...\n")
    ckpt = load_checkpoint(ckpt_dir)

    # Fallback: if tab_features wasn't saved, use the 53 standard lexical features
    if ckpt["tab_features"] is None:
        ckpt["tab_features"] = list(STANDARD_FEATURES)
        print(f"  (tab_features reconstructed → {len(STANDARD_FEATURES)} standard features)")

    # Auto-calibrate: detect label inversion
    print("  Running auto-calibration on reference URLs ...")
    invert = calibrate(ckpt)

    alpha = ckpt["adaptive"]["alpha"]
    beta = ckpt["adaptive"]["beta"]
    print(f"\nModel ready.  Adaptive weights: alpha={alpha:.4f}, beta={beta:.4f}")
    print(f"Type a URL to classify, or 'quit' to exit.\n")

    # ── Step 3: Prediction loop ──
    while True:
        url = input("URL> ").strip()
        if not url:
            continue
        if url.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        try:
            label, conf, pa, pb, ph = predict_url(url, ckpt, invert=invert)
        except Exception as e:
            print(f"  Error: {e}\n")
            continue

        # Display results
        tag = "⚠ PHISHING" if label == "Phishing" else "✓ LEGITIMATE"
        print(f"\n  Result : {tag}")
        print(f"  Confidence : {conf:.2%}")
        print(f"  ─────────────────────────────────")
        print(f"  Path A (RF)          : {pa:.4f}")
        print(f"  Path B (CNN-BiLSTM)  : {pb:.4f}")
        print(f"  Hybrid (α·A + β·B)   : {ph:.4f}")
        print(f"  Threshold            : 0.50")
        print()


if __name__ == "__main__":
    main()
