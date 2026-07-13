#!/usr/bin/env python
"""
Notebook 05 — Training Monitoring & Learning Curves
=====================================================

Reads the CSV log from a training run and produces publication-quality
plots to monitor progress.

Can be run **while training is still running** — reads whatever is
available so far.

Saves
-----
    outputs/figs/05_training/<run_id>/learning_curves.png
    outputs/figs/05_training/<run_id>/loss_components.png
    outputs/figs/05_training/<run_id>/lr_schedule.png
    outputs/figs/05_training/<run_id>/val_vs_train.png
    outputs/figs/05_training/<run_id>/summary.txt

Usage
-----
    cd /gpfs0/bgu-rgilad/users/orelgr/deep2
    /gpfs0/bgu-rgilad/users/orelgr/env/deep2_env/bin/python scripts/run_notebook05.py --run_id run01
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True

ROOT = Path("/gpfs0/bgu-rgilad/users/orelgr/deep2")

# ── Parse args ──
parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, default="run01")
args = parser.parse_args()
run_id = args.run_id

csv_path = ROOT / "outputs" / run_id / "metrics.csv"
FIGS = ROOT / "outputs" / "figs" / "05_training" / run_id
FIGS.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print(f"Notebook 05 — Training Monitoring: {run_id}")
print("=" * 60)

# ═════════════════════════════════════════════════════════════════════
# 1. Load metrics
# ═════════════════════════════════════════════════════════════════════
if not csv_path.exists():
    print(f"ERROR: {csv_path} not found. Is the training running?")
    sys.exit(1)

df = pd.read_csv(csv_path)
n_epochs = len(df)
print(f"\nLoaded {n_epochs} epochs from {csv_path}")
print(f"Epochs: {df['epoch'].min()} to {df['epoch'].max()}")
print(f"\nLatest metrics (epoch {df['epoch'].iloc[-1]}):")
print(f"  Train loss: {df['train_loss'].iloc[-1]:.6f}")
print(f"  Val loss:   {df['val_loss'].iloc[-1]:.6f}")
print(f"  Train coeff MAE: {df['train_loss_c'].iloc[-1]:.6f}")
print(f"  Val coeff MAE:   {df['val_loss_c'].iloc[-1]:.6f}")

best_idx = df["val_loss"].idxmin()
print(f"\nBest val loss: {df['val_loss'].iloc[best_idx]:.6f} at epoch {df['epoch'].iloc[best_idx]}")

# NNLS baseline from notebook 03 (for comparison)
nnls_csv = ROOT / "outputs" / "metrics" / "nnls_baseline.csv"
nnls_mae = None
if nnls_csv.exists():
    nnls_df = pd.read_csv(nnls_csv)
    nnls_mae = nnls_df["mae"].mean()
    print(f"NNLS baseline MAE: {nnls_mae:.4f} (for reference)")

# ═════════════════════════════════════════════════════════════════════
# 2. Plots
# ═════════════════════════════════════════════════════════════════════
print("\n── Generating plots ──")
epochs = df["epoch"].values

# --- 2a. Learning curves (train vs val total loss) ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs, df["train_loss"], "b-", lw=1.5, label="Train loss")
ax.plot(epochs, df["val_loss"], "r-", lw=1.5, label="Val loss")
ax.axvline(df["epoch"].iloc[best_idx], color="green", ls="--", lw=0.8, alpha=0.5,
           label=f"Best val (epoch {df['epoch'].iloc[best_idx]})")
ax.set_xlabel("Epoch")
ax.set_ylabel("Total Loss")
ax.set_title(f"Learning Curves — {run_id}")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "learning_curves.png", dpi=200)
plt.close(fig)
print(f"  Saved: learning_curves.png")

# --- 2b. Loss components (train) ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs, df["train_loss_c"], label="Coeff MAE", lw=1.2)
ax.plot(epochs, df["train_loss_r"], label="Reconstruction MSE", lw=1.2)
ax.plot(epochs, df["train_loss_b"], label="Baseline MAE", lw=1.2)
if nnls_mae is not None:
    ax.axhline(nnls_mae, color="gray", ls=":", lw=1.2, label=f"NNLS MAE = {nnls_mae:.4f}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title(f"Train Loss Components — {run_id}")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "loss_components.png", dpi=200)
plt.close(fig)
print(f"  Saved: loss_components.png")

# --- 2c. LR schedule ---
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(epochs, df["lr"].astype(float), "k-", lw=1.2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Learning Rate")
ax.set_title(f"LR Schedule — {run_id}")
ax.set_yscale("log")
fig.tight_layout()
fig.savefig(FIGS / "lr_schedule.png", dpi=200)
plt.close(fig)
print(f"  Saved: lr_schedule.png")

# --- 2d. Val vs Train coefficient MAE ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs, df["train_loss_c"], "b-", lw=1.5, label="Train coeff MAE")
ax.plot(epochs, df["val_loss_c"], "r-", lw=1.5, label="Val coeff MAE")
if nnls_mae is not None:
    ax.axhline(nnls_mae, color="gray", ls=":", lw=1.2, label=f"NNLS MAE = {nnls_mae:.4f}")
ax.axvline(df["epoch"].iloc[best_idx], color="green", ls="--", lw=0.8, alpha=0.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Coefficient MAE")
ax.set_title(f"Coefficient MAE: Train vs Validation — {run_id}")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "val_vs_train.png", dpi=200)
plt.close(fig)
print(f"  Saved: val_vs_train.png")

# ═════════════════════════════════════════════════════════════════════
# 3. Summary
# ═════════════════════════════════════════════════════════════════════
gap = df["val_loss"].iloc[-1] - df["train_loss"].iloc[-1]
overfitting = "possible" if gap > 0.02 else "minimal"

summary = f"""
Notebook 05 — Training Monitoring Summary
==========================================
Run ID: {run_id}
Epochs completed: {n_epochs}

Latest metrics (epoch {df['epoch'].iloc[-1]}):
  Train loss:      {df['train_loss'].iloc[-1]:.6f}
  Val loss:        {df['val_loss'].iloc[-1]:.6f}
  Train coeff MAE: {df['train_loss_c'].iloc[-1]:.6f}
  Val coeff MAE:   {df['val_loss_c'].iloc[-1]:.6f}
  Train-Val gap:   {gap:.6f} ({overfitting} overfitting)

Best validation:
  Val loss:        {df['val_loss'].iloc[best_idx]:.6f} at epoch {df['epoch'].iloc[best_idx]}
  Val coeff MAE:   {df['val_loss_c'].iloc[best_idx]:.6f}

{"NNLS baseline MAE: " + f"{nnls_mae:.4f}" if nnls_mae else "NNLS baseline: not available"}
{"Model beats NNLS: " + ("YES" if df['val_loss_c'].iloc[best_idx] < nnls_mae else "NOT YET") if nnls_mae else ""}

Figures saved to: {FIGS}/
"""
print(summary)

with open(FIGS / "summary.txt", "w") as f:
    f.write(summary)
print(f"  Saved: summary.txt")
print("DONE")
