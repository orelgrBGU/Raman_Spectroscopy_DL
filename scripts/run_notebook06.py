#!/usr/bin/env python
"""
Notebook 06 — Model Evaluation & Comparison with NNLS Baseline
===============================================================

Purpose
-------
After training, load the best checkpoint and evaluate on **holdout chemicals**
(never seen during training). Compare head-to-head with the NNLS baseline.

This notebook answers the key thesis question:
    "Does the DL model outperform classical NNLS on spectral decomposition?"

Evaluation protocol
-------------------
1. Generate 500 synthetic mixtures from **holdout** chemicals.
2. Run both the DL model and NNLS on the same samples.
3. Compare coefficient MAE, reconstruction quality, and robustness to noise/complexity.

Saves
-----
    outputs/figs/06_eval/
    ├── scatter_model.png           — model pred vs true coefficients
    ├── scatter_nnls.png            — NNLS pred vs true coefficients
    ├── mae_comparison_bar.png      — MAE bar chart: model vs NNLS
    ├── mae_vs_snr.png              — MAE as function of SNR
    ├── mae_vs_K.png                — MAE as function of # components
    ├── example_good_0..2.png       — best model predictions
    ├── example_hard_0..2.png       — hardest cases (high K, low SNR)
    ├── improvement_histogram.png   — per-sample MAE improvement
    └── summary.txt                 — text summary

Usage
-----
    cd /gpfs0/bgu-rgilad/users/orelgr/deep2
    python scripts/run_notebook06.py --run_id run01
    python scripts/run_notebook06.py --run_id run01 --ckpt best  # or: last
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True

ROOT = Path("/gpfs0/bgu-rgilad/users/orelgr/deep2")

parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, default="run01")
parser.add_argument("--ckpt", type=str, default="best", help="best | last | epoch_XXXX")
parser.add_argument("--n_samples", type=int, default=500)
args = parser.parse_args()

FIGS = ROOT / "outputs" / "figs" / "06_eval"
FIGS.mkdir(parents=True, exist_ok=True)

ckpt_path = ROOT / "checkpoints" / args.run_id / f"{args.ckpt}.pt"
if not ckpt_path.exists():
    print(f"ERROR: Checkpoint not found: {ckpt_path}")
    print("Is training still running? Check with: tail -5 logs/run01.log")
    sys.exit(1)

import torch
from src.data.synth_mixtures import ChemicalPool, make_fixed_batch
from src.eval import load_model_from_checkpoint, predict_batch, nnls_predict_batch, compute_metrics

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Notebook 06 — Model Evaluation")
print("=" * 60)

# ═════════════════════════════════════════════════════════════════════
# 1. Load model & data
# ═════════════════════════════════════════════════════════════════════
print(f"\nCheckpoint: {ckpt_path}")
model, info = load_model_from_checkpoint(ckpt_path, device)
print(f"  Trained for {info['epoch']+1} epochs, {info['step']} steps")
print(f"  Best val loss during training: {info['best_metric']:.6f}")

print("\nLoading chemical pool...")
pool = ChemicalPool.load()
_, holdout_pool = pool.split(holdout_frac=0.2, seed=0)
grid = holdout_pool.grid
print(f"  Holdout chemicals: {len(holdout_pool.chemicals)}")

print(f"\nGenerating {args.n_samples} evaluation samples from holdout chemicals...")
samples = make_fixed_batch(holdout_pool, n=args.n_samples, seed=2024)

# Stats
Ks = [s["K"] for s in samples]
snrs = [s["snr_db"] for s in samples]
print(f"  K range: {min(Ks)}-{max(Ks)}, mean={np.mean(Ks):.1f}")
print(f"  SNR range: {min(snrs):.0f}-{max(snrs):.0f} dB, mean={np.mean(snrs):.0f} dB")

# ═════════════════════════════════════════════════════════════════════
# 2. Run both methods
# ═════════════════════════════════════════════════════════════════════
print("\n── Running DL model ──")
model_results = predict_batch(model, samples, device)
model_metrics = compute_metrics(model_results)
print(f"  MAE: {model_metrics['mae_mean']:.4f} +/- {model_metrics['mae_std']:.4f}")
print(f"  Spearman: {model_metrics['spearman_mean']:.3f}")

print("\n── Running NNLS baseline ──")
nnls_results = nnls_predict_batch(samples, grid, poly_order=5)
nnls_metrics = compute_metrics(nnls_results)
print(f"  MAE: {nnls_metrics['mae_mean']:.4f} +/- {nnls_metrics['mae_std']:.4f}")
print(f"  Spearman: {nnls_metrics['spearman_mean']:.3f}")

# ═════════════════════════════════════════════════════════════════════
# 3. Plots
# ═════════════════════════════════════════════════════════════════════
print("\n── Generating plots ──")

# Helper: collect all coefficients
def collect_coeffs(results):
    c_true_all, c_pred_all = [], []
    for r in results:
        c_true_all.extend(r["coeffs_true"].tolist())
        c_pred_all.extend(r["coeffs_pred"].tolist())
    return np.array(c_true_all), np.array(c_pred_all)

model_ct, model_cp = collect_coeffs(model_results)
nnls_ct, nnls_cp = collect_coeffs(nnls_results)

# --- 3a. Scatter: Model ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(model_ct, model_cp, alpha=0.3, s=10, edgecolors="none")
lim = max(model_ct.max(), model_cp.max(), nnls_cp.max()) * 1.1
ax.plot([0, lim], [0, lim], "r--", lw=1)
ax.set_xlabel("True Coefficient")
ax.set_ylabel("Predicted Coefficient")
ax.set_title(f"DL Model: Pred vs True (MAE={model_metrics['mae_mean']:.4f})")
ax.set_xlim(-0.02, lim)
ax.set_ylim(-0.05, lim)
fig.tight_layout()
fig.savefig(FIGS / "scatter_model.png", dpi=200)
plt.close(fig)
print(f"  Saved: scatter_model.png")

# --- 3b. Scatter: NNLS ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(nnls_ct, nnls_cp, alpha=0.3, s=10, edgecolors="none", color="orange")
ax.plot([0, lim], [0, lim], "r--", lw=1)
ax.set_xlabel("True Coefficient")
ax.set_ylabel("Predicted Coefficient")
ax.set_title(f"NNLS Baseline: Pred vs True (MAE={nnls_metrics['mae_mean']:.4f})")
ax.set_xlim(-0.02, lim)
ax.set_ylim(-0.05, lim)
fig.tight_layout()
fig.savefig(FIGS / "scatter_nnls.png", dpi=200)
plt.close(fig)
print(f"  Saved: scatter_nnls.png")

# --- 3c. Bar chart comparison ---
fig, ax = plt.subplots(figsize=(7, 5))
methods = ["NNLS", "DL Model"]
maes = [nnls_metrics["mae_mean"], model_metrics["mae_mean"]]
stds = [nnls_metrics["mae_std"], model_metrics["mae_std"]]
colors = ["#ff9800", "#2196f3"]
bars = ax.bar(methods, maes, yerr=stds, color=colors, capsize=8, edgecolor="black", lw=0.5)
ax.set_ylabel("Coefficient MAE")
ax.set_title("Model vs NNLS: Coefficient MAE (lower is better)")
for bar, mae in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{mae:.4f}", ha="center", va="bottom", fontweight="bold")
fig.tight_layout()
fig.savefig(FIGS / "mae_comparison_bar.png", dpi=200)
plt.close(fig)
print(f"  Saved: mae_comparison_bar.png")

# --- 3d. MAE vs SNR ---
snr_arr = np.array([r["snr_db"] for r in model_results])
model_mae_arr = model_metrics["per_sample_mae"]
nnls_mae_arr = nnls_metrics["per_sample_mae"]

snr_bins = np.arange(10, 65, 10)
snr_centers = snr_bins[:-1] + 5

model_binned = [model_mae_arr[(snr_arr >= lo) & (snr_arr < lo+10)] for lo in snr_bins[:-1]]
nnls_binned = [nnls_mae_arr[(snr_arr >= lo) & (snr_arr < lo+10)] for lo in snr_bins[:-1]]

fig, ax = plt.subplots(figsize=(8, 5))
w = 3.5
ax.bar(snr_centers - w/2, [b.mean() if len(b) else 0 for b in nnls_binned],
       width=w, color="#ff9800", alpha=0.8, label="NNLS")
ax.bar(snr_centers + w/2, [b.mean() if len(b) else 0 for b in model_binned],
       width=w, color="#2196f3", alpha=0.8, label="DL Model")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Mean Coefficient MAE")
ax.set_title("Coefficient MAE vs Signal-to-Noise Ratio")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "mae_vs_snr.png", dpi=200)
plt.close(fig)
print(f"  Saved: mae_vs_snr.png")

# --- 3e. MAE vs K (number of components) ---
K_arr = np.array([r["K"] for r in model_results])
K_vals = sorted(set(K_arr))

fig, ax = plt.subplots(figsize=(8, 5))
model_by_k = [model_mae_arr[K_arr == k] for k in K_vals]
nnls_by_k = [nnls_mae_arr[K_arr == k] for k in K_vals]

x = np.array(K_vals)
w = 0.35
ax.bar(x - w/2, [b.mean() if len(b) else 0 for b in nnls_by_k],
       width=w, color="#ff9800", alpha=0.8, label="NNLS")
ax.bar(x + w/2, [b.mean() if len(b) else 0 for b in model_by_k],
       width=w, color="#2196f3", alpha=0.8, label="DL Model")
ax.set_xlabel("Number of Components (K)")
ax.set_ylabel("Mean Coefficient MAE")
ax.set_title("Coefficient MAE vs Mixture Complexity")
ax.set_xticks(K_vals)
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "mae_vs_K.png", dpi=200)
plt.close(fig)
print(f"  Saved: mae_vs_K.png")

# --- 3f. Per-sample improvement histogram ---
improvement = nnls_mae_arr - model_mae_arr  # positive = model is better

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(improvement, bins=50, color="#4caf50", edgecolor="black", lw=0.3, alpha=0.8)
ax.axvline(0, color="red", ls="--", lw=1.2, label="break-even")
ax.axvline(improvement.mean(), color="blue", ls="-", lw=1.2,
           label=f"mean = {improvement.mean():.4f}")
pct_better = 100 * (improvement > 0).mean()
ax.set_xlabel("NNLS MAE - Model MAE (positive = model wins)")
ax.set_ylabel("Count")
ax.set_title(f"Per-Sample Improvement: Model beats NNLS in {pct_better:.0f}% of samples")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "improvement_histogram.png", dpi=200)
plt.close(fig)
print(f"  Saved: improvement_histogram.png")

# --- 3g. Example decompositions (3 good + 3 hard) ---
def plot_example(result, title_prefix, save_name):
    r = result
    y = r["y"]
    R = r["R"]
    m = r["mask"].astype(bool)
    c_t = r["coeffs_true"]
    c_p = r["coeffs_pred"]
    b_t = r["baseline_true"]
    b_p = r["baseline_pred"]
    names = r["ref_names"]

    recon = (c_p[:, None] * R).sum(axis=0) + b_p
    mae_i = np.mean(np.abs(c_t - c_p))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(grid[m], y[m], "k-", lw=0.6, alpha=0.5, label="mixture")
    ax.plot(grid[m], recon[m], "r-", lw=1.2, alpha=0.8, label="model reconstruction")
    ax.plot(grid[m], b_p[m], "--", color="gray", lw=0.8, label="pred baseline")
    ax.plot(grid[m], b_t[m], ":", color="blue", lw=0.8, alpha=0.5, label="true baseline")

    for j in range(len(c_t)):
        if c_t[j] > 1e-6 or abs(c_p[j]) > 0.01:
            short = names[j].split(":")[-1][:18]
            ax.plot([], [], " ", label=f"{short}: true={c_t[j]:.3f} pred={c_p[j]:.3f}")

    ax.set_title(f"{title_prefix}: K={r['K']}, M={r['M']}, SNR={r['snr_db']:.0f}dB, MAE={mae_i:.4f}")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity")
    ax.legend(fontsize=6.5, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / save_name, dpi=200, bbox_inches="tight")
    plt.close(fig)

# Sort by MAE to find best and worst
sorted_idx = np.argsort(model_mae_arr)

for i in range(3):
    plot_example(model_results[sorted_idx[i]], "[GOOD]", f"example_good_{i}.png")
    print(f"  Saved: example_good_{i}.png")

for i in range(3):
    idx = sorted_idx[-(i+1)]
    plot_example(model_results[idx], "[HARD]", f"example_hard_{i}.png")
    print(f"  Saved: example_hard_{i}.png")

# ═════════════════════════════════════════════════════════════════════
# 4. Summary
# ═════════════════════════════════════════════════════════════════════
pct_better = 100 * (improvement > 0).mean()
winner = "DL Model" if model_metrics["mae_mean"] < nnls_metrics["mae_mean"] else "NNLS"

summary = f"""
Notebook 06 — Model Evaluation Summary
========================================

Evaluation Setup:
  Checkpoint: {ckpt_path.name} (epoch {info['epoch']+1})
  Holdout chemicals: {len(holdout_pool.chemicals)} (never seen in training)
  Test samples: {args.n_samples}
  K range: {min(Ks)}-{max(Ks)}, SNR range: {min(snrs):.0f}-{max(snrs):.0f} dB

Results:
  ┌─────────────┬───────────────┬───────────────┐
  │ Metric      │ DL Model      │ NNLS          │
  ├─────────────┼───────────────┼───────────────┤
  │ MAE (mean)  │ {model_metrics['mae_mean']:.4f} +/- {model_metrics['mae_std']:.4f} │ {nnls_metrics['mae_mean']:.4f} +/- {nnls_metrics['mae_std']:.4f} │
  │ MAE (median)│ {model_metrics['mae_median']:.4f}         │ {nnls_metrics['mae_median']:.4f}         │
  │ Spearman    │ {model_metrics['spearman_mean']:.3f}          │ {nnls_metrics['spearman_mean']:.3f}          │
  │ Recon MSE   │ {model_metrics['recon_mse_mean']:.6f}      │ {nnls_metrics['recon_mse_mean']:.6f}      │
  └─────────────┴───────────────┴───────────────┘

  Winner: {winner}
  Model beats NNLS in {pct_better:.0f}% of samples
  Mean improvement: {improvement.mean():.4f}

Figures saved to: {FIGS}/
"""
print(summary)

with open(FIGS / "summary.txt", "w") as f:
    f.write(summary)
print(f"  Saved: summary.txt")
print("DONE")
