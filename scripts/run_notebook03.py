#!/usr/bin/env python
"""
Run notebook 03 (classical baselines) as a standalone script.
Evaluates NNLS and MCR-ALS on synthetic Raman mixtures.

Each plot is saved as a SEPARATE image file.

Saves:
  - outputs/figs/03_baselines/*.png  (one file per figure)
  - outputs/metrics/nnls_baseline.csv

Usage:
    cd /gpfs0/bgu-rgilad/users/orelgr/deep2
    /gpfs0/bgu-rgilad/users/orelgr/env/deep2_env/bin/python scripts/run_notebook03.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True

ROOT = Path("/gpfs0/bgu-rgilad/users/orelgr/deep2")
FIGS = ROOT / "outputs" / "figs" / "03_baselines"
METRICS = ROOT / "outputs" / "metrics"
FIGS.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)

from src.data.synth_mixtures import ChemicalPool, SynthConfig, make_fixed_batch
from src.baselines.nnls import nnls_batch
from src.baselines.mcr_als import mcr_batch

# ── Load pool & generate eval set ──────────────────────────────────────
print("Loading ChemicalPool...")
pool = ChemicalPool.load()
train_pool, hold_pool = pool.split(holdout_frac=0.2, seed=0)
grid = train_pool.grid

N_EVAL = 200
print(f"Generating {N_EVAL} synthetic samples for evaluation...")
samples = make_fixed_batch(train_pool, n=N_EVAL, seed=123)

# ── Run NNLS ───────────────────────────────────────────────────────────
print("Running NNLS baseline...")
t0 = time.time()
nnls_results = nnls_batch(samples, grid, poly_order=5)
t_nnls = time.time() - t0
print(f"  NNLS done in {t_nnls:.1f}s ({t_nnls/N_EVAL*1000:.0f} ms/sample)")

# ── Run MCR-ALS ────────────────────────────────────────────────────────
print("Running MCR-ALS baseline...")
t0 = time.time()
mcr_results = mcr_batch(samples, grid, max_iter=100)
t_mcr = time.time() - t0
print(f"  MCR-ALS done in {t_mcr:.1f}s ({t_mcr/N_EVAL*1000:.0f} ms/sample)")


# ── Metrics ────────────────────────────────────────────────────────────
def compute_metrics(results, method_name):
    rows = []
    for r in results:
        c_true = r["coeffs_true"]
        c_pred = r["coeffs_pred"]
        n = min(len(c_true), len(c_pred))
        c_t, c_p = c_true[:n], c_pred[:n]

        mae = float(np.mean(np.abs(c_t - c_p)))
        if np.std(c_t) > 1e-9 and np.std(c_p) > 1e-9:
            spear, _ = spearmanr(c_t, c_p)
        else:
            spear = np.nan

        is_dist = c_t < 1e-6
        dist_mean = float(c_p[is_dist].mean()) if is_dist.any() else np.nan

        rows.append({
            "method": method_name,
            "K": r["K"],
            "M": r["M"],
            "snr_db": r["snr_db"],
            "mae": mae,
            "spearman": spear,
            "distractor_coeff_mean": dist_mean,
        })
    return pd.DataFrame(rows)


df_nnls = compute_metrics(nnls_results, "NNLS")
df_mcr = compute_metrics(mcr_results, "MCR-ALS")
df_all = pd.concat([df_nnls, df_mcr], ignore_index=True)

df_all.to_csv(METRICS / "nnls_baseline.csv", index=False)
print(f"\nSaved metrics → {METRICS / 'nnls_baseline.csv'}")

print("\n" + "=" * 60)
print("SUMMARY (mean +/- std)")
print("=" * 60)
for method in ["NNLS", "MCR-ALS"]:
    sub = df_all[df_all["method"] == method]
    print(f"\n{method}:")
    print(f"  MAE(coeffs):     {sub['mae'].mean():.4f} +/- {sub['mae'].std():.4f}")
    print(f"  Spearman:        {sub['spearman'].mean():.4f} +/- {sub['spearman'].std():.4f}")
    print(f"  Distractor leak: {sub['distractor_coeff_mean'].mean():.4f} +/- {sub['distractor_coeff_mean'].std():.4f}")


# ── Plotting (each figure is a SEPARATE file) ─────────────────────────
print("\nPlotting figures...")


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGS / name}")


# --- NNLS scatter: pred vs true ---
fig, ax = plt.subplots(figsize=(6, 6))
all_true = np.concatenate([r["coeffs_true"] for r in nnls_results])
all_pred = np.concatenate([r["coeffs_pred"] for r in nnls_results])
ax.scatter(all_true, all_pred, s=4, alpha=0.3)
lim = max(all_true.max(), all_pred.max()) * 1.05
ax.plot([0, lim], [0, lim], "r--", lw=1, label="ideal")
ax.set_xlabel("True coefficient")
ax.set_ylabel("Predicted coefficient")
ax.set_title("NNLS: predicted vs true coefficients")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.legend()
save_fig(fig, "nnls_pred_vs_true.png")

# --- MCR scatter: pred vs true ---
fig, ax = plt.subplots(figsize=(6, 6))
all_true_m = np.concatenate([r["coeffs_true"] for r in mcr_results])
all_pred_m = np.concatenate([r["coeffs_pred"] for r in mcr_results])
ax.scatter(all_true_m, all_pred_m, s=4, alpha=0.3, color="tab:orange")
lim = max(all_true_m.max(), all_pred_m.max()) * 1.05
ax.plot([0, lim], [0, lim], "r--", lw=1, label="ideal")
ax.set_xlabel("True coefficient")
ax.set_ylabel("Predicted coefficient")
ax.set_title("MCR-ALS: predicted vs true coefficients")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.legend()
save_fig(fig, "mcr_pred_vs_true.png")

# --- MAE vs SNR ---
fig, ax = plt.subplots(figsize=(8, 5))
for method, color in [("NNLS", "tab:blue"), ("MCR-ALS", "tab:orange")]:
    sub = df_all[df_all["method"] == method].copy()
    bins = np.arange(10, 65, 5)
    sub["snr_bin"] = pd.cut(sub["snr_db"], bins)
    grouped = sub.groupby("snr_bin", observed=True)["mae"].agg(["mean", "std"])
    centers = [(b.left + b.right) / 2 for b in grouped.index]
    ax.errorbar(centers, grouped["mean"], yerr=grouped["std"], label=method,
                color=color, marker="o", capsize=3, lw=1.5)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("MAE (coefficients)")
ax.set_title("Coefficient MAE vs SNR — classical baselines")
ax.legend()
save_fig(fig, "mae_vs_snr.png")

# --- MAE vs K ---
fig, ax = plt.subplots(figsize=(8, 5))
for method, color in [("NNLS", "tab:blue"), ("MCR-ALS", "tab:orange")]:
    sub = df_all[df_all["method"] == method]
    grouped = sub.groupby("K")["mae"].agg(["mean", "std"])
    ax.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"], label=method,
                color=color, marker="o", capsize=3, lw=1.5)
ax.set_xlabel("K (number of true components)")
ax.set_ylabel("MAE (coefficients)")
ax.set_title("Coefficient MAE vs K — classical baselines")
ax.legend()
save_fig(fig, "mae_vs_K.png")

# --- NNLS decomposition examples (one file per sample) ---
for idx in range(4):
    s = samples[idx]
    r = nnls_results[idx]

    y = s["y"].numpy()
    R = s["R"].numpy()
    mask = s["mask"].numpy().astype(bool)
    c_true = r["coeffs_true"]
    c_pred = r["coeffs_pred"]
    b_pred = r["baseline_pred"]
    names = r["ref_names"]

    recon = (c_pred[:, None] * R).sum(axis=0) + b_pred

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(grid[mask], y[mask], "k-", lw=0.6, alpha=0.5, label="mixture")
    ax.plot(grid[mask], recon[mask], "r-", lw=1.2, alpha=0.8, label="NNLS recon")
    ax.plot(grid[mask], b_pred[mask], "--", color="gray", lw=0.8, label="baseline")

    for j in range(min(len(c_true), len(c_pred))):
        if c_true[j] > 1e-6 or c_pred[j] > 0.01:
            short = names[j].split(":")[-1][:18]
            ax.plot([], [], " ",
                    label=f"{short}: true={c_true[j]:.3f} pred={c_pred[j]:.3f}")

    mae_i = float(np.mean(np.abs(c_true - c_pred)))
    ax.set_title(f"[PRED / NNLS] Sample {idx}: K={r['K']}, M={r['M']}, "
                 f"SNR={r['snr_db']:.0f}dB, MAE={mae_i:.4f}")
    ax.set_xlabel("cm\u207b\u00b9")
    ax.set_ylabel("Intensity")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    save_fig(fig, f"nnls_example_{idx}.png")


print(f"\nDONE — all outputs saved")
