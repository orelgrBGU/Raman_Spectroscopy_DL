#!/usr/bin/env python
"""
Run notebook 02 (synth sanity) as a standalone script.
Saves all figures to outputs/figs/02_synth/.

Usage:
    cd /gpfs0/bgu-rgilad/users/orelgr/deep2
    /gpfs0/bgu-rgilad/users/orelgr/env/deep2_env/bin/python scripts/run_notebook02.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True

ROOT = Path("/gpfs0/bgu-rgilad/users/orelgr/deep2")
FIGS = ROOT / "outputs" / "figs" / "02_synth"
FIGS.mkdir(parents=True, exist_ok=True)

from src.data.synth_mixtures import ChemicalPool, SynthConfig, make_fixed_batch

print("Loading ChemicalPool...")
pool = ChemicalPool.load()
train_pool, hold_pool = pool.split(holdout_frac=0.2, seed=0)
print(f"Train pool: {len(train_pool.chemicals)} chemicals")
print(f"Holdout pool: {len(hold_pool.chemicals)} chemicals")

# --- Holdout check ---
overlap = set(train_pool.chemicals) & set(hold_pool.chemicals)
print(f"Overlap: {len(overlap)}  {'PASS' if len(overlap) == 0 else 'FAIL'}")

# --- Generate samples ---
print("Generating 8 samples for construction plots...")
samples = make_fixed_batch(train_pool, n=8, seed=42)
for i, s in enumerate(samples):
    c = s["c"].numpy()
    print(f"  Sample {i}: K={s['K']} M={s['M']} SNR={s['snr_db']:.0f}dB "
          f"coeffs(>0)={c[c>0].round(3)}")

grid = train_pool.grid


def plot_construction(ax, sample, title):
    y = sample["y"].numpy()
    R = sample["R"].numpy()
    c = sample["c"].numpy()
    bl = sample["baseline"].numpy()
    mask = sample["mask"].numpy().astype(bool)
    names = sample["ref_names"]

    clean_sum = (c[:, None] * R).sum(axis=0) + bl

    for j in range(len(c)):
        if c[j] < 1e-6:
            continue
        short = names[j].split(":")[-1][:20]
        ax.plot(grid[mask], (c[j] * R[j])[mask], lw=0.9, alpha=0.8,
                label=f"{short} ({c[j]:.3f})")

    ax.plot(grid[mask], bl[mask], "--k", lw=0.8, alpha=0.6, label="baseline")
    ax.plot(grid[mask], clean_sum[mask], color="navy", lw=1.6, alpha=0.7,
            label="clean sum")
    ax.plot(grid[mask], y[mask], color="red", lw=0.7, alpha=0.5,
            label="corrupted y")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("cm\u207b\u00b9", fontsize=8)
    ax.legend(fontsize=6, loc="upper right", ncol=2)


# --- Figure 1: 8 construction plots ---
print("Plotting 8 construction panels...")
fig, axes = plt.subplots(4, 2, figsize=(15, 18))
for i, (ax, s) in enumerate(zip(axes.flat, samples)):
    plot_construction(
        ax, s,
        f"[GT / construction] Sample {i}:  K={s['K']}, M={s['M']}, SNR={s['snr_db']:.0f} dB"
    )
fig.suptitle("Synthetic mixture construction \u2014 ground truth decomposition", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(FIGS / "construction_8_samples.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {FIGS / 'construction_8_samples.png'}")

# --- Figure 2: Low vs high SNR ---
print("Plotting SNR comparison...")
cfg_low  = SynthConfig(snr_db_range=(10.0, 15.0), seed=99)
cfg_high = SynthConfig(snr_db_range=(55.0, 60.0), seed=99)
low_samples  = make_fixed_batch(train_pool, n=2, cfg=cfg_low,  seed=99)
high_samples = make_fixed_batch(train_pool, n=2, cfg=cfg_high, seed=99)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for row, (lo, hi) in enumerate(zip(low_samples, high_samples)):
    plot_construction(axes[row, 0], lo,
        f"[GT / construction] Low SNR={lo['snr_db']:.0f} dB  K={lo['K']}")
    plot_construction(axes[row, 1], hi,
        f"[GT / construction] High SNR={hi['snr_db']:.0f} dB  K={hi['K']}")
fig.suptitle("Corruption comparison: low SNR (left) vs high SNR (right)", fontsize=12)
fig.tight_layout()
fig.savefig(FIGS / "snr_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {FIGS / 'snr_comparison.png'}")

# --- Figure 3: Distributions ---
print("Generating 500 samples for distribution analysis...")
big_batch = make_fixed_batch(train_pool, n=500, seed=7)

all_K = [s["K"] for s in big_batch]
all_M = [s["M"] for s in big_batch]
all_snr = [s["snr_db"] for s in big_batch]
all_coeffs = np.concatenate([s["c"].numpy()[s["c"].numpy() > 1e-6] for s in big_batch])

fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
axes[0].hist(all_K, bins=np.arange(0.5, 9.5, 1), edgecolor="k", alpha=0.7)
axes[0].set_xlabel("K (true components)")
axes[0].set_title("K distribution")

axes[1].hist(all_M, bins=np.arange(-0.5, 6.5, 1), edgecolor="k", alpha=0.7, color="orange")
axes[1].set_xlabel("M (distractors)")
axes[1].set_title("M distribution")

axes[2].hist(all_coeffs, bins=40, edgecolor="k", alpha=0.7, color="green")
axes[2].set_xlabel("Coefficient value")
axes[2].set_title("Dirichlet coefficient distribution")

axes[3].hist(all_snr, bins=30, edgecolor="k", alpha=0.7, color="purple")
axes[3].set_xlabel("SNR (dB)")
axes[3].set_title("SNR distribution")

fig.suptitle("Synthetic generator statistics (n=500 samples)", fontsize=12)
fig.tight_layout()
fig.savefig(FIGS / "generator_distributions.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {FIGS / 'generator_distributions.png'}")

print("\nDONE — all figures saved to outputs/figs/02_synth/")
