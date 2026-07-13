#!/usr/bin/env python
"""
Notebook 07 — Detailed Decomposition Visualization
=====================================================

Creates publication-quality figures showing:
  1. Pure reference spectra
  2. How each component looks after weighting (c_i * R_i)
  3. The superposition that builds the mixture
  4. The model's predicted decomposition vs ground truth
  5. Demo: decompose mixtures of UNSEEN chemicals

Each mixture gets a multi-panel figure:
  - Top: mixture spectrum + model reconstruction
  - Middle: each weighted component (true vs predicted)
  - Bottom: coefficient bar chart (true vs predicted)

Saves
-----
    outputs/figs/07_decomposition/
    ├── train_mixture_0..4/        — 5 mixtures from training chemicals
    │   ├── full_decomposition.png
    │   ├── component_0..K.png
    │   └── coefficients.png
    ├── holdout_mixture_0..4/      — 5 mixtures from unseen chemicals (DEMO)
    │   ├── full_decomposition.png
    │   ├── component_0..K.png
    │   └── coefficients.png
    └── summary.txt

Usage
-----
    cd /gpfs0/bgu-rgilad/users/orelgr/deep2
    python scripts/run_notebook07_decomposition_viz.py --run_id run02
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.axisbelow"] = True

ROOT = Path("/gpfs0/bgu-rgilad/users/orelgr/deep2")

parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, default="run02")
parser.add_argument("--n_examples", type=int, default=5)
args = parser.parse_args()

FIGS = ROOT / "outputs" / "figs" / "07_decomposition"
FIGS.mkdir(parents=True, exist_ok=True)

from src.data.synth_mixtures import ChemicalPool, SynthConfig, SyntheticMixtures, make_fixed_batch
from src.eval import load_model_from_checkpoint, predict_batch, nnls_predict_batch
from src.baselines.nnls import nnls_decompose

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Notebook 07 — Detailed Decomposition Visualization")
print("=" * 60)

# ── Load model ──
ckpt_path = ROOT / "checkpoints" / args.run_id / "best.pt"
model, info = load_model_from_checkpoint(ckpt_path, device)
print(f"Model: {ckpt_path.name} (epoch {info['epoch']+1})")

# ── Load chemical pool ──
pool = ChemicalPool.load()
train_pool, holdout_pool = pool.split(holdout_frac=0.2, seed=0)
grid = pool.grid

# Nice color palette for components
COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f",
]


def plot_full_decomposition(sample, result, grid, save_dir, title_prefix, method="DL Model"):
    """Create all plots for one mixture decomposition."""
    save_dir.mkdir(parents=True, exist_ok=True)

    y = sample["y"].numpy()
    R = sample["R"].numpy()
    m = sample["mask"].numpy().astype(bool)
    c_true = sample["c"].numpy()
    c_pred = result["coeffs_pred"]
    b_true = sample["baseline"].numpy()
    b_pred = result["baseline_pred"]
    names = sample["ref_names"]
    K = sample["K"]
    M = sample["M"]
    K_total = len(c_true)
    wn = grid[m]

    # Identify active components (true coeff > 0) and distractors
    active_idx = [j for j in range(K_total) if c_true[j] > 1e-6]
    distractor_idx = [j for j in range(K_total) if c_true[j] <= 1e-6]

    # ─── 1. Full decomposition overview ───
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot mixture
    ax.plot(wn, y[m], "k-", lw=1.5, alpha=0.7, label="Observed mixture", zorder=10)

    # Plot each TRUE weighted component stacked
    cumulative = np.zeros_like(y[m])
    for j in active_idx:
        weighted = c_true[j] * R[j][m]
        short = names[j].split(":")[-1][:25]
        color = COLORS[j % len(COLORS)]
        ax.fill_between(wn, cumulative, cumulative + weighted,
                        alpha=0.3, color=color, label=f"{short} (c={c_true[j]:.3f})")
        cumulative += weighted

    # Plot baseline
    ax.plot(wn, b_true[m], "--", color="gray", lw=1, alpha=0.7, label="Baseline (true)")

    # Plot reconstruction
    recon = (c_pred[:, None] * R).sum(axis=0) + b_pred
    ax.plot(wn, recon[m], "r-", lw=1.5, alpha=0.8, label=f"{method} reconstruction")

    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=12)
    ax.set_ylabel("Intensity", fontsize=12)
    ax.set_title(f"{title_prefix}: Mixture Decomposition (K={K}, M={M})", fontsize=13)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(save_dir / "full_decomposition.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ─── 2. Individual component plots ───
    for j in active_idx:
        short = names[j].split(":")[-1][:25]
        color = COLORS[j % len(COLORS)]

        fig, ax = plt.subplots(figsize=(11, 4))

        # Pure reference (unweighted)
        ax.plot(wn, R[j][m], "-", color=color, lw=0.8, alpha=0.4,
                label=f"Pure {short}")

        # True weighted
        ax.fill_between(wn, 0, c_true[j] * R[j][m],
                        alpha=0.3, color=color,
                        label=f"True weight: c={c_true[j]:.3f}")

        # Predicted weighted
        ax.plot(wn, c_pred[j] * R[j][m], "--", color="red", lw=1.2,
                label=f"Predicted weight: c={c_pred[j]:.3f}")

        ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=11)
        ax.set_ylabel("Intensity", fontsize=11)
        ax.set_title(f"Component: {short}", fontsize=12)
        ax.legend(fontsize=9)
        ax.invert_xaxis()
        fig.tight_layout()
        fig.savefig(save_dir / f"component_{j}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ─── 3. Coefficient bar chart ───
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(K_total)
    width = 0.35
    short_names = [n.split(":")[-1][:15] for n in names]

    bars_true = ax.bar(x - width/2, c_true, width, label="True", color="#2196f3",
                       edgecolor="black", lw=0.5)
    bars_pred = ax.bar(x + width/2, c_pred, width, label=f"{method} Predicted",
                       color="#f44336", edgecolor="black", lw=0.5)

    # Mark distractors
    for j in distractor_idx:
        ax.axvspan(j - 0.5, j + 0.5, alpha=0.08, color="gray")

    ax.set_xlabel("Reference", fontsize=11)
    ax.set_ylabel("Coefficient", fontsize=11)
    ax.set_title(f"{title_prefix}: Coefficients (MAE={np.mean(np.abs(c_true - c_pred)):.4f})", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=10)
    ax.axhline(0, color="black", lw=0.5)

    # Annotate distractors
    if distractor_idx:
        ax.text(0.02, 0.95, "Gray = distractor (true coeff = 0)",
                transform=ax.transAxes, fontsize=8, color="gray", va="top")

    fig.tight_layout()
    fig.savefig(save_dir / "coefficients.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ─── 4. Superposition buildup (how the mixture is constructed) ───
    fig, ax = plt.subplots(figsize=(12, 6))

    cumulative = np.zeros_like(y[m])
    ax.plot(wn, y[m], "k-", lw=2, alpha=0.3, label="Final mixture (with noise)", zorder=0)

    for step, j in enumerate(active_idx):
        weighted = c_true[j] * R[j][m]
        cumulative += weighted
        short = names[j].split(":")[-1][:20]
        color = COLORS[j % len(COLORS)]
        ax.plot(wn, cumulative, "-", color=color, lw=1.5,
                label=f"+ {short} (c={c_true[j]:.3f})", alpha=0.8)

    # Add baseline
    cumulative_with_base = cumulative + b_true[m]
    ax.plot(wn, cumulative_with_base, "--", color="gray", lw=1.5,
            label="+ baseline", alpha=0.8)

    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=12)
    ax.set_ylabel("Intensity", fontsize=12)
    ax.set_title(f"{title_prefix}: How the Mixture is Built (Superposition)", fontsize=13)
    ax.legend(fontsize=8, loc="upper right")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(save_dir / "superposition_buildup.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════
# PART 1: Training chemicals (model has seen these chemicals before)
# ═════════════════════════════════════════════════════════════════════
print("\n── Part 1: Training chemicals ──")
# Use a config that forces K=3-5 for clear visualization
train_samples = make_fixed_batch(train_pool, n=args.n_examples, seed=42,
                                  cfg=SynthConfig(K_min=3, K_max=5, M_min=1, M_max=3, seed=42))
train_results = predict_batch(model, train_samples, device)

for i in range(args.n_examples):
    s = train_samples[i]
    r = train_results[i]
    save_dir = FIGS / f"train_mixture_{i}"
    title = f"[TRAIN] Mixture {i}"
    plot_full_decomposition(s, r, grid, save_dir, title)
    mae = np.mean(np.abs(s["c"].numpy() - r["coeffs_pred"]))
    print(f"  Mixture {i}: K={s['K']}, M={s['M']}, MAE={mae:.4f} — saved to {save_dir.name}/")

# ═════════════════════════════════════════════════════════════════════
# PART 2: DEMO — Holdout chemicals (model has NEVER seen these)
# ═════════════════════════════════════════════════════════════════════
print("\n── Part 2: DEMO — Unseen chemicals ──")
print(f"  Using {len(holdout_pool.chemicals)} holdout chemicals the model never trained on")

holdout_samples = make_fixed_batch(holdout_pool, n=args.n_examples, seed=777,
                                    cfg=SynthConfig(K_min=2, K_max=5, M_min=1, M_max=2, seed=777))
holdout_results = predict_batch(model, holdout_samples, device)

# Also run NNLS on the same samples for comparison
holdout_nnls = []
for s in holdout_samples:
    c_nnls, b_nnls = nnls_decompose(
        s["y"].numpy(), s["R"].numpy(), s["mask"].numpy(), grid, poly_order=5
    )
    holdout_nnls.append({"coeffs_pred": c_nnls, "baseline_pred": b_nnls})

for i in range(args.n_examples):
    s = holdout_samples[i]
    r_model = holdout_results[i]
    r_nnls = holdout_nnls[i]

    # Model prediction
    save_dir = FIGS / f"holdout_mixture_{i}"
    title = f"[DEMO / UNSEEN] Mixture {i}"
    plot_full_decomposition(s, r_model, grid, save_dir, title, method="DL Model")
    mae_model = np.mean(np.abs(s["c"].numpy() - r_model["coeffs_pred"]))

    # NNLS prediction (same mixture)
    save_dir_nnls = FIGS / f"holdout_mixture_{i}_nnls"
    plot_full_decomposition(s, r_nnls, grid, save_dir_nnls, title, method="NNLS")
    mae_nnls = np.mean(np.abs(s["c"].numpy() - r_nnls["coeffs_pred"]))

    winner = "DL" if mae_model < mae_nnls else "NNLS"
    print(f"  Mixture {i}: K={s['K']}, M={s['M']}, "
          f"DL MAE={mae_model:.4f}, NNLS MAE={mae_nnls:.4f} — Winner: {winner}")

    # Print chemical names for this mixture
    active = [(s["ref_names"][j].split(":")[-1], s["c"][j].item())
              for j in range(len(s["c"])) if s["c"][j] > 1e-6]
    print(f"    Components: {', '.join(f'{name} ({c:.1%})' for name, c in active)}")

# ═════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════
print("\n── Summary ──")

all_figs = list(FIGS.rglob("*.png"))
print(f"Total figures generated: {len(all_figs)}")
print(f"Saved to: {FIGS}/")

# List all directories
for d in sorted(FIGS.iterdir()):
    if d.is_dir():
        n_files = len(list(d.glob("*.png")))
        print(f"  {d.name}/ — {n_files} figures")

print("\nDONE")
