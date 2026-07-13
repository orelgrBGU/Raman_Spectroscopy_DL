#!/usr/bin/env python
"""
Stage 4 overfit test: train on 16 fixed samples until loss → 0.

Verifies that the model architecture + loss + forward pass are wired correctly.
If the model can't memorize 16 samples, there's a bug.

Saves:
    outputs/figs/04_overfit/loss_curve.png
    outputs/figs/04_overfit/overfit_example_*.png   (4 samples)

Usage:
    cd /gpfs0/bgu-rgilad/users/orelgr/deep2
    /gpfs0/bgu-rgilad/users/orelgr/env/deep2_env/bin/python scripts/run_overfit_test.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True

ROOT = Path("/gpfs0/bgu-rgilad/users/orelgr/deep2")
FIGS = ROOT / "outputs" / "figs" / "04_overfit"
FIGS.mkdir(parents=True, exist_ok=True)

from src.data.synth_mixtures import ChemicalPool, SynthConfig, make_fixed_batch
from src.model.decompose import DecomposeModel, collate_decompose
from src.model.loss import DecomposeLoss

# ── Settings ──────────────────────────────────────────────────────────
N_SAMPLES = 16
N_STEPS = 5000
LR = 1e-3
PRINT_EVERY = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Overfit test: {N_SAMPLES} samples, {N_STEPS} steps, lr={LR}")

# ── Data ──────────────────────────────────────────────────────────────
print("Loading ChemicalPool...")
pool = ChemicalPool.load()
train_pool, _ = pool.split(holdout_frac=0.2, seed=0)
grid = train_pool.grid

print(f"Generating {N_SAMPLES} fixed samples...")
samples = make_fixed_batch(train_pool, n=N_SAMPLES, seed=77)
batch = collate_decompose(samples)

# Move to device
y = batch["y"].to(DEVICE)
R = batch["R"].to(DEVICE)
c_true = batch["c"].to(DEVICE)
baseline_true = batch["baseline"].to(DEVICE)
mask = batch["mask"].to(DEVICE)
ref_mask = batch["ref_mask"].to(DEVICE)

print(f"Batch shapes: y={y.shape}, R={R.shape}, c={c_true.shape}, mask={mask.shape}")

# ── Model ─────────────────────────────────────────────────────────────
model = DecomposeModel(
    d_model=256,        # full size for overfit test
    n_transformer_layers=2,
    n_heads=4,
    dropout=0.0,        # no dropout for overfit
    poly_order=5,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

criterion = DecomposeLoss(lambda_c=1.0, lambda_r=1.0, lambda_b=0.5, lambda_l1=0.0)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_STEPS, eta_min=1e-5)

# ── Train ─────────────────────────────────────────────────────────────
print(f"\nTraining for {N_STEPS} steps...")
history = {"loss": [], "loss_c": [], "loss_r": [], "loss_b": []}

for step in range(1, N_STEPS + 1):
    model.train()
    optimizer.zero_grad()

    c_pred, b_pred = model(y, R, ref_mask)
    loss, detail = criterion(c_pred, c_true, b_pred, baseline_true, y, R, mask, ref_mask)

    loss.backward()
    optimizer.step()
    scheduler.step()

    for k in history:
        history[k].append(detail[k])

    if step % PRINT_EVERY == 0 or step == 1:
        print(f"  step {step:4d}  loss={detail['loss']:.5f}  "
              f"c={detail['loss_c']:.5f}  r={detail['loss_r']:.6f}  b={detail['loss_b']:.6f}")

final_loss = history["loss"][-1]
print(f"\nFinal loss: {final_loss:.6f}")
print("PASS" if final_loss < 0.05 else "FAIL — loss did not converge near zero")

# ── Plot loss curve ───────────────────────────────────────────────────
print("\nPlotting...")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history["loss"], label="total", lw=1.5)
ax.plot(history["loss_c"], label="coeff MAE", lw=1, alpha=0.7)
ax.plot(history["loss_r"], label="reconstruction", lw=1, alpha=0.7)
ax.plot(history["loss_b"], label="baseline", lw=1, alpha=0.7)
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title(f"Overfit test: {N_SAMPLES} samples, {N_STEPS} steps")
ax.set_yscale("log")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "loss_curve.png", dpi=200)
plt.close(fig)
print(f"  Saved: {FIGS / 'loss_curve.png'}")

# ── Plot example predictions ─────────────────────────────────────────
model.eval()
with torch.no_grad():
    c_pred, b_pred = model(y, R, ref_mask)

c_pred_np = c_pred.cpu().numpy()
b_pred_np = b_pred.cpu().numpy()

for idx in range(min(4, N_SAMPLES)):
    s = samples[idx]
    y_np = s["y"].numpy()
    R_np = s["R"].numpy()
    m = s["mask"].numpy().astype(bool)
    c_t = s["c"].numpy()
    c_p = c_pred_np[idx, :len(c_t)]
    b_t = s["baseline"].numpy()
    b_p = b_pred_np[idx]
    names = s["ref_names"]

    # Reconstruction
    recon = (c_p[:, None] * R_np).sum(axis=0) + b_p

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(grid[m], y_np[m], "k-", lw=0.6, alpha=0.5, label="mixture")
    ax.plot(grid[m], recon[m], "r-", lw=1.2, alpha=0.8, label="model recon")
    ax.plot(grid[m], b_p[m], "--", color="gray", lw=0.8, label="pred baseline")
    ax.plot(grid[m], b_t[m], ":", color="blue", lw=0.8, alpha=0.5, label="true baseline")

    for j in range(len(c_t)):
        if c_t[j] > 1e-6 or c_p[j] > 0.01:
            short = names[j].split(":")[-1][:18]
            ax.plot([], [], " ",
                    label=f"{short}: true={c_t[j]:.3f} pred={c_p[j]:.3f}")

    mae_i = float(np.mean(np.abs(c_t - c_p)))
    ax.set_title(f"[PRED / overfit] Sample {idx}: K={s['K']}, M={s['M']}, MAE={mae_i:.4f}")
    ax.set_xlabel("cm\u207b\u00b9")
    ax.set_ylabel("Intensity")
    ax.legend(fontsize=6.5, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / f"overfit_example_{idx}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGS / f'overfit_example_{idx}.png'}")

print("\nDONE")
