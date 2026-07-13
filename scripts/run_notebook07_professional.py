#!/usr/bin/env python
"""
Notebook 07 — Professional Decomposition Analysis
===================================================

Publication-quality evaluation following EGU-Net (Hong et al., IEEE TNNLS 2021)
and RamanFormer (PMC 2024) methodologies.

Evaluation protocol:
  - Metrics: MAE, RMSE, R², SAD, SRE, AUC-ROC (detection)
  - Visualizations:
    Part A — "Forward": How a mixture is constructed step-by-step
    Part B — "Reverse": How the model decomposes it back
    Part C — DEMO on unseen chemicals
    Part D — Comprehensive metrics dashboard

References:
  - EGU-Net: SAD + RMSE for abundance estimation
  - RamanFormer: MAE + RMSE + noise robustness analysis

Usage:
    cd /gpfs0/bgu-rgilad/users/orelgr/deep2
    python scripts/run_notebook07_professional.py --run_id run02
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.grid": True,
    "axes.axisbelow": True,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "grid.alpha": 0.3,
})

ROOT = Path("/gpfs0/bgu-rgilad/users/orelgr/deep2")

parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, default="run02")
args = parser.parse_args()

FIGS = ROOT / "outputs" / "figs" / "07_professional"
FIGS.mkdir(parents=True, exist_ok=True)

from src.data.synth_mixtures import ChemicalPool, SynthConfig, make_fixed_batch
from src.eval import load_model_from_checkpoint, predict_batch
from src.baselines.nnls import nnls_decompose

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 65)
print("  Notebook 07 — Professional Decomposition Analysis")
print("  Following EGU-Net & RamanFormer evaluation protocols")
print("=" * 65)

# ── Load ──
ckpt_path = ROOT / "checkpoints" / args.run_id / "best.pt"
model, info = load_model_from_checkpoint(ckpt_path, device)
pool = ChemicalPool.load()
train_pool, holdout_pool = pool.split(holdout_frac=0.2, seed=0)
grid = pool.grid

COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
           "#a65628", "#f781bf", "#66c2a5", "#fc8d62", "#8da0cb",
           "#e78ac3", "#a6d854", "#ffd92f", "#b3b3b3"]


# ═══════════════════════════════════════════════════════════════════
# METRICS (EGU-Net + RamanFormer style)
# ═══════════════════════════════════════════════════════════════════

def compute_all_metrics(results):
    """Compute comprehensive metrics following EGU-Net & RamanFormer."""
    maes, rmses, sads, sres, r2s = [], [], [], [], []
    y_detect_true, y_detect_pred = [], []

    for r in results:
        ct = r["coeffs_true"]
        cp = r["coeffs_pred"]

        # MAE (RamanFormer)
        maes.append(np.mean(np.abs(ct - cp)))

        # RMSE (EGU-Net: abundance RMSE)
        rmses.append(np.sqrt(np.mean((ct - cp) ** 2)))

        # SAD — Spectral Angle Distance (EGU-Net)
        # Treat coefficient vectors as "spectra"
        norm_t = np.linalg.norm(ct) + 1e-10
        norm_p = np.linalg.norm(cp) + 1e-10
        cos_angle = np.clip(np.dot(ct, cp) / (norm_t * norm_p), -1, 1)
        sads.append(np.arccos(cos_angle))

        # SRE — Signal-to-Reconstruction Error (dB)
        signal_power = np.sum(ct ** 2) + 1e-10
        error_power = np.sum((ct - cp) ** 2) + 1e-10
        sres.append(10 * np.log10(signal_power / error_power))

        # R² — computed globally after loop (per-sample is unstable with few points)

        # Detection: binary — is component present (c > threshold)?
        threshold = 0.02
        for j in range(len(ct)):
            y_detect_true.append(1 if ct[j] > threshold else 0)
            y_detect_pred.append(float(cp[j]))

    # AUC-ROC for component detection
    from sklearn.metrics import roc_auc_score, roc_curve
    auc = roc_auc_score(y_detect_true, y_detect_pred) if len(set(y_detect_true)) > 1 else float("nan")
    fpr, tpr, thresholds = roc_curve(y_detect_true, y_detect_pred)

    # Global R² (over all coefficients, not per-sample)
    all_ct = np.array(y_detect_true, dtype=float)  # reuse detection arrays
    all_cp = np.array(y_detect_pred)
    # But for proper R², use raw coefficient arrays
    ct_concat = np.concatenate([r["coeffs_true"] for r in results])
    cp_concat = np.concatenate([r["coeffs_pred"] for r in results])
    ss_res = np.sum((ct_concat - cp_concat) ** 2)
    ss_tot = np.sum((ct_concat - ct_concat.mean()) ** 2) + 1e-10
    r2_global = 1 - ss_res / ss_tot

    return {
        "MAE": np.array(maes),
        "RMSE": np.array(rmses),
        "SAD": np.array(sads),
        "SRE_dB": np.array(sres),
        "R2": r2_global,
        "AUC": auc,
        "roc_fpr": fpr,
        "roc_tpr": tpr,
        "detect_true": np.array(y_detect_true),
        "detect_pred": np.array(y_detect_pred),
    }


# ═══════════════════════════════════════════════════════════════════
# PART A — FORWARD: How a Mixture is Constructed
# ═══════════════════════════════════════════════════════════════════
print("\n══ Part A: Forward Process — Mixture Construction ══")

# Generate one clear example for the forward process
demo_samples = make_fixed_batch(train_pool, n=1, seed=2025,
    cfg=SynthConfig(K_min=4, K_max=4, M_min=0, M_max=0,
                    snr_db_range=(40, 50), seed=2025))
s = demo_samples[0]
y_np = s["y"].numpy()
R_np = s["R"].numpy()
m = s["mask"].numpy().astype(bool)
c_true = s["c"].numpy()
b_true = s["baseline"].numpy()
names = s["ref_names"]
K = s["K"]
wn = grid[m]

# Figure A1: Pure reference spectra (individual panels)
for j in range(K):
    short = names[j].split(":")[-1][:30]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(wn, R_np[j][m], color=COLORS[j], lw=1.5)
    ax.fill_between(wn, 0, R_np[j][m], alpha=0.15, color=COLORS[j])
    ax.set_title(f"Step 1 — Pure Reference Spectrum: {short}")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(FIGS / f"A1_pure_reference_{j}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# Figure A2: Weighting — multiply each pure spectrum by its coefficient
fig, ax = plt.subplots(figsize=(11, 5))
for j in range(K):
    short = names[j].split(":")[-1][:20]
    weighted = c_true[j] * R_np[j][m]
    ax.plot(wn, weighted, color=COLORS[j], lw=1.3, alpha=0.8,
            label=f"{short} × {c_true[j]:.3f}")
    ax.fill_between(wn, 0, weighted, alpha=0.1, color=COLORS[j])
ax.set_title("Step 2 — Weighted Components (c$_i$ × R$_i$)")
ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_ylabel("Intensity")
ax.legend(loc="upper right")
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(FIGS / "A2_weighted_components.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Figure A3: Superposition buildup
fig, ax = plt.subplots(figsize=(11, 5))
cumulative = np.zeros_like(wn, dtype=float)
for j in range(K):
    short = names[j].split(":")[-1][:20]
    cumulative = cumulative + c_true[j] * R_np[j][m]
    ax.plot(wn, cumulative, color=COLORS[j], lw=1.5,
            label=f"After adding {short}")
ax.plot(wn, cumulative + b_true[m], "--", color="gray", lw=1.2, label="+ Baseline")
ax.plot(wn, y_np[m], "k-", lw=0.8, alpha=0.3, label="Final (with noise)")
ax.set_title("Step 3 — Superposition: Building the Mixture")
ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_ylabel("Intensity")
ax.legend(loc="upper right", fontsize=8)
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(FIGS / "A3_superposition_buildup.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Figure A4: Final mixture equation
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(wn, y_np[m], "k-", lw=1.5, label="y = Σ(c$_i$·R$_i$) + baseline + noise")
ax.fill_between(wn, 0, y_np[m], alpha=0.08, color="black")
eq_parts = " + ".join([f"{c_true[j]:.2f}·R$_{j}$" for j in range(K)])
ax.set_title(f"Step 4 — Observed Mixture:  y = {eq_parts} + b + ε")
ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_ylabel("Intensity")
ax.legend(fontsize=10)
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(FIGS / "A4_final_mixture.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"  Saved A1-A4: Forward construction ({K} components)")


# ═══════════════════════════════════════════════════════════════════
# PART B — REVERSE: Model Decomposes the Mixture
# ═══════════════════════════════════════════════════════════════════
print("\n══ Part B: Reverse Engineering — Model Decomposition ══")

result_model = predict_batch(model, demo_samples, device)[0]
c_nnls, b_nnls = nnls_decompose(y_np, R_np, s["mask"].numpy(), grid)

cp_model = result_model["coeffs_pred"]
bp_model = result_model["baseline_pred"]

# Figure B1: The challenge — we only see the mixture
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(wn, y_np[m], "k-", lw=1.5, alpha=0.8)
ax.fill_between(wn, 0, y_np[m], alpha=0.08, color="black")
ax.set_title("The Challenge: Given only this mixture, find c$_1$, c$_2$, ..., c$_K$")
ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_ylabel("Intensity")
ax.text(0.5, 0.85, "y = Σ c$_i$ · R$_i$ + baseline + noise\n"
        "Find: c$_i$ = ? for each known reference R$_i$",
        transform=ax.transAxes, fontsize=13, ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(FIGS / "B1_challenge.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Figure B2: Model's predicted decomposition
recon_model = (cp_model[:, None] * R_np).sum(axis=0) + bp_model

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(wn, y_np[m], "k-", lw=0.8, alpha=0.3, label="Observed mixture")

cumulative = np.zeros_like(wn, dtype=float)
for j in range(K):
    if abs(cp_model[j]) > 0.005:
        short = names[j].split(":")[-1][:20]
        weighted = cp_model[j] * R_np[j][m]
        ax.fill_between(wn, cumulative, cumulative + weighted,
                        alpha=0.35, color=COLORS[j],
                        label=f"{short}: pred={cp_model[j]:.3f} (true={c_true[j]:.3f})")
        cumulative = cumulative + weighted

ax.plot(wn, recon_model[m], "r-", lw=1.5, alpha=0.8, label="DL reconstruction")
ax.set_title(f"DL Model Decomposition (MAE={np.mean(np.abs(c_true - cp_model)):.4f})")
ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_ylabel("Intensity")
ax.legend(loc="upper right", fontsize=8)
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(FIGS / "B2_model_decomposition.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Figure B3: NNLS decomposition (same mixture)
recon_nnls = (c_nnls[:, None] * R_np).sum(axis=0) + b_nnls

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(wn, y_np[m], "k-", lw=0.8, alpha=0.3, label="Observed mixture")

cumulative = np.zeros_like(wn, dtype=float)
for j in range(K):
    if abs(c_nnls[j]) > 0.005:
        short = names[j].split(":")[-1][:20]
        weighted = c_nnls[j] * R_np[j][m]
        ax.fill_between(wn, cumulative, cumulative + weighted,
                        alpha=0.35, color=COLORS[j],
                        label=f"{short}: pred={c_nnls[j]:.3f} (true={c_true[j]:.3f})")
        cumulative = cumulative + weighted

ax.plot(wn, recon_nnls[m], "r-", lw=1.5, alpha=0.8, label="NNLS reconstruction")
ax.set_title(f"NNLS Decomposition (MAE={np.mean(np.abs(c_true - c_nnls)):.4f})")
ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_ylabel("Intensity")
ax.legend(loc="upper right", fontsize=8)
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(FIGS / "B3_nnls_decomposition.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Figure B4: Coefficient comparison bar chart
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(K)
w = 0.25
short_names = [n.split(":")[-1][:15] for n in names[:K]]
ax.bar(x - w, c_true[:K], w, label="Ground Truth", color="#2196f3", edgecolor="k", lw=0.5)
ax.bar(x, cp_model[:K], w, label="DL Model", color="#f44336", edgecolor="k", lw=0.5)
ax.bar(x + w, c_nnls[:K], w, label="NNLS", color="#ff9800", edgecolor="k", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(short_names, rotation=30, ha="right")
ax.set_ylabel("Coefficient")
ax.set_title("Coefficient Comparison: True vs DL vs NNLS")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "B4_coefficient_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"  Saved B1-B4: Reverse engineering decomposition")


# ═══════════════════════════════════════════════════════════════════
# PART C — DEMO: Unseen Chemicals
# ═══════════════════════════════════════════════════════════════════
print("\n══ Part C: DEMO — Unseen Chemicals ══")

N_DEMO = 6
demo_hold = make_fixed_batch(holdout_pool, n=N_DEMO, seed=2024,
    cfg=SynthConfig(K_min=3, K_max=5, M_min=1, M_max=3, seed=2024))
demo_model_results = predict_batch(model, demo_hold, device)

for i in range(N_DEMO):
    s = demo_hold[i]
    r = demo_model_results[i]
    y_d = s["y"].numpy()
    R_d = s["R"].numpy()
    m_d = s["mask"].numpy().astype(bool)
    ct = s["c"].numpy()
    cp = r["coeffs_pred"]
    bt = s["baseline"].numpy()
    bp = r["baseline_pred"]
    ns = s["ref_names"]
    K_d = s["K"]
    K_total = len(ct)
    wn_d = grid[m_d]

    # NNLS for comparison
    c_nnls_d, b_nnls_d = nnls_decompose(y_d, R_d, s["mask"].numpy(), grid)

    recon_dl = (cp[:, None] * R_d).sum(axis=0) + bp
    recon_nnls_d = (c_nnls_d[:, None] * R_d).sum(axis=0) + b_nnls_d

    mae_dl = np.mean(np.abs(ct - cp))
    mae_nnls = np.mean(np.abs(ct - c_nnls_d))

    active = [(j, ns[j].split(":")[-1][:20], ct[j]) for j in range(K_total) if ct[j] > 1e-6]
    active_str = ", ".join(f"{name} ({c:.1%})" for _, name, c in active)
    winner = "DL" if mae_dl < mae_nnls else "NNLS"

    # Main decomposition figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2, 1]})

    # Top: spectral decomposition
    ax = axes[0]
    ax.plot(wn_d, y_d[m_d], "k-", lw=1.2, alpha=0.4, label="Observed mixture")
    ax.plot(wn_d, recon_dl[m_d], "r-", lw=1.5, alpha=0.8,
            label=f"DL reconstruction (MAE={mae_dl:.4f})")
    ax.plot(wn_d, recon_nnls_d[m_d], "--", color="#ff9800", lw=1.3, alpha=0.8,
            label=f"NNLS reconstruction (MAE={mae_nnls:.4f})")

    cumulative = np.zeros_like(wn_d, dtype=float)
    for j, name, c_j in active:
        weighted = ct[j] * R_d[j][m_d]
        ax.fill_between(wn_d, cumulative, cumulative + weighted,
                        alpha=0.2, color=COLORS[j % len(COLORS)])
        cumulative = cumulative + weighted

    ax.set_ylabel("Intensity")
    ax.set_title(f"[UNSEEN] Demo {i}: {active_str}  —  Winner: {winner}")
    ax.legend(fontsize=8, loc="upper right")
    ax.invert_xaxis()

    # Bottom: coefficient bars
    ax2 = axes[1]
    x = np.arange(K_total)
    w = 0.25
    short_ns = [n.split(":")[-1][:12] for n in ns]
    ax2.bar(x - w, ct, w, label="True", color="#2196f3", edgecolor="k", lw=0.3)
    ax2.bar(x, cp, w, label="DL", color="#f44336", edgecolor="k", lw=0.3)
    ax2.bar(x + w, c_nnls_d, w, label="NNLS", color="#ff9800", edgecolor="k", lw=0.3)
    # Mark distractors
    for j in range(K_total):
        if ct[j] < 1e-6:
            ax2.axvspan(j - 0.45, j + 0.45, alpha=0.06, color="gray")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_ns, rotation=40, ha="right", fontsize=7)
    ax2.set_ylabel("Coefficient")
    ax2.set_xlabel("Reference (gray = distractor)")
    ax2.legend(fontsize=8)
    ax2.axhline(0, color="k", lw=0.5)

    fig.tight_layout()
    fig.savefig(FIGS / f"C_demo_unseen_{i}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Demo {i}: K={K_d}, DL MAE={mae_dl:.4f}, NNLS MAE={mae_nnls:.4f} — {winner}")


# ═══════════════════════════════════════════════════════════════════
# PART D — Comprehensive Metrics Dashboard
# ═══════════════════════════════════════════════════════════════════
print("\n══ Part D: Comprehensive Metrics (EGU-Net + RamanFormer) ══")

N_EVAL = 500
eval_samples = make_fixed_batch(holdout_pool, n=N_EVAL, seed=2024)
eval_model = predict_batch(model, eval_samples, device)

# NNLS results
eval_nnls = []
for s in eval_samples:
    c_n, b_n = nnls_decompose(s["y"].numpy(), s["R"].numpy(), s["mask"].numpy(), grid)
    eval_nnls.append({"coeffs_pred": c_n, "baseline_pred": b_n,
                      "coeffs_true": s["c"].numpy(), "K": s["K"],
                      "M": s["M"], "snr_db": s["snr_db"]})
for i, s in enumerate(eval_samples):
    eval_nnls[i]["coeffs_true"] = s["c"].numpy()

m_dl = compute_all_metrics(eval_model)
m_nnls = compute_all_metrics(eval_nnls)

# Print results table
print(f"""
  ┌──────────────────────┬────────────────┬────────────────┐
  │ Metric               │ DL Model       │ NNLS           │
  ├──────────────────────┼────────────────┼────────────────┤
  │ MAE (mean±std)       │ {m_dl['MAE'].mean():.4f} ± {m_dl['MAE'].std():.4f}  │ {m_nnls['MAE'].mean():.4f} ± {m_nnls['MAE'].std():.4f}  │
  │ RMSE (mean)          │ {m_dl['RMSE'].mean():.4f}          │ {m_nnls['RMSE'].mean():.4f}          │
  │ SAD (mean, rad)      │ {m_dl['SAD'].mean():.4f}          │ {m_nnls['SAD'].mean():.4f}          │
  │ SRE (mean, dB)       │ {m_dl['SRE_dB'].mean():.2f}           │ {m_nnls['SRE_dB'].mean():.2f}           │
  │ R² (mean)            │ {m_dl['R2'].mean():.4f}          │ {m_nnls['R2'].mean():.4f}          │
  │ AUC-ROC (detection)  │ {m_dl['AUC']:.4f}          │ {m_nnls['AUC']:.4f}          │
  └──────────────────────┴────────────────┴────────────────┘
""")

# Figure D1: Metrics bar chart
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
metric_names = ["MAE", "RMSE", "SAD", "R²", "AUC"]
dl_vals = [m_dl["MAE"].mean(), m_dl["RMSE"].mean(), m_dl["SAD"].mean(),
           max(0, m_dl["R2"]), m_dl["AUC"]]
nnls_vals = [m_nnls["MAE"].mean(), m_nnls["RMSE"].mean(), m_nnls["SAD"].mean(),
             max(0, m_nnls["R2"]), m_nnls["AUC"]]
# For MAE, RMSE, SAD: lower is better. For R², AUC: higher is better
lower_better = [True, True, True, False, False]

for ax, name, dl_v, nnls_v, lb in zip(axes, metric_names, dl_vals, nnls_vals, lower_better):
    colors_bar = ["#f44336", "#ff9800"]
    ax.bar(["DL", "NNLS"], [dl_v, nnls_v], color=colors_bar, edgecolor="k", lw=0.5)
    ax.set_title(name + (" ↓" if lb else " ↑"))
    for bar_i, v in enumerate([dl_v, nnls_v]):
        ax.text(bar_i, v + 0.01 * max(dl_v, nnls_v), f"{v:.3f}",
                ha="center", va="bottom", fontweight="bold", fontsize=9)
fig.suptitle("Comprehensive Metrics: DL Model vs NNLS (EGU-Net / RamanFormer protocol)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGS / "D1_metrics_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Figure D2: ROC Curve (component detection)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(m_dl["roc_fpr"], m_dl["roc_tpr"], "r-", lw=2,
        label=f"DL Model (AUC={m_dl['AUC']:.3f})")
ax.plot(m_nnls["roc_fpr"], m_nnls["roc_tpr"], "-", color="#ff9800", lw=2,
        label=f"NNLS (AUC={m_nnls['AUC']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Component Detection (c > 0.02?)")
ax.legend(loc="lower right")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(FIGS / "D2_roc_curve.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Figure D3: Scatter — predicted vs true (side by side)
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

for ax, results, name, color in [
    (axes[0], eval_model, "DL Model", "#f44336"),
    (axes[1], eval_nnls, "NNLS", "#ff9800")
]:
    ct_all = np.concatenate([r["coeffs_true"] for r in results])
    cp_all = np.concatenate([r["coeffs_pred"] for r in results])
    ax.scatter(ct_all, cp_all, alpha=0.15, s=8, color=color, edgecolors="none")
    lim = max(ct_all.max(), cp_all.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("True Coefficient")
    ax.set_ylabel("Predicted Coefficient")
    mae = np.mean(np.abs(ct_all - cp_all))
    ax.set_title(f"{name} (MAE={mae:.4f})")
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.05, lim)
    ax.set_aspect("equal")

fig.suptitle("Predicted vs True Coefficients (500 holdout samples)", fontweight="bold")
fig.tight_layout()
fig.savefig(FIGS / "D3_scatter_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Figure D4: MAE vs SNR (RamanFormer-style noise robustness)
snr_arr = np.array([s["snr_db"] for s in eval_samples])
dl_mae_arr = m_dl["MAE"]
nnls_mae_arr = m_nnls["MAE"]

snr_bins = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]
fig, ax = plt.subplots(figsize=(9, 5))
centers = [(a+b)/2 for a, b in snr_bins]
dl_means = [dl_mae_arr[(snr_arr >= a) & (snr_arr < b)].mean() for a, b in snr_bins]
dl_stds = [dl_mae_arr[(snr_arr >= a) & (snr_arr < b)].std() for a, b in snr_bins]
nnls_means = [nnls_mae_arr[(snr_arr >= a) & (snr_arr < b)].mean() for a, b in snr_bins]
nnls_stds = [nnls_mae_arr[(snr_arr >= a) & (snr_arr < b)].std() for a, b in snr_bins]

ax.errorbar(centers, dl_means, yerr=dl_stds, fmt="o-", color="#f44336",
            lw=2, capsize=5, label="DL Model", markersize=8)
ax.errorbar(centers, nnls_means, yerr=nnls_stds, fmt="s-", color="#ff9800",
            lw=2, capsize=5, label="NNLS", markersize=8)
ax.set_xlabel("Signal-to-Noise Ratio (dB)")
ax.set_ylabel("Coefficient MAE")
ax.set_title("Noise Robustness Analysis (RamanFormer protocol)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "D4_mae_vs_snr.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Figure D5: MAE vs K (mixture complexity)
K_arr = np.array([s["K"] for s in eval_samples])
K_vals = sorted(set(K_arr))

fig, ax = plt.subplots(figsize=(9, 5))
dl_by_k = [dl_mae_arr[K_arr == k] for k in K_vals]
nnls_by_k = [nnls_mae_arr[K_arr == k] for k in K_vals]

bp1 = ax.boxplot([d for d in dl_by_k], positions=np.array(K_vals) - 0.2,
                  widths=0.35, patch_artist=True, showfliers=False)
bp2 = ax.boxplot([d for d in nnls_by_k], positions=np.array(K_vals) + 0.2,
                  widths=0.35, patch_artist=True, showfliers=False)
for patch in bp1["boxes"]: patch.set_facecolor("#f44336"); patch.set_alpha(0.5)
for patch in bp2["boxes"]: patch.set_facecolor("#ff9800"); patch.set_alpha(0.5)

ax.set_xlabel("Number of Components (K)")
ax.set_ylabel("Coefficient MAE")
ax.set_title("Performance vs Mixture Complexity")
ax.set_xticks(K_vals)
# Manual legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#f44336", alpha=0.5, label="DL Model"),
                    Patch(facecolor="#ff9800", alpha=0.5, label="NNLS")])
fig.tight_layout()
fig.savefig(FIGS / "D5_mae_vs_K_boxplot.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Figure D6: Per-sample improvement histogram
improvement = nnls_mae_arr - dl_mae_arr  # positive = DL wins
pct_better = 100 * (improvement > 0).mean()

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(improvement, bins=60, color="#4caf50", edgecolor="black", lw=0.3, alpha=0.8)
ax.axvline(0, color="red", ls="--", lw=1.5, label="Break-even")
ax.axvline(improvement.mean(), color="blue", ls="-", lw=1.5,
           label=f"Mean = {improvement.mean():.4f}")
ax.set_xlabel("MAE$_{NNLS}$ − MAE$_{DL}$ (positive = DL wins)")
ax.set_ylabel("Count")
ax.set_title(f"Per-Sample Improvement: DL wins in {pct_better:.0f}% of {N_EVAL} samples")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "D6_improvement_histogram.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"  Saved D1-D6: Comprehensive metrics dashboard")


# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
winner = "DL Model" if m_dl["MAE"].mean() < m_nnls["MAE"].mean() else "NNLS"
all_figs = list(FIGS.glob("*.png"))
print(f"\n  Total figures: {len(all_figs)}")
print(f"  Overall winner: {winner}")
print(f"  All saved to: {FIGS}/")
print("\nDONE")
