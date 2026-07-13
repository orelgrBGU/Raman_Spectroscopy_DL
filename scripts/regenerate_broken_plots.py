#!/usr/bin/env python
"""
Regenerate all plots that had the negative-baseline bug.

Parts B, C, D from notebook07_professional were generated with old
synth_mixtures code that produced negative baselines. This script
regenerates them using the FIXED synth_mixtures (positive baselines)
and NNLS only (no DL model needed — v2 model not yet trained).

Usage:
    cd /gpfs0/bgu-rgilad/users/orelgr/deep2
    /gpfs0/bgu-rgilad/users/orelgr/env/deep2_env/bin/python scripts/regenerate_broken_plots.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import numpy as np
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
FIGS = ROOT / "outputs" / "figs" / "07_professional"
FIGS.mkdir(parents=True, exist_ok=True)

from src.data.synth_mixtures import ChemicalPool, SynthConfig, make_fixed_batch
from src.baselines.nnls import nnls_decompose

COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
           "#a65628", "#f781bf", "#66c2a5", "#fc8d62", "#8da0cb",
           "#e78ac3", "#a6d854", "#ffd92f", "#b3b3b3"]

print("=" * 65)
print("  Regenerating broken plots (B, C, D) with fixed baselines")
print("  Using NNLS only (v2 DL model not yet trained)")
print("=" * 65)

pool = ChemicalPool.load()
train_pool, holdout_pool = pool.split(holdout_frac=0.2, seed=0)
grid = pool.grid


# ═══════════════════════════════════════════════════════════════════
# PART B — REVERSE: NNLS Decomposes the Mixture
# ═══════════════════════════════════════════════════════════════════
print("\n══ Part B: Reverse Engineering — NNLS Decomposition ══")

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

c_nnls, b_nnls = nnls_decompose(y_np, R_np, s["mask"].numpy(), grid)

# Figure B1: The challenge — we only see the mixture
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(wn, y_np[m], "k-", lw=1.5, alpha=0.8)
ax.fill_between(wn, 0, y_np[m], alpha=0.08, color="black")
ax.set_title("The Challenge: Given only this mixture, find c$_1$, c$_2$, ..., c$_K$")
ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_ylabel("Intensity")
ax.text(0.5, 0.85, "y = \u03a3 c$_i$ \u00b7 R$_i$ + baseline + noise\n"
        "Find: c$_i$ = ? for each known reference R$_i$",
        transform=ax.transAxes, fontsize=13, ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(FIGS / "B1_challenge.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved B1_challenge.png")

# Figure B2: NNLS decomposition (replaces old DL model decomposition)
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
ax.plot(wn, b_nnls[m], "--", color="gray", lw=1, alpha=0.6, label="NNLS baseline")
ax.set_title(f"NNLS Decomposition (MAE={np.mean(np.abs(c_true - c_nnls)):.4f})")
ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_ylabel("Intensity")
ax.legend(loc="upper right", fontsize=8)
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(FIGS / "B2_model_decomposition.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved B2_model_decomposition.png (NNLS)")

# Figure B3: same NNLS but showing component stacking
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

ax.plot(wn, b_nnls[m], "--", color="gray", lw=1.2, label="Estimated baseline")
ax.plot(wn, b_true[m], ":", color="blue", lw=0.8, alpha=0.5, label="True baseline")
ax.plot(wn, recon_nnls[m], "r-", lw=1.5, alpha=0.8, label="NNLS reconstruction")
ax.set_title(f"NNLS Decomposition — Component Stacking (MAE={np.mean(np.abs(c_true - c_nnls)):.4f})")
ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_ylabel("Intensity")
ax.legend(loc="upper right", fontsize=8)
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(FIGS / "B3_nnls_decomposition.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved B3_nnls_decomposition.png")

# Figure B4: Coefficient comparison bar chart (True vs NNLS)
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(K)
w = 0.35
short_names = [n.split(":")[-1][:15] for n in names[:K]]
ax.bar(x - w/2, c_true[:K], w, label="Ground Truth", color="#2196f3", edgecolor="k", lw=0.5)
ax.bar(x + w/2, c_nnls[:K], w, label="NNLS", color="#ff9800", edgecolor="k", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(short_names, rotation=30, ha="right")
ax.set_ylabel("Coefficient")
ax.set_title("Coefficient Comparison: True vs NNLS")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "B4_coefficient_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved B4_coefficient_comparison.png")


# ═══════════════════════════════════════════════════════════════════
# PART C — DEMO: Unseen Chemicals (NNLS only)
# ═══════════════════════════════════════════════════════════════════
print("\n══ Part C: DEMO — Unseen Chemicals (NNLS) ══")

N_DEMO = 6
demo_hold = make_fixed_batch(holdout_pool, n=N_DEMO, seed=2024,
    cfg=SynthConfig(K_min=3, K_max=5, M_min=1, M_max=3, seed=2024))

for i in range(N_DEMO):
    s = demo_hold[i]
    y_d = s["y"].numpy()
    R_d = s["R"].numpy()
    m_d = s["mask"].numpy().astype(bool)
    ct = s["c"].numpy()
    bt = s["baseline"].numpy()
    ns = s["ref_names"]
    K_d = s["K"]
    K_total = len(ct)
    wn_d = grid[m_d]

    c_nnls_d, b_nnls_d = nnls_decompose(y_d, R_d, s["mask"].numpy(), grid)
    recon_nnls_d = (c_nnls_d[:, None] * R_d).sum(axis=0) + b_nnls_d
    mae_nnls = np.mean(np.abs(ct - c_nnls_d))

    active = [(j, ns[j].split(":")[-1][:20], ct[j]) for j in range(K_total) if ct[j] > 1e-6]
    active_str = ", ".join(f"{name} ({c:.1%})" for _, name, c in active)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2, 1]})

    # Top: spectral decomposition
    ax = axes[0]
    ax.plot(wn_d, y_d[m_d], "k-", lw=1.2, alpha=0.4, label="Observed mixture")
    ax.plot(wn_d, recon_nnls_d[m_d], "-", color="#ff9800", lw=1.5, alpha=0.8,
            label=f"NNLS reconstruction (MAE={mae_nnls:.4f})")

    cumulative = np.zeros_like(wn_d, dtype=float)
    for j, name, c_j in active:
        weighted = ct[j] * R_d[j][m_d]
        ax.fill_between(wn_d, cumulative, cumulative + weighted,
                        alpha=0.2, color=COLORS[j % len(COLORS)])
        cumulative = cumulative + weighted

    ax.plot(wn_d, b_nnls_d[m_d], "--", color="gray", lw=1, alpha=0.6, label="NNLS baseline")
    ax.set_ylabel("Intensity")
    ax.set_title(f"[UNSEEN] Demo {i}: {active_str}")
    ax.legend(fontsize=8, loc="upper right")
    ax.invert_xaxis()

    # Bottom: coefficient bars
    ax2 = axes[1]
    x = np.arange(K_total)
    w = 0.35
    short_ns = [n.split(":")[-1][:12] for n in ns]
    ax2.bar(x - w/2, ct, w, label="True", color="#2196f3", edgecolor="k", lw=0.3)
    ax2.bar(x + w/2, c_nnls_d, w, label="NNLS", color="#ff9800", edgecolor="k", lw=0.3)
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
    print(f"  Demo {i}: K={K_d}, NNLS MAE={mae_nnls:.4f}")


# ═══════════════════════════════════════════════════════════════════
# PART D — Metrics Dashboard (NNLS baseline only)
# ═══════════════════════════════════════════════════════════════════
print("\n══ Part D: NNLS Baseline Metrics ══")

N_EVAL = 500
eval_samples = make_fixed_batch(holdout_pool, n=N_EVAL, seed=2024)

eval_nnls = []
for idx, s in enumerate(eval_samples):
    c_n, b_n = nnls_decompose(s["y"].numpy(), s["R"].numpy(), s["mask"].numpy(), grid)
    eval_nnls.append({
        "coeffs_pred": c_n,
        "baseline_pred": b_n,
        "coeffs_true": s["c"].numpy(),
        "K": s["K"],
        "M": s["M"],
        "snr_db": s["snr_db"],
    })
    if (idx + 1) % 100 == 0:
        print(f"    NNLS eval: {idx+1}/{N_EVAL}")

# Compute metrics
from sklearn.metrics import roc_auc_score, roc_curve
maes, rmses, sads = [], [], []
y_detect_true, y_detect_pred = [], []

for r in eval_nnls:
    ct = r["coeffs_true"]
    cp = r["coeffs_pred"]
    maes.append(np.mean(np.abs(ct - cp)))
    rmses.append(np.sqrt(np.mean((ct - cp) ** 2)))
    norm_t = np.linalg.norm(ct) + 1e-10
    norm_p = np.linalg.norm(cp) + 1e-10
    cos_angle = np.clip(np.dot(ct, cp) / (norm_t * norm_p), -1, 1)
    sads.append(np.arccos(cos_angle))
    for j in range(len(ct)):
        y_detect_true.append(1 if ct[j] > 0.02 else 0)
        y_detect_pred.append(float(cp[j]))

maes = np.array(maes)
rmses = np.array(rmses)
sads = np.array(sads)

ct_concat = np.concatenate([r["coeffs_true"] for r in eval_nnls])
cp_concat = np.concatenate([r["coeffs_pred"] for r in eval_nnls])
ss_res = np.sum((ct_concat - cp_concat) ** 2)
ss_tot = np.sum((ct_concat - ct_concat.mean()) ** 2) + 1e-10
r2_global = 1 - ss_res / ss_tot

auc = roc_auc_score(y_detect_true, y_detect_pred) if len(set(y_detect_true)) > 1 else float("nan")
fpr, tpr, _ = roc_curve(y_detect_true, y_detect_pred)

print(f"""
  NNLS Baseline Results (500 holdout samples):
  MAE:     {maes.mean():.4f} +/- {maes.std():.4f}
  RMSE:    {rmses.mean():.4f}
  SAD:     {sads.mean():.4f}
  R2:      {r2_global:.4f}
  AUC-ROC: {auc:.4f}
""")

# Figure D1: NNLS metrics summary
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
metric_names = ["MAE", "RMSE", "SAD", "R\u00b2", "AUC"]
vals = [maes.mean(), rmses.mean(), sads.mean(), max(0, r2_global), auc]
lower_better = [True, True, True, False, False]

for ax, name, v, lb in zip(axes, metric_names, vals, lower_better):
    ax.bar(["NNLS"], [v], color="#ff9800", edgecolor="k", lw=0.5)
    ax.set_title(name + (" \u2193" if lb else " \u2191"))
    ax.text(0, v + 0.01 * v, f"{v:.3f}", ha="center", va="bottom",
            fontweight="bold", fontsize=11)
fig.suptitle("NNLS Baseline Metrics (500 holdout samples, fixed data)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGS / "D1_metrics_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Figure D2: ROC Curve
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, "-", color="#ff9800", lw=2,
        label=f"NNLS (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve \u2014 Component Detection (c > 0.02?)")
ax.legend(loc="lower right")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(FIGS / "D2_roc_curve.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Figure D3: Scatter \u2014 predicted vs true
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(ct_concat, cp_concat, alpha=0.15, s=8, color="#ff9800", edgecolors="none")
lim = max(ct_concat.max(), cp_concat.max()) * 1.1
ax.plot([0, lim], [0, lim], "k--", lw=1)
ax.set_xlabel("True Coefficient")
ax.set_ylabel("Predicted Coefficient")
mae_all = np.mean(np.abs(ct_concat - cp_concat))
ax.set_title(f"NNLS: Predicted vs True Coefficients (MAE={mae_all:.4f})")
ax.set_xlim(-0.02, lim)
ax.set_ylim(-0.05, lim)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(FIGS / "D3_scatter_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Figure D4: MAE vs SNR
snr_arr = np.array([s["snr_db"] for s in eval_samples])
snr_bins = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]
fig, ax = plt.subplots(figsize=(9, 5))
centers = [(a+b)/2 for a, b in snr_bins]
nnls_means = [maes[(snr_arr >= a) & (snr_arr < b)].mean() for a, b in snr_bins]
nnls_stds = [maes[(snr_arr >= a) & (snr_arr < b)].std() for a, b in snr_bins]

ax.errorbar(centers, nnls_means, yerr=nnls_stds, fmt="s-", color="#ff9800",
            lw=2, capsize=5, label="NNLS", markersize=8)
ax.set_xlabel("Signal-to-Noise Ratio (dB)")
ax.set_ylabel("Coefficient MAE")
ax.set_title("Noise Robustness Analysis (NNLS Baseline)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "D4_mae_vs_snr.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Figure D5: MAE vs K
K_arr = np.array([s["K"] for s in eval_samples])
K_vals = sorted(set(K_arr))

fig, ax = plt.subplots(figsize=(9, 5))
nnls_by_k = [maes[K_arr == k] for k in K_vals]

bp = ax.boxplot(nnls_by_k, positions=K_vals,
                widths=0.5, patch_artist=True, showfliers=False)
for patch in bp["boxes"]:
    patch.set_facecolor("#ff9800")
    patch.set_alpha(0.5)

ax.set_xlabel("Number of Components (K)")
ax.set_ylabel("Coefficient MAE")
ax.set_title("NNLS Performance vs Mixture Complexity")
ax.set_xticks(K_vals)
fig.tight_layout()
fig.savefig(FIGS / "D5_mae_vs_K_boxplot.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Figure D6: MAE distribution
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(maes, bins=40, color="#ff9800", edgecolor="black", lw=0.3, alpha=0.8)
ax.axvline(maes.mean(), color="blue", ls="-", lw=1.5,
           label=f"Mean MAE = {maes.mean():.4f}")
ax.axvline(np.median(maes), color="green", ls="--", lw=1.5,
           label=f"Median MAE = {np.median(maes):.4f}")
ax.set_xlabel("Coefficient MAE")
ax.set_ylabel("Count")
ax.set_title(f"NNLS Baseline: MAE Distribution ({N_EVAL} holdout samples)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "D6_improvement_histogram.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"\n  Saved D1-D6: NNLS baseline metrics dashboard")
print(f"  All plots regenerated with FIXED positive baselines")
print("\nDONE")
