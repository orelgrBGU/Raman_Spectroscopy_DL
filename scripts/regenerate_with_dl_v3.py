#!/usr/bin/env python
"""
Regenerate all professional plots (D-series) and eval_v3 with BOTH DL v3 and NNLS.

Previously D-series only showed NNLS. Now that v3 is trained, we add DL results
to all comparison plots: D1 metrics, D2 ROC, D3 scatter, D4 MAE vs SNR,
D5 MAE vs K, D6 MAE distribution. Also adds missing plots to 06_eval_v3/:
ROC curve, MAE vs SNR.

Usage:
    cd /gpfs0/bgu-rgilad/users/orelgr/deep2
    /gpfs0/bgu-rgilad/users/orelgr/env/deep2_env/bin/python scripts/regenerate_with_dl_v3.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import numpy as np
import torch
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
FIGS_PROF = ROOT / "outputs" / "figs" / "07_professional"
FIGS_EVAL = ROOT / "outputs" / "figs" / "06_eval_v3"
FIGS_PROF.mkdir(parents=True, exist_ok=True)
FIGS_EVAL.mkdir(parents=True, exist_ok=True)

from src.data.synth_mixtures import ChemicalPool, SynthConfig, make_fixed_batch
from src.baselines.nnls import nnls_decompose
from src.eval import load_model_from_checkpoint, predict_batch, compute_metrics, nnls_predict_batch

COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
           "#a65628", "#f781bf", "#66c2a5", "#fc8d62", "#8da0cb"]

DL_COLOR = "#2196f3"
NNLS_COLOR = "#ff9800"

print("=" * 65)
print("  Regenerating D-series + eval_v3 with DL v3 + NNLS")
print("=" * 65)

# ── Load model ──────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_path = ROOT / "checkpoints" / "v3_run01" / "best.pt"
print(f"\nLoading DL v3 from {ckpt_path} on {device}")
model, info = load_model_from_checkpoint(ckpt_path, device=device)
print(f"  Epoch {info['epoch']}, best metric: {info['best_metric']:.6f}")

# ── Load data ───────────────────────────────────────────────────────
pool = ChemicalPool.load()
train_pool, holdout_pool = pool.split(holdout_frac=0.2, seed=0)
grid = pool.grid

N_EVAL = 500
eval_samples = make_fixed_batch(holdout_pool, n=N_EVAL, seed=2024)
print(f"\nGenerated {N_EVAL} holdout samples")

# ── Run DL v3 inference ────────────────────────────────────────────
print("\nRunning DL v3 inference...")
BATCH = 64
dl_results = []
for start in range(0, N_EVAL, BATCH):
    batch = eval_samples[start:start+BATCH]
    dl_results.extend(predict_batch(model, batch, device=device))
    print(f"  DL inference: {min(start+BATCH, N_EVAL)}/{N_EVAL}")

# ── Run NNLS ───────────────────────────────────────────────────────
print("\nRunning NNLS baseline...")
nnls_results = []
for idx, s in enumerate(eval_samples):
    c_n, b_n = nnls_decompose(s["y"].numpy(), s["R"].numpy(), s["mask"].numpy(), grid)
    nnls_results.append({
        "coeffs_pred": c_n,
        "baseline_pred": b_n,
        "coeffs_true": s["c"].numpy(),
        "K": s["K"],
        "M": s["M"],
        "snr_db": s["snr_db"],
    })
    if (idx + 1) % 100 == 0:
        print(f"  NNLS eval: {idx+1}/{N_EVAL}")

# ── Compute metrics for both ────────────────────────────────────────
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

# DL metrics
dl_maes, dl_rmses, dl_sads = [], [], []
dl_detect_true, dl_detect_pred = [], []
for r in dl_results:
    ct = r["coeffs_true"]
    cp = r["coeffs_pred"]
    dl_maes.append(np.mean(np.abs(ct - cp)))
    dl_rmses.append(np.sqrt(np.mean((ct - cp) ** 2)))
    norm_t = np.linalg.norm(ct) + 1e-10
    norm_p = np.linalg.norm(cp) + 1e-10
    cos_angle = np.clip(np.dot(ct, cp) / (norm_t * norm_p), -1, 1)
    dl_sads.append(np.arccos(cos_angle))
    for j in range(len(ct)):
        dl_detect_true.append(1 if ct[j] > 0.02 else 0)
        dl_detect_pred.append(float(cp[j]))

dl_maes = np.array(dl_maes)
dl_rmses = np.array(dl_rmses)
dl_sads = np.array(dl_sads)

dl_ct_all = np.concatenate([r["coeffs_true"] for r in dl_results])
dl_cp_all = np.concatenate([r["coeffs_pred"] for r in dl_results])
dl_ss_res = np.sum((dl_ct_all - dl_cp_all) ** 2)
dl_ss_tot = np.sum((dl_ct_all - dl_ct_all.mean()) ** 2) + 1e-10
dl_r2 = 1 - dl_ss_res / dl_ss_tot
dl_auc = roc_auc_score(dl_detect_true, dl_detect_pred)
dl_fpr, dl_tpr, _ = roc_curve(dl_detect_true, dl_detect_pred)

# DL Spearman
dl_spearman_list = []
for r in dl_results:
    ct = r["coeffs_true"]
    cp = r["coeffs_pred"]
    if np.std(ct) > 1e-9 and np.std(cp) > 1e-9:
        rho, _ = spearmanr(ct, cp)
        dl_spearman_list.append(rho)
dl_spearman = np.mean(dl_spearman_list) if dl_spearman_list else float("nan")

# NNLS metrics
nnls_maes, nnls_rmses, nnls_sads = [], [], []
nnls_detect_true, nnls_detect_pred = [], []
for r in nnls_results:
    ct = r["coeffs_true"]
    cp = r["coeffs_pred"]
    nnls_maes.append(np.mean(np.abs(ct - cp)))
    nnls_rmses.append(np.sqrt(np.mean((ct - cp) ** 2)))
    norm_t = np.linalg.norm(ct) + 1e-10
    norm_p = np.linalg.norm(cp) + 1e-10
    cos_angle = np.clip(np.dot(ct, cp) / (norm_t * norm_p), -1, 1)
    nnls_sads.append(np.arccos(cos_angle))
    for j in range(len(ct)):
        nnls_detect_true.append(1 if ct[j] > 0.02 else 0)
        nnls_detect_pred.append(float(cp[j]))

nnls_maes = np.array(nnls_maes)
nnls_rmses = np.array(nnls_rmses)
nnls_sads = np.array(nnls_sads)

nnls_ct_all = np.concatenate([r["coeffs_true"] for r in nnls_results])
nnls_cp_all = np.concatenate([r["coeffs_pred"] for r in nnls_results])
nnls_ss_res = np.sum((nnls_ct_all - nnls_cp_all) ** 2)
nnls_ss_tot = np.sum((nnls_ct_all - nnls_ct_all.mean()) ** 2) + 1e-10
nnls_r2 = 1 - nnls_ss_res / nnls_ss_tot
nnls_auc = roc_auc_score(nnls_detect_true, nnls_detect_pred)
nnls_fpr, nnls_tpr, _ = roc_curve(nnls_detect_true, nnls_detect_pred)

# NNLS Spearman
nnls_spearman_list = []
for r in nnls_results:
    ct = r["coeffs_true"]
    cp = r["coeffs_pred"]
    if np.std(ct) > 1e-9 and np.std(cp) > 1e-9:
        rho, _ = spearmanr(ct, cp)
        nnls_spearman_list.append(rho)
nnls_spearman = np.mean(nnls_spearman_list) if nnls_spearman_list else float("nan")

print(f"""
  ┌─────────────┬───────────────┬───────────────┐
  │ Metric      │ DL v3         │ NNLS          │
  ├─────────────┼───────────────┼───────────────┤
  │ MAE         │ {dl_maes.mean():.4f}         │ {nnls_maes.mean():.4f}         │
  │ RMSE        │ {dl_rmses.mean():.4f}         │ {nnls_rmses.mean():.4f}         │
  │ SAD         │ {dl_sads.mean():.4f}         │ {nnls_sads.mean():.4f}         │
  │ R²          │ {dl_r2:.4f}         │ {nnls_r2:.4f}         │
  │ AUC-ROC     │ {dl_auc:.4f}         │ {nnls_auc:.4f}         │
  │ Spearman    │ {dl_spearman:.4f}         │ {nnls_spearman:.4f}         │
  └─────────────┴───────────────┴───────────────┘
""")

snr_arr = np.array([s["snr_db"] for s in eval_samples])
K_arr = np.array([s["K"] for s in eval_samples])

# ═══════════════════════════════════════════════════════════════════
# D1: Metrics Comparison (DL v3 vs NNLS)
# ═══════════════════════════════════════════════════════════════════
print("Generating D1: Metrics comparison...")
fig, axes = plt.subplots(1, 6, figsize=(22, 4))
metric_names = ["MAE", "RMSE", "SAD", "R²", "AUC-ROC", "Spearman"]
dl_vals = [dl_maes.mean(), dl_rmses.mean(), dl_sads.mean(), max(0, dl_r2), dl_auc, dl_spearman]
nnls_vals = [nnls_maes.mean(), nnls_rmses.mean(), nnls_sads.mean(), max(0, nnls_r2), nnls_auc, nnls_spearman]
lower_better = [True, True, True, False, False, False]

for ax, name, dv, nv, lb in zip(axes, metric_names, dl_vals, nnls_vals, lower_better):
    x = np.arange(2)
    bars = ax.bar(x, [dv, nv], color=[DL_COLOR, NNLS_COLOR], edgecolor="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["DL v3", "NNLS"])
    ax.set_title(name + (" ↓" if lb else " ↑"))
    for bar, v in zip(bars, [dv, nv]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    # Highlight winner
    if lb:
        winner = 0 if dv < nv else 1
    else:
        winner = 0 if dv > nv else 1
    bars[winner].set_edgecolor("gold")
    bars[winner].set_linewidth(2.5)

fig.suptitle("DL v3 vs NNLS — Comprehensive Metrics (500 holdout samples)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGS_PROF / "D1_metrics_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved D1_metrics_comparison.png")

# ═══════════════════════════════════════════════════════════════════
# D2: ROC Curve (DL v3 + NNLS)
# ═══════════════════════════════════════════════════════════════════
print("Generating D2: ROC curve...")
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(dl_fpr, dl_tpr, "-", color=DL_COLOR, lw=2.5,
        label=f"DL v3 (AUC={dl_auc:.3f})")
ax.plot(nnls_fpr, nnls_tpr, "-", color=NNLS_COLOR, lw=2.5,
        label=f"NNLS (AUC={nnls_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Component Detection (c > 0.02?)")
ax.legend(loc="lower right", fontsize=11)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(FIGS_PROF / "D2_roc_curve.png", dpi=200, bbox_inches="tight")
fig.savefig(FIGS_EVAL / "roc_curve.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved D2_roc_curve.png + 06_eval_v3/roc_curve.png")

# ═══════════════════════════════════════════════════════════════════
# D3: Scatter — Predicted vs True (DL v3 + NNLS side by side)
# ═══════════════════════════════════════════════════════════════════
print("Generating D3: Scatter comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

lim = max(dl_ct_all.max(), dl_cp_all.max(), nnls_cp_all.max()) * 1.1

ax1.scatter(dl_ct_all, dl_cp_all, alpha=0.12, s=6, color=DL_COLOR, edgecolors="none")
ax1.plot([0, lim], [0, lim], "k--", lw=1)
ax1.set_xlabel("True Coefficient")
ax1.set_ylabel("Predicted Coefficient")
ax1.set_title(f"DL v3 (MAE={np.mean(np.abs(dl_ct_all - dl_cp_all)):.4f})")
ax1.set_xlim(-0.02, lim)
ax1.set_ylim(-0.05, lim)
ax1.set_aspect("equal")

ax2.scatter(nnls_ct_all, nnls_cp_all, alpha=0.12, s=6, color=NNLS_COLOR, edgecolors="none")
ax2.plot([0, lim], [0, lim], "k--", lw=1)
ax2.set_xlabel("True Coefficient")
ax2.set_ylabel("Predicted Coefficient")
ax2.set_title(f"NNLS (MAE={np.mean(np.abs(nnls_ct_all - nnls_cp_all)):.4f})")
ax2.set_xlim(-0.02, lim)
ax2.set_ylim(-0.05, lim)
ax2.set_aspect("equal")

fig.suptitle("Predicted vs True Coefficients (500 holdout samples)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGS_PROF / "D3_scatter_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved D3_scatter_comparison.png")

# ═══════════════════════════════════════════════════════════════════
# D4: MAE vs SNR (DL v3 + NNLS)
# ═══════════════════════════════════════════════════════════════════
print("Generating D4: MAE vs SNR...")
snr_bins = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]
centers = [(a+b)/2 for a, b in snr_bins]

dl_snr_means = [dl_maes[(snr_arr >= a) & (snr_arr < b)].mean() for a, b in snr_bins]
dl_snr_stds = [dl_maes[(snr_arr >= a) & (snr_arr < b)].std() for a, b in snr_bins]
nnls_snr_means = [nnls_maes[(snr_arr >= a) & (snr_arr < b)].mean() for a, b in snr_bins]
nnls_snr_stds = [nnls_maes[(snr_arr >= a) & (snr_arr < b)].std() for a, b in snr_bins]

fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(centers, dl_snr_means, yerr=dl_snr_stds, fmt="o-", color=DL_COLOR,
            lw=2, capsize=5, label="DL v3", markersize=8)
ax.errorbar(centers, nnls_snr_means, yerr=nnls_snr_stds, fmt="s-", color=NNLS_COLOR,
            lw=2, capsize=5, label="NNLS", markersize=8)
ax.set_xlabel("Signal-to-Noise Ratio (dB)")
ax.set_ylabel("Coefficient MAE")
ax.set_title("Noise Robustness: DL v3 vs NNLS")
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(FIGS_PROF / "D4_mae_vs_snr.png", dpi=200, bbox_inches="tight")
fig.savefig(FIGS_EVAL / "mae_vs_snr.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved D4_mae_vs_snr.png + 06_eval_v3/mae_vs_snr.png")

# ═══════════════════════════════════════════════════════════════════
# D5: MAE vs K (DL v3 + NNLS boxplots side by side)
# ═══════════════════════════════════════════════════════════════════
print("Generating D5: MAE vs K...")
K_vals = sorted(set(K_arr))

fig, ax = plt.subplots(figsize=(10, 5))
width = 0.35
for i, k in enumerate(K_vals):
    mask_k = K_arr == k
    dl_data = dl_maes[mask_k]
    nnls_data = nnls_maes[mask_k]

    bp_dl = ax.boxplot([dl_data], positions=[k - width/2], widths=width,
                       patch_artist=True, showfliers=False)
    bp_nnls = ax.boxplot([nnls_data], positions=[k + width/2], widths=width,
                         patch_artist=True, showfliers=False)
    for patch in bp_dl["boxes"]:
        patch.set_facecolor(DL_COLOR)
        patch.set_alpha(0.5)
    for patch in bp_nnls["boxes"]:
        patch.set_facecolor(NNLS_COLOR)
        patch.set_alpha(0.5)

# Legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=DL_COLOR, alpha=0.5, label="DL v3"),
                   Patch(facecolor=NNLS_COLOR, alpha=0.5, label="NNLS")],
          fontsize=11)
ax.set_xlabel("Number of Components (K)")
ax.set_ylabel("Coefficient MAE")
ax.set_title("Performance vs Mixture Complexity: DL v3 vs NNLS")
ax.set_xticks(K_vals)
fig.tight_layout()
fig.savefig(FIGS_PROF / "D5_mae_vs_K_boxplot.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved D5_mae_vs_K_boxplot.png")

# ═══════════════════════════════════════════════════════════════════
# D6: MAE Distribution (DL v3 + NNLS overlaid)
# ═══════════════════════════════════════════════════════════════════
print("Generating D6: MAE distribution...")
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(0, max(dl_maes.max(), nnls_maes.max()), 45)
ax.hist(dl_maes, bins=bins, color=DL_COLOR, edgecolor="black", lw=0.3,
        alpha=0.5, label=f"DL v3 (mean={dl_maes.mean():.4f})")
ax.hist(nnls_maes, bins=bins, color=NNLS_COLOR, edgecolor="black", lw=0.3,
        alpha=0.5, label=f"NNLS (mean={nnls_maes.mean():.4f})")
ax.axvline(dl_maes.mean(), color=DL_COLOR, ls="-", lw=2)
ax.axvline(nnls_maes.mean(), color=NNLS_COLOR, ls="-", lw=2)
ax.set_xlabel("Coefficient MAE")
ax.set_ylabel("Count")
ax.set_title(f"MAE Distribution: DL v3 vs NNLS ({N_EVAL} holdout samples)")
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(FIGS_PROF / "D6_improvement_histogram.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved D6_improvement_histogram.png")

# ═══════════════════════════════════════════════════════════════════
# B2: DL v3 Model Decomposition (was NNLS-only before)
# ═══════════════════════════════════════════════════════════════════
print("\nGenerating B2: DL v3 model decomposition example...")

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

# DL v3 prediction on this sample
dl_demo = predict_batch(model, demo_samples, device=device)[0]
c_dl = dl_demo["coeffs_pred"]
b_dl = dl_demo["baseline_pred"]
recon_dl = (c_dl[:, None] * R_np).sum(axis=0) + b_dl

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(wn, y_np[m], "k-", lw=0.8, alpha=0.3, label="Observed mixture")

cumulative = np.zeros_like(wn, dtype=float)
for j in range(K):
    if abs(c_dl[j]) > 0.005:
        short = names[j].split(":")[-1][:20]
        weighted = c_dl[j] * R_np[j][m]
        ax.fill_between(wn, cumulative, cumulative + weighted,
                        alpha=0.35, color=COLORS[j],
                        label=f"{short}: pred={c_dl[j]:.3f} (true={c_true[j]:.3f})")
        cumulative = cumulative + weighted

ax.plot(wn, recon_dl[m], "r-", lw=1.5, alpha=0.8, label="DL v3 reconstruction")
ax.plot(wn, b_dl[m], "--", color="gray", lw=1, alpha=0.6, label="DL v3 baseline")
ax.set_title(f"DL v3 Decomposition (MAE={np.mean(np.abs(c_true - c_dl)):.4f})")
ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_ylabel("Intensity")
ax.legend(loc="upper right", fontsize=8)
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(FIGS_PROF / "B2_model_decomposition.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved B2_model_decomposition.png (DL v3)")

# ═══════════════════════════════════════════════════════════════════
# B4: Coefficient comparison (True vs DL v3 vs NNLS)
# ═══════════════════════════════════════════════════════════════════
print("Generating B4: Coefficient comparison (3-way)...")
c_nnls_demo, b_nnls_demo = nnls_decompose(y_np, R_np, s["mask"].numpy(), grid)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(K)
w = 0.25
short_names = [n.split(":")[-1][:15] for n in names[:K]]
ax.bar(x - w, c_true[:K], w, label="Ground Truth", color="#4CAF50", edgecolor="k", lw=0.5)
ax.bar(x, c_dl[:K], w, label="DL v3", color=DL_COLOR, edgecolor="k", lw=0.5)
ax.bar(x + w, c_nnls_demo[:K], w, label="NNLS", color=NNLS_COLOR, edgecolor="k", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(short_names, rotation=30, ha="right")
ax.set_ylabel("Coefficient")
ax.set_title("Coefficient Comparison: True vs DL v3 vs NNLS")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS_PROF / "B4_coefficient_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  Saved B4_coefficient_comparison.png (3-way)")

# ═══════════════════════════════════════════════════════════════════
# C: Demo Unseen Chemicals (DL v3 + NNLS)
# ═══════════════════════════════════════════════════════════════════
print("\nGenerating Part C: Unseen chemicals demo (DL v3 + NNLS)...")

N_DEMO = 6
demo_hold = make_fixed_batch(holdout_pool, n=N_DEMO, seed=2024,
    cfg=SynthConfig(K_min=3, K_max=5, M_min=1, M_max=3, seed=2024))

# Run DL on all demos
dl_demo_results = predict_batch(model, demo_hold, device=device)

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

    # DL v3
    c_dl_d = dl_demo_results[i]["coeffs_pred"]
    b_dl_d = dl_demo_results[i]["baseline_pred"]
    recon_dl_d = (c_dl_d[:, None] * R_d).sum(axis=0) + b_dl_d
    mae_dl_d = np.mean(np.abs(ct - c_dl_d))

    # NNLS
    c_nnls_d, b_nnls_d = nnls_decompose(y_d, R_d, s["mask"].numpy(), grid)
    recon_nnls_d = (c_nnls_d[:, None] * R_d).sum(axis=0) + b_nnls_d
    mae_nnls_d = np.mean(np.abs(ct - c_nnls_d))

    active = [(j, ns[j].split(":")[-1][:20], ct[j]) for j in range(K_total) if ct[j] > 1e-6]
    active_str = ", ".join(f"{name} ({c:.1%})" for _, name, c in active)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2, 1]})

    # Top: spectral decomposition
    ax = axes[0]
    ax.plot(wn_d, y_d[m_d], "k-", lw=1.2, alpha=0.4, label="Observed mixture")
    ax.plot(wn_d, recon_dl_d[m_d], "-", color=DL_COLOR, lw=1.5, alpha=0.8,
            label=f"DL v3 (MAE={mae_dl_d:.4f})")
    ax.plot(wn_d, recon_nnls_d[m_d], "-", color=NNLS_COLOR, lw=1.5, alpha=0.8,
            label=f"NNLS (MAE={mae_nnls_d:.4f})")

    cumulative = np.zeros_like(wn_d, dtype=float)
    for j, name, c_j in active:
        weighted = ct[j] * R_d[j][m_d]
        ax.fill_between(wn_d, cumulative, cumulative + weighted,
                        alpha=0.2, color=COLORS[j % len(COLORS)])
        cumulative = cumulative + weighted

    ax.set_ylabel("Intensity")
    ax.set_title(f"[UNSEEN] Demo {i}: {active_str}")
    ax.legend(fontsize=8, loc="upper right")
    ax.invert_xaxis()

    # Bottom: coefficient bars (True vs DL v3 vs NNLS)
    ax2 = axes[1]
    x = np.arange(K_total)
    w = 0.25
    short_ns = [n.split(":")[-1][:12] for n in ns]
    ax2.bar(x - w, ct, w, label="True", color="#4CAF50", edgecolor="k", lw=0.3)
    ax2.bar(x, c_dl_d, w, label="DL v3", color=DL_COLOR, edgecolor="k", lw=0.3)
    ax2.bar(x + w, c_nnls_d, w, label="NNLS", color=NNLS_COLOR, edgecolor="k", lw=0.3)
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
    fig.savefig(FIGS_PROF / f"C_demo_unseen_{i}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Demo {i}: K={K_d}, DL MAE={mae_dl_d:.4f}, NNLS MAE={mae_nnls_d:.4f}")


# ═══════════════════════════════════════════════════════════════════
# Update summary.txt
# ═══════════════════════════════════════════════════════════════════
summary = f"""Evaluation Summary — DL v3 vs NNLS (with all comparison plots)
==================================================
Checkpoint: best.pt (epoch {info['epoch']})
Holdout chemicals: 64
Test samples: {N_EVAL}

Metric                         DL v3            NNLS
--------------------------------------------------
MAE (mean)                    {dl_maes.mean():.4f}          {nnls_maes.mean():.4f}
MAE (median)                  {np.median(dl_maes):.4f}          {np.median(nnls_maes):.4f}
RMSE                          {dl_rmses.mean():.4f}          {nnls_rmses.mean():.4f}
SAD                           {dl_sads.mean():.4f}          {nnls_sads.mean():.4f}
R²                            {dl_r2:.4f}          {nnls_r2:.4f}
AUC-ROC                       {dl_auc:.4f}          {nnls_auc:.4f}
Spearman                       {dl_spearman:.3f}           {nnls_spearman:.3f}
DL win rate                    {(dl_maes < nnls_maes).mean()*100:.1f}%
"""
(FIGS_EVAL / "summary.txt").write_text(summary)
print(f"\n  Updated summary.txt")

print("\n" + "=" * 65)
print("  ALL PLOTS REGENERATED WITH DL v3 + NNLS")
print("=" * 65)
