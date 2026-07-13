#!/usr/bin/env python
"""
Project Summary — Deep Raman Spectral Decomposition
=====================================================

Generates a visual summary of the entire project pipeline for
presentation to advisor. Collects key figures from all stages
and creates an overview panel.

Saves
-----
    outputs/figs/00_summary/
    ├── stage1_data_overview.png       — data sources & chemical coverage
    ├── stage2_synth_example.png       — synthetic mixture construction
    ├── stage3_nnls_baseline.png       — NNLS baseline performance
    ├── stage4_overfit_proof.png       — architecture validation
    ├── stage5_training_curves.png     — training progress
    ├── stage6_final_comparison.png    — model vs NNLS
    └── project_overview.txt           — text summary of all stages

Usage
-----
    cd /gpfs0/bgu-rgilad/users/orelgr/deep2
    python scripts/run_project_summary.py --run_id run01
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")

ROOT = Path("/gpfs0/bgu-rgilad/users/orelgr/deep2")
SUMMARY_DIR = ROOT / "outputs" / "figs" / "00_summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, default="run01")
args = parser.parse_args()

print("=" * 60)
print("Project Summary — Deep Raman Spectral Decomposition")
print("=" * 60)

# ═════════════════════════════════════════════════════════════════════
# Collect key figures from each stage
# ═════════════════════════════════════════════════════════════════════

stage_figs = {
    # Stage 2: Synthetic data
    "stage2_construction": ROOT / "outputs" / "figs" / "02_synth" / "construction_8_samples.png",
    "stage2_snr": ROOT / "outputs" / "figs" / "02_synth" / "snr_comparison.png",
    "stage2_distributions": ROOT / "outputs" / "figs" / "02_synth" / "generator_distributions.png",
    # Stage 3: Baselines
    "stage3_nnls_scatter": ROOT / "outputs" / "figs" / "03_baselines" / "nnls_pred_vs_true.png",
    "stage3_mae_vs_snr": ROOT / "outputs" / "figs" / "03_baselines" / "mae_vs_snr.png",
    "stage3_mae_vs_K": ROOT / "outputs" / "figs" / "03_baselines" / "mae_vs_K.png",
    "stage3_nnls_example": ROOT / "outputs" / "figs" / "03_baselines" / "nnls_example_0.png",
    # Stage 4: Overfit test
    "stage4_loss_curve": ROOT / "outputs" / "figs" / "04_overfit" / "loss_curve.png",
    "stage4_components": ROOT / "outputs" / "figs" / "04_overfit" / "loss_components.png",
    "stage4_scatter": ROOT / "outputs" / "figs" / "04_overfit" / "coeff_scatter.png",
    "stage4_example": ROOT / "outputs" / "figs" / "04_overfit" / "overfit_example_0.png",
    # Stage 5: Training
    "stage5_learning_curves": ROOT / "outputs" / "figs" / "05_training" / args.run_id / "learning_curves.png",
    "stage5_loss_components": ROOT / "outputs" / "figs" / "05_training" / args.run_id / "loss_components.png",
    "stage5_val_vs_train": ROOT / "outputs" / "figs" / "05_training" / args.run_id / "val_vs_train.png",
    # Stage 6: Evaluation
    "stage6_scatter_model": ROOT / "outputs" / "figs" / "06_eval" / "scatter_model.png",
    "stage6_scatter_nnls": ROOT / "outputs" / "figs" / "06_eval" / "scatter_nnls.png",
    "stage6_bar": ROOT / "outputs" / "figs" / "06_eval" / "mae_comparison_bar.png",
    "stage6_improvement": ROOT / "outputs" / "figs" / "06_eval" / "improvement_histogram.png",
    "stage6_vs_snr": ROOT / "outputs" / "figs" / "06_eval" / "mae_vs_snr.png",
    "stage6_vs_K": ROOT / "outputs" / "figs" / "06_eval" / "mae_vs_K.png",
    "stage6_good": ROOT / "outputs" / "figs" / "06_eval" / "example_good_0.png",
    "stage6_hard": ROOT / "outputs" / "figs" / "06_eval" / "example_hard_0.png",
}

print("\nCollecting figures from all stages...")
found, missing = 0, 0
for name, src in stage_figs.items():
    if src.exists():
        dst = SUMMARY_DIR / f"{name}.png"
        shutil.copy2(src, dst)
        found += 1
    else:
        print(f"  [SKIP] {name}: {src.name} not found")
        missing += 1

print(f"  Collected: {found}/{found+missing} figures")

# ═════════════════════════════════════════════════════════════════════
# Read summaries from each stage
# ═════════════════════════════════════════════════════════════════════
summaries = {}
for stage, path in [
    ("04_overfit", ROOT / "outputs" / "figs" / "04_overfit" / "summary.txt"),
    ("05_training", ROOT / "outputs" / "figs" / "05_training" / args.run_id / "summary.txt"),
    ("06_eval", ROOT / "outputs" / "figs" / "06_eval" / "summary.txt"),
]:
    if path.exists():
        summaries[stage] = path.read_text()

# ═════════════════════════════════════════════════════════════════════
# Generate project overview text
# ═════════════════════════════════════════════════════════════════════
overview = """
================================================================================
      Deep Learning for Raman Spectral Decomposition — Project Overview
================================================================================

GOAL
----
Build a generic deep learning model that decomposes an unknown Raman spectrum
into a linear combination of known reference spectra, estimating both mixing
coefficients and a polynomial baseline.

Given:  y = sum(c_i * R_i) + baseline + noise
Find:   c_i (coefficients) and baseline

PIPELINE
--------

Stage 1 — Data Preparation
    - Collected pure Raman spectra from RRUFF and SDBS databases
    - 320 unique chemicals, 4,057 spectra
    - Unified wavenumber grid: 400-3,400 cm^-1 (3,001 points)
    - Chemical-level holdout split: 256 train / 64 holdout

Stage 2 — Synthetic Mixture Generator
    - On-the-fly infinite dataset: no two training samples are identical
    - K=1-8 components (Dirichlet coefficients) + M=0-5 distractors
    - Realistic corruptions: polynomial baseline, Gaussian+Poisson noise
      (SNR 10-60 dB), peak shifts, broadening, intensity scaling
    - Key figures: construction_8_samples, snr_comparison, distributions

Stage 3 — Classical Baselines
    - NNLS (Non-Negative Least Squares) with polynomial baseline:
      Augments reference matrix with +/- polynomial columns, solved via
      scipy.optimize.nnls. MAE ≈ 0.088, Spearman ≈ 0.55.
    - MCR-ALS: Blind source separation — wrong paradigm for informed
      decomposition (MAE = 2.65). Documented as negative finding.

Stage 4 — Model Architecture & Overfit Test
    - Architecture: Shared Conv1D Encoder + Transformer → embeddings
      Cross-Attention (unknown queries references) → MLP scorer → coefficients
      Separate MLP → polynomial baseline
    - 2.3M parameters (d_model=256, 2 Transformer layers, 4 heads)
    - Key design decisions:
      * Removed softplus activation (caused gradient collapse, loss stuck at 0.16)
      * Decoupled reconstruction loss from baseline (prevents baseline absorbing signal)
      * Soft non-negativity penalty instead of hard constraint
    - Overfit test: 16 samples → loss 0.0001 in 5000 steps ✓

Stage 5 — Training
    - 100 epochs × 500 steps × batch_size=64 = 3.2M unique mixtures
    - Adam optimizer + cosine LR schedule (1e-3 → 1e-5)
    - Mixed precision (AMP) on NVIDIA L40S GPU
    - Auto-resume from checkpoint (survives preemption)

Stage 6 — Evaluation
    - 500 synthetic mixtures from holdout chemicals (unseen)
    - Head-to-head comparison with NNLS baseline
    - Metrics: MAE, Spearman correlation, reconstruction MSE
    - Analysis by SNR level and mixture complexity (K)

"""

# Append individual stage summaries
for stage, text in summaries.items():
    overview += f"\n{'='*60}\n"
    overview += text

overview += """
================================================================================
  Code structure:
    deep2/
    ├── data/processed/          — preprocessed spectra stacks
    ├── configs/base.yaml        — training hyperparameters
    ├── src/
    │   ├── data/synth_mixtures.py  — synthetic mixture generator
    │   ├── model/
    │   │   ├── encoder.py          — Conv1D + Transformer encoder
    │   │   ├── decompose.py        — full DecomposeModel
    │   │   └── loss.py             — combined loss function
    │   ├── baselines/nnls.py       — NNLS baseline
    │   ├── train.py                — training script
    │   └── eval.py                 — evaluation utilities
    ├── scripts/
    │   ├── run_notebook02.py       — synthetic data sanity check
    │   ├── run_notebook03.py       — classical baselines
    │   ├── run_notebook04.py       — overfit test documentation
    │   ├── run_notebook05.py       — training monitoring
    │   ├── run_notebook06.py       — model evaluation
    │   └── run_project_summary.py  — this file
    ├── checkpoints/                — saved model weights
    ├── logs/                       — training logs
    └── outputs/figs/               — all generated figures
================================================================================
"""

with open(SUMMARY_DIR / "project_overview.txt", "w") as f:
    f.write(overview)

print(overview)
print(f"\nAll summary figures collected in: {SUMMARY_DIR}/")
print("DONE")
