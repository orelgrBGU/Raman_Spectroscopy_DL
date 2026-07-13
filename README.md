# Raman Spectral Decomposition via Deep Learning

A deep learning framework for decomposing Raman spectra of unknown mixtures into their pure chemical components. Given a mixture spectrum and a set of reference (pure) spectra, the model predicts the concentration coefficient of each component — enabling **generic, reference-swappable** spectral unmixing without retraining.

> **Thesis project** — Ben-Gurion University of the Negev, Department of Industrial Engineering and Management

---

## Table of Contents

- [Motivation](#motivation)
- [Method Overview](#method-overview)
- [Architecture (v3)](#architecture-v3)
- [Pipeline: From Data to Evaluation](#pipeline-from-data-to-evaluation)
  - [Stage 1: Data Collection & Preprocessing](#stage-1-data-collection--preprocessing)
  - [Stage 2: Synthetic Mixture Generation](#stage-2-synthetic-mixture-generation)
  - [Stage 3: Classical Baseline (NNLS)](#stage-3-classical-baseline-nnls)
  - [Stage 4: Overfit Sanity Check](#stage-4-overfit-sanity-check)
  - [Stage 5: Full Training (v3)](#stage-5-full-training-v3)
- [Results: DL v3 vs NNLS](#results-dl-v3-vs-nnls)
- [NNLS Baseline Results](#nnls-baseline-results)
- [Real Data: Sugar Mixture Test](#real-data-sugar-mixture-test)
- [Demo: Unseen Chemicals](#demo-unseen-chemicals)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Version History](#version-history)
- [References](#references)

---

## Motivation

Raman spectroscopy is a powerful non-destructive technique for identifying chemical compositions. In real-world applications (food quality, pharmaceuticals, forensics), samples are typically **mixtures** of multiple compounds. The challenge: given an observed mixture spectrum and a library of pure reference spectra, determine **which** components are present and in **what proportions**.

Traditional methods like NNLS (Non-Negative Least Squares) solve this as a linear algebra problem, but struggle with:
- Fluorescent baselines that distort the signal
- Noisy spectra with low SNR
- Overlapping spectral features between components

Our approach uses a **deep learning model** that learns the "art of decomposition" from millions of synthetically generated mixtures, then generalizes to **unseen chemicals** at inference time.

---

## Method Overview

The key insight: rather than training on specific chemicals, we train the model to understand **how spectra combine**. The model receives:

1. **Unknown mixture spectrum** `y` (corrupted with noise + baseline)
2. **Reference spectra** `R = [r_1, r_2, ..., r_K]` (pure components, possibly including distractors)

And predicts:
- **Coefficients** `c = [c_1, c_2, ..., c_K]` — the abundance of each reference in the mixture (independent non-negative via softplus)
- **Baseline** `b` — the fluorescent background (non-negative via soft penalty)

The model never memorizes specific chemicals. At inference, you can swap in completely new reference spectra and the model still works.

---

## Architecture (v3)

![Architecture diagram](outputs/figs/architecture_diagram.png)

The architecture incorporates insights from the literature (EGU-Net, PNAS 2024 Georgiev, RamanFormer) with key improvements across three versions:

**Mixture-Only Encoder** (references are NOT encoded through the deep network):
- 3 Conv1D blocks (1→32→64→128 channels, kernel=7, MaxPool=4)
- 1 Transformer encoder layer (4 heads, d=128)
- Global average pooling → 128-dim embedding `z_u`

**Reference Projection** (lightweight, no deep encoder):
- `Linear(3001, 128) + LayerNorm` → `z_r` (K × 128)
- This avoids encoder collapse (v1 had cosine_sim=0.999 between all ref embeddings)

**Cross-Attention:** The unknown spectrum queries the references to build a context-aware representation.

**Spectral Similarity Features:** Direct signal-space features (cosine similarity, dot product, L2 distance) bypass the encoder to provide explicit matching signals.

**Scorer:** MLP that takes `[z_r_i, context, spectral_features]` (dim = 2d+3) → logit per reference.

**Softplus Output (v3):** Each coefficient is independently non-negative via `softplus(logit)`. Unlike v2's softmax (which forced sum-to-one across all references including distractors), softplus allows each coefficient to be pushed to true zero independently — critical for **distractor suppression**.

**Baseline Head:** MLP on `z_u` → polynomial coefficients (order 5) → smooth baseline. Non-negativity enforced via soft penalty in the loss (v3), since hard ReLU clamping kills gradients for small baselines.

---

## Pipeline: From Data to Evaluation

### Stage 1: Data Collection & Preprocessing

We collected **115 pure Raman spectra** of food-relevant organic compounds from multiple open spectral databases:

- **RamanBioLib** — large library of biological/organic Raman spectra
- **SDBS (AIST)** — organic molecule spectral database
- **Olive oil dataset** — food-specific Raman spectra
- **Sugar mixtures** — controlled mixture experiments

All spectra are interpolated to a unified grid (400–3400 cm⁻¹, 1 cm⁻¹ resolution) and L2-normalized.

![Data exploration: spectral coverage per source](outputs/figs/01_exploration/coverage_per_source.png)

![Pure spectrum examples from each source](outputs/figs/01_exploration/pure_examples_per_source.png)

---

### Stage 2: Synthetic Mixture Generation

Training data is generated **on-the-fly** — each batch contains fresh, never-before-seen mixtures:

1. **Sample K** components (K ∈ [1..8]) from the chemical pool
2. **Draw coefficients** from Dirichlet(α), where α ∈ [0.3, 2.0]
3. **Linear superposition**: `y = Σ c_i · r_i`
4. **Add distractors**: M ∈ [0..5] extra references with true coefficient = 0
5. **Corrupt** with realistic noise:
   - Positive fluorescent baseline (exponential-quadratic bumps, always ≥ 0)
   - Gaussian + Poisson noise (SNR 10–60 dB)
   - Peak shift (±1–3 cm⁻¹)
   - Gaussian broadening (σ ∈ [0.5, 2] cm⁻¹)

The forward process — from pure spectra to corrupted mixture — is illustrated below:

![Step 1: Pure reference spectra of individual components](outputs/figs/07_professional/A1_pure_reference_0.png)

![Step 2: Each component weighted by its Dirichlet coefficient](outputs/figs/07_professional/A2_weighted_components.png)

![Step 3: Superposition builds the mixture step by step](outputs/figs/07_professional/A3_superposition_buildup.png)

![Step 4: Final mixture with baseline and noise corruption](outputs/figs/07_professional/A4_final_mixture.png)

**Generator statistics** — distribution of K (number of components), coefficient magnitudes, and SNR levels across generated batches:

![Generator distributions](outputs/figs/02_synth/generator_distributions.png)

---

### Stage 3: Classical Baseline (NNLS)

Before training the DL model, we established a classical baseline using **Non-Negative Least Squares** (NNLS) with polynomial baseline columns:

```
minimize ||y - [R | P] · x||²   subject to x ≥ 0
```

where `P` contains polynomial basis functions (order 5) to absorb the baseline.

NNLS achieves strong results on synthetic mixtures, providing the performance floor our model must beat:

![NNLS decomposition example](outputs/figs/03_baselines/nnls_example_0.png)

![NNLS predicted vs true coefficients](outputs/figs/03_baselines/nnls_pred_vs_true.png)

![NNLS MAE vs number of components](outputs/figs/03_baselines/mae_vs_K.png)

---

### Stage 4: Overfit Sanity Check

Before full training, we verified the architecture can learn by **overfitting to 16 samples**. This confirms the wiring (encoder → cross-attention → scorer → softplus → loss) is correct.

Result (v3): **coefficient MAE = 0.012** after 5000 steps, baseline non-negative (0%), distractor coefficients < 0.001.

![Overfit loss curve](outputs/figs/04_overfit/loss_curve.png)

![Overfit decomposition example](outputs/figs/04_overfit/overfit_example_0.png)

---

### Stage 5: Full Training (v3)

Training configuration (v3):
- **Optimizer:** Adam (lr=1e-3, cosine decay to 1e-5)
- **Batch size:** 64, synthetic data generated on-the-fly
- **Duration:** 100 epochs × 500 steps/epoch = 50,000 gradient steps
- **Mixed precision** (AMP) for efficiency
- **Holdout:** 20% of chemicals (64 compounds) reserved for validation
- **Infrastructure:** Run:AI GPU cluster with checkpoint auto-resume

**Loss function (v3):**
```
L = λ_c    · MAE(c_pred, c_true)           coefficient accuracy        (λ=1.0)
  + λ_r    · MSE(y - b_pred, Σ c·R)        reconstruction              (λ=50.0)
  + λ_sad  · SAD(y - b_pred, Σ c·R)        spectral angle distance     (λ=1.0)
  + λ_b    · MAE(b_pred, b_true)            baseline estimation         (λ=0.5)
  + λ_l1   · ||c_pred||₁                    sparsity (distractor→0)    (λ=0.1)
  + λ_bneg · mean(relu(-b_pred))            baseline non-negativity    (λ=10.0)
```

**Key v3 changes from v2:**
- **softmax → softplus:** Each coefficient is independent; distractors can be driven to true zero without affecting in-mixture coefficients
- **λ_r: 1.0 → 50.0:** Reconstruction loss was negligible in v2; now it provides meaningful physics-based regularization
- **λ_l1: 0.01 → 0.1:** Stronger sparsity needed for unbounded softplus coefficients
- **Baseline non-negativity:** Soft penalty replaces hard ReLU (which killed gradients for small baselines)

![Training curves](outputs/figs/06_eval_v3/training_curves.png)

---

## Results: DL v3 vs NNLS

### Summary Table (500 holdout samples — unseen chemicals)

| Metric | DL v2 (softmax) | **DL v3 (softplus)** | NNLS |
|--------|:---:|:---:|:---:|
| MAE (mean) ↓ | 0.1511 | **0.1234** | **0.0937** |
| MAE (median) ↓ | 0.1218 | **0.0983** | **0.0634** |
| Spearman ↑ | 0.279 | **0.513** | 0.483 |
| DL win rate | 30% | **40.6%** | — |

v3 achieves the **highest Spearman rank correlation** (0.513 > NNLS 0.483), meaning it predicts the **relative ordering** of component contributions better than NNLS. NNLS still wins on absolute MAE due to its direct per-sample optimization.

### Coefficient Scatter: Predicted vs True

![Scatter comparison](outputs/figs/06_eval_v3/scatter_comparison.png)

### Model Comparison

![MAE comparison](outputs/figs/06_eval_v3/mae_comparison_bar.png)

![Spearman comparison](outputs/figs/06_eval_v3/spearman_comparison_bar.png)

### Robustness: MAE vs Number of Components

![MAE vs K](outputs/figs/06_eval_v3/mae_vs_K.png)

### Per-Sample Improvement Distribution

![Improvement histogram](outputs/figs/06_eval_v3/improvement_histogram.png)

### Example Decompositions

Cases where DL v3 outperforms NNLS:

![DL wins example 0](outputs/figs/06_eval_v3/example_dl_wins_0.png)

![DL wins example 1](outputs/figs/06_eval_v3/example_dl_wins_1.png)

Cases where NNLS outperforms DL v3:

![NNLS wins example 0](outputs/figs/06_eval_v3/example_nnls_wins_0.png)

![NNLS wins example 1](outputs/figs/06_eval_v3/example_nnls_wins_1.png)

---

## NNLS Baseline Results

### The Decomposition Challenge

Given a corrupted mixture spectrum, recover each component's contribution:

![The challenge: only the mixture is observed](outputs/figs/07_professional/B1_challenge.png)

### NNLS Decomposition

NNLS decomposes the mixture into its components with high accuracy:

![NNLS decomposition with component stacking](outputs/figs/07_professional/B3_nnls_decomposition.png)

![Coefficient comparison: True vs NNLS](outputs/figs/07_professional/B4_coefficient_comparison.png)

### Comprehensive Metrics (500 holdout samples)

![NNLS metrics summary](outputs/figs/07_professional/D1_metrics_comparison.png)

| Metric | NNLS |
|--------|------|
| MAE ↓ | 0.093 |
| RMSE ↓ | 0.139 |
| SAD ↓ | 0.555 |
| R² ↑ | 0.240 |
| AUC-ROC ↑ | 0.737 |

### ROC Curve — Component Detection

![ROC curve for component detection](outputs/figs/07_professional/D2_roc_curve.png)

### Predicted vs True Coefficients

![Scatter plot: NNLS predicted vs true](outputs/figs/07_professional/D3_scatter_comparison.png)

### Robustness Analysis

![MAE vs SNR: noise robustness](outputs/figs/07_professional/D4_mae_vs_snr.png)

![MAE vs K: complexity scaling](outputs/figs/07_professional/D5_mae_vs_K_boxplot.png)

### MAE Distribution

![NNLS MAE distribution](outputs/figs/07_professional/D6_improvement_histogram.png)

---

## Real Data: Sugar Mixture Test

We tested the model on **9,600 real sugar mixture spectra** — physical mixtures of glucose, fructose, sucrose, maltose, and water measured in a lab. The model was given the 5 pure reference spectra and asked to decompose each mixture.

This test reveals the **domain gap** between synthetic training data and real measurements:

| Metric | DL v3 | NNLS |
|--------|:---:|:---:|
| Reconstruction RMSE | 0.0073 | **0.0003** |
| Coefficient sum | 2.56 | 1.01 |
| Baseline non-negative | 67.9% | 100% |

NNLS performs well because it directly solves the linear system per-sample. The DL model struggles because real sugar spectra are highly similar to each other, and the synthetic training distribution doesn't perfectly match real measurement conditions.

Example coefficient comparisons on real sugar mixtures:

![Sugar example 0](outputs/figs/08_real_data_test_v3/example_00_coefficients.png)

![Sugar example 2](outputs/figs/08_real_data_test_v3/example_02_coefficients.png)

**Next steps for real-data improvement:** fine-tuning on labeled real mixtures, improved augmentation to close the domain gap, and hybrid NNLS-DL approaches.

---

## Demo: Unseen Chemicals

The NNLS baseline's performance on **holdout chemicals** — compounds that were completely excluded from the training chemical pool:

![Demo: unseen chemical mixture 1](outputs/figs/07_professional/C_demo_unseen_0.png)

![Demo: unseen chemical mixture 2](outputs/figs/07_professional/C_demo_unseen_1.png)

![Demo: unseen chemical mixture 3](outputs/figs/07_professional/C_demo_unseen_2.png)

![Demo: unseen chemical mixture 4](outputs/figs/07_professional/C_demo_unseen_3.png)

![Demo: unseen chemical mixture 5](outputs/figs/07_professional/C_demo_unseen_4.png)

![Demo: unseen chemical mixture 6](outputs/figs/07_professional/C_demo_unseen_5.png)

---

## Project Structure

```
deep2/
├── src/
│   ├── data/
│   │   ├── ingest.py             # Download & parse spectral databases
│   │   ├── preprocess.py         # Interpolation to unified grid + L2 normalization
│   │   └── synth_mixtures.py     # On-the-fly synthetic mixture generator
│   ├── baselines/
│   │   ├── nnls.py               # NNLS with polynomial baseline columns
│   │   └── mcr_als.py            # MCR-ALS wrapper
│   ├── model/
│   │   ├── encoder.py            # Conv1D + Transformer (mixture-only encoder)
│   │   ├── decompose.py          # Cross-attention + scorer + softplus + baseline head
│   │   └── loss.py               # Multi-component loss (MAE + recon + SAD + baseline + L1 + bneg)
│   ├── train.py                  # Full training script (CLI, auto-resume, AMP)
│   └── eval.py                   # Evaluation utilities & metrics
├── configs/
│   └── base.yaml                 # Training hyperparameters (v3)
├── scripts/
│   ├── run_notebook02.py         # Synthetic mixture visualization
│   ├── run_notebook03.py         # Classical baseline evaluation
│   ├── run_overfit_test.py       # Overfit test execution
│   ├── create_architecture_diagram.py  # Architecture diagram generator
│   ├── regenerate_broken_plots.py      # Plot regeneration with fixed data
│   └── submit_train.sh           # Run:AI GPU submission script
├── data/
│   ├── raw/                      # Original spectra from databases
│   ├── processed/                # Preprocessed .npz files on unified grid
│   └── manifest.csv              # Chemical index with metadata
├── outputs/
│   └── figs/                     # All generated figures
│       ├── 01_exploration/       # Data exploration
│       ├── 02_synth/             # Synthetic mixture construction
│       ├── 03_baselines/         # NNLS baseline results
│       ├── 04_overfit/           # Overfit sanity check
│       ├── 06_eval_v3/          # DL v3 evaluation plots & comparison
│       ├── 07_professional/      # Publication-quality figures
│       └── 08_real_data_test_v3/ # Real sugar mixture test (v3)
├── checkpoints/                  # Model weights (not in git)
└── requirements.txt
```

---

## Usage

### Prerequisites

```bash
pip install torch numpy scipy pandas matplotlib seaborn pyyaml tqdm tensorboard pymcr scikit-learn
```

### Training

```bash
# Overfit sanity check (CPU, ~2 min)
python scripts/run_overfit_test.py

# Full training (GPU recommended)
python -m src.train --config configs/base.yaml --run_id v3_run01 --max_epochs 100

# Resume interrupted training
python -m src.train --config configs/base.yaml --run_id v3_run01  # auto-detects checkpoint
```

### Evaluation

```bash
# Generate NNLS baseline plots
python scripts/regenerate_broken_plots.py

# Architecture diagram
python scripts/create_architecture_diagram.py
```

---

## Version History

| Version | Key Changes | Synthetic MAE | Spearman |
|---------|------------|:---:|:---:|
| **v1** | Shared encoder, raw linear output, soft non-negativity penalty | — | — |
| **v2** | Mixture-only encoder, lightweight ref projection, softmax, SAD loss | 0.1511 | 0.279 |
| **v3** | softplus coefficients, non-negative baseline penalty, rebalanced loss | **0.1234** | **0.513** |

**v1 → v2:** Fixed encoder collapse (cosine sim 0.999 between all references). Moved from shared deep encoder to mixture-only encoder + lightweight reference projection.

**v2 → v3:** Fixed distractor suppression. Softmax forced sum-to-one across all references (including distractors), preventing true zeros. Softplus allows independent non-negative coefficients. Boosted reconstruction loss weight (1→50) and sparsity (0.01→0.1). Added soft baseline non-negativity penalty.

---

## References

- **EGU-Net:** Qi et al., *"EGU-Net: Endmember Guided Unmixing Network for Hyperspectral Images"* — SAD and RMSE abundance metrics
- **Georgiev et al. (PNAS 2024):** Physics-constrained autoencoder for spectral unmixing — softmax output, linear mixing decoder
- **RamanFormer:** Transformer-based Raman spectral analysis, MAE/RMSE protocol
- **NNLS:** Lawson & Hanson, *Solving Least Squares Problems* (1995)
- **MCR-ALS:** Tauler, *"Multivariate Curve Resolution"* (1995)

---

## License

This project is part of a thesis at Ben-Gurion University of the Negev. For academic use.
