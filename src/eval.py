"""
Evaluation utilities for the spectral decomposition model.

Provides functions to:
  - Load a trained model from checkpoint
  - Run inference on a batch of samples
  - Compute per-sample and aggregate metrics
  - Compare model predictions with NNLS baseline
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.model.decompose import DecomposeModel, collate_decompose
from src.baselines.nnls import nnls_decompose


def load_model_from_checkpoint(
    ckpt_path: str | Path,
    device: str = "cuda",
) -> tuple[DecomposeModel, dict]:
    """Load a trained DecomposeModel from a checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    mcfg = cfg["model"]

    model = DecomposeModel(
        d_model=mcfg["d_model"],
        n_transformer_layers=mcfg["n_transformer_layers"],
        n_heads=mcfg["n_heads"],
        dropout=0.0,  # no dropout at inference
        poly_order=mcfg["poly_order"],
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    info = {
        "epoch": ckpt["epoch"],
        "step": ckpt["step"],
        "best_metric": ckpt["best_metric"],
    }
    return model, info


@torch.no_grad()
def predict_batch(
    model: DecomposeModel,
    samples: list[dict],
    device: str = "cuda",
) -> list[dict]:
    """Run model inference on a list of synthetic samples."""
    batch = collate_decompose(samples)
    y = batch["y"].to(device)
    R = batch["R"].to(device)
    ref_mask = batch["ref_mask"].to(device)

    c_pred, b_pred = model(y, R, ref_mask)
    c_pred = c_pred.cpu().numpy()
    b_pred = b_pred.cpu().numpy()

    results = []
    for i, s in enumerate(samples):
        K_total = len(s["c"])
        results.append({
            "coeffs_pred": c_pred[i, :K_total],
            "baseline_pred": b_pred[i],
            "coeffs_true": s["c"].numpy(),
            "baseline_true": s["baseline"].numpy(),
            "y": s["y"].numpy(),
            "R": s["R"].numpy(),
            "mask": s["mask"].numpy(),
            "K": s["K"],
            "M": s["M"],
            "snr_db": s["snr_db"],
            "ref_names": s["ref_names"],
        })
    return results


def compute_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics from a list of result dicts."""
    maes, recon_mses, spearman_corrs = [], [], []

    for r in results:
        c_true = r["coeffs_true"]
        c_pred = r["coeffs_pred"]
        maes.append(np.mean(np.abs(c_true - c_pred)))

        # Reconstruction error
        y = r["y"]
        R = r["R"]
        mask = r["mask"].astype(bool)
        recon = (c_pred[:, None] * R).sum(axis=0) + r["baseline_pred"]
        mse = np.mean((y[mask] - recon[mask]) ** 2)
        recon_mses.append(mse)

        # Spearman rank correlation
        from scipy.stats import spearmanr
        if np.std(c_true) > 1e-9 and np.std(c_pred) > 1e-9:
            rho, _ = spearmanr(c_true, c_pred)
            spearman_corrs.append(rho)

    return {
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "mae_median": float(np.median(maes)),
        "recon_mse_mean": float(np.mean(recon_mses)),
        "spearman_mean": float(np.mean(spearman_corrs)) if spearman_corrs else float("nan"),
        "n_samples": len(results),
        "per_sample_mae": np.array(maes),
    }


def nnls_predict_batch(
    samples: list[dict],
    grid: np.ndarray,
    poly_order: int = 5,
) -> list[dict]:
    """Run NNLS baseline on a list of samples (same output format as predict_batch)."""
    results = []
    for s in samples:
        y = s["y"].numpy()
        R = s["R"].numpy()
        mask = s["mask"].numpy()
        c_pred, b_pred = nnls_decompose(y, R, mask, grid, poly_order)

        results.append({
            "coeffs_pred": c_pred,
            "baseline_pred": b_pred,
            "coeffs_true": s["c"].numpy(),
            "baseline_true": s["baseline"].numpy(),
            "y": y,
            "R": R,
            "mask": mask,
            "K": s["K"],
            "M": s["M"],
            "snr_db": s["snr_db"],
            "ref_names": s["ref_names"],
        })
    return results
