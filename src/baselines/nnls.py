"""
NNLS baseline for Raman spectral decomposition.

Augments the reference matrix R with polynomial columns so that NNLS
simultaneously estimates non-negative mixture coefficients **and** a
smooth baseline.  The polynomial is constructed on a normalised [-1, 1]
axis to keep condition numbers sane.

Usage:
    coeffs, baseline = nnls_decompose(y, R, mask, grid, poly_order=5)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def _poly_basis(grid: np.ndarray, order: int) -> np.ndarray:
    """Polynomial basis columns (order+1 columns) on [-1, 1]."""
    x = np.linspace(-1.0, 1.0, grid.size, dtype=np.float64)
    return np.stack([x ** k for k in range(order + 1)], axis=1)


def nnls_decompose(
    y: np.ndarray,
    R: np.ndarray,
    mask: np.ndarray,
    grid: np.ndarray,
    poly_order: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve  y ≈ R^T c + P β  subject to c ≥ 0, β ≥ 0  via NNLS.

    To allow negative polynomial coefficients, each basis column p_k is
    duplicated as (p_k, -p_k) so NNLS can pick either sign.

    Parameters
    ----------
    y : (N,) mixture spectrum
    R : (K, N) reference spectra (rows = components)
    mask : (N,) bool — valid wavenumber region
    grid : (N,) wavenumber axis (only used for polynomial basis shape)
    poly_order : max polynomial order for baseline

    Returns
    -------
    coeffs : (K,) estimated mixing coefficients (≥ 0)
    baseline : (N,) estimated baseline on full grid (zeros where ~mask)
    """
    m = mask.astype(bool)
    K, N = R.shape

    # Polynomial basis — duplicate for +/- so NNLS can represent negative baseline
    P = _poly_basis(grid, poly_order).astype(np.float64)  # (N, order+1)
    P_ext = np.hstack([P, -P])  # (N, 2*(order+1))

    # Build augmented design matrix: [R^T | P_ext] restricted to valid points
    A = np.hstack([R[:, m].T, P_ext[m]])  # (sum(m), K + 2*(order+1))
    b = y[m].astype(np.float64)

    x, _ = nnls(A, b)

    coeffs = x[:K].astype(np.float32)

    # Reconstruct baseline on full grid
    n_poly = poly_order + 1
    beta_pos = x[K : K + n_poly]
    beta_neg = x[K + n_poly : K + 2 * n_poly]
    beta = beta_pos - beta_neg
    baseline = (P @ beta).astype(np.float32)
    baseline[~m] = 0.0

    return coeffs, baseline


def nnls_batch(
    samples: list[dict],
    grid: np.ndarray,
    poly_order: int = 5,
) -> list[dict]:
    """
    Run NNLS on a list of synthetic samples (from make_fixed_batch).

    Returns list of dicts with keys: coeffs_pred, baseline_pred, coeffs_true,
    baseline_true, K, M, snr_db, ref_names.
    """
    results = []
    for s in samples:
        y = s["y"].numpy()
        R = s["R"].numpy()
        mask = s["mask"].numpy()
        c_true = s["c"].numpy()
        b_true = s["baseline"].numpy()

        c_pred, b_pred = nnls_decompose(y, R, mask, grid, poly_order)

        results.append({
            "coeffs_pred": c_pred,
            "baseline_pred": b_pred,
            "coeffs_true": c_true,
            "baseline_true": b_true,
            "K": s["K"],
            "M": s["M"],
            "snr_db": s["snr_db"],
            "ref_names": s["ref_names"],
        })
    return results
