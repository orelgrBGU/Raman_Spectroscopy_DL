"""
MCR-ALS baseline for Raman spectral decomposition.

Wraps pymcr to perform alternating least-squares with non-negativity
constraints on both concentrations and spectra.

Unlike NNLS, MCR-ALS does NOT require prior knowledge of the pure spectra —
it can discover them.  Here we seed it with the known references and let it
refine, so that the comparison with NNLS is fair.

Usage:
    coeffs = mcr_decompose(y, R, mask, max_iter=100)
"""

from __future__ import annotations

import numpy as np
from pymcr.mcr import McrAR
from pymcr.constraints import ConstraintNonneg


def mcr_decompose(
    y: np.ndarray,
    R: np.ndarray,
    mask: np.ndarray,
    max_iter: int = 100,
    tol_increase: float = 100.0,
) -> np.ndarray:
    """
    Estimate mixing coefficients via MCR-ALS.

    Parameters
    ----------
    y : (N,) mixture spectrum
    R : (K, N) reference spectra
    mask : (N,) bool — valid wavenumber region
    max_iter : ALS iterations
    tol_increase : allowed relative increase in error before stopping

    Returns
    -------
    coeffs : (K,) estimated coefficients (≥ 0)
    """
    m = mask.astype(bool)
    K = R.shape[0]

    # MCR expects (n_samples, n_features) for D and (n_components, n_features) for ST
    D = y[m].reshape(1, -1).astype(np.float64)
    ST_init = R[:, m].astype(np.float64)

    mcr = McrAR(
        max_iter=max_iter,
        tol_increase=tol_increase,
        c_constraints=[ConstraintNonneg()],
        st_constraints=[ConstraintNonneg()],
    )
    mcr.fit(D, ST=ST_init)

    coeffs = mcr.C_opt_.flatten().astype(np.float32)
    return coeffs


def mcr_batch(
    samples: list[dict],
    grid: np.ndarray,
    max_iter: int = 100,
) -> list[dict]:
    """
    Run MCR-ALS on a list of synthetic samples.

    Returns list of dicts with keys: coeffs_pred, coeffs_true, K, M, snr_db.
    """
    results = []
    for s in samples:
        y = s["y"].numpy()
        R = s["R"].numpy()
        mask = s["mask"].numpy()
        c_true = s["c"].numpy()

        try:
            c_pred = mcr_decompose(y, R, mask, max_iter=max_iter)
        except Exception:
            # MCR can fail to converge on difficult samples
            c_pred = np.zeros_like(c_true)

        results.append({
            "coeffs_pred": c_pred,
            "coeffs_true": c_true,
            "K": s["K"],
            "M": s["M"],
            "snr_db": s["snr_db"],
            "ref_names": s["ref_names"],
        })
    return results
