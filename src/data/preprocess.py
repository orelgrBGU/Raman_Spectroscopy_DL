"""
Preprocess raw-standardized spectra onto a unified cm^-1 grid + L2 normalize.

Grid: 400..3400 cm^-1 at step 1 (3001 points) — per the project spec.
Per-spectrum mask marks which grid points fall inside the source's native range.

Output: data/processed/{source}/*.npz with keys:
    intensity : float32 (N,) — L2-normalized on valid grid points
    mask      : bool    (N,) — True where the native range covers this cm^-1
Plus a per-source stack for efficient batch loading:
    data/processed/{source}_stack.npz with keys:
        intensity : float32 (M, N)
        mask      : bool    (M, N)
        spectrum_id : object array (M,)

Grid is stored once in data/processed/grid.npz.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

PROJECT_ROOT = Path("/gpfs0/bgu-rgilad/users/orelgr/deep2")
RAW_STD_DIR = PROJECT_ROOT / "data" / "raw_std"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.csv"

GRID_MIN = 400.0
GRID_MAX = 3400.0
GRID_STEP = 1.0
GRID = np.arange(GRID_MIN, GRID_MAX + GRID_STEP / 2, GRID_STEP, dtype=np.float32)


def _l2_normalize(intensity: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """L2-normalize on masked points; out-of-range points stay zero."""
    out = np.zeros_like(intensity)
    if mask.any():
        norm = np.sqrt(float(np.sum(intensity[mask] ** 2)))
        if norm > 0:
            out[mask] = intensity[mask] / norm
    return out


def _process_one(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a raw_std spectrum, interpolate onto GRID, L2-normalize. Return (intensity, mask)."""
    with np.load(npz_path, allow_pickle=True) as z:
        wn = z["wavenumber"].astype(np.float64)
        it = z["intensity"].astype(np.float64)

    it = np.where(np.isfinite(it), it, 0.0)
    order = np.argsort(wn)
    wn = wn[order]
    it = it[order]

    keep = np.concatenate([[True], np.diff(wn) > 0])
    wn = wn[keep]
    it = it[keep]

    mask = (GRID >= wn[0]) & (GRID <= wn[-1])
    interpolated = np.zeros(GRID.shape, dtype=np.float32)
    if mask.any():
        f = interp1d(wn, it, kind="linear", bounds_error=False, fill_value=0.0, assume_sorted=True)
        interpolated[mask] = f(GRID[mask]).astype(np.float32)

    normalized = _l2_normalize(interpolated, mask)
    return normalized, mask


def preprocess_all(only: list[str] | None = None) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(PROCESSED_DIR / "grid.npz", wavenumber=GRID)

    manifest = pd.read_csv(MANIFEST_PATH)
    sources = sorted(manifest["source"].unique()) if only is None else only

    for source in sources:
        sub = manifest[manifest["source"] == source]
        print(f"[preprocess] {source}: {len(sub)} spectra")

        stack_int = np.zeros((len(sub), GRID.size), dtype=np.float32)
        stack_mask = np.zeros((len(sub), GRID.size), dtype=bool)
        spectrum_ids: list[str] = []

        for i, (_, row) in enumerate(sub.iterrows()):
            npz_path = PROJECT_ROOT / row["npz_path"]
            interp, mask = _process_one(npz_path)
            stack_int[i] = interp
            stack_mask[i] = mask
            spectrum_ids.append(row["spectrum_id"])
            if (i + 1) % 500 == 0:
                print(f"[preprocess] {source}: {i + 1}/{len(sub)}")

        stack_path = PROCESSED_DIR / f"{source}_stack.npz"
        np.savez_compressed(
            stack_path,
            intensity=stack_int,
            mask=stack_mask,
            spectrum_id=np.array(spectrum_ids, dtype=object),
        )
        print(f"[preprocess] wrote {stack_path} ({stack_int.shape})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None)
    args = ap.parse_args()
    preprocess_all(only=args.only)
