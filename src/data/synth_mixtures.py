"""
On-the-fly generator of synthetic Raman mixtures with ground truth.

A sample = (mixture y, reference matrix R, true coefficients c, true baseline b).

Design (per project spec, Stage 2):
- K in [1..8] components drawn per sample.
- Dirichlet coefficients (positive, sum to 1).
- M in [0..5] distractor references appended with true coeff = 0.
- Realistic corruptions: polynomial baseline, Gaussian + Poisson noise (SNR 10-60 dB),
  peak shift +/-1-3 cm^-1, jitter (Gaussian broadening), global intensity scale.
- Holdout split: 20% of chemicals reserved for val/test (never seen in train).
- All spectra live on the unified 400-3400 cm^-1 grid (3001 points).

Returned tensors have shape:
    y:   (N,)         — mixture on the unified grid, corrupted
    R:   (K+M, N)     — reference stack (in-mixture + distractors, shuffled)
    c:   (K+M,)       — Dirichlet coeffs for in-mixture, zeros for distractors
    b:   (N,)         — clean polynomial baseline (label for baseline head)
    mask: (N,)        — union of native masks of the sampled references
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

PROJECT_ROOT = Path("/gpfs0/bgu-rgilad/users/orelgr/deep2")
PROC = PROJECT_ROOT / "data" / "processed"
MANIFEST = PROJECT_ROOT / "data" / "manifest.csv"

# ---------------------------------------------------------------------------
# Chemical pool: load pure spectra from all sources, index by canonical chemical
# ---------------------------------------------------------------------------


@dataclass
class ChemicalPool:
    """Index of pure spectra. `by_chem[chemical]` -> list of (intensity, mask) tuples."""
    grid: np.ndarray
    by_chem: dict[str, list[tuple[np.ndarray, np.ndarray]]]
    chemicals: list[str]

    @classmethod
    def load(cls, sources: list[str] | None = None) -> "ChemicalPool":
        grid = np.load(PROC / "grid.npz")["wavenumber"]
        manifest = pd.read_csv(MANIFEST)
        pure = manifest[manifest["kind"] == "pure"].copy()
        if sources:
            pure = pure[pure["source"].isin(sources)]

        # Load each stack ONCE into plain numpy arrays (not lazy NpzFile)
        # NpzFile re-decompresses the full array on every __getitem__ call,
        # and arr[i] returns a view that keeps the decompressed copy alive,
        # causing O(N * stack_size) memory growth.
        stacks: dict[str, dict[str, np.ndarray]] = {}
        for s in pure["source"].unique():
            z = np.load(PROC / f"{s}_stack.npz", allow_pickle=True)
            stacks[s] = {"intensity": z["intensity"], "mask": z["mask"],
                         "spectrum_id": z["spectrum_id"]}
            z.close()

        id_to_index = {s: {sid: i for i, sid in enumerate(list(d["spectrum_id"]))} for s, d in stacks.items()}

        by_chem: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for _, row in pure.iterrows():
            src = row["source"]
            i = id_to_index[src][row["spectrum_id"]]
            key = f"{src}:{row['chemical']}"
            entry = (stacks[src]["intensity"][i].copy(), stacks[src]["mask"][i].copy())
            by_chem.setdefault(key, []).append(entry)

        chemicals = sorted(by_chem.keys())
        return cls(grid=grid, by_chem=by_chem, chemicals=chemicals)

    def split(self, holdout_frac: float = 0.2, seed: int = 0) -> tuple["ChemicalPool", "ChemicalPool"]:
        """Deterministic chemical-level split → (train pool, holdout pool)."""
        rng = np.random.default_rng(seed)
        idx = np.arange(len(self.chemicals))
        rng.shuffle(idx)
        n_hold = max(1, int(round(holdout_frac * len(idx))))
        hold = set(self.chemicals[i] for i in idx[:n_hold])
        train_chems = [c for c in self.chemicals if c not in hold]
        hold_chems = [c for c in self.chemicals if c in hold]
        return (
            ChemicalPool(self.grid, {c: self.by_chem[c] for c in train_chems}, train_chems),
            ChemicalPool(self.grid, {c: self.by_chem[c] for c in hold_chems}, hold_chems),
        )


# ---------------------------------------------------------------------------
# Corruptions
# ---------------------------------------------------------------------------


def _random_baseline(grid: np.ndarray, rng: np.random.Generator, amp: float = 0.05) -> np.ndarray:
    """Generate a physically realistic fluorescence baseline (always non-negative).

    Real fluorescence backgrounds are smooth, broad, and always ADD intensity
    to the Raman signal.  We use an exponential-quadratic shape so the
    baseline is guaranteed positive everywhere.
    """
    x = np.linspace(-1, 1, grid.size, dtype=np.float32)
    # Smooth positive bump: exp(-(x-center)^2 / width^2) scaled randomly
    center = float(rng.uniform(-0.5, 0.5))
    width = float(rng.uniform(0.5, 2.0))
    bump = np.exp(-((x - center) ** 2) / (width ** 2 + 1e-9)).astype(np.float32)
    # Optional second bump for more complex shapes
    if rng.random() > 0.5:
        c2 = float(rng.uniform(-0.8, 0.8))
        w2 = float(rng.uniform(0.3, 1.5))
        bump2 = np.exp(-((x - c2) ** 2) / (w2 ** 2 + 1e-9)).astype(np.float32)
        bump = bump + float(rng.uniform(0.2, 0.8)) * bump2
    # Add a gentle linear slope
    slope = float(rng.uniform(-0.3, 0.3))
    bump = bump + np.clip(slope * x + 0.5, 0, None).astype(np.float32) * 0.3
    # Normalize and scale
    bump = bump / (np.max(bump) + 1e-9)
    return (bump * amp * float(rng.uniform(0.5, 1.5))).astype(np.float32)


def _add_noise(y: np.ndarray, rng: np.random.Generator, snr_db: float) -> np.ndarray:
    signal_p = float(np.mean(y ** 2)) + 1e-12
    noise_p = signal_p / (10 ** (snr_db / 10))
    sigma = np.sqrt(noise_p)
    gauss = rng.normal(0.0, sigma, size=y.shape).astype(np.float32)
    poisson_scale = np.clip(np.abs(y), 1e-6, None) * 0.05
    poisson = rng.normal(0.0, poisson_scale, size=y.shape).astype(np.float32)
    return (y + gauss + poisson).astype(np.float32)


def _peak_shift(spectrum: np.ndarray, rng: np.random.Generator, max_shift: int = 3) -> np.ndarray:
    shift = int(rng.integers(-max_shift, max_shift + 1))
    return np.roll(spectrum, shift).astype(np.float32)


def _jitter_broadening(spectrum: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = float(rng.uniform(0.5, 2.0))
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return np.convolve(spectrum, kernel, mode="same").astype(np.float32)


# ---------------------------------------------------------------------------
# Mixture generator
# ---------------------------------------------------------------------------


@dataclass
class SynthConfig:
    K_min: int = 1
    K_max: int = 8
    M_min: int = 0
    M_max: int = 5
    dirichlet_alpha_range: tuple[float, float] = (0.3, 2.0)
    baseline_amp: float = 0.05
    snr_db_range: tuple[float, float] = (10.0, 60.0)
    max_peak_shift: int = 3
    jitter_prob: float = 0.5
    p_correlated_pair: float = 0.33  # fraction of samples that force >=2 similar refs
    seed: int | None = None


class SyntheticMixtures(IterableDataset):
    """Torch IterableDataset that yields (y, R, c, b, mask) tensors on-the-fly."""

    def __init__(self, pool: ChemicalPool, cfg: SynthConfig | None = None):
        super().__init__()
        self.pool = pool
        self.cfg = cfg or SynthConfig()
        self._sim_pairs: list[tuple[str, str]] | None = None

    def _rng(self) -> np.random.Generator:
        worker = torch.utils.data.get_worker_info()
        seed = self.cfg.seed
        if seed is None:
            seed = np.random.SeedSequence().entropy
        if worker is not None:
            seed = int(seed) ^ int(worker.id) * 100003
        return np.random.default_rng(seed)

    def _compute_similarity_pairs(self, top_frac: float = 0.02) -> list[tuple[str, str]]:
        """Precompute a small pool of highly-correlated chemical pairs (once per instance)."""
        if self._sim_pairs is not None:
            return self._sim_pairs
        centroids = np.stack(
            [np.mean(np.stack([e[0] for e in self.pool.by_chem[c]], 0), 0) for c in self.pool.chemicals]
        )
        norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9
        C = (centroids / norms) @ (centroids / norms).T
        np.fill_diagonal(C, -np.inf)
        n = len(self.pool.chemicals)
        thresh = np.quantile(C[C > -np.inf], 1 - top_frac)
        i_idx, j_idx = np.where(C >= thresh)
        pairs = [(self.pool.chemicals[i], self.pool.chemicals[j]) for i, j in zip(i_idx, j_idx) if i < j]
        self._sim_pairs = pairs
        return pairs

    def _sample_spectrum(self, chemical: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        entries = self.pool.by_chem[chemical]
        i = int(rng.integers(0, len(entries)))
        it, mask = entries[i]
        return it.astype(np.float32), mask.astype(bool)

    def _sample_component_names(self, rng: np.random.Generator, K: int) -> list[str]:
        force_pair = rng.random() < self.cfg.p_correlated_pair and K >= 2 and len(self.pool.chemicals) >= 2
        chosen: list[str] = []
        if force_pair:
            pairs = self._compute_similarity_pairs()
            if pairs:
                p = pairs[int(rng.integers(0, len(pairs)))]
                chosen.extend(p)
        remaining = [c for c in self.pool.chemicals if c not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[: K - len(chosen)])
        return chosen[:K]

    def _sample_distractors(self, rng: np.random.Generator, exclude: set[str], M: int) -> list[str]:
        pool = [c for c in self.pool.chemicals if c not in exclude]
        rng.shuffle(pool)
        return pool[:M]

    def _build_one(self, rng: np.random.Generator) -> dict:
        cfg = self.cfg
        grid = self.pool.grid
        N = grid.size

        K = int(rng.integers(cfg.K_min, cfg.K_max + 1))
        M = int(rng.integers(cfg.M_min, cfg.M_max + 1))
        in_names = self._sample_component_names(rng, K)
        alpha = float(rng.uniform(*cfg.dirichlet_alpha_range))
        coeffs_in = rng.dirichlet(np.full(K, alpha)).astype(np.float32)

        refs = []
        masks = []
        for name in in_names:
            it, mask = self._sample_spectrum(name, rng)
            refs.append(it)
            masks.append(mask)

        dist_names = self._sample_distractors(rng, set(in_names), M)
        for name in dist_names:
            it, mask = self._sample_spectrum(name, rng)
            refs.append(it)
            masks.append(mask)

        R_clean = np.stack(refs, 0).astype(np.float32)
        M_mask = np.stack(masks, 0)
        union_mask = M_mask[:K].any(0)

        # Global intensity scale applied BEFORE mixing so labels stay consistent
        intensity_scale = float(rng.uniform(0.7, 1.3))
        R_scaled = R_clean * intensity_scale

        y = (coeffs_in[:, None] * R_scaled[:K]).sum(0)

        baseline = _random_baseline(grid, rng, amp=cfg.baseline_amp)
        y = y + baseline
        snr = float(rng.uniform(*cfg.snr_db_range))
        y = _add_noise(y, rng, snr)
        y = _peak_shift(y, rng, cfg.max_peak_shift)
        if rng.random() < cfg.jitter_prob:
            y = _jitter_broadening(y, rng)
        y[~union_mask] = 0.0

        # Shuffle in-mixture vs distractors so the model can't cheat on ordering
        total = K + M
        c_full = np.concatenate([coeffs_in, np.zeros(M, dtype=np.float32)])
        perm = np.arange(total)
        rng.shuffle(perm)
        R_out = R_scaled[perm]  # use scaled references so c * R matches y
        c_out = c_full[perm]
        ref_names = [(*in_names, *dist_names)[i] for i in perm]

        return {
            "y": torch.from_numpy(y),
            "R": torch.from_numpy(R_out),
            "c": torch.from_numpy(c_out),
            "baseline": torch.from_numpy(baseline),
            "mask": torch.from_numpy(union_mask),
            "K": K,
            "M": M,
            "snr_db": snr,
            "ref_names": ref_names,
        }

    def __iter__(self) -> Iterator[dict]:
        rng = self._rng()
        while True:
            yield self._build_one(rng)


# ---------------------------------------------------------------------------
# Convenience: build a fixed val batch
# ---------------------------------------------------------------------------


def make_fixed_batch(pool: ChemicalPool, n: int, cfg: SynthConfig | None = None, seed: int = 42) -> list[dict]:
    """Build a deterministic list of n samples for validation/plots."""
    cfg = cfg or SynthConfig(seed=seed)
    ds = SyntheticMixtures(pool, cfg)
    rng = np.random.default_rng(seed)
    return [ds._build_one(rng) for _ in range(n)]


if __name__ == "__main__":
    pool = ChemicalPool.load()
    print(f"pool: {len(pool.chemicals)} unique chemicals across {sum(len(v) for v in pool.by_chem.values())} spectra")
    train, hold = pool.split(holdout_frac=0.2, seed=0)
    print(f"train pool: {len(train.chemicals)}, holdout pool: {len(hold.chemicals)}")

    ds = SyntheticMixtures(train, SynthConfig(seed=1))
    it = iter(ds)
    for _ in range(3):
        s = next(it)
        print(f"K={s['K']} M={s['M']} snr={s['snr_db']:.1f} c[:K]={s['c'].numpy()[s['c'].numpy() > 0]}")
