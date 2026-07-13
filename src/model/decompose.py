"""
Decomposition model v2: mixture encoder + direct reference matching + physics decoder.

Key changes from v1 (based on EGU-Net, PNAS 2024, RamanFormer literature):
  1. References are NOT encoded through a deep neural network.
     Instead, a lightweight linear projection is used. This prevents
     encoder collapse (v1 had cosine_sim=0.999 between all ref embeddings).
  2. Coefficients use softmax → architectural sum-to-one + non-negativity.
     No soft penalty needed.
  3. The "decoder" is just physics: y_hat = sum(c_i * R_i).
     The model must reconstruct the mixture from its own predictions.
  4. Smaller model (d=128 vs 256, 3 conv blocks vs 4, 1 transformer vs 2).

Flow:
    1. Encode mixture → z_u (B, d)
    2. Project each reference with lightweight linear → z_r (B, K, d)
    3. Compute spectral features in signal space (cos_sim, dot, L2)
    4. Cross-attention: z_u queries z_r → context
    5. Scorer MLP([z_r_i, context, spec_feats]) → logit per reference
    6. Softmax over references → coefficients c (sum-to-one, non-negative)
    7. Physics decoder: y_hat = sum(c_i * R_i) (no learned params)
    8. Baseline head: MLP(z_u) → polynomial baseline
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SpectrumEncoder


class DecomposeModel(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_transformer_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.1,
        poly_order: int = 5,
        spectrum_len: int = 3001,
    ):
        super().__init__()
        self.d_model = d_model
        self.poly_order = poly_order
        n_poly = poly_order + 1

        # Mixture encoder (only for the unknown spectrum)
        self.encoder = SpectrumEncoder(
            d_model=d_model,
            n_transformer_layers=n_transformer_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Lightweight reference projection (NOT a deep encoder)
        # Simple linear map from signal space to embedding space
        self.ref_proj = nn.Sequential(
            nn.Linear(spectrum_len, d_model),
            nn.LayerNorm(d_model),
        )

        # Cross-attention: unknown (query) attends to references (key/value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Per-reference coefficient scorer
        # Input: [z_r, ctx, spectral_features(3)] → logit
        self.scorer = nn.Sequential(
            nn.Linear(2 * d_model + 3, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Baseline head: mixture embedding → polynomial coefficients
        self.baseline_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, n_poly),
        )

    def forward(
        self,
        y: torch.Tensor,
        R: torch.Tensor,
        ref_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        y : (B, N) unknown mixture spectrum
        R : (B, K_max, N) reference spectra (padded)
        ref_mask : (B, K_max) bool — True for real references, False for padding

        Returns
        -------
        coeffs : (B, K_max) predicted coefficients (sum-to-one via softmax, ≥ 0)
        baseline : (B, N) predicted baseline polynomial
        """
        B, K_max, N = R.shape

        # 1. Encode mixture only (NOT references)
        z_u = self.encoder(y)  # (B, d)

        # 2. Lightweight reference projection (linear, not deep encoder)
        z_r = self.ref_proj(R)  # (B, K_max, d)

        # 3. Cross-attention: mixture queries references
        z_u_q = z_u.unsqueeze(1)  # (B, 1, d)
        key_pad = ~ref_mask if ref_mask is not None else None
        attn_out, _ = self.cross_attn(z_u_q, z_r, z_r, key_padding_mask=key_pad)
        # attn_out: (B, 1, d)

        # 4. Spectral similarity features (direct signal space — bypass encoder)
        y_exp = y.unsqueeze(1).expand_as(R)  # (B, K, N)
        cos_sim = F.cosine_similarity(y_exp, R, dim=-1, eps=1e-8)  # (B, K)
        dot_prod = (y_exp * R).sum(dim=-1)  # (B, K)
        l2_dist = torch.norm(y_exp - R, dim=-1)  # (B, K)
        spec_feats = torch.stack([cos_sim, dot_prod, l2_dist], dim=-1)  # (B, K, 3)

        # 5. Score each reference
        ctx = attn_out.expand(-1, K_max, -1)  # (B, K, d)
        combined = torch.cat([z_r, ctx, spec_feats], dim=-1)  # (B, K, 2d+3)
        logits = self.scorer(combined).squeeze(-1)  # (B, K)

        # 6. Softmax over valid references → sum-to-one, non-negative
        # Mask out padding positions with large negative value
        if ref_mask is not None:
            logits = logits.masked_fill(~ref_mask, -1e9)
        coeffs = F.softmax(logits, dim=-1)  # (B, K)
        if ref_mask is not None:
            coeffs = coeffs * ref_mask.float()  # zero out padding

        # 7. Baseline head
        poly_coeffs = self.baseline_head(z_u)  # (B, n_poly)
        x = torch.linspace(-1.0, 1.0, N, device=y.device, dtype=y.dtype)
        powers = torch.stack([x ** k for k in range(self.poly_order + 1)], dim=0)
        baseline = torch.mm(poly_coeffs, powers)  # (B, N)

        return coeffs, baseline


def collate_decompose(samples: list[dict]) -> dict:
    """
    Collate variable-K samples into padded batch tensors.

    Input: list of dicts from SyntheticMixtures / make_fixed_batch.
    Output: dict with keys y, R, c, baseline, mask, ref_mask, K_list, M_list.
    """
    B = len(samples)
    K_max = max(s["R"].shape[0] for s in samples)
    N = samples[0]["y"].shape[0]

    y = torch.stack([s["y"] for s in samples])           # (B, N)
    baseline = torch.stack([s["baseline"] for s in samples])  # (B, N)
    mask = torch.stack([s["mask"] for s in samples])      # (B, N)

    R = torch.zeros(B, K_max, N)
    c = torch.zeros(B, K_max)
    ref_mask = torch.zeros(B, K_max, dtype=torch.bool)

    for i, s in enumerate(samples):
        k = s["R"].shape[0]
        R[i, :k] = s["R"]
        c[i, :k] = s["c"]
        ref_mask[i, :k] = True

    return {
        "y": y,
        "R": R,
        "c": c,
        "baseline": baseline,
        "mask": mask,
        "ref_mask": ref_mask,
        "K_list": [s["K"] for s in samples],
        "M_list": [s["M"] for s in samples],
    }
