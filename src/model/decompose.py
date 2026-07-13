"""
Decomposition model: shared encoder + cross-attention + coefficient & baseline heads.

Flow:
    1. Encode unknown spectrum → z_u (B, d)
    2. Encode each reference → z_r (B, K, d)   [shared weights with step 1]
    3. Cross-attention: z_u queries z_r → contextualized unknown (B, 1, d)
    4. For each reference: concat(z_r_i, context) → MLP → softplus → c_i
    5. Baseline head: MLP(z_u) → 6 polynomial coefficients → baseline curve

Variable K is handled via padding + a ref_mask that excludes pad slots.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SpectrumEncoder


class DecomposeModel(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_transformer_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        poly_order: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.poly_order = poly_order
        n_poly = poly_order + 1

        # Shared encoder for unknown + references
        self.encoder = SpectrumEncoder(
            d_model=d_model,
            n_transformer_layers=n_transformer_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Cross-attention: unknown (query) attends to references (key/value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Per-reference coefficient scorer
        # Input: [z_r, ctx, spectral_features(3)] → coefficient
        self.scorer = nn.Sequential(
            nn.Linear(2 * d_model + 3, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Baseline head: unknown embedding → polynomial coefficients
        self.baseline_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_poly),
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
        coeffs : (B, K_max) predicted coefficients (≥ 0 via softplus)
        baseline : (B, N) predicted baseline polynomial
        """
        B, K_max, N = R.shape

        # 1. Encode unknown
        z_u = self.encoder(y)  # (B, d)

        # 2. Encode references (reshape to batch, encode, reshape back)
        R_flat = R.reshape(B * K_max, N)  # (B*K, N)
        z_r_flat = self.encoder(R_flat)    # (B*K, d)
        z_r = z_r_flat.reshape(B, K_max, self.d_model)  # (B, K, d)

        # 3. Cross-attention: unknown queries references
        z_u_q = z_u.unsqueeze(1)  # (B, 1, d)
        # key_padding_mask: True where padding → invert ref_mask
        key_pad = ~ref_mask if ref_mask is not None else None
        attn_out, _ = self.cross_attn(z_u_q, z_r, z_r, key_padding_mask=key_pad)
        # attn_out: (B, 1, d)

        # 4. Spectral similarity features (computed directly in signal space)
        # These bypass the encoder and give the scorer direct signal about
        # which references match the unknown spectrum.
        y_exp = y.unsqueeze(1).expand_as(R)  # (B, K, N)
        # a) cosine similarity
        cos_sim = F.cosine_similarity(y_exp, R, dim=-1, eps=1e-8)  # (B, K)
        # b) dot product (unnormalized correlation)
        dot_prod = (y_exp * R).sum(dim=-1)  # (B, K)
        # c) L2 distance (inverse)
        l2_dist = torch.norm(y_exp - R, dim=-1)  # (B, K)

        spec_feats = torch.stack([cos_sim, dot_prod, l2_dist], dim=-1)  # (B, K, 3)

        # Score each reference
        ctx = attn_out.expand(-1, K_max, -1)  # (B, K, d)
        combined = torch.cat([z_r, ctx, spec_feats], dim=-1)  # (B, K, 2d+3)
        coeffs = self.scorer(combined).squeeze(-1)  # (B, K)
        # No activation — let the model learn raw coefficients.
        # Non-negativity enforced by loss penalty, not activation.

        # Zero out padding positions
        if ref_mask is not None:
            coeffs = coeffs * ref_mask.float()

        # 5. Baseline head
        poly_coeffs = self.baseline_head(z_u)  # (B, n_poly)
        x = torch.linspace(-1.0, 1.0, N, device=y.device, dtype=y.dtype)
        # powers: (n_poly, N)
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
