"""
Combined loss for spectral decomposition (v3).

v3 changes:
  1. Increased lambda_r (reconstruction) — with softplus coefficients (no sum-to-1
     constraint), reconstruction loss provides essential physics regularization
     that prevents coefficient drift.
  2. Increased lambda_l1 (sparsity) — with softplus, coefficients are unbounded
     above; stronger L1 drives distractors to zero and prevents over-estimation.
  3. All v2 improvements kept: b_pred in reconstruction, SAD loss.

L = lambda_c    * MAE(c_pred, c_true)
  + lambda_r    * MSE(y - b_pred, c_pred @ R)     reconstruction with MODEL's baseline
  + lambda_sad  * SAD(y - b_pred, c_pred @ R)      spectral angle distance
  + lambda_b    * MAE(b_pred, b_true)              baseline supervision
  + lambda_l1   * L1(c_pred)                       sparsity (stronger for softplus)
  + lambda_bneg * mean(relu(-b_pred))              soft non-negativity for baseline
"""

from __future__ import annotations

import torch
import torch.nn as nn


def spectral_angle_distance(y: torch.Tensor, y_hat: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
    """
    Compute mean SAD between target and prediction over the batch.

    SAD = arccos(cos_sim(y, y_hat)) / pi,  normalized to [0, 1].
    Masked points are excluded.
    """
    # Apply mask
    y_m = y * mask.float()
    y_hat_m = y_hat * mask.float()

    # Cosine similarity per sample
    dot = (y_m * y_hat_m).sum(dim=-1)  # (B,)
    norm_y = torch.norm(y_m, dim=-1).clamp(min=1e-8)
    norm_yh = torch.norm(y_hat_m, dim=-1).clamp(min=1e-8)
    cos_sim = dot / (norm_y * norm_yh)
    cos_sim = cos_sim.clamp(-1 + 1e-7, 1 - 1e-7)  # numerical stability

    sad = torch.acos(cos_sim) / torch.pi  # [0, 1]
    return sad.mean()


class DecomposeLoss(nn.Module):
    def __init__(
        self,
        lambda_c: float = 1.0,
        lambda_r: float = 50.0,
        lambda_sad: float = 1.0,
        lambda_b: float = 0.5,
        lambda_l1: float = 0.1,
        lambda_bneg: float = 10.0,
    ):
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_r = lambda_r
        self.lambda_sad = lambda_sad
        self.lambda_b = lambda_b
        self.lambda_l1 = lambda_l1
        self.lambda_bneg = lambda_bneg

    def forward(
        self,
        c_pred: torch.Tensor,
        c_true: torch.Tensor,
        b_pred: torch.Tensor,
        b_true: torch.Tensor,
        y: torch.Tensor,
        R: torch.Tensor,
        mask: torch.Tensor,
        ref_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Parameters
        ----------
        c_pred : (B, K_max) — softplus output (non-negative, independent)
        c_true : (B, K_max)
        b_pred : (B, N) — model's predicted baseline
        b_true : (B, N) — ground truth baseline
        y      : (B, N) — corrupted mixture
        R      : (B, K_max, N) — reference spectra
        mask   : (B, N) — valid wavenumber mask
        ref_mask : (B, K_max) — real reference mask
        """
        # 1. Coefficient MAE (only on real references)
        coeff_diff = torch.abs(c_pred - c_true) * ref_mask.float()
        loss_c = coeff_diff.sum() / ref_mask.float().sum().clamp(min=1)

        # 2. Reconstruction: (y - b_pred) ≈ sum(c_pred * R)
        # Uses b_pred (model's own prediction), NOT b_true.
        # This forces the model to learn a useful baseline.
        y_signal = y - b_pred  # model's estimate of clean signal
        y_recon = (c_pred.unsqueeze(-1) * R).sum(dim=1)  # (B, N)
        recon_diff = (y_signal - y_recon) ** 2 * mask.float()
        loss_r = recon_diff.sum() / mask.float().sum().clamp(min=1)

        # 3. SAD (Spectral Angle Distance) — scale-invariant shape matching
        loss_sad = spectral_angle_distance(y_signal, y_recon, mask)

        # 4. Baseline MAE (supervised)
        base_diff = torch.abs(b_pred - b_true) * mask.float()
        loss_b = base_diff.sum() / mask.float().sum().clamp(min=1)

        # 5. L1 sparsity on coefficients (encourage distractor suppression)
        loss_l1 = (c_pred * ref_mask.float()).abs().mean()

        # 6. Soft non-negativity penalty for baseline
        # (fluorescence is always additive; ReLU kills gradients so we use soft penalty)
        loss_bneg = torch.relu(-b_pred).mean()

        # Total
        loss = (self.lambda_c * loss_c
                + self.lambda_r * loss_r
                + self.lambda_sad * loss_sad
                + self.lambda_b * loss_b
                + self.lambda_l1 * loss_l1
                + self.lambda_bneg * loss_bneg)

        detail = {
            "loss": loss.item(),
            "loss_c": loss_c.item(),
            "loss_r": loss_r.item(),
            "loss_sad": loss_sad.item(),
            "loss_b": loss_b.item(),
            "loss_l1": loss_l1.item(),
            "loss_bneg": loss_bneg.item(),
        }
        return loss, detail
