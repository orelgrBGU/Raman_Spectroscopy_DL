"""
Combined loss for spectral decomposition.

L = lambda_c   * MAE(c_pred, c_true)
  + lambda_r   * reconstruction_loss(y - b_true, R, c_pred, mask)
  + lambda_b   * MAE(b_pred, b_true)  on masked points
  + lambda_l1  * L1(c_pred)           sparsity on predicted coefficients
  + lambda_neg * mean(relu(-c_pred))  penalize negative coefficients
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DecomposeLoss(nn.Module):
    def __init__(
        self,
        lambda_c: float = 1.0,
        lambda_r: float = 1.0,
        lambda_b: float = 0.5,
        lambda_l1: float = 0.01,
        lambda_neg: float = 0.1,
    ):
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_r = lambda_r
        self.lambda_b = lambda_b
        self.lambda_l1 = lambda_l1
        self.lambda_neg = lambda_neg

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
        c_pred : (B, K_max)
        c_true : (B, K_max)
        b_pred : (B, N)
        b_true : (B, N)
        y      : (B, N) corrupted mixture
        R      : (B, K_max, N) reference spectra
        mask   : (B, N) valid wavenumber mask
        ref_mask : (B, K_max) real reference mask

        Returns
        -------
        loss : scalar
        detail : dict of individual loss components (for logging)
        """
        # 1. Coefficient MAE (only on real references)
        coeff_diff = torch.abs(c_pred - c_true) * ref_mask.float()
        loss_c = coeff_diff.sum() / ref_mask.float().sum().clamp(min=1)

        # 2. Reconstruction: (y - b_true) ≈ sum(c_pred * R)
        # Use b_true so the baseline head can't absorb the signal.
        # (B, K, 1) * (B, K, N) → sum over K → (B, N)
        y_signal = y - b_true  # true signal without baseline
        y_recon = (c_pred.unsqueeze(-1) * R).sum(dim=1)
        recon_diff = (y_signal - y_recon) ** 2 * mask.float()
        loss_r = recon_diff.sum() / mask.float().sum().clamp(min=1)

        # 3. Baseline MAE (on masked points)
        base_diff = torch.abs(b_pred - b_true) * mask.float()
        loss_b = base_diff.sum() / mask.float().sum().clamp(min=1)

        # 4. L1 sparsity on coefficients
        loss_l1 = (c_pred * ref_mask.float()).abs().mean()

        # 5. Non-negativity penalty (soft constraint instead of activation)
        loss_neg = torch.relu(-c_pred * ref_mask.float()).mean()

        # Total
        loss = (self.lambda_c * loss_c
                + self.lambda_r * loss_r
                + self.lambda_b * loss_b
                + self.lambda_l1 * loss_l1
                + self.lambda_neg * loss_neg)

        detail = {
            "loss": loss.item(),
            "loss_c": loss_c.item(),
            "loss_r": loss_r.item(),
            "loss_b": loss_b.item(),
            "loss_l1": loss_l1.item(),
            "loss_neg": loss_neg.item(),
        }
        return loss, detail
