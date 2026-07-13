"""
Shared spectrum encoder: Conv1D stack + optional Transformer → embedding.

The same encoder processes both the unknown mixture and each reference
spectrum, so they live in the same representation space.

Architecture:
    Conv1D(1→32, k=7) + BN + ReLU + MaxPool(4)    → 750 tokens
    Conv1D(32→64, k=7) + BN + ReLU + MaxPool(4)    → 187 tokens
    Conv1D(64→128, k=7) + BN + ReLU + MaxPool(4)   → 46 tokens
    Conv1D(128→d, k=7) + BN + ReLU + MaxPool(4)    → 11 tokens
    [optional] 2× TransformerEncoderLayer(d, nhead=4)
    Global Average Pooling → (B, d)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, pool: int = 4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpectrumEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_transformer_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.conv_stack = nn.Sequential(
            ConvBlock(1, 32, kernel=7, pool=4),
            ConvBlock(32, 64, kernel=7, pool=4),
            ConvBlock(64, 128, kernel=7, pool=4),
            ConvBlock(128, d_model, kernel=7, pool=4),
        )

        self.use_transformer = n_transformer_layers > 0
        if self.use_transformer:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=n_transformer_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N) raw spectrum on the unified grid (3001 points)

        Returns
        -------
        emb : (B, d_model) global embedding
        """
        x = x.unsqueeze(1)  # (B, 1, N) — single channel
        x = self.conv_stack(x)  # (B, d_model, T) where T ≈ 11

        if self.use_transformer:
            x = x.transpose(1, 2)  # (B, T, d_model) — batch_first
            x = self.transformer(x)
            x = x.transpose(1, 2)  # (B, d_model, T)

        # Global average pooling
        emb = x.mean(dim=2)  # (B, d_model)
        return emb
