"""Shared observation encoder: a small CNN over the egocentric grid plus an MLP
over (pos, own_status), reused by both Speaker and Listener so the two stages
can later share/swap encoders (stage-3 prep, not implemented yet).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ObsEncoder(nn.Module):
    def __init__(self, n_channels: int, V: int, pos_dim: int = 2, status_dim: int = 8, cnn_channels: int = 32, hidden: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        cnn_out_dim = cnn_channels * V * V
        self.proj = nn.Sequential(
            nn.Linear(cnn_out_dim + pos_dim + status_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.out_dim = hidden

    def forward(self, grid: torch.Tensor, pos: torch.Tensor, own_status: torch.Tensor) -> torch.Tensor:
        x = self.cnn(grid)
        x = x.flatten(1)
        x = torch.cat([x, pos, own_status], dim=-1)
        return self.proj(x)
