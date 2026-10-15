"""Speaker: obs -> message logits. `--gumbel` switches on a Gumbel-Softmax head
for a later end-to-end fine-tuning stage (not implemented here); normal
training uses plain softmax cross-entropy on the logits."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import ObsEncoder


class Speaker(nn.Module):
    def __init__(self, encoder: ObsEncoder, vocab_size: int, gumbel: bool = False):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.out_dim, vocab_size)
        self.vocab_size = vocab_size
        self.gumbel = gumbel

    def logits(self, grid: torch.Tensor, pos: torch.Tensor, own_status: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(grid, pos, own_status)
        return self.head(feat)

    def forward(self, grid: torch.Tensor, pos: torch.Tensor, own_status: torch.Tensor) -> torch.Tensor:
        return self.logits(grid, pos, own_status)

    @torch.no_grad()
    def act(self, grid: torch.Tensor, pos: torch.Tensor, own_status: torch.Tensor) -> torch.Tensor:
        """Greedy message index, for rollout/eval use."""
        return self.logits(grid, pos, own_status).argmax(dim=-1)

    def gumbel_sample(self, grid: torch.Tensor, pos: torch.Tensor, own_status: torch.Tensor, tau: float = 1.0, hard: bool = True) -> torch.Tensor:
        logits = self.logits(grid, pos, own_status)
        return F.gumbel_softmax(logits, tau=tau, hard=hard)
