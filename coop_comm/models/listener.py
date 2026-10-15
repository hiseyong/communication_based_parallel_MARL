"""Listener: obs (grid, pos, own_status, msg_in) -> move-action logits + value.
`use_gru=True` adds a recurrent layer for the hold/DONE-wait behavior that may
need memory beyond the Markovian (own_status, msg_in) signal; off by default
since both of those already encode "am I currently on the plate" / "what is
being asked of me right now" every step.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .encoders import ObsEncoder


class Listener(nn.Module):
    def __init__(
        self,
        encoder: ObsEncoder,
        vocab_size: int,
        n_actions: int = 6,
        msg_embed_dim: int = 16,
        use_gru: bool = False,
        gru_hidden: int = 128,
    ):
        super().__init__()
        self.encoder = encoder
        self.msg_embed = nn.Sequential(nn.Linear(vocab_size, msg_embed_dim), nn.ReLU())
        self.use_gru = use_gru
        feat_dim = encoder.out_dim + msg_embed_dim
        if use_gru:
            self.gru = nn.GRU(feat_dim, gru_hidden, batch_first=True)
            head_in = gru_hidden
        else:
            self.gru = None
            head_in = feat_dim
        self.gru_hidden = gru_hidden
        self.policy_head = nn.Linear(head_in, n_actions)
        self.value_head = nn.Linear(head_in, 1)

    def _features(self, grid: torch.Tensor, pos: torch.Tensor, own_status: torch.Tensor, msg_in: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(grid, pos, own_status)
        msg_feat = self.msg_embed(msg_in)
        return torch.cat([feat, msg_feat], dim=-1)

    def initial_hidden(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        if not self.use_gru:
            return None
        return torch.zeros(1, batch_size, self.gru_hidden, device=device)

    def forward_step(
        self,
        grid: torch.Tensor,
        pos: torch.Tensor,
        own_status: torch.Tensor,
        msg_in: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Single timestep: all tensors (B, ...). Returns (logits, value, new_hidden)."""
        x = self._features(grid, pos, own_status, msg_in)
        if self.use_gru:
            out, hidden = self.gru(x.unsqueeze(1), hidden)
            x = out.squeeze(1)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value, hidden

    def forward_seq(
        self,
        grid: torch.Tensor,
        pos: torch.Tensor,
        own_status: torch.Tensor,
        msg_in: torch.Tensor,
        hidden0: Optional[torch.Tensor] = None,
        done_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Whole-rollout sequence per env, for PPO updates. Shapes (B, T, ...).
        `done_mask` (B, T) marks episode boundaries within the chunk -> hidden
        state is zeroed right after a done step so memory never leaks across
        episodes. Returns (logits (B,T,A), value (B,T))."""
        B, T = grid.shape[0], grid.shape[1]
        x = self._features(grid.flatten(0, 1), pos.flatten(0, 1), own_status.flatten(0, 1), msg_in.flatten(0, 1))
        x = x.view(B, T, -1)
        if self.use_gru:
            if hidden0 is None:
                hidden0 = self.initial_hidden(B, grid.device)
            outs = []
            h = hidden0
            for t in range(T):
                out, h = self.gru(x[:, t : t + 1], h)
                outs.append(out)
                if done_mask is not None:
                    reset = done_mask[:, t].view(1, B, 1)
                    h = h * (1.0 - reset)
            x = torch.cat(outs, dim=1)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value
