"""Builds Speaker/Listener from a small hyperparam dict, and derives the
grid-channel / status-vector / vocab sizes from K the same way DualRouteEnv
does -- kept in one place so every train/eval script agrees with the env."""
from __future__ import annotations

from typing import Dict

from .encoders import ObsEncoder
from .listener import Listener
from .speaker import Speaker


def channel_dims(K: int) -> Dict[str, int]:
    return {
        "n_channels": 5 + 4 * K,
        "status_dim": 2 * (K + 1),
        "vocab_size": 2 * K + 3,
    }


def build_speaker(K: int, V: int, cnn_channels: int = 32, hidden: int = 128, gumbel: bool = False) -> Speaker:
    dims = channel_dims(K)
    encoder = ObsEncoder(dims["n_channels"], V, pos_dim=2, status_dim=dims["status_dim"], cnn_channels=cnn_channels, hidden=hidden)
    return Speaker(encoder, dims["vocab_size"], gumbel=gumbel)


def build_listener(K: int, V: int, cnn_channels: int = 32, hidden: int = 128, msg_embed_dim: int = 16, use_gru: bool = False, gru_hidden: int = 128) -> Listener:
    dims = channel_dims(K)
    encoder = ObsEncoder(dims["n_channels"], V, pos_dim=2, status_dim=dims["status_dim"], cnn_channels=cnn_channels, hidden=hidden)
    return Listener(encoder, dims["vocab_size"], n_actions=6, msg_embed_dim=msg_embed_dim, use_gru=use_gru, gru_hidden=gru_hidden)
