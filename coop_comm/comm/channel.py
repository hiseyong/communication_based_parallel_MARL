"""Message channel: applied to a sender's token before it reaches the receiver's
`msg_in` on the next step. `identity` is the default (noiseless) channel;
`symbol_flip` / `drop` are for later robustness experiments.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Channel:
    kind: str = "identity"  # "identity" | "symbol_flip" | "drop"
    p: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("identity", "symbol_flip", "drop"):
            raise ValueError(f"unknown channel kind {self.kind!r}")

    def transmit(self, msg_idx: int, vocab_size: int, none_idx: int, rng: np.random.Generator) -> int:
        if self.kind == "identity" or self.p <= 0.0:
            return msg_idx
        if self.kind == "drop":
            if rng.random() < self.p:
                return none_idx
            return msg_idx
        if self.kind == "symbol_flip":
            if rng.random() < self.p:
                others = [i for i in range(vocab_size) if i != msg_idx]
                return int(rng.choice(others))
            return msg_idx
        raise AssertionError("unreachable")
