"""Fixed communication vocabulary: NONE, REQ_PLATE_<c>, REQ_LEVER_<c>, DONE, WAIT.

Size is always 2K + 3 for K colors, generated automatically from the color list
so experiments can change K without touching this file. A `Vocab` also exposes
a flat-index <-> (intent, device, color) slot view, so a future slotted
representation (`[intent][device][color]`) can reuse the same index space.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

INTENTS = ["NONE", "REQ", "DONE", "WAIT"]
DEVICES = ["NONE", "PLATE", "LEVER"]


@dataclass(frozen=True)
class Vocab:
    colors: Tuple[str, ...]

    def __post_init__(self) -> None:
        tokens: List[str] = ["NONE"]
        tokens += [f"REQ_PLATE_{c}" for c in self.colors]
        tokens += [f"REQ_LEVER_{c}" for c in self.colors]
        tokens += ["DONE", "WAIT"]
        object.__setattr__(self, "tokens", tuple(tokens))
        object.__setattr__(self, "_index", {t: i for i, t in enumerate(tokens)})

    @property
    def K(self) -> int:
        return len(self.colors)

    @property
    def size(self) -> int:
        return len(self.tokens)  # 2K + 3

    @property
    def none_idx(self) -> int:
        return 0

    @property
    def done_idx(self) -> int:
        return self.size - 2

    @property
    def wait_idx(self) -> int:
        return self.size - 1

    def req_plate(self, color: int) -> int:
        return 1 + color

    def req_lever(self, color: int) -> int:
        return 1 + self.K + color

    def encode(self, token: str) -> int:
        return self._index[token]

    def decode(self, idx: int) -> str:
        return self.tokens[idx]

    def one_hot(self, idx: int) -> np.ndarray:
        vec = np.zeros(self.size, dtype=np.float32)
        vec[idx] = 1.0
        return vec

    # --- flat <-> slot (intent, device, color) ---------------------------
    def to_slot(self, idx: int) -> Tuple[int, int, int]:
        """color index is 0-based into self.colors, or -1 if not applicable."""
        if idx == self.none_idx:
            return INTENTS.index("NONE"), DEVICES.index("NONE"), -1
        if idx == self.done_idx:
            return INTENTS.index("DONE"), DEVICES.index("NONE"), -1
        if idx == self.wait_idx:
            return INTENTS.index("WAIT"), DEVICES.index("NONE"), -1
        if 1 <= idx <= self.K:
            return INTENTS.index("REQ"), DEVICES.index("PLATE"), idx - 1
        if self.K + 1 <= idx <= 2 * self.K:
            return INTENTS.index("REQ"), DEVICES.index("LEVER"), idx - self.K - 1
        raise ValueError(f"index {idx} out of range for vocab size {self.size}")

    def from_slot(self, intent: int, device: int, color: int) -> int:
        intent_name = INTENTS[intent]
        device_name = DEVICES[device]
        if intent_name == "NONE":
            return self.none_idx
        if intent_name == "DONE":
            return self.done_idx
        if intent_name == "WAIT":
            return self.wait_idx
        if intent_name == "REQ" and device_name == "PLATE":
            return self.req_plate(color)
        if intent_name == "REQ" and device_name == "LEVER":
            return self.req_lever(color)
        raise ValueError(f"invalid slot combination intent={intent_name} device={device_name}")
