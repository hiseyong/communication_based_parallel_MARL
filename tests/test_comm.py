import numpy as np

from coop_comm.comm.channel import Channel
from coop_comm.comm.vocab import Vocab


def test_vocab_size_and_roundtrip():
    v = Vocab(("RED", "GREEN", "BLUE"))
    assert v.size == 2 * 3 + 3
    for tok in v.tokens:
        idx = v.encode(tok)
        assert v.decode(idx) == tok


def test_vocab_flat_slot_roundtrip():
    v = Vocab(("RED", "GREEN", "BLUE"))
    for idx in range(v.size):
        intent, device, color = v.to_slot(idx)
        assert v.from_slot(intent, device, color) == idx


def test_vocab_scales_with_k():
    for K in (1, 2, 5):
        v = Vocab(tuple(f"C{i}" for i in range(K)))
        assert v.size == 2 * K + 3


def test_channel_identity_passthrough():
    ch = Channel(kind="identity")
    rng = np.random.default_rng(0)
    for m in range(5):
        assert ch.transmit(m, vocab_size=9, none_idx=0, rng=rng) == m


def test_channel_drop_eventually_hits_none():
    ch = Channel(kind="drop", p=1.0)
    rng = np.random.default_rng(0)
    assert ch.transmit(3, vocab_size=9, none_idx=0, rng=rng) == 0


def test_channel_symbol_flip_changes_symbol_when_forced():
    ch = Channel(kind="symbol_flip", p=1.0)
    rng = np.random.default_rng(0)
    out = ch.transmit(3, vocab_size=9, none_idx=0, rng=rng)
    assert out != 3
    assert 0 <= out < 9
