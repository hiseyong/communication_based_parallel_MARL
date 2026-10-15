from coop_comm.comm.vocab import Vocab
from coop_comm.envs.layout_generator import LayoutGenerator
from coop_comm.envs.oracle import oracle_message


def test_oracle_requests_before_barrier_then_done_after():
    gen = LayoutGenerator(K=3, excluded_combos=[("OBSTACLE", 2)])
    layout = gen.sample("train", "L1", 0)
    vocab = Vocab(tuple(layout.colors))

    # whichever route has a barrier should: REQ before it, DONE just after, NONE elsewhere.
    for label in ("a", "b"):
        route = layout.route(label)
        other_route = layout.route(layout.other_label(label))
        if not route.barriers:
            continue
        barrier = route.barriers[0]
        bc = barrier.pos[1]

        before = (barrier.pos[0], max(0, bc - 1))
        msg = oracle_message(route, before, other_route, other_route.start, frozenset(), vocab)
        assert vocab.decode(msg).startswith("REQ_")

        at = barrier.pos
        msg_at = oracle_message(route, at, other_route, other_route.start, frozenset(), vocab)
        assert vocab.decode(msg_at) == "DONE"

        far_after = (barrier.pos[0], min(route.W - 1, bc + 10))
        if far_after[1] > bc + 2:
            msg_far = oracle_message(route, far_after, other_route, other_route.start, frozenset(), vocab)
            assert vocab.decode(msg_far) == "NONE"
        return
    raise AssertionError("expected at least one route with a barrier in this L1 layout")


def test_oracle_does_not_leak_a_barrier_outside_view_radius():
    """spec 3.8: REQ_* only once the barrier is '시야 내' (within sight) --
    an un-trainable Speaker would result if the oracle announced REQ_* for a
    barrier far outside the agent's egocentric observation window."""
    gen = LayoutGenerator(K=3, excluded_combos=[("OBSTACLE", 2)])
    vocab = Vocab(("RED", "GREEN", "BLUE"))
    view_radius = 3  # matches V=7 egocentric crop (V // 2)

    for idx in range(50):
        layout = gen.sample("train", "L1", idx)
        for label in ("a", "b"):
            route = layout.route(label)
            other_route = layout.route(layout.other_label(label))
            if not route.barriers:
                continue
            barrier = route.barriers[0]
            bc = barrier.pos[1]
            far_col = bc - view_radius - 2
            if far_col < 0:
                continue  # barrier too close to START on this layout; try another
            far_before = (barrier.pos[0], far_col)
            msg_far = oracle_message(route, far_before, other_route, other_route.start, frozenset(), vocab, view_radius=view_radius)
            assert vocab.decode(msg_far) == "NONE", "oracle must not request a barrier that isn't within view_radius yet"

            near_before = (barrier.pos[0], bc - 1)
            msg_near = oracle_message(route, near_before, other_route, other_route.start, frozenset(), vocab, view_radius=view_radius)
            assert vocab.decode(msg_near).startswith("REQ_"), "oracle must request once the barrier enters view_radius"
            return
    raise AssertionError("no L1 layout in the first 50 train indices had enough room to test the view-radius gap")
