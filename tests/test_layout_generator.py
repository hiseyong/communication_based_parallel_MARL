import pytest

from coop_comm.envs.layout_generator import LayoutGenerator, is_solvable

LEVELS = ["L1", "L2", "L3"]


@pytest.mark.parametrize("level", LEVELS)
def test_1000_generated_layouts_all_solvable(level):
    gen = LayoutGenerator(K=3, excluded_combos=[("OBSTACLE", 2)])
    for i in range(1000):
        layout = gen.sample("train", level, i)
        ok, _ = is_solvable(layout)
        assert ok, f"layout train/{level}/{i} (seed={layout.seed}) is not solvable"


def test_splits_are_seed_disjoint():
    gen = LayoutGenerator(K=3, excluded_combos=[("OBSTACLE", 2)])
    seeds = {}
    for split in ["train", "val", "test_layout", "test_combo"]:
        seeds[split] = {gen.sample(split, "L2", i).seed for i in range(20)}
    all_seeds = [s for group in seeds.values() for s in group]
    assert len(all_seeds) == len(set(all_seeds))


def test_test_combo_forces_excluded_combo():
    excluded = [("OBSTACLE", 2)]
    gen = LayoutGenerator(K=3, excluded_combos=excluded)
    for i in range(50):
        layout = gen.sample("test_combo", "L2", i)
        combos = {(b.kind.name, b.color) for b in layout.route_a.barriers + layout.route_b.barriers}
        assert combos & set(excluded), f"layout {i} does not contain the forced excluded combo"


def test_train_val_test_layout_avoid_excluded_combo():
    excluded = [("OBSTACLE", 2)]
    gen = LayoutGenerator(K=3, excluded_combos=excluded)
    for split in ["train", "val", "test_layout"]:
        for i in range(100):
            layout = gen.sample(split, "L2", i)
            combos = {(b.kind.name, b.color) for b in layout.route_a.barriers + layout.route_b.barriers}
            assert not (combos & set(excluded)), f"{split} layout {i} leaked excluded combo"
