"""Procedural generation of DualRouteEnv layouts, with joint-state BFS solvability checks.

Each route is a single main corridor (row `H // 2`) from START (col 0) to EXIT
(col W-1), with short perpendicular alcoves branching off it that hold
devices (PLATE/LEVER). A device in one route always controls the
same-colored barrier (DOOR/OBSTACLE) in the *other* route.

Difficulty levels:
  L1 - one route has a single barrier; the other holds the (unconditioned)
       controlling device. No dependency ordering.
  L2 - both routes have one barrier each, independently resolvable in any
       order (parallel dependency).
  L3 - a forced A -> B -> A chain: route `first` holds an unconditioned
       device1 that opens route `second`'s barrier1; only after passing
       barrier1 can route `second` reach device2, which opens route
       `first`'s barrier2.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .dynamics import Action, step_agent
from .objects import (
    DEFAULT_COLORS,
    BarrierSpec,
    DeviceSpec,
    Layout,
    ObjectType,
    Pos,
    RouteLayout,
)

Combo = Tuple[str, int]  # (ObjectType.name, color_idx) for DOOR/OBSTACLE barriers


def _device_kind_for_barrier(kind: ObjectType) -> ObjectType:
    return ObjectType.PLATE if kind == ObjectType.DOOR else ObjectType.LEVER


def _sample_combo(rng: np.random.Generator, K: int, excluded: Optional[Set[Combo]]) -> Tuple[ObjectType, int]:
    excluded = excluded or set()
    for _ in range(200):
        kind = ObjectType.DOOR if rng.random() < 0.5 else ObjectType.OBSTACLE
        color = int(rng.integers(0, K))
        if (kind.name, color) not in excluded:
            return kind, color
    raise RuntimeError("could not sample a (barrier, color) combo avoiding excluded_combos; K too small?")


def _sample_combos(
    rng: np.random.Generator,
    n: int,
    K: int,
    excluded: Optional[Set[Combo]],
    force: Optional[Combo],
) -> List[Tuple[ObjectType, int]]:
    combos = [_sample_combo(rng, K, excluded) for _ in range(n)]
    if force is not None:
        kind = ObjectType[force[0]]
        idx = int(rng.integers(0, n))
        combos[idx] = (kind, force[1])
    return combos


def _pick_branch_col(rng: np.random.Generator, col_lo: int, col_hi: int, used_cols: Sequence[int]) -> int:
    candidates = [c for c in range(col_lo, col_hi + 1) if c not in used_cols]
    if not candidates:
        raise RuntimeError("no free column available for a branch alcove")
    return int(rng.choice(candidates))


def _pick_branch_dir_len(rng: np.random.Generator, H: int, main_row: int, min_len: int, max_len: int) -> Tuple[int, int]:
    """Returns (direction, length) with direction in {-1 (up), +1 (down)}."""
    options = []
    for length in range(min_len, max_len + 1):
        if length <= main_row:
            options.append((-1, length))
        if length <= H - 1 - main_row:
            options.append((1, length))
    if not options:
        raise RuntimeError("grid too small to fit any branch alcove")
    idx = int(rng.integers(len(options)))
    return options[idx]


def build_route(
    H: int,
    W: int,
    main_row: int,
    barriers: Sequence[Tuple[int, ObjectType, int]],
    devices: Sequence[Tuple[int, ObjectType, int, int, int]],
) -> RouteLayout:
    """barriers: (col, kind, color). devices: (col, kind, color, branch_dir, branch_len)."""
    grid_type = np.full((H, W), int(ObjectType.WALL), dtype=np.int64)
    grid_color = np.full((H, W), -1, dtype=np.int64)
    grid_type[main_row, :] = int(ObjectType.FLOOR)

    start: Pos = (main_row, 0)
    exit_: Pos = (main_row, W - 1)
    grid_type[start] = int(ObjectType.START)
    grid_type[exit_] = int(ObjectType.EXIT)

    barrier_specs: List[BarrierSpec] = []
    for col, kind, color in barriers:
        grid_type[main_row, col] = int(kind)
        grid_color[main_row, col] = color
        barrier_specs.append(BarrierSpec(kind, color, (main_row, col)))

    device_specs: List[DeviceSpec] = []
    for col, kind, color, branch_dir, branch_len in devices:
        for i in range(1, branch_len + 1):
            r = main_row + branch_dir * i
            grid_type[r, col] = int(ObjectType.FLOOR)
        tip_r = main_row + branch_dir * branch_len
        grid_type[tip_r, col] = int(kind)
        grid_color[tip_r, col] = color
        device_specs.append(DeviceSpec(kind, color, (tip_r, col)))

    return RouteLayout(grid_type, grid_color, start, exit_, barrier_specs, device_specs)


def generate_layout(
    rng: np.random.Generator,
    level: str,
    K: int = 3,
    H: int = 7,
    W: int = 15,
    min_branch_len: int = 1,
    max_branch_len: int = 3,
    excluded_combos: Optional[Set[Combo]] = None,
    force_combo: Optional[Combo] = None,
    seed: Optional[int] = None,
) -> Layout:
    if W < 8:
        raise RuntimeError("W too small for layout generation (need >= 8)")
    main_row = H // 2
    col_lo, col_hi = 2, W - 3
    routes: Dict[str, RouteLayout] = {}

    if level == "L1":
        first = "a" if rng.random() < 0.5 else "b"
        second = "b" if first == "a" else "a"
        (kind, color), = _sample_combos(rng, 1, K, excluded_combos, force_combo)
        barrier_col = int(rng.integers(col_lo, col_hi + 1))
        dev_kind = _device_kind_for_barrier(kind)
        dev_col = _pick_branch_col(rng, 1, W - 2, [barrier_col])
        dev_dir, dev_len = _pick_branch_dir_len(rng, H, main_row, min_branch_len, max_branch_len)
        routes[first] = build_route(H, W, main_row, [(barrier_col, kind, color)], [])
        routes[second] = build_route(H, W, main_row, [], [(dev_col, dev_kind, color, dev_dir, dev_len)])

    elif level == "L2":
        (kind_a, color_a), (kind_b, color_b) = _sample_combos(rng, 2, K, excluded_combos, force_combo)
        col_a = int(rng.integers(col_lo, col_hi + 1))
        col_b = int(rng.integers(col_lo, col_hi + 1))
        # Each route's own device must sit *before* that same route's own barrier:
        # a DOOR-type barrier re-closes once the presser leaves, so a device
        # positioned behind it could become permanently unreachable once the
        # holder has already advanced past its own (now re-closed) barrier
        # before a view-gated request arrives (see DESIGN_NOTES.md).
        dcol_a = _pick_branch_col(rng, 1, col_a - 1, [])
        ddir_a, dlen_a = _pick_branch_dir_len(rng, H, main_row, min_branch_len, max_branch_len)
        dcol_b = _pick_branch_col(rng, 1, col_b - 1, [])
        ddir_b, dlen_b = _pick_branch_dir_len(rng, H, main_row, min_branch_len, max_branch_len)
        dev_kind_for_b_barrier = _device_kind_for_barrier(kind_b)  # lives in route a
        dev_kind_for_a_barrier = _device_kind_for_barrier(kind_a)  # lives in route b
        routes["a"] = build_route(
            H, W, main_row, [(col_a, kind_a, color_a)], [(dcol_a, dev_kind_for_b_barrier, color_b, ddir_a, dlen_a)]
        )
        routes["b"] = build_route(
            H, W, main_row, [(col_b, kind_b, color_b)], [(dcol_b, dev_kind_for_a_barrier, color_a, ddir_b, dlen_b)]
        )

    elif level == "L3":
        first = "a" if rng.random() < 0.5 else "b"
        second = "b" if first == "a" else "a"
        (kind1, color1), (kind2, color2) = _sample_combos(rng, 2, K, excluded_combos, force_combo)
        # device1 (in `first`, free) -> barrier1 (in `second`)
        # device2 (in `second`, gated behind barrier1) -> barrier2 (in `first`)
        mid = max(col_lo, W // 3 - 1)
        col_b1 = min(max(int(rng.integers(mid, mid + 2)), col_lo), col_hi - 1)
        col_d1 = int(rng.integers(1, max(2, col_b1 - 1)))
        col_b2 = int(rng.integers(col_d1 + 1, col_hi + 1))
        col_d2 = int(rng.integers(col_b1 + 1, col_hi + 1))

        dev1_kind = _device_kind_for_barrier(kind1)
        dev2_kind = _device_kind_for_barrier(kind2)
        d1_dir, d1_len = _pick_branch_dir_len(rng, H, main_row, min_branch_len, max_branch_len)
        d2_dir, d2_len = _pick_branch_dir_len(rng, H, main_row, min_branch_len, max_branch_len)

        routes[first] = build_route(
            H, W, main_row, [(col_b2, kind2, color2)], [(col_d1, dev1_kind, color1, d1_dir, d1_len)]
        )
        routes[second] = build_route(
            H, W, main_row, [(col_b1, kind1, color1)], [(col_d2, dev2_kind, color2, d2_dir, d2_len)]
        )
    else:
        raise ValueError(f"unknown level {level!r}")

    K_eff = K
    colors = DEFAULT_COLORS[:K_eff] if K_eff <= len(DEFAULT_COLORS) else [f"C{i}" for i in range(K_eff)]
    return Layout(
        H=H,
        W=W,
        colors=colors,
        route_a=routes["a"],
        route_b=routes["b"],
        level=level,
        seed=int(seed) if seed is not None else -1,
    )


def is_solvable(layout: Layout, max_expansions: int = 200_000) -> Tuple[bool, int]:
    """Joint-state BFS over (pos_a, pos_b, toggles_a, toggles_b). Both agents observe full
    joint state here (unlike the live env) -- this checks *existence* of a solution, not
    whether a partially-observed policy can find it."""
    route_a, route_b = layout.route_a, layout.route_b
    start_state = (route_a.start, route_b.start, frozenset(), frozenset())

    def is_goal(state) -> bool:
        return state[0] == route_a.exit and state[1] == route_b.exit

    if is_goal(start_state):
        return True, 0

    visited = {start_state}
    frontier = deque([start_state])
    actions = list(Action)
    expansions = 0
    while frontier:
        pos_a, pos_b, tog_a, tog_b = frontier.popleft()
        for act_a in actions:
            new_pos_a, new_tog_a, _ = step_agent(route_a, pos_a, tog_a, act_a, route_b, pos_b, tog_b)
            for act_b in actions:
                new_pos_b, new_tog_b, _ = step_agent(route_b, pos_b, tog_b, act_b, route_a, pos_a, tog_a)
                new_state = (new_pos_a, new_pos_b, new_tog_a, new_tog_b)
                if new_state in visited:
                    continue
                visited.add(new_state)
                if is_goal(new_state):
                    return True, expansions + 1
                frontier.append(new_state)
        expansions += 1
        if expansions > max_expansions:
            return False, expansions
    return False, expansions


class LayoutGenerator:
    """Split-aware sampler. `excluded_combos` is a set of (ObjectType.name, color_idx) tuples
    withheld from train/val/test_layout and force-included (one per layout) in test_combo."""

    SEED_OFFSETS = {"train": 0, "val": 10_000_000, "test_layout": 20_000_000, "test_combo": 30_000_000}

    def __init__(
        self,
        K: int = 3,
        H: int = 7,
        W: int = 15,
        min_branch_len: int = 1,
        max_branch_len: int = 3,
        excluded_combos: Optional[Sequence[Combo]] = None,
        base_seed: int = 0,
    ):
        self.K = K
        self.H = H
        self.W = W
        self.min_branch_len = min_branch_len
        self.max_branch_len = max_branch_len
        self.excluded_combos: Set[Combo] = set(excluded_combos or [])
        self.base_seed = base_seed

    def sample(self, split: str, level: str, index: int, max_tries: int = 50) -> Layout:
        if split not in self.SEED_OFFSETS:
            raise ValueError(f"unknown split {split!r}")
        offset = self.SEED_OFFSETS[split]
        excluded_list = sorted(self.excluded_combos)
        last_err: Optional[Exception] = None
        for attempt in range(max_tries):
            seed = self.base_seed + offset + index * 97 + attempt
            rng = np.random.default_rng(seed)
            excluded = None if split == "test_combo" else self.excluded_combos
            force = None
            if split == "test_combo":
                if not excluded_list:
                    raise ValueError("test_combo split requires excluded_combos to be configured")
                force = excluded_list[int(rng.integers(len(excluded_list)))]
            try:
                layout = generate_layout(
                    rng,
                    level,
                    K=self.K,
                    H=self.H,
                    W=self.W,
                    min_branch_len=self.min_branch_len,
                    max_branch_len=self.max_branch_len,
                    excluded_combos=excluded,
                    force_combo=force,
                    seed=seed,
                )
                ok, _ = is_solvable(layout)
                if ok:
                    layout.meta["attempt"] = attempt
                    return layout
            except RuntimeError as e:
                last_err = e
                continue
        raise RuntimeError(
            f"failed to generate a solvable layout for split={split} level={level} index={index} after {max_tries} tries ({last_err})"
        )
