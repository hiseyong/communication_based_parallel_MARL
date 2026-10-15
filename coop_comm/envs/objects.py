"""Grid object constants and the data structures describing a generated DualRouteEnv layout."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

Pos = Tuple[int, int]

DEFAULT_COLORS: List[str] = ["RED", "GREEN", "BLUE"]


class ObjectType(IntEnum):
    """Static cell type. Barrier/device colors are stored separately in `grid_color`."""

    WALL = 0
    FLOOR = 1
    START = 2
    EXIT = 3
    DOOR = 4
    OBSTACLE = 5
    PLATE = 6
    LEVER = 7


# Cell types that are always walkable regardless of dynamic barrier state.
ALWAYS_PASSABLE = {
    ObjectType.FLOOR,
    ObjectType.START,
    ObjectType.EXIT,
    ObjectType.PLATE,
    ObjectType.LEVER,
}
# Cell types whose passability depends on the *other* route's device state.
CONDITIONAL_BARRIER = {ObjectType.DOOR, ObjectType.OBSTACLE}


@dataclass(frozen=True)
class BarrierSpec:
    """A DOOR or OBSTACLE cell in a route, controlled by the opposite route's device."""

    kind: ObjectType  # ObjectType.DOOR or ObjectType.OBSTACLE
    color: int
    pos: Pos


@dataclass(frozen=True)
class DeviceSpec:
    """A PLATE or LEVER cell in a route, controlling the opposite route's barrier of the same color."""

    kind: ObjectType  # ObjectType.PLATE or ObjectType.LEVER
    color: int
    pos: Pos


@dataclass
class RouteLayout:
    """One agent's independent grid."""

    grid_type: np.ndarray  # (H, W) int array of ObjectType values
    grid_color: np.ndarray  # (H, W) int array; -1 where not colored
    start: Pos
    exit: Pos
    barriers: List[BarrierSpec]  # barriers IN this route (controlled by the other route's devices)
    devices: List[DeviceSpec]  # devices IN this route (controlling the other route's barriers)

    @property
    def H(self) -> int:
        return self.grid_type.shape[0]

    @property
    def W(self) -> int:
        return self.grid_type.shape[1]

    def plate_pos_by_color(self) -> Dict[int, Pos]:
        return {d.color: d.pos for d in self.devices if d.kind == ObjectType.PLATE}

    def lever_colors(self) -> List[int]:
        return [d.color for d in self.devices if d.kind == ObjectType.LEVER]

    def barrier_by_color(self) -> Dict[int, BarrierSpec]:
        return {b.color: b for b in self.barriers}

    def in_bounds(self, pos: Pos) -> bool:
        r, c = pos
        return 0 <= r < self.H and 0 <= c < self.W

    def cell(self, pos: Pos) -> Tuple[ObjectType, int]:
        r, c = pos
        return ObjectType(int(self.grid_type[r, c])), int(self.grid_color[r, c])


@dataclass
class Layout:
    H: int
    W: int
    colors: List[str]
    route_a: RouteLayout
    route_b: RouteLayout
    level: str
    seed: int
    meta: dict = field(default_factory=dict)

    @property
    def K(self) -> int:
        return len(self.colors)

    def route(self, label: str) -> RouteLayout:
        return self.route_a if label == "a" else self.route_b

    def other_label(self, label: str) -> str:
        return "b" if label == "a" else "a"
