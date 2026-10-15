"""Single source of truth for movement/interaction dynamics.

Shared by the live env (dual_route_env.py), the layout solvability BFS
(layout_generator.py) and the scripted oracle (oracle.py) so that "solvable
on paper" always matches "solvable in the actual env".
"""
from __future__ import annotations

from enum import IntEnum
from typing import FrozenSet, Tuple

from .objects import ALWAYS_PASSABLE, ObjectType, Pos, RouteLayout

ToggleSet = FrozenSet[int]


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    INTERACT = 5


N_ACTIONS = len(Action)

MOVES = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
    Action.STAY: (0, 0),
    Action.INTERACT: (0, 0),
}


def is_passable(route: RouteLayout, pos: Pos, other_route: RouteLayout, other_pos: Pos, other_toggles: ToggleSet) -> bool:
    """Whether `pos` in `route` can be entered, given the other route's live state."""
    cell_type, cell_color = route.cell(pos)
    if cell_type in ALWAYS_PASSABLE:
        return True
    if cell_type == ObjectType.DOOR:
        return other_route.plate_pos_by_color().get(cell_color) == other_pos
    if cell_type == ObjectType.OBSTACLE:
        return cell_color in other_toggles
    return False  # WALL


def step_agent(
    route: RouteLayout,
    pos: Pos,
    own_toggles: ToggleSet,
    action: Action,
    other_route: RouteLayout,
    other_pos: Pos,
    other_toggles: ToggleSet,
) -> Tuple[Pos, ToggleSet, bool]:
    """Apply one action for a single agent in its own route.

    Returns (new_pos, new_own_toggles, moved) where `moved` is True iff the
    agent's position changed (used for "first barrier pass" bookkeeping).
    """
    if action == Action.INTERACT:
        cell_type, cell_color = route.cell(pos)
        if cell_type == ObjectType.LEVER:
            if cell_color in own_toggles:
                own_toggles = own_toggles - {cell_color}
            else:
                own_toggles = own_toggles | {cell_color}
        return pos, own_toggles, False

    dr, dc = MOVES[action]
    if (dr, dc) == (0, 0):
        return pos, own_toggles, False
    new_pos = (pos[0] + dr, pos[1] + dc)
    if not route.in_bounds(new_pos):
        return pos, own_toggles, False
    if not is_passable(route, new_pos, other_route, other_pos, other_toggles):
        return pos, own_toggles, False
    return new_pos, own_toggles, True
