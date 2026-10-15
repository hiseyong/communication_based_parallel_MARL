"""Ground-truth message policy and a BFS-driven scripted agent used to:
  (1) generate labeled (obs, message) data for stage-1 speaker training, and
  (2) act as the fixed "oracle partner" for stage-2 listener training.

oracle_message rule (single place this is decided, see module docstring):
  1. If there is a barrier in my own route still ahead of me (column-wise,
     i.e. not yet reached) *and within my sight* (`view_radius`, matching the
     egocentric observation window -- spec 3.8: "시야 내 첫 번째 닫힌 장벽")
     -> request the matching device: REQ_PLATE_<c> for a DOOR, REQ_LEVER_<c>
     for an OBSTACLE. A barrier I haven't walked close enough to see yet
     stays NONE -- the oracle must not leak global map knowledge the agent's
     observation doesn't contain (see DESIGN_NOTES.md "observation leak").
  2. Else if I am at or within `done_hold` columns past one of my barriers
     -> DONE (purely a function of current position, so no extra history
     needs to be tracked: "just passed" == "within done_hold columns of it").
  3. Else -> NONE.
ScriptedAgent reacts to an incoming REQ_* by BFS-pathing to the matching
device in its own route and holding a PLATE (until DONE) or toggling a
LEVER once; if that device is currently unreachable (e.g. gated behind its
own still-closed barrier, as in the L3 dependency chain) it falls back to
advancing toward its own exit instead of idling, so the dependency resolves
itself as soon as the partner's help opens the path.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..comm.vocab import DEVICES, Vocab
from .dynamics import MOVES, Action, N_ACTIONS, is_passable
from .objects import DeviceSpec, Layout, ObjectType, Pos, RouteLayout

DEFAULT_DONE_HOLD = 2


def oracle_message(
    route: RouteLayout,
    own_pos: Pos,
    other_route: RouteLayout,
    other_pos: Pos,
    other_toggles: frozenset,
    vocab: Vocab,
    done_hold: int = DEFAULT_DONE_HOLD,
    view_radius: Optional[int] = None,
) -> int:
    col = own_pos[1]
    ahead = sorted((b for b in route.barriers if col <= b.pos[1]), key=lambda b: b.pos[1])
    if ahead:
        nearest = ahead[0]
        in_sight = view_radius is None or (
            abs(own_pos[0] - nearest.pos[0]) <= view_radius and abs(own_pos[1] - nearest.pos[1]) <= view_radius
        )
        if col < nearest.pos[1] and in_sight:
            return vocab.req_lever(nearest.color) if nearest.kind == ObjectType.OBSTACLE else vocab.req_plate(nearest.color)
        # col == barrier col: just stepped onto an (open) barrier tile; fall through to DONE check.
    if any(b.pos[1] <= col <= b.pos[1] + done_hold for b in route.barriers):
        return vocab.done_idx
    return vocab.none_idx


def bfs_path(
    route: RouteLayout,
    start: Pos,
    goal: Pos,
    other_route: RouteLayout,
    other_pos: Pos,
    other_toggles: frozenset,
) -> Optional[List[Action]]:
    """Shortest move sequence (UP/DOWN/LEFT/RIGHT only) from start to goal under
    the *current* dynamic state, or None if goal is unreachable right now."""
    if start == goal:
        return []
    move_actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
    visited = {start}
    frontier = deque([start])
    came_from: Dict[Pos, Tuple[Pos, Action]] = {}
    while frontier:
        cur = frontier.popleft()
        for act in move_actions:
            dr, dc = MOVES[act]
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in visited or not route.in_bounds(nxt):
                continue
            if not is_passable(route, nxt, other_route, other_pos, other_toggles):
                continue
            visited.add(nxt)
            came_from[nxt] = (cur, act)
            if nxt == goal:
                path = [act]
                node = cur
                while node != start:
                    node, a2 = came_from[node]
                    path.append(a2)
                path.reverse()
                return path
            frontier.append(nxt)
    return None


def bfs_path_towards(
    route: RouteLayout,
    start: Pos,
    goal: Pos,
    other_route: RouteLayout,
    other_pos: Pos,
    other_toggles: frozenset,
) -> List[Action]:
    """Like `bfs_path`, but if `goal` is currently unreachable (e.g. gated
    behind a still-closed barrier), walks as far as currently possible
    instead of giving up -- otherwise, under a `view_radius`-gated oracle, an
    agent waiting at its own closed barrier would never approach close enough
    to bring it into view and request help (a real deadlock caught by
    test_env.py's clear-rate test; see DESIGN_NOTES.md)."""
    if start == goal:
        return []
    move_actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
    visited = {start}
    frontier = deque([start])
    came_from: Dict[Pos, Tuple[Pos, Action]] = {}
    best, best_dist = start, abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    while frontier:
        cur = frontier.popleft()
        for act in move_actions:
            dr, dc = MOVES[act]
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in visited or not route.in_bounds(nxt):
                continue
            if not is_passable(route, nxt, other_route, other_pos, other_toggles):
                continue
            visited.add(nxt)
            came_from[nxt] = (cur, act)
            dist = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
            if dist < best_dist:
                best, best_dist = nxt, dist
            if nxt == goal:
                best, best_dist = nxt, 0
                frontier.clear()
                break
            frontier.append(nxt)
    if best == start:
        return []
    path = []
    node = best
    while node != start:
        node, act = came_from[node]
        path.append(act)
    path.reverse()
    return path


class ScriptedAgent:
    """BFS-based baseline policy for one route. `epsilon` mixes in uniform-random
    *movement* actions for data diversity; the emitted message always follows
    `oracle_message` (stage-1 training needs correct labels regardless of how
    noisy the underlying trajectory is)."""

    def __init__(
        self,
        label: str,
        layout: Layout,
        epsilon: float = 0.0,
        rng: Optional[np.random.Generator] = None,
        done_hold: int = DEFAULT_DONE_HOLD,
        view_radius: Optional[int] = None,
    ):
        self.label = label
        self.other_label = layout.other_label(label)
        self.route = layout.route(label)
        self.other_route = layout.route(self.other_label)
        self.epsilon = epsilon
        self.rng = rng if rng is not None else np.random.default_rng()
        self.done_hold = done_hold
        self.view_radius = view_radius
        self._target: Optional[DeviceSpec] = None

    def reset(self) -> None:
        self._target = None

    def act(self, own_pos: Pos, own_toggles: frozenset, other_pos: Pos, other_toggles: frozenset, msg_in: int, vocab: Vocab) -> Tuple[Action, int]:
        msg_out = oracle_message(self.route, own_pos, self.other_route, other_pos, other_toggles, vocab, self.done_hold, self.view_radius)

        if msg_in == vocab.done_idx:
            self._target = None
        elif msg_in not in (vocab.none_idx, vocab.wait_idx):
            _, device_idx, color = vocab.to_slot(msg_in)
            want_kind = ObjectType.PLATE if DEVICES[device_idx] == "PLATE" else ObjectType.LEVER
            dev = next((d for d in self.route.devices if d.kind == want_kind and d.color == color), None)
            if dev is not None and not (dev.kind == ObjectType.LEVER and dev.color in own_toggles):
                self._target = dev

        if self._target is not None and self._target.kind == ObjectType.LEVER and self._target.color in own_toggles:
            self._target = None  # already toggled; one-shot device needs no further action

        if self._target is not None:
            action = self._act_toward(own_pos, self._target, other_pos, other_toggles)
        else:
            action = self._advance(own_pos, other_pos, other_toggles)

        if self.rng.random() < self.epsilon:
            action = Action(int(self.rng.integers(N_ACTIONS)))
        return action, msg_out

    def _act_toward(self, own_pos: Pos, device: DeviceSpec, other_pos: Pos, other_toggles: frozenset) -> Action:
        if own_pos == device.pos:
            return Action.INTERACT if device.kind == ObjectType.LEVER else Action.STAY
        path = bfs_path(self.route, own_pos, device.pos, self.other_route, other_pos, other_toggles)
        if not path:
            return self._advance(own_pos, other_pos, other_toggles)
        return path[0]

    def _still_possibly_needed(self, other_pos: Pos) -> bool:
        """True if one of my own devices controls a partner barrier the partner
        hasn't passed yet -- used only to avoid finalizing at my exit (a
        permanent STAY per the env's rules) while a late, correctly
        view-gated request might still be on its way. Uses privileged state
        (other_pos) for *patience* only, never to trigger the help action
        itself -- that stays strictly message-gated, so cutting messages
        still collapses success (see tests/test_env.py)."""
        for dev in self.route.devices:
            barrier = next((b for b in self.other_route.barriers if b.color == dev.color), None)
            if barrier is not None and other_pos[1] < barrier.pos[1]:
                return True
        return False

    def _crosses_own_barrier(self, own_pos: Pos, next_pos: Pos) -> bool:
        """True if taking this step would move from at-or-before one of my own
        barriers to beyond it. My own devices are always placed before my own
        barrier (layout_generator.py), but a DOOR-type barrier re-closes once
        its presser leaves -- so crossing it forecloses ever going back for an
        unconsumed device on the near side. Combined with `_still_possibly_needed`,
        this holds the agent at its own (currently open) barrier instead of
        stranding itself on the far side."""
        for b in self.route.barriers:
            if own_pos[1] <= b.pos[1] < next_pos[1]:
                return True
        return False

    def _advance(self, own_pos: Pos, other_pos: Pos, other_toggles: frozenset) -> Action:
        goal = self.route.exit
        if own_pos == goal:
            return Action.STAY
        path = bfs_path_towards(self.route, own_pos, goal, self.other_route, other_pos, other_toggles)
        if not path:
            return Action.STAY
        dr, dc = MOVES[path[0]]
        next_pos = (own_pos[0] + dr, own_pos[1] + dc)
        if self._still_possibly_needed(other_pos) and (next_pos == goal or self._crosses_own_barrier(own_pos, next_pos)):
            return Action.STAY
        return path[0]
