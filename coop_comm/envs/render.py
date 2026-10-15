"""ASCII rendering (debugging / tests) and a matplotlib-free RGB + GIF renderer
(used by eval/ to save example episodes)."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from ..comm.vocab import Vocab
from .objects import Layout, ObjectType, Pos, RouteLayout

CELL_PX = 20
GAP_PX = 10

_PALETTE = [(214, 64, 64), (60, 170, 90), (70, 100, 215), (210, 170, 40), (160, 80, 190), (60, 180, 180)]


def _base_color(color_idx: int) -> tuple:
    return _PALETTE[color_idx % len(_PALETTE)]


def _ascii_cell(route: RouteLayout, pos: Pos, other_route: RouteLayout, other_pos: Pos, other_toggles: frozenset) -> str:
    ctype, ccolor = route.cell(pos)
    if ctype == ObjectType.WALL:
        return "##"
    if ctype == ObjectType.FLOOR:
        return ".."
    if ctype == ObjectType.START:
        return "S."
    if ctype == ObjectType.EXIT:
        return "E."
    if ctype == ObjectType.DOOR:
        is_open = other_route.plate_pos_by_color().get(ccolor) == other_pos
        return ("d" if is_open else "D") + str(ccolor)
    if ctype == ObjectType.OBSTACLE:
        is_open = ccolor in other_toggles
        return ("o" if is_open else "O") + str(ccolor)
    if ctype == ObjectType.PLATE:
        return "P" + str(ccolor)
    if ctype == ObjectType.LEVER:
        return "L" + str(ccolor)
    return "??"


def _route_ascii_lines(route: RouteLayout, other_route: RouteLayout, own_pos: Pos, other_pos: Pos, other_toggles: frozenset) -> List[str]:
    lines = []
    for r in range(route.H):
        chars = []
        for c in range(route.W):
            if (r, c) == own_pos:
                chars.append("@@")
            else:
                chars.append(_ascii_cell(route, (r, c), other_route, other_pos, other_toggles))
        lines.append("".join(chars))
    return lines


def render_ascii(
    layout: Layout,
    pos: Dict[str, Pos],
    toggles: Dict[str, frozenset],
    exited: Dict[str, bool],
    msg_in: Dict[str, int],
    vocab: Vocab,
) -> str:
    lines_a = _route_ascii_lines(layout.route_a, layout.route_b, pos["a"], pos["b"], toggles["b"])
    lines_b = _route_ascii_lines(layout.route_b, layout.route_a, pos["b"], pos["a"], toggles["a"])
    width_a = layout.route_a.W * 2
    out = [f"{'Route A (agent_0)':<{width_a}}    Route B (agent_1)"]
    for la, lb in zip(lines_a, lines_b):
        out.append(f"{la:<{width_a}}    {lb}")
    out.append("")
    out.append(f"agent_0: msg_in={vocab.decode(msg_in['a']):<14} exited={exited['a']}")
    out.append(f"agent_1: msg_in={vocab.decode(msg_in['b']):<14} exited={exited['b']}")
    return "\n".join(out)


def _cell_rgb(ctype: ObjectType, ccolor: int, is_open: bool) -> tuple:
    if ctype == ObjectType.WALL:
        return (25, 25, 25)
    if ctype == ObjectType.FLOOR:
        return (238, 238, 238)
    if ctype == ObjectType.START:
        return (200, 210, 255)
    if ctype == ObjectType.EXIT:
        return (255, 210, 60)
    base = _base_color(ccolor)
    if ctype in (ObjectType.DOOR, ObjectType.OBSTACLE):
        if is_open:
            return tuple(int(0.5 * b + 0.5 * 255) for b in base)
        return tuple(int(0.6 * b) for b in base)
    if ctype in (ObjectType.PLATE, ObjectType.LEVER):
        return tuple(int(0.5 * b + 0.5 * 255) for b in base)
    return (255, 0, 255)


def _render_route_rgb(route: RouteLayout, other_route: RouteLayout, own_pos: Pos, other_pos: Pos, other_toggles: frozenset) -> np.ndarray:
    H, W = route.H, route.W
    img = np.zeros((H * CELL_PX, W * CELL_PX, 3), dtype=np.uint8)
    plate_pos_other = other_route.plate_pos_by_color()
    for r in range(H):
        for c in range(W):
            ctype, ccolor = route.cell((r, c))
            is_open = False
            if ctype == ObjectType.DOOR:
                is_open = plate_pos_other.get(ccolor) == other_pos
            elif ctype == ObjectType.OBSTACLE:
                is_open = ccolor in other_toggles
            rgb = _cell_rgb(ctype, ccolor, is_open)
            img[r * CELL_PX : (r + 1) * CELL_PX, c * CELL_PX : (c + 1) * CELL_PX] = rgb
    ar, ac = own_pos
    pad = 4
    img[ar * CELL_PX + pad : (ar + 1) * CELL_PX - pad, ac * CELL_PX + pad : (ac + 1) * CELL_PX - pad] = (20, 20, 20)
    return img


def render_frame_rgb(layout: Layout, pos: Dict[str, Pos], toggles: Dict[str, frozenset]) -> np.ndarray:
    img_a = _render_route_rgb(layout.route_a, layout.route_b, pos["a"], pos["b"], toggles["b"])
    img_b = _render_route_rgb(layout.route_b, layout.route_a, pos["b"], pos["a"], toggles["a"])
    H = max(img_a.shape[0], img_b.shape[0])

    def pad_h(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == H:
            return img
        padded = np.full((H, img.shape[1], 3), 255, dtype=np.uint8)
        padded[: img.shape[0]] = img
        return padded

    gap = np.full((H, GAP_PX, 3), 255, dtype=np.uint8)
    return np.concatenate([pad_h(img_a), gap, pad_h(img_b)], axis=1)


def save_episode_gif(frames: Sequence[np.ndarray], path: str, fps: int = 4) -> None:
    from PIL import Image

    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
