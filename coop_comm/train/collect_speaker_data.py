"""Stage 1 data collection: roll out ScriptedAgent(epsilon) pairs and dump
(obs, oracle_message_label) pairs for every step/agent to a .npz file.

Usage:
    python -m coop_comm.train.collect_speaker_data --config coop_comm/configs/speaker.yaml --split train
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List

import numpy as np

from ..comm.vocab import Vocab
from ..config_utils import load_config
from ..envs.dual_route_env import AGENT_OF, DualRouteEnv
from ..envs.layout_generator import LayoutGenerator
from ..envs.objects import ObjectType
from ..envs.oracle import ScriptedAgent

DEFAULT_CONFIG = dict(
    out_dir="data/speaker",
    split="train",
    levels="L1,L2,L3",
    episodes_per_level=500,
    epsilon_min=0.0,
    epsilon_max=0.3,
    K=3,
    H=7,
    W=15,
    V=7,
    max_steps=200,
    excluded_combos="OBSTACLE:BLUE",
    seed=0,
)


def _parse_excluded_combos(spec: str, K: int):
    from ..envs.objects import DEFAULT_COLORS

    colors = DEFAULT_COLORS[:K] if K <= len(DEFAULT_COLORS) else [f"C{i}" for i in range(K)]
    out = set()
    for item in [s for s in spec.split(",") if s]:
        kind, color = item.split(":")
        color = color.strip()
        color_idx = int(color) if color.isdigit() else colors.index(color.strip().upper())
        out.add((kind.strip().upper(), color_idx))
    return out


def collect(cfg: Dict) -> str:
    levels = cfg["levels"] if isinstance(cfg["levels"], list) else cfg["levels"].split(",")
    excluded = _parse_excluded_combos(cfg["excluded_combos"], cfg["K"]) if cfg["excluded_combos"] else None
    gen = LayoutGenerator(K=cfg["K"], H=cfg["H"], W=cfg["W"], excluded_combos=excluded, base_seed=cfg["seed"])
    env = DualRouteEnv(H=cfg["H"], W=cfg["W"], K=cfg["K"], V=cfg["V"], max_steps=cfg["max_steps"], layout_generator=gen, split=cfg["split"])
    vocab: Vocab = env.vocab
    rng = np.random.default_rng(cfg["seed"])

    grids: List[np.ndarray] = []
    poss: List[np.ndarray] = []
    statuses: List[np.ndarray] = []
    labels: List[int] = []
    level_ids: List[int] = []
    label_counts = Counter()

    for level_idx, level in enumerate(levels):
        for ep in range(cfg["episodes_per_level"]):
            obs, infos = env.reset(options={"layout_index": ep, "level": level, "split": cfg["split"]})
            eps_a = float(rng.uniform(cfg["epsilon_min"], cfg["epsilon_max"]))
            eps_b = float(rng.uniform(cfg["epsilon_min"], cfg["epsilon_max"]))
            agents = {
                "a": ScriptedAgent("a", env._layout, epsilon=eps_a, rng=rng, view_radius=cfg["V"] // 2),
                "b": ScriptedAgent("b", env._layout, epsilon=eps_b, rng=rng, view_radius=cfg["V"] // 2),
            }
            for t in range(cfg["max_steps"]):
                actions = {}
                for label_lbl in ("a", "b"):
                    other = "b" if label_lbl == "a" else "a"
                    act, msg = agents[label_lbl].act(
                        env._pos[label_lbl], env._toggles[label_lbl], env._pos[other], env._toggles[other],
                        env._msg_in[label_lbl], vocab,
                    )
                    o = obs[AGENT_OF[label_lbl]]
                    grids.append(o["grid"])
                    poss.append(o["pos"])
                    statuses.append(o["own_status"])
                    labels.append(msg)
                    level_ids.append(level_idx)
                    label_counts[vocab.decode(msg)] += 1
                    actions[AGENT_OF[label_lbl]] = np.array([int(act), msg])
                obs, rewards, terms, truncs, infos = env.step(actions)
                if all(terms.values()) or all(truncs.values()):
                    break

    os.makedirs(cfg["out_dir"], exist_ok=True)
    out_path = os.path.join(cfg["out_dir"], f"{cfg['split']}.npz")
    np.savez_compressed(
        out_path,
        grid=np.stack(grids).astype(np.float32),
        pos=np.stack(poss).astype(np.float32),
        own_status=np.stack(statuses).astype(np.float32),
        label=np.array(labels, dtype=np.int64),
        level_id=np.array(level_ids, dtype=np.int64),
        levels=np.array(levels),
        vocab_tokens=np.array(vocab.tokens),
    )

    report = {
        "n_samples": len(labels),
        "class_counts": dict(label_counts),
        "class_fractions": {k: v / len(labels) for k, v in label_counts.items()},
    }
    with open(os.path.join(cfg["out_dir"], f"{cfg['split']}_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"saved {len(labels)} samples -> {out_path}")
    print("class distribution:", report["class_fractions"])
    return out_path


if __name__ == "__main__":
    cfg = load_config(DEFAULT_CONFIG, description="collect speaker training data")
    collect(cfg)
