"""Stage 3 prep, filled in: plug a trained Speaker into the message head and a
trained Listener into the move head of BOTH agents (one shared pair of
weights, used symmetrically -- see DESIGN_NOTES.md "parameter sharing"), and
run pure self-play with no ScriptedAgent on either side. This is the actual
"can the separately-learned language components solve the real task"
check -- no additional joint fine-tuning happens here, only composition.

Reports self-play success rate against three reference points: the
ScriptedAgent/ScriptedAgent oracle ceiling, Speaker+ScriptedAgent-listener
(eval_speaker's online eval), and ScriptedAgent-speaker+Listener
(eval_listener). The gap between self-play and the ceiling is the answer to
"does composing the two separately-trained modules work."
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import torch

from ..comm.vocab import Vocab
from ..config_utils import load_config
from ..envs.dual_route_env import AGENT_OF, DualRouteEnv
from ..envs.layout_generator import LayoutGenerator
from ..envs.oracle import ScriptedAgent
from ..models.factory import build_listener, build_speaker
from ..train.collect_speaker_data import _parse_excluded_combos

DEFAULT_CONFIG = dict(
    speaker_checkpoint="runs/speaker_exp1/0/speaker.pt",
    listener_checkpoint="runs/listener_exp1/0/listener.pt",
    out_dir="runs/eval_pair",
    splits="val,test_layout,test_combo",
    levels="L1,L2,L3",
    n_episodes=300,
    excluded_combos="OBSTACLE:BLUE",
    K=3,
    H=7,
    W=15,
    V=7,
    max_steps=200,
    device="cpu",
)


class CommAgent:
    """obs -> (message, move), composing a trained Speaker (message head) and
    a trained Listener (move head). The message is computed from obs *without*
    msg_in (the Speaker never sees it); the move uses the full obs including
    the msg_in the env delivered from the partner's previous message."""

    def __init__(self, speaker, listener, device: torch.device):
        self.speaker = speaker
        self.listener = listener
        self.device = device
        self.hidden: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.hidden = self.listener.initial_hidden(1, self.device) if self.listener.use_gru else None

    @torch.no_grad()
    def act(self, obs: Dict[str, np.ndarray]):
        g = torch.from_numpy(obs["grid"]).float().unsqueeze(0).to(self.device)
        p = torch.from_numpy(obs["pos"]).float().unsqueeze(0).to(self.device)
        s = torch.from_numpy(obs["own_status"]).float().unsqueeze(0).to(self.device)
        m = torch.from_numpy(obs["msg_in"]).float().unsqueeze(0).to(self.device)
        msg = int(self.speaker.act(g, p, s).item())
        logits, _, self.hidden = self.listener.forward_step(g, p, s, m, self.hidden)
        move = int(logits.argmax(-1).item())
        return move, msg


def load_pair(speaker_ckpt: str, listener_ckpt: str, device: str):
    sckpt = torch.load(speaker_ckpt, map_location=device)
    scfg = sckpt["config"]
    speaker = build_speaker(scfg["K"], scfg["V"], cnn_channels=scfg["cnn_channels"], hidden=scfg["hidden"], gumbel=scfg.get("gumbel", False))
    speaker.load_state_dict(sckpt["model_state"])
    speaker.to(device).eval()

    lckpt = torch.load(listener_ckpt, map_location=device)
    lcfg = lckpt["config"]
    listener = build_listener(lcfg["K"], lcfg["V"], use_gru=lcfg.get("use_gru", False))
    listener.load_state_dict(lckpt["model_state"])
    listener.to(device).eval()

    assert scfg["K"] == lcfg["K"], "speaker/listener K mismatch"
    return speaker, listener, scfg["K"]


def run_self_play_episode(env: DualRouteEnv, agent_a: CommAgent, agent_b: CommAgent, max_steps: int, seed: int, layout_index: int, level: str, split: str):
    obs, infos = env.reset(seed=seed, options={"layout_index": layout_index, "level": level, "split": split})
    agent_a.reset()
    agent_b.reset()
    for t in range(max_steps):
        move_a, msg_a = agent_a.act(obs["agent_0"])
        move_b, msg_b = agent_b.act(obs["agent_1"])
        actions = {"agent_0": np.array([move_a, msg_a]), "agent_1": np.array([move_b, msg_b])}
        obs, rewards, terms, truncs, infos = env.step(actions)
        if all(terms.values()) or all(truncs.values()):
            return all(terms.values()), t + 1
    return False, max_steps


def run_scripted_baseline(env: DualRouteEnv, max_steps: int, seed: int, layout_index: int, level: str, split: str, view_radius: int):
    obs, infos = env.reset(seed=seed, options={"layout_index": layout_index, "level": level, "split": split})
    agents = {
        "a": ScriptedAgent("a", env._layout, epsilon=0.0, rng=np.random.default_rng(seed), view_radius=view_radius),
        "b": ScriptedAgent("b", env._layout, epsilon=0.0, rng=np.random.default_rng(seed + 1), view_radius=view_radius),
    }
    for t in range(max_steps):
        actions = {}
        for label in ("a", "b"):
            other = "b" if label == "a" else "a"
            act, msg = agents[label].act(env._pos[label], env._toggles[label], env._pos[other], env._toggles[other], env._msg_in[label], env.vocab)
            actions[AGENT_OF[label]] = np.array([int(act), msg])
        obs, rewards, terms, truncs, infos = env.step(actions)
        if all(terms.values()) or all(truncs.values()):
            return all(terms.values()), t + 1
    return False, max_steps


def run(cfg: Dict) -> Dict:
    device = torch.device(cfg["device"])
    speaker, listener, K = load_pair(cfg["speaker_checkpoint"], cfg["listener_checkpoint"], cfg["device"])
    agent_a = CommAgent(speaker, listener, device)
    agent_b = CommAgent(speaker, listener, device)

    excluded = _parse_excluded_combos(cfg["excluded_combos"], cfg["K"]) if cfg["excluded_combos"] else None
    gen = LayoutGenerator(K=cfg["K"], H=cfg["H"], W=cfg["W"], excluded_combos=excluded)
    env = DualRouteEnv(H=cfg["H"], W=cfg["W"], K=cfg["K"], V=cfg["V"], max_steps=cfg["max_steps"], layout_generator=gen)

    os.makedirs(cfg["out_dir"], exist_ok=True)
    splits = cfg["splits"].split(",") if isinstance(cfg["splits"], str) else cfg["splits"]
    levels = cfg["levels"].split(",") if isinstance(cfg["levels"], str) else cfg["levels"]

    results = {}
    for split in splits:
        for level in levels:
            self_play, scripted = [], []
            for ep in range(cfg["n_episodes"]):
                ok, steps = run_self_play_episode(env, agent_a, agent_b, cfg["max_steps"], seed=ep, layout_index=ep, level=level, split=split)
                self_play.append((ok, steps))
                ok2, steps2 = run_scripted_baseline(env, cfg["max_steps"], seed=ep, layout_index=ep, level=level, split=split, view_radius=cfg["V"] // 2)
                scripted.append((ok2, steps2))
            sp_rate = float(np.mean([o for o, _ in self_play]))
            sc_rate = float(np.mean([o for o, _ in scripted]))
            results.setdefault(split, {})[level] = {
                "self_play_success_rate": sp_rate,
                "self_play_avg_steps": float(np.mean([s for _, s in self_play])),
                "scripted_ceiling_success_rate": sc_rate,
                "gap_to_ceiling": sc_rate - sp_rate,
            }
            print(f"[{split}/{level}] self-play(Speaker+Listener both sides)={sp_rate:.3f}  scripted ceiling={sc_rate:.3f}  gap={sc_rate - sp_rate:.3f}")

    with open(os.path.join(cfg["out_dir"], "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    cfg = load_config(DEFAULT_CONFIG, description="stage-3 check: self-play with trained Speaker+Listener on both agents")
    run(cfg)
