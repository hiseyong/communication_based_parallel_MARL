"""Stage 2 training: PPO for one agent (the "listener"), with the partner fixed
to a ScriptedAgent emitting oracle messages (epsilon=0) and an L1->L2->L3
curriculum that advances once the rolling success rate crosses a threshold.

Note: the curriculum's rolling success rate is measured over *stochastic*
rollouts (PPO needs sampling for exploration), which can look better than the
policy's actual (greedy/deployed) competence while entropy is still high --
see `quick_greedy_eval` below and DESIGN_NOTES.md ("selfish solo-exit" risk).
"""
from __future__ import annotations

import csv
import os
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch

from ..comm.vocab import Vocab
from ..config_utils import load_config
from ..envs.dual_route_env import AGENT_OF, DualRouteEnv
from ..envs.layout_generator import LayoutGenerator
from ..envs.oracle import ScriptedAgent, oracle_message
from ..models.factory import build_listener
from .collect_speaker_data import _parse_excluded_combos
from .ppo import PPOConfig, train_ppo

LEVELS = ["L1", "L2", "L3"]

DEFAULT_CONFIG = dict(
    out_dir="runs/listener_exp1/0",
    seed=0,
    K=3,
    H=7,
    W=15,
    V=7,
    max_steps=200,
    excluded_combos="OBSTACLE:BLUE",
    split="train",
    own_send_mode="oracle",  # "none" | "oracle" -- stage 2 doesn't train the message head
    channel_kind="identity",
    channel_p=0.0,
    use_gru=False,
    curriculum_threshold=0.8,
    curriculum_window=50,
    total_timesteps=200_000,
    n_envs=8,
    n_steps=128,
    update_epochs=4,
    num_minibatches=4,
    lr=2.5e-4,
    ent_coef=0.01,
    vf_coef=0.5,
    device="cpu",
    ckpt_every_updates=50,
    greedy_eval_episodes=20,
)


class SingleAgentListenerEnv:
    """Gym-like single-agent view of DualRouteEnv for `label`, with the other
    agent always a fixed ScriptedAgent (oracle messages, epsilon=0)."""

    def __init__(self, label: str, layout_generator: LayoutGenerator, level: str, split: str, cfg: Dict):
        self.label = label
        self.other_label = "b" if label == "a" else "a"
        self.level = level
        self.split = split
        self.own_send_mode = cfg["own_send_mode"]
        self.view_radius = cfg["V"] // 2
        self.env = DualRouteEnv(
            H=cfg["H"], W=cfg["W"], K=cfg["K"], V=cfg["V"], max_steps=cfg["max_steps"],
            layout_generator=layout_generator, level=level, split=split,
            channel_kind=cfg["channel_kind"], channel_p=cfg["channel_p"],
        )
        self.vocab: Vocab = self.env.vocab
        self.partner: Optional[ScriptedAgent] = None
        self._rng = np.random.default_rng(cfg["seed"])

    def set_level(self, level: str) -> None:
        self.level = level

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        opts = dict(options or {})
        opts.setdefault("level", self.level)
        opts.setdefault("split", self.split)
        obs, infos = self.env.reset(seed=seed, options=opts)
        self.partner = ScriptedAgent(self.other_label, self.env._layout, epsilon=0.0, rng=self._rng, view_radius=self.view_radius)
        return obs[AGENT_OF[self.label]]

    def step(self, action_move: int):
        own, other = self.label, self.other_label
        partner_act, partner_msg = self.partner.act(
            self.env._pos[other], self.env._toggles[other], self.env._pos[own], self.env._toggles[own],
            self.env._msg_in[other], self.vocab,
        )
        if self.own_send_mode == "oracle":
            own_route, other_route = self.env._layout.route(own), self.env._layout.route(other)
            own_msg = oracle_message(own_route, self.env._pos[own], other_route, self.env._pos[other], self.env._toggles[other], self.vocab, view_radius=self.view_radius)
        else:
            own_msg = self.vocab.none_idx
        actions = {
            AGENT_OF[own]: np.array([action_move, own_msg]),
            AGENT_OF[other]: np.array([int(partner_act), partner_msg]),
        }
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        return obs[AGENT_OF[own]], rewards[AGENT_OF[own]], terms[AGENT_OF[own]], truncs[AGENT_OF[own]], infos[AGENT_OF[own]]


class Curriculum:
    def __init__(self, envs: List[SingleAgentListenerEnv], threshold: float, window: int, out_dir: str):
        self.envs = envs
        self.threshold = threshold
        self.window = window
        self.level_idx = 0
        self.history = deque(maxlen=window)
        self.log: List[dict] = []
        self.out_dir = out_dir
        self.global_step = 0

    def on_episode_end(self, env_idx: int, success: bool, info: dict) -> None:
        self.history.append(1.0 if success else 0.0)
        if len(self.history) >= self.window and np.mean(self.history) >= self.threshold and self.level_idx < len(LEVELS) - 1:
            self.level_idx += 1
            for env in self.envs:
                env.set_level(LEVELS[self.level_idx])
            self.log.append({"advanced_to": LEVELS[self.level_idx], "global_step": self.global_step})
            print(f"curriculum: advancing to {LEVELS[self.level_idx]} at step {self.global_step}")
            self.history.clear()

    def current_level(self) -> str:
        return LEVELS[self.level_idx]


@torch.no_grad()
def quick_greedy_eval(model, gen: LayoutGenerator, level: str, split: str, cfg: Dict, n_episodes: int) -> float:
    """Greedy (argmax) success rate -- unlike the training-time rolling success
    rate (measured over exploratory *stochastic* rollouts), this reflects the
    policy actually deployed at eval/inference time."""
    model.eval()
    device = torch.device(cfg["device"])
    successes = 0
    for ep in range(n_episodes):
        env = SingleAgentListenerEnv("a", gen, level, split, cfg)
        obs = env.reset(seed=10_000 + ep, options={"layout_index": ep})
        hidden = model.initial_hidden(1, device) if model.use_gru else None
        for t in range(cfg["max_steps"]):
            g = torch.from_numpy(obs["grid"]).float().unsqueeze(0).to(device)
            p = torch.from_numpy(obs["pos"]).float().unsqueeze(0).to(device)
            s = torch.from_numpy(obs["own_status"]).float().unsqueeze(0).to(device)
            m = torch.from_numpy(obs["msg_in"]).float().unsqueeze(0).to(device)
            logits, _, hidden = model.forward_step(g, p, s, m, hidden)
            action = int(logits.argmax(-1).item())
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                successes += int(term)
                break
    model.train()
    return successes / n_episodes


def train(cfg: Dict) -> str:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)

    excluded = _parse_excluded_combos(cfg["excluded_combos"], cfg["K"]) if cfg["excluded_combos"] else None
    gen = LayoutGenerator(K=cfg["K"], H=cfg["H"], W=cfg["W"], excluded_combos=excluded, base_seed=cfg["seed"])

    label = "a"
    envs = [SingleAgentListenerEnv(label, gen, LEVELS[0], cfg["split"], cfg) for _ in range(cfg["n_envs"])]
    curriculum = Curriculum(envs, cfg["curriculum_threshold"], cfg["curriculum_window"], cfg["out_dir"])

    model = build_listener(cfg["K"], cfg["V"], use_gru=cfg["use_gru"])

    try:
        from torch.utils.tensorboard import SummaryWriter

        tb = SummaryWriter(cfg["out_dir"])
    except Exception:
        tb = None

    ppo_cfg = PPOConfig(
        total_timesteps=cfg["total_timesteps"], n_envs=cfg["n_envs"], n_steps=cfg["n_steps"],
        update_epochs=cfg["update_epochs"], num_minibatches=cfg["num_minibatches"], lr=cfg["lr"],
        ent_coef=cfg["ent_coef"], vf_coef=cfg["vf_coef"], device=cfg["device"], seed=cfg["seed"],
    )

    def on_episode_end(env_idx, success, info, global_step):
        curriculum.global_step = global_step
        curriculum.on_episode_end(env_idx, success, info)

    greedy_csv = os.path.join(cfg["out_dir"], "greedy_eval.csv")
    with open(greedy_csv, "w", newline="") as f:
        csv.writer(f).writerow(["update", "global_step", "level", "greedy_success_rate"])
    ckpt_dir = os.path.join(cfg["out_dir"], "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    def on_update_end(update, global_step, model_):
        if update % cfg["ckpt_every_updates"] != 0:
            return
        level = curriculum.current_level()
        rate = quick_greedy_eval(model_, gen, level, cfg["split"], cfg, cfg["greedy_eval_episodes"])
        with open(greedy_csv, "a", newline="") as f:
            csv.writer(f).writerow([update, global_step, level, rate])
        print(f"  [greedy-eval] update={update} step={global_step} level={level} success_rate={rate:.3f}")
        torch.save(
            {"model_state": model_.state_dict(), "config": cfg, "global_step": global_step, "level": level},
            os.path.join(ckpt_dir, f"update_{update:06d}.pt"),
        )
        if tb is not None:
            tb.add_scalar("ppo/greedy_success_rate", rate, global_step)

    model = train_ppo(
        envs, model, ppo_cfg, on_episode_end=on_episode_end,
        log_csv_path=os.path.join(cfg["out_dir"], "log.csv"), tb_writer=tb,
        on_update_end=on_update_end,
    )

    ckpt_path = os.path.join(cfg["out_dir"], "listener.pt")
    torch.save({"model_state": model.state_dict(), "config": cfg, "curriculum_log": curriculum.log, "final_level": curriculum.current_level()}, ckpt_path)
    if tb is not None:
        tb.close()
    print(f"saved checkpoint -> {ckpt_path} (reached {curriculum.current_level()})")
    return ckpt_path


if __name__ == "__main__":
    cfg = load_config(DEFAULT_CONFIG, description="train the stage-2 listener with PPO")
    train(cfg)
