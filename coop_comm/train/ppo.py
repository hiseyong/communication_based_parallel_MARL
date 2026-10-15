"""CleanRL-style single-file PPO, decoupled from any specific env/model framework
so the communication module (Listener) stays easy to swap/modify. Expects:
  - `envs`: a list of objects with `reset(seed=None, options=None) -> obs_dict`
    and `step(action: int) -> (obs_dict, reward, terminated, truncated, info)`,
    obs_dict having numpy arrays under "grid", "pos", "own_status", "msg_in".
  - `model`: a `coop_comm.models.listener.Listener`.
An optional `on_episode_end(env_idx, success, info)` callback lets the caller
(train_listener.py) drive curriculum progression.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class PPOConfig:
    total_timesteps: int = 200_000
    n_envs: int = 8
    n_steps: int = 128
    update_epochs: int = 4
    num_minibatches: int = 4
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    device: str = "cpu"
    seed: int = 0


def _stack_obs(obs_list: List[Dict[str, np.ndarray]], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        k: torch.from_numpy(np.stack([o[k] for o in obs_list])).float().to(device)
        for k in ("grid", "pos", "own_status", "msg_in")
    }


def train_ppo(
    envs: List,
    model: nn.Module,
    cfg: PPOConfig,
    on_episode_end: Optional[Callable[[int, bool, dict, int], None]] = None,
    log_csv_path: Optional[str] = None,
    tb_writer=None,
    reset_options: Optional[Callable[[int], dict]] = None,
    on_update_end: Optional[Callable[[int, int, nn.Module], None]] = None,
) -> nn.Module:
    device = torch.device(cfg.device)
    model.to(device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    n_envs = cfg.n_envs
    assert len(envs) == n_envs
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    obs = [env.reset(seed=cfg.seed + i, options=(reset_options(i) if reset_options else None)) for i, env in enumerate(envs)]
    hidden = model.initial_hidden(n_envs, device) if model.use_gru else None

    n_updates = max(1, cfg.total_timesteps // (cfg.n_steps * n_envs))
    batch_size = cfg.n_steps * n_envs
    minibatch_size = max(1, batch_size // cfg.num_minibatches)

    if log_csv_path:
        os.makedirs(os.path.dirname(log_csv_path) or ".", exist_ok=True)
        with open(log_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["update", "global_step", "mean_reward", "success_rate", "policy_loss", "value_loss", "entropy"])

    global_step = 0
    for update in range(n_updates):
        if cfg.anneal_lr:
            frac = 1.0 - update / n_updates
            optimizer.param_groups[0]["lr"] = frac * cfg.lr

        buf_grid, buf_pos, buf_status, buf_msgin = [], [], [], []
        buf_actions = torch.zeros(cfg.n_steps, n_envs, device=device)
        buf_logprobs = torch.zeros(cfg.n_steps, n_envs, device=device)
        buf_rewards = torch.zeros(cfg.n_steps, n_envs, device=device)
        buf_dones = torch.zeros(cfg.n_steps, n_envs, device=device)
        buf_values = torch.zeros(cfg.n_steps, n_envs, device=device)
        ep_rewards = [0.0] * n_envs
        hidden0 = hidden if model.use_gru else None
        update_successes: List[float] = []

        for step in range(cfg.n_steps):
            obs_t = _stack_obs(obs, device)
            buf_grid.append(obs_t["grid"])
            buf_pos.append(obs_t["pos"])
            buf_status.append(obs_t["own_status"])
            buf_msgin.append(obs_t["msg_in"])

            with torch.no_grad():
                logits, value, hidden = model.forward_step(obs_t["grid"], obs_t["pos"], obs_t["own_status"], obs_t["msg_in"], hidden)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)

            buf_actions[step] = action
            buf_logprobs[step] = logprob
            buf_values[step] = value

            next_obs = []
            for i, env in enumerate(envs):
                o, r, term, trunc, info = env.step(int(action[i].item()))
                ep_rewards[i] += r
                buf_rewards[step, i] = r
                done = term or trunc
                buf_dones[step, i] = float(done)
                if done:
                    update_successes.append(1.0 if term else 0.0)
                    if on_episode_end is not None:
                        on_episode_end(i, bool(term), info, global_step)
                    o = env.reset(options=(reset_options(i) if reset_options else None))
                    ep_rewards[i] = 0.0
                    if model.use_gru:
                        hidden[:, i, :] = 0.0
                next_obs.append(o)
            obs = next_obs
            global_step += n_envs

        with torch.no_grad():
            obs_t = _stack_obs(obs, device)
            _, next_value, _ = model.forward_step(obs_t["grid"], obs_t["pos"], obs_t["own_status"], obs_t["msg_in"], hidden)

        advantages = torch.zeros(cfg.n_steps, n_envs, device=device)
        lastgaelam = torch.zeros(n_envs, device=device)
        for t in reversed(range(cfg.n_steps)):
            nextnonterminal = 1.0 - buf_dones[t]
            nextvalues = next_value if t == cfg.n_steps - 1 else buf_values[t + 1]
            delta = buf_rewards[t] + cfg.gamma * nextvalues * nextnonterminal - buf_values[t]
            lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + buf_values

        grid_b = torch.stack(buf_grid)  # (T, n_envs, ...)
        pos_b = torch.stack(buf_pos)
        status_b = torch.stack(buf_status)
        msgin_b = torch.stack(buf_msgin)

        if model.use_gru:
            policy_loss_acc = value_loss_acc = ent_acc = 0.0
            n_batches = 0
            env_idx_all = np.arange(n_envs)
            for epoch in range(cfg.update_epochs):
                np.random.shuffle(env_idx_all)
                for start in range(0, n_envs, max(1, n_envs // cfg.num_minibatches)):
                    mb = env_idx_all[start : start + max(1, n_envs // cfg.num_minibatches)]
                    g = grid_b[:, mb].transpose(0, 1)
                    p = pos_b[:, mb].transpose(0, 1)
                    s = status_b[:, mb].transpose(0, 1)
                    m = msgin_b[:, mb].transpose(0, 1)
                    done_mask = buf_dones[:, mb].transpose(0, 1)
                    h0 = hidden0[:, mb] if hidden0 is not None else None
                    logits, value = model.forward_seq(g, p, s, m, hidden0=h0, done_mask=done_mask)
                    logits = logits.transpose(0, 1).reshape(-1, logits.shape[-1])
                    value = value.transpose(0, 1).reshape(-1)
                    a = buf_actions[:, mb].reshape(-1)
                    oldlp = buf_logprobs[:, mb].reshape(-1)
                    adv = advantages[:, mb].reshape(-1)
                    ret = returns[:, mb].reshape(-1)
                    pl, vl, ent = _ppo_losses(logits, value, a, oldlp, adv, ret, cfg)
                    loss = pl + cfg.vf_coef * vl - cfg.ent_coef * ent
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    policy_loss_acc += pl.item(); value_loss_acc += vl.item(); ent_acc += ent.item(); n_batches += 1
        else:
            grid_f = grid_b.reshape(batch_size, *grid_b.shape[2:])
            pos_f = pos_b.reshape(batch_size, *pos_b.shape[2:])
            status_f = status_b.reshape(batch_size, *status_b.shape[2:])
            msgin_f = msgin_b.reshape(batch_size, *msgin_b.shape[2:])
            actions_f = buf_actions.reshape(batch_size)
            logprobs_f = buf_logprobs.reshape(batch_size)
            advantages_f = advantages.reshape(batch_size)
            returns_f = returns.reshape(batch_size)

            idx = np.arange(batch_size)
            policy_loss_acc = value_loss_acc = ent_acc = 0.0
            n_batches = 0
            for epoch in range(cfg.update_epochs):
                np.random.shuffle(idx)
                for start in range(0, batch_size, minibatch_size):
                    mb = idx[start : start + minibatch_size]
                    logits, value, _ = model.forward_step(grid_f[mb], pos_f[mb], status_f[mb], msgin_f[mb], None)
                    pl, vl, ent = _ppo_losses(logits, value, actions_f[mb], logprobs_f[mb], advantages_f[mb], returns_f[mb], cfg)
                    loss = pl + cfg.vf_coef * vl - cfg.ent_coef * ent
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    policy_loss_acc += pl.item(); value_loss_acc += vl.item(); ent_acc += ent.item(); n_batches += 1

        mean_reward = buf_rewards.mean().item()
        success_rate = float(np.mean(update_successes)) if update_successes else float("nan")
        if log_csv_path:
            with open(log_csv_path, "a", newline="") as f:
                csv.writer(f).writerow([update, global_step, mean_reward, success_rate, policy_loss_acc / n_batches, value_loss_acc / n_batches, ent_acc / n_batches])
        if tb_writer is not None:
            tb_writer.add_scalar("ppo/mean_reward", mean_reward, global_step)
            tb_writer.add_scalar("ppo/success_rate", success_rate, global_step)
            tb_writer.add_scalar("ppo/policy_loss", policy_loss_acc / n_batches, global_step)
            tb_writer.add_scalar("ppo/value_loss", value_loss_acc / n_batches, global_step)
            tb_writer.add_scalar("ppo/entropy", ent_acc / n_batches, global_step)

        if on_update_end is not None:
            on_update_end(update, global_step, model)

    return model


def _ppo_losses(logits, value, actions, old_logprobs, advantages, returns, cfg: PPOConfig):
    dist = Categorical(logits=logits)
    newlogprob = dist.log_prob(actions)
    entropy = dist.entropy().mean()
    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()

    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss1 = -adv * ratio
    pg_loss2 = -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
    policy_loss = torch.max(pg_loss1, pg_loss2).mean()
    value_loss = 0.5 * ((value - returns) ** 2).mean()
    return policy_loss, value_loss, entropy
