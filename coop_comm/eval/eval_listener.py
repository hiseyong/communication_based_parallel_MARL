"""Stage 2 evaluation: task success rate / avg steps, response accuracy (REQ_*
resolved within T steps), hold/release behavior around DONE, false-trigger
rate, an msg_in ablation (zero / shuffle), and test_layout/test_combo
generalization. The partner is always a fixed ScriptedAgent (oracle messages,
epsilon=0), same as during training.

Assumption (documented in DESIGN_NOTES.md): at most one barrier/request is
ever pending per route in our L1-L3 layouts, so DONE (which carries no color)
is unambiguous -- it always refers to the single currently-held PLATE, if any.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import torch

from ..comm.vocab import DEVICES, INTENTS, Vocab
from ..config_utils import load_config
from ..envs.layout_generator import LayoutGenerator
from ..envs.objects import ObjectType
from ..models.factory import build_listener
from .plots import plot_bar_with_error
from ..train.train_listener import SingleAgentListenerEnv
from ..train.collect_speaker_data import _parse_excluded_combos

DEFAULT_CONFIG = dict(
    checkpoint="runs/listener_exp1/0/listener.pt",
    out_dir="runs/listener_exp1/0/eval",
    splits="val,test_layout,test_combo",
    levels="L1,L2,L3",
    n_episodes=200,
    response_T=30,
    excluded_combos="OBSTACLE:BLUE",
    K=3,
    H=7,
    W=15,
    V=7,
    max_steps=200,
    device="cpu",
)


def load_listener(checkpoint: str, device: str):
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = build_listener(cfg["K"], cfg["V"], use_gru=cfg.get("use_gru", False))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, cfg


def _apply_ablation(obs: Dict[str, np.ndarray], mode: str, vocab_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    if mode == "normal":
        return obs
    obs = dict(obs)
    if mode == "zero":
        obs["msg_in"] = np.zeros_like(obs["msg_in"])
    elif mode == "shuffle":
        v = np.zeros_like(obs["msg_in"])
        v[rng.integers(vocab_size)] = 1.0
        obs["msg_in"] = v
    return obs


def run_instrumented_episode(model, env, vocab: Vocab, device: str, response_T: int, msg_mode: str, rng: np.random.Generator, max_steps: int, seed: int, layout_index: int) -> Dict:
    obs = env.reset(seed=seed, options={"layout_index": layout_index})
    hidden = model.initial_hidden(1, device) if model.use_gru else None
    route = env.env._layout.route(env.label)
    plate_by_color = {d.color: d for d in route.devices if d.kind == ObjectType.PLATE}
    lever_by_color = {d.color: d for d in route.devices if d.kind == ObjectType.LEVER}

    pending: Dict[int, dict] = {}
    response_times = []
    missed = 0
    premature_releases = 0
    release_delays = []
    false_triggers = 0
    total_steps = 0
    success = False
    t = 0

    for t in range(max_steps):
        true_msg = env.env._msg_in[env.label]
        intent, dev_idx, color = vocab.to_slot(true_msg)
        if INTENTS[intent] == "REQ" and color not in pending:
            pending[color] = {"kind": DEVICES[dev_idx], "start": t, "state": "REQUESTED"}

        own_pos = env.env._pos[env.label]
        ctype, ccolor = route.cell(own_pos)
        if ctype == ObjectType.PLATE and ccolor not in pending:
            false_triggers += 1

        obs_model = _apply_ablation(obs, msg_mode, vocab.size, rng)
        with torch.no_grad():
            g = torch.from_numpy(obs_model["grid"]).float().unsqueeze(0).to(device)
            p = torch.from_numpy(obs_model["pos"]).float().unsqueeze(0).to(device)
            s = torch.from_numpy(obs_model["own_status"]).float().unsqueeze(0).to(device)
            m = torch.from_numpy(obs_model["msg_in"]).float().unsqueeze(0).to(device)
            logits, _, hidden = model.forward_step(g, p, s, m, hidden)
            action = int(logits.argmax(-1).item())
            if action == 5 and ctype == ObjectType.LEVER and ccolor not in pending:  # INTERACT misfire
                false_triggers += 1

        obs, reward, term, trunc, info = env.step(action)
        total_steps += 1
        new_pos = env.env._pos[env.label]
        new_toggles = env.env._toggles[env.label]
        new_true_msg = env.env._msg_in[env.label]

        for color, pend in list(pending.items()):
            if pend["state"] == "REQUESTED":
                if pend["kind"] == "PLATE" and plate_by_color.get(color) and new_pos == plate_by_color[color].pos:
                    response_times.append((t + 1) - pend["start"])
                    pend["state"] = "HOLDING"
                elif pend["kind"] == "LEVER" and color in new_toggles:
                    response_times.append((t + 1) - pend["start"])
                    del pending[color]
                elif (t + 1) - pend["start"] > response_T:
                    missed += 1
                    del pending[color]

        new_intent, _, _ = vocab.to_slot(new_true_msg)
        for color, pend in list(pending.items()):
            if pend["state"] != "HOLDING":
                continue
            plate = plate_by_color[color]
            if INTENTS[new_intent] == "DONE" and "done_t" not in pend:
                pend["done_t"] = t + 1
            if "done_t" in pend:
                if new_pos != plate.pos:
                    release_delays.append((t + 1) - pend["done_t"])
                    del pending[color]
            else:
                if new_pos != plate.pos:
                    premature_releases += 1
                    del pending[color]

        if term or trunc:
            success = bool(term)
            break

    return {
        "success": success, "steps": t + 1, "response_times": response_times, "missed": missed,
        "premature_releases": premature_releases, "release_delays": release_delays,
        "false_triggers": false_triggers, "total_steps": total_steps,
    }


def evaluate(model, cfg: Dict, vocab: Vocab, level: str, split: str, msg_mode: str, device: str) -> Dict:
    excluded = _parse_excluded_combos(cfg["excluded_combos"], cfg["K"]) if cfg["excluded_combos"] else None
    gen = LayoutGenerator(K=cfg["K"], H=cfg["H"], W=cfg["W"], excluded_combos=excluded)
    rng = np.random.default_rng(0)
    results = []
    for ep in range(cfg["n_episodes"]):
        env = SingleAgentListenerEnv("a", gen, level, split, {**cfg, "own_send_mode": "oracle", "channel_kind": "identity", "channel_p": 0.0, "seed": ep})
        results.append(run_instrumented_episode(model, env, vocab, device, cfg["response_T"], msg_mode, rng, cfg["max_steps"], seed=ep, layout_index=ep))

    n_req = sum(len(r["response_times"]) + r["missed"] for r in results)
    n_resolved = sum(len(r["response_times"]) for r in results)
    n_holds = sum(len(r["release_delays"]) + r["premature_releases"] for r in results)
    return {
        "success_rate": float(np.mean([r["success"] for r in results])),
        "avg_steps": float(np.mean([r["steps"] for r in results])),
        "response_accuracy": n_resolved / n_req if n_req else float("nan"),
        "avg_response_time": float(np.mean([v for r in results for v in r["response_times"]])) if n_resolved else float("nan"),
        "premature_release_rate": sum(r["premature_releases"] for r in results) / n_holds if n_holds else float("nan"),
        "avg_release_delay": float(np.mean([v for r in results for v in r["release_delays"]])) if any(r["release_delays"] for r in results) else float("nan"),
        "false_trigger_rate": sum(r["false_triggers"] for r in results) / sum(r["total_steps"] for r in results),
    }


def run(cfg: Dict) -> Dict:
    model, model_cfg = load_listener(cfg["checkpoint"], cfg["device"])
    K = model_cfg["K"]
    colors = ["RED", "GREEN", "BLUE"][:K] if K <= 3 else [f"C{i}" for i in range(K)]
    vocab = Vocab(tuple(colors))
    os.makedirs(cfg["out_dir"], exist_ok=True)

    splits = cfg["splits"].split(",") if isinstance(cfg["splits"], str) else cfg["splits"]
    levels = cfg["levels"].split(",") if isinstance(cfg["levels"], str) else cfg["levels"]

    results = {}
    for split in splits:
        for level in levels:
            normal = evaluate(model, cfg, vocab, level, split, "normal", cfg["device"])
            zero = evaluate(model, cfg, vocab, level, split, "zero", cfg["device"])
            shuffle = evaluate(model, cfg, vocab, level, split, "shuffle", cfg["device"])
            results.setdefault(split, {})[level] = {"normal": normal, "ablation_zero": zero, "ablation_shuffle": shuffle}
            print(f"[{split}/{level}] success={normal['success_rate']:.3f} resp_acc={normal['response_accuracy']:.3f} "
                  f"false_trig={normal['false_trigger_rate']:.4f} | zero_msg_success={zero['success_rate']:.3f} shuffle_success={shuffle['success_rate']:.3f}")

    with open(os.path.join(cfg["out_dir"], "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    cats, means, stds = [], [], []
    for split, per_level in results.items():
        for level, modes in per_level.items():
            for mode_name in ["normal", "ablation_zero", "ablation_shuffle"]:
                cats.append(f"{split}/{level}/{mode_name}")
                means.append(modes[mode_name]["success_rate"])
                stds.append(0.0)
    plot_bar_with_error(cats, means, stds, os.path.join(cfg["out_dir"], "success_rate_summary.png"), ylabel="success rate", title="Listener success rate (incl. msg_in ablation)")
    return results


if __name__ == "__main__":
    cfg = load_config(DEFAULT_CONFIG, description="evaluate the stage-2 listener")
    run(cfg)
