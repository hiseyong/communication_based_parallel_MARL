"""Stage 1 evaluation: offline (accuracy / per-class P-R-F1 / confusion matrix on
val, test_layout, test_combo) and online (swap the learned speaker in for one or
both ScriptedAgents' outgoing message inside a real episode; measure message
accuracy against the oracle label and task success rate with a ScriptedAgent
listener on the receiving end).
"""
from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import numpy as np
import torch

from ..comm.vocab import Vocab
from ..config_utils import load_config
from ..envs.dual_route_env import AGENT_OF, DualRouteEnv
from ..envs.layout_generator import LayoutGenerator
from ..envs.oracle import ScriptedAgent
from ..models.factory import build_speaker
from .plots import plot_confusion_matrix
from ..train.collect_speaker_data import _parse_excluded_combos

DEFAULT_CONFIG = dict(
    checkpoint="runs/speaker_exp1/0/speaker.pt",
    data_dir="data/speaker",
    out_dir="runs/speaker_exp1/0/eval",
    splits="val,test_layout,test_combo",
    online_episodes=200,
    online_levels="L1,L2,L3",
    sender_side="both",  # "a" | "b" | "both"
    excluded_combos="OBSTACLE:BLUE",
    K=3,
    H=7,
    W=15,
    V=7,
    max_steps=200,
    device="cpu",
)


def load_speaker(checkpoint: str, device: str):
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = build_speaker(cfg["K"], cfg["V"], cnn_channels=cfg["cnn_channels"], hidden=cfg["hidden"], gumbel=cfg.get("gumbel", False))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, cfg


def offline_eval(model, npz_path: str, vocab: Vocab, device: str) -> Dict:
    d = np.load(npz_path, allow_pickle=True)
    grid = torch.from_numpy(d["grid"]).float().to(device)
    pos = torch.from_numpy(d["pos"]).float().to(device)
    status = torch.from_numpy(d["own_status"]).float().to(device)
    label = d["label"]

    with torch.no_grad():
        pred = model.act(grid, pos, status).cpu().numpy()

    n_classes = vocab.size
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(label, pred):
        cm[t, p] += 1

    acc = float((pred == label).mean())
    per_class = {}
    for c in range(n_classes):
        tp = cm[c, c]
        support = cm[c, :].sum()
        pred_pos = cm[:, c].sum()
        precision = tp / pred_pos if pred_pos > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[vocab.decode(c)] = {"precision": precision, "recall": recall, "f1": f1, "support": int(support)}

    return {"accuracy": acc, "per_class": per_class, "confusion_matrix": cm.tolist(), "n_samples": len(label)}


def online_eval(model, cfg: Dict, vocab: Vocab, level: str, split: str, device: str) -> Dict:
    excluded = _parse_excluded_combos(cfg["excluded_combos"], cfg["K"]) if cfg["excluded_combos"] else None
    gen = LayoutGenerator(K=cfg["K"], H=cfg["H"], W=cfg["W"], excluded_combos=excluded)
    env = DualRouteEnv(H=cfg["H"], W=cfg["W"], K=cfg["K"], V=cfg["V"], max_steps=cfg["max_steps"], layout_generator=gen, split=split)

    n_msg, n_msg_correct = 0, 0
    n_success = 0
    total_steps = []

    for ep in range(cfg["online_episodes"]):
        obs, infos = env.reset(options={"layout_index": ep, "level": level, "split": split})
        agents = {
            "a": ScriptedAgent("a", env._layout, epsilon=0.0, rng=np.random.default_rng(ep), view_radius=cfg["V"] // 2),
            "b": ScriptedAgent("b", env._layout, epsilon=0.0, rng=np.random.default_rng(ep + 1), view_radius=cfg["V"] // 2),
        }
        use_speaker = {"a": cfg["sender_side"] in ("a", "both"), "b": cfg["sender_side"] in ("b", "both")}
        done = False
        for t in range(cfg["max_steps"]):
            actions = {}
            for label_lbl in ("a", "b"):
                other = "b" if label_lbl == "a" else "a"
                act, oracle_msg = agents[label_lbl].act(
                    env._pos[label_lbl], env._toggles[label_lbl], env._pos[other], env._toggles[other],
                    env._msg_in[label_lbl], vocab,
                )
                send_msg = oracle_msg
                if use_speaker[label_lbl]:
                    o = obs[AGENT_OF[label_lbl]]
                    with torch.no_grad():
                        g = torch.from_numpy(o["grid"]).float().unsqueeze(0).to(device)
                        p = torch.from_numpy(o["pos"]).float().unsqueeze(0).to(device)
                        s = torch.from_numpy(o["own_status"]).float().unsqueeze(0).to(device)
                        send_msg = int(model.act(g, p, s).item())
                    n_msg += 1
                    n_msg_correct += int(send_msg == oracle_msg)
                actions[AGENT_OF[label_lbl]] = np.array([int(act), send_msg])
            obs, rewards, terms, truncs, infos = env.step(actions)
            if all(terms.values()) or all(truncs.values()):
                done = all(terms.values())
                total_steps.append(t + 1)
                break
        n_success += int(done)

    return {
        "message_accuracy": n_msg_correct / n_msg if n_msg else float("nan"),
        "task_success_rate": n_success / cfg["online_episodes"],
        "avg_steps": float(np.mean(total_steps)) if total_steps else float("nan"),
    }


def run(cfg: Dict) -> Dict:
    model, model_cfg = load_speaker(cfg["checkpoint"], cfg["device"])
    K = model_cfg["K"]
    colors = ["RED", "GREEN", "BLUE"][:K] if K <= 3 else [f"C{i}" for i in range(K)]
    vocab = Vocab(tuple(colors))

    os.makedirs(cfg["out_dir"], exist_ok=True)
    results = {}
    for split in cfg["splits"].split(",") if isinstance(cfg["splits"], str) else cfg["splits"]:
        npz_path = os.path.join(cfg["data_dir"], f"{split}.npz")
        if not os.path.exists(npz_path):
            print(f"skip offline eval for {split}: {npz_path} not found")
            continue
        off = offline_eval(model, npz_path, vocab, cfg["device"])
        plot_confusion_matrix(
            np.array(off["confusion_matrix"]), vocab.tokens, os.path.join(cfg["out_dir"], f"confusion_{split}.png"), title=f"Speaker confusion ({split})"
        )
        results[split] = {"offline": off}
        print(f"[{split}] offline accuracy={off['accuracy']:.4f}")

    online_levels = cfg["online_levels"].split(",") if isinstance(cfg["online_levels"], str) else cfg["online_levels"]
    for split in ["val", "test_layout", "test_combo"]:
        for level in online_levels:
            on = online_eval(model, cfg, vocab, level, split, cfg["device"])
            results.setdefault(split, {}).setdefault("online", {})[level] = on
            print(f"[{split}/{level}] online msg_acc={on['message_accuracy']:.4f} task_success={on['task_success_rate']:.4f}")

    with open(os.path.join(cfg["out_dir"], "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    cfg = load_config(DEFAULT_CONFIG, description="evaluate the stage-1 speaker")
    run(cfg)
