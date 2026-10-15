"""Aggregate per-seed listener training logs + eval results into mean+-std
tables/figures (`claude_code_prompt_stage1-2.md` §8: "시드 3개 이상... 평균±표준편차
표/그림 생성"). Expects `scripts/run_seeds.sh` to have already produced, for each
seed under --exp_dir/<seed>/: log.csv (training curve) and eval/results.json
(from eval_listener.py).
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List

import numpy as np

from ..config_utils import load_config
from .plots import plot_bar_with_error, plot_mean_std_curve

DEFAULT_CONFIG = dict(
    exp_dir="runs/listener_exp1",
    seeds="0,1,2",
    out_dir="runs/listener_exp1/aggregate",
)


def _read_training_curve(path: str, column: str = "success_rate") -> List[float]:
    with open(path) as f:
        reader = csv.DictReader(f)
        vals = []
        for row in reader:
            v = row[column]
            vals.append(float(v) if v and v != "nan" else np.nan)
    # forward-fill nans so the mean/std curve isn't dominated by missing updates
    arr = np.array(vals)
    mask = np.isnan(arr)
    if mask.all():
        return [0.0] * len(arr)
    arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
    return arr.tolist()


def run(cfg: Dict) -> Dict:
    seeds = cfg["seeds"].split(",") if isinstance(cfg["seeds"], str) else cfg["seeds"]
    os.makedirs(cfg["out_dir"], exist_ok=True)

    curves = []
    for seed in seeds:
        path = os.path.join(cfg["exp_dir"], str(seed), "log.csv")
        if os.path.exists(path):
            curves.append(_read_training_curve(path))
    if curves:
        plot_mean_std_curve(
            {"stochastic success rate (training)": curves},
            os.path.join(cfg["out_dir"], "training_curve.png"),
            xlabel="PPO update", ylabel="success rate", title="Listener training curve (mean +- std over seeds)",
        )

    per_split_level: Dict[str, Dict[str, List[float]]] = {}
    for seed in seeds:
        results_path = os.path.join(cfg["exp_dir"], str(seed), "eval", "results.json")
        if not os.path.exists(results_path):
            continue
        with open(results_path) as f:
            results = json.load(f)
        for split, per_level in results.items():
            for level, modes in per_level.items():
                key = f"{split}/{level}"
                per_split_level.setdefault(key, []).append(modes["normal"]["success_rate"])

    table = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n_seeds": len(v)} for k, v in per_split_level.items()}
    with open(os.path.join(cfg["out_dir"], "summary_table.json"), "w") as f:
        json.dump(table, f, indent=2)

    if table:
        cats = list(table.keys())
        means = [table[k]["mean"] for k in cats]
        stds = [table[k]["std"] for k in cats]
        plot_bar_with_error(cats, means, stds, os.path.join(cfg["out_dir"], "summary_bar.png"), ylabel="success rate", title=f"Listener success rate, mean +- std over {len(seeds)} seeds")
        print("summary (mean +- std over seeds):")
        for k in cats:
            print(f"  {k}: {table[k]['mean']:.3f} +- {table[k]['std']:.3f} (n={table[k]['n_seeds']})")
    return table


if __name__ == "__main__":
    cfg = load_config(DEFAULT_CONFIG, description="aggregate listener results across seeds")
    run(cfg)
