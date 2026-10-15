"""Matplotlib plotting helpers shared by eval_speaker.py / eval_listener.py /
scripts/run_seeds.sh aggregation."""
from __future__ import annotations

import os
from typing import Dict, List, Sequence

import numpy as np


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], path: str, title: str = "Confusion matrix") -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(class_names)), max(5, 0.6 * len(class_names))))
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j] > 0:
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=7, color="black" if norm[i, j] < 0.5 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mean_std_curve(series: Dict[str, List[Sequence[float]]], path: str, xlabel: str = "step", ylabel: str = "value", title: str = "") -> None:
    """series: {label: [run1_values, run2_values, ...]} (one list per seed, same length ideally)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, runs in series.items():
        min_len = min(len(r) for r in runs)
        arr = np.stack([np.asarray(r[:min_len], dtype=float) for r in runs])
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        x = np.arange(min_len)
        ax.plot(x, mean, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bar_with_error(categories: Sequence[str], means: Sequence[float], stds: Sequence[float], path: str, ylabel: str = "value", title: str = "") -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(4, 0.8 * len(categories)), 4))
    x = np.arange(len(categories))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
