"""Stage 1 supervised training: obs -> oracle message label, cross-entropy
(class-weighted by default to counter the NONE-majority class imbalance).
"""
from __future__ import annotations

import csv
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..config_utils import load_config
from ..models.factory import build_speaker, channel_dims

DEFAULT_CONFIG = dict(
    data_dir="data/speaker",
    out_dir="runs/speaker_exp1/0",
    seed=0,
    K=3,
    V=7,
    epochs=20,
    batch_size=256,
    lr=1e-3,
    cnn_channels=32,
    hidden=128,
    gumbel=False,
    class_weighted=True,
    device="cpu",
)


def _load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    return (
        torch.from_numpy(d["grid"]).float(),
        torch.from_numpy(d["pos"]).float(),
        torch.from_numpy(d["own_status"]).float(),
        torch.from_numpy(d["label"]).long(),
    )


def train(cfg: Dict) -> str:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device(cfg["device"])
    os.makedirs(cfg["out_dir"], exist_ok=True)

    grid, pos, status, label = _load_npz(os.path.join(cfg["data_dir"], "train.npz"))
    vocab_size = channel_dims(cfg["K"])["vocab_size"]
    train_ds = TensorDataset(grid, pos, status, label)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)

    val_path = os.path.join(cfg["data_dir"], "val.npz")
    val_loader = None
    if os.path.exists(val_path):
        vgrid, vpos, vstatus, vlabel = _load_npz(val_path)
        val_loader = DataLoader(TensorDataset(vgrid, vpos, vstatus, vlabel), batch_size=512)

    class_weight = None
    if cfg["class_weighted"]:
        counts = torch.bincount(label, minlength=vocab_size).float()
        inv = 1.0 / torch.clamp(counts, min=1.0)
        class_weight = (inv / inv.sum() * vocab_size).to(device)

    model = build_speaker(cfg["K"], cfg["V"], cnn_channels=cfg["cnn_channels"], hidden=cfg["hidden"], gumbel=cfg["gumbel"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)

    try:
        from torch.utils.tensorboard import SummaryWriter

        tb = SummaryWriter(cfg["out_dir"])
    except Exception:
        tb = None

    log_path = os.path.join(cfg["out_dir"], "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_val_acc = -1.0
    best_path = os.path.join(cfg["out_dir"], "speaker.pt")

    def _eval(loader):
        model.eval()
        n, correct, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for g, p, s, y in loader:
                g, p, s, y = g.to(device), p.to(device), s.to(device), y.to(device)
                logits = model(g, p, s)
                loss = nn.functional.cross_entropy(logits, y)
                total_loss += loss.item() * len(y)
                correct += (logits.argmax(-1) == y).sum().item()
                n += len(y)
        return total_loss / n, correct / n

    for epoch in range(cfg["epochs"]):
        model.train()
        n, correct, total_loss = 0, 0, 0.0
        for g, p, s, y in train_loader:
            g, p, s, y = g.to(device), p.to(device), s.to(device), y.to(device)
            logits = model(g, p, s)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(-1) == y).sum().item()
            n += len(y)
        train_loss, train_acc = total_loss / n, correct / n

        val_loss, val_acc = (float("nan"), float("nan"))
        if val_loader is not None:
            val_loss, val_acc = _eval(val_loader)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({"model_state": model.state_dict(), "config": cfg}, best_path)
        else:
            torch.save({"model_state": model.state_dict(), "config": cfg}, best_path)

        print(f"epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, train_acc, val_loss, val_acc])
        if tb is not None:
            tb.add_scalar("train/loss", train_loss, epoch)
            tb.add_scalar("train/acc", train_acc, epoch)
            if val_loader is not None:
                tb.add_scalar("val/loss", val_loss, epoch)
                tb.add_scalar("val/acc", val_acc, epoch)

    if tb is not None:
        tb.close()
    print(f"saved checkpoint -> {best_path}")
    return best_path


if __name__ == "__main__":
    cfg = load_config(DEFAULT_CONFIG, description="train the stage-1 speaker")
    train(cfg)
