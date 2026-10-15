"""Minimal YAML + argparse-override config loading, shared by every train/*.py
and eval/*.py entry point (`--config foo.yaml --some_key override_value`)."""
from __future__ import annotations

import argparse
from typing import Any, Dict, Optional, Sequence

import yaml


def load_config(default: Dict[str, Any], argv: Optional[Sequence[str]] = None, description: str = "") -> Dict[str, Any]:
    pre = argparse.ArgumentParser(description=description, add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args(argv)

    cfg = dict(default)
    if pre_args.config:
        with open(pre_args.config) as f:
            loaded = yaml.safe_load(f) or {}
        cfg.update(loaded)

    parser = argparse.ArgumentParser(description=description, parents=[pre])
    for key, val in cfg.items():
        if isinstance(val, bool):
            parser.add_argument(f"--{key}", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=None)
        elif isinstance(val, list):
            parser.add_argument(f"--{key}", type=str, default=None)
        else:
            parser.add_argument(f"--{key}", type=type(val) if val is not None else str, default=None)
    args = parser.parse_args(argv)

    for key in cfg:
        override = getattr(args, key, None)
        if override is not None:
            if isinstance(cfg[key], list) and isinstance(override, str):
                cfg[key] = [v for v in override.split(",") if v]
            else:
                cfg[key] = override
    return cfg
