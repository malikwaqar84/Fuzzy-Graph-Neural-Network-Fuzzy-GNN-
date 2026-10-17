"""Configuration loading with dotted-key overrides (e.g. model.use_eai=false)."""
from __future__ import annotations

import copy
import yaml

ABLATIONS = {
    "full": {},
    "hard_gcn": {"model.membership": "hard_gcn"},
    "gaussian": {"model.membership": "gaussian"},
    "fixed_membership": {"model.learn_membership": False},
    "static": {"model.membership": "static"},
    "no_eai": {"model.use_eai": False},
    "short_only": {"model.multiscale": False},
    "no_attention": {"model.use_attention": False},
    "no_temporal": {"train.lambda_temp": 0.0},
    # "no_firing_power" is evaluated from the predictions of the full model (see run_loso.py)
}


def load_config(path: str, overrides: dict | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in (overrides or {}).items():
        set_key(cfg, k, v)
    return cfg


def set_key(cfg: dict, dotted: str, value):
    d = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def parse_override(s: str):
    k, v = s.split("=", 1)
    return k, yaml.safe_load(v)


def with_overrides(cfg: dict, overrides: dict) -> dict:
    out = copy.deepcopy(cfg)
    for k, v in overrides.items():
        set_key(out, k, v)
    return out
