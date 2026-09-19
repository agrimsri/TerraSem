"""Configuration loading with CLI override support.

Loads a YAML config file and merges dot-separated ``key=value`` overrides
supplied on the command line, returning a plain dict.  No magic numbers inside
modules — all tunables live in ``configs/`` and are accessed via this module.

Usage::

    from terrasem.utils.config import load_config
    cfg = load_config("configs/model/segformer_b0.yaml",
                      overrides=["training.lr=1e-4", "training.batch_size=4"])
    lr = cfg["training"]["lr"]  # float 1e-4
"""

from __future__ import annotations

import contextlib
import copy
from pathlib import Path
from typing import Any

import yaml


def _set_nested(d: dict, keys: list[str], value: Any) -> None:
    """Set a value in a nested dict given a list of key segments."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    # Attempt to infer type from existing value; fall back to string.
    existing = d.get(keys[-1])
    if existing is not None:
        with contextlib.suppress(ValueError, TypeError):
            value = type(existing)(value)
    else:
        # Try int → float → str
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except (ValueError, TypeError):
                pass
    d[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config file and apply CLI ``key=value`` overrides.

    Args:
        path: Path to the YAML config file.
        overrides: List of ``"dot.separated.key=value"`` strings.
            Keys use ``"."`` as separator for nested dicts.

    Returns:
        Merged config dict.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If an override string is not in ``"key=value"`` form.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open() as fh:
        cfg: dict = yaml.safe_load(fh) or {}

    cfg = copy.deepcopy(cfg)

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(
                f"Override must be in 'key=value' form, got: {override!r}"
            )
        raw_key, _, raw_val = override.partition("=")
        _set_nested(cfg, raw_key.split("."), raw_val)

    return cfg
