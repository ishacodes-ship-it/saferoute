"""Centralized experiment configuration loading.

Every pipeline stage reads its parameters through `load_config`, never from
hardcoded literals in the calling module. This is what makes an experiment
(city, H3 resolution, lambda/mu grids, seeds, ...) reproducible and
swappable from a single YAML file per PRD §16/§17.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "base.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


@dataclass(frozen=True)
class Config:
    """Thin wrapper over the merged config dict with dict-like and dotted access."""

    _data: dict

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def as_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def path(self, key: str) -> Path:
        """Resolve a `paths.<key>` entry to an absolute Path under the repo root."""
        rel = self._data["paths"][key]
        p = REPO_ROOT / rel
        p.mkdir(parents=True, exist_ok=True)
        return p

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"Config(experiment_name={self._data.get('experiment_name')!r})"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH, override_path: str | Path | None = None) -> Config:
    """Load base.yaml, optionally deep-merging an override YAML on top.

    Args:
        config_path: path to the base config (defaults to config/base.yaml).
        override_path: optional path to a YAML file whose keys override the base
            config (used by ablation scripts to vary e.g. routing.default_lambda
            without duplicating the whole file).
    """
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    if override_path is not None:
        with open(override_path, "r") as f:
            override_data = yaml.safe_load(f) or {}
        data = _deep_merge(data, override_data)

    return Config(_data=data)
