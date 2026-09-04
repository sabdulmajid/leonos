"""Single-source YAML configuration loading and hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/base.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the experiment configuration and attach its stable SHA-256 hash."""
    config_path = Path(path)
    raw = config_path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    config["_meta"] = {
        "path": str(config_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return config


def canonical_config_json(config: dict[str, Any]) -> str:
    """Return a deterministic serialization suitable for run manifests."""
    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
