from __future__ import annotations

import json

from leonos.artifacts import atomic_write_json, stable_hash
from leonos.config import load_config


def test_config_has_frozen_contract() -> None:
    config = load_config()
    assert config["forecast"]["context_sessions"] == 90
    assert config["forecast"]["horizon_sessions"] == 10
    assert config["forecast"]["kronos"]["sample_count"] == 10
    assert len(config["sources"]["dataset"]["revision"]) == 40


def test_atomic_json_and_stable_hash(tmp_path) -> None:
    path = atomic_write_json(tmp_path / "nested" / "run.json", {"b": 2, "a": 1})
    assert json.loads(path.read_text()) == {"a": 1, "b": 2}
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
