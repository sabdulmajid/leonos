from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from leonos.data import build_manifest, exchange_sessions
from leonos.features import FEATURE_COLUMNS
from leonos.pipeline import fit_baseline, prepare_data

REVISION = "b" * 40


def _market_rows(sessions: pd.DatetimeIndex, ticker: str, ticker_number: int) -> pd.DataFrame:
    number = np.arange(len(sessions), dtype=float)
    rate = 0.00015 * (ticker_number + 1)
    close = (90.0 + 15.0 * ticker_number) * np.exp(
        rate * number + 0.008 * np.sin(number / 17.0 + ticker_number)
    )
    open_ = close * (1.0 + 0.001 * np.sin(number / 5.0 + ticker_number))
    local = pd.DatetimeIndex(sessions).tz_localize("America/New_York")
    return pd.DataFrame(
        {
            "datetime": local,
            "symbol": ticker,
            "timeframe": "1day",
            "open": open_,
            "high": np.maximum(open_, close) * 1.005,
            "low": np.minimum(open_, close) * 0.995,
            "close": close,
            "volume": 1_000_000.0
            + ticker_number * 100_000.0
            + 20_000.0 * (1.0 + np.sin(number / 11.0)),
            "close_adj": close,
        }
    )


def _write_fixture_snapshot(root: Path) -> None:
    sessions = exchange_sessions("2023-01-03", "2025-03-14")
    panel = pd.concat(
        [_market_rows(sessions, ticker, number) for number, ticker in enumerate("ABC")],
        ignore_index=True,
    )
    # Both accepted quality-policy reasons are exercised. Neither is hard-coded
    # into the pipeline; the real snapshot determines its own counts.
    zero_key = (panel["symbol"] == "A") & (
        panel["datetime"].dt.date == sessions[100].date()
    )
    panel.loc[zero_key, "volume"] = 0.0
    weekend = _market_rows(pd.DatetimeIndex(["2024-06-15"]), "B", 1)
    panel = pd.concat([panel, weekend], ignore_index=True)

    split_dates = pd.to_datetime(panel["datetime"], utc=True).dt.tz_convert(None).dt.normalize()
    partitions = {
        "train": panel.loc[split_dates <= pd.Timestamp("2023-12-31")],
        "val": panel.loc[
            (split_dates >= pd.Timestamp("2024-01-01"))
            & (split_dates <= pd.Timestamp("2024-12-31"))
        ],
        "test": panel.loc[split_dates >= pd.Timestamp("2025-01-01")],
    }
    bars_root = root / "bars_1day"
    bars_root.mkdir(parents=True)
    for split, frame in partitions.items():
        frame.to_parquet(bars_root / f"{split}.parquet", index=False)
    (root / "README.md").write_text("pipeline fixture\n", encoding="utf-8")
    build_manifest(
        root,
        revision=REVISION,
        retrieved_at_utc="2026-01-01T00:00:00+00:00",
    )


def _config(root: Path) -> dict[str, object]:
    return {
        "experiment": {"name": "pipeline-fixture", "seed": 42},
        "sources": {"dataset": {"revision": REVISION}},
        "paths": {
            "raw_data": str(root / "raw"),
            "prepared_data": str(root / "prepared"),
            "artifacts": str(root / "artifacts"),
            "summaries": str(root / "results" / "summary"),
        },
        "data": {
            "documented_splits": [],
            "require_adjustment_check": False,
            "drop_identical_duplicates": True,
            "fatal_ohlc": "raise",
        },
        "forecast": {"context_sessions": 90, "horizon_sessions": 10},
        "splits": {
            "development_label_end_max": "2023-12-31",
            "validation_origin_min": "2024-07-01",
            "validation_origin_max": "2024-12-31",
            "validation_label_end_max": "2024-12-31",
            "final_refit_label_end_max": "2024-12-31",
            "test_origin_min": "2025-01-01",
            "test_origin_max": "2025-02-28",
            "test_label_end_max": "2025-03-14",
        },
        "lightgbm": {
            "num_threads": 2,
            "max_search_configs": 1,
            "early_stopping_rounds": 5,
            "max_rounds": 30,
            "candidates": [
                {
                    "candidate_id": "fixture",
                    "learning_rate": 0.1,
                    "num_leaves": 7,
                    "min_data_in_leaf": 5,
                }
            ],
        },
        "evaluation": {"minimum_daily_coverage": 3},
    }


@pytest.fixture(scope="module")
def prepared_fixture(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("pipeline")
    config = _config(root)
    _write_fixture_snapshot(Path(config["paths"]["raw_data"]))
    outputs = prepare_data(config)
    return config, outputs


def test_prepare_writes_accepted_panel_exact_splits_and_one_feature_cache(
    prepared_fixture,
) -> None:
    config, outputs = prepared_fixture
    prepared = Path(config["paths"]["prepared_data"])
    assert outputs["calendar"] == str(prepared / "sessions.parquet")
    assert outputs["labels_all"] == str(prepared / "labels" / "all.parquet")
    assert outputs["labels_test"] == str(prepared / "labels" / "test.parquet")
    assert outputs["features"] == str(prepared / "features.parquet")
    assert all(isinstance(value, str) for value in outputs.values())

    raw_audit = json.loads(Path(outputs["raw_audit_json"]).read_text(encoding="utf-8"))
    accepted_audit = json.loads(
        Path(outputs["accepted_audit_json"]).read_text(encoding="utf-8")
    )
    assert raw_audit["acceptance"]["accepted"] is False
    assert accepted_audit["acceptance"]["accepted"] is True
    exclusions = pd.read_parquet(outputs["exclusions"])
    assert set(exclusions["reason"]) == {
        "non_exchange_session",
        "nonpositive_or_nonfinite_volume",
    }
    accepted = pd.read_parquet(outputs["accepted_bars"])
    assert accepted["session"].dt.tz is None
    assert accepted["datetime"].dt.tz is not None

    development = pd.read_parquet(outputs["labels_development"])
    validation = pd.read_parquet(outputs["labels_validation"])
    refit = pd.read_parquet(outputs["labels_refit"])
    test = pd.read_parquet(outputs["labels_test"])
    assert development["label_end"].max() <= pd.Timestamp("2023-12-31")
    assert validation["origin"].min() >= pd.Timestamp("2024-07-01")
    assert validation["label_end"].max() <= pd.Timestamp("2024-12-31")
    assert refit["label_end"].max() <= pd.Timestamp("2024-12-31")
    assert test["origin"].min() >= pd.Timestamp("2025-01-01")
    assert test["origin"].max() <= pd.Timestamp("2025-02-28")
    assert test["label_end"].max() <= pd.Timestamp("2025-03-14")

    features = pd.read_parquet(outputs["features"])
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert len(FEATURE_COLUMNS) == 61
    labels_all = pd.read_parquet(outputs["labels_all"])
    assert len(features) == len(labels_all)

    # Complete markers make the expensive feature preparation safely resumable.
    feature_mtime = Path(outputs["features"]).stat().st_mtime_ns
    resumed = prepare_data(config)
    assert resumed == outputs
    assert Path(outputs["features"]).stat().st_mtime_ns == feature_mtime


def test_fit_baseline_persists_search_refit_and_immutable_predictions(
    prepared_fixture,
) -> None:
    config, outputs = prepared_fixture
    result = fit_baseline(config, seed=42)
    artifact_root = Path(config["paths"]["artifacts"])
    assert result["tuning_root"] == str(
        artifact_root / "models" / "lightgbm" / "seed=42" / "tuning"
    )
    assert result["final_model_root"] == str(
        artifact_root / "models" / "lightgbm" / "seed=42" / "final"
    )
    assert result["validation_predictions"] == str(
        artifact_root
        / "predictions"
        / "lightgbm"
        / "validation"
        / "seed=42"
        / "predictions.parquet"
    )
    assert result["test_predictions"] == str(
        artifact_root
        / "predictions"
        / "lightgbm"
        / "test"
        / "seed=42"
        / "predictions.parquet"
    )
    assert all(isinstance(value, str) for value in result.values())

    candidates = pd.read_parquet(Path(result["tuning_root"]) / "candidates.parquet")
    assert len(candidates) == 1
    assert candidates.loc[0, "status"] == "ok"
    validation_predictions = pd.read_parquet(result["validation_predictions"])
    test_predictions = pd.read_parquet(result["test_predictions"])
    assert len(validation_predictions) == len(
        pd.read_parquet(outputs["labels_validation"])
    )
    assert len(test_predictions) == len(pd.read_parquet(outputs["labels_test"]))
    assert validation_predictions["status"].eq("ok").all()
    assert test_predictions["status"].eq("ok").all()

    metadata = json.loads(Path(result["run_metadata"]).read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert metadata["search"]["candidate_count"] == 1
    assert metadata["search"]["selected"]["candidate_id"] == "fixture"
    assert metadata["prepare_signature"]
    assert metadata["git"]["dirty"] is True

    prediction_mtime = Path(result["test_predictions"]).stat().st_mtime_ns
    resumed = fit_baseline(config, seed=42)
    assert resumed == result
    assert Path(result["test_predictions"]).stat().st_mtime_ns == prediction_mtime
