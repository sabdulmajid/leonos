from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from leonos.features import FEATURE_COLUMNS
from leonos.models.lightgbm import (
    DEFAULT_CANDIDATES,
    CandidateConfig,
    CandidateResult,
    LightGBMModel,
    SearchConfig,
    fit_final_lightgbm,
    predict_lightgbm,
    read_prediction_artifacts,
    select_candidate,
    tune_lightgbm,
    validate_search_splits,
    write_prediction_artifact,
)


def _feature_rows(dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    rows = []
    for date_number, date in enumerate(dates):
        for ticker_number, ticker in enumerate(tickers):
            signal = ticker_number / max(len(tickers) - 1, 1) + date_number * 0.001
            row: dict[str, object] = {
                "ticker": ticker,
                "origin": date,
                "label_end": date + pd.Timedelta(days=14),
                "target": signal * 0.05 - 0.02,
            }
            for feature_number, feature in enumerate(FEATURE_COLUMNS):
                row[feature] = signal + feature_number * 1e-4
            rows.append(row)
    return pd.DataFrame(rows)


def test_default_search_is_bounded_and_cpu_threads_are_capped() -> None:
    assert 1 <= len(DEFAULT_CANDIDATES) <= 12
    assert len({candidate.candidate_id for candidate in DEFAULT_CANDIDATES}) == len(
        DEFAULT_CANDIDATES
    )
    SearchConfig(num_threads=16)
    with pytest.raises(ValueError, match="capped"):
        SearchConfig(num_threads=17)


def test_validation_timing_boundaries_fail_closed() -> None:
    train = _feature_rows(pd.to_datetime(["2023-12-01"]), ["A", "B"])
    validation = _feature_rows(pd.to_datetime(["2024-07-01"]), ["A", "B"])
    validate_search_splits(train, validation)

    leaked = train.copy()
    leaked["label_end"] = pd.Timestamp("2024-01-01")
    with pytest.raises(ValueError, match="end by 2023-12-31"):
        validate_search_splits(leaked, validation)

    wrong_validation = validation.copy()
    wrong_validation["origin"] = pd.Timestamp("2024-06-28")
    with pytest.raises(ValueError, match="July-December 2024"):
        validate_search_splits(train, wrong_validation)


def test_close_rankic_tie_prefers_lower_complexity_deterministically() -> None:
    complex_candidate = CandidateConfig("complex", 0.05, 63, 20)
    simple_candidate = CandidateConfig("simple", 0.05, 15, 20)
    results = [
        CandidateResult(complex_candidate, "ok", 100, 0.2000005, 0.1, 1.0),
        CandidateResult(simple_candidate, "ok", 80, 0.2, 0.1, 1.0),
    ]
    assert select_candidate(results, tie_tolerance=1e-6).candidate == simple_candidate
    assert select_candidate(results, tie_tolerance=1e-8).candidate == complex_candidate


class _DummyBooster:
    def predict(self, matrix: pd.DataFrame, num_iteration: int) -> np.ndarray:
        del num_iteration
        return matrix.iloc[:, 0].to_numpy(dtype=float)


def test_prediction_contract_excludes_labels_and_persists_atomic_shards(tmp_path) -> None:
    features = _feature_rows(pd.to_datetime(["2025-01-02"]), ["A", "B", "C"])
    model = LightGBMModel(
        booster=_DummyBooster(),
        candidate=DEFAULT_CANDIDATES[0],
        feature_columns=FEATURE_COLUMNS,
        seed=42,
        boosting_rounds=10,
        training_rows=100,
        training_label_end_max=pd.Timestamp("2024-12-31"),
        fit_seconds=0.1,
    )
    predictions = predict_lightgbm(model, features)
    assert list(predictions.columns) == [
        "model",
        "seed",
        "ticker",
        "origin",
        "horizon",
        "score",
        "status",
    ]
    assert "target" not in predictions
    assert predictions["horizon"].eq(10).all()
    assert predictions["status"].eq("ok").all()

    path = tmp_path / "shard.parquet"
    write_prediction_artifact(predictions, path)
    loaded = read_prediction_artifacts([path])
    pd.testing.assert_frame_equal(loaded, predictions)
    with pytest.raises(FileExistsError):
        write_prediction_artifact(predictions, path)
    with pytest.raises(ValueError, match="duplicate full keys"):
        read_prediction_artifacts([path, path])


@pytest.mark.skipif(
    importlib.util.find_spec("lightgbm") is None,
    reason="pinned LightGBM dependency is not installed in the bootstrap interpreter",
)
def test_small_real_lightgbm_search_and_declared_refit() -> None:
    tickers = ["A", "B", "C", "D", "E", "F"]
    train = _feature_rows(pd.bdate_range("2023-10-02", periods=20), tickers)
    validation = _feature_rows(pd.bdate_range("2024-07-01", periods=10), tickers)
    candidate = CandidateConfig("tiny", 0.1, 7, 5)
    tuning = tune_lightgbm(
        train,
        validation,
        candidates=[candidate],
        config=SearchConfig(
            max_boost_rounds=30,
            early_stopping_rounds=5,
            minimum_daily_coverage=5,
            num_threads=2,
        ),
    )
    assert tuning.selected.status == "ok"
    assert 1 <= tuning.selected.best_iteration <= 30
    assert len(tuning.records()) == 1

    refit = pd.concat([train, validation], ignore_index=True)
    final = fit_final_lightgbm(refit, tuning)
    predictions = predict_lightgbm(final, validation)
    assert len(predictions) == len(validation)
    assert predictions["score"].map(np.isfinite).all()
