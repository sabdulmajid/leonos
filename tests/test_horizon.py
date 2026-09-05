from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from leonos.features import FEATURE_COLUMNS
from leonos.horizon import (
    HORIZON_CANDIDATES,
    KRONOS_MAX_CONTEXT_SESSIONS,
    KRONOS_MAX_OUTPUT_SESSIONS,
    MAX_SEARCH_CONFIGS_PER_HORIZON,
    SUPPORTED_HORIZONS,
    HorizonContractError,
    HorizonKronosAdapter,
    HorizonKronosConfig,
    UnsupportedHorizonError,
    build_horizon_targets,
    chronological_train_validation_split,
    collapse_kronos_horizon_scores,
    fit_final_horizon_lightgbm,
    kronos_horizon_capabilities,
    predict_horizon_lightgbm,
    tune_horizon_lightgbm,
    validate_horizon,
    validate_horizon_search_splits,
    validate_search_candidates,
    validation_metrics_by_instrument,
)
from leonos.models.kronos import ForecastRequest
from leonos.models.lightgbm import CandidateConfig, LightGBMModel, SearchConfig


def _close_bars(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    close = 100.0 + np.arange(len(calendar), dtype=float)
    return pd.DataFrame({"ticker": "ALFA", "session": calendar, "close": close})


def test_supported_horizons_and_public_kronos_limits_are_explicit() -> None:
    assert SUPPORTED_HORIZONS == (21, 63, 126, 252)
    assert len(HORIZON_CANDIDATES) == MAX_SEARCH_CONFIGS_PER_HORIZON == 8
    capabilities = kronos_horizon_capabilities()
    assert capabilities.supported_horizons == SUPPORTED_HORIZONS
    assert capabilities.max_context_sessions == KRONOS_MAX_CONTEXT_SESSIONS == 512
    assert capabilities.max_output_sessions == KRONOS_MAX_OUTPUT_SESSIONS == 512
    assert capabilities.exposes_sample_paths is False
    assert "averages" in capabilities.output_semantics

    for horizon in SUPPORTED_HORIZONS:
        assert validate_horizon(horizon) == horizon
    with pytest.raises(UnsupportedHorizonError, match="supported horizons"):
        validate_horizon(10)
    with pytest.raises(UnsupportedHorizonError, match="technical output maximum is 512"):
        validate_horizon(513)


def test_horizon_targets_are_exchange_session_exact_and_matured() -> None:
    calendar = pd.bdate_range("2023-01-02", periods=320)
    bars = _close_bars(calendar)
    origins = [calendar[100], calendar[170]]
    as_of = calendar[200]

    actual = build_horizon_targets(
        bars,
        calendar,
        horizons=[21, 63],
        context_sessions=5,
        label_as_of=as_of,
        origin_dates=origins,
    )

    assert actual["label_end"].le(as_of).all()
    assert set(actual["horizon_sessions"]) == {21, 63}
    # The later 63-session label has not matured at the declared as-of session.
    assert not (actual["origin"].eq(calendar[170]) & actual["horizon_sessions"].eq(63)).any()
    row = actual.loc[actual["origin"].eq(calendar[100]) & actual["horizon_sessions"].eq(21)].iloc[0]
    terminal = bars.loc[121, "close"] / bars.loc[100, "close"] - 1.0
    path_mean = bars.loc[101:121, "close"].mean() / bars.loc[100, "close"] - 1.0
    assert terminal != pytest.approx(path_mean)
    assert row["target"] == pytest.approx(terminal)
    assert row["target_kind"] == "terminal_close_return"
    assert row["execution_session"] == calendar[101]
    assert row["label_end"] == calendar[121]

    # Bars that were not observable as of the label cutoff cannot alter any
    # retained label.
    mutated = bars.copy()
    mutated.loc[mutated["session"] > as_of, "close"] *= 10_000.0
    after = build_horizon_targets(
        mutated,
        calendar,
        horizons=[21, 63],
        context_sessions=5,
        label_as_of=as_of,
        origin_dates=origins,
    )
    pdt.assert_frame_equal(actual, after)


def test_terminal_label_needs_endpoint_not_intermediate_future_bars() -> None:
    calendar = pd.bdate_range("2023-01-02", periods=100)
    bars = _close_bars(calendar)
    origin = calendar[50]
    terminal = calendar[71]
    bars = bars.loc[bars["session"] != calendar[60]].copy()

    label = build_horizon_targets(
        bars,
        calendar,
        horizons=21,
        context_sessions=5,
        label_as_of=terminal,
        origin_dates=[origin],
    )
    assert len(label) == 1
    assert label.loc[0, "label_end"] == terminal
    assert bool(label.loc[0, "label_complete"])
    assert label.loc[0, "target"] == pytest.approx((171.0 / 150.0) - 1.0)


def _split_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["ALFA"] * 4,
            "origin": pd.to_datetime(["2022-01-03", "2023-12-20", "2024-01-02", "2024-05-01"]),
            "label_end": pd.to_datetime(["2022-02-01", "2024-01-15", "2024-02-01", "2024-06-03"]),
            "target": [0.01, 0.02, 0.03, 0.04],
            "horizon_sessions": [21] * 4,
        }
    )


def test_chronological_split_purges_overlap_and_immature_validation_labels() -> None:
    train, validation = chronological_train_validation_split(
        _split_rows(),
        validation_origin_start="2024-01-01",
        validation_origin_end="2024-05-31",
        label_as_of="2024-05-31",
        horizon_sessions=21,
    )
    assert train["origin"].tolist() == [pd.Timestamp("2022-01-03")]
    assert validation["origin"].tolist() == [pd.Timestamp("2024-01-02")]
    assert train["label_end"].max() < validation["origin"].min()
    assert train["split"].eq("training").all()
    assert validation["split"].eq("validation").all()
    validate_horizon_search_splits(
        train,
        validation,
        horizon_sessions=21,
        label_as_of="2024-05-31",
    )

    leaked = train.copy()
    leaked["label_end"] = validation["origin"].min()
    with pytest.raises(HorizonContractError, match="must end before"):
        validate_horizon_search_splits(
            leaked,
            validation,
            horizon_sessions=21,
            label_as_of="2024-05-31",
        )


def test_candidate_search_rejects_more_than_eight_or_duplicate_ids() -> None:
    too_many = [CandidateConfig(f"c{number}", 0.05, 7, 5) for number in range(9)]
    with pytest.raises(HorizonContractError, match="between one and 8"):
        validate_search_candidates(too_many)
    duplicate = [
        CandidateConfig("same", 0.05, 7, 5),
        CandidateConfig("same", 0.02, 15, 5),
    ]
    with pytest.raises(HorizonContractError, match="identifiers must be unique"):
        validate_search_candidates(duplicate)


class _DummyBooster:
    def predict(self, matrix: pd.DataFrame, num_iteration: int) -> np.ndarray:
        del num_iteration
        return matrix.iloc[:, 0].to_numpy(dtype=float)


def test_generic_lightgbm_prediction_keeps_labels_out_and_sets_horizon() -> None:
    features = pd.DataFrame(
        {
            "ticker": ["ALFA", "BRAVO"],
            "origin": pd.to_datetime(["2025-01-02", "2025-01-02"]),
            "target": [999.0, 999.0],
            **{
                feature: np.array([0.1, 0.2]) + number * 1e-4
                for number, feature in enumerate(FEATURE_COLUMNS)
            },
        }
    )
    model = LightGBMModel(
        booster=_DummyBooster(),
        candidate=HORIZON_CANDIDATES[0],
        feature_columns=FEATURE_COLUMNS,
        seed=42,
        boosting_rounds=3,
        training_rows=10,
        training_label_end_max=pd.Timestamp("2024-12-31"),
        fit_seconds=0.01,
    )
    prediction = predict_horizon_lightgbm(
        model,
        features,
        horizon_sessions=126,
    )
    assert prediction["horizon"].eq(126).all()
    assert prediction["score"].tolist() == [0.1, 0.2]
    assert prediction["status"].eq("ok").all()
    assert "target" not in prediction


def test_validation_metrics_report_instrument_coverage_and_constant_score_na() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    labels = pd.DataFrame(
        {
            "ticker": ["ALFA"] * 3 + ["BRAVO"] * 3,
            "origin": list(dates) * 2,
            "horizon_sessions": [21] * 6,
            "target": [0.0, 0.1, 0.2, -0.1, 0.0, 0.1],
        }
    )
    predictions = pd.DataFrame(
        {
            "model": ["lightgbm"] * 5,
            "seed": [42] * 5,
            "ticker": ["ALFA"] * 3 + ["BRAVO"] * 2,
            "origin": [*dates, dates[0], dates[1]],
            "horizon": [21] * 5,
            "score": [0.1, 0.1, 0.1, -0.1, 0.0],
            "status": ["ok"] * 5,
        }
    )
    metrics = validation_metrics_by_instrument(predictions, labels).set_index("ticker")
    assert metrics.loc["ALFA", "coverage"] == 1.0
    assert metrics.loc["ALFA", "scored"] == 3
    assert np.isnan(metrics.loc["ALFA", "spearman_correlation"])
    assert metrics.loc["BRAVO", "coverage"] == pytest.approx(2 / 3)
    assert metrics.loc["BRAVO", "mae"] == 0.0


class _FakeModule:
    def __init__(self) -> None:
        self.training = True

    def eval(self) -> _FakeModule:
        self.training = False
        return self

    def requires_grad_(self, value: bool) -> _FakeModule:
        assert value is False
        return self


class _PointPathPredictor:
    def __init__(self, max_context: int = 512) -> None:
        self.model = _FakeModule()
        self.tokenizer = _FakeModule()
        self.max_context = max_context
        self.calls: list[dict[str, object]] = []

    def predict_batch(self, **kwargs: object) -> list[pd.DataFrame]:
        self.calls.append(kwargs)
        frames = kwargs["df_list"]
        dates = kwargs["y_timestamp_list"]
        pred_len = int(kwargs["pred_len"])
        assert isinstance(frames, list)
        assert isinstance(dates, list)
        outputs = []
        for frame, future_dates in zip(frames, dates, strict=True):
            current = float(frame["close"].iloc[-1])
            close = current * np.linspace(1.001, 1.252, pred_len)
            outputs.append(pd.DataFrame({"close": close}, index=pd.DatetimeIndex(future_dates)))
        return outputs


def _forecast_request(context: int, horizon: int) -> ForecastRequest:
    all_dates = pd.bdate_range("2023-01-02", periods=context + horizon)
    history_dates = all_dates[:context]
    future_dates = all_dates[context:]
    close = 100.0 + np.arange(context, dtype=float)
    history = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000.0 + np.arange(context),
            # The adapter must strip both columns before calling upstream.
            "target": np.arange(context),
            "future_close": close + 50_000.0,
        }
    )
    return ForecastRequest("ALFA", history_dates[-1], history, history_dates, future_dates)


def test_kronos_252_step_point_path_smoke_uses_unmodified_batch_surface() -> None:
    predictor = _PointPathPredictor()
    seeds: list[int] = []
    clock = iter([10.0, 10.25])
    adapter = HorizonKronosAdapter(
        predictor,
        HorizonKronosConfig(horizon_sessions=252, context_sessions=8),
        seed_setter=seeds.append,
        clock=lambda: next(clock),
    )
    path = adapter.predict_batch([_forecast_request(8, 252)], split="validation")
    assert len(path) == 252
    assert path["forecast_step"].tolist() == list(range(1, 253))
    assert path["target_horizon"].eq(252).all()
    assert path["samples_exposed"].eq(False).all()  # noqa: E712
    assert seeds == [int(path["effective_seed"].iloc[0])]
    call = predictor.calls[0]
    assert call["pred_len"] == 252
    assert call["sample_count"] == 10
    assert list(call["df_list"][0].columns) == ["open", "high", "low", "close", "volume"]
    score = collapse_kronos_horizon_scores(path)
    assert len(score) == 1
    assert score.loc[0, "horizon"] == 252
    assert score.loc[0, "status"] == "ok"
    terminal_return = path.loc[path["forecast_step"].eq(252), "predicted_close"].iloc[0]
    terminal_return = terminal_return / path["current_close"].iloc[0] - 1.0
    mean_path_return = path["predicted_close"].mean() / path["current_close"].iloc[0] - 1.0
    assert score.loc[0, "score"] == pytest.approx(terminal_return)
    assert score.loc[0, "score"] != pytest.approx(mean_path_return)
    assert not {"target", "future_close"}.intersection(path.columns)

    with pytest.raises(HorizonContractError, match="exposes only a point path"):
        adapter.predict_batch(
            [_forecast_request(8, 252)],
            split="validation",
            return_samples=True,
        )


def test_kronos_rejects_horizon_above_actual_predictor_decode_limit() -> None:
    with pytest.raises(UnsupportedHorizonError, match="output limit 128"):
        HorizonKronosAdapter(
            _PointPathPredictor(max_context=128),
            HorizonKronosConfig(horizon_sessions=252, context_sessions=8),
            seed_setter=lambda _: None,
        )


def _model_rows(dates: pd.DatetimeIndex, tickers: list[str], horizon: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_number, date in enumerate(dates):
        for ticker_number, ticker in enumerate(tickers):
            signal = ticker_number / (len(tickers) - 1) + date_number * 0.002
            row: dict[str, object] = {
                "ticker": ticker,
                "origin": date,
                "label_end": pd.Timestamp(date) + pd.offsets.Day(40),
                "horizon_sessions": horizon,
                "target": signal * 0.08 - 0.03,
            }
            for feature_number, feature in enumerate(FEATURE_COLUMNS):
                row[feature] = signal + feature_number * 1e-4
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.skipif(
    importlib.util.find_spec("lightgbm") is None,
    reason="pinned LightGBM dependency is unavailable",
)
def test_tiny_real_horizon_lightgbm_search_refit_and_prediction() -> None:
    tickers = ["ALFA", "BRAVO", "CHARLIE", "DELTA"]
    train = _model_rows(pd.bdate_range("2022-01-03", periods=12), tickers, 21)
    validation = _model_rows(pd.bdate_range("2024-01-02", periods=6), tickers, 21)
    candidate = CandidateConfig("tiny", 0.1, 7, 2)
    tuning = tune_horizon_lightgbm(
        train,
        validation,
        horizon_sessions=21,
        label_as_of="2024-12-31",
        candidates=[candidate],
        config=SearchConfig(
            max_boost_rounds=20,
            early_stopping_rounds=5,
            minimum_daily_coverage=3,
            num_threads=1,
        ),
    )
    assert tuning.selected.status == "ok"
    refit = pd.concat([train, validation], ignore_index=True)
    model = fit_final_horizon_lightgbm(
        refit,
        tuning,
        horizon_sessions=21,
        label_as_of="2024-12-31",
    )
    assert model.horizon_sessions == 21
    assert model.target_kind == "terminal_close_return"
    prediction = predict_horizon_lightgbm(
        model,
        validation,
        horizon_sessions=21,
    )
    assert len(prediction) == len(validation)
    assert prediction["score"].map(np.isfinite).all()
    with pytest.raises(HorizonContractError, match="does not match prediction horizon"):
        predict_horizon_lightgbm(model, validation, horizon_sessions=63)
