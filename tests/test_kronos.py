from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from leonos.models.kronos import (
    AMOUNT_PROXY_POLICY,
    CONTEXT_SESSIONS,
    FORECAST_SESSIONS,
    INVESTMENT_SAMPLE_COUNT,
    INVESTMENT_TEMPERATURE,
    INVESTMENT_TOP_K,
    INVESTMENT_TOP_P,
    ForecastRequest,
    KronosAdapter,
    KronosContractError,
    KronosInferenceConfig,
    PredictionShardError,
    _official_model_package,
    derive_effective_seed,
    iter_completed_origin_keys,
    logical_shard_identity,
    pending_requests,
    prediction_shard_path,
    validate_prediction_frame,
    write_prediction_shard,
)


class FakeModule:
    def __init__(self) -> None:
        self.training = True
        self.eval_calls = 0
        self.requires_grad_calls: list[bool] = []

    def eval(self) -> FakeModule:
        self.eval_calls += 1
        self.training = False
        return self

    def requires_grad_(self, value: bool) -> FakeModule:
        self.requires_grad_calls.append(value)
        return self


class FakeOfficialPredictor:
    def __init__(self) -> None:
        self.model = FakeModule()
        self.tokenizer = FakeModule()
        self.calls: list[dict[str, object]] = []
        self.estimated_amount: list[np.ndarray] = []

    def predict_batch(self, **kwargs: object) -> list[pd.DataFrame]:
        self.calls.append(kwargs)
        frames = kwargs["df_list"]
        future_dates = kwargs["y_timestamp_list"]
        assert isinstance(frames, list)
        assert isinstance(future_dates, list)
        results = []
        for frame, dates in zip(frames, future_dates, strict=True):
            assert isinstance(frame, pd.DataFrame)
            assert isinstance(dates, pd.Series)
            if "amount" not in frame:
                proxy = frame["volume"] * frame[["open", "high", "low", "close"]].mean(axis=1)
                self.estimated_amount.append(proxy.to_numpy())
            current_close = float(frame["close"].iloc[-1])
            closes = current_close * np.linspace(1.01, 1.10, FORECAST_SESSIONS)
            results.append(pd.DataFrame({"close": closes}, index=pd.DatetimeIndex(dates)))
        return results


def make_request(
    ticker: str = "AAPL", shift: int = 0, *, with_amount: bool = False
) -> ForecastRequest:
    dates = pd.bdate_range("2024-01-02", periods=CONTEXT_SESSIONS + shift)[shift:]
    base = np.arange(CONTEXT_SESSIONS, dtype=float) + 100.0 + shift
    history = pd.DataFrame(
        {
            "open": base,
            "high": base + 2.0,
            "low": base - 2.0,
            "close": base + 1.0,
            "volume": np.arange(CONTEXT_SESSIONS, dtype=float) + 1_000.0,
            # These must never cross the model boundary.
            "label": np.linspace(-1.0, 1.0, CONTEXT_SESSIONS),
            "future_close": base + 50_000.0,
        }
    )
    if with_amount:
        history["amount"] = history["volume"] * history["close"]
    future = pd.bdate_range(dates[-1] + pd.Timedelta(1, unit="D"), periods=FORECAST_SESSIONS)
    return ForecastRequest(ticker, dates[-1], history, dates, future)


def run_fake(
    requests: list[ForecastRequest],
) -> tuple[pd.DataFrame, FakeOfficialPredictor, list[int]]:
    predictor = FakeOfficialPredictor()
    seeds: list[int] = []
    times = iter([100.0, 100.25])
    adapter = KronosAdapter(predictor, seed_setter=seeds.append, clock=lambda: next(times))
    return adapter.predict_batch(requests, split="validation"), predictor, seeds


def test_fixed_contract_and_official_call_are_enforced() -> None:
    request = make_request()
    output, predictor, seeds = run_fake([request])

    assert predictor.model.training is False
    assert predictor.tokenizer.training is False
    assert predictor.model.requires_grad_calls == [False]
    assert predictor.tokenizer.requires_grad_calls == [False]
    call = predictor.calls[0]
    assert call["pred_len"] == FORECAST_SESSIONS
    assert call["T"] == INVESTMENT_TEMPERATURE
    assert call["top_p"] == INVESTMENT_TOP_P
    assert call["top_k"] == INVESTMENT_TOP_K  # 0 is upstream's disabled sentinel.
    assert call["sample_count"] == INVESTMENT_SAMPLE_COUNT
    assert call["verbose"] is False
    assert len(call["df_list"][0]) == CONTEXT_SESSIONS
    assert list(call["df_list"][0].columns) == ["open", "high", "low", "close", "volume"]
    assert len(call["y_timestamp_list"][0]) == FORECAST_SESSIONS

    assert seeds == [int(output["effective_seed"].iloc[0])]
    assert len(output) == FORECAST_SESSIONS
    assert output["horizon"].tolist() == list(range(1, FORECAST_SESSIONS + 1))
    assert output["input_end"].eq(request.origin).all()
    expected_score = np.linspace(1.01, 1.10, FORECAST_SESSIONS).mean() - 1.0
    assert output["score"].iloc[0] == pytest.approx(expected_score)
    assert output["amount_source"].unique().tolist() == [
        "estimated_ohlcv_proxy_by_official_predictor"
    ]
    expected_proxy = request.history["volume"] * request.history[
        ["open", "high", "low", "close"]
    ].mean(axis=1)
    np.testing.assert_allclose(predictor.estimated_amount[0], expected_proxy)
    assert "observed dollar turnover" in AMOUNT_PROXY_POLICY
    assert not {"label", "future_close"}.intersection(output.columns)


@pytest.mark.parametrize(
    ("changed", "value"),
    [
        ("context_sessions", 89),
        ("forecast_sessions", 11),
        ("temperature", 1.0),
        ("top_p", 0.95),
        ("top_k", 5),
        ("sample_count", 1),
    ],
)
def test_v1_sampling_contract_cannot_drift(changed: str, value: object) -> None:
    with pytest.raises(KronosContractError, match="settings are fixed"):
        KronosInferenceConfig(**{changed: value})


def test_request_requires_exact_past_and_future_session_shapes() -> None:
    request = make_request()
    bad = ForecastRequest(
        request.ticker,
        request.origin,
        request.history.iloc[1:],
        request.history_dates[1:],
        request.forecast_dates,
    )
    predictor = FakeOfficialPredictor()
    adapter = KronosAdapter(predictor, seed_setter=lambda _: None)
    with pytest.raises(KronosContractError, match="exactly 90"):
        adapter.predict_batch([bad], split="validation")
    assert predictor.calls == []


def test_future_value_mutation_cannot_reach_predictor_or_change_score() -> None:
    request = make_request()
    unrelated_future_values = pd.Series(np.arange(10.0), index=request.forecast_dates)
    first, predictor_one, _ = run_fake([request])
    unrelated_future_values.iloc[:] = 1e12
    second, predictor_two, _ = run_fake([request])

    pd.testing.assert_series_equal(first["score"], second["score"])
    pd.testing.assert_frame_equal(
        predictor_one.calls[0]["df_list"][0], predictor_two.calls[0]["df_list"][0]
    )
    assert all(series.name is None for series in predictor_two.calls[0]["y_timestamp_list"])


def test_shard_identity_is_canonical_and_gpu_independent() -> None:
    first, second = make_request("MSFT"), make_request("AAPL")
    left = logical_shard_identity("test", 42, [first.key, second.key])
    right = logical_shard_identity("test", 42, [second.key, first.key])
    assert left == right
    assert left != logical_shard_identity("test", 43, [first.key, second.key])
    assert left != logical_shard_identity("validation", 42, [first.key, second.key])
    assert derive_effective_seed(42, left) == derive_effective_seed(42, right)
    path = prediction_shard_path(
        Path("artifacts/predictions"), split="test", seed=42, requests=[first, second]
    )
    assert left in path.name
    assert "cuda" not in path.name


def test_atomic_parquet_shard_and_resume_keys(tmp_path: Path) -> None:
    done_request = make_request("AAPL")
    todo_request = make_request("MSFT")
    output, _, _ = run_fake([done_request])
    path = tmp_path / "predictions.parquet"
    write_prediction_shard(output, path)

    restored = pd.read_parquet(path)
    validate_prediction_frame(restored)
    completed = list(iter_completed_origin_keys([path]))
    assert completed == [("kronos", 42, "AAPL", pd.Timestamp(done_request.origin))]
    pending = pending_requests(
        [todo_request, done_request], seed=42, completed_origin_keys=completed
    )
    assert [request.ticker for request in pending] == ["MSFT"]
    assert not list(tmp_path.glob(".*.tmp.parquet"))

    with pytest.raises(PredictionShardError, match="duplicate forecast key across shards"):
        list(iter_completed_origin_keys([path, path]))


def test_prediction_writer_rejects_labels_and_preserves_old_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _, _ = run_fake([make_request()])
    path = tmp_path / "predictions.parquet"
    write_prediction_shard(output, path)
    original_bytes = path.read_bytes()

    contaminated = output.assign(label=0.01)
    with pytest.raises(PredictionShardError, match="stored separately"):
        write_prediction_shard(contaminated, tmp_path / "contaminated.parquet")

    def fail_write(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated interrupted parquet write")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_write)
    with pytest.raises(RuntimeError, match="interrupted"):
        write_prediction_shard(output, path)
    assert path.read_bytes() == original_bytes
    assert not list(tmp_path.glob(".*.tmp.parquet"))


def test_official_forecast_date_reordering_is_rejected() -> None:
    request = make_request()

    class MisalignedPredictor(FakeOfficialPredictor):
        def predict_batch(self, **kwargs: object) -> list[pd.DataFrame]:
            predictions = super().predict_batch(**kwargs)
            predictions[0] = predictions[0].iloc[::-1]
            return predictions

    adapter = KronosAdapter(MisalignedPredictor(), seed_setter=lambda _: None)
    with pytest.raises(KronosContractError, match="strictly increasing|date alignment"):
        adapter.predict_batch([request], split="validation")


def test_official_loader_supports_upstreams_hard_coded_model_import(tmp_path: Path) -> None:
    package = tmp_path / "model"
    package.mkdir()
    (package / "__init__.py").write_text("from .kronos import imported_marker\n")
    (package / "module.py").write_text("marker = 'from-pinned-source'\n")
    # The real pinned kronos.py likewise imports from the absolute top-level `model` package.
    (package / "kronos.py").write_text(
        "from model.module import marker\nimported_marker = marker\n"
    )

    with _official_model_package(tmp_path) as official:
        assert official.imported_marker == "from-pinned-source"

    assert "model" not in __import__("sys").modules
    assert "model.module" not in __import__("sys").modules
