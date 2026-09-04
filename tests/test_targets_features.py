from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from leonos.features import (
    FEATURE_COLUMNS,
    MAX_FEATURE_LOOKBACK,
    build_ohlcv_features,
    feature_manifest,
    validate_feature_frame,
)
from leonos.targets import (
    SplitSpec,
    TargetSpec,
    apply_split,
    build_targets,
    extract_context,
)


def _bars(calendar: pd.DatetimeIndex, tickers: tuple[str, ...] = ("AAA",)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker_number, ticker in enumerate(tickers):
        close = 100.0 + ticker_number * 10.0 + np.arange(len(calendar), dtype=float) * 0.2
        for date, price, number in zip(calendar, close, range(len(calendar)), strict=True):
            rows.append(
                {
                    "ticker": ticker,
                    "session": date,
                    "open": price * (1.0 - 0.001 * ((number % 3) - 1)),
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1_000_000.0 + number * 1_000.0 + ticker_number,
                }
            )
    return pd.DataFrame(rows)


def test_exact_average_future_close_target_uses_explicit_sessions() -> None:
    # The four-day date gap is intentional: horizons count supplied exchange
    # sessions, not weekdays or calendar days.
    calendar = pd.to_datetime(
        ["2024-07-01", "2024-07-02", "2024-07-03", "2024-07-08", "2024-07-09"]
    )
    bars = pd.DataFrame(
        {
            "ticker": "AAA",
            "session": calendar,
            "close": [10.0, 11.0, 12.0, 13.0, 15.0],
        }
    )
    actual = build_targets(
        bars,
        calendar,
        spec=TargetSpec(context_sessions=3, horizon_sessions=2),
    )
    assert len(actual) == 1
    row = actual.iloc[0]
    assert row["origin"] == pd.Timestamp("2024-07-03")
    assert row["execution_session"] == pd.Timestamp("2024-07-08")
    assert row["label_end"] == pd.Timestamp("2024-07-09")
    assert row["target"] == pytest.approx(((13.0 + 15.0) / 2.0) / 12.0 - 1.0)


def test_missing_exchange_session_invalidates_label_instead_of_row_shifting() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=6)
    bars = pd.DataFrame(
        {"ticker": "AAA", "session": calendar, "close": np.arange(10.0, 16.0)}
    )
    bars = bars.loc[bars["session"] != calendar[3]].copy()
    audited = build_targets(
        bars,
        calendar,
        spec=TargetSpec(context_sessions=3, horizon_sessions=2),
        origin_dates=[calendar[2]],
        include_incomplete=True,
    )
    assert len(audited) == 1
    assert not bool(audited.loc[0, "label_complete"])
    assert np.isnan(audited.loc[0, "target"])


def test_split_purges_by_actual_label_end() -> None:
    targets = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "origin": pd.to_datetime(["2023-12-15", "2023-12-20", "2024-07-01"]),
            "label_end": pd.to_datetime(["2023-12-29", "2024-01-05", "2024-07-15"]),
            "target": [0.0, 0.0, 0.0],
        }
    )
    selected = apply_split(
        targets, SplitSpec("development", label_end_max="2023-12-31")
    )
    assert selected["origin"].tolist() == [pd.Timestamp("2023-12-15")]


def test_future_price_mutation_does_not_change_context_or_features() -> None:
    calendar = pd.bdate_range("2023-01-02", periods=110)
    origin = calendar[94]
    bars = _bars(calendar)
    keys = pd.DataFrame({"ticker": ["AAA"], "origin": [origin]})

    context_before = extract_context(bars, calendar, ticker="AAA", origin=origin)
    features_before = build_ohlcv_features(bars, calendar, keys=keys)

    mutated = bars.copy()
    future = mutated["session"] > origin
    mutated.loc[future, ["open", "high", "low", "close", "volume"]] *= 1000.0
    context_after = extract_context(mutated, calendar, ticker="AAA", origin=origin)
    features_after = build_ohlcv_features(mutated, calendar, keys=keys)

    pdt.assert_frame_equal(context_before, context_after)
    pdt.assert_frame_equal(features_before, features_after)


def test_feature_inventory_is_ohlcv_only_and_inside_context() -> None:
    manifest = feature_manifest()
    assert manifest["feature_count"] == len(FEATURE_COLUMNS)
    assert manifest["uses_ticker_identity"] is False
    assert manifest["uses_amount_or_vwap"] is False
    assert MAX_FEATURE_LOOKBACK <= 90
    assert not {"ticker", "amount", "vwap", "target"}.intersection(FEATURE_COLUMNS)


def test_missing_bar_inside_90_session_context_is_not_silently_bridged() -> None:
    calendar = pd.bdate_range("2023-01-02", periods=100)
    bars = _bars(calendar)
    origin = calendar[94]
    keys = pd.DataFrame({"ticker": ["AAA"], "origin": [origin]})
    bars = bars.loc[bars["session"] != calendar[20]].copy()

    usable = build_ohlcv_features(bars, calendar, keys=keys)
    audited = build_ohlcv_features(
        bars, calendar, keys=keys, include_incomplete=True
    )
    assert usable.empty
    assert len(audited) == 1
    assert not bool(audited.loc[0, "context_complete"])


def test_feature_frame_validator_rejects_unintended_inventory() -> None:
    frame = pd.DataFrame(columns=FEATURE_COLUMNS)
    validate_feature_frame(frame)
    with pytest.raises(ValueError, match="inventory mismatch"):
        validate_feature_frame(frame, (*FEATURE_COLUMNS, "ticker"))


def test_duplicate_market_keys_fail_closed() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=5)
    bars = pd.DataFrame(
        {
            "ticker": ["AAA"] * 6,
            "session": [*calendar, calendar[0]],
            "close": [10.0] * 6,
        }
    )
    with pytest.raises(ValueError, match="duplicate ticker/session"):
        build_targets(
            bars,
            calendar,
            spec=TargetSpec(context_sessions=2, horizon_sessions=1),
        )
