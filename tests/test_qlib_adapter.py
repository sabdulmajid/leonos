from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import leonos.qlib_adapter as adapter
from leonos.qlib_adapter import (
    QLIB_REVISION,
    InvalidOpenFillError,
    QlibAdapterError,
    QlibBacktestSpec,
    build_information_date_signal,
    make_open_only_exchange,
    order_indicator_metric_series,
    read_day_binary,
    run_qlib_topk_backtest,
    verify_topk_shift_contract,
    write_qlib_day_dataset,
)


def fixture_bars() -> pd.DataFrame:
    sessions = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"])
    return pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "session": sessions[2],
                "open": 11.5,
                "high": 12.5,
                "low": 11.0,
                "close": 12.0,
                "volume": 1_200.0,
            },
            {
                "ticker": "AAPL",
                "session": sessions[0],
                "open": 9.5,
                "high": 10.5,
                "low": 9.0,
                "close": 10.0,
                "volume": 1_000.0,
            },
            {
                "ticker": "AAPL",
                "session": sessions[1],
                "open": 10.5,
                "high": 11.5,
                "low": 10.0,
                "close": 11.0,
                "volume": 1_100.0,
            },
            {
                "ticker": "BRK.B",
                "session": sessions[1],
                "open": 49.0,
                "high": 51.0,
                "low": 48.0,
                "close": 50.0,
                "volume": 500.0,
            },
            {
                "ticker": "BRK.B",
                "session": sessions[3],
                "open": 54.0,
                "high": 56.0,
                "low": 53.0,
                "close": 55.0,
                "volume": 550.0,
            },
        ]
    )


def relative_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_day_writer_matches_qlib_binary_layout_and_adjusted_basis(tmp_path: Path) -> None:
    calendar = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"])
    destination = tmp_path / "qlib"
    manifest = write_qlib_day_dataset(
        fixture_bars(), destination, calendar=calendar, price_basis="verified_split_adjusted"
    )

    assert manifest["qlib_revision"] == QLIB_REVISION
    assert manifest["factor"] == 1.0
    assert manifest["price_normalization"] == "none"
    assert (destination / "calendars/day.txt").read_text().splitlines() == [
        "2025-01-02",
        "2025-01-03",
        "2025-01-06",
        "2025-01-07",
    ]
    assert (destination / "instruments/all.txt").read_text().splitlines() == [
        "AAPL\t2025-01-02\t2025-01-06",
        "BRK.B\t2025-01-03\t2025-01-07",
    ]

    start, close = read_day_binary(destination / "features/aapl/close.day.bin")
    assert start == 0
    np.testing.assert_allclose(close, [10.0, 11.0, 12.0])
    _, factor = read_day_binary(destination / "features/aapl/factor.day.bin")
    np.testing.assert_array_equal(factor, np.ones(3, dtype=np.float32))
    _, change = read_day_binary(destination / "features/aapl/change.day.bin")
    np.testing.assert_allclose(change[1:], [0.1, 12.0 / 11.0 - 1.0], rtol=1e-6)
    assert np.isnan(change[0])

    start, brk_close = read_day_binary(destination / "features/brk.b/close.day.bin")
    assert start == 1
    np.testing.assert_allclose(brk_close[[0, 2]], [50.0, 55.0])
    assert np.isnan(brk_close[1])  # Missing exchange session remains missing, never filled.
    _, brk_change = read_day_binary(destination / "features/brk.b/change.day.bin")
    assert np.isnan(brk_change).all()  # No two-session return masquerades as a daily change.


def test_day_writer_is_deterministic_and_refuses_replacement(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest_one = write_qlib_day_dataset(fixture_bars(), first)
    manifest_two = write_qlib_day_dataset(fixture_bars().iloc[::-1], second)
    assert manifest_one == manifest_two
    assert relative_bytes(first) == relative_bytes(second)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_qlib_day_dataset(fixture_bars(), first)


def test_day_writer_rejects_duplicate_and_case_colliding_tickers(tmp_path: Path) -> None:
    duplicated = pd.concat([fixture_bars(), fixture_bars().iloc[[0]]], ignore_index=True)
    with pytest.raises(QlibAdapterError, match="duplicate"):
        write_qlib_day_dataset(duplicated, tmp_path / "duplicates")

    collided = fixture_bars()
    extra = collided[collided["ticker"] == "AAPL"].copy()
    extra["ticker"] = "aapl"
    extra["session"] += pd.Timedelta(30, unit="D")
    with pytest.raises(QlibAdapterError, match="collide"):
        write_qlib_day_dataset(pd.concat([collided, extra]), tmp_path / "collision")


def test_signal_keeps_information_date_unshifted() -> None:
    scores = pd.DataFrame(
        {
            "ticker": ["MSFT", "AAPL", "MSFT", "AAPL"],
            "origin": pd.to_datetime(["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"]),
            "score": [0.2, 0.1, -0.1, 0.3],
        }
    )
    original = scores.copy(deep=True)
    signal = build_information_date_signal(scores)
    assert signal.index.names == ["datetime", "instrument"]
    assert signal.loc[(pd.Timestamp("2025-01-02"), "MSFT")] == 0.0
    assert signal.loc[(pd.Timestamp("2025-01-02"), "AAPL")] == -1.0
    assert signal.index.get_level_values("datetime").min() == pd.Timestamp("2025-01-02")
    assert pd.Timestamp("2025-01-04") not in signal.index.get_level_values("datetime")
    pd.testing.assert_frame_equal(scores, original)


def test_eight_way_cutoff_tie_is_ticker_ascending_and_permutation_stable() -> None:
    tickers = ["HHH", "AAA", "FFF", "CCC", "BBB", "GGG", "EEE", "DDD"]
    origin = pd.Timestamp("2025-01-02")
    scores = pd.DataFrame({"ticker": tickers, "origin": origin, "score": 0.0})

    selections = []
    for random_state in (1, 2, 3, 4):
        shuffled = scores.sample(frac=1.0, random_state=random_state)
        signal = build_information_date_signal(shuffled).xs(origin)
        assert signal.is_unique
        selections.append(signal.nlargest(5).index.tolist())

    assert selections == [["AAA", "BBB", "CCC", "DDD", "EEE"]] * 4


def test_tie_break_is_stable_across_python_hash_seed_processes() -> None:
    code = """
import json
import pandas as pd
from leonos.qlib_adapter import build_information_date_signal
tickers = set(['HHH','AAA','FFF','CCC','BBB','GGG','EEE','DDD'])
frame = pd.DataFrame({'ticker': list(tickers), 'origin': pd.Timestamp('2025-01-02'), 'score': 0.0})
signal = build_information_date_signal(frame).xs(pd.Timestamp('2025-01-02'))
print(json.dumps(signal.nlargest(5).index.tolist()))
"""
    selections = []
    for seed in ("1", "987654"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        output = subprocess.check_output([sys.executable, "-c", code], text=True, env=environment)
        selections.append(json.loads(output))
    assert selections == [["AAA", "BBB", "CCC", "DDD", "EEE"]] * 2


def test_exact_us_exchange_settings_are_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    class CapturingExchange:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(adapter, "open_only_exchange_class", lambda: CapturingExchange)
    exchange = make_open_only_exchange(
        start_time="2025-01-03", end_time="2025-01-31", codes=["MSFT", "AAPL"]
    )
    assert exchange.kwargs == {
        "freq": "day",
        "start_time": pd.Timestamp("2025-01-03"),
        "end_time": pd.Timestamp("2025-01-31"),
        "codes": ["AAPL", "MSFT"],
        "deal_price": "$open",
        "open_cost": 0.0005,
        "close_cost": 0.0005,
        "min_cost": 0.0,
        "impact_cost": 0.0,
        "limit_threshold": None,
        "volume_threshold": None,
        "trade_unit": 1,
    }
    assert QlibBacktestSpec(cost_bps_per_side=15).proportional_cost == 0.0015
    with pytest.raises(QlibAdapterError, match="declared"):
        QlibBacktestSpec(cost_bps_per_side=7)


def test_open_only_exchange_rejects_invalid_open_without_close_fallback() -> None:
    pytest.importorskip("qlib", reason="locked Qlib extra is optional in CPU tests")
    exchange_class = adapter.open_only_exchange_class()

    class Quote:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get_data(self, *args: object, field: str, **kwargs: object) -> float:
            self.calls.append(field)
            return np.nan

    exchange = object.__new__(exchange_class)
    exchange.quote = Quote()
    now = pd.Timestamp("2025-01-03")
    assert exchange.is_stock_tradable("AAPL", now, now) is False
    with pytest.raises(InvalidOpenFillError, match="invalid next-open"):
        exchange.get_deal_price("AAPL", now, now, direction=1)
    assert exchange.quote.calls == ["$open", "$open"]
    assert "$close" not in exchange.quote.calls


def test_installed_topk_uses_exactly_one_prior_signal_step() -> None:
    pytest.importorskip("qlib", reason="locked Qlib extra is optional in CPU tests")
    verify_topk_shift_contract()


def test_qlib_executes_information_date_signal_at_next_open(tmp_path: Path) -> None:
    pytest.importorskip("qlib", reason="locked Qlib extra is optional in CPU tests")
    sessions = pd.bdate_range("2025-01-02", periods=5)
    rows = []
    tickers = ["HHH", "AAA", "FFF", "CCC", "BBB", "GGG", "EEE", "DDD"]
    for day_index, session in enumerate(sessions):
        for ticker_index, ticker in enumerate(tickers):
            open_price = 100.0 + 10.0 * ticker_index + day_index
            rows.append(
                {
                    "ticker": ticker,
                    "session": session,
                    "open": open_price,
                    "high": open_price + 2.0,
                    "low": open_price - 2.0,
                    "close": open_price + 1.0,
                    "volume": 10_000.0,
                }
            )
    provider = tmp_path / "qlib"
    write_qlib_day_dataset(pd.DataFrame(rows), provider, calendar=sessions)
    scores = pd.DataFrame(
        {
            "ticker": tickers,
            "origin": sessions[0],
            "score": np.zeros(len(tickers), dtype=float),
        }
    )

    outputs = run_qlib_topk_backtest(
        scores,
        provider_uri=provider,
        start_time=sessions[1],
        end_time=sessions[2],
    )
    execution_date = pd.Timestamp(sessions[1])
    assert execution_date in outputs.positions
    selected = sorted(tickers)[:5]
    assert set(outputs.positions[execution_date].get_stock_list()) == set(selected)
    trade_prices = order_indicator_metric_series(
        outputs.order_history[execution_date], "trade_price"
    )
    expected_opens = {ticker: 101.0 + 10.0 * tickers.index(ticker) for ticker in selected}
    for ticker, price in expected_opens.items():
        assert trade_prices.loc[ticker] == pytest.approx(price)
    assert outputs.report.loc[execution_date, "total_cost"] > 0
    assert outputs.report["bench"].isna().all()
