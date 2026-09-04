from __future__ import annotations

import numpy as np
import pandas as pd

from leonos.portfolio import assert_account_reconciles, portfolio_metrics, simulate_topk_dropout


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2025-01-02", periods=9)
    bars = pd.DataFrame(
        [
            {"session": date, "ticker": ticker, "open": price, "close": price + 1}
            for date_index, date in enumerate(dates)
            for ticker, price in (("A", 100 + date_index), ("B", 50 + date_index))
        ]
    )
    # A is selected on day 0; B becomes preferred after A has met its hold threshold.
    scores = pd.DataFrame(
        [
            {"origin": date, "ticker": ticker, "score": score}
            for index, date in enumerate(dates[:-1])
            for ticker, score in (
                ("A", 2.0 if index < 5 else 0.0),
                ("B", 1.0 if index < 5 else 3.0),
            )
        ]
    )
    return bars, scores


def test_signal_executes_once_at_next_open_and_reconciles() -> None:
    bars, scores = _fixture()
    result = simulate_topk_dropout(
        bars, scores, topk=1, n_drop=1, hold_thresh=5, initial_cash=1_000, cost_bps_per_side=5
    )
    assert result.fills.iloc[0]["signal_date"] == bars["session"].min()
    assert result.fills.iloc[0]["execution_date"] == sorted(bars["session"].unique())[1]
    assert result.fills.iloc[0]["price"] == 101
    assert set(result.fills["side"]) == {"buy", "sell"}
    assert_account_reconciles(result)
    metrics = portfolio_metrics(result.account)
    assert np.isfinite(metrics["net_cumulative_return"])
    assert metrics["transaction_costs_dollars"] == result.fills["fee"].sum()


def test_missing_open_rejects_trade_without_using_close() -> None:
    bars, scores = _fixture()
    next_session = sorted(bars["session"].unique())[1]
    bars.loc[(bars["session"] == next_session) & (bars["ticker"] == "A"), "open"] = np.nan
    result = simulate_topk_dropout(bars, scores, topk=1, initial_cash=1_000)
    assert (
        result.fills.empty
        or not (
            (result.fills["ticker"] == "A") & (result.fills["execution_date"] == next_session)
        ).any()
    )
    assert result.rejected_orders.iloc[0]["reason"] == "missing_or_invalid_open"
