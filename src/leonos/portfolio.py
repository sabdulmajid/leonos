"""Transparent U.S. daily-bar ledger used to verify Qlib portfolio results.

The production report is expected to retain Qlib's raw output.  This small ledger
exists as an independent timing/accounting reconciliation and as a fallback for
worked examples; it deliberately mirrors the declared TopkDropout policy without
China-market defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fill:
    signal_date: pd.Timestamp
    execution_date: pd.Timestamp
    ticker: str
    side: str
    shares: float
    price: float
    gross_value: float
    fee: float
    reason: str


@dataclass
class Holding:
    shares: float
    entry_date: pd.Timestamp
    sessions_held: int = 0


@dataclass(frozen=True)
class LedgerResult:
    account: pd.DataFrame
    fills: pd.DataFrame
    final_positions: dict[str, float]
    rejected_orders: pd.DataFrame


def _validate_inputs(bars: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_bars = {"session", "ticker", "open", "close"}
    required_scores = {"origin", "ticker", "score"}
    missing_bars = required_bars - set(bars.columns)
    missing_scores = required_scores - set(scores.columns)
    if missing_bars or missing_scores:
        raise ValueError(
            f"missing columns: bars={sorted(missing_bars)}, scores={sorted(missing_scores)}"
        )
    clean_bars = bars.copy()
    clean_scores = scores.copy()
    clean_bars["session"] = pd.to_datetime(clean_bars["session"]).dt.tz_localize(None)
    clean_scores["origin"] = pd.to_datetime(clean_scores["origin"]).dt.tz_localize(None)
    if clean_bars.duplicated(["session", "ticker"]).any():
        raise ValueError("duplicate bar keys")
    if clean_scores.duplicated(["origin", "ticker"]).any():
        raise ValueError("duplicate score keys")
    return clean_bars.sort_values(["session", "ticker"]), clean_scores.sort_values(
        ["origin", "ticker"]
    )


def simulate_topk_dropout(
    bars: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    topk: int = 5,
    n_drop: int = 1,
    hold_thresh: int = 5,
    target_exposure: float = 0.95,
    initial_cash: float = 1_000_000.0,
    cost_bps_per_side: float = 5.0,
) -> LedgerResult:
    """Execute each origin's decision at the immediately following session open.

    Selection uses only the saved score at the information date.  Whether an order
    fills is determined solely from the next session's open.  Missing closes may be
    carried for valuation, but a carried value is never tradable.
    """
    if topk <= 0 or n_drop < 0 or hold_thresh < 0:
        raise ValueError("invalid strategy parameters")
    if not (0.0 <= target_exposure <= 1.0) or initial_cash <= 0 or cost_bps_per_side < 0:
        raise ValueError("invalid capital/cost parameters")
    bars, scores = _validate_inputs(bars, scores)
    sessions = pd.Index(bars["session"].drop_duplicates().sort_values())
    if len(sessions) < 2:
        raise ValueError("at least two sessions are required")
    bar_lookup = bars.set_index(["session", "ticker"])
    score_groups = {
        date: frame.set_index("ticker")["score"] for date, frame in scores.groupby("origin")
    }
    cash = float(initial_cash)
    holdings: dict[str, Holding] = {}
    last_marks: dict[str, float] = {}
    fills: list[Fill] = []
    rejects: list[dict[str, Any]] = []
    accounts: list[dict[str, Any]] = []
    fee_rate = cost_bps_per_side / 10_000.0

    for index, session in enumerate(sessions):
        session = pd.Timestamp(session)
        # Age positions once per exchange session before deciding at this open.
        for holding in holdings.values():
            holding.sessions_held += 1

        signal_date = pd.Timestamp(sessions[index - 1]) if index else None
        signal = score_groups.get(signal_date) if signal_date is not None else None
        if signal is not None:
            signal = signal[np.isfinite(signal)].sort_values(ascending=False, kind="mergesort")
            ranked = list(signal.index.astype(str))
            held_ranked = [ticker for ticker in ranked if ticker in holdings]
            held_without_score = sorted(set(holdings) - set(held_ranked))
            current_order = held_ranked + held_without_score
            desired_pool = ranked[: max(topk, len(current_order) + n_drop)]
            bottom = sorted(
                current_order,
                key=lambda ticker: (float(signal.get(ticker, -np.inf)), ticker),
            )
            sell_candidates = [
                ticker
                for ticker in bottom
                if ticker not in desired_pool[:topk]
                and holdings[ticker].sessions_held >= hold_thresh
            ][:n_drop]

            for ticker in sell_candidates:
                key = (session, ticker)
                row = bar_lookup.loc[key] if key in bar_lookup.index else None
                price = float(row["open"]) if row is not None else np.nan
                if not np.isfinite(price) or price <= 0:
                    rejects.append(
                        {
                            "signal_date": signal_date,
                            "execution_date": session,
                            "ticker": ticker,
                            "side": "sell",
                            "reason": "missing_or_invalid_open",
                        }
                    )
                    continue
                holding = holdings.pop(ticker)
                gross = holding.shares * price
                fee = gross * fee_rate
                cash += gross - fee
                fills.append(
                    Fill(
                        signal_date,
                        session,
                        ticker,
                        "sell",
                        holding.shares,
                        price,
                        gross,
                        fee,
                        "topk_dropout",
                    )
                )

            slots = max(0, topk - len(holdings))
            buy_candidates = [ticker for ticker in ranked if ticker not in holdings][:slots]
            if buy_candidates:
                budget_each = cash * target_exposure / len(buy_candidates)
                for ticker in buy_candidates:
                    key = (session, ticker)
                    row = bar_lookup.loc[key] if key in bar_lookup.index else None
                    price = float(row["open"]) if row is not None else np.nan
                    if not np.isfinite(price) or price <= 0:
                        rejects.append(
                            {
                                "signal_date": signal_date,
                                "execution_date": session,
                                "ticker": ticker,
                                "side": "buy",
                                "reason": "missing_or_invalid_open",
                            }
                        )
                        continue
                    # Cost is included in the budget, preventing negative cash.
                    gross = min(budget_each / (1.0 + fee_rate), cash / (1.0 + fee_rate))
                    shares = gross / price
                    fee = gross * fee_rate
                    cash -= gross + fee
                    holdings[ticker] = Holding(shares=shares, entry_date=session, sessions_held=0)
                    fills.append(
                        Fill(
                            signal_date,
                            session,
                            ticker,
                            "buy",
                            shares,
                            price,
                            gross,
                            fee,
                            "initial_or_replacement",
                        )
                    )

        market_value = 0.0
        carried = 0
        for ticker, holding in holdings.items():
            key = (session, ticker)
            row = bar_lookup.loc[key] if key in bar_lookup.index else None
            close = float(row["close"]) if row is not None else np.nan
            if np.isfinite(close) and close > 0:
                last_marks[ticker] = close
            else:
                close = last_marks.get(ticker, np.nan)
                carried += 1
            if not np.isfinite(close):
                raise ValueError(f"no valid valuation for held {ticker} on {session.date()}")
            market_value += holding.shares * close
        fees_today = sum(fill.fee for fill in fills if fill.execution_date == session)
        turnover_today = sum(fill.gross_value for fill in fills if fill.execution_date == session)
        accounts.append(
            {
                "session": session,
                "cash": cash,
                "market_value": market_value,
                "account_value": cash + market_value,
                "fees": fees_today,
                "traded_value": turnover_today,
                "positions": len(holdings),
                "carried_valuations": carried,
            }
        )

    account = pd.DataFrame(accounts)
    account["net_return"] = (
        account["account_value"]
        .pct_change()
        .fillna(account["account_value"].iloc[0] / initial_cash - 1.0)
    )
    fill_frame = pd.DataFrame([asdict(fill) for fill in fills], columns=list(Fill.__annotations__))
    reject_frame = pd.DataFrame(rejects)
    return LedgerResult(
        account=account,
        fills=fill_frame,
        final_positions={ticker: holding.shares for ticker, holding in holdings.items()},
        rejected_orders=reject_frame,
    )


def simulate_equal_weight_buy_hold(
    bars: pd.DataFrame,
    *,
    start_session: object,
    end_session: object,
    target_exposure: float = 0.95,
    initial_cash: float = 1_000_000.0,
    cost_bps_per_side: float = 5.0,
) -> LedgerResult:
    """Buy the first-open eligible basket once and mark it without liquidation.

    Whole shares match the Qlib U.S. backtest's one-share trade unit. Eligibility
    and fills inspect only the first session's open; later missing closes may carry
    the most recent mark but are never treated as tradable quotes.
    """

    if not 0.0 <= target_exposure <= 1.0:
        raise ValueError("target_exposure must be between zero and one")
    if initial_cash <= 0 or cost_bps_per_side < 0:
        raise ValueError("capital must be positive and cost nonnegative")
    required = {"session", "ticker", "open", "close"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {sorted(missing)}")
    clean = bars.loc[:, ["session", "ticker", "open", "close"]].copy()
    clean["session"] = (
        pd.to_datetime(clean["session"], utc=True).dt.tz_convert(None).dt.normalize()
    )
    clean["ticker"] = clean["ticker"].astype(str)
    if clean.duplicated(["session", "ticker"]).any():
        raise ValueError("duplicate bar keys")
    start = pd.Timestamp(start_session).tz_localize(None).normalize()
    end = pd.Timestamp(end_session).tz_localize(None).normalize()
    if end < start:
        raise ValueError("end_session precedes start_session")
    clean = clean.loc[clean["session"].between(start, end)].sort_values(
        ["session", "ticker"], kind="stable"
    )
    sessions = pd.DatetimeIndex(clean["session"].unique()).sort_values()
    if len(sessions) == 0 or sessions[0] != start or sessions[-1] != end:
        raise ValueError("requested start/end sessions are absent from bars")

    first = clean.loc[clean["session"].eq(start)].copy()
    numeric_open = pd.to_numeric(first["open"], errors="coerce")
    eligible = first.loc[np.isfinite(numeric_open) & numeric_open.gt(0)].sort_values(
        "ticker", kind="stable"
    )
    if eligible.empty:
        raise ValueError("no equities have a tradable first-evaluation open")
    fee_rate = cost_bps_per_side / 10_000.0
    per_name_budget = initial_cash * target_exposure / len(eligible)
    holdings: dict[str, Holding] = {}
    fills: list[Fill] = []
    cash = float(initial_cash)
    for row in eligible.itertuples(index=False):
        shares = float(
            np.floor(per_name_budget / ((1.0 + fee_rate) * float(row.open)))
        )
        if shares <= 0:
            continue
        gross = shares * float(row.open)
        fee = gross * fee_rate
        cash -= gross + fee
        ticker = str(row.ticker)
        holdings[ticker] = Holding(shares=shares, entry_date=start)
        fills.append(
            Fill(
                start,
                start,
                ticker,
                "buy",
                shares,
                float(row.open),
                gross,
                fee,
                "reference_entry",
            )
        )
    if not holdings:
        raise ValueError("capital was insufficient to buy an eligible equity")

    lookup = clean.set_index(["session", "ticker"])
    last_marks: dict[str, float] = {}
    accounts: list[dict[str, Any]] = []
    total_entry_value = sum(fill.gross_value for fill in fills)
    total_fees = sum(fill.fee for fill in fills)
    for session in sessions:
        market_value = 0.0
        carried = 0
        for ticker, holding in holdings.items():
            key = (pd.Timestamp(session), ticker)
            close = float(lookup.loc[key, "close"]) if key in lookup.index else np.nan
            if np.isfinite(close) and close > 0:
                last_marks[ticker] = close
            else:
                close = last_marks.get(ticker, np.nan)
                carried += 1
            if not np.isfinite(close):
                raise ValueError(
                    f"no valid valuation for held {ticker} on {pd.Timestamp(session).date()}"
                )
            market_value += holding.shares * close
        accounts.append(
            {
                "session": pd.Timestamp(session),
                "cash": cash,
                "market_value": market_value,
                "account_value": cash + market_value,
                "fees": total_fees if session == sessions[0] else 0.0,
                "traded_value": total_entry_value if session == sessions[0] else 0.0,
                "positions": len(holdings),
                "carried_valuations": carried,
            }
        )
    account = pd.DataFrame(accounts)
    account["net_return"] = account["account_value"].pct_change()
    account.loc[account.index[0], "net_return"] = (
        account.loc[account.index[0], "account_value"] / initial_cash - 1.0
    )
    return LedgerResult(
        account=account,
        fills=pd.DataFrame(
            [asdict(fill) for fill in fills], columns=list(Fill.__annotations__)
        ),
        final_positions={ticker: holding.shares for ticker, holding in holdings.items()},
        rejected_orders=pd.DataFrame(),
    )


def portfolio_metrics(account: pd.DataFrame, *, annualization: int = 252) -> dict[str, float]:
    """Compute metrics from reconciled, compounded account value."""
    if account.empty or "account_value" not in account or "net_return" not in account:
        raise ValueError("account series is empty or incomplete")
    wealth = account["account_value"].astype(float)
    returns = account["net_return"].astype(float)
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    std = returns.std(ddof=1)
    sharpe = np.sqrt(annualization) * returns.mean() / std if std > 0 else np.nan
    initial_value = wealth.iloc[0] / (1.0 + returns.iloc[0])
    return {
        "net_cumulative_return": float(wealth.iloc[-1] / initial_value - 1.0),
        "net_sharpe_zero_cash": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "turnover_dollars": float(account["traded_value"].sum()),
        "transaction_costs_dollars": float(account["fees"].sum()),
        "ending_value": float(wealth.iloc[-1]),
    }


def assert_account_reconciles(result: LedgerResult, *, atol: float = 1e-8) -> None:
    """Raise if any daily identity or terminal cash/position identity fails."""
    account = result.account
    if not np.allclose(
        account["account_value"], account["cash"] + account["market_value"], atol=atol, rtol=0
    ):
        raise AssertionError("daily account value does not reconcile")
    if (account["cash"] < -atol).any():
        raise AssertionError("ledger used leverage")
