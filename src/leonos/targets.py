"""Leakage-safe forecast origins and ten-session appreciation targets.

The functions in this module deliberately operate on an explicit exchange-session
calendar.  A ticker's next row is not assumed to be the next exchange session:
missing rows remain missing after reindexing and invalidate the affected context or
label.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

KEY_COLUMNS = ("ticker", "origin")


@dataclass(frozen=True)
class TargetSpec:
    """Definition of the published Kronos investment target."""

    context_sessions: int = 90
    horizon_sessions: int = 10

    def __post_init__(self) -> None:
        if self.context_sessions < 1:
            raise ValueError("context_sessions must be positive")
        if self.horizon_sessions < 1:
            raise ValueError("horizon_sessions must be positive")


@dataclass(frozen=True)
class SplitSpec:
    """Inclusive origin/label-end constraints for a chronological split."""

    name: str
    origin_start: str | pd.Timestamp | None = None
    origin_end: str | pd.Timestamp | None = None
    label_end_min: str | pd.Timestamp | None = None
    label_end_max: str | pd.Timestamp | None = None


DEVELOPMENT_SPLIT = SplitSpec(
    name="development",
    label_end_max="2023-12-31",
)
VALIDATION_SPLIT = SplitSpec(
    name="validation",
    origin_start="2024-07-01",
    origin_end="2024-12-31",
    label_end_max="2024-12-31",
)
TEST_SPLIT = SplitSpec(name="test", origin_start="2025-01-01")
DEFAULT_TARGET_SPEC = TargetSpec()


def canonical_session_index(values: Iterable[object]) -> pd.DatetimeIndex:
    """Return unique, sorted, timezone-naive UTC calendar dates.

    The accepted-data layer is responsible for deciding what a source timestamp
    means.  Once that decision has been made, this helper gives all downstream
    components one stable daily-session representation.
    """

    parsed = pd.to_datetime(list(values), utc=True, errors="raise")
    index = pd.DatetimeIndex(parsed).tz_convert(None).normalize()
    if index.has_duplicates:
        duplicates = index[index.duplicated()].unique().strftime("%Y-%m-%d").tolist()
        raise ValueError(f"exchange calendar contains duplicate sessions: {duplicates[:5]}")
    return index.sort_values()


def _canonicalize_session_series(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="raise")
    return parsed.dt.tz_convert(None).dt.normalize()


def _prepare_bars(
    bars: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    ticker_col: str,
    session_col: str,
) -> pd.DataFrame:
    missing = {ticker_col, session_col}.difference(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {sorted(missing)}")
    out = bars.copy()
    out[ticker_col] = out[ticker_col].astype("string")
    if out[ticker_col].isna().any():
        raise ValueError("ticker keys contain missing values")
    out[session_col] = _canonicalize_session_series(out[session_col])
    duplicated = out.duplicated([ticker_col, session_col], keep=False)
    if duplicated.any():
        examples = (
            out.loc[duplicated, [ticker_col, session_col]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(f"duplicate ticker/session bars must be resolved upstream: {examples}")
    unexpected = pd.Index(out[session_col].unique()).difference(calendar)
    if len(unexpected):
        rendered = pd.DatetimeIndex(unexpected[:5]).strftime("%Y-%m-%d").tolist()
        raise ValueError(f"bars contain sessions absent from exchange calendar: {rendered}")
    return out.sort_values([ticker_col, session_col], kind="stable").reset_index(drop=True)


def build_targets(
    bars: pd.DataFrame,
    calendar: Sequence[object],
    *,
    spec: TargetSpec = DEFAULT_TARGET_SPEC,
    ticker_col: str = "ticker",
    session_col: str = "session",
    close_col: str = "close",
    origin_dates: Sequence[object] | None = None,
    include_incomplete: bool = False,
) -> pd.DataFrame:
    """Construct exact average-future-close targets on an exchange calendar.

    The result contains one row per ticker/origin with explicit input and label
    boundaries.  By default only origins with a complete close context and all ten
    future closes are returned.  ``include_incomplete=True`` is useful for audits;
    such rows have a non-finite ``target`` and explicit completeness flags.
    """

    cal = canonical_session_index(calendar)
    if len(cal) < spec.context_sessions + spec.horizon_sessions:
        raise ValueError("calendar is shorter than the requested context plus horizon")
    if close_col not in bars.columns:
        raise ValueError(f"bars missing close column {close_col!r}")
    frame = _prepare_bars(
        bars, cal, ticker_col=ticker_col, session_col=session_col
    )
    frame[close_col] = pd.to_numeric(frame[close_col], errors="coerce")

    requested_origins: set[pd.Timestamp] | None = None
    if origin_dates is not None:
        requested_origins = set(canonical_session_index(origin_dates))
        absent = requested_origins.difference(cal)
        if absent:
            rendered = sorted(absent)[:5]
            raise ValueError(f"requested origins absent from exchange calendar: {rendered}")

    records: list[dict[str, object]] = []
    first_position = spec.context_sessions - 1
    last_position = len(cal) - spec.horizon_sessions - 1
    positions = range(first_position, last_position + 1)

    for ticker, group in frame.groupby(ticker_col, sort=True, observed=True):
        close = (
            group.set_index(session_col)[close_col]
            .reindex(cal)
            .to_numpy(dtype=np.float64, na_value=np.nan)
        )
        valid_close = np.isfinite(close) & (close > 0.0)
        for position in positions:
            origin = cal[position]
            if requested_origins is not None and origin not in requested_origins:
                continue
            context_slice = slice(position - spec.context_sessions + 1, position + 1)
            label_slice = slice(position + 1, position + spec.horizon_sessions + 1)
            context_complete = bool(valid_close[context_slice].all())
            label_complete = bool(valid_close[label_slice].all())
            eligible = context_complete and label_complete
            if not eligible and not include_incomplete:
                continue
            target = (
                float(close[label_slice].mean() / close[position] - 1.0)
                if eligible
                else np.nan
            )
            records.append(
                {
                    "ticker": str(ticker),
                    "origin": origin,
                    "context_start": cal[position - spec.context_sessions + 1],
                    "input_end": origin,
                    "execution_session": cal[position + 1],
                    "forecast_start": cal[position + 1],
                    "label_end": cal[position + spec.horizon_sessions],
                    "current_close": float(close[position])
                    if valid_close[position]
                    else np.nan,
                    "target": target,
                    "context_complete": context_complete,
                    "label_complete": label_complete,
                    "context_sessions": spec.context_sessions,
                    "horizon_sessions": spec.horizon_sessions,
                }
            )

    columns = [
        "ticker",
        "origin",
        "context_start",
        "input_end",
        "execution_session",
        "forecast_start",
        "label_end",
        "current_close",
        "target",
        "context_complete",
        "label_complete",
        "context_sessions",
        "horizon_sessions",
    ]
    result = pd.DataFrame.from_records(records, columns=columns)
    if not result.empty:
        result = result.sort_values(["origin", "ticker"], kind="stable").reset_index(drop=True)
    return result


def apply_split(targets: pd.DataFrame, split: SplitSpec) -> pd.DataFrame:
    """Filter targets by actual inclusive origin and label-end constraints."""

    required = {"origin", "label_end"}
    missing = required.difference(targets.columns)
    if missing:
        raise ValueError(f"targets missing split columns: {sorted(missing)}")
    out = targets.copy()
    out["origin"] = _canonicalize_session_series(out["origin"])
    out["label_end"] = _canonicalize_session_series(out["label_end"])
    keep = pd.Series(True, index=out.index)

    def boundary(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
        if value is None:
            return None
        return canonical_session_index([value])[0]

    origin_start = boundary(split.origin_start)
    origin_end = boundary(split.origin_end)
    label_end_min = boundary(split.label_end_min)
    label_end_max = boundary(split.label_end_max)
    if origin_start is not None:
        keep &= out["origin"] >= origin_start
    if origin_end is not None:
        keep &= out["origin"] <= origin_end
    if label_end_min is not None:
        keep &= out["label_end"] >= label_end_min
    if label_end_max is not None:
        keep &= out["label_end"] <= label_end_max
    selected = out.loc[keep].copy()
    selected["split"] = split.name
    return selected.reset_index(drop=True)


def assert_no_label_overlap(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Assert that every left label ends before the first right forecast origin."""

    if left.empty or right.empty:
        return
    left_end = pd.to_datetime(left["label_end"]).max()
    right_origin = pd.to_datetime(right["origin"]).min()
    if left_end >= right_origin:
        raise ValueError(
            f"label leakage across split boundary: {left_end=} is not before {right_origin=}"
        )


def extract_context(
    bars: pd.DataFrame,
    calendar: Sequence[object],
    *,
    ticker: str,
    origin: object,
    columns: Sequence[str] = ("open", "high", "low", "close", "volume"),
    spec: TargetSpec = DEFAULT_TARGET_SPEC,
    ticker_col: str = "ticker",
    session_col: str = "session",
) -> pd.DataFrame:
    """Return exactly the completed sessions visible at an origin.

    This is a small reference adapter used in alignment/leakage tests.  It never
    receives or joins future prices.
    """

    cal = canonical_session_index(calendar)
    origin_ts = canonical_session_index([origin])[0]
    try:
        position = cal.get_loc(origin_ts)
    except KeyError as exc:
        raise ValueError(f"origin {origin_ts.date()} is absent from calendar") from exc
    if not isinstance(position, (int, np.integer)) or position < spec.context_sessions - 1:
        raise ValueError("origin has insufficient calendar history")
    missing_columns = set(columns).difference(bars.columns)
    if missing_columns:
        raise ValueError(f"bars missing context columns: {sorted(missing_columns)}")
    frame = _prepare_bars(
        bars, cal, ticker_col=ticker_col, session_col=session_col
    )
    sessions = cal[position - spec.context_sessions + 1 : position + 1]
    selected = frame.loc[frame[ticker_col].astype(str) == str(ticker)].set_index(session_col)
    context = selected.reindex(sessions)
    if context[list(columns)].isna().any(axis=None):
        missing_sessions = context.index[context[list(columns)].isna().any(axis=1)]
        raise ValueError(
            "context contains missing fields/sessions: "
            + ", ".join(missing_sessions[:5].strftime("%Y-%m-%d"))
        )
    context = context.loc[:, list(columns)].copy()
    context.insert(0, "session", sessions)
    context.insert(0, "ticker", str(ticker))
    return context.reset_index(drop=True)


__all__ = [
    "DEVELOPMENT_SPLIT",
    "DEFAULT_TARGET_SPEC",
    "KEY_COLUMNS",
    "SplitSpec",
    "TEST_SPLIT",
    "TargetSpec",
    "VALIDATION_SPLIT",
    "apply_split",
    "assert_no_label_overlap",
    "build_targets",
    "canonical_session_index",
    "extract_context",
]
