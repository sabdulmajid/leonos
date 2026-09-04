"""Causal, OHLCV-supported factors for the pooled LightGBM baseline.

This is a deliberately explicit Alpha158-style adaptation, not a claim that the
full Qlib Alpha158 handler is available from the source data.  It omits VWAP and
amount factors, uses only same-equity OHLCV and known calendar fields, and caps
every effective lookback inside the shared 90-session information envelope.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .targets import _prepare_bars, canonical_session_index

FEATURE_SET_NAME = "ohlcv_alpha158_90_v1"
ROLLING_WINDOWS = (5, 10, 20, 30, 60)
REQUIRED_BAR_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class FeatureDefinition:
    """Auditable description of one predictive column."""

    name: str
    family: str
    lookback_sessions: int
    expression: str


def feature_definitions() -> tuple[FeatureDefinition, ...]:
    """Return the frozen ordered feature inventory."""

    definitions = [
        FeatureDefinition("KMID", "candlestick", 1, "close/open - 1"),
        FeatureDefinition("KLEN", "candlestick", 1, "(high-low)/close"),
        FeatureDefinition(
            "KMID2", "candlestick", 1, "(close-open)/(high-low), zero if flat"
        ),
        FeatureDefinition(
            "KUP", "candlestick", 1, "(high-max(open,close))/close"
        ),
        FeatureDefinition(
            "KLOW", "candlestick", 1, "(min(open,close)-low)/close"
        ),
        FeatureDefinition(
            "KSFT", "candlestick", 1, "(2*close-high-low)/close"
        ),
        FeatureDefinition("VROC1", "volume", 2, "volume/lag(volume,1) - 1"),
        FeatureDefinition("DOW_SIN", "calendar", 1, "sin(2*pi*weekday/7)"),
        FeatureDefinition("DOW_COS", "calendar", 1, "cos(2*pi*weekday/7)"),
        FeatureDefinition("MONTH_SIN", "calendar", 1, "sin(2*pi*(month-1)/12)"),
        FeatureDefinition("MONTH_COS", "calendar", 1, "cos(2*pi*(month-1)/12)"),
    ]
    for window in ROLLING_WINDOWS:
        definitions.extend(
            [
                FeatureDefinition(
                    f"ROC{window}",
                    "price",
                    window + 1,
                    f"close/lag(close,{window}) - 1",
                ),
                FeatureDefinition(
                    f"MA{window}",
                    "price",
                    window,
                    f"close/mean(close,{window}) - 1",
                ),
                FeatureDefinition(
                    f"STD{window}",
                    "price",
                    window + 1,
                    f"std(close/lag(close,1)-1,{window})",
                ),
                FeatureDefinition(
                    f"MAX{window}",
                    "price",
                    window,
                    f"close/max(close,{window}) - 1",
                ),
                FeatureDefinition(
                    f"MIN{window}",
                    "price",
                    window,
                    f"close/min(close,{window}) - 1",
                ),
                FeatureDefinition(
                    f"RSV{window}",
                    "price",
                    window,
                    f"(close-min(low,{window}))/(max(high,{window})-min(low,{window}))",
                ),
                FeatureDefinition(
                    f"RANK{window}",
                    "price",
                    window,
                    f"percentile rank of close in trailing {window} sessions",
                ),
                FeatureDefinition(
                    f"VMA{window}",
                    "volume",
                    window,
                    f"volume/mean(volume,{window}) - 1",
                ),
                FeatureDefinition(
                    f"VSTD{window}",
                    "volume",
                    window,
                    f"std(volume,{window})/mean(volume,{window})",
                ),
                FeatureDefinition(
                    f"PV_CORR{window}",
                    "price_volume",
                    window + 1,
                    f"corr(return_1,volume_change_1,{window})",
                ),
            ]
        )
    return tuple(definitions)


FEATURE_COLUMNS = tuple(definition.name for definition in feature_definitions())
MAX_FEATURE_LOOKBACK = max(
    definition.lookback_sessions for definition in feature_definitions()
)
if MAX_FEATURE_LOOKBACK > 90:  # pragma: no cover - import-time invariant
    raise RuntimeError("feature inventory exceeds the 90-session contract")


def feature_manifest() -> dict[str, object]:
    """Return a JSON-serializable description for run metadata."""

    return {
        "name": FEATURE_SET_NAME,
        "feature_count": len(FEATURE_COLUMNS),
        "max_lookback_sessions": MAX_FEATURE_LOOKBACK,
        "uses_ticker_identity": False,
        "uses_amount_or_vwap": False,
        "definitions": [asdict(definition) for definition in feature_definitions()],
    }


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.where(denominator != 0.0)
    return numerator / denominator


def _rolling_rank(series: pd.Series, window: int) -> pd.Series:
    rolling = series.rolling(window, min_periods=window)
    try:
        return rolling.rank(method="average", pct=True)
    except (AttributeError, TypeError):  # pragma: no cover - old pandas fallback
        return rolling.apply(
            lambda values: pd.Series(values).rank(method="average", pct=True).iloc[-1],
            raw=False,
        )


def _features_for_ticker(values: pd.DataFrame) -> pd.DataFrame:
    open_ = values["open"]
    high = values["high"]
    low = values["low"]
    close = values["close"]
    volume = values["volume"]
    spread = high - low
    midpoint_extreme = 2.0 * close - high - low

    out = pd.DataFrame(index=values.index)
    out["KMID"] = _safe_ratio(close, open_) - 1.0
    out["KLEN"] = _safe_ratio(spread, close)
    out["KMID2"] = _safe_ratio(close - open_, spread).fillna(0.0)
    out["KUP"] = _safe_ratio(high - pd.concat([open_, close], axis=1).max(axis=1), close)
    out["KLOW"] = _safe_ratio(pd.concat([open_, close], axis=1).min(axis=1) - low, close)
    out["KSFT"] = _safe_ratio(midpoint_extreme, close)

    return_1 = _safe_ratio(close, close.shift(1)) - 1.0
    volume_change_1 = _safe_ratio(volume, volume.shift(1)) - 1.0
    out["VROC1"] = volume_change_1

    weekday = pd.Series(values.index.dayofweek, index=values.index, dtype=float)
    month = pd.Series(values.index.month - 1, index=values.index, dtype=float)
    out["DOW_SIN"] = np.sin(2.0 * np.pi * weekday / 7.0)
    out["DOW_COS"] = np.cos(2.0 * np.pi * weekday / 7.0)
    out["MONTH_SIN"] = np.sin(2.0 * np.pi * month / 12.0)
    out["MONTH_COS"] = np.cos(2.0 * np.pi * month / 12.0)

    for window in ROLLING_WINDOWS:
        rolling_close = close.rolling(window, min_periods=window)
        rolling_low = low.rolling(window, min_periods=window).min()
        rolling_high = high.rolling(window, min_periods=window).max()
        rolling_volume = volume.rolling(window, min_periods=window)
        mean_volume = rolling_volume.mean()
        out[f"ROC{window}"] = _safe_ratio(close, close.shift(window)) - 1.0
        out[f"MA{window}"] = _safe_ratio(close, rolling_close.mean()) - 1.0
        out[f"STD{window}"] = return_1.rolling(
            window, min_periods=window
        ).std(ddof=0)
        out[f"MAX{window}"] = _safe_ratio(close, rolling_close.max()) - 1.0
        out[f"MIN{window}"] = _safe_ratio(close, rolling_close.min()) - 1.0
        out[f"RSV{window}"] = _safe_ratio(
            close - rolling_low, rolling_high - rolling_low
        ).fillna(0.0)
        out[f"RANK{window}"] = _rolling_rank(close, window)
        out[f"VMA{window}"] = _safe_ratio(volume, mean_volume) - 1.0
        out[f"VSTD{window}"] = _safe_ratio(
            rolling_volume.std(ddof=0), mean_volume
        )
        out[f"PV_CORR{window}"] = return_1.rolling(
            window, min_periods=window
        ).corr(volume_change_1)

    return out.loc[:, FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)


def build_ohlcv_features(
    bars: pd.DataFrame,
    calendar: Sequence[object],
    *,
    keys: pd.DataFrame | None = None,
    context_sessions: int = 90,
    ticker_col: str = "ticker",
    session_col: str = "session",
    column_map: dict[str, str] | None = None,
    include_incomplete: bool = False,
) -> pd.DataFrame:
    """Compute causal factors and select valid ticker/origin contexts.

    ``column_map`` maps canonical names (open/high/low/close/volume) to a verified,
    internally consistent price-basis in ``bars``.  This function never derives an
    adjustment factor or combines adjusted close with raw OHLC.
    """

    if context_sessions < MAX_FEATURE_LOOKBACK or context_sessions > 90:
        raise ValueError(
            f"context_sessions must be in [{MAX_FEATURE_LOOKBACK}, 90], got {context_sessions}"
        )
    cal = canonical_session_index(calendar)
    mapping = column_map or {name: name for name in REQUIRED_BAR_COLUMNS}
    if set(mapping) != set(REQUIRED_BAR_COLUMNS):
        raise ValueError(
            "column_map must provide exactly open/high/low/close/volume canonical keys"
        )
    source_columns = [mapping[name] for name in REQUIRED_BAR_COLUMNS]
    missing = set(source_columns).difference(bars.columns)
    if missing:
        raise ValueError(f"bars missing OHLCV columns: {sorted(missing)}")
    frame = _prepare_bars(
        bars, cal, ticker_col=ticker_col, session_col=session_col
    )

    requested: pd.DataFrame | None = None
    if keys is not None:
        if not {"ticker", "origin"}.issubset(keys.columns):
            raise ValueError("keys must contain ticker and origin")
        requested = keys.loc[:, ["ticker", "origin"]].copy()
        requested["ticker"] = requested["ticker"].astype("string")
        requested["origin"] = pd.to_datetime(
            requested["origin"], utc=True, errors="raise"
        ).dt.tz_convert(None).dt.normalize()
        if requested.duplicated(["ticker", "origin"]).any():
            raise ValueError("requested feature keys are not unique")
        absent_dates = pd.Index(requested["origin"].unique()).difference(cal)
        if len(absent_dates):
            raise ValueError("requested feature origins are absent from exchange calendar")

    outputs: list[pd.DataFrame] = []
    for ticker, group in frame.groupby(ticker_col, sort=True, observed=True):
        values = pd.DataFrame(index=cal)
        indexed = group.set_index(session_col)
        for canonical, source in mapping.items():
            values[canonical] = pd.to_numeric(indexed[source], errors="coerce").reindex(cal)

        finite = np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
        price_positive = (values[["open", "high", "low", "close"]] > 0.0).all(axis=1)
        volume_valid = values["volume"] >= 0.0
        raw_valid = pd.Series(
            finite & price_positive.to_numpy() & volume_valid.to_numpy(), index=cal
        )
        context_complete = (
            raw_valid.astype(np.int16)
            .rolling(context_sessions, min_periods=context_sessions)
            .sum()
            .eq(context_sessions)
        )
        computed = _features_for_ticker(values)
        computed.insert(0, "context_complete", context_complete)
        computed.insert(0, "context_start", pd.Series(cal, index=cal).shift(context_sessions - 1))
        computed.insert(0, "origin", cal)
        computed.insert(0, "ticker", str(ticker))

        if requested is not None:
            ticker_dates = requested.loc[
                requested["ticker"].astype(str) == str(ticker), "origin"
            ]
            computed = computed.loc[computed["origin"].isin(ticker_dates)]
        if not include_incomplete:
            computed = computed.loc[computed["context_complete"]]
        outputs.append(computed.reset_index(drop=True))

    columns = ["ticker", "origin", "context_start", "context_complete", *FEATURE_COLUMNS]
    result = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(columns=columns)
    result = result.loc[:, columns]
    if requested is not None:
        available = result.loc[:, ["ticker", "origin"]].copy()
        available["ticker"] = available["ticker"].astype("string")
        missing_keys = requested.merge(
            available, on=["ticker", "origin"], how="left", indicator=True
        )
        missing_keys = missing_keys.loc[missing_keys["_merge"] == "left_only"]
        if not include_incomplete and not missing_keys.empty:
            # Missing keys are expected when a predeclared origin lacks a usable
            # context.  Coverage is made visible by callers joining against keys.
            pass
    return result.sort_values(["origin", "ticker"], kind="stable").reset_index(drop=True)


def validate_feature_frame(
    frame: pd.DataFrame, feature_columns: Sequence[str] = FEATURE_COLUMNS
) -> None:
    """Fail closed if model inputs depart from the frozen causal inventory."""

    selected = tuple(feature_columns)
    if selected != FEATURE_COLUMNS:
        unknown = sorted(set(selected).difference(FEATURE_COLUMNS))
        omitted = sorted(set(FEATURE_COLUMNS).difference(selected))
        raise ValueError(
            f"feature inventory mismatch; unknown={unknown[:5]}, omitted={omitted[:5]}"
        )
    missing = set(selected).difference(frame.columns)
    if missing:
        raise ValueError(f"feature frame missing columns: {sorted(missing)}")
    forbidden = {"ticker", "target", "label", "amount", "vwap"}.intersection(selected)
    if forbidden:
        raise ValueError(f"non-permitted predictive columns: {sorted(forbidden)}")
    if MAX_FEATURE_LOOKBACK > 90:
        raise ValueError("feature lookback exceeds the 90-session information set")


__all__ = [
    "FEATURE_COLUMNS",
    "FEATURE_SET_NAME",
    "FeatureDefinition",
    "MAX_FEATURE_LOOKBACK",
    "REQUIRED_BAR_COLUMNS",
    "ROLLING_WINDOWS",
    "build_ohlcv_features",
    "feature_definitions",
    "feature_manifest",
    "validate_feature_frame",
]
