"""Pinned Qlib day-data and U.S. next-open backtest integration.

The saved score index is the information date.  Pinned Qlib's
``TopkDropoutStrategy.generate_trade_decision`` explicitly requests the prior trading
step with ``shift=1``; this module therefore does not shift signals a second time.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

QLIB_URL = "https://github.com/microsoft/qlib"
QLIB_REVISION = "79633dd9506ea689e5400dea0197717b5b3d74b7"
QLIB_FREQUENCY = "day"
QLIB_FIELDS = ("open", "high", "low", "close", "volume", "factor", "change")
BAR_COLUMNS = ("ticker", "session", "open", "high", "low", "close", "volume")
SIGNAL_SHIFT_SESSIONS = 1
PRIMARY_COST_BPS = 5


class QlibAdapterError(ValueError):
    """Input data or pinned-Qlib behavior violates the Leonos contract."""


class QlibUnavailableError(ImportError):
    """The optional pinned Qlib dependency is not installed."""


class InvalidOpenFillError(RuntimeError):
    """An order reached execution without a finite, strictly positive open."""


@dataclass(frozen=True)
class QlibBacktestSpec:
    """Fixed Leonos v1 U.S. portfolio settings (cost may use declared sensitivities)."""

    topk: int = 5
    n_drop: int = 1
    hold_thresh: int = 5
    risk_degree: float = 0.95
    initial_cash: float = 1_000_000.0
    cost_bps_per_side: int = PRIMARY_COST_BPS
    min_cost: float = 0.0
    trade_unit: int = 1

    def __post_init__(self) -> None:
        fixed = {
            "topk": (self.topk, 5),
            "n_drop": (self.n_drop, 1),
            "hold_thresh": (self.hold_thresh, 5),
            "risk_degree": (self.risk_degree, 0.95),
            "initial_cash": (self.initial_cash, 1_000_000.0),
            "min_cost": (self.min_cost, 0.0),
            "trade_unit": (self.trade_unit, 1),
        }
        changed = [name for name, (actual, expected) in fixed.items() if actual != expected]
        if changed:
            raise QlibAdapterError(f"Leonos v1 portfolio settings are fixed; changed: {changed}")
        if self.cost_bps_per_side not in {0, 5, 15}:
            raise QlibAdapterError("cost must be the declared 0, 5, or 15 bps per-side setting")

    @property
    def proportional_cost(self) -> float:
        return self.cost_bps_per_side / 10_000.0


@dataclass(frozen=True)
class QlibBacktestOutputs:
    """Raw upstream outputs retained alongside their directly useful components."""

    raw_portfolio_metrics: dict[str, tuple[pd.DataFrame, dict[Any, Any]]]
    raw_indicator_metrics: dict[str, tuple[pd.DataFrame, Any]]
    report: pd.DataFrame
    positions: dict[Any, Any]
    trade_indicators: pd.DataFrame
    order_history: dict[Any, Any]


def _daily_index(values: Iterable[Any], name: str) -> pd.DatetimeIndex:
    try:
        result = pd.DatetimeIndex(pd.to_datetime(list(values)))
    except (TypeError, ValueError) as exc:
        raise QlibAdapterError(f"{name} must contain parseable daily sessions") from exc
    if result.empty or result.hasnans:
        raise QlibAdapterError(f"{name} must be non-empty and contain no missing values")
    if result.tz is not None:
        raise QlibAdapterError(f"{name} must already be timezone-naive exchange sessions")
    if not result.equals(result.normalize()):
        raise QlibAdapterError(f"{name} must be normalized daily session labels")
    return result


def _validated_bars(bars: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(BAR_COLUMNS).difference(bars.columns))
    if missing:
        raise QlibAdapterError(f"standardized bars are missing columns: {missing}")
    clean = bars.loc[:, BAR_COLUMNS].copy()
    if clean.empty:
        raise QlibAdapterError("cannot build Qlib data from an empty frame")
    clean["ticker"] = clean["ticker"].astype(str)
    if clean["ticker"].str.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*").eq(False).any():
        raise QlibAdapterError("ticker contains characters unsafe for Qlib feature paths")
    original_sessions = _daily_index(clean["session"], "bars.session")
    clean["session"] = original_sessions
    if clean.duplicated(["ticker", "session"]).any():
        raise QlibAdapterError("duplicate (ticker, session) rows are not allowed")
    folded = clean[["ticker"]].drop_duplicates().assign(path=lambda x: x["ticker"].str.lower())
    if folded["path"].duplicated().any():
        raise QlibAdapterError("tickers collide under Qlib's lowercase feature-path convention")

    numeric = list(BAR_COLUMNS[2:])
    try:
        values = clean[numeric].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise QlibAdapterError("OHLCV values must be numeric") from exc
    if not np.isfinite(values).all():
        raise QlibAdapterError("OHLCV values must be finite")
    if (clean[["open", "high", "low", "close"]].to_numpy(dtype=np.float64) <= 0).any():
        raise QlibAdapterError("OHLC values must be strictly positive")
    if (clean["volume"].to_numpy(dtype=np.float64) < 0).any():
        raise QlibAdapterError("volume must be non-negative")
    max_oc = clean[["open", "close"]].max(axis=1)
    min_oc = clean[["open", "close"]].min(axis=1)
    if (clean["high"] < max_oc).any() or (clean["low"] > min_oc).any():
        raise QlibAdapterError("OHLC candle bounds are inconsistent")
    if (clean["high"] < clean["low"]).any():
        raise QlibAdapterError("high must be greater than or equal to low")
    if np.abs(values).max() > np.finfo(np.float32).max:
        raise QlibAdapterError("OHLCV value exceeds Qlib's float32 binary range")
    return clean.sort_values(["session", "ticker"], kind="mergesort").reset_index(drop=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_day_binary(path: Path, start_index: int, values: np.ndarray) -> None:
    payload = np.concatenate(
        [np.asarray([start_index], dtype="<f4"), np.asarray(values, dtype="<f4")]
    )
    path.write_bytes(payload.tobytes(order="C"))


def read_day_binary(path: Path) -> tuple[int, np.ndarray]:
    """Small independent reader used to verify Qlib's header-plus-values layout."""

    payload = np.fromfile(Path(path), dtype="<f4")
    if len(payload) < 2 or not np.isfinite(payload[0]) or payload[0] != int(payload[0]):
        raise QlibAdapterError(f"invalid Qlib day binary: {path}")
    return int(payload[0]), payload[1:]


def _dataset_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def write_qlib_day_dataset(
    bars: pd.DataFrame,
    destination: Path,
    *,
    calendar: Sequence[Any] | None = None,
    price_basis: str = "split_adjusted_ohlcv",
) -> dict[str, Any]:
    """Atomically write deterministic Qlib day binaries without normalizing prices.

    OHLC and volume are assumed to have already been placed on one accepted,
    split-consistent basis.  Prices are written as supplied, ``factor`` is exactly 1,
    and ``change`` is close-to-prior-exchange-session close on that same basis.
    """

    if not price_basis.strip():
        raise QlibAdapterError("price_basis must explicitly describe the accepted input basis")
    clean = _validated_bars(bars)
    if calendar is None:
        sessions = pd.DatetimeIndex(clean["session"].drop_duplicates().sort_values())
    else:
        sessions = _daily_index(calendar, "calendar")
        if sessions.has_duplicates or not sessions.is_monotonic_increasing:
            raise QlibAdapterError("calendar must be strictly increasing and unique")
    unknown_sessions = pd.Index(clean["session"].unique()).difference(sessions)
    if len(unknown_sessions):
        raise QlibAdapterError(
            f"bars contain sessions absent from calendar: {unknown_sessions.tolist()}"
        )

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing Qlib dataset: {destination}")
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=destination.parent)
    )
    staging = staging_parent / "dataset"
    try:
        (staging / "calendars").mkdir(parents=True)
        (staging / "instruments").mkdir(parents=True)
        (staging / "features").mkdir(parents=True)
        calendar_text = "".join(f"{session:%Y-%m-%d}\n" for session in sessions)
        (staging / "calendars" / "day.txt").write_text(calendar_text, encoding="utf-8")

        session_locations = {session: index for index, session in enumerate(sessions)}
        instrument_lines: list[str] = []
        for ticker, ticker_bars in clean.groupby("ticker", sort=True):
            ticker_bars = ticker_bars.sort_values("session").set_index("session")
            first_session = pd.Timestamp(ticker_bars.index.min())
            last_session = pd.Timestamp(ticker_bars.index.max())
            start_index = session_locations[first_session]
            end_index = session_locations[last_session]
            active_sessions = sessions[start_index : end_index + 1]
            aligned = ticker_bars.reindex(active_sessions)
            aligned.index.name = "session"

            feature_values: dict[str, np.ndarray] = {
                field: aligned[field].to_numpy(dtype=np.float64)
                for field in ("open", "high", "low", "close", "volume")
            }
            feature_values["factor"] = np.ones(len(aligned), dtype=np.float64)
            feature_values["change"] = (
                aligned["close"].pct_change(fill_method=None).to_numpy(dtype=np.float64)
            )
            feature_dir = staging / "features" / ticker.lower()
            feature_dir.mkdir()
            for field in QLIB_FIELDS:
                _write_day_binary(
                    feature_dir / f"{field}.day.bin", start_index, feature_values[field]
                )
            instrument_lines.append(
                f"{ticker}\t{first_session:%Y-%m-%d}\t{last_session:%Y-%m-%d}\n"
            )
        (staging / "instruments" / "all.txt").write_text(
            "".join(instrument_lines), encoding="utf-8"
        )

        files = {
            str(path.relative_to(staging)): {"sha256": _sha256(path), "size": path.stat().st_size}
            for path in _dataset_files(staging)
        }
        manifest: dict[str, Any] = {
            "format": "qlib-file-provider-day-v1",
            "qlib_revision": QLIB_REVISION,
            "frequency": QLIB_FREQUENCY,
            "binary_dtype": "little-endian-float32",
            "binary_layout": "first float=start calendar index; remaining floats=values",
            "price_basis": price_basis,
            "price_normalization": "none",
            "factor": 1.0,
            "change": "close[t] / close[prior exchange session] - 1 on price_basis",
            "source_rows": len(clean),
            "calendar_count": len(sessions),
            "calendar_start": sessions[0].strftime("%Y-%m-%d"),
            "calendar_end": sessions[-1].strftime("%Y-%m-%d"),
            "tickers": sorted(clean["ticker"].unique().tolist()),
            "fields": list(QLIB_FIELDS),
            "files": files,
        }
        manifest_payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        (staging / "leonos_qlib_manifest.json").write_text(
            manifest_payload, encoding="utf-8"
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging_parent, ignore_errors=True)
        raise
    else:
        shutil.rmtree(staging_parent, ignore_errors=True)
    return manifest


def build_information_date_signal(scores: pd.DataFrame) -> pd.Series:
    """Build Qlib signal indexed by the unshifted post-close information date."""

    required = {"ticker", "origin", "score"}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise QlibAdapterError(f"scores are missing columns: {missing}")
    clean = scores.loc[:, ["ticker", "origin", "score"]].copy()
    if clean.empty:
        raise QlibAdapterError("scores must not be empty")
    clean["ticker"] = clean["ticker"].astype(str)
    clean["origin"] = _daily_index(clean["origin"], "scores.origin")
    if clean.duplicated(["ticker", "origin"]).any():
        raise QlibAdapterError("scores contain duplicate (ticker, origin) keys")
    try:
        values = clean["score"].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise QlibAdapterError("scores must be numeric") from exc
    if not np.isfinite(values).all():
        raise QlibAdapterError("scores must be finite before portfolio simulation")
    index = pd.MultiIndex.from_arrays(
        [clean["origin"], clean["ticker"]], names=["datetime", "instrument"]
    )
    return pd.Series(values, index=index, name="score").sort_index()


def verify_topk_shift_contract() -> None:
    """Fail fast if installed Qlib no longer has the inspected one-step lookup."""

    try:
        from qlib.backtest.utils import TradeCalendarManager
        from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
    except ImportError as exc:
        raise QlibUnavailableError("install the locked 'qlib' optional dependency") from exc
    source = re.sub(r"\s+", " ", inspect.getsource(TopkDropoutStrategy.generate_trade_decision))
    if not re.search(r"get_step_time\(trade_step,\s*shift\s*=\s*1\)", source):
        raise QlibAdapterError(
            "installed TopkDropoutStrategy does not expose the inspected shift=1 lookup"
        )
    calendar_source = re.sub(r"\s+", " ", inspect.getsource(TradeCalendarManager.get_step_time))
    if not re.search(r"trade_step\s*-\s*shift", calendar_source):
        raise QlibAdapterError("installed Qlib no longer defines positive shift as an earlier bar")


def init_qlib_us(provider_uri: Path) -> None:
    """Initialize the locked file provider with U.S. market conventions."""

    try:
        import qlib
        from qlib.constant import REG_US
    except ImportError as exc:
        raise QlibUnavailableError("install the locked 'qlib' optional dependency") from exc
    verify_topk_shift_contract()
    qlib.init(
        provider_uri=str(Path(provider_uri).resolve(strict=True)),
        region=REG_US,
        dataset_cache=None,
        expression_cache=None,
    )


def _finite_positive_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and bool(
        np.isfinite(value) and value > 1e-8
    )


@lru_cache(maxsize=1)
def open_only_exchange_class() -> type:
    """Return a small Exchange subclass that never substitutes close for open."""

    try:
        from qlib.backtest.exchange import Exchange
    except ImportError as exc:
        raise QlibUnavailableError("install the locked 'qlib' optional dependency") from exc

    class OpenOnlyExchange(Exchange):
        def _execution_open(
            self,
            stock_id: str,
            start_time: pd.Timestamp,
            end_time: pd.Timestamp,
            method: str | None = "ts_data_last",
        ) -> Any:
            return self.quote.get_data(
                stock_id, start_time, end_time, field="$open", method=method
            )

        def is_stock_tradable(
            self,
            stock_id: str,
            start_time: pd.Timestamp,
            end_time: pd.Timestamp,
            direction: int | None = None,
        ) -> bool:
            # Deliberately inspect only the execution open: not high/low/close/full-day volume.
            del direction
            return _finite_positive_scalar(
                self._execution_open(stock_id, start_time, end_time)
            )

        def get_deal_price(
            self,
            stock_id: str,
            start_time: pd.Timestamp,
            end_time: pd.Timestamp,
            direction: Any,
            method: str | None = "ts_data_last",
        ) -> Any:
            del direction  # Buys and sells both use the same next-session open.
            value = self._execution_open(stock_id, start_time, end_time, method)
            if method is not None and not _finite_positive_scalar(value):
                raise InvalidOpenFillError(
                    f"invalid next-open fill for {stock_id} at {pd.Timestamp(start_time)}"
                )
            return value

    OpenOnlyExchange.__name__ = "OpenOnlyExchange"
    OpenOnlyExchange.__qualname__ = "OpenOnlyExchange"
    return OpenOnlyExchange


def make_open_only_exchange(
    *,
    start_time: Any,
    end_time: Any,
    codes: Sequence[str],
    spec: QlibBacktestSpec | None = None,
) -> Any:
    """Create the U.S. day exchange with explicit open fills and no market limits."""

    spec = spec or QlibBacktestSpec()
    exchange_class = open_only_exchange_class()
    return exchange_class(
        freq="day",
        start_time=pd.Timestamp(start_time),
        end_time=pd.Timestamp(end_time),
        codes=sorted(map(str, codes)),
        deal_price="$open",
        open_cost=spec.proportional_cost,
        close_cost=spec.proportional_cost,
        min_cost=spec.min_cost,
        impact_cost=0.0,
        limit_threshold=None,
        volume_threshold=None,
        trade_unit=spec.trade_unit,
    )


def order_indicator_metric_series(order_indicator: Any, metric: str) -> pd.Series:
    """Convert one Qlib order metric without its pandas-incompatible ``to_series``.

    Pinned Qlib's numpy-backed indicator passes its custom, non-iterable ``Index``
    directly to pandas.  pandas 2.3 rejects that object.  The documented Qlib
    ``get_index_data`` representation already exposes both values and ``tolist``;
    converting only that boundary avoids monkey-patching either dependency.
    """

    if not isinstance(metric, str) or not metric:
        raise QlibAdapterError("order-indicator metric must be a non-empty string")
    try:
        indexed = order_indicator.get_index_data(metric)
        values = np.asarray(indexed.values, dtype=np.float64)
        labels = indexed.index.tolist()
    except (AttributeError, TypeError, ValueError) as exc:
        raise QlibAdapterError(f"cannot convert Qlib order metric {metric!r}") from exc
    if values.ndim != 1 or len(values) != len(labels):
        raise QlibAdapterError(f"invalid Qlib order metric shape for {metric!r}")
    return pd.Series(values.copy(), index=pd.Index(labels), name=metric)


def run_qlib_topk_backtest(
    scores: pd.DataFrame,
    *,
    provider_uri: Path,
    start_time: Any,
    end_time: Any,
    spec: QlibBacktestSpec | None = None,
) -> QlibBacktestOutputs:
    """Run pinned TopkDropout; Qlib itself applies the one-session signal lookup."""

    spec = spec or QlibBacktestSpec()
    signal = build_information_date_signal(scores)
    init_qlib_us(provider_uri)
    try:
        from qlib.backtest.account import Account
        from qlib.backtest.backtest import backtest_loop
        from qlib.backtest.executor import SimulatorExecutor
        from qlib.backtest.utils import CommonInfrastructure
        from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
    except ImportError as exc:  # pragma: no cover - guarded by init_qlib_us
        raise QlibUnavailableError("install the locked 'qlib' optional dependency") from exc

    codes = signal.index.get_level_values("instrument").unique().tolist()
    exchange = make_open_only_exchange(
        start_time=start_time, end_time=end_time, codes=codes, spec=spec
    )
    strategy = TopkDropoutStrategy(
        signal=signal,
        topk=spec.topk,
        n_drop=spec.n_drop,
        hold_thresh=spec.hold_thresh,
        risk_degree=spec.risk_degree,
        only_tradable=False,
        forbid_all_trade_at_limit=False,
    )
    executor = SimulatorExecutor(
        time_per_step="day",
        generate_portfolio_metrics=True,
        track_data=False,
        verbose=False,
        trade_type="serial",
    )
    # Qlib's public helper converts benchmark=None to {}, which the pinned report code
    # interprets as its default CSI300 identifier.  Build the same infrastructure with
    # an explicit null benchmark to prevent that unwanted external dependency.
    account = Account(init_cash=spec.initial_cash, benchmark_config={"benchmark": None})
    common_infrastructure = CommonInfrastructure(
        trade_account=account, trade_exchange=exchange
    )
    strategy.reset_common_infra(common_infrastructure)
    executor.reset_common_infra(common_infrastructure)
    portfolio_metrics, indicator_metrics = backtest_loop(
        start_time=pd.Timestamp(start_time),
        end_time=pd.Timestamp(end_time),
        trade_strategy=strategy,
        trade_executor=executor,
    )
    if len(portfolio_metrics) != 1 or len(indicator_metrics) != 1:
        raise QlibAdapterError("expected exactly one daily Qlib executor output")
    portfolio_key = next(iter(portfolio_metrics))
    indicator_key = next(iter(indicator_metrics))
    if portfolio_key != indicator_key:
        raise QlibAdapterError("Qlib portfolio and indicator frequencies do not match")
    report, positions = portfolio_metrics[portfolio_key]
    trade_indicators, indicator_object = indicator_metrics[indicator_key]
    order_history = dict(indicator_object.order_indicator_his)
    return QlibBacktestOutputs(
        raw_portfolio_metrics=portfolio_metrics,
        raw_indicator_metrics=indicator_metrics,
        report=report,
        positions=positions,
        trade_indicators=trade_indicators,
        order_history=order_history,
    )
