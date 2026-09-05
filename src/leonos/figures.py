"""Deterministic figures rendered only from saved evaluation artifacts."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg", force=True)
from matplotlib import dates as mdates  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

_FIGURE_DPI = 150
_ROLLING_SESSIONS = 20
_PRIMARY_COST_BPS = 5


def _require_columns(frame: pd.DataFrame, required: set[str], *, source: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _rankic_frame(path: str | Path, *, rolling_sessions: int = _ROLLING_SESSIONS) -> pd.DataFrame:
    """Load valid daily paired differences and calculate the declared rolling mean."""

    source = Path(path)
    frame = pd.read_parquet(source)
    _require_columns(frame, {"origin", "coverage_ok", "delta_rankic"}, source=source)
    if rolling_sessions <= 0:
        raise ValueError("rolling_sessions must be positive")
    if frame["coverage_ok"].isna().any() or not pd.api.types.is_bool_dtype(frame["coverage_ok"]):
        raise ValueError(f"{source} coverage_ok must be a complete boolean column")

    valid = frame.loc[frame["coverage_ok"], ["origin", "delta_rankic"]].copy()
    if valid.empty:
        raise ValueError(f"{source} contains no dates with valid paired RankIC")
    valid["origin"] = pd.to_datetime(valid["origin"], errors="raise")
    valid["delta_rankic"] = pd.to_numeric(valid["delta_rankic"], errors="raise")
    if valid["origin"].duplicated().any():
        raise ValueError(f"{source} contains duplicate forecast origins")
    if not np.isfinite(valid["delta_rankic"].to_numpy(dtype=float)).all():
        raise ValueError(f"{source} contains non-finite valid paired RankIC differences")

    valid = valid.sort_values("origin", kind="stable").reset_index(drop=True)
    valid["rolling_mean"] = (
        valid["delta_rankic"]
        .rolling(
            window=rolling_sessions,
            min_periods=rolling_sessions,
        )
        .mean()
    )
    return valid


def _wealth_frame(
    evaluation_root: str | Path,
    *,
    initial_cash: float,
    cost_bps: int = _PRIMARY_COST_BPS,
) -> pd.DataFrame:
    """Load and reconcile the three persisted net account-value series."""

    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash must be finite and positive")
    root = Path(evaluation_root) / "portfolio"
    specifications = {
        "Kronos-base": (
            root / "kronos" / f"cost_bps={cost_bps}" / "qlib_report.parquet",
            "datetime",
            "compounded_wealth",
            "account",
        ),
        "LightGBM": (
            root / "lightgbm" / f"cost_bps={cost_bps}" / "qlib_report.parquet",
            "datetime",
            "compounded_wealth",
            "account",
        ),
        "95%-invested equal-dollar reference": (
            root / "equal_weight_buy_hold" / f"cost_bps={cost_bps}" / "account.parquet",
            "session",
            "account_value",
            None,
        ),
    }

    series: dict[str, pd.Series] = {}
    common_dates: pd.DatetimeIndex | None = None
    for label, (path, date_column, value_column, account_column) in specifications.items():
        frame = pd.read_parquet(path)
        required = {date_column, value_column, "net_return"}
        if account_column is not None:
            required.add(account_column)
        _require_columns(frame, required, source=path)
        selected = frame.loc[:, list(required)].copy()
        selected[date_column] = pd.to_datetime(selected[date_column], errors="raise")
        if selected[date_column].duplicated().any():
            raise ValueError(f"{path} contains duplicate account dates")
        selected = selected.sort_values(date_column, kind="stable").reset_index(drop=True)
        if selected.empty:
            raise ValueError(f"{path} contains no account rows")

        net_return = pd.to_numeric(selected["net_return"], errors="raise").to_numpy(dtype=float)
        values = pd.to_numeric(selected[value_column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(net_return).all() or not np.isfinite(values).all():
            raise ValueError(f"{path} contains non-finite net returns or account values")
        if (net_return <= -1.0).any() or (values <= 0.0).any():
            raise ValueError(f"{path} contains an invalid return or account value")

        compounded = initial_cash * np.cumprod(1.0 + net_return)
        if not np.allclose(values, compounded, rtol=1e-10, atol=1e-6):
            raise ValueError(f"{path} account values do not reconcile to compounded net returns")
        if account_column is not None:
            accounts = pd.to_numeric(selected[account_column], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(accounts).all() or not np.allclose(
                accounts, values, rtol=1e-10, atol=1e-6
            ):
                raise ValueError(f"{path} account and compounded wealth columns do not reconcile")

        dates = pd.DatetimeIndex(selected[date_column])
        if common_dates is None:
            common_dates = dates
        elif not dates.equals(common_dates):
            raise ValueError("5-bps portfolio account artifacts do not share the same dates")
        series[label] = pd.Series(values, index=dates, name=label)

    assert common_dates is not None  # all fixed specifications are traversed above
    return pd.DataFrame(series, index=common_dates)


def _atomic_save_png(figure: Figure, destination: str | Path) -> Path:
    """Save a PNG through a same-directory temporary file and atomically replace it."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    try:
        figure.savefig(
            temporary,
            format="png",
            dpi=_FIGURE_DPI,
            bbox_inches="tight",
            facecolor="white",
            metadata={"Software": "Leonos"},
        )
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


def plot_rankic_difference(
    daily_metrics_path: str | Path,
    destination: str | Path,
    *,
    seed: int,
    rolling_sessions: int = _ROLLING_SESSIONS,
) -> Path:
    """Plot daily paired RankIC differences and their full-window rolling mean."""

    frame = _rankic_frame(daily_metrics_path, rolling_sessions=rolling_sessions)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        figure, axis = plt.subplots(figsize=(10.0, 4.8))
        axis.plot(
            frame["origin"],
            frame["delta_rankic"],
            color="#7a8793",
            alpha=0.55,
            linewidth=0.8,
            label="Daily Δ RankIC",
        )
        axis.plot(
            frame["origin"],
            frame["rolling_mean"],
            color="#1f5a94",
            linewidth=2.0,
            label=f"{rolling_sessions}-session rolling mean",
        )
        axis.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0, label="Zero")
        axis.set_title(f"Paired daily RankIC difference — primary seed {seed}")
        axis.set_xlabel("Forecast origin")
        axis.set_ylabel("RankIC difference (Kronos − LightGBM)")
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        axis.grid(axis="y", color="#d7dde3", linewidth=0.6)
        axis.legend(frameon=False, ncol=3, loc="upper right")
        figure.tight_layout()
        try:
            return _atomic_save_png(figure, destination)
        finally:
            plt.close(figure)


def plot_net_wealth(
    evaluation_root: str | Path,
    destination: str | Path,
    *,
    seed: int,
    initial_cash: float,
    cost_bps: int = _PRIMARY_COST_BPS,
) -> Path:
    """Plot reconciled saved account values for the shared 5-bps simulation."""

    wealth = _wealth_frame(evaluation_root, initial_cash=initial_cash, cost_bps=cost_bps)
    colors = {
        "Kronos-base": "#1f5a94",
        "LightGBM": "#d66a1f",
        "95%-invested equal-dollar reference": "#606a73",
    }
    styles = {"95%-invested equal-dollar reference": "--"}
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        figure, axis = plt.subplots(figsize=(10.0, 4.8))
        for label in wealth.columns:
            axis.plot(
                wealth.index,
                wealth[label] / 1_000_000.0,
                label=label,
                color=colors[label],
                linestyle=styles.get(label, "-"),
                linewidth=1.8,
            )
        axis.set_title(f"Compounded net wealth at {cost_bps} bps per side — seed {seed}")
        axis.set_xlabel("Execution session (first point follows initial fills)")
        axis.set_ylabel("End-of-session net wealth (USD millions)")
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        axis.grid(axis="y", color="#d7dde3", linewidth=0.6)
        axis.legend(frameon=False, loc="upper left")
        figure.tight_layout()
        try:
            return _atomic_save_png(figure, destination)
        finally:
            plt.close(figure)


def render_result_figures(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path = Path("reports") / "figures",
) -> dict[str, Path]:
    """Regenerate the two primary-seed result figures and return their paths."""

    seed = int(config.get("experiment", {}).get("seed", 42))
    cost_bps = int(config.get("portfolio", {}).get("cost_bps_per_side", _PRIMARY_COST_BPS))
    if cost_bps != _PRIMARY_COST_BPS:
        raise ValueError("the declared primary result figure requires 5 bps per side")
    initial_cash = float(config["portfolio"]["initial_cash"])
    evaluation_root = Path(config["paths"]["artifacts"]) / "evaluation" / f"seed={seed}"
    destination = Path(output_dir)
    return {
        "rankic_difference": plot_rankic_difference(
            evaluation_root / "daily_metrics.parquet",
            destination / "rankic-difference.png",
            seed=seed,
        ),
        "net_wealth": plot_net_wealth(
            evaluation_root,
            destination / "net-wealth.png",
            seed=seed,
            initial_cash=initial_cash,
            cost_bps=cost_bps,
        ),
    }
