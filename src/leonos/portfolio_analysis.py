"""Generic paired multi-asset scenarios for private portfolio research.

All values are expressed in one reporting currency before they enter this module.
The engine never contains or writes account-specific data; callers supply weights,
return matrices, and explicit private destinations.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class Policy:
    """One transparent target policy and its fixed rebalance cadence."""

    name: str
    target_weights: Mapping[str, float]
    rebalance_every_sessions: int | None
    initial_rebalance: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("policy name must be non-empty")
        if self.rebalance_every_sessions is not None and self.rebalance_every_sessions < 1:
            raise ValueError("rebalance cadence must be positive or None")


@dataclass(frozen=True)
class Friction:
    """Per-asset trading friction plus reporting-currency conversion friction."""

    default_trade_bps: float = 5.0
    trade_bps_by_asset: Mapping[str, float] | None = None
    usd_assets: frozenset[str] = frozenset()
    cash_assets: frozenset[str] = frozenset()
    fx_bps: float = 0.0

    def __post_init__(self) -> None:
        rates = [self.default_trade_bps, self.fx_bps]
        rates.extend((self.trade_bps_by_asset or {}).values())
        if any(not np.isfinite(value) or value < 0 for value in rates):
            raise ValueError("friction rates must be finite and nonnegative")


@dataclass(frozen=True)
class PolicyPathResult:
    """Paired pathwise outputs for one policy/horizon."""

    policy: str
    horizon_sessions: int
    initial_value: float
    sample_fingerprint: str
    ending_value: np.ndarray
    maximum_drawdown: np.ndarray
    traded_notional: np.ndarray
    trading_cost: np.ndarray
    fx_converted_notional: np.ndarray
    fx_cost: np.ndarray


def validate_return_frame(returns: pd.DataFrame) -> pd.DataFrame:
    """Require a finite, ordered joint return panel with no impossible losses."""

    if returns.empty or returns.columns.empty:
        raise ValueError("return panel must be non-empty")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("return panel must have a DatetimeIndex")
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("return dates must be unique and increasing")
    if returns.columns.has_duplicates:
        raise ValueError("return assets must be unique")
    numeric = returns.apply(pd.to_numeric, errors="coerce").astype(float)
    values = numeric.to_numpy()
    if not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("returns must be finite and greater than -1")
    return numeric


def reporting_currency_returns(
    native_returns: pd.DataFrame,
    *,
    currencies: Mapping[str, str],
    usd_cad_returns: pd.Series,
    reporting_currency: str = "CAD",
) -> pd.DataFrame:
    """Translate native total returns into CAD while preserving joint dates.

    For a USD asset, ``(1 + asset_return) * (1 + USD/CAD_return) - 1`` is used.
    CAD assets are unchanged. No other currency is silently approximated.
    """

    if reporting_currency != "CAD":
        raise ValueError("the current generic converter supports CAD reporting only")
    frame = validate_return_frame(native_returns)
    fx = pd.to_numeric(usd_cad_returns, errors="coerce").reindex(frame.index)
    if not np.isfinite(fx.to_numpy(float)).all() or (fx <= -1.0).any():
        raise ValueError("USD/CAD returns must align and be finite")
    if set(currencies) != set(frame.columns):
        raise ValueError("currency mapping must match return assets exactly")
    out = frame.copy()
    for asset in frame.columns:
        currency = str(currencies[asset]).upper()
        if currency == "USD":
            out[asset] = (1.0 + frame[asset]) * (1.0 + fx) - 1.0
        elif currency != "CAD":
            raise ValueError(f"unsupported currency for {asset}: {currency}")
    return out


def add_non_interest_cash_returns(
    reporting_returns: pd.DataFrame,
    *,
    usd_cad_returns: pd.Series,
    cad_name: str = "cash_cad",
    usd_name: str = "cash_usd",
) -> pd.DataFrame:
    """Add zero-yield CAD cash and zero-yield USD cash viewed from CAD."""

    out = validate_return_frame(reporting_returns).copy()
    fx = pd.to_numeric(usd_cad_returns, errors="coerce").reindex(out.index)
    if not np.isfinite(fx.to_numpy(float)).all() or (fx <= -1.0).any():
        raise ValueError("USD/CAD returns must align and be finite")
    if cad_name == usd_name:
        raise ValueError("CAD and USD cash names must differ")
    if cad_name in out or usd_name in out:
        raise ValueError("cash column already exists")
    out[cad_name] = 0.0
    out[usd_name] = fx.to_numpy(float)
    return out


def circular_block_indices(
    *,
    observations: int,
    path_count: int,
    horizon_sessions: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    """Draw one shared set of circular moving-block indices."""

    if min(observations, path_count, horizon_sessions, block_length) < 1:
        raise ValueError("block sampler dimensions must be positive")
    effective = min(observations, block_length)
    blocks = math.ceil(horizon_sessions / effective)
    rng = np.random.default_rng(int(seed))
    starts = rng.integers(0, observations, size=(path_count, blocks), dtype=np.int64)
    offsets = np.arange(effective, dtype=np.int64)
    return ((starts[:, :, None] + offsets) % observations).reshape(path_count, -1)[
        :, :horizon_sessions
    ]


def sample_joint_return_paths(returns: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    """Apply shared indices to the complete asset vector on every date."""

    frame = validate_return_frame(returns)
    if indices.ndim != 2 or indices.size == 0:
        raise ValueError("indices must be a non-empty path-by-session matrix")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("sample indices must be integers")
    if indices.min() < 0 or indices.max() >= len(frame):
        raise ValueError("sample index is outside the return panel")
    return frame.to_numpy(float)[indices]


def rolling_origin_return_paths(
    returns: pd.DataFrame,
    *,
    horizon_sessions: int,
    step_sessions: int = 21,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Create chronological realized windows without crossing the data endpoint."""

    frame = validate_return_frame(returns)
    if horizon_sessions < 1 or step_sessions < 1 or len(frame) < horizon_sessions:
        raise ValueError("invalid rolling-origin horizon or step")
    starts = np.arange(0, len(frame) - horizon_sessions + 1, step_sessions)
    paths = np.stack(
        [frame.iloc[start : start + horizon_sessions].to_numpy(float) for start in starts]
    )
    return pd.DatetimeIndex(frame.index[starts]), paths


def _weights(values: Mapping[str, float], assets: Sequence[str], role: str) -> np.ndarray:
    if set(values) != set(assets):
        raise ValueError(f"{role} weights must match return assets exactly")
    result = np.asarray([float(values[asset]) for asset in assets], dtype=np.float64)
    if not np.isfinite(result).all() or (result < 0.0).any():
        raise ValueError(f"{role} weights must be finite and nonnegative")
    if not np.isclose(result.sum(), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError(f"{role} weights must sum to one")
    return result


def _rebalance(
    values: np.ndarray,
    *,
    target_weights: np.ndarray,
    assets: Sequence[str],
    friction: Friction,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total_before = values.sum(axis=1)
    if (total_before <= 0.0).any():
        raise ValueError("portfolio value must remain positive")
    asset_rates = np.asarray(
        [
            float((friction.trade_bps_by_asset or {}).get(asset, friction.default_trade_bps))
            / 10_000.0
            for asset in assets
        ]
    )
    tradable = np.asarray([asset not in friction.cash_assets for asset in assets])
    usd = np.asarray([asset in friction.usd_assets for asset in assets])
    fx_rate = friction.fx_bps / 10_000.0
    target_usd_weight = float(target_weights[usd].sum())
    current_usd_value = values[:, usd].sum(axis=1)
    # Currency conversion is the intentional change in the USD bucket at the
    # pre-cost target. Trading friction within that bucket is not another FX
    # conversion (for example, moving USD cash into a USD-listed fund).
    fx_notional = np.abs(total_before * target_usd_weight - current_usd_value)
    fx_cost = fx_notional * fx_rate
    total_after = total_before - fx_cost
    desired = total_after[:, None] * target_weights
    for _ in range(12):
        traded = np.abs(desired - values)[:, tradable].sum(axis=1)
        trading_cost = (
            np.abs(desired - values)[:, tradable]
            * asset_rates[None, tradable]
        ).sum(axis=1)
        next_total = total_before - trading_cost - fx_cost
        if (next_total <= 0.0).any():
            raise ValueError("friction exhausted portfolio value")
        next_desired = next_total[:, None] * target_weights
        if np.allclose(next_desired, desired, rtol=1e-12, atol=1e-10):
            desired = next_desired
            total_after = next_total
            break
        desired = next_desired
        total_after = next_total
    else:  # pragma: no cover - contraction should converge for realistic rates
        raise RuntimeError("rebalance friction calculation did not converge")
    traded = np.abs(desired - values)[:, tradable].sum(axis=1)
    trading_cost = (
        np.abs(desired - values)[:, tradable] * asset_rates[None, tradable]
    ).sum(axis=1)
    if not np.allclose(
        desired.sum(axis=1) + trading_cost + fx_cost,
        total_before,
        rtol=1e-12,
        atol=1e-8,
    ):
        raise AssertionError("post-rebalance values do not reconcile")
    return desired, traded, trading_cost, fx_notional, fx_cost


def simulate_policy_paths(
    sampled_returns: np.ndarray,
    *,
    assets: Sequence[str],
    initial_weights: Mapping[str, float],
    policy: Policy,
    friction: Friction | None = None,
    initial_value: float = 100.0,
) -> PolicyPathResult:
    """Replay one fixed policy on paired joint return paths.

    Each return row represents the change between two permitted valuation/trade
    instants. Rebalancing occurs before the first return when requested and then
    after each completed cadence, before the following return.
    """

    paths = np.asarray(sampled_returns, dtype=np.float64)
    if paths.ndim != 3 or paths.shape[2] != len(assets) or paths.shape[1] < 1:
        raise ValueError("sampled returns must have shape paths x sessions x assets")
    if not np.isfinite(paths).all() or (paths <= -1.0).any():
        raise ValueError("sampled returns must be finite and greater than -1")
    if not np.isfinite(initial_value) or initial_value <= 0.0:
        raise ValueError("initial value must be finite and positive")
    if not assets or len(set(assets)) != len(assets):
        raise ValueError("assets must be non-empty and unique")
    initial = _weights(initial_weights, assets, "initial")
    target = _weights(policy.target_weights, assets, "target")
    friction = friction or Friction()
    count, horizon, _ = paths.shape
    if count < 1:
        raise ValueError("sampled returns must contain at least one path")
    known_assets = set(assets)
    for name, configured in (
        ("USD assets", friction.usd_assets),
        ("cash assets", friction.cash_assets),
        ("per-asset friction", (friction.trade_bps_by_asset or {}).keys()),
    ):
        unknown = set(configured).difference(known_assets)
        if unknown:
            raise ValueError(f"{name} contain unknown assets: {sorted(unknown)}")
    values = np.broadcast_to(initial_value * initial, (count, len(assets))).copy()
    traded_total = np.zeros(count)
    trading_cost_total = np.zeros(count)
    fx_notional_total = np.zeros(count)
    fx_cost_total = np.zeros(count)
    running_peak = np.full(count, float(initial_value))
    maximum_drawdown = np.zeros(count)

    def rebalance() -> None:
        nonlocal values
        values, traded, trade_cost, fx_notional, fx_cost = _rebalance(
            values,
            target_weights=target,
            assets=assets,
            friction=friction,
        )
        traded_total[:] += traded
        trading_cost_total[:] += trade_cost
        fx_notional_total[:] += fx_notional
        fx_cost_total[:] += fx_cost

    if policy.initial_rebalance:
        rebalance()
        total = values.sum(axis=1)
        maximum_drawdown = np.maximum(maximum_drawdown, 1.0 - total / running_peak)

    for session in range(horizon):
        values *= 1.0 + paths[:, session, :]
        total = values.sum(axis=1)
        running_peak = np.maximum(running_peak, total)
        maximum_drawdown = np.maximum(maximum_drawdown, 1.0 - total / running_peak)
        cadence = policy.rebalance_every_sessions
        if cadence is not None and (session + 1) % cadence == 0 and session + 1 < horizon:
            rebalance()
            total = values.sum(axis=1)
            maximum_drawdown = np.maximum(
                maximum_drawdown, 1.0 - total / running_peak
            )

    ending = values.sum(axis=1)
    if (
        not np.isfinite(ending).all()
        or (values < -1e-10).any()
    ):
        raise AssertionError("portfolio paths failed reconciliation")
    contiguous_paths = np.ascontiguousarray(paths)
    fingerprint = hashlib.sha256()
    fingerprint.update(
        json.dumps(
            {"assets": list(assets), "shape": list(paths.shape), "dtype": str(paths.dtype)},
            separators=(",", ":"),
        ).encode()
    )
    fingerprint.update(memoryview(contiguous_paths).cast("B"))
    sample_fingerprint = fingerprint.hexdigest()
    return PolicyPathResult(
        policy=policy.name,
        horizon_sessions=horizon,
        initial_value=float(initial_value),
        sample_fingerprint=sample_fingerprint,
        ending_value=ending,
        maximum_drawdown=maximum_drawdown,
        traded_notional=traded_total,
        trading_cost=trading_cost_total,
        fx_converted_notional=fx_notional_total,
        fx_cost=fx_cost_total,
    )


def summarize_policy_paths(result: PolicyPathResult) -> dict[str, float | int | str]:
    """Summarize pathwise outputs without interpreting frequencies as probabilities."""

    ending = np.asarray(result.ending_value, dtype=float)
    drawdown = np.asarray(result.maximum_drawdown, dtype=float)
    return {
        "policy": result.policy,
        "horizon_sessions": int(result.horizon_sessions),
        "path_count": int(len(ending)),
        "ending_value_mean": float(ending.mean()),
        "ending_value_p05": float(np.quantile(ending, 0.05)),
        "ending_value_median": float(np.median(ending)),
        "ending_value_p95": float(np.quantile(ending, 0.95)),
        "scenario_fraction_below_start": float(
            np.mean(ending < result.initial_value)
        ),
        "maximum_drawdown_p95": float(np.quantile(drawdown, 0.95)),
        "maximum_drawdown_median": float(np.median(drawdown)),
        "traded_notional_mean": float(np.mean(result.traded_notional)),
        "trading_cost_mean": float(np.mean(result.trading_cost)),
        "fx_converted_notional_mean": float(np.mean(result.fx_converted_notional)),
        "fx_cost_mean": float(np.mean(result.fx_cost)),
    }


def paired_ending_value_difference(
    result: PolicyPathResult,
    reference: PolicyPathResult,
) -> np.ndarray:
    """Return pathwise ending-value differences only for identical scenario draws."""

    if (
        result.sample_fingerprint != reference.sample_fingerprint
        or result.horizon_sessions != reference.horizon_sessions
        or result.initial_value != reference.initial_value
        or result.ending_value.shape != reference.ending_value.shape
    ):
        raise ValueError("policy results do not use the same paired scenario paths")
    return np.asarray(result.ending_value) - np.asarray(reference.ending_value)


def shrunk_covariance(returns: pd.DataFrame, *, annualization: float = 252.0) -> pd.DataFrame:
    """Estimate a stable annualized covariance using Ledoit-Wolf shrinkage."""

    frame = validate_return_frame(returns)
    if len(frame) < 3 or annualization <= 0:
        raise ValueError("covariance requires at least three rows and positive annualization")
    estimate = LedoitWolf().fit(frame.to_numpy(float)).covariance_ * annualization
    return pd.DataFrame(estimate, index=frame.columns, columns=frame.columns)


def deterministic_stress_value(
    *,
    initial_value: float,
    weights: Mapping[str, float],
    asset_returns: Mapping[str, float],
) -> float:
    """Apply a declared one-step joint shock without summing asset quantiles."""

    if set(weights) != set(asset_returns):
        raise ValueError("stress returns must match policy assets")
    assets = tuple(weights)
    weight_array = _weights(weights, assets, "stress")
    shocks = np.asarray([float(asset_returns[asset]) for asset in assets])
    if not np.isfinite(shocks).all() or (shocks <= -1.0).any():
        raise ValueError("stress returns must be finite and greater than -1")
    return float(initial_value * np.sum(weight_array * (1.0 + shocks)))


__all__ = [
    "Friction",
    "Policy",
    "PolicyPathResult",
    "add_non_interest_cash_returns",
    "circular_block_indices",
    "deterministic_stress_value",
    "paired_ending_value_difference",
    "reporting_currency_returns",
    "rolling_origin_return_paths",
    "sample_joint_return_paths",
    "shrunk_covariance",
    "simulate_policy_paths",
    "summarize_policy_paths",
    "validate_return_frame",
]
