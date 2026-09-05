"""Generic covariance, tail-risk, scenario, and constrained-allocation helpers.

The functions in this module operate only on caller-supplied return matrices and
generic asset metadata.  They do not read accounts, infer private paths, forecast
returns, or turn historical scenario frequencies into claims about future odds.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.covariance import LedoitWolf

from .portfolio_analysis import circular_block_indices


@dataclass(frozen=True)
class FilteredSimulationResult:
    """Joint EWMA-filtered paths and the shared residual-row draws behind them."""

    assets: tuple[str, ...]
    paths: np.ndarray
    sampled_indices: np.ndarray
    decay: float
    block_length: int


@dataclass(frozen=True)
class AllocationConstraints:
    """Long-only portfolio bounds for a small scenario-CVaR allocation problem."""

    equity_assets: frozenset[str]
    cash_assets: frozenset[str]
    direct_stock_assets: frozenset[str] = frozenset()
    sector_by_asset: Mapping[str, str] = field(default_factory=dict)
    sector_caps: Mapping[str, float] = field(default_factory=dict)
    min_equity_weight: float = 0.0
    max_equity_weight: float = 1.0
    max_cash_weight: float = 1.0
    min_direct_stock_weight: float = 0.0
    max_direct_stock_weight: float = 1.0
    max_individual_weight: float = 1.0
    max_direct_stock_individual_weight: float = 1.0


@dataclass(frozen=True)
class CvarAllocationResult:
    """Auditable outputs from the constrained empirical-CVaR linear program."""

    weights: pd.Series
    confidence_level: float
    expected_terminal_return: float
    terminal_return_expected_shortfall: float
    expected_shortfall_loss: float
    value_at_risk_loss: float
    anchor_deviation: float
    turnover: float
    objective_value: float
    solver: str


def _return_frame(
    returns: pd.DataFrame,
    *,
    minimum_rows: int,
    role: str,
) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame) or len(returns) < minimum_rows:
        raise ValueError(f"{role} must contain at least {minimum_rows} rows")
    if returns.columns.empty or returns.columns.has_duplicates:
        raise ValueError(f"{role} assets must be non-empty and unique")
    if any(not isinstance(asset, str) or not asset for asset in returns.columns):
        raise ValueError(f"{role} asset names must be non-empty strings")
    if isinstance(returns.index, pd.DatetimeIndex) and (
        returns.index.has_duplicates or not returns.index.is_monotonic_increasing
    ):
        raise ValueError(f"{role} dates must be unique and increasing")
    numeric = returns.apply(pd.to_numeric, errors="coerce").astype(np.float64)
    values = numeric.to_numpy()
    if not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError(f"{role} must be finite simple returns greater than -1")
    return numeric


def _positive_scale(value: float, role: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{role} must be finite and positive")
    return result


def _decay_weights(observations: int, decay: float) -> np.ndarray:
    decay = float(decay)
    if not np.isfinite(decay) or not 0.0 < decay < 1.0:
        raise ValueError("decay must be finite and strictly between zero and one")
    ages = np.arange(observations - 1, -1, -1, dtype=np.float64)
    weights = np.power(decay, ages)
    return weights / weights.sum()


def ledoit_wolf_covariance(
    returns: pd.DataFrame,
    *,
    annualization: float = 1.0,
) -> pd.DataFrame:
    """Estimate a centered Ledoit-Wolf covariance matrix.

    The convention matches :class:`sklearn.covariance.LedoitWolf`: the empirical
    covariance is centered and normalized by ``1 / n`` and is shrunk toward
    ``mu * I``, where ``mu = trace(empirical_covariance) / asset_count``.  The
    fitted covariance is multiplied by ``annualization`` only after shrinkage.
    Use ``annualization=1`` for covariance in the input return frequency.
    """

    frame = _return_frame(returns, minimum_rows=2, role="returns")
    scale = _positive_scale(annualization, "annualization")
    estimate = LedoitWolf(assume_centered=False).fit(frame.to_numpy()).covariance_
    covariance = np.asarray(estimate, dtype=np.float64) * scale
    if not np.isfinite(covariance).all():
        raise AssertionError("Ledoit-Wolf covariance is not finite")
    return pd.DataFrame(covariance, index=frame.columns, columns=frame.columns)


def exponentially_weighted_covariance(
    returns: pd.DataFrame,
    *,
    decay: float = 0.94,
    annualization: float = 1.0,
    center: bool = True,
) -> pd.DataFrame:
    """Return a normalized population EW covariance with newest rows weighted most.

    Row weights are proportional to ``decay ** age`` and sum to one.  With
    ``center=True`` (the default), the same weights define the mean that is removed;
    with ``center=False`` the result is the exponentially weighted second moment.
    No effective-sample-size bias correction is applied.
    """

    frame = _return_frame(returns, minimum_rows=2, role="returns")
    scale = _positive_scale(annualization, "annualization")
    values = frame.to_numpy()
    weights = _decay_weights(len(frame), decay)
    mean = weights @ values if center else np.zeros(values.shape[1], dtype=np.float64)
    centered = values - mean
    covariance = (centered * weights[:, None]).T @ centered
    covariance = 0.5 * (covariance + covariance.T) * scale
    if not np.isfinite(covariance).all():
        raise AssertionError("exponentially weighted covariance is not finite")
    return pd.DataFrame(covariance, index=frame.columns, columns=frame.columns)


def terminal_return_expected_shortfall(
    terminal_returns: np.ndarray | pd.Series,
    *,
    confidence_level: float = 0.95,
) -> float:
    """Average the lower tail of terminal returns at the requested confidence.

    This is a *return* convention: worse results are more negative.  At 95%, the
    result averages the worst 5% of equally weighted terminal-return scenarios; it
    is not the 5th percentile.  A fractional boundary observation is weighted when
    the requested tail contains a non-integer number of scenarios.
    """

    values = np.asarray(terminal_returns, dtype=np.float64)
    confidence = float(confidence_level)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("terminal returns must be a non-empty finite vector")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be strictly between zero and one")

    ordered = np.sort(values)
    tail_mass = (1.0 - confidence) * len(ordered)
    rounded_mass = round(tail_mass)
    if np.isclose(tail_mass, rounded_mass, rtol=0.0, atol=1e-12):
        whole = int(rounded_mass)
        fraction = 0.0
        denominator = float(whole)
    else:
        whole = int(np.floor(tail_mass))
        fraction = tail_mass - whole
        denominator = tail_mass
    if whole == 0:
        return float(ordered[0])
    tail_sum = float(ordered[:whole].sum())
    if fraction > 0.0:
        tail_sum += fraction * float(ordered[whole])
    return tail_sum / denominator


def simulate_ewma_filtered_paths(
    returns: pd.DataFrame,
    *,
    path_count: int,
    horizon_sessions: int,
    block_length: int,
    decay: float = 0.94,
    seed: int = 0,
    variance_floor: float = 1e-12,
) -> FilteredSimulationResult:
    """Simulate joint arithmetic-return paths from blocked EWMA residual vectors.

    Source simple returns are converted to log returns, centered by an exponential
    mean, and standardized by one-step EWMA variances.  Circular blocks resample
    complete residual rows, so every asset on a simulated step uses the same source
    row and cross-asset dependence is retained.  Volatility then evolves pathwise
    under the same EWMA recursion before results are converted back to simple returns.
    """

    frame = _return_frame(returns, minimum_rows=2, role="returns")
    for value, role in (
        (path_count, "path_count"),
        (horizon_sessions, "horizon_sessions"),
        (block_length, "block_length"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{role} must be a positive integer")
    weights = _decay_weights(len(frame), decay)
    floor = _positive_scale(variance_floor, "variance_floor")
    log_returns = np.log1p(frame.to_numpy())
    drift = weights @ log_returns
    centered = log_returns - drift

    state = np.maximum(weights @ np.square(centered), floor)
    conditional_variance = np.empty_like(centered)
    for observation in range(len(frame)):
        conditional_variance[observation] = state
        state = np.maximum(
            float(decay) * state + (1.0 - float(decay)) * np.square(centered[observation]),
            floor,
        )
    residuals = centered / np.sqrt(conditional_variance)
    residuals -= residuals.mean(axis=0)
    residual_scale = np.sqrt(np.mean(np.square(residuals), axis=0))
    active = residual_scale > np.sqrt(floor)
    residuals[:, active] /= residual_scale[active]
    residuals[:, ~active] = 0.0

    indices = circular_block_indices(
        observations=len(frame),
        path_count=int(path_count),
        horizon_sessions=int(horizon_sessions),
        block_length=int(block_length),
        seed=int(seed),
    )
    variances = np.broadcast_to(state, (int(path_count), len(frame.columns))).copy()
    simulated_log = np.empty(
        (int(path_count), int(horizon_sessions), len(frame.columns)), dtype=np.float64
    )
    decay_value = float(decay)
    for session in range(int(horizon_sessions)):
        shocks = residuals[indices[:, session]]
        innovations = np.sqrt(variances) * shocks
        simulated_log[:, session, :] = drift + innovations
        variances = np.maximum(
            decay_value * variances + (1.0 - decay_value) * np.square(innovations),
            floor,
        )
    with np.errstate(over="ignore", invalid="ignore"):
        paths = np.expm1(simulated_log)
    if not np.isfinite(paths).all() or (paths <= -1.0).any():
        raise ValueError("EWMA-filtered simulation produced invalid simple returns")
    return FilteredSimulationResult(
        assets=tuple(frame.columns),
        paths=paths,
        sampled_indices=indices,
        decay=decay_value,
        block_length=min(int(block_length), len(frame)),
    )


def _cap(value: float, role: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{role} must be finite and between zero and one")
    return result


def _weight_vector(
    weights: Mapping[str, float] | None,
    assets: tuple[str, ...],
    role: str,
) -> np.ndarray | None:
    if weights is None:
        return None
    if set(weights) != set(assets):
        raise ValueError(f"{role} must match scenario assets exactly")
    values = np.asarray([float(weights[asset]) for asset in assets], dtype=np.float64)
    if (
        not np.isfinite(values).all()
        or (values < 0.0).any()
        or not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-10)
    ):
        raise ValueError(f"{role} must be finite, nonnegative, and sum to one")
    return values


def optimize_cvar_allocation(
    scenario_returns: pd.DataFrame,
    *,
    constraints: AllocationConstraints,
    confidence_level: float = 0.95,
    anchor_weights: Mapping[str, float] | None = None,
    current_weights: Mapping[str, float] | None = None,
    anchor_penalty: float = 0.0,
    turnover_penalty: float = 0.0,
    minimum_expected_return: float | None = None,
) -> CvarAllocationResult:
    """Minimize empirical terminal-loss CVaR plus optional full-L1 penalties.

    The SciPy/HiGHS linear program is long-only and fully invested.  Equity and
    direct-stock minima/maxima are aggregate. ``max_individual_weight`` applies to
    every asset, while ``max_direct_stock_individual_weight`` further restricts
    direct names. Anchor and turnover penalties multiply
    ``sum(abs(weight - reference_weight))`` (without a one-half turnover convention).
    Historical scenarios remain equally weighted.
    """

    frame = _return_frame(scenario_returns, minimum_rows=2, role="scenario returns")
    assets = tuple(frame.columns)
    known = set(assets)
    confidence = float(confidence_level)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be strictly between zero and one")
    penalties = (float(anchor_penalty), float(turnover_penalty))
    if any(not np.isfinite(value) or value < 0.0 for value in penalties):
        raise ValueError("allocation penalties must be finite and nonnegative")

    equity = set(constraints.equity_assets)
    cash = set(constraints.cash_assets)
    direct = set(constraints.direct_stock_assets)
    sector_by_asset = dict(constraints.sector_by_asset)
    for role, configured in (
        ("equity_assets", equity),
        ("cash_assets", cash),
        ("direct_stock_assets", direct),
        ("sector_by_asset", set(sector_by_asset)),
    ):
        unknown = configured.difference(known)
        if unknown:
            raise ValueError(f"{role} contains unknown assets: {sorted(unknown)}")
    if equity.intersection(cash):
        raise ValueError("equity_assets and cash_assets must be disjoint")
    if not direct.issubset(equity):
        raise ValueError("direct_stock_assets must be a subset of equity_assets")

    equity_floor = _cap(constraints.min_equity_weight, "min_equity_weight")
    equity_cap = _cap(constraints.max_equity_weight, "max_equity_weight")
    cash_cap = _cap(constraints.max_cash_weight, "max_cash_weight")
    direct_floor = _cap(constraints.min_direct_stock_weight, "min_direct_stock_weight")
    direct_cap = _cap(constraints.max_direct_stock_weight, "max_direct_stock_weight")
    individual_cap = _cap(constraints.max_individual_weight, "max_individual_weight")
    direct_individual_cap = _cap(
        constraints.max_direct_stock_individual_weight,
        "max_direct_stock_individual_weight",
    )
    if equity_floor > equity_cap:
        raise ValueError("min_equity_weight cannot exceed max_equity_weight")
    if direct_floor > direct_cap:
        raise ValueError("min_direct_stock_weight cannot exceed max_direct_stock_weight")
    sector_caps = {
        str(sector): _cap(cap, f"sector cap {sector}")
        for sector, cap in constraints.sector_caps.items()
    }
    for sector in sector_caps:
        if sector not in set(sector_by_asset.values()):
            raise ValueError(f"sector cap has no matching assets: {sector}")

    anchor = _weight_vector(anchor_weights, assets, "anchor_weights")
    current = _weight_vector(current_weights, assets, "current_weights")
    if anchor_penalty > 0.0 and anchor is None:
        raise ValueError("anchor_weights are required when anchor_penalty is positive")
    if turnover_penalty > 0.0 and current is None:
        raise ValueError("current_weights are required when turnover_penalty is positive")
    if minimum_expected_return is not None and not np.isfinite(minimum_expected_return):
        raise ValueError("minimum_expected_return must be finite")

    scenarios = frame.to_numpy()
    scenario_count, asset_count = scenarios.shape
    threshold_index = asset_count
    slack_start = threshold_index + 1
    next_index = slack_start + scenario_count
    anchor_start: int | None = None
    turnover_start: int | None = None
    if anchor_penalty > 0.0:
        anchor_start = next_index
        next_index += asset_count
    if turnover_penalty > 0.0:
        turnover_start = next_index
        next_index += asset_count

    objective = np.zeros(next_index, dtype=np.float64)
    objective[threshold_index] = 1.0
    objective[slack_start : slack_start + scenario_count] = 1.0 / (
        (1.0 - confidence) * scenario_count
    )
    if anchor_start is not None:
        objective[anchor_start : anchor_start + asset_count] = float(anchor_penalty)
    if turnover_start is not None:
        objective[turnover_start : turnover_start + asset_count] = float(turnover_penalty)

    inequalities: list[np.ndarray] = []
    upper_bounds: list[float] = []
    for scenario_index, scenario in enumerate(scenarios):
        row = np.zeros(next_index, dtype=np.float64)
        row[:asset_count] = -scenario
        row[threshold_index] = -1.0
        row[slack_start + scenario_index] = -1.0
        inequalities.append(row)
        upper_bounds.append(0.0)

    def add_cap(members: set[str], cap: float) -> None:
        row = np.zeros(next_index, dtype=np.float64)
        row[:asset_count] = [float(asset in members) for asset in assets]
        inequalities.append(row)
        upper_bounds.append(cap)

    def add_floor(members: set[str], floor: float) -> None:
        row = np.zeros(next_index, dtype=np.float64)
        row[:asset_count] = [-float(asset in members) for asset in assets]
        inequalities.append(row)
        upper_bounds.append(-floor)

    add_floor(equity, equity_floor)
    add_cap(equity, equity_cap)
    add_cap(cash, cash_cap)
    add_floor(direct, direct_floor)
    add_cap(direct, direct_cap)
    for sector, cap in sector_caps.items():
        add_cap(
            {asset for asset, asset_sector in sector_by_asset.items() if asset_sector == sector},
            cap,
        )

    if minimum_expected_return is not None:
        row = np.zeros(next_index, dtype=np.float64)
        row[:asset_count] = -scenarios.mean(axis=0)
        inequalities.append(row)
        upper_bounds.append(-float(minimum_expected_return))

    def add_absolute_deviation(start: int, reference: np.ndarray) -> None:
        for asset_index in range(asset_count):
            positive = np.zeros(next_index, dtype=np.float64)
            positive[asset_index] = 1.0
            positive[start + asset_index] = -1.0
            inequalities.append(positive)
            upper_bounds.append(float(reference[asset_index]))

            negative = np.zeros(next_index, dtype=np.float64)
            negative[asset_index] = -1.0
            negative[start + asset_index] = -1.0
            inequalities.append(negative)
            upper_bounds.append(float(-reference[asset_index]))

    if anchor_start is not None:
        assert anchor is not None
        add_absolute_deviation(anchor_start, anchor)
    if turnover_start is not None:
        assert current is not None
        add_absolute_deviation(turnover_start, current)

    equality = np.zeros((1, next_index), dtype=np.float64)
    equality[0, :asset_count] = 1.0
    bounds: list[tuple[float | None, float | None]] = [
        (
            0.0,
            min(individual_cap, direct_individual_cap) if asset in direct else individual_cap,
        )
        for asset in assets
    ]
    bounds.append((None, None))
    bounds.extend((0.0, None) for _ in range(scenario_count))
    if anchor_start is not None:
        bounds.extend((0.0, None) for _ in range(asset_count))
    if turnover_start is not None:
        bounds.extend((0.0, None) for _ in range(asset_count))

    solution = linprog(
        objective,
        A_ub=np.vstack(inequalities),
        b_ub=np.asarray(upper_bounds),
        A_eq=equality,
        b_eq=np.array([1.0]),
        bounds=bounds,
        method="highs",
    )
    if not solution.success or solution.x is None:
        raise ValueError(f"CVaR allocation is infeasible or failed: {solution.message}")

    weights = np.asarray(solution.x[:asset_count], dtype=np.float64)
    weights[np.abs(weights) < 1e-12] = 0.0
    if (
        not np.isfinite(weights).all()
        or (weights < -1e-8).any()
        or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-8)
    ):
        raise AssertionError("CVaR allocation weights failed reconciliation")
    portfolio_returns = scenarios @ weights
    tail_return = terminal_return_expected_shortfall(portfolio_returns, confidence_level=confidence)
    anchor_deviation = float(np.abs(weights - anchor).sum()) if anchor is not None else 0.0
    turnover = float(np.abs(weights - current).sum()) if current is not None else 0.0
    objective_value = (
        -tail_return + float(anchor_penalty) * anchor_deviation + float(turnover_penalty) * turnover
    )
    weight_series = pd.Series(weights, index=assets, name="weight")
    return CvarAllocationResult(
        weights=weight_series,
        confidence_level=confidence,
        expected_terminal_return=float(portfolio_returns.mean()),
        terminal_return_expected_shortfall=tail_return,
        expected_shortfall_loss=-tail_return,
        value_at_risk_loss=float(np.quantile(-portfolio_returns, confidence)),
        anchor_deviation=anchor_deviation,
        turnover=turnover,
        objective_value=float(objective_value),
        solver="scipy.optimize.linprog(method='highs')",
    )


__all__ = [
    "AllocationConstraints",
    "CvarAllocationResult",
    "FilteredSimulationResult",
    "exponentially_weighted_covariance",
    "ledoit_wolf_covariance",
    "optimize_cvar_allocation",
    "simulate_ewma_filtered_paths",
    "terminal_return_expected_shortfall",
]
