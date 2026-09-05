from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from leonos.aggressive_risk import (
    AllocationConstraints,
    exponentially_weighted_covariance,
    ledoit_wolf_covariance,
    optimize_cvar_allocation,
    simulate_ewma_filtered_paths,
    terminal_return_expected_shortfall,
)


def test_covariances_are_finite_and_ewma_responds_to_recent_volatility() -> None:
    rng = np.random.default_rng(17)
    first = rng.normal(0.0, 0.002, 40)
    recent = rng.normal(0.0, 0.030, 40)
    common = np.concatenate([first, recent])
    returns = pd.DataFrame(
        {
            "fictional_equity": common,
            "fictional_fund": 0.5 * common + rng.normal(0.0, 0.003, len(common)),
        },
        index=pd.bdate_range("2024-01-02", periods=len(common)),
    )

    shrunk = ledoit_wolf_covariance(returns)
    annualized = ledoit_wolf_covariance(returns, annualization=252.0)
    responsive = exponentially_weighted_covariance(returns, decay=0.70)
    persistent = exponentially_weighted_covariance(returns, decay=0.99)

    np.testing.assert_allclose(shrunk, shrunk.T)
    np.testing.assert_allclose(annualized, shrunk * 252.0)
    assert np.linalg.eigvalsh(shrunk.to_numpy()).min() >= -1e-12
    assert (
        responsive.loc["fictional_equity", "fictional_equity"]
        > persistent.loc["fictional_equity", "fictional_equity"]
    )
    assert np.isfinite(responsive.to_numpy()).all()


def test_terminal_return_expected_shortfall_is_not_the_fifth_percentile() -> None:
    terminal_returns = np.array([-0.50, -0.20, *([0.0] * 38)])

    expected_shortfall = terminal_return_expected_shortfall(terminal_returns)
    fifth_percentile = float(np.quantile(terminal_returns, 0.05))

    assert expected_shortfall == pytest.approx(-0.35)
    assert expected_shortfall < fifth_percentile


def test_ewma_filtered_blocks_preserve_joint_asset_vectors() -> None:
    rng = np.random.default_rng(23)
    common = rng.normal(0.0005, 0.015, 72)
    returns = pd.DataFrame(
        {
            "fictional_a": common,
            "fictional_a_clone": common,
            "fictional_b": 0.4 * common + rng.normal(0.0, 0.008, len(common)),
        },
        index=pd.bdate_range("2024-01-02", periods=len(common)),
    )

    result = simulate_ewma_filtered_paths(
        returns,
        path_count=24,
        horizon_sessions=11,
        block_length=3,
        decay=0.94,
        seed=31,
    )

    assert result.paths.shape == (24, 11, 3)
    assert result.sampled_indices.shape == (24, 11)
    assert np.isfinite(result.paths).all()
    assert (result.paths > -1.0).all()
    np.testing.assert_allclose(result.paths[:, :, 0], result.paths[:, :, 1])
    for start in range(0, result.sampled_indices.shape[1], result.block_length):
        block = result.sampled_indices[:, start : start + result.block_length]
        if block.shape[1] > 1:
            assert (np.diff(block, axis=1) % len(returns) == 1).all()


def _allocation_inputs() -> tuple[pd.DataFrame, AllocationConstraints]:
    rng = np.random.default_rng(29)
    common = rng.normal(0.008, 0.055, 80)
    scenarios = pd.DataFrame(
        {
            "broad_equity": 0.6 * common + rng.normal(0.003, 0.015, len(common)),
            "direct_alpha": common + rng.normal(0.005, 0.040, len(common)),
            "direct_beta": 0.5 * common + rng.normal(0.002, 0.050, len(common)),
            "cash": np.full(len(common), 0.002),
        }
    )
    limits = AllocationConstraints(
        equity_assets=frozenset({"broad_equity", "direct_alpha", "direct_beta"}),
        cash_assets=frozenset({"cash"}),
        direct_stock_assets=frozenset({"direct_alpha", "direct_beta"}),
        sector_by_asset={
            "broad_equity": "broad",
            "direct_alpha": "technology",
            "direct_beta": "healthcare",
        },
        sector_caps={"broad": 0.45, "technology": 0.20, "healthcare": 0.20},
        min_equity_weight=0.60,
        max_equity_weight=0.65,
        max_cash_weight=0.35,
        min_direct_stock_weight=0.15,
        max_direct_stock_weight=0.20,
        max_individual_weight=0.45,
        max_direct_stock_individual_weight=0.10,
    )
    return scenarios, limits


def test_cvar_allocation_obeys_all_caps_and_reports_finite_tail_risk() -> None:
    scenarios, limits = _allocation_inputs()
    reference = {
        "broad_equity": 0.25,
        "direct_alpha": 0.15,
        "direct_beta": 0.10,
        "cash": 0.50,
    }

    result = optimize_cvar_allocation(
        scenarios,
        constraints=limits,
        anchor_weights=reference,
        current_weights=reference,
        anchor_penalty=0.05,
        turnover_penalty=0.05,
    )
    weights = result.weights

    assert weights.sum() == pytest.approx(1.0)
    assert weights[["broad_equity", "direct_alpha", "direct_beta"]].sum() >= 0.60 - 1e-9
    assert weights[["broad_equity", "direct_alpha", "direct_beta"]].sum() <= 0.65 + 1e-9
    assert weights[["cash"]].sum() <= 0.35 + 1e-9
    assert weights[["direct_alpha", "direct_beta"]].sum() >= 0.15 - 1e-9
    assert weights[["direct_alpha", "direct_beta"]].sum() <= 0.20 + 1e-9
    assert weights.max() <= 0.45 + 1e-9
    assert weights[["direct_alpha", "direct_beta"]].max() <= 0.10 + 1e-9
    assert weights["broad_equity"] <= 0.45 + 1e-9
    assert weights["direct_alpha"] <= 0.10 + 1e-9
    assert weights["direct_beta"] <= 0.10 + 1e-9
    np.testing.assert_allclose(
        weights.loc[["broad_equity", "direct_alpha", "direct_beta", "cash"]],
        [0.45, 0.10, 0.10, 0.35],
        atol=1e-8,
    )
    portfolio_returns = scenarios.to_numpy() @ weights.to_numpy()
    assert result.terminal_return_expected_shortfall == pytest.approx(
        terminal_return_expected_shortfall(portfolio_returns)
    )
    assert np.isfinite(
        [
            result.expected_terminal_return,
            result.expected_shortfall_loss,
            result.value_at_risk_loss,
            result.anchor_deviation,
            result.turnover,
            result.objective_value,
        ]
    ).all()

    infeasible = replace(
        limits,
        min_equity_weight=0.70,
        max_equity_weight=0.75,
    )
    with pytest.raises(ValueError, match="infeasible"):
        optimize_cvar_allocation(scenarios, constraints=infeasible)


def test_anchor_and_turnover_penalties_reduce_reference_deviation() -> None:
    scenarios, _ = _allocation_inputs()
    relaxed = AllocationConstraints(
        equity_assets=frozenset({"broad_equity", "direct_alpha", "direct_beta"}),
        cash_assets=frozenset({"cash"}),
        direct_stock_assets=frozenset({"direct_alpha", "direct_beta"}),
        sector_by_asset={"direct_alpha": "technology", "direct_beta": "healthcare"},
        sector_caps={"technology": 0.20, "healthcare": 0.20},
        max_equity_weight=0.75,
        max_cash_weight=0.35,
        max_direct_stock_weight=0.30,
        max_individual_weight=0.55,
    )
    reference = {
        "broad_equity": 0.45,
        "direct_alpha": 0.10,
        "direct_beta": 0.10,
        "cash": 0.35,
    }

    unpenalized = optimize_cvar_allocation(
        scenarios,
        constraints=relaxed,
        anchor_weights=reference,
        current_weights=reference,
    )
    penalized = optimize_cvar_allocation(
        scenarios,
        constraints=relaxed,
        anchor_weights=reference,
        current_weights=reference,
        anchor_penalty=0.10,
        turnover_penalty=0.10,
    )

    assert penalized.anchor_deviation < unpenalized.anchor_deviation
    assert penalized.turnover < unpenalized.turnover
    np.testing.assert_allclose(penalized.weights.to_numpy(), list(reference.values()), atol=1e-8)
