from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leonos.portfolio_analysis import (
    Friction,
    Policy,
    add_non_interest_cash_returns,
    circular_block_indices,
    deterministic_stress_value,
    paired_ending_value_difference,
    reporting_currency_returns,
    rolling_origin_return_paths,
    sample_joint_return_paths,
    shrunk_covariance,
    simulate_policy_path_endpoints,
    simulate_policy_paths,
    summarize_policy_paths,
)


def _returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fund_usd": [0.01, -0.02, 0.03, 0.00, 0.01, -0.01],
            "fund_cad": [0.00, 0.01, -0.01, 0.02, -0.01, 0.01],
        },
        index=pd.bdate_range("2024-01-02", periods=6),
    )


def test_currency_translation_and_cash_are_jointly_aligned() -> None:
    native = _returns()
    fx = pd.Series([0.01, 0.0, -0.01, 0.0, 0.02, 0.0], index=native.index)
    reporting = reporting_currency_returns(
        native,
        currencies={"fund_usd": "USD", "fund_cad": "CAD"},
        usd_cad_returns=fx,
    )
    assert reporting.loc[native.index[0], "fund_usd"] == pytest.approx(0.0201)
    assert reporting["fund_cad"].equals(native["fund_cad"])
    with_cash = add_non_interest_cash_returns(reporting, usd_cad_returns=fx)
    assert (with_cash["cash_cad"] == 0.0).all()
    pd.testing.assert_series_equal(with_cash["cash_usd"], fx, check_names=False)


def test_joint_block_sampling_is_deterministic_and_preserves_asset_rows() -> None:
    returns = _returns()
    first = circular_block_indices(
        observations=len(returns),
        path_count=20,
        horizon_sessions=5,
        block_length=2,
        seed=7,
    )
    second = circular_block_indices(
        observations=len(returns),
        path_count=20,
        horizon_sessions=5,
        block_length=2,
        seed=7,
    )
    np.testing.assert_array_equal(first, second)
    sampled = sample_joint_return_paths(returns, first)
    for path in range(len(first)):
        np.testing.assert_allclose(sampled[path], returns.to_numpy()[first[path]])


def test_policy_rebalance_costs_and_values_reconcile_without_negative_cash() -> None:
    assets = ("fund_usd", "fund_cad", "cash_cad")
    sampled = np.zeros((4, 4, 3), dtype=float)
    policy = Policy(
        "fictional-growth",
        {"fund_usd": 0.4, "fund_cad": 0.4, "cash_cad": 0.2},
        rebalance_every_sessions=2,
    )
    result = simulate_policy_paths(
        sampled,
        assets=assets,
        initial_weights={"fund_usd": 0.8, "fund_cad": 0.0, "cash_cad": 0.2},
        policy=policy,
        friction=Friction(
            default_trade_bps=10,
            usd_assets=frozenset({"fund_usd"}),
            cash_assets=frozenset({"cash_cad"}),
            fx_bps=100,
        ),
        initial_value=100.0,
    )
    assert (result.ending_value > 0).all()
    assert (result.ending_value < 100.0).all()
    assert (result.trading_cost > 0).all()
    assert (result.fx_cost > 0).all()
    summary = summarize_policy_paths(result)
    assert summary["scenario_fraction_below_start"] == 1.0
    assert summary["path_count"] == 4


def test_hold_policy_does_not_trade_and_matches_manual_compounding() -> None:
    base = _returns()
    sampled = base.to_numpy()[None, :, :]
    weights = {"fund_usd": 0.6, "fund_cad": 0.4}
    result = simulate_policy_paths(
        sampled,
        assets=tuple(base.columns),
        initial_weights=weights,
        policy=Policy("hold", weights, None, initial_rebalance=False),
    )
    manual = 100.0 * sum(
        weights[column] * np.prod(1.0 + base[column].to_numpy()) for column in base
    )
    assert result.ending_value[0] == pytest.approx(manual)
    assert result.traded_notional[0] == 0.0
    assert result.trading_cost[0] == 0.0


def test_summary_uses_simulation_start_and_usd_switch_does_not_charge_fx() -> None:
    assets = ("fund_usd", "cash_usd")
    sampled = np.full((1, 1, 2), -0.01)
    result = simulate_policy_paths(
        sampled,
        assets=assets,
        initial_weights={"fund_usd": 0.0, "cash_usd": 1.0},
        policy=Policy(
            "usd-switch",
            {"fund_usd": 1.0, "cash_usd": 0.0},
            rebalance_every_sessions=None,
        ),
        friction=Friction(
            default_trade_bps=10.0,
            usd_assets=frozenset(assets),
            cash_assets=frozenset({"cash_usd"}),
            fx_bps=100.0,
        ),
        initial_value=1_000.0,
    )
    assert result.fx_converted_notional[0] == pytest.approx(0.0)
    assert result.fx_cost[0] == pytest.approx(0.0)
    assert summarize_policy_paths(result)["scenario_fraction_below_start"] == 1.0


def test_results_require_identical_scenario_paths_for_paired_difference() -> None:
    assets = ("fund_usd", "cash_usd")
    policy = Policy("hold", {"fund_usd": 1.0, "cash_usd": 0.0}, None, False)
    first = simulate_policy_paths(
        np.zeros((2, 2, 2)),
        assets=assets,
        initial_weights=policy.target_weights,
        policy=policy,
    )
    second = simulate_policy_paths(
        np.zeros((2, 2, 2)),
        assets=assets,
        initial_weights=policy.target_weights,
        policy=policy,
    )
    np.testing.assert_array_equal(paired_ending_value_difference(first, second), 0.0)

    unpaired = simulate_policy_paths(
        np.full((2, 2, 2), 0.01),
        assets=assets,
        initial_weights=policy.target_weights,
        policy=policy,
    )
    with pytest.raises(ValueError, match="same paired"):
        paired_ending_value_difference(first, unpaired)


def test_friction_keys_and_sampling_indices_fail_closed() -> None:
    frame = _returns()
    with pytest.raises(ValueError, match="integers"):
        sample_joint_return_paths(frame, np.array([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="unknown assets"):
        simulate_policy_paths(
            np.zeros((1, 1, 2)),
            assets=tuple(frame.columns),
            initial_weights={"fund_usd": 0.5, "fund_cad": 0.5},
            policy=Policy("hold", {"fund_usd": 0.5, "fund_cad": 0.5}, None),
            friction=Friction(usd_assets=frozenset({"typo"})),
        )
    with pytest.raises(ValueError, match="unique"):
        simulate_policy_paths(
            np.zeros((1, 1, 2)),
            assets=("duplicate", "duplicate"),
            initial_weights={"duplicate": 1.0},
            policy=Policy("bad", {"duplicate": 1.0}, None),
        )


def test_pairing_fingerprint_includes_asset_order_and_large_values_reconcile() -> None:
    paths = np.array([[[0.1, -0.1]]])
    policy = Policy("single", {"a": 1.0, "b": 0.0}, None, False)
    first = simulate_policy_paths(
        paths,
        assets=("a", "b"),
        initial_weights=policy.target_weights,
        policy=policy,
        initial_value=72_832_611.77,
    )
    second = simulate_policy_paths(
        paths,
        assets=("b", "a"),
        initial_weights={"b": 0.0, "a": 1.0},
        policy=policy,
        initial_value=72_832_611.77,
    )
    with pytest.raises(ValueError, match="same paired"):
        paired_ending_value_difference(first, second)


def test_rolling_origins_covariance_and_joint_stress() -> None:
    returns = _returns()
    origins, paths = rolling_origin_return_paths(returns, horizon_sessions=3, step_sessions=2)
    assert len(origins) == 2
    assert paths.shape == (2, 3, 2)
    covariance = shrunk_covariance(returns)
    assert covariance.shape == (2, 2)
    np.testing.assert_allclose(covariance, covariance.T)
    stressed = deterministic_stress_value(
        initial_value=100.0,
        weights={"fund_usd": 0.75, "fund_cad": 0.25},
        asset_returns={"fund_usd": -0.20, "fund_cad": -0.10},
    )
    assert stressed == pytest.approx(82.5)


def test_multi_endpoint_replay_matches_standalone_accounting() -> None:
    assets = ("fund", "cash")
    rng = np.random.default_rng(23)
    paths = rng.normal(0.0002, 0.01, size=(12, 9, 2))
    policy = Policy(
        "fictional-quarterly",
        {"fund": 0.8, "cash": 0.2},
        rebalance_every_sessions=3,
    )
    friction = Friction(
        default_trade_bps=7.0,
        cash_assets=frozenset({"cash"}),
    )
    endpoints = simulate_policy_path_endpoints(
        paths,
        horizons=(2, 3, 5, 9),
        assets=assets,
        initial_weights={"fund": 0.3, "cash": 0.7},
        policy=policy,
        friction=friction,
    )
    for horizon, combined in endpoints.items():
        standalone = simulate_policy_paths(
            paths[:, :horizon],
            assets=assets,
            initial_weights={"fund": 0.3, "cash": 0.7},
            policy=policy,
            friction=friction,
        )
        np.testing.assert_allclose(combined.ending_value, standalone.ending_value)
        np.testing.assert_allclose(combined.maximum_drawdown, standalone.maximum_drawdown)
        np.testing.assert_allclose(combined.traded_notional, standalone.traded_notional)
        np.testing.assert_allclose(combined.trading_cost, standalone.trading_cost)


def test_quarterly_review_with_drift_band_does_not_force_a_trade() -> None:
    assets = ("fund", "cash")
    paths = np.zeros((2, 8, 2))
    paths[:, :, 0] = 0.001
    always = Policy("always", {"fund": 0.8, "cash": 0.2}, 2)
    banded = Policy(
        "banded",
        {"fund": 0.8, "cash": 0.2},
        2,
        rebalance_absolute_drift=0.05,
        rebalance_relative_drift=0.25,
    )
    friction = Friction(default_trade_bps=5.0, cash_assets=frozenset({"cash"}))
    always_result = simulate_policy_paths(
        paths,
        assets=assets,
        initial_weights={"fund": 0.8, "cash": 0.2},
        policy=always,
        friction=friction,
    )
    banded_result = simulate_policy_paths(
        paths,
        assets=assets,
        initial_weights={"fund": 0.8, "cash": 0.2},
        policy=banded,
        friction=friction,
    )
    assert np.all(banded_result.traded_notional == 0.0)
    assert np.all(always_result.traded_notional > 0.0)
    with pytest.raises(ValueError, match="review cadence"):
        Policy(
            "invalid",
            {"fund": 0.8, "cash": 0.2},
            None,
            rebalance_absolute_drift=0.05,
        )
