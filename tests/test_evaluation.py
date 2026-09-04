from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leonos.evaluation import (
    align_model_predictions,
    compare_predictions,
    daily_cross_sectional_metrics,
    daily_rankic,
    moving_block_bootstrap_mean,
    reconcile_daily_rankic_with_qlib,
    spearman_average_rank,
)


def _labels(days: int = 3, tickers: int = 6) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=days)
    return pd.DataFrame(
        [
            {
                "ticker": f"S{ticker}",
                "origin": date,
                "target": (ticker - 2.5) / 100.0 + day / 1000.0,
            }
            for day, date in enumerate(dates)
            for ticker in range(tickers)
        ]
    )


def _prediction(labels: pd.DataFrame, name: str, scores: np.ndarray) -> pd.DataFrame:
    values = np.asarray(scores, dtype=float)
    return labels[["ticker", "origin"]].assign(
        model=name,
        seed=42,
        horizon=10,
        score=values,
        status=np.where(np.isfinite(values), "ok", "nonfinite"),
    )


def test_spearman_uses_average_ranks_and_constant_is_undefined() -> None:
    assert spearman_average_rank([1.0, 1.0, 2.0], [1.0, 2.0, 3.0]) == pytest.approx(
        np.sqrt(3.0) / 2.0
    )
    assert np.isnan(spearman_average_rank([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]))


def test_daily_rankic_is_cross_sectional_not_along_forecast_path() -> None:
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    panel = pd.DataFrame(
        {
            "origin": np.repeat(dates, 3),
            "ticker": ["A", "B", "C"] * 2,
            "target": [1, 2, 3, 1, 2, 3],
            "score": [1, 2, 3, 3, 2, 1],
        }
    )
    result = daily_rankic(panel, minimum_coverage=3)
    assert result.loc[dates[0]] == pytest.approx(1.0)
    assert result.loc[dates[1]] == pytest.approx(-1.0)
    assert result.mean() == pytest.approx(0.0)


def test_alignment_preserves_prediction_failures_and_common_coverage() -> None:
    labels = _labels(days=1, tickers=4)
    kronos = _prediction(labels, "kronos", labels["target"].to_numpy())
    kronos.loc[0, "score"] = np.nan
    kronos.loc[0, "status"] = "nonfinite"
    lightgbm = _prediction(labels, "lightgbm", labels["target"].to_numpy())
    aligned = align_model_predictions(
        labels,
        {"kronos": kronos, "lightgbm": lightgbm},
        expected_seed=42,
    )
    assert len(aligned) == len(labels)
    assert aligned["kronos_score"].isna().sum() == 1
    daily = daily_cross_sectional_metrics(
        aligned, ["kronos", "lightgbm"], minimum_coverage=3
    )
    assert daily.loc[0, "eligible_count"] == 4
    assert daily.loc[0, "kronos_available"] == 3
    assert daily.loc[0, "lightgbm_available"] == 4
    assert daily.loc[0, "common_count"] == 3
    assert daily.loc[0, "common_coverage"] == pytest.approx(0.75)


def test_oracle_shuffled_and_zero_score_sanity_references() -> None:
    labels = _labels(days=4, tickers=6)
    oracle = _prediction(labels, "kronos", labels["target"].to_numpy())
    reversed_scores = labels.groupby("origin", sort=False)["target"].transform(
        lambda values: values.iloc[::-1].to_numpy()
    )
    shuffled = _prediction(labels, "lightgbm", reversed_scores.to_numpy())

    result = compare_predictions(
        labels,
        {"kronos": oracle, "lightgbm": shuffled},
        expected_seed=42,
        minimum_coverage=5,
        bootstrap_replicates=100,
    )
    assert result.summary["models"]["kronos"]["mean_daily_rankic"] == pytest.approx(1.0)
    assert result.summary["models"]["lightgbm"]["mean_daily_rankic"] == pytest.approx(-1.0)
    assert result.summary["mean_daily_rankic_difference"] == pytest.approx(2.0)
    assert result.summary["zero_score"]["mean_daily_rankic"] is None
    assert np.allclose(result.bootstrap["lower"], 2.0)
    assert np.allclose(result.bootstrap["upper"], 2.0)

    constant = labels.assign(score=0.0)
    assert daily_rankic(constant, minimum_coverage=5).isna().all()


def test_mae_is_averaged_by_date_not_weighted_by_number_of_tickers() -> None:
    date_one, date_two = pd.to_datetime(["2025-01-02", "2025-01-03"])
    rows = [
        {"ticker": "A", "origin": date_one, "target": 0.0},
        {"ticker": "B", "origin": date_one, "target": 1.0},
        *[
            {"ticker": ticker, "origin": date_two, "target": float(number)}
            for number, ticker in enumerate(["A", "B", "C", "D"])
        ],
    ]
    labels = pd.DataFrame(rows)
    # Errors are exactly 1 on date one and 3 on date two.  A row-weighted result
    # would differ from the required equal-date mean of 2.
    first_scores = labels["target"] + np.where(labels["origin"] == date_one, 1.0, 3.0)
    second_scores = labels["target"] + np.where(labels["origin"] == date_one, 2.0, 4.0)
    first = _prediction(labels, "kronos", first_scores.to_numpy())
    second = _prediction(labels, "lightgbm", second_scores.to_numpy())
    result = compare_predictions(
        labels,
        {"kronos": first, "lightgbm": second},
        expected_seed=42,
        minimum_coverage=2,
        bootstrap_replicates=20,
    )
    assert result.summary["models"]["kronos"]["mean_daily_mae"] == pytest.approx(2.0)
    assert result.summary["models"]["lightgbm"]["mean_daily_mae"] == pytest.approx(3.0)


def test_moving_block_bootstrap_is_deterministic_and_keeps_date_statistic() -> None:
    values = np.linspace(-0.2, 0.3, 25)
    first, samples_one = moving_block_bootstrap_mean(
        values, block_length=5, replicates=100, seed=7
    )
    second, samples_two = moving_block_bootstrap_mean(
        values, block_length=5, replicates=100, seed=7
    )
    np.testing.assert_array_equal(samples_one, samples_two)
    assert first == second
    assert first.estimate == pytest.approx(values.mean())
    assert first.observations == 25


def test_duplicate_prediction_keys_fail_closed() -> None:
    labels = _labels(days=1, tickers=5)
    predictions = _prediction(labels, "kronos", labels["target"].to_numpy())
    predictions = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate prediction keys"):
        align_model_predictions(labels, {"kronos": predictions}, expected_seed=42)


@pytest.mark.parametrize("missing_column", ["model", "seed", "horizon", "status"])
def test_prediction_provenance_columns_are_mandatory(missing_column: str) -> None:
    labels = _labels(days=1, tickers=5)
    predictions = _prediction(labels, "kronos", labels["target"].to_numpy()).drop(
        columns=missing_column
    )
    with pytest.raises(ValueError, match="missing provenance columns"):
        align_model_predictions(labels, {"kronos": predictions}, expected_seed=42)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("model", "lightgbm", "model provenance is mixed or wrong"),
        ("seed", 43, "seed provenance is mixed or wrong"),
        ("horizon", 9, "horizon provenance is mixed or wrong"),
    ],
)
def test_prediction_provenance_rejects_wrong_or_mixed_values(
    column: str, value: object, message: str
) -> None:
    labels = _labels(days=1, tickers=5)
    predictions = _prediction(labels, "kronos", labels["target"].to_numpy())
    predictions.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        align_model_predictions(labels, {"kronos": predictions}, expected_seed=42)


def test_prediction_status_must_exactly_match_score_finiteness() -> None:
    labels = _labels(days=1, tickers=5)
    predictions = _prediction(labels, "kronos", labels["target"].to_numpy())
    predictions.loc[0, "score"] = np.nan
    with pytest.raises(ValueError, match="inconsistent with score finiteness"):
        align_model_predictions(labels, {"kronos": predictions}, expected_seed=42)

    predictions.loc[0, "status"] = "nonfinite"
    aligned = align_model_predictions(
        labels, {"kronos": predictions}, expected_seed=42, horizon=10
    )
    assert np.isnan(aligned.loc[0, "kronos_score"])


def test_compare_records_enforced_seed_and_horizon() -> None:
    labels = _labels(days=2, tickers=5)
    kronos = _prediction(labels, "kronos", labels["target"].to_numpy())
    lightgbm = _prediction(labels, "lightgbm", labels["target"].to_numpy())
    result = compare_predictions(
        labels,
        {"kronos": kronos, "lightgbm": lightgbm},
        expected_seed=42,
        bootstrap_replicates=10,
    )
    assert result.summary["prediction_seed"] == 42
    assert result.summary["prediction_horizon_sessions"] == 10


def test_independent_fixture_reconciles_with_qlib_when_available() -> None:
    labels = _labels(days=3, tickers=6)
    panel = labels.assign(score=labels["target"] * 2.0 + 0.1)
    result = reconcile_daily_rankic_with_qlib(panel, minimum_coverage=5)
    if result.available:
        assert result.matched, result.detail
        assert result.maximum_absolute_difference == pytest.approx(0.0, abs=1e-12)
    else:
        assert result.matched is None
