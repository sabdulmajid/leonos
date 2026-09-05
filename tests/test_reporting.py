from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import leonos.reporting as reporting
from leonos.evaluation import QlibReconciliation


def _predictions(labels: pd.DataFrame, model: str, score: np.ndarray) -> pd.DataFrame:
    return labels[["ticker", "origin"]].assign(
        model=model,
        seed=42,
        horizon=10,
        score=np.asarray(score, dtype=float),
        status="ok",
    )


def test_qlib_metrics_subtract_cost_once_and_use_final_cumulative_counters() -> None:
    initial = 1_000.0
    net = np.asarray([0.09, -0.025])
    account = initial * np.cumprod(1.0 + net)
    report = pd.DataFrame(
        {
            "account": account,
            "return": [0.10, -0.02],
            "cost": [0.01, 0.005],
            "total_turnover": [1_000.0, 1_500.0],
            "turnover": [1.0, 0.5],
            "total_cost": [10.0, 15.0],
            "cash": [100.0, 200.0],
            "value": account - [100.0, 200.0],
        },
        index=pd.to_datetime(["2025-01-03", "2025-01-06"]),
    )
    persisted, metrics = reporting._qlib_portfolio_metrics(
        report, initial_cash=initial, terminal_positions=5
    )

    np.testing.assert_allclose(persisted["net_return"], net)
    assert metrics["net_cumulative_return"] == pytest.approx(np.prod(1.0 + net) - 1.0)
    assert metrics["turnover_dollars"] == 1_500.0
    assert metrics["transaction_costs_dollars"] == 15.0
    assert metrics["turnover_rate_sum"] == 1.5
    assert metrics["account_reconciliation_max_abs_dollars"] < 1e-10


def test_position_diagnostics_measure_drift_and_reject_short_stock_value() -> None:
    dates = pd.to_datetime(["2025-01-03", "2025-01-06"])
    positions = pd.DataFrame(
        [
            {"datetime": dates[0], "ticker": "__CASH__", "amount": 50, "price": 1, "value": 50},
            {"datetime": dates[0], "ticker": "AAA", "amount": 6, "price": 100, "value": 600},
            {"datetime": dates[0], "ticker": "BBB", "amount": 7, "price": 50, "value": 350},
            {"datetime": dates[1], "ticker": "__CASH__", "amount": -1, "price": 1, "value": -1},
            {"datetime": dates[1], "ticker": "AAA", "amount": 7.01, "price": 100, "value": 701},
            {"datetime": dates[1], "ticker": "BBB", "amount": 6, "price": 50, "value": 300},
        ]
    )
    report = pd.DataFrame({"account": [1_000.0, 1_000.0]}, index=dates)

    diagnostics = reporting._position_diagnostics(positions, report)

    assert diagnostics["median_gross_invested_exposure"] == pytest.approx(0.9755)
    assert diagnostics["max_gross_invested_exposure"] == pytest.approx(1.001)
    assert diagnostics["median_largest_single_stock_weight"] == pytest.approx(0.6505)
    assert diagnostics["max_largest_single_stock_weight"] == pytest.approx(0.701)
    assert diagnostics["minimum_cash_dollars"] == -1.0
    assert diagnostics["minimum_cash_account_weight"] == pytest.approx(-0.001)

    invalid = positions.copy()
    invalid.loc[invalid["ticker"].eq("AAA"), "value"] = -1.0
    with pytest.raises(ValueError, match="long-only"):
        reporting._position_diagnostics(invalid)


def test_worked_example_pairs_a_completed_trade_and_includes_both_fees() -> None:
    orders = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2025-01-02", "2025-01-10"]),
            "execution_date": pd.to_datetime(["2025-01-03", "2025-01-13"]),
            "ticker": ["AAA", "AAA"],
            # Qlib's saved order metrics sign sell quantities negatively.
            "deal_amount": [10.0, -10.0],
            "trade_price": [100.0, 110.0],
            "trade_cost": [0.5, 0.55],
            "side": ["buy", "sell"],
        }
    )
    example = reporting._worked_example(orders)
    assert example is not None
    assert example["entry_signal_date"] == "2025-01-02T00:00:00"
    assert example["entry_execution_date"] == "2025-01-03T00:00:00"
    assert example["exit_execution_date"] == "2025-01-13T00:00:00"
    assert example["selected_stock"] == "AAA"
    assert example["gross_result_dollars"] == 100.0
    assert example["fees_dollars"] == pytest.approx(1.05)
    assert example["net_result_dollars"] == pytest.approx(98.95)


class _Position:
    def __init__(self, account: float) -> None:
        self.account = account

    def get_cash(self, include_settle: bool = False) -> float:
        del include_settle
        return self.account * 0.05

    def get_stock_list(self) -> list[str]:
        return ["AAA"]

    def get_stock_amount(self, ticker: str) -> float:
        assert ticker == "AAA"
        return 1.0

    def get_stock_price(self, ticker: str) -> float:
        assert ticker == "AAA"
        return self.account * 0.95


def test_saved_evaluation_uses_complete_artifacts_and_renders_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    origins = pd.to_datetime(["2025-01-02", "2025-01-03"])
    labels = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "origin": origin,
                "target": ticker_index / 100.0,
                "input_end": origin,
                "horizon_sessions": 10,
                "context_complete": True,
                "label_complete": True,
                "split": "test",
            }
            for origin in origins
            for ticker_index, ticker in enumerate(tickers)
        ]
    )
    sessions = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    bars = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "session": session,
                "open": 100.0 + index,
                "high": 102.0 + index,
                "low": 99.0 + index,
                "close": 101.0 + index,
                "volume": 1_000.0,
            }
            for session in sessions
            for index, ticker in enumerate(tickers)
        ]
    )
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    label_path = prepared / "labels.parquet"
    bars_path = prepared / "bars.parquet"
    calendar_path = prepared / "calendar.parquet"
    labels.to_parquet(label_path, index=False)
    bars.to_parquet(bars_path, index=False)
    pd.DataFrame({"session": sessions}).to_parquet(calendar_path, index=False)
    prepared_paths = {
        "labels_test": label_path,
        "accepted_bars": bars_path,
        "calendar": calendar_path,
    }
    prepare_marker = {
        "run_signature": "prepare-signature",
        "dataset_revision": "dataset-revision",
        "accepted_rows": len(bars),
        "calendar_sessions": len(sessions),
    }
    config = {
        "_meta": {"sha256": "config-hash"},
        "experiment": {"seed": 42, "sensitivity_seeds": [43, 44]},
        "paths": {
            "prepared_data": str(prepared),
            "artifacts": str(tmp_path / "artifacts"),
            "summaries": str(tmp_path / "summaries"),
        },
        "forecast": {"horizon_sessions": 10},
        "evaluation": {
            "minimum_daily_coverage": 3,
            "bootstrap": {
                "block_length": 20,
                "sensitivity_block_lengths": [10, 40],
                "replicates": 20,
                "seed": 42,
            },
        },
        "portfolio": {"initial_cash": 1_000_000.0, "target_exposure": 0.95},
        "sources": {"qlib": {"revision": reporting.QLIB_REVISION}},
    }
    kronos = _predictions(labels, "kronos", labels["target"].to_numpy())
    lightgbm = _predictions(labels, "lightgbm", -labels["target"].to_numpy())
    worker = {
        "started_at_utc": "2025-01-01T00:00:00Z",
        "finished_at_utc": "2025-01-01T00:00:10Z",
        "elapsed_seconds": 10.0,
        "gpu_memory": {
            "peak_allocated_bytes": 2**30,
            "peak_reserved_bytes": 2**31,
        },
        "device": "cuda:0",
    }
    monkeypatch.setattr(
        reporting,
        "load_complete_preparation",
        lambda _config: (prepared_paths, prepare_marker),
    )
    monkeypatch.setattr(
        reporting,
        "_load_lightgbm_artifact",
        lambda *_args: (
            lightgbm,
            {
                "timing_seconds": {"test_prediction": 0.0277, "total": 2.0},
                "search": {
                    "selected": {
                        "candidate_id": "l31_lr02",
                        "best_iteration": 1,
                        "validation_mean_daily_rankic": 0.065,
                    }
                },
            },
            tmp_path / "lightgbm.parquet",
        ),
    )
    monkeypatch.setattr(
        reporting,
        "_load_kronos_artifacts",
        lambda *_args: (kronos, [worker], [tmp_path / "kronos.parquet"]),
    )
    monkeypatch.setattr(reporting, "_ensure_qlib_provider", lambda *_args: tmp_path / "qlib")
    monkeypatch.setattr(
        reporting,
        "reconcile_daily_rankic_with_qlib",
        lambda *_args, **_kwargs: QlibReconciliation(True, True, 0.0, "matched"),
    )

    calls: list[tuple[int, int]] = []

    def fake_backtest(scores: pd.DataFrame, **kwargs: object) -> SimpleNamespace:
        spec = kwargs["spec"]
        assert isinstance(spec, reporting.QlibBacktestSpec)
        calls.append((len(scores), spec.cost_bps_per_side))
        cost = spec.proportional_cost
        gross = np.asarray([0.01, -0.002])
        net = gross - cost
        account = 1_000_000.0 * np.cumprod(1.0 + net)
        report = pd.DataFrame(
            {
                "account": account,
                "return": gross,
                "cost": cost,
                "total_turnover": [950_000.0, 1_050_000.0],
                "turnover": [0.95, 0.10],
                "total_cost": [950_000.0 * cost, 1_050_000.0 * cost],
                "cash": account * 0.05,
                "value": account * 0.95,
                "bench": np.nan,
            },
            index=sessions[1:],
        )
        return SimpleNamespace(
            report=report,
            positions={
                date: _Position(value) for date, value in zip(sessions[1:], account, strict=True)
            },
            trade_indicators=pd.DataFrame({"orders": [0, 0]}, index=sessions[1:]),
            order_history={},
        )

    monkeypatch.setattr(reporting, "run_qlib_topk_backtest", fake_backtest)
    monkeypatch.chdir(tmp_path)

    summary = reporting.evaluate_saved_predictions(config, seed=42)
    assert summary["status"] == "complete"
    assert summary["comparison"]["common_observations"] == len(labels)
    assert summary["portfolio"]["models"]["kronos"]["5"]["return_definition"].startswith(
        "Qlib return - Qlib cost"
    )
    assert sorted(calls) == sorted([(len(labels), cost) for cost in (0, 5, 15)] * 2)
    assert (tmp_path / "summaries/evaluation_seed_42.json").is_file()
    assert (tmp_path / "artifacts/evaluation/seed=42/aligned.parquet").is_file()

    aggregate = tmp_path / "summaries/evaluation_seed_42.json"
    saved = json.loads(aggregate.read_text(encoding="utf-8"))
    saved["portfolio"]["worked_trade"] = {
        "model": "kronos",
        "selected_stock": "AAA",
        "entry_signal_date": "2025-01-02T00:00:00",
        "entry_execution_date": "2025-01-03T00:00:00",
        "entry_price": 100.0,
        "exit_signal_date": "2025-01-10T00:00:00",
        "exit_execution_date": "2025-01-13T00:00:00",
        "exit_price": 110.0,
        "shares": 10.0,
        "gross_result_dollars": 100.0,
        "fees_dollars": 1.05,
        "net_result_dollars": 98.95,
    }
    saved["comparison"]["models"]["kronos"]["mean_daily_rankic"] = -0.003
    saved["comparison"]["models"]["lightgbm"]["mean_daily_rankic"] = 0.002
    saved["comparison"]["mean_daily_rankic_difference"] = -0.005
    saved["comparison"]["primary_confidence_interval"].update({"lower": -0.07, "upper": 0.05})
    saved["comparison"]["bootstrap_sensitivity"] = [
        {"block_length": 10, "estimate": -0.005, "lower": -0.06, "upper": 0.04},
        {"block_length": 20, "estimate": -0.005, "lower": -0.07, "upper": 0.05},
        {"block_length": 40, "estimate": -0.005, "lower": -0.08, "upper": 0.06},
    ]
    saved["portfolio"]["models"]["kronos"]["5"]["net_cumulative_return"] = 0.006
    saved["portfolio"]["models"]["lightgbm"]["5"]["net_cumulative_return"] = 0.007
    saved["calendar_year_metrics"] = [
        {
            "calendar_year": 2025,
            "dates": 250,
            "kronos_mean_rankic": 0.001,
            "lightgbm_mean_rankic": -0.004,
            "mean_rankic_difference": 0.005,
        },
        {
            "calendar_year": 2026,
            "dates": 159,
            "kronos_mean_rankic": -0.009,
            "lightgbm_mean_rankic": 0.013,
            "mean_rankic_difference": -0.022,
        },
    ]
    reporting.atomic_write_json(aggregate, saved)
    seed_settings = {
        43: (-0.004, 0.001, -0.005, -0.06, 0.04, 0.35, 1.26),
        44: (-0.002, 0.007, -0.009, -0.08, 0.047, 0.54, 0.515),
    }
    for sensitivity_seed, values in seed_settings.items():
        sensitivity = json.loads(json.dumps(saved))
        k_rankic, l_rankic, rankic_delta, ci_lower, ci_upper, k_net, l_net = values
        sensitivity["seed"] = sensitivity_seed
        sensitivity["comparison"]["prediction_seed"] = sensitivity_seed
        sensitivity["comparison"]["models"]["kronos"]["mean_daily_rankic"] = k_rankic
        sensitivity["comparison"]["models"]["lightgbm"]["mean_daily_rankic"] = l_rankic
        sensitivity["comparison"]["mean_daily_rankic_difference"] = rankic_delta
        sensitivity["comparison"]["primary_confidence_interval"].update(
            {"lower": ci_lower, "upper": ci_upper}
        )
        sensitivity["portfolio"]["models"]["kronos"]["5"]["net_cumulative_return"] = k_net
        sensitivity["portfolio"]["models"]["lightgbm"]["5"]["net_cumulative_return"] = l_net
        sensitivity["lightgbm_validation_selection"].update(
            {
                "candidate_id": "l31_subsample" if sensitivity_seed == 43 else "l31_regularized",
                "best_iteration": 1 if sensitivity_seed == 43 else 3,
            }
        )
        if sensitivity_seed == 43:
            sensitivity["portfolio"]["models"]["kronos"]["0"]["position_diagnostics"].update(
                {
                    "minimum_cash_dollars": -14.33,
                    "minimum_cash_account_weight": -1.52e-5,
                }
            )
            sensitivity["portfolio"]["models"]["kronos"]["5"]["position_diagnostics"].update(
                {
                    "minimum_cash_dollars": -13.03,
                    "minimum_cash_account_weight": -1.42e-5,
                }
            )
        reporting.atomic_write_json(
            tmp_path / f"summaries/evaluation_seed_{sensitivity_seed}.json",
            sensitivity,
        )

    report_path = reporting.render_results_report(config)
    text = report_path.read_text(encoding="utf-8")
    assert "RankIC difference" in text
    assert "95% CI" in text
    assert "split-adjusted price-return simulation" in text
    assert (
        "All three declared-seed RankIC differences are negative, but each paired 95% CI "
        "contains zero" in text
    )
    assert "calendar-year RankIC difference changes sign" in text
    assert (
        "portfolio winner is not seed-stable: LightGBM wins seeds 42 and 43; Kronos "
        "narrowly wins seed 44" in text
    )
    assert "| kronos | 0.80% | 0.60% | 0.50% |" in text
    assert "| lightgbm | 0.80% | 0.70% | 0.50% |" in text
    assert "| 10 | -0.0050 | -0.0600 | 0.0400 |" in text
    assert "| 40 | -0.0050 | -0.0800 | 0.0600 |" in text
    assert (
        "| 42 | -0.0030 | 0.0020 | -0.0050 | [-0.0700, 0.0500] | 0.60% | "
        "0.70% | l31_lr02@1 | LightGBM (+0.10 pp) |" in text
    )
    assert "| 44 | -0.0020 | 0.0070 | -0.0090 | [-0.0800, 0.0470]" in text
    assert "| 2025 | 250 | 0.0010 | -0.0040 | 0.0050 |" in text
    assert "| 2026 | 159 | -0.0090 | 0.0130 | -0.0220 |" in text
    assert "Forecast origins span 2025-01-02 through 2025-01-03" in text
    assert "Portfolio execution/valuation spans 2025-01-03 through 2025-01-06" in text
    assert "Primary-seed realized position drift at 5 bps" in text
    assert "Median gross exposure" in text
    assert "Maximum largest-stock weight" in text
    assert "unmodified Qlib strategy's realized weights" in text
    assert "seed 43 Kronos reached minimum cash -$13.03" in text
    assert "cash/account -1.42e-05" in text
    assert "worst declared-cost case was seed 43 Kronos at 0 bps: -$14.33" in text
    assert "whole-share rounding overdrafts" in text
    assert "strict no-leverage cannot be claimed" in text
    assert "l31_lr02@1" in text
    assert "l31_subsample@1" in text
    assert "l31_regularized@3" in text
    assert "reran the full development-fit, validation-selection" in text
    assert "CAGR" in text
    assert "Σ daily turnover rate" in text
    assert "0.028" in text
    assert "Peak GPU allocated (per-worker maximum)" in text
    assert "Peak GPU reserved (per-worker maximum)" in text
    assert "1.00 GiB" in text
    assert "2.00 GiB" in text
    assert "selected **AAA**" in text
    assert "2025-01-02 post-close signal" in text
    assert "next open on 2025-01-03" in text
    assert "gross result $100.00, fees $1.05, and net result $98.95" in text
    assert "not evidence of model skill" in text
    assert "not a forced final liquidation" in text
    assert "![Paired daily RankIC difference](figures/rankic-difference.png)" in text
    assert "![Compounded net wealth at five bps per side](figures/net-wealth.png)" in text


def test_complete_coverage_gate_rejects_a_failed_forecast() -> None:
    labels = pd.DataFrame({"ticker": ["AAA", "BBB"], "origin": pd.to_datetime(["2025-01-02"] * 2)})
    prediction = labels.assign(score=[0.1, np.nan], status=["ok", "nonfinite"])
    with pytest.raises(ValueError, match="failed or non-finite"):
        reporting._require_complete_prediction_keys(labels, prediction, "kronos")


def test_runtime_summary_uses_first_start_and_cumulative_resume_time() -> None:
    workers = [
        {
            "started_at_utc": "2025-01-01T00:05:00Z",
            "first_started_at_utc": "2025-01-01T00:00:00Z",
            "finished_at_utc": "2025-01-01T00:10:00Z",
            "elapsed_seconds": 7.0,
            "cumulative_elapsed_seconds": 7.0,
            "attempt_count": 2,
            "gpu_memory": {},
            "device": "cuda:0",
        },
        {
            "started_at_utc": "2025-01-01T00:02:00Z",
            "first_started_at_utc": "2025-01-01T00:02:00Z",
            "finished_at_utc": "2025-01-01T00:08:00Z",
            "elapsed_seconds": 4.0,
            "cumulative_elapsed_seconds": 4.0,
            "attempt_count": 1,
            "gpu_memory": {},
            "device": "cuda:1",
        },
    ]

    runtime = reporting._runtime_summary({"timing_seconds": {}}, workers)["kronos"]

    assert runtime["test_wall_seconds"] == 600.0
    assert runtime["sum_worker_elapsed_seconds"] == 11.0
    assert runtime["total_worker_attempts"] == 3
