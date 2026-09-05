"""Saved-artifact evaluation, portfolio evidence, and concise result reporting.

This module never invokes a forecaster.  It fails closed unless both models cover
the complete predeclared test label set with finite, correctly-provenanced scores.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import atomic_write_json, git_state, runtime_environment, stable_hash
from .evaluation import compare_predictions, reconcile_daily_rankic_with_qlib
from .kronos_runner import (
    collapse_kronos_scores,
    forecast_key_set_metadata,
    frozen_kronos_execution_plan,
)
from .pipeline import baseline_stage_config_hash, load_complete_preparation
from .portfolio import (
    assert_account_reconciles,
    portfolio_metrics,
    simulate_equal_weight_buy_hold,
)
from .qlib_adapter import (
    QLIB_REVISION,
    QlibBacktestOutputs,
    QlibBacktestSpec,
    order_indicator_metric_series,
    run_qlib_topk_backtest,
    write_qlib_day_dataset,
)

EVALUATION_SCHEMA = "leonos.saved_evaluation.v1"
MODEL_NAMES = ("kronos", "lightgbm")
DECLARED_COSTS_BPS = (0, 5, 15)


def _evaluation_implementation_hash() -> str:
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for relative in (
        "reporting.py",
        "evaluation.py",
        "portfolio.py",
        "qlib_adapter.py",
    ):
        digest.update(relative.encode())
        digest.update((package / relative).read_bytes())
    return digest.hexdigest()


def _path(config: Mapping[str, Any], key: str) -> Path:
    try:
        return Path(str(config["paths"][key]))
    except (KeyError, TypeError) as exc:
        raise ValueError(f"configuration is missing paths.{key}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _atomic_write_parquet(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".parquet", dir=destination.parent
    )
    os.close(fd)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination


def _atomic_write_text(text: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination


def _finite_json(value: Any) -> Any:
    """Replace non-finite scalars with JSON ``null`` and normalize NumPy values."""

    if isinstance(value, Mapping):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    return value


def _validate_test_labels(labels: pd.DataFrame, horizon: int) -> pd.DataFrame:
    required = {"ticker", "origin", "target"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"test labels are missing columns: {sorted(missing)}")
    clean = labels.copy()
    clean["ticker"] = clean["ticker"].astype("string")
    clean["origin"] = pd.to_datetime(clean["origin"], utc=True).dt.tz_convert(None).dt.normalize()
    if clean.empty or clean.duplicated(["ticker", "origin"]).any():
        raise ValueError("test labels must be non-empty with unique ticker/origin keys")
    if not np.isfinite(pd.to_numeric(clean["target"], errors="coerce")).all():
        raise ValueError("every predeclared test label must be finite")
    for flag in ("context_complete", "label_complete"):
        if flag in clean and not clean[flag].astype(bool).all():
            raise ValueError(f"test labels contain rows with {flag}=false")
    if "horizon_sessions" in clean and not clean["horizon_sessions"].eq(horizon).all():
        raise ValueError("test labels do not use the configured horizon")
    if "input_end" in clean:
        input_end = pd.to_datetime(clean["input_end"], utc=True).dt.tz_convert(None).dt.normalize()
        if not input_end.eq(clean["origin"]).all():
            raise ValueError("test label input_end does not equal origin")
    if "split" in clean and not clean["split"].astype(str).eq("test").all():
        raise ValueError("non-test rows were found in the test label artifact")
    return clean.sort_values(["origin", "ticker"], kind="stable").reset_index(drop=True)


def _require_complete_prediction_keys(
    labels: pd.DataFrame, predictions: pd.DataFrame, model: str
) -> None:
    if not {"ticker", "origin", "score", "status"}.issubset(predictions.columns):
        raise ValueError(f"{model} prediction artifact is incomplete")
    keys = predictions.loc[:, ["ticker", "origin"]].copy()
    keys["ticker"] = keys["ticker"].astype("string")
    keys["origin"] = pd.to_datetime(keys["origin"], utc=True).dt.tz_convert(None).dt.normalize()
    expected = pd.MultiIndex.from_frame(labels[["ticker", "origin"]])
    observed = pd.MultiIndex.from_frame(keys)
    if observed.has_duplicates or len(observed) != len(expected) or set(observed) != set(expected):
        missing = len(set(expected).difference(observed))
        extra = len(set(observed).difference(expected))
        raise ValueError(
            f"{model} must cover every test key exactly once: missing={missing}, extra={extra}"
        )
    scores = pd.to_numeric(predictions["score"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(scores).all() or not predictions["status"].astype(str).eq("ok").all():
        raise ValueError(f"{model} has failed or non-finite test forecasts; evaluation aborted")


def _load_kronos_artifacts(
    config: Mapping[str, Any],
    seed: int,
    prepare_marker: Mapping[str, Any],
    expected_keys: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[Path]]:
    root = _path(config, "artifacts") / "predictions" / "kronos" / "test" / f"seed={seed}"
    expected = forecast_key_set_metadata(expected_keys)
    expected_rows = int(expected["count"])
    frozen_batch_size, frozen_num_shards = frozen_kronos_execution_plan(config)
    run_plan_path = root / "run-plan.json"
    if not run_plan_path.is_file():
        raise FileNotFoundError(f"immutable Kronos run plan is required: {run_plan_path}")
    run_plan = _read_json(run_plan_path)
    payload = run_plan.get("plan")
    if not isinstance(payload, dict) or run_plan.get("run_signature") != stable_hash(payload):
        raise ValueError("Kronos root run-plan signature is invalid")
    config_hash = config.get("_meta", {}).get("sha256") or stable_hash(config)
    required_plan = {
        "split": "test",
        "run_name": "test",
        "seed": int(seed),
        "batch_size": frozen_batch_size,
        "num_shards": frozen_num_shards,
        "limit": None,
        "eligible_key_count": expected_rows,
        "eligible_key_sha256": expected["sha256"],
        "selected_key_count": expected_rows,
        "selected_key_sha256": expected["sha256"],
        "config_hash": config_hash,
        "prepare_signature": str(prepare_marker["run_signature"]),
        "dataset_revision": str(prepare_marker["dataset_revision"]),
    }
    wrong_plan = [name for name, value in required_plan.items() if payload.get(name) != value]
    if wrong_plan:
        raise ValueError(f"Kronos root run-plan provenance mismatch: {sorted(wrong_plan)}")
    shard_paths = sorted(root.glob("*.parquet"))
    manifest_paths = sorted(root.glob("worker-*-of-*.json"))
    if not shard_paths or not manifest_paths:
        raise FileNotFoundError(
            f"complete Kronos test shards and worker manifests are required: {root}"
        )
    manifests = [_read_json(path) for path in manifest_paths]
    num_shards = {int(item.get("num_shards", -1)) for item in manifests}
    indices = {int(item.get("shard_index", -1)) for item in manifests}
    if len(num_shards) != 1 or indices != set(range(next(iter(num_shards)))):
        raise ValueError("Kronos worker manifests do not describe one complete execution plan")
    for item in manifests:
        required = {
            "status": "complete",
            "split": "test",
            "run_name": "test",
            "seed": int(seed),
            "prepare_signature": str(prepare_marker["run_signature"]),
            "eligible_origins": int(expected_rows),
            "batch_size": frozen_batch_size,
            "num_shards": frozen_num_shards,
            "run_plan_signature": str(run_plan["run_signature"]),
            "run_plan_path": str(run_plan_path),
        }
        wrong = [name for name, expected in required.items() if item.get(name) != expected]
        if item.get("config_hash") != config_hash:
            wrong.append("config_hash")
        if wrong:
            raise ValueError(f"Kronos worker manifest provenance mismatch: {sorted(set(wrong))}")
        if int(item.get("completed_origins", -1)) != int(item.get("assigned_origins", -2)):
            raise ValueError("Kronos worker manifest is incomplete")
    if sum(int(item["assigned_origins"]) for item in manifests) != expected_rows:
        raise ValueError("Kronos worker assignment does not cover all eligible origins")
    listed = {Path(str(path)).resolve() for item in manifests for path in item.get("artifacts", [])}
    observed = {path.resolve() for path in shard_paths}
    if listed != observed:
        raise ValueError("Kronos worker manifests and saved shard set disagree")
    return collapse_kronos_scores(shard_paths), manifests, shard_paths


def _load_lightgbm_artifact(
    config: Mapping[str, Any], seed: int, prepare_marker: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    artifact_root = _path(config, "artifacts")
    prediction_path = (
        artifact_root / "predictions" / "lightgbm" / "test" / f"seed={seed}" / "predictions.parquet"
    )
    run_path = artifact_root / "models" / "lightgbm" / f"seed={seed}" / "run.json"
    if not prediction_path.is_file() or not run_path.is_file():
        raise FileNotFoundError(
            "complete saved LightGBM test predictions and run metadata are required"
        )
    metadata = _read_json(run_path)
    required = {
        "status": "complete",
        "seed": int(seed),
        "prepare_signature": str(prepare_marker["run_signature"]),
    }
    wrong = [name for name, expected in required.items() if metadata.get(name) != expected]
    if metadata.get("stage_config_hash") != baseline_stage_config_hash(config):
        wrong.append("stage_config_hash")
    if wrong:
        raise ValueError(f"LightGBM run provenance mismatch: {sorted(set(wrong))}")
    return pd.read_parquet(prediction_path), metadata, prediction_path


def _ensure_qlib_provider(
    config: Mapping[str, Any], prepared: Mapping[str, Path], prepare_marker: Mapping[str, Any]
) -> Path:
    signature = str(prepare_marker["run_signature"])
    provider = _path(config, "prepared_data") / "qlib" / signature
    marker_path = provider / "leonos_provider.json"
    expected = {
        "schema_version": "leonos.qlib_provider.v1",
        "prepare_signature": signature,
        "qlib_revision": QLIB_REVISION,
        "source_rows": int(prepare_marker["accepted_rows"]),
        "calendar_count": int(prepare_marker["calendar_sessions"]),
    }
    if provider.exists():
        if not marker_path.is_file() or _read_json(marker_path) != expected:
            raise ValueError("existing Qlib provider is not keyed to the current preparation")
        upstream = _read_json(provider / "leonos_qlib_manifest.json")
        if (
            upstream.get("qlib_revision") != QLIB_REVISION
            or int(upstream.get("source_rows", -1)) != expected["source_rows"]
            or int(upstream.get("calendar_count", -1)) != expected["calendar_count"]
        ):
            raise ValueError("existing Qlib provider manifest failed provenance checks")
        return provider
    bars = pd.read_parquet(prepared["accepted_bars"])
    calendar = pd.read_parquet(prepared["calendar"])["session"]
    write_qlib_day_dataset(
        bars,
        provider,
        calendar=calendar,
        price_basis="source split-adjusted OHLCV; price returns exclude dividends",
    )
    atomic_write_json(marker_path, expected)
    return provider


def _flatten_orders(order_history: Mapping[Any, Any], calendar: pd.DatetimeIndex) -> pd.DataFrame:
    metrics = (
        "amount",
        "deal_amount",
        "trade_price",
        "trade_value",
        "trade_cost",
        "trade_dir",
        "ffr",
    )
    previous = {calendar[index]: calendar[index - 1] for index in range(1, len(calendar))}
    frames: list[pd.DataFrame] = []
    ordered_history = sorted(order_history.items(), key=lambda item: pd.Timestamp(item[0]))
    for raw_date, indicator in ordered_history:
        execution = pd.Timestamp(raw_date).tz_localize(None).normalize()
        columns: list[pd.Series] = []
        for metric in metrics:
            series = order_indicator_metric_series(indicator, metric)
            if len(series):
                columns.append(series.rename(metric))
        if not columns:
            continue
        frame = pd.concat(columns, axis=1).rename_axis("ticker").reset_index()
        frame.insert(0, "signal_date", previous.get(execution, pd.NaT))
        frame.insert(1, "execution_date", execution)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["signal_date", "execution_date", "ticker", *metrics])
    result = pd.concat(frames, ignore_index=True)
    result["ticker"] = result["ticker"].astype(str)
    result["side"] = result["trade_dir"].map({0.0: "sell", 1.0: "buy"})
    return result.sort_values(["execution_date", "ticker"], kind="stable").reset_index(drop=True)


def _flatten_positions(positions: Mapping[Any, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw_date, position in sorted(positions.items(), key=lambda item: pd.Timestamp(item[0])):
        date = pd.Timestamp(raw_date).tz_localize(None).normalize()
        cash = float(position.get_cash(include_settle=True))
        rows.append(
            {
                "datetime": date,
                "ticker": "__CASH__",
                "amount": cash,
                "price": 1.0,
                "value": cash,
            }
        )
        for ticker in sorted(position.get_stock_list()):
            amount = float(position.get_stock_amount(ticker))
            price = float(position.get_stock_price(ticker))
            rows.append(
                {
                    "datetime": date,
                    "ticker": str(ticker),
                    "amount": amount,
                    "price": price,
                    "value": amount * price,
                }
            )
    return pd.DataFrame(rows, columns=["datetime", "ticker", "amount", "price", "value"])


def _qlib_portfolio_metrics(
    report: pd.DataFrame, *, initial_cash: float, terminal_positions: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "account",
        "return",
        "cost",
        "total_turnover",
        "turnover",
        "total_cost",
        "cash",
        "value",
    }
    missing = required.difference(report.columns)
    if report.empty or missing:
        raise ValueError(f"Qlib report is empty or missing columns: {sorted(missing)}")
    clean = report.sort_index().copy()
    numeric = clean[list(required)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Qlib report contains non-finite accounting values")
    # Pinned Qlib reports return gross of fees and cost as the matching fee rate.
    clean["net_return"] = numeric["return"] - numeric["cost"]
    clean["compounded_wealth"] = initial_cash * (1.0 + clean["net_return"]).cumprod()
    difference = (clean["compounded_wealth"] - numeric["account"]).abs()
    tolerance = max(1e-6, initial_cash * 1e-8)
    if float(difference.max()) > tolerance:
        raise AssertionError(
            "Qlib account does not reconcile to one compounding of report.return - report.cost"
        )
    starting = pd.Series([initial_cash], dtype=float)
    wealth_for_drawdown = pd.concat(
        [starting, clean["compounded_wealth"].reset_index(drop=True)], ignore_index=True
    )
    drawdown = wealth_for_drawdown / wealth_for_drawdown.cummax() - 1.0
    returns = clean["net_return"].to_numpy(dtype=float)
    standard_deviation = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sharpe = (
        float(np.sqrt(252.0) * np.mean(returns) / standard_deviation)
        if standard_deviation > 0
        else np.nan
    )
    metrics = {
        "net_cumulative_return": float(clean["compounded_wealth"].iloc[-1] / initial_cash - 1.0),
        "cagr_252_session": float(
            (clean["compounded_wealth"].iloc[-1] / initial_cash) ** (252.0 / len(clean)) - 1.0
        ),
        "net_sharpe_zero_cash": sharpe,
        "max_drawdown": float(drawdown.min()),
        "turnover_rate_sum": float(numeric["turnover"].sum()),
        # Qlib's total_* fields are cumulative account counters, not daily flows.
        "turnover_dollars": float(numeric["total_turnover"].iloc[-1]),
        "transaction_costs_dollars": float(numeric["total_cost"].iloc[-1]),
        "ending_value": float(numeric["account"].iloc[-1]),
        "ending_cash": float(numeric["cash"].iloc[-1]),
        "ending_market_value": float(numeric["value"].iloc[-1]),
        "unrealized_position_count": int(terminal_positions),
        "sessions": int(len(clean)),
        "account_reconciliation_max_abs_dollars": float(difference.max()),
        "cash_return_convention": "zero",
        "return_definition": "Qlib return - Qlib cost, compounded once",
        "forced_final_liquidation": False,
    }
    persisted = clean.rename_axis("datetime").reset_index()
    return persisted, metrics


def _worked_example(orders: pd.DataFrame) -> dict[str, Any] | None:
    if orders.empty:
        return None
    ordered = orders.sort_values(["execution_date", "ticker"], kind="stable")
    buys: dict[str, dict[str, Any]] = {}
    for row in ordered.to_dict("records"):
        ticker = str(row["ticker"])
        # Pinned Qlib stores sell deal_amount/trade_value with a negative sign,
        # even though trade_dir separately identifies the side.  Share matching
        # uses magnitude; preserving the sign here previously skipped every exit.
        amount = abs(float(row.get("deal_amount", 0.0)))
        price = float(row.get("trade_price", np.nan))
        if amount <= 0 or not np.isfinite(price):
            continue
        if row.get("side") == "buy" and ticker not in buys:
            buys[ticker] = row
        elif row.get("side") == "sell" and ticker in buys:
            entry = buys[ticker]
            entry_amount = abs(float(entry["deal_amount"]))
            shares = min(entry_amount, amount)
            if shares <= 0:
                continue
            entry_fee = float(entry.get("trade_cost", 0.0)) * shares / entry_amount
            exit_fee = float(row.get("trade_cost", 0.0)) * shares / amount
            gross = shares * (price - float(entry["trade_price"]))
            return _finite_json(
                {
                    "ticker": ticker,
                    "selected_stock": ticker,
                    "entry_signal_date": entry.get("signal_date"),
                    "entry_execution_date": entry["execution_date"],
                    "entry_timing": "signal after close; fill at next-session open",
                    "entry_price": float(entry["trade_price"]),
                    "exit_signal_date": row.get("signal_date"),
                    "exit_execution_date": row["execution_date"],
                    "exit_price": price,
                    "shares": shares,
                    "gross_result_dollars": gross,
                    "fees_dollars": entry_fee + exit_fee,
                    "net_result_dollars": gross - entry_fee - exit_fee,
                    "note": "deterministic accounting example; not evidence of model skill",
                }
            )
    return None


def _runtime_summary(
    lightgbm_metadata: Mapping[str, Any], kronos_manifests: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    timing = lightgbm_metadata.get("timing_seconds", {})
    starts = [
        pd.Timestamp(item.get("first_started_at_utc", item["started_at_utc"]))
        for item in kronos_manifests
    ]
    finishes = [pd.Timestamp(item["finished_at_utc"]) for item in kronos_manifests]
    allocated = [
        item.get("gpu_memory", {}).get("peak_allocated_bytes") for item in kronos_manifests
    ]
    reserved = [item.get("gpu_memory", {}).get("peak_reserved_bytes") for item in kronos_manifests]
    allocated_finite = [int(value) for value in allocated if value is not None]
    reserved_finite = [int(value) for value in reserved if value is not None]
    return {
        "lightgbm": {
            "test_inference_seconds": timing.get("test_prediction"),
            "total_fit_and_inference_seconds": timing.get("total"),
            "peak_memory_bytes": None,
            "device": "CPU",
        },
        "kronos": {
            "test_wall_seconds": float((max(finishes) - min(starts)).total_seconds()),
            "sum_worker_elapsed_seconds": float(
                sum(
                    float(item.get("cumulative_elapsed_seconds", item["elapsed_seconds"]))
                    for item in kronos_manifests
                )
            ),
            "total_worker_attempts": sum(
                int(item.get("attempt_count", 1)) for item in kronos_manifests
            ),
            "peak_allocated_bytes_per_worker_max": max(allocated_finite)
            if allocated_finite
            else None,
            "peak_reserved_bytes_per_worker_max": max(reserved_finite) if reserved_finite else None,
            "worker_count": len(kronos_manifests),
            "devices": sorted({str(item["device"]) for item in kronos_manifests}),
        },
    }


def evaluate_saved_predictions(config: Mapping[str, Any], seed: int = 42) -> dict[str, Any]:
    """Evaluate only complete, immutable test predictions and persist all evidence."""

    prepared, prepare_marker = load_complete_preparation(config)
    horizon = int(config["forecast"]["horizon_sessions"])
    labels = _validate_test_labels(pd.read_parquet(prepared["labels_test"]), horizon)
    lightgbm, lightgbm_metadata, lightgbm_path = _load_lightgbm_artifact(
        config, seed, prepare_marker
    )
    kronos, worker_manifests, kronos_paths = _load_kronos_artifacts(
        config, seed, prepare_marker, labels[["ticker", "origin"]]
    )
    _require_complete_prediction_keys(labels, kronos, "kronos")
    _require_complete_prediction_keys(labels, lightgbm, "lightgbm")

    evaluation_config = config.get("evaluation", {})
    bootstrap_config = evaluation_config.get("bootstrap", {})
    primary_block = int(bootstrap_config.get("block_length", 20))
    sensitivity = [
        int(value) for value in bootstrap_config.get("sensitivity_block_lengths", [10, 40])
    ]
    comparison = compare_predictions(
        labels,
        {"kronos": kronos, "lightgbm": lightgbm},
        expected_seed=int(seed),
        horizon=horizon,
        minimum_coverage=int(evaluation_config.get("minimum_daily_coverage", 3)),
        block_lengths=(primary_block, *sensitivity),
        bootstrap_replicates=int(bootstrap_config.get("replicates", 2_000)),
        bootstrap_seed=int(bootstrap_config.get("seed", 42)),
    )
    if comparison.summary["common_observations"] != len(labels):
        raise AssertionError("strict complete coverage was lost during model alignment")
    period_metrics = comparison.daily.copy()
    period_metrics["calendar_year"] = pd.to_datetime(period_metrics["origin"]).dt.year
    period_metrics = (
        period_metrics.groupby("calendar_year", sort=True, observed=True)
        .agg(
            dates=("origin", "size"),
            kronos_mean_rankic=("kronos_rankic", "mean"),
            lightgbm_mean_rankic=("lightgbm_rankic", "mean"),
            mean_rankic_difference=("delta_rankic", "mean"),
            kronos_mean_mae=("kronos_mae", "mean"),
            lightgbm_mean_mae=("lightgbm_mae", "mean"),
        )
        .reset_index()
    )

    reconciliation: dict[str, Any] = {}
    for model in MODEL_NAMES:
        panel = comparison.aligned[["ticker", "origin", "target", f"{model}_score"]].rename(
            columns={f"{model}_score": "score"}
        )
        result = reconcile_daily_rankic_with_qlib(
            panel,
            minimum_coverage=int(evaluation_config.get("minimum_daily_coverage", 3)),
        )
        reconciliation[model] = _finite_json(vars(result))
        if result.available and not result.matched:
            raise AssertionError(f"{model} RankIC does not reconcile with pinned Qlib")

    provider = _ensure_qlib_provider(config, prepared, prepare_marker)
    bars = pd.read_parquet(prepared["accepted_bars"])
    calendar = pd.DatetimeIndex(pd.read_parquet(prepared["calendar"])["session"])
    origins = pd.DatetimeIndex(comparison.aligned["origin"].unique()).sort_values()
    origin_locations = calendar.get_indexer([origins.min(), origins.max()])
    if (origin_locations < 0).any() or origin_locations[1] + 1 >= len(calendar):
        raise ValueError("test origins cannot be mapped to next-session execution bounds")
    start_time = calendar[origin_locations[0] + 1]
    end_time = calendar[origin_locations[1] + 1]

    portfolio_config = config.get("portfolio", {})
    initial_cash = float(portfolio_config.get("initial_cash", 1_000_000.0))
    evaluation_root = _path(config, "artifacts") / "evaluation" / f"seed={seed}"
    portfolio_summary: dict[str, Any] = {model: {} for model in MODEL_NAMES}
    primary_orders: dict[str, pd.DataFrame] = {}
    for model in MODEL_NAMES:
        scores = comparison.aligned[["ticker", "origin", f"{model}_score"]].rename(
            columns={f"{model}_score": "score"}
        )
        for cost_bps in DECLARED_COSTS_BPS:
            spec = QlibBacktestSpec(cost_bps_per_side=cost_bps)
            outputs: QlibBacktestOutputs = run_qlib_topk_backtest(
                scores,
                provider_uri=provider,
                start_time=start_time,
                end_time=end_time,
                spec=spec,
            )
            positions = _flatten_positions(outputs.positions)
            terminal_positions = (
                int(
                    (
                        positions.loc[
                            positions["datetime"].eq(positions["datetime"].max()),
                            "ticker",
                        ]
                        != "__CASH__"
                    ).sum()
                )
                if len(positions)
                else 0
            )
            raw_report, metrics = _qlib_portfolio_metrics(
                outputs.report, initial_cash=initial_cash, terminal_positions=terminal_positions
            )
            orders = _flatten_orders(outputs.order_history, calendar)
            prefix = evaluation_root / "portfolio" / model / f"cost_bps={cost_bps}"
            _atomic_write_parquet(raw_report, prefix / "qlib_report.parquet")
            trade_indicators = outputs.trade_indicators.rename_axis("datetime").reset_index()
            _atomic_write_parquet(trade_indicators, prefix / "trade_indicators.parquet")
            _atomic_write_parquet(orders, prefix / "orders.parquet")
            _atomic_write_parquet(positions, prefix / "positions.parquet")
            atomic_write_json(prefix / "metrics.json", _finite_json(metrics))
            portfolio_summary[model][str(cost_bps)] = _finite_json(metrics)
            if cost_bps == 5:
                primary_orders[model] = orders

    reference_summary: dict[str, Any] = {}
    for cost_bps in DECLARED_COSTS_BPS:
        reference = simulate_equal_weight_buy_hold(
            bars,
            start_session=start_time,
            end_session=end_time,
            target_exposure=float(portfolio_config.get("target_exposure", 0.95)),
            initial_cash=initial_cash,
            cost_bps_per_side=float(cost_bps),
        )
        assert_account_reconciles(reference)
        metrics = portfolio_metrics(reference.account)
        metrics.update(
            {
                "cagr_252_session": float(
                    (1.0 + metrics["net_cumulative_return"]) ** (252.0 / len(reference.account))
                    - 1.0
                ),
                "unrealized_position_count": len(reference.final_positions),
                "cash_return_convention": "zero",
                "forced_final_liquidation": False,
                "reference": "95%-invested equal-dollar buy-and-hold",
            }
        )
        prefix = evaluation_root / "portfolio" / "equal_weight_buy_hold" / f"cost_bps={cost_bps}"
        _atomic_write_parquet(reference.account, prefix / "account.parquet")
        _atomic_write_parquet(reference.fills, prefix / "orders.parquet")
        _atomic_write_parquet(reference.rejected_orders, prefix / "rejected_orders.parquet")
        atomic_write_json(prefix / "metrics.json", _finite_json(metrics))
        reference_summary[str(cost_bps)] = _finite_json(metrics)

    worked_model = next(
        (model for model in MODEL_NAMES if _worked_example(primary_orders[model]) is not None),
        None,
    )
    worked = _worked_example(primary_orders[worked_model]) if worked_model else None
    if worked is not None:
        worked = {"model": worked_model, **worked}
    atomic_write_json(evaluation_root / "worked_trade.json", _finite_json(worked))

    _atomic_write_parquet(comparison.aligned, evaluation_root / "aligned.parquet")
    _atomic_write_parquet(comparison.daily, evaluation_root / "daily_metrics.parquet")
    _atomic_write_parquet(period_metrics, evaluation_root / "calendar_year_metrics.parquet")
    _atomic_write_parquet(comparison.bootstrap, evaluation_root / "bootstrap.parquet")
    runtime = _runtime_summary(lightgbm_metadata, worker_manifests)
    summary = {
        "schema_version": EVALUATION_SCHEMA,
        "status": "complete",
        "seed": int(seed),
        "prepare_signature": str(prepare_marker["run_signature"]),
        "dataset_revision": str(prepare_marker["dataset_revision"]),
        "comparison": {
            **comparison.summary,
            "bootstrap_sensitivity": _finite_json(comparison.bootstrap.to_dict("records")),
        },
        "calendar_year_metrics": _finite_json(period_metrics.to_dict("records")),
        "rankic_qlib_reconciliation": reconciliation,
        "portfolio": {
            "execution": "signal after origin close; Qlib shift=1; next-session open",
            "accounting_convention": (
                "split-adjusted price-return simulation; not literal historical share "
                "accounting; dividends excluded"
            ),
            "start_session": pd.Timestamp(start_time).date().isoformat(),
            "end_session": pd.Timestamp(end_time).date().isoformat(),
            "models": portfolio_summary,
            "equal_weight_buy_hold": reference_summary,
            "worked_trade": worked,
        },
        "runtime": runtime,
        "run_provenance": {
            "config_hash": config.get("_meta", {}).get("sha256"),
            "implementation_hash": _evaluation_implementation_hash(),
            "git": git_state(Path(__file__).resolve().parents[2]),
            "environment": runtime_environment(),
            "source_revisions": config.get("sources", {}),
        },
        "provenance": {
            "lightgbm_prediction": str(lightgbm_path),
            "kronos_shards": [str(path) for path in kronos_paths],
            "qlib_revision": QLIB_REVISION,
            "qlib_provider_prepare_signature": str(prepare_marker["run_signature"]),
        },
    }
    summary = _finite_json(summary)
    atomic_write_json(evaluation_root / "summary.json", summary)
    atomic_write_json(_path(config, "summaries") / f"evaluation_seed_{seed}.json", summary)
    return summary


def _number(value: Any, digits: int = 4) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _percent(value: Any, digits: int = 2) -> str:
    return "NA" if value is None else f"{100.0 * float(value):.{digits}f}%"


def _duration_seconds(value: Any) -> str:
    if value is None:
        return "NA"
    seconds = float(value)
    return f"{seconds:.3f}" if abs(seconds) < 1.0 else f"{seconds:.1f}"


def _gibibytes(value: Any) -> str:
    return "NA" if value is None else f"{float(value) / 2**30:.2f} GiB"


def _date(value: Any) -> str:
    return "NA" if value is None else pd.Timestamp(value).date().isoformat()


def render_results_report(config: Mapping[str, Any]) -> Path:
    """Render a compact human conclusion from aggregate evaluation JSON only."""

    summary_root = _path(config, "summaries")
    summaries = {
        int(path.stem.rsplit("_", 1)[1]): _read_json(path)
        for path in sorted(summary_root.glob("evaluation_seed_*.json"))
    }
    primary_seed = int(config.get("experiment", {}).get("seed", 42))
    if primary_seed not in summaries:
        raise FileNotFoundError(f"primary saved evaluation is absent for seed {primary_seed}")
    _, prepare_marker = load_complete_preparation(config)
    declared_seeds = {
        primary_seed,
        *map(int, config.get("experiment", {}).get("sensitivity_seeds", [])),
    }
    for seed in declared_seeds.intersection(summaries):
        saved = summaries[seed]
        provenance = saved.get("run_provenance", {})
        if saved.get("prepare_signature") != prepare_marker.get("run_signature"):
            raise ValueError(f"saved seed {seed} evaluation uses stale prepared data")
        if provenance.get("config_hash") != config.get("_meta", {}).get("sha256"):
            raise ValueError(f"saved seed {seed} evaluation uses a stale configuration")
        if provenance.get("implementation_hash") != _evaluation_implementation_hash():
            raise ValueError(f"saved seed {seed} evaluation uses stale evaluation code")
    primary = summaries[primary_seed]
    comparison = primary["comparison"]
    models = comparison["models"]
    interval = comparison["primary_confidence_interval"]
    delta = float(comparison["mean_daily_rankic_difference"])
    lower, upper = float(interval["lower"]), float(interval["upper"])
    if lower > 0:
        ranking_answer = "Kronos ranked the panel better on the primary test"
    elif upper < 0:
        ranking_answer = "LightGBM ranked the panel better on the primary test"
    else:
        ranking_answer = (
            "The ranking evidence is inconclusive because the paired 95% interval contains zero"
        )

    portfolios = primary["portfolio"]
    kronos_net = portfolios["models"]["kronos"]["5"]["net_cumulative_return"]
    lightgbm_net = portfolios["models"]["lightgbm"]["5"]["net_cumulative_return"]
    if delta > 0:
        portfolio_answer = (
            "The positive ranking difference also produced the higher 5-bps net return."
            if kronos_net > lightgbm_net
            else "The positive ranking difference did not produce the higher 5-bps net return."
        )
    else:
        portfolio_answer = (
            "LightGBM also produced the higher 5-bps net return."
            if lightgbm_net > kronos_net
            else "The ranking and 5-bps net-return ordering differ."
        )

    declared = declared_seeds
    available = declared.intersection(summaries)
    completed_seeds = sorted(available)
    seed_deltas = [
        float(summaries[seed]["comparison"]["mean_daily_rankic_difference"])
        for seed in completed_seeds
    ]
    seed_intervals = [
        summaries[seed]["comparison"]["primary_confidence_interval"] for seed in completed_seeds
    ]
    count_words = {1: "one", 2: "two", 3: "three"}
    completed_count = count_words.get(len(completed_seeds), str(len(completed_seeds)))
    if available != declared:
        seed_rank_stability = (
            f"Seed sensitivity is incomplete ({len(available)}/{len(declared)} declared seeds)."
        )
    elif seed_deltas and all(value < 0 for value in seed_deltas):
        uncertainty = (
            "but each paired 95% CI contains zero"
            if all(float(item["lower"]) <= 0 <= float(item["upper"]) for item in seed_intervals)
            else "with mixed confidence-interval evidence"
        )
        seed_rank_stability = (
            f"All {completed_count} declared-seed RankIC differences are negative, "
            f"{uncertainty}; ranking evidence is not robust proof of a difference."
        )
    elif seed_deltas and all(np.sign(value) == np.sign(seed_deltas[0]) for value in seed_deltas):
        seed_rank_stability = (
            f"All {completed_count} declared seeds have the same RankIC-difference sign; "
            "the per-seed intervals below determine its uncertainty."
        )
    else:
        seed_rank_stability = "The RankIC-difference sign changes across declared seeds."
    period_deltas = [
        float(item["mean_rankic_difference"])
        for item in primary.get("calendar_year_metrics", [])
        if item.get("mean_rankic_difference") is not None
    ]
    year_metrics = primary.get("calendar_year_metrics", [])
    positive_years = [
        int(item["calendar_year"])
        for item in year_metrics
        if float(item["mean_rankic_difference"]) > 0
    ]
    negative_years = [
        int(item["calendar_year"])
        for item in year_metrics
        if float(item["mean_rankic_difference"]) < 0
    ]
    if positive_years and negative_years:
        period_stability = (
            "The primary-seed calendar-year RankIC difference changes sign "
            f"(positive in {', '.join(map(str, positive_years))}; negative in "
            f"{', '.join(map(str, negative_years))})."
        )
    elif len(period_deltas) > 1:
        period_stability = "The RankIC-difference sign is consistent across calendar-year segments."
    else:
        period_stability = "Only one calendar-year segment is available."

    seed_rows: list[str] = []
    winner_by_seed: dict[int, tuple[str, float]] = {}
    for seed in completed_seeds:
        saved = summaries[seed]
        saved_comparison = saved["comparison"]
        saved_models = saved_comparison["models"]
        saved_interval = saved_comparison["primary_confidence_interval"]
        saved_portfolios = saved["portfolio"]["models"]
        kronos_return = float(saved_portfolios["kronos"]["5"]["net_cumulative_return"])
        lightgbm_return = float(saved_portfolios["lightgbm"]["5"]["net_cumulative_return"])
        return_difference_pp = 100.0 * (kronos_return - lightgbm_return)
        if return_difference_pp > 0:
            winner = "Kronos"
        elif return_difference_pp < 0:
            winner = "LightGBM"
        else:
            winner = "Tie"
        winner_by_seed[seed] = (winner, abs(return_difference_pp))
        seed_rows.append(
            "| "
            + " | ".join(
                [
                    str(seed),
                    _number(saved_models["kronos"]["mean_daily_rankic"]),
                    _number(saved_models["lightgbm"]["mean_daily_rankic"]),
                    _number(saved_comparison["mean_daily_rankic_difference"]),
                    (f"[{_number(saved_interval['lower'])}, {_number(saved_interval['upper'])}]"),
                    _percent(kronos_return),
                    _percent(lightgbm_return),
                    f"{winner} (+{abs(return_difference_pp):.2f} pp)"
                    if winner != "Tie"
                    else "Tie (0.00 pp)",
                ]
            )
            + " |"
        )

    distinct_winners = {winner for winner, _ in winner_by_seed.values()}
    if len(distinct_winners) > 1:
        lightgbm_seeds = [
            seed for seed, (winner, _) in winner_by_seed.items() if winner == "LightGBM"
        ]
        kronos_seeds = [seed for seed, (winner, _) in winner_by_seed.items() if winner == "Kronos"]
        kronos_narrow = bool(kronos_seeds) and all(
            winner_by_seed[seed][1] < 5.0 for seed in kronos_seeds
        )
        lightgbm_label = " and ".join(map(str, lightgbm_seeds))
        kronos_label = " and ".join(map(str, kronos_seeds))
        portfolio_stability = (
            "The 5-bps portfolio winner is not seed-stable: "
            f"LightGBM wins seeds {lightgbm_label}; Kronos "
            f"{'narrowly ' if kronos_narrow else ''}wins seed {kronos_label}."
        )
    elif winner_by_seed:
        only_winner = next(iter(distinct_winners))
        portfolio_stability = (
            f"The 5-bps portfolio winner is {only_winner} across all completed declared seeds."
        )
    else:  # pragma: no cover - primary seed is required above
        portfolio_stability = "No completed declared-seed portfolio results are available."

    year_rows = [
        (
            f"| {int(item['calendar_year'])} | {int(item['dates'])} | "
            f"{_number(item['kronos_mean_rankic'])} | "
            f"{_number(item['lightgbm_mean_rankic'])} | "
            f"{_number(item['mean_rankic_difference'])} |"
        )
        for item in sorted(year_metrics, key=lambda value: int(value["calendar_year"]))
    ]

    rows = []
    for model in MODEL_NAMES:
        signal = models[model]
        portfolio = portfolios["models"][model]["5"]
        runtime = primary["runtime"][model]
        elapsed = runtime.get("test_wall_seconds", runtime.get("test_inference_seconds"))
        allocated = runtime.get(
            "peak_allocated_bytes_per_worker_max", runtime.get("peak_memory_bytes")
        )
        reserved = runtime.get("peak_reserved_bytes_per_worker_max")
        rows.append(
            "| "
            + " | ".join(
                [
                    model,
                    _percent(signal["prediction_coverage"]),
                    _number(signal["mean_daily_rankic"]),
                    (
                        f"{_number(delta)} [{_number(lower)}, {_number(upper)}]"
                        if model == "kronos"
                        else "reference"
                    ),
                    _number(signal["mean_daily_mae_bps"], 1),
                    _percent(portfolio["net_cumulative_return"]),
                    _percent(portfolio["cagr_252_session"]),
                    _number(portfolio["net_sharpe_zero_cash"], 2),
                    _percent(portfolio["max_drawdown"]),
                    _number(portfolio["turnover_rate_sum"], 2),
                    f"${portfolio['transaction_costs_dollars']:,.0f}",
                    _duration_seconds(elapsed),
                    _gibibytes(allocated),
                    _gibibytes(reserved),
                ]
            )
            + " |"
        )
    reference = portfolios["equal_weight_buy_hold"]["5"]
    intro = (
        f"Primary seed: {primary_seed}. Test span: {portfolios['start_session']} through "
        f"{portfolios['end_session']} (signals trade at the next session open). Costs are "
        "5 bps per side; cash return is zero."
    )
    conclusion = (
        f"{ranking_answer}: mean daily RankIC difference (Kronos − LightGBM) was "
        f"{_number(delta)} with paired moving-block 95% CI [{_number(lower)}, "
        f"{_number(upper)}] across {comparison['paired_rankic_dates']} dates. "
        f"{portfolio_answer}"
    )
    reference_text = (
        "The zero-score reference has RankIC `NA` and MAE "
        f"{_number(comparison['zero_score']['mean_daily_mae_bps'], 1)} bp. The "
        "95%-invested equal-weight buy-and-hold reference returned "
        f"{_percent(reference['net_cumulative_return'])} net, with Sharpe "
        f"{_number(reference['net_sharpe_zero_cash'], 2)}, CAGR "
        f"{_percent(reference['cagr_252_session'])}, and maximum drawdown "
        f"{_percent(reference['max_drawdown'])}."
    )
    table_header = (
        "| Model | Coverage | Mean RankIC | Paired Δ RankIC (95% CI) | MAE (bp) | "
        "Net return | CAGR | Net Sharpe | Max drawdown | Σ daily turnover rate | "
        "Costs | Inference seconds | Peak GPU allocated | Peak GPU reserved |"
    )
    cost_rows = [
        (
            f"| {model} | "
            f"{_percent(portfolios['models'][model]['0']['net_cumulative_return'])} | "
            f"{_percent(portfolios['models'][model]['5']['net_cumulative_return'])} | "
            f"{_percent(portfolios['models'][model]['15']['net_cumulative_return'])} |"
        )
        for model in MODEL_NAMES
    ]
    bootstrap_by_length = {
        int(item["block_length"]): item for item in comparison.get("bootstrap_sensitivity", [])
    }
    missing_blocks = {10, 40}.difference(bootstrap_by_length)
    if missing_blocks:
        raise ValueError(
            "saved evaluation JSON lacks declared bootstrap blocks "
            f"{sorted(missing_blocks)}; regenerate evaluation"
        )
    bootstrap_rows = [
        (
            f"| {block_length} | {_number(item['estimate'])} | "
            f"{_number(item['lower'])} | {_number(item['upper'])} |"
        )
        for block_length, item in sorted(bootstrap_by_length.items())
    ]
    worked = portfolios.get("worked_trade")
    if worked is None:
        worked_text = "No completed round trip was available for a worked example."
    else:
        worked_text = (
            f"The {worked['model']} strategy selected **{worked['selected_stock']}** "
            f"from the {_date(worked['entry_signal_date'])} post-close signal and "
            f"bought {float(worked['shares']):,.0f} shares at the next open on "
            f"{_date(worked['entry_execution_date'])} for "
            f"${float(worked['entry_price']):,.4f}. Its {_date(worked['exit_signal_date'])} "
            f"exit signal filled at the {_date(worked['exit_execution_date'])} open for "
            f"${float(worked['exit_price']):,.4f}: gross result "
            f"${float(worked['gross_result_dollars']):,.2f}, fees "
            f"${float(worked['fees_dollars']):,.2f}, and net result "
            f"${float(worked['net_result_dollars']):,.2f}. This deterministic accounting "
            "example is not evidence of model skill and is not a forced final liquidation."
        )
    text = "\n".join(
        [
            "# Leonos v1 results",
            "",
            intro,
            "",
            conclusion,
            "",
            table_header,
            (
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: | ---: | ---: |"
            ),
            *rows,
            "",
            reference_text,
            "",
            "## Figures",
            "",
            "![Paired daily RankIC difference](figures/rankic-difference.png)",
            "",
            "![Compounded net wealth at five bps per side](figures/net-wealth.png)",
            "",
            "## Seed and period stability",
            "",
            seed_rank_stability,
            "",
            portfolio_stability,
            "",
            period_stability,
            "",
            (
                "| Seed | Kronos RankIC | LightGBM RankIC | Δ RankIC (K−L) | "
                "Paired 95% CI | Kronos net, 5 bps | LightGBM net, 5 bps | "
                "Portfolio winner (margin) |"
            ),
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            *seed_rows,
            "",
            "Primary-seed calendar-year RankIC:",
            "",
            "| Year | Dates | Kronos RankIC | LightGBM RankIC | Δ RankIC (K−L) |",
            "| ---: | ---: | ---: | ---: | ---: |",
            *year_rows,
            "",
            "## Sensitivities",
            "",
            "| Model | Net return, 0 bps | Net return, 5 bps | Net return, 15 bps |",
            "| --- | ---: | ---: | ---: |",
            *cost_rows,
            "",
            "| Bootstrap block (sessions) | Mean Δ RankIC | 95% CI lower | 95% CI upper |",
            "| ---: | ---: | ---: | ---: |",
            *bootstrap_rows,
            "",
            "## Worked accounting example",
            "",
            worked_text,
            "",
            ("Remaining positions are marked at the last close and are not forcibly liquidated."),
            "",
            "## Limits",
            "",
            (
                "This is a finite test on a fixed surviving-stock basket using a "
                "present-day historical snapshot, not a point-in-time universe. The "
                "checkpoint's June 2024 pretraining cutoff is author-reported rather "
                "than independently audited. Kronos and LightGBM differ in architecture "
                "and training history, so this does not isolate pretraining causally. "
                "The portfolio is a split-adjusted price-return simulation, not literal "
                "historical share accounting; dividends are excluded. Daily next-open "
                "fills and fixed proportional costs simplify execution and say nothing "
                "about live fill quality or capacity."
            ),
            "",
        ]
    )
    return _atomic_write_text(text, Path("reports") / "results.md")


__all__ = ["EVALUATION_SCHEMA", "evaluate_saved_predictions", "render_results_report"]
