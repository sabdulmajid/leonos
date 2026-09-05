"""Paired return-path scenarios from completed, saved portfolio ledgers.

This module deliberately performs no forecasting and no retraining. It applies
the same circular moving-block draws to saved daily RankIC and to Kronos,
LightGBM, and reference returns so every comparison remains paired. The result is
a conditional historical stress distribution, not a generative market model or
a claim about future odds.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .artifacts import atomic_write_json, git_state, runtime_environment, stable_hash

SCENARIO_SCHEMA = "leonos.paired_rankic_return_block_bootstrap.v1"
SCENARIO_METHOD = (
    "paired_circular_moving_block_bootstrap_of_saved_rankic_and_net_returns"
)
SERIES = ("kronos", "lightgbm", "reference")
DEFAULT_SCENARIO_CONFIG = Path("configs/scenario.yaml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scenario_config(path: str | Path = DEFAULT_SCENARIO_CONFIG) -> dict[str, Any]:
    """Load and hash the post-v1 scenario settings without mutating v1 config."""

    source = Path(path)
    raw = source.read_bytes()
    settings = yaml.safe_load(raw)
    if not isinstance(settings, dict):
        raise ValueError(f"scenario configuration must be a mapping: {source}")
    settings["_meta"] = {"path": str(source), "sha256": hashlib.sha256(raw).hexdigest()}
    return settings


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


def _read_returns(path: Path, *, date_column: str, value_column: str) -> pd.Series:
    if not path.is_file():
        raise FileNotFoundError(f"saved portfolio ledger is required: {path}")
    frame = pd.read_parquet(path, columns=[date_column, value_column])
    dates = pd.to_datetime(frame[date_column], utc=True, errors="raise").dt.tz_convert(None)
    dates = dates.dt.normalize()
    values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=np.float64)
    if len(frame) == 0 or dates.duplicated().any():
        raise ValueError(f"ledger must have non-empty unique dates: {path}")
    if not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError(f"ledger returns must be finite and greater than -1: {path}")
    return pd.Series(values, index=pd.DatetimeIndex(dates), name=value_column).sort_index()


def load_saved_paired_returns(
    artifacts_root: str | Path,
    *,
    forecast_seed: int,
    cost_bps_per_side: int,
) -> pd.DataFrame:
    """Load and strictly align the three saved net-return series."""

    root = (
        Path(artifacts_root)
        / "evaluation"
        / f"seed={int(forecast_seed)}"
        / "portfolio"
    )
    values: dict[str, pd.Series] = {}
    for model in ("kronos", "lightgbm"):
        values[model] = _read_returns(
            root / model / f"cost_bps={int(cost_bps_per_side)}" / "qlib_report.parquet",
            date_column="datetime",
            value_column="net_return",
        )
    values["reference"] = _read_returns(
        root
        / "equal_weight_buy_hold"
        / f"cost_bps={int(cost_bps_per_side)}"
        / "account.parquet",
        date_column="session",
        value_column="net_return",
    )
    indexes = [series.index for series in values.values()]
    if any(not indexes[0].equals(index) for index in indexes[1:]):
        raise ValueError(
            f"saved return dates do not align for seed={forecast_seed}, "
            f"cost={cost_bps_per_side}"
        )
    frame = pd.DataFrame(values).rename_axis("session").reset_index()
    if not frame["session"].is_monotonic_increasing:
        raise AssertionError("paired return dates are not ordered")
    return frame


def load_saved_daily_rankic(
    artifacts_root: str | Path,
    *,
    forecast_seed: int,
) -> pd.DataFrame:
    """Load one completed seed's paired daily cross-sectional RankIC rows."""

    path = (
        Path(artifacts_root)
        / "evaluation"
        / f"seed={int(forecast_seed)}"
        / "daily_metrics.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(f"saved daily metrics are required: {path}")
    columns = ["origin", "kronos_rankic", "lightgbm_rankic", "delta_rankic"]
    frame = pd.read_parquet(path, columns=columns)
    frame["origin"] = (
        pd.to_datetime(frame["origin"], utc=True, errors="raise")
        .dt.tz_convert(None)
        .dt.normalize()
    )
    numeric = frame[columns[1:]].apply(pd.to_numeric, errors="coerce")
    if (
        frame.empty
        or frame["origin"].duplicated().any()
        or not frame["origin"].is_monotonic_increasing
        or not np.isfinite(numeric.to_numpy(dtype=float)).all()
    ):
        raise ValueError(f"daily RankIC rows must be finite, unique, and ordered: {path}")
    if not np.allclose(
        numeric["kronos_rankic"] - numeric["lightgbm_rankic"],
        numeric["delta_rankic"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"daily RankIC differences do not reconcile: {path}")
    return pd.concat([frame[["origin"]], numeric], axis=1)


def _validate_paired_returns(frame: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray]:
    required = {"session", *SERIES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"paired returns missing columns: {sorted(missing)}")
    dates = pd.DatetimeIndex(
        pd.to_datetime(frame["session"], utc=True, errors="raise").dt.tz_convert(None)
    ).normalize()
    if len(dates) == 0 or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("paired returns require non-empty, unique, ordered sessions")
    returns = frame.loc[:, SERIES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(returns).all() or (returns <= -1.0).any():
        raise ValueError("paired returns must be finite and greater than -1")
    return dates, returns


def paired_block_bootstrap_paths(
    returns_by_scenario: Mapping[tuple[int, int], pd.DataFrame],
    rankic_by_seed: Mapping[int, pd.DataFrame],
    *,
    replicates: int,
    block_length: int,
    seed: int,
    initial_value: float = 100.0,
    batch_size: int = 1_024,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Bootstrap terminal wealth and drawdown with shared date-block draws.

    The exact same sampled date indices are used for RankIC and every model,
    forecast seed, and cost case. Circular blocks give every observed date equal
    start weight, preserve paired comparisons, and avoid the edge-weighting bias
    of a non-circular block sampler.
    """

    if not returns_by_scenario:
        raise ValueError("at least one scenario is required")
    if not rankic_by_seed:
        raise ValueError("paired daily RankIC rows are required")
    if replicates < 1 or block_length < 1 or batch_size < 1:
        raise ValueError("replicates, block_length, and batch_size must be positive")
    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("initial_value must be finite and positive")

    keys = sorted((int(seed_), int(cost)) for seed_, cost in returns_by_scenario)
    matrices: list[np.ndarray] = []
    canonical_dates: pd.DatetimeIndex | None = None
    for key in keys:
        dates, values = _validate_paired_returns(returns_by_scenario[key])
        if canonical_dates is None:
            canonical_dates = dates
        elif not canonical_dates.equals(dates):
            raise ValueError("all scenario ledgers must cover identical ordered sessions")
        matrices.append(values)
    assert canonical_dates is not None

    seed_keys = sorted({forecast_seed for forecast_seed, _ in keys})
    if set(seed_keys) != {int(value) for value in rankic_by_seed}:
        raise ValueError("daily RankIC seeds must exactly match return scenario seeds")
    rankic_matrices: list[np.ndarray] = []
    canonical_rankic_dates: pd.DatetimeIndex | None = None
    for forecast_seed in seed_keys:
        frame = rankic_by_seed[forecast_seed]
        required = {
            "origin",
            "kronos_rankic",
            "lightgbm_rankic",
            "delta_rankic",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"daily RankIC rows missing columns: {sorted(missing)}")
        dates = pd.DatetimeIndex(
            pd.to_datetime(frame["origin"], utc=True, errors="raise").dt.tz_convert(None)
        ).normalize()
        values = frame[
            ["kronos_rankic", "lightgbm_rankic", "delta_rankic"]
        ].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if (
            len(dates) != len(canonical_dates)
            or dates.has_duplicates
            or not dates.is_monotonic_increasing
            or not np.isfinite(values).all()
        ):
            raise ValueError("daily RankIC rows must align in length and be finite")
        if not np.allclose(values[:, 0] - values[:, 1], values[:, 2], atol=1e-12):
            raise ValueError("daily RankIC difference does not reconcile")
        if canonical_rankic_dates is None:
            canonical_rankic_dates = dates
        elif not canonical_rankic_dates.equals(dates):
            raise ValueError("daily RankIC origins must be identical across seeds")
        rankic_matrices.append(values)

    observations = len(canonical_dates)
    effective_block = min(int(block_length), observations)
    blocks_needed = int(math.ceil(observations / effective_block))
    combined = np.concatenate(matrices, axis=1)
    combined_rankic = np.concatenate(rankic_matrices, axis=1)
    series_per_scenario = len(SERIES)
    scenario_count = len(keys)
    terminal = np.empty((replicates, scenario_count, series_per_scenario), dtype=np.float64)
    drawdown = np.empty_like(terminal)
    rankic_means = np.empty((replicates, len(seed_keys), 3), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    offsets = np.arange(effective_block, dtype=np.int64)

    written = 0
    while written < replicates:
        count = min(int(batch_size), replicates - written)
        starts = rng.integers(0, observations, size=(count, blocks_needed), dtype=np.int64)
        indices = (
            (starts[:, :, None] + offsets) % observations
        ).reshape(count, -1)[:, :observations]
        sampled = combined[indices]
        wealth = initial_value * np.cumprod(1.0 + sampled, axis=1)
        terminal[written : written + count] = wealth[:, -1, :].reshape(
            count, scenario_count, series_per_scenario
        )
        running_peak = np.maximum.accumulate(np.maximum(wealth, initial_value), axis=1)
        minimum_drawdown = np.min(wealth / running_peak - 1.0, axis=1)
        drawdown[written : written + count] = minimum_drawdown.reshape(
            count, scenario_count, series_per_scenario
        )
        rankic_means[written : written + count] = combined_rankic[indices].mean(
            axis=1
        ).reshape(count, len(seed_keys), 3)
        written += count

    records: list[pd.DataFrame] = []
    summaries: dict[str, Any] = {}
    quantile_levels = np.array([0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975])
    for scenario_index, (forecast_seed, cost) in enumerate(keys):
        ends = terminal[:, scenario_index, :]
        dds = drawdown[:, scenario_index, :]
        observed_returns = matrices[scenario_index]
        observed_paths = initial_value * np.cumprod(1.0 + observed_returns, axis=0)
        observed_peaks = np.maximum.accumulate(
            np.maximum(observed_paths, initial_value), axis=0
        )
        observed_drawdowns = np.min(observed_paths / observed_peaks - 1.0, axis=0)
        rankic_index = seed_keys.index(forecast_seed)
        sampled_rankic = rankic_means[:, rankic_index, :]
        observed_rankic = rankic_matrices[rankic_index].mean(axis=0)
        frame = pd.DataFrame(
            {
                "forecast_seed": forecast_seed,
                "cost_bps_per_side": cost,
                "replicate": np.arange(replicates, dtype=np.int64),
                "kronos_ending_value": ends[:, 0],
                "lightgbm_ending_value": ends[:, 1],
                "reference_ending_value": ends[:, 2],
                "kronos_max_drawdown": dds[:, 0],
                "lightgbm_max_drawdown": dds[:, 1],
                "reference_max_drawdown": dds[:, 2],
                "kronos_minus_lightgbm": ends[:, 0] - ends[:, 1],
                "kronos_mean_rankic": sampled_rankic[:, 0],
                "lightgbm_mean_rankic": sampled_rankic[:, 1],
                "delta_mean_rankic": sampled_rankic[:, 2],
            }
        )
        records.append(frame)
        model_summary: dict[str, Any] = {}
        for model_index, model in enumerate(SERIES):
            model_ends = ends[:, model_index]
            model_dds = dds[:, model_index]
            quantiles = np.quantile(model_ends, quantile_levels)
            model_summary[model] = {
                "ending_value_quantiles": {
                    f"{level:.3f}": float(value)
                    for level, value in zip(quantile_levels, quantiles, strict=True)
                },
                "observed_ending_value": float(observed_paths[-1, model_index]),
                "observed_max_drawdown": float(observed_drawdowns[model_index]),
                "mean_ending_value": float(model_ends.mean()),
                "resampled_fraction_ending_below_initial": float(
                    np.mean(model_ends < initial_value)
                ),
                "median_max_drawdown": float(np.median(model_dds)),
                "max_drawdown_0.025_quantile": float(np.quantile(model_dds, 0.025)),
            }
        key = f"seed={forecast_seed}/cost_bps={cost}"
        log_terminal_ratio = np.log(ends[:, 0] / ends[:, 1])
        delta_rankic_interval = np.quantile(sampled_rankic[:, 2], [0.025, 0.975])
        summaries[key] = {
            "forecast_seed": forecast_seed,
            "cost_bps_per_side": cost,
            "models": model_summary,
            "rankic": {
                "observed_kronos": float(observed_rankic[0]),
                "observed_lightgbm": float(observed_rankic[1]),
                "observed_delta_kronos_minus_lightgbm": float(observed_rankic[2]),
                "resampled_mean_delta": float(sampled_rankic[:, 2].mean()),
                "resampled_delta_95_interval": [
                    float(delta_rankic_interval[0]),
                    float(delta_rankic_interval[1]),
                ],
                "resampled_fraction_delta_above_zero": float(
                    np.mean(sampled_rankic[:, 2] > 0.0)
                ),
            },
            "paired_comparison": {
                "resampled_fraction_kronos_beats_lightgbm": float(
                    np.mean(ends[:, 0] > ends[:, 1])
                ),
                "resampled_fraction_lightgbm_beats_kronos": float(
                    np.mean(ends[:, 1] > ends[:, 0])
                ),
                "resampled_fraction_kronos_beats_reference": float(
                    np.mean(ends[:, 0] > ends[:, 2])
                ),
                "resampled_fraction_lightgbm_beats_reference": float(
                    np.mean(ends[:, 1] > ends[:, 2])
                ),
                "median_kronos_minus_lightgbm": float(
                    np.median(ends[:, 0] - ends[:, 1])
                ),
                "kronos_minus_lightgbm_95_interval": [
                    float(np.quantile(ends[:, 0] - ends[:, 1], 0.025)),
                    float(np.quantile(ends[:, 0] - ends[:, 1], 0.975)),
                ],
                "median_log_terminal_ratio_kronos_over_lightgbm": float(
                    np.median(log_terminal_ratio)
                ),
                "log_terminal_ratio_kronos_over_lightgbm_95_interval": [
                    float(np.quantile(log_terminal_ratio, 0.025)),
                    float(np.quantile(log_terminal_ratio, 0.975)),
                ],
            },
        }

    distribution = pd.concat(records, ignore_index=True)
    summary = {
        "schema_version": SCENARIO_SCHEMA,
        "status": "complete",
        "method": SCENARIO_METHOD,
        "interpretation": (
            "conditional historical return-path stress analysis; not independent future "
            "market simulations, fresh model fits, or calibrated probabilities"
        ),
        "state_limitation": (
            "a block can begin from an already-invested original portfolio state; sampled "
            "returns retain original strategy states inside blocks, while holdings, orders, "
            "costs, and deployment are not recomputed at concatenated block boundaries"
        ),
        "rankic_state_note": (
            "daily cross-sectional RankIC resampling has no portfolio-state splice; "
            "the state limitation applies to resampled account-return paths"
        ),
        "replicates_per_scenario": int(replicates),
        "scenario_count": int(scenario_count),
        "resampled_date_paths": int(replicates),
        "scenario_path_evaluations": int(replicates * scenario_count),
        "block_length_sessions": int(block_length),
        "effective_block_length_sessions": int(effective_block),
        "sessions_per_path": int(observations),
        "bootstrap_seed": int(seed),
        "normalized_initial_value": float(initial_value),
        "shared_block_draws_across_all_scenarios": True,
        "first_session": canonical_dates[0].date().isoformat(),
        "last_session": canonical_dates[-1].date().isoformat(),
        "scenarios": summaries,
    }
    return distribution, summary


def run_saved_scenario_analysis(
    config: Mapping[str, Any],
    *,
    scenario_config: Mapping[str, Any] | None = None,
    replicates: int | None = None,
    block_length: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run the configured CPU-only scenario analysis and persist its evidence."""

    settings = scenario_config
    if not isinstance(settings, Mapping):
        settings = load_scenario_config()
    if settings.get("method") != SCENARIO_METHOD:
        raise ValueError(
            f"unsupported scenario method {settings.get('method')!r}; "
            f"expected {SCENARIO_METHOD!r}"
        )
    forecast_seeds = [int(value) for value in settings["forecast_seeds"]]
    costs = [int(value) for value in settings["cost_bps_per_side"]]
    artifacts_root = Path(str(config["paths"]["artifacts"]))
    source_ledgers: list[dict[str, Any]] = []
    evaluation_summaries: dict[int, dict[str, Any]] = {}
    expected_config_hash = str(config.get("_meta", {}).get("sha256"))
    compatibility: set[tuple[Any, ...]] = set()
    for forecast_seed in forecast_seeds:
        evaluation_summary_path = (
            artifacts_root / "evaluation" / f"seed={forecast_seed}" / "summary.json"
        )
        if not evaluation_summary_path.is_file():
            raise FileNotFoundError(
                f"completed evaluation summary is required: {evaluation_summary_path}"
            )
        evaluation_summary = json.loads(evaluation_summary_path.read_text(encoding="utf-8"))
        if (
            evaluation_summary.get("status") != "complete"
            or int(evaluation_summary.get("seed", -1)) != forecast_seed
            or evaluation_summary.get("schema_version") != "leonos.saved_evaluation.v1"
            or evaluation_summary.get("run_provenance", {}).get("config_hash")
            != expected_config_hash
        ):
            raise ValueError(
                f"evaluation is incomplete or incompatible for forecast seed "
                f"{forecast_seed}"
            )
        compatibility.add(
            (
                evaluation_summary.get("prepare_signature"),
                evaluation_summary.get("forecast_origin_start"),
                evaluation_summary.get("forecast_origin_end"),
                evaluation_summary.get("comparison", {}).get("paired_rankic_dates"),
                evaluation_summary.get("portfolio", {}).get("start_session"),
                evaluation_summary.get("portfolio", {}).get("end_session"),
                evaluation_summary.get("run_provenance", {}).get(
                    "implementation_hash"
                ),
            )
        )
        evaluation_summaries[forecast_seed] = evaluation_summary
        source_ledgers.append(
            {
                "role": "evaluation_summary",
                "forecast_seed": forecast_seed,
                "path": str(evaluation_summary_path),
                "sha256": _sha256(evaluation_summary_path),
            }
        )
        daily_metrics_path = (
            artifacts_root / "evaluation" / f"seed={forecast_seed}" / "daily_metrics.parquet"
        )
        if not daily_metrics_path.is_file():
            raise FileNotFoundError(f"saved daily metrics are required: {daily_metrics_path}")
        source_ledgers.append(
            {
                "role": "daily_metrics",
                "forecast_seed": forecast_seed,
                "path": str(daily_metrics_path),
                "sha256": _sha256(daily_metrics_path),
            }
        )
        for cost in costs:
            for model, filename in (
                ("kronos", "qlib_report.parquet"),
                ("lightgbm", "qlib_report.parquet"),
                ("equal_weight_buy_hold", "account.parquet"),
            ):
                ledger_path = (
                    artifacts_root
                    / "evaluation"
                    / f"seed={forecast_seed}"
                    / "portfolio"
                    / model
                    / f"cost_bps={cost}"
                    / filename
                )
                if not ledger_path.is_file():
                    raise FileNotFoundError(f"saved portfolio ledger is required: {ledger_path}")
                source_ledgers.append(
                    {
                        "role": model,
                        "forecast_seed": forecast_seed,
                        "cost_bps_per_side": cost,
                        "path": str(ledger_path),
                        "sha256": _sha256(ledger_path),
                    }
                )
    if len(compatibility) != 1:
        raise ValueError(
            "evaluation seeds disagree on preparation, coverage, or implementation"
        )

    calendar_path = Path(str(config["paths"]["prepared_data"])) / "sessions.parquet"
    if not calendar_path.is_file():
        raise FileNotFoundError(f"prepared exchange calendar is required: {calendar_path}")
    calendar = pd.DatetimeIndex(
        pd.to_datetime(
            pd.read_parquet(calendar_path, columns=["session"])["session"],
            utc=True,
            errors="raise",
        )
        .dt.tz_convert(None)
        .dt.normalize()
    )
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("prepared exchange calendar must be non-empty, unique, and ordered")
    source_ledgers.append(
        {
            "role": "prepared_exchange_calendar",
            "path": str(calendar_path),
            "sha256": _sha256(calendar_path),
        }
    )
    scenario_returns = {
        (forecast_seed, cost): load_saved_paired_returns(
            config["paths"]["artifacts"],
            forecast_seed=forecast_seed,
            cost_bps_per_side=cost,
        )
        for forecast_seed in forecast_seeds
        for cost in costs
    }
    rankic_by_seed = {
        forecast_seed: load_saved_daily_rankic(
            artifacts_root, forecast_seed=forecast_seed
        )
        for forecast_seed in forecast_seeds
    }
    for (forecast_seed, cost), frame in scenario_returns.items():
        portfolio = evaluation_summaries[forecast_seed].get("portfolio", {})
        expected_start = pd.Timestamp(portfolio.get("start_session")).normalize()
        expected_end = pd.Timestamp(portfolio.get("end_session")).normalize()
        observed_dates = pd.DatetimeIndex(frame["session"])
        if (
            len(observed_dates) == 0
            or observed_dates[0] != expected_start
            or observed_dates[-1] != expected_end
        ):
            raise ValueError(
                f"ledger date coverage disagrees with evaluation summary for "
                f"seed={forecast_seed}, cost={cost}"
            )
        expected_nodes = {
            "kronos": portfolio.get("models", {}).get("kronos", {}).get(str(cost), {}),
            "lightgbm": portfolio.get("models", {})
            .get("lightgbm", {})
            .get(str(cost), {}),
            "reference": portfolio.get("equal_weight_buy_hold", {}).get(str(cost), {}),
        }
        for name, node in expected_nodes.items():
            expected_return = node.get("net_cumulative_return")
            realized_return = float(np.prod(1.0 + frame[name].to_numpy(float)) - 1.0)
            if expected_return is None or not np.isclose(
                realized_return, float(expected_return), rtol=0.0, atol=1e-10
            ):
                raise ValueError(
                    f"saved {name} returns do not reconcile to evaluation summary "
                    f"for seed={forecast_seed}, cost={cost}"
                )
    canonical_origins: pd.DatetimeIndex | None = None
    for forecast_seed, frame in rankic_by_seed.items():
        comparison = evaluation_summaries[forecast_seed].get("comparison", {})
        expected_start = pd.Timestamp(
            evaluation_summaries[forecast_seed].get("forecast_origin_start")
        ).normalize()
        expected_end = pd.Timestamp(
            evaluation_summaries[forecast_seed].get("forecast_origin_end")
        ).normalize()
        observed_dates = pd.DatetimeIndex(frame["origin"])
        if (
            len(observed_dates) != int(comparison.get("paired_rankic_dates", -1))
            or observed_dates[0] != expected_start
            or observed_dates[-1] != expected_end
        ):
            raise ValueError(
                f"daily RankIC coverage disagrees with evaluation summary for "
                f"seed={forecast_seed}"
            )
        observed_means = frame[
            ["kronos_rankic", "lightgbm_rankic", "delta_rankic"]
        ].mean()
        expected_means = np.array(
            [
                comparison.get("models", {})
                .get("kronos", {})
                .get("mean_daily_rankic"),
                comparison.get("models", {})
                .get("lightgbm", {})
                .get("mean_daily_rankic"),
                comparison.get("mean_daily_rankic_difference"),
            ],
            dtype=float,
        )
        if not np.allclose(
            observed_means.to_numpy(float),
            expected_means,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"daily RankIC means do not reconcile to evaluation summary for "
                f"seed={forecast_seed}"
            )
        if canonical_origins is None:
            canonical_origins = observed_dates
        elif not canonical_origins.equals(observed_dates):
            raise ValueError("daily RankIC origins disagree across forecast seeds")
    assert canonical_origins is not None
    calendar_positions = calendar.get_indexer(canonical_origins)
    if (calendar_positions < 0).any() or (calendar_positions + 1 >= len(calendar)).any():
        raise ValueError("a RankIC origin is absent from the prepared exchange calendar")
    expected_execution_sessions = calendar[calendar_positions + 1]
    for (forecast_seed, cost), frame in scenario_returns.items():
        observed_sessions = pd.DatetimeIndex(frame["session"])
        if not observed_sessions.equals(expected_execution_sessions):
            raise ValueError(
                "portfolio rows must be the exact next exchange sessions after RankIC "
                f"origins for seed={forecast_seed}, cost={cost}"
            )
    selected_replicates = int(
        settings["replicates"] if replicates is None else replicates
    )
    selected_block = int(
        settings["block_length_sessions"] if block_length is None else block_length
    )
    selected_seed = int(settings["seed"] if seed is None else seed)
    implementation_hash = _sha256(Path(__file__))
    run_settings = {
        "method": SCENARIO_METHOD,
        "forecast_seeds": forecast_seeds,
        "cost_bps_per_side": costs,
        "replicates": selected_replicates,
        "block_length_sessions": selected_block,
        "seed": selected_seed,
        "batch_size": int(settings["batch_size"]),
        "normalized_initial_value": float(settings["normalized_initial_value"]),
        "scenario_config_hash": settings.get("_meta", {}).get("sha256"),
    }
    run_signature = stable_hash(
        {
            "schema_version": SCENARIO_SCHEMA,
            "implementation_hash": implementation_hash,
            "settings": run_settings,
            "source_sha256": [item["sha256"] for item in source_ledgers],
        }
    )
    summaries_root = Path(str(config["paths"]["summaries"]))
    destination = (
        artifacts_root
        / "scenarios"
        / run_signature
        / "paired_rankic_return_bootstrap.parquet"
    )
    summary_path = summaries_root / "scenarios" / f"{run_signature}.json"
    if destination.is_file() and summary_path.is_file():
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            cached.get("status") == "complete"
            and cached.get("run_signature") == run_signature
            and cached.get("distribution_sha256") == _sha256(destination)
        ):
            return {
                "summary": str(summary_path),
                "distribution": str(destination),
                "replicates_per_scenario": selected_replicates,
                "scenario_count": len(scenario_returns),
                "resampled_date_paths": selected_replicates,
                "scenario_path_evaluations": selected_replicates
                * len(scenario_returns),
                "cached": True,
            }
        raise ValueError(f"incompatible cached scenario artifacts: {run_signature}")

    distribution, summary = paired_block_bootstrap_paths(
        scenario_returns,
        rankic_by_seed,
        replicates=selected_replicates,
        block_length=selected_block,
        seed=selected_seed,
        initial_value=float(settings["normalized_initial_value"]),
        batch_size=int(settings["batch_size"]),
    )
    _atomic_write_parquet(distribution, destination)
    summary["source_ledgers"] = source_ledgers
    summary["run_signature"] = run_signature
    summary["run_settings"] = run_settings
    summary["run_provenance"] = {
        "git": git_state(),
        "environment": runtime_environment(),
        "config_hash": config.get("_meta", {}).get("sha256"),
        "scenario_config_hash": settings.get("_meta", {}).get("sha256"),
        "implementation_hash": implementation_hash,
    }
    summary["distribution_path"] = str(destination)
    summary["distribution_sha256"] = _sha256(destination)
    atomic_write_json(summary_path, summary)
    return {
        "summary": str(summary_path),
        "distribution": str(destination),
        "replicates_per_scenario": selected_replicates,
        "scenario_count": len(scenario_returns),
        "resampled_date_paths": selected_replicates,
        "scenario_path_evaluations": len(distribution),
        "cached": False,
    }


__all__ = [
    "SCENARIO_SCHEMA",
    "SCENARIO_METHOD",
    "SERIES",
    "DEFAULT_SCENARIO_CONFIG",
    "load_scenario_config",
    "load_saved_paired_returns",
    "load_saved_daily_rankic",
    "paired_block_bootstrap_paths",
    "run_saved_scenario_analysis",
]
