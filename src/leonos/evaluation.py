"""Cross-sectional signal evaluation for the fixed-equity experiment.

All primary statistics are calculated from persisted prediction rows.  Dates are
the sampling unit: equities are ranked within a date, and complete date rows are
resampled in the moving-block bootstrap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

KEY_COLUMNS = ("ticker", "origin")
DEFAULT_MINIMUM_DAILY_COVERAGE = 3


@dataclass(frozen=True)
class BootstrapInterval:
    block_length: int
    replicates: int
    seed: int
    observations: int
    estimate: float
    lower: float
    upper: float
    standard_error: float


@dataclass
class ComparisonResult:
    """Evaluation tables plus a compact JSON-serializable summary."""

    aligned: pd.DataFrame
    daily: pd.DataFrame
    bootstrap: pd.DataFrame
    summary: dict[str, object]


@dataclass(frozen=True)
class QlibReconciliation:
    available: bool
    matched: bool | None
    maximum_absolute_difference: float | None
    detail: str


def spearman_average_rank(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman correlation with ordinary average-rank tie handling."""

    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[finite]
    y_array = y_array[finite]
    if len(x_array) < 2:
        return np.nan
    x_rank = pd.Series(x_array).rank(method="average").to_numpy(dtype=np.float64)
    y_rank = pd.Series(y_array).rank(method="average").to_numpy(dtype=np.float64)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = float(
        np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    )
    if denominator == 0.0:
        return np.nan
    return float(np.dot(x_centered, y_centered) / denominator)


def _canonical_keys(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = set(KEY_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing key columns: {sorted(missing)}")
    out = frame.copy()
    out["ticker"] = out["ticker"].astype("string")
    if out["ticker"].isna().any():
        raise ValueError(f"{name} contains missing tickers")
    out["origin"] = pd.to_datetime(out["origin"], utc=True, errors="raise").dt.tz_convert(
        None
    ).dt.normalize()
    duplicate = out.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicate.any():
        examples = out.loc[duplicate, list(KEY_COLUMNS)].head(5).to_dict("records")
        raise ValueError(f"{name} has duplicate prediction keys: {examples}")
    return out


def _validate_prediction_provenance(
    frame: pd.DataFrame,
    *,
    model_name: str,
    expected_seed: int,
    horizon: int,
    score_col: str,
) -> pd.DataFrame:
    """Validate an origin-level prediction artifact before using any scores."""

    required = {"model", "seed", "horizon", "status", score_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{model_name} predictions missing provenance columns: {sorted(missing)}"
        )
    if frame.empty:
        raise ValueError(f"{model_name} predictions are empty")

    declared_model = frame["model"].astype("string")
    if declared_model.isna().any() or not declared_model.eq(model_name).all():
        values = sorted(declared_model.dropna().astype(str).unique().tolist())
        raise ValueError(
            f"{model_name} prediction model provenance is mixed or wrong: {values}"
        )

    try:
        declared_seed = pd.to_numeric(frame["seed"], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{model_name} prediction seed provenance is not numeric") from exc
    if not np.isfinite(declared_seed).all() or not np.equal(
        declared_seed, int(expected_seed)
    ).all():
        values = sorted(set(declared_seed[np.isfinite(declared_seed)].tolist()))
        raise ValueError(
            f"{model_name} prediction seed provenance is mixed or wrong: "
            f"expected {expected_seed}, observed {values}"
        )

    try:
        declared_horizon = pd.to_numeric(frame["horizon"], errors="raise").to_numpy(
            dtype=float
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{model_name} prediction horizon provenance is not numeric") from exc
    if not np.isfinite(declared_horizon).all() or not np.equal(
        declared_horizon, horizon
    ).all():
        values = sorted(set(declared_horizon[np.isfinite(declared_horizon)].tolist()))
        raise ValueError(
            f"{model_name} prediction horizon provenance is mixed or wrong: "
            f"expected {horizon}, observed {values}"
        )

    try:
        score = pd.to_numeric(frame[score_col], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{model_name} prediction scores are not numeric") from exc
    status = frame["status"].astype("string")
    if status.isna().any() or status.str.strip().eq("").any():
        raise ValueError(f"{model_name} prediction status provenance is missing")
    finite = np.isfinite(score)
    successful = status.eq("ok").to_numpy(dtype=bool)
    if not np.array_equal(finite, successful):
        raise ValueError(
            f"{model_name} prediction status is inconsistent with score finiteness"
        )

    validated = frame.copy()
    validated[score_col] = score
    validated.loc[~finite, score_col] = np.nan
    return validated


def align_model_predictions(
    labels: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    *,
    expected_seed: int,
    horizon: int = 10,
    target_col: str = "target",
    score_col: str = "score",
) -> pd.DataFrame:
    """Left-align every model to the same predeclared eligible label keys.

    Prediction failures remain explicit NaNs rather than disappearing through an
    inner join.  Predictions outside the selected label period are harmlessly
    excluded; duplicate keys or model-column mismatches fail closed.
    """

    if isinstance(expected_seed, (bool, np.bool_)) or int(expected_seed) != expected_seed:
        raise ValueError("expected_seed must be an integer")
    if isinstance(horizon, (bool, np.bool_)) or int(horizon) != horizon or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if len(predictions) < 1:
        raise ValueError("at least one model prediction table is required")
    label_frame = _canonical_keys(labels, "labels")
    if target_col not in label_frame.columns:
        raise ValueError(f"labels missing target column {target_col!r}")
    label_frame[target_col] = pd.to_numeric(label_frame[target_col], errors="coerce")
    label_frame = label_frame.loc[
        np.isfinite(label_frame[target_col]), [*KEY_COLUMNS, target_col]
    ].copy()
    if label_frame.empty:
        raise ValueError("no finite realized labels were supplied")

    aligned = label_frame
    for model_name, raw_predictions in predictions.items():
        if not model_name or model_name.endswith("_score"):
            raise ValueError(f"invalid model name {model_name!r}")
        model_frame = _canonical_keys(raw_predictions, f"{model_name} predictions")
        model_frame = _validate_prediction_provenance(
            model_frame,
            model_name=model_name,
            expected_seed=int(expected_seed),
            horizon=int(horizon),
            score_col=score_col,
        )
        model_frame = model_frame.loc[:, [*KEY_COLUMNS, score_col]].copy()
        aligned = aligned.merge(
            model_frame.rename(columns={score_col: f"{model_name}_score"}),
            on=list(KEY_COLUMNS),
            how="left",
            validate="one_to_one",
        )
    return aligned.sort_values(["origin", "ticker"], kind="stable").reset_index(drop=True)


def daily_cross_sectional_metrics(
    aligned: pd.DataFrame,
    model_names: Sequence[str],
    *,
    target_col: str = "target",
    minimum_coverage: int = DEFAULT_MINIMUM_DAILY_COVERAGE,
) -> pd.DataFrame:
    """Calculate equal-date-weighted RankIC and MAE inputs on common rows."""

    if minimum_coverage < 2:
        raise ValueError("minimum_coverage must be at least two equities")
    if not model_names:
        raise ValueError("model_names cannot be empty")
    score_columns = [f"{name}_score" for name in model_names]
    missing = {"origin", "ticker", target_col, *score_columns}.difference(aligned.columns)
    if missing:
        raise ValueError(f"aligned table missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for origin, group in aligned.groupby("origin", sort=True, observed=True):
        target = pd.to_numeric(group[target_col], errors="coerce").to_numpy(dtype=float)
        target_finite = np.isfinite(target)
        common = target_finite.copy()
        row: dict[str, object] = {
            "origin": pd.Timestamp(origin),
            "eligible_count": int(target_finite.sum()),
        }
        score_arrays: dict[str, np.ndarray] = {}
        for name, column in zip(model_names, score_columns, strict=True):
            scores = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
            score_arrays[name] = scores
            finite = np.isfinite(scores) & target_finite
            row[f"{name}_available"] = int(finite.sum())
            row[f"{name}_coverage"] = (
                float(finite.sum() / target_finite.sum()) if target_finite.any() else np.nan
            )
            common &= np.isfinite(scores)

        common_count = int(common.sum())
        row["common_count"] = common_count
        row["common_coverage"] = (
            float(common_count / target_finite.sum()) if target_finite.any() else np.nan
        )
        coverage_ok = common_count >= minimum_coverage
        row["coverage_ok"] = coverage_ok
        common_target = target[common]
        for name in model_names:
            common_score = score_arrays[name][common]
            row[f"{name}_rankic"] = (
                spearman_average_rank(common_score, common_target)
                if coverage_ok
                else np.nan
            )
            row[f"{name}_mae"] = (
                float(np.mean(np.abs(common_score - common_target)))
                if coverage_ok
                else np.nan
            )
        row["zero_mae"] = (
            float(np.mean(np.abs(common_target))) if coverage_ok else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("origin", kind="stable").reset_index(drop=True)


def daily_rankic(
    panel: pd.DataFrame,
    *,
    score_col: str = "score",
    target_col: str = "target",
    minimum_coverage: int = DEFAULT_MINIMUM_DAILY_COVERAGE,
) -> pd.Series:
    """Convenience wrapper for one model's daily cross-sectional RankIC."""

    canonical = _canonical_keys(panel, "panel")
    required = {score_col, target_col}
    missing = required.difference(canonical.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")
    aligned = canonical.rename(columns={score_col: "model_score"})
    result = daily_cross_sectional_metrics(
        aligned,
        ["model"],
        target_col=target_col,
        minimum_coverage=minimum_coverage,
    )
    return result.set_index("origin")["model_rankic"].rename("rankic")


def moving_block_bootstrap_mean(
    values: Sequence[float],
    *,
    block_length: int = 20,
    replicates: int = 2_000,
    seed: int = 42,
) -> tuple[BootstrapInterval, np.ndarray]:
    """Bootstrap a time-ordered mean using non-circular moving blocks."""

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        raise ValueError("bootstrap requires at least one finite date statistic")
    if block_length < 1:
        raise ValueError("block_length must be positive")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    effective_block = min(block_length, len(array))
    blocks_needed = int(np.ceil(len(array) / effective_block))
    maximum_start = len(array) - effective_block
    rng = np.random.default_rng(seed)
    distribution = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        starts = rng.integers(0, maximum_start + 1, size=blocks_needed)
        sampled = np.concatenate(
            [array[start : start + effective_block] for start in starts]
        )[: len(array)]
        distribution[replicate] = sampled.mean()
    lower, upper = np.quantile(distribution, [0.025, 0.975])
    interval = BootstrapInterval(
        block_length=block_length,
        replicates=replicates,
        seed=seed,
        observations=len(array),
        estimate=float(array.mean()),
        lower=float(lower),
        upper=float(upper),
        standard_error=float(distribution.std(ddof=1)) if replicates > 1 else 0.0,
    )
    return interval, distribution


def bootstrap_sensitivity(
    daily_deltas: Sequence[float],
    *,
    block_lengths: Sequence[int] = (20, 10, 40),
    replicates: int = 2_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Return declared primary and block-length sensitivity intervals."""

    if not block_lengths:
        raise ValueError("block_lengths cannot be empty")
    records = []
    for block_length in block_lengths:
        interval, _ = moving_block_bootstrap_mean(
            daily_deltas,
            block_length=int(block_length),
            replicates=replicates,
            seed=seed,
        )
        records.append(asdict(interval))
    return pd.DataFrame.from_records(records)


def compare_predictions(
    labels: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    *,
    expected_seed: int,
    horizon: int = 10,
    first_model: str = "kronos",
    second_model: str = "lightgbm",
    target_col: str = "target",
    score_col: str = "score",
    minimum_coverage: int = DEFAULT_MINIMUM_DAILY_COVERAGE,
    block_lengths: Sequence[int] = (20, 10, 40),
    bootstrap_replicates: int = 2_000,
    bootstrap_seed: int = 42,
) -> ComparisonResult:
    """Evaluate two models on common rows and bootstrap their paired RankIC gap."""

    if first_model == second_model:
        raise ValueError("the two comparison model names must differ")
    missing_models = {first_model, second_model}.difference(predictions)
    if missing_models:
        raise ValueError(f"missing comparison predictions: {sorted(missing_models)}")
    selected_predictions = {
        first_model: predictions[first_model],
        second_model: predictions[second_model],
    }
    aligned = align_model_predictions(
        labels,
        selected_predictions,
        expected_seed=expected_seed,
        horizon=horizon,
        target_col=target_col,
        score_col=score_col,
    )
    model_names = [first_model, second_model]
    daily = daily_cross_sectional_metrics(
        aligned,
        model_names,
        target_col=target_col,
        minimum_coverage=minimum_coverage,
    )
    daily["delta_rankic"] = (
        daily[f"{first_model}_rankic"] - daily[f"{second_model}_rankic"]
    )
    valid_primary = np.isfinite(daily["delta_rankic"])
    if not valid_primary.any():
        raise ValueError("no dates have defined paired RankIC at the required coverage")
    bootstrap = bootstrap_sensitivity(
        daily.loc[valid_primary, "delta_rankic"].to_numpy(dtype=float),
        block_lengths=block_lengths,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )

    primary_block = int(block_lengths[0])
    primary_interval = bootstrap.loc[bootstrap["block_length"] == primary_block].iloc[0]
    eligible_observations = int(daily["eligible_count"].sum())
    common_observations = int(daily["common_count"].sum())
    models_summary: dict[str, dict[str, float | int]] = {}
    for name in model_names:
        available = int(daily[f"{name}_available"].sum())
        mean_mae = float(daily[f"{name}_mae"].mean())
        models_summary[name] = {
            "available_observations": available,
            "prediction_coverage": float(available / eligible_observations),
            "mean_daily_rankic": float(
                daily.loc[valid_primary, f"{name}_rankic"].mean()
            ),
            "mean_daily_mae": mean_mae,
            "mean_daily_mae_bps": mean_mae * 10_000.0,
        }
    zero_mae = float(daily["zero_mae"].mean())
    summary: dict[str, object] = {
        "first_model": first_model,
        "second_model": second_model,
        "prediction_seed": int(expected_seed),
        "prediction_horizon_sessions": int(horizon),
        "rankic_difference_definition": f"{first_model} - {second_model}",
        "eligible_observations": eligible_observations,
        "common_observations": common_observations,
        "common_coverage": float(common_observations / eligible_observations),
        "total_dates": int(len(daily)),
        "dates_meeting_minimum_coverage": int(daily["coverage_ok"].sum()),
        "paired_rankic_dates": int(valid_primary.sum()),
        "minimum_daily_coverage": minimum_coverage,
        "models": models_summary,
        "zero_score": {
            "mean_daily_rankic": None,
            "mean_daily_mae": zero_mae,
            "mean_daily_mae_bps": zero_mae * 10_000.0,
        },
        "mean_daily_rankic_difference": float(
            daily.loc[valid_primary, "delta_rankic"].mean()
        ),
        "primary_confidence_interval": {
            "method": "paired_moving_block_bootstrap_of_complete_dates",
            "block_length": primary_block,
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "lower": float(primary_interval["lower"]),
            "upper": float(primary_interval["upper"]),
        },
    }
    return ComparisonResult(
        aligned=aligned,
        daily=daily,
        bootstrap=bootstrap,
        summary=summary,
    )


def reconcile_daily_rankic_with_qlib(
    panel: pd.DataFrame,
    *,
    score_col: str = "score",
    target_col: str = "target",
    minimum_coverage: int = 2,
    tolerance: float = 1e-12,
) -> QlibReconciliation:
    """Reconcile the independent implementation with Qlib when installed."""

    try:
        from qlib.contrib.eva.alpha import calc_ic
    except (ImportError, ModuleNotFoundError) as exc:
        return QlibReconciliation(
            available=False,
            matched=None,
            maximum_absolute_difference=None,
            detail=f"Qlib unavailable: {exc}",
        )

    canonical = _canonical_keys(panel, "Qlib reconciliation panel")
    finite = np.isfinite(pd.to_numeric(canonical[score_col], errors="coerce")) & np.isfinite(
        pd.to_numeric(canonical[target_col], errors="coerce")
    )
    canonical = canonical.loc[finite].copy()
    counts = canonical.groupby("origin", observed=True).size()
    allowed_dates = counts.index[counts >= minimum_coverage]
    canonical = canonical.loc[canonical["origin"].isin(allowed_dates)]
    index = pd.MultiIndex.from_frame(
        canonical[["origin", "ticker"]].rename(
            columns={"origin": "datetime", "ticker": "instrument"}
        )
    )
    prediction = pd.Series(
        canonical[score_col].to_numpy(dtype=float), index=index, name="score"
    )
    label = pd.Series(
        canonical[target_col].to_numpy(dtype=float), index=index, name="label"
    )
    _, qlib_rankic = calc_ic(prediction, label)
    independent = daily_rankic(
        canonical,
        score_col=score_col,
        target_col=target_col,
        minimum_coverage=minimum_coverage,
    )
    qlib_rankic = pd.Series(qlib_rankic).rename_axis("origin")
    joined = pd.concat(
        [independent.rename("independent"), qlib_rankic.rename("qlib")], axis=1
    ).dropna(how="all")
    differences = np.abs(joined["independent"] - joined["qlib"])
    both_nan = joined["independent"].isna() & joined["qlib"].isna()
    matched = bool(((differences <= tolerance) | both_nan).all())
    maximum = float(differences.max()) if differences.notna().any() else 0.0
    return QlibReconciliation(
        available=True,
        matched=matched,
        maximum_absolute_difference=maximum,
        detail=f"compared {len(joined)} daily rows",
    )


__all__ = [
    "BootstrapInterval",
    "ComparisonResult",
    "DEFAULT_MINIMUM_DAILY_COVERAGE",
    "KEY_COLUMNS",
    "QlibReconciliation",
    "align_model_predictions",
    "bootstrap_sensitivity",
    "compare_predictions",
    "daily_cross_sectional_metrics",
    "daily_rankic",
    "moving_block_bootstrap_mean",
    "reconcile_daily_rankic_with_qlib",
    "spearman_average_rank",
]
