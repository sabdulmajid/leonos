"""Leakage-safe public OHLCV forecasts at month-to-year session horizons.

This module is deliberately separate from the frozen Leonos v1 90/10 contract.
It reuses the same causal OHLCV features, LightGBM model objects, and official
Kronos predictor surface while making the target horizon explicit.  It does not
change the v1 CLI, experiment configuration, or average-close target.

The target at horizon ``H`` is::

    close[t + H] / close[t] - 1

where all offsets are positions in the supplied exchange-session calendar.
Only labels whose explicit ``label_end`` is on or before an as-of date may enter
a split.  No intermediate future close is used in the target.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from .evaluation import daily_rankic, spearman_average_rank
from .features import (
    FEATURE_COLUMNS,
    MAX_FEATURE_LOOKBACK,
    build_ohlcv_features,
)
from .models.kronos import (
    INPUT_COLUMNS,
    INVESTMENT_SAMPLE_COUNT,
    INVESTMENT_TEMPERATURE,
    INVESTMENT_TOP_K,
    INVESTMENT_TOP_P,
    KRONOS_IMPLEMENTATION_REVISION,
    KRONOS_MODEL_ID,
    KRONOS_MODEL_REVISION,
    KRONOS_TOKENIZER_ID,
    KRONOS_TOKENIZER_REVISION,
    ForecastOriginKey,
    ForecastRequest,
    OfficialPredictor,
    derive_effective_seed,
    freeze_official_predictor,
    logical_shard_identity,
    set_global_inference_seed,
)
from .models.lightgbm import (
    DEFAULT_CANDIDATES,
    DEFAULT_SEARCH_CONFIG,
    CandidateConfig,
    CandidateResult,
    LightGBMModel,
    SearchConfig,
    TuningResult,
    _candidate_params,
    _prediction_arrays,
    _rankic_feval,
    _require_lightgbm,
    _supervised_arrays,
    select_candidate,
)
from .targets import _prepare_bars, assert_no_label_overlap, canonical_session_index

# These are exact exchange-session counts, conventionally described as roughly
# one month, one quarter, half a year, and one year of U.S. trading sessions.
SUPPORTED_HORIZONS = (21, 63, 126, 252)
MAX_SEARCH_CONFIGS_PER_HORIZON = 8
TERMINAL_RETURN_TARGET_KIND = "terminal_close_return"

# The pinned public Kronos README declares a 512-token context for base/small.
# More importantly, pinned auto_regressive_inference decodes only
# ``[-max_context:]`` before KronosPredictor tries to return ``pred_len`` rows.
# Consequently the unmodified public predictor cannot return more than 512 rows,
# even though generation itself rolls its attention window for arbitrary steps.
KRONOS_MAX_CONTEXT_SESSIONS = 512
KRONOS_MAX_OUTPUT_SESSIONS = 512
KRONOS_EXPOSES_SAMPLE_PATHS = False
KRONOS_POINT_OUTPUT_SEMANTICS = (
    "official predictor averages decoded sampled trajectories internally; "
    "the public return value is one point path, not samples or calibrated quantiles"
)

HORIZON_CANDIDATES = tuple(DEFAULT_CANDIDATES[:MAX_SEARCH_CONFIGS_PER_HORIZON])
if not 1 <= len(HORIZON_CANDIDATES) <= MAX_SEARCH_CONFIGS_PER_HORIZON:  # pragma: no cover
    raise RuntimeError("horizon LightGBM candidate inventory is empty or exceeds its cap")

_LABEL_COLUMN_NAMES = frozenset(
    {"label", "labels", "target", "targets", "actual", "actual_y", "realized", "y"}
)

KRONOS_PATH_COLUMNS = (
    "model",
    "seed",
    "ticker",
    "origin",
    "target_horizon",
    "forecast_step",
    "input_end",
    "forecast_date",
    "predicted_close",
    "current_close",
    "score",
    "split",
    "logical_shard_id",
    "effective_seed",
    "model_id",
    "model_revision",
    "tokenizer_id",
    "tokenizer_revision",
    "implementation_revision",
    "temperature",
    "top_p",
    "top_k",
    "sample_count",
    "samples_exposed",
    "amount_source",
    "output_semantics",
    "batch_elapsed_seconds",
)


class UnsupportedHorizonError(ValueError):
    """A requested horizon is outside the declared public comparison."""


class HorizonContractError(ValueError):
    """A horizon request or chronology violated the public-data contract."""


@dataclass(frozen=True)
class HorizonSpec:
    """One declared target horizon and its same-instrument input context."""

    horizon_sessions: int
    context_sessions: int = 90

    def __post_init__(self) -> None:
        validate_horizon(self.horizon_sessions)
        if isinstance(self.context_sessions, (bool, np.bool_)) or not isinstance(
            self.context_sessions, Integral
        ):
            raise HorizonContractError("context_sessions must be an integer")
        if not 1 <= int(self.context_sessions) <= KRONOS_MAX_CONTEXT_SESSIONS:
            raise HorizonContractError(
                f"context_sessions must be in [1, {KRONOS_MAX_CONTEXT_SESSIONS}]"
            )


@dataclass(frozen=True)
class KronosHorizonCapabilities:
    """Capabilities of the pinned, unmodified public Kronos-base predictor."""

    model_id: str = KRONOS_MODEL_ID
    implementation_revision: str = KRONOS_IMPLEMENTATION_REVISION
    max_context_sessions: int = KRONOS_MAX_CONTEXT_SESSIONS
    max_output_sessions: int = KRONOS_MAX_OUTPUT_SESSIONS
    supported_horizons: tuple[int, ...] = SUPPORTED_HORIZONS
    exposes_sample_paths: bool = KRONOS_EXPOSES_SAMPLE_PATHS
    output_semantics: str = KRONOS_POINT_OUTPUT_SEMANTICS


@dataclass(frozen=True)
class HorizonKronosConfig:
    """Frozen sampling settings plus one explicit supported horizon."""

    horizon_sessions: int
    context_sessions: int = 90
    seed: int = 42
    temperature: float = INVESTMENT_TEMPERATURE
    top_p: float = INVESTMENT_TOP_P
    top_k: int = INVESTMENT_TOP_K
    sample_count: int = INVESTMENT_SAMPLE_COUNT
    model_id: str = KRONOS_MODEL_ID
    model_revision: str = KRONOS_MODEL_REVISION
    tokenizer_id: str = KRONOS_TOKENIZER_ID
    tokenizer_revision: str = KRONOS_TOKENIZER_REVISION
    implementation_revision: str = KRONOS_IMPLEMENTATION_REVISION

    def __post_init__(self) -> None:
        HorizonSpec(self.horizon_sessions, self.context_sessions)
        fixed = {
            "temperature": (self.temperature, INVESTMENT_TEMPERATURE),
            "top_p": (self.top_p, INVESTMENT_TOP_P),
            "top_k": (self.top_k, INVESTMENT_TOP_K),
            "sample_count": (self.sample_count, INVESTMENT_SAMPLE_COUNT),
            "model_id": (self.model_id, KRONOS_MODEL_ID),
            "model_revision": (self.model_revision, KRONOS_MODEL_REVISION),
            "tokenizer_id": (self.tokenizer_id, KRONOS_TOKENIZER_ID),
            "tokenizer_revision": (self.tokenizer_revision, KRONOS_TOKENIZER_REVISION),
            "implementation_revision": (
                self.implementation_revision,
                KRONOS_IMPLEMENTATION_REVISION,
            ),
        }
        changed = [name for name, (actual, expected) in fixed.items() if actual != expected]
        if changed:
            raise HorizonContractError(
                f"generic horizons retain frozen Kronos identity/sampling; changed: {changed}"
            )
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, Integral):
            raise HorizonContractError("seed must be a non-negative integer")
        if self.seed < 0:
            raise HorizonContractError("seed must be a non-negative integer")


@dataclass
class HorizonTuningResult(TuningResult):
    """Existing LightGBM tuning result with immutable target-horizon provenance."""

    horizon_sessions: int
    label_as_of: pd.Timestamp
    target_kind: str = TERMINAL_RETURN_TARGET_KIND


@dataclass
class HorizonLightGBMModel(LightGBMModel):
    """Existing fitted model fields plus the horizon it was trained to predict."""

    horizon_sessions: int
    label_as_of: pd.Timestamp
    target_kind: str = TERMINAL_RETURN_TARGET_KIND


def validate_horizon(value: int) -> int:
    """Return a declared horizon or fail with the exact supported inventory."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise UnsupportedHorizonError(
            f"horizon must be an integer; supported horizons are {SUPPORTED_HORIZONS}"
        )
    horizon = int(value)
    if horizon not in SUPPORTED_HORIZONS:
        detail = (
            f"; pinned Kronos technical output maximum is {KRONOS_MAX_OUTPUT_SESSIONS}"
            if horizon > KRONOS_MAX_OUTPUT_SESSIONS
            else ""
        )
        raise UnsupportedHorizonError(
            f"unsupported horizon {horizon}; supported horizons are {SUPPORTED_HORIZONS}{detail}"
        )
    return horizon


def validate_horizons(values: Sequence[int] | int) -> tuple[int, ...]:
    """Validate a nonempty, duplicate-free sequence of declared horizons."""

    raw: Sequence[int] = (values,) if isinstance(values, Integral) else values
    horizons = tuple(validate_horizon(value) for value in raw)
    if not horizons:
        raise UnsupportedHorizonError("at least one horizon is required")
    if len(set(horizons)) != len(horizons):
        raise UnsupportedHorizonError("horizons must be unique")
    return horizons


def kronos_horizon_capabilities() -> KronosHorizonCapabilities:
    """Return inspectable limits for the pinned public implementation."""

    return KronosHorizonCapabilities()


def _as_timestamp(value: object, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise HorizonContractError(f"{name} is not a parseable timestamp") from exc
    if pd.isna(timestamp):
        raise HorizonContractError(f"{name} is missing")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.normalize()


def _as_date_series(values: pd.Series, name: str) -> pd.Series:
    try:
        return pd.to_datetime(values, utc=True, errors="raise").dt.tz_convert(None).dt.normalize()
    except (TypeError, ValueError) as exc:
        raise HorizonContractError(f"{name} contains invalid timestamps") from exc


def build_horizon_targets(
    bars: pd.DataFrame,
    calendar: Sequence[object],
    *,
    horizons: Sequence[int] | int = SUPPORTED_HORIZONS,
    context_sessions: int = 90,
    label_as_of: object | None = None,
    ticker_col: str = "ticker",
    session_col: str = "session",
    close_col: str = "close",
    origin_dates: Sequence[object] | None = None,
    include_incomplete: bool = False,
) -> pd.DataFrame:
    """Build calendar-exact, matured terminal-close return labels.

    ``label_as_of`` is the last session whose close may be used in a label.  When
    omitted it defaults to the last supplied calendar session.  Rows ending after
    that date are excluded even if later bars happen to be present in ``bars``.
    """

    declared = validate_horizons(horizons)
    if isinstance(context_sessions, (bool, np.bool_)) or not isinstance(context_sessions, Integral):
        raise HorizonContractError("context_sessions must be a positive integer")
    if not 1 <= int(context_sessions) <= KRONOS_MAX_CONTEXT_SESSIONS:
        raise HorizonContractError(
            f"context_sessions must be in [1, {KRONOS_MAX_CONTEXT_SESSIONS}]"
        )
    sessions = canonical_session_index(calendar)
    if sessions.empty:
        raise HorizonContractError("calendar is empty")
    as_of = sessions[-1] if label_as_of is None else _as_timestamp(label_as_of, "label_as_of")
    if len(sessions) < int(context_sessions) + max(declared):
        raise HorizonContractError("calendar is shorter than the requested context plus horizon")
    if close_col not in bars.columns:
        raise HorizonContractError(f"bars are missing close column {close_col!r}")
    prepared = _prepare_bars(
        bars,
        sessions,
        ticker_col=ticker_col,
        session_col=session_col,
    )
    prepared[close_col] = pd.to_numeric(prepared[close_col], errors="coerce")

    requested_origins: set[pd.Timestamp] | None = None
    if origin_dates is not None:
        requested_origins = set(canonical_session_index(origin_dates))
        absent = requested_origins.difference(sessions)
        if absent:
            rendered = sorted(absent)[:5]
            raise HorizonContractError(
                f"requested origins are absent from exchange calendar: {rendered}"
            )

    frames: list[pd.DataFrame] = []
    for horizon in declared:
        records: list[dict[str, object]] = []
        first_position = int(context_sessions) - 1
        last_position = len(sessions) - horizon - 1
        for ticker, group in prepared.groupby(ticker_col, sort=True, observed=True):
            close = (
                group.set_index(session_col)[close_col]
                .reindex(sessions)
                .to_numpy(dtype=np.float64, na_value=np.nan)
            )
            valid = np.isfinite(close) & (close > 0.0)
            for position in range(first_position, last_position + 1):
                origin = sessions[position]
                label_end = sessions[position + horizon]
                if label_end > as_of:
                    continue
                if requested_origins is not None and origin not in requested_origins:
                    continue
                context_start_position = position - int(context_sessions) + 1
                context_complete = bool(valid[context_start_position : position + 1].all())
                # A terminal-return label needs the exact exchange-session endpoint,
                # not every intermediate future bar.
                label_complete = bool(valid[position + horizon])
                eligible = context_complete and label_complete
                if not eligible and not include_incomplete:
                    continue
                current_close = float(close[position]) if valid[position] else np.nan
                target = (
                    float(close[position + horizon] / current_close - 1.0) if eligible else np.nan
                )
                records.append(
                    {
                        "ticker": str(ticker),
                        "origin": origin,
                        "context_start": sessions[context_start_position],
                        "input_end": origin,
                        "execution_session": sessions[position + 1],
                        "forecast_start": sessions[position + 1],
                        "label_end": label_end,
                        "current_close": current_close,
                        "target": target,
                        "context_complete": context_complete,
                        "label_complete": label_complete,
                        "context_sessions": int(context_sessions),
                        "horizon_sessions": horizon,
                        "target_kind": TERMINAL_RETURN_TARGET_KIND,
                    }
                )
        columns = [
            "ticker",
            "origin",
            "context_start",
            "input_end",
            "execution_session",
            "forecast_start",
            "label_end",
            "current_close",
            "target",
            "context_complete",
            "label_complete",
            "context_sessions",
            "horizon_sessions",
            "target_kind",
        ]
        targets = pd.DataFrame.from_records(records, columns=columns)
        targets["label_as_of"] = as_of
        frames.append(targets)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not result.empty:
        result = result.sort_values(
            ["horizon_sessions", "origin", "ticker"], kind="stable"
        ).reset_index(drop=True)
    return result


def build_horizon_supervised_panel(
    bars: pd.DataFrame,
    calendar: Sequence[object],
    *,
    horizons: Sequence[int] | int = SUPPORTED_HORIZONS,
    context_sessions: int = 90,
    label_as_of: object | None = None,
    ticker_col: str = "ticker",
    session_col: str = "session",
    column_map: dict[str, str] | None = None,
    origin_dates: Sequence[object] | None = None,
) -> pd.DataFrame:
    """Compute the shared causal feature matrix once and join matured labels.

    Features are calculated only from bars through each origin.  The current
    Alpha158-style adapter has a 90-session ceiling, so its context must stay
    inside ``[MAX_FEATURE_LOOKBACK, 90]`` even though Kronos itself supports 512.
    """

    if not MAX_FEATURE_LOOKBACK <= int(context_sessions) <= 90:
        raise HorizonContractError(
            f"LightGBM feature context must be in [{MAX_FEATURE_LOOKBACK}, 90]"
        )
    sessions = canonical_session_index(calendar)
    labels = build_horizon_targets(
        bars,
        sessions,
        horizons=horizons,
        context_sessions=context_sessions,
        label_as_of=label_as_of,
        ticker_col=ticker_col,
        session_col=session_col,
        close_col=(column_map or {}).get("close", "close"),
        origin_dates=origin_dates,
    )
    if labels.empty:
        return labels.assign(**{column: pd.Series(dtype=float) for column in FEATURE_COLUMNS})
    keys = labels.loc[:, ["ticker", "origin"]].drop_duplicates().reset_index(drop=True)
    features = build_ohlcv_features(
        bars,
        sessions,
        keys=keys,
        context_sessions=context_sessions,
        ticker_col=ticker_col,
        session_col=session_col,
        column_map=column_map,
    )
    feature_values = features.loc[:, ["ticker", "origin", *FEATURE_COLUMNS]]
    result = labels.merge(
        feature_values,
        on=["ticker", "origin"],
        how="inner",
        validate="many_to_one",
    )
    return result.sort_values(["horizon_sessions", "origin", "ticker"], kind="stable").reset_index(
        drop=True
    )


def matured_labels(frame: pd.DataFrame, *, label_as_of: object) -> pd.DataFrame:
    """Select only rows whose complete label window was observable as of a date."""

    if "label_end" not in frame.columns:
        raise HorizonContractError("labels are missing label_end")
    as_of = _as_timestamp(label_as_of, "label_as_of")
    out = frame.copy()
    out["label_end"] = _as_date_series(out["label_end"], "label_end")
    return out.loc[out["label_end"] <= as_of].reset_index(drop=True)


def _validate_single_horizon(frame: pd.DataFrame, expected: int | None = None) -> int:
    if "horizon_sessions" not in frame.columns:
        if expected is None:
            raise HorizonContractError("frame is missing horizon_sessions")
        return validate_horizon(expected)
    numeric = pd.to_numeric(frame["horizon_sessions"], errors="coerce")
    unique = numeric.dropna().unique()
    if len(unique) != 1 or not np.isfinite(unique[0]) or int(unique[0]) != unique[0]:
        raise HorizonContractError("a model split must contain exactly one integer horizon")
    horizon = validate_horizon(int(unique[0]))
    if expected is not None and horizon != validate_horizon(expected):
        raise HorizonContractError(
            f"frame horizon {horizon} does not match requested horizon {expected}"
        )
    if "target_kind" in frame.columns:
        target_kind = frame["target_kind"].astype("string")
        if target_kind.isna().any() or not target_kind.eq(TERMINAL_RETURN_TARGET_KIND).all():
            raise HorizonContractError(
                f"horizon labels must use target_kind={TERMINAL_RETURN_TARGET_KIND!r}"
            )
    return horizon


def chronological_train_validation_split(
    frame: pd.DataFrame,
    *,
    validation_origin_start: object,
    validation_origin_end: object,
    label_as_of: object,
    horizon_sessions: int | None = None,
    require_nonempty: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a purged chronological train/validation split.

    Training rows are retained only when their ``label_end`` is strictly before
    the first validation origin.  Validation rows must have origins in the
    requested inclusive window and labels matured by ``label_as_of``.
    """

    required = {"ticker", "origin", "label_end", "target"}
    missing = required.difference(frame.columns)
    if missing:
        raise HorizonContractError(f"supervised frame missing columns: {sorted(missing)}")
    horizon = _validate_single_horizon(frame, horizon_sessions)
    start = _as_timestamp(validation_origin_start, "validation_origin_start")
    end = _as_timestamp(validation_origin_end, "validation_origin_end")
    as_of = _as_timestamp(label_as_of, "label_as_of")
    if start > end:
        raise HorizonContractError("validation_origin_start is after validation_origin_end")

    out = frame.copy()
    out["ticker"] = out["ticker"].astype("string")
    out["origin"] = _as_date_series(out["origin"], "origin")
    out["label_end"] = _as_date_series(out["label_end"], "label_end")
    if out["ticker"].isna().any():
        raise HorizonContractError("ticker keys contain missing values")
    if out.duplicated(["ticker", "origin", "horizon_sessions"]).any():
        raise HorizonContractError("supervised frame contains duplicate horizon keys")
    target = pd.to_numeric(out["target"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(target).all():
        raise HorizonContractError("split input must contain only finite eligible targets")
    if (out["label_end"] <= out["origin"]).any():
        raise HorizonContractError("label_end must follow its forecast origin")

    out = matured_labels(out, label_as_of=as_of)
    train = out.loc[(out["origin"] < start) & (out["label_end"] < start)].copy()
    validation = out.loc[(out["origin"] >= start) & (out["origin"] <= end)].copy()
    train["split"] = "training"
    validation["split"] = "validation"
    train = train.sort_values(["origin", "ticker"], kind="stable").reset_index(drop=True)
    validation = validation.sort_values(["origin", "ticker"], kind="stable").reset_index(drop=True)
    if require_nonempty and (train.empty or validation.empty):
        raise HorizonContractError(
            f"horizon {horizon} produced an empty chronological training or validation split"
        )
    assert_no_label_overlap(train, validation)
    return train, validation


def chronological_splits_by_horizon(
    frame: pd.DataFrame,
    *,
    validation_origin_start: object,
    validation_origin_end: object,
    label_as_of: object,
    horizons: Sequence[int] | int = SUPPORTED_HORIZONS,
    require_nonempty: bool = True,
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """Apply the same predeclared chronological window independently per horizon."""

    result: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for horizon in validate_horizons(horizons):
        subset = frame.loc[pd.to_numeric(frame["horizon_sessions"]) == horizon].copy()
        result[horizon] = chronological_train_validation_split(
            subset,
            validation_origin_start=validation_origin_start,
            validation_origin_end=validation_origin_end,
            label_as_of=label_as_of,
            horizon_sessions=horizon,
            require_nonempty=require_nonempty,
        )
    return result


def validate_search_candidates(
    candidates: Sequence[CandidateConfig],
) -> tuple[CandidateConfig, ...]:
    """Freeze a nonempty search with no more than eight configurations."""

    bounded = tuple(candidates)
    if not 1 <= len(bounded) <= MAX_SEARCH_CONFIGS_PER_HORIZON:
        raise HorizonContractError(
            "each horizon LightGBM search must contain between one and "
            f"{MAX_SEARCH_CONFIGS_PER_HORIZON} candidates"
        )
    identifiers = [candidate.candidate_id for candidate in bounded]
    if len(set(identifiers)) != len(identifiers):
        raise HorizonContractError("LightGBM candidate identifiers must be unique")
    return bounded


def validate_horizon_search_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    horizon_sessions: int,
    label_as_of: object,
) -> None:
    """Fail closed on mixed horizons, immature labels, or chronological overlap."""

    horizon = validate_horizon(horizon_sessions)
    for name, values in (("training", train), ("validation", validation)):
        required = {"ticker", "origin", "label_end", "target"}
        missing = required.difference(values.columns)
        if missing:
            raise HorizonContractError(f"{name} frame missing columns: {sorted(missing)}")
        if values.empty:
            raise HorizonContractError(f"{name} frame is empty")
        _validate_single_horizon(values, horizon)
        if values.duplicated(["ticker", "origin"]).any():
            raise HorizonContractError(f"{name} frame contains duplicate keys")
        target = pd.to_numeric(values["target"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(target).all():
            raise HorizonContractError(f"{name} frame contains non-finite targets")

    train_origin = _as_date_series(train["origin"], "training origin")
    train_end = _as_date_series(train["label_end"], "training label_end")
    validation_origin = _as_date_series(validation["origin"], "validation origin")
    validation_end = _as_date_series(validation["label_end"], "validation label_end")
    as_of = _as_timestamp(label_as_of, "label_as_of")
    if (train_end > as_of).any() or (validation_end > as_of).any():
        raise HorizonContractError("search includes a label that had not matured by label_as_of")
    if train_origin.max() >= validation_origin.min():
        raise HorizonContractError("training origins must precede validation origins")
    if train_end.max() >= validation_origin.min():
        raise HorizonContractError("training labels must end before the first validation origin")
    overlap = train[["ticker", "origin"]].merge(
        validation[["ticker", "origin"]], on=["ticker", "origin"], how="inner"
    )
    if not overlap.empty:
        raise HorizonContractError("training and validation keys overlap")


def tune_horizon_lightgbm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    horizon_sessions: int,
    label_as_of: object,
    candidates: Sequence[CandidateConfig] = HORIZON_CANDIDATES,
    config: SearchConfig = DEFAULT_SEARCH_CONFIG,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    target_col: str = "target",
) -> HorizonTuningResult:
    """Tune one pooled LightGBM with at most eight validation-only candidates."""

    bounded = validate_search_candidates(candidates)
    validate_horizon_search_splits(
        train,
        validation,
        horizon_sessions=horizon_sessions,
        label_as_of=label_as_of,
    )
    columns = tuple(feature_columns)
    train_x, train_y, _ = _supervised_arrays(train, columns, target_col=target_col)
    validation_x, validation_y, validation_rows = _supervised_arrays(
        validation, columns, target_col=target_col
    )
    lgb = _require_lightgbm()
    train_dataset = lgb.Dataset(
        train_x,
        label=train_y,
        feature_name=list(columns),
        free_raw_data=False,
    )
    validation_dataset = lgb.Dataset(
        validation_x,
        label=validation_y,
        reference=train_dataset,
        feature_name=list(columns),
        free_raw_data=False,
    )
    feval = _rankic_feval(validation_rows["origin"], minimum_coverage=config.minimum_daily_coverage)
    outcomes: list[CandidateResult] = []
    boosters: dict[str, Any] = {}
    for candidate in bounded:
        started = time.perf_counter()
        try:
            booster = lgb.train(
                _candidate_params(candidate, config, config.seed),
                train_dataset,
                num_boost_round=config.max_boost_rounds,
                valid_sets=[validation_dataset],
                valid_names=["validation"],
                feval=feval,
                callbacks=[
                    lgb.early_stopping(config.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            best_iteration = int(booster.best_iteration or config.max_boost_rounds)
            prediction = np.asarray(
                booster.predict(validation_x, num_iteration=best_iteration), dtype=float
            )
            if prediction.shape != (len(validation_x),) or not np.isfinite(prediction).all():
                raise ValueError("candidate returned malformed or non-finite validation scores")
            scored = validation_rows.loc[:, ["ticker", "origin", target_col]].rename(
                columns={target_col: "target"}
            )
            scored["score"] = prediction
            per_date = daily_rankic(
                scored,
                minimum_coverage=config.minimum_daily_coverage,
            )
            defined = per_date[np.isfinite(per_date)]
            if defined.empty:
                raise ValueError("candidate produced no defined validation daily RankIC")
            daily_mae = (
                scored.assign(abs_error=np.abs(scored["score"] - scored["target"]))
                .groupby("origin", observed=True)["abs_error"]
                .mean()
            )
            outcome = CandidateResult(
                candidate=candidate,
                status="ok",
                best_iteration=best_iteration,
                validation_mean_daily_rankic=float(defined.mean()),
                validation_mean_daily_mae=float(daily_mae.mean()),
                fit_seconds=float(time.perf_counter() - started),
            )
            boosters[candidate.candidate_id] = booster
        except Exception as exc:
            outcome = CandidateResult(
                candidate=candidate,
                status="failed",
                best_iteration=0,
                validation_mean_daily_rankic=np.nan,
                validation_mean_daily_mae=np.nan,
                fit_seconds=float(time.perf_counter() - started),
                error=f"{type(exc).__name__}: {exc}",
            )
        outcomes.append(outcome)
    selected = select_candidate(outcomes, tie_tolerance=config.tie_tolerance)
    return HorizonTuningResult(
        selected=selected,
        candidates=tuple(outcomes),
        validation_booster=boosters[selected.candidate.candidate_id],
        feature_columns=columns,
        search_config=config,
        horizon_sessions=validate_horizon(horizon_sessions),
        label_as_of=_as_timestamp(label_as_of, "label_as_of"),
    )


def fit_final_horizon_lightgbm(
    refit: pd.DataFrame,
    tuning: TuningResult,
    *,
    horizon_sessions: int,
    label_as_of: object,
    seed: int | None = None,
    target_col: str = "target",
) -> HorizonLightGBMModel:
    """Refit a selected horizon model using only labels matured by ``label_as_of``."""

    horizon = validate_horizon(horizon_sessions)
    tuned_horizon = getattr(tuning, "horizon_sessions", horizon)
    if tuned_horizon != horizon:
        raise HorizonContractError(
            f"tuning horizon {tuned_horizon} does not match refit horizon {horizon}"
        )
    if refit.empty:
        raise HorizonContractError("final refit frame is empty")
    _validate_single_horizon(refit, horizon)
    if "label_end" not in refit:
        raise HorizonContractError("final refit frame is missing label_end")
    if target_col not in refit:
        raise HorizonContractError(f"final refit frame is missing {target_col!r}")
    if refit.duplicated(["ticker", "origin"]).any():
        raise HorizonContractError("final refit frame contains duplicate keys")
    label_end = _as_date_series(refit["label_end"], "final refit label_end")
    origin = _as_date_series(refit["origin"], "final refit origin")
    as_of = _as_timestamp(label_as_of, "label_as_of")
    if (label_end > as_of).any():
        raise HorizonContractError("final refit includes an immature label")
    if (label_end <= origin).any():
        raise HorizonContractError("final refit label_end must follow its origin")
    target = pd.to_numeric(refit[target_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(target).all():
        raise HorizonContractError("final refit includes non-finite targets")
    train_x, train_y, _ = _supervised_arrays(refit, tuning.feature_columns, target_col=target_col)
    rounds = int(tuning.selected.best_iteration)
    if rounds < 1:
        raise HorizonContractError("selected candidate has no valid boosting round count")
    selected_seed = tuning.search_config.seed if seed is None else int(seed)
    lgb = _require_lightgbm()
    dataset = lgb.Dataset(
        train_x,
        label=train_y,
        feature_name=list(tuning.feature_columns),
        free_raw_data=False,
    )
    started = time.perf_counter()
    booster = lgb.train(
        _candidate_params(tuning.selected.candidate, tuning.search_config, selected_seed),
        dataset,
        num_boost_round=rounds,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    return HorizonLightGBMModel(
        booster=booster,
        candidate=tuning.selected.candidate,
        feature_columns=tuning.feature_columns,
        seed=selected_seed,
        boosting_rounds=rounds,
        training_rows=len(train_x),
        training_label_end_max=pd.Timestamp(label_end.max()),
        fit_seconds=float(time.perf_counter() - started),
        horizon_sessions=horizon,
        label_as_of=as_of,
    )


def predict_horizon_lightgbm(
    model: LightGBMModel,
    features: pd.DataFrame,
    *,
    horizon_sessions: int,
    model_name: str = "lightgbm",
) -> pd.DataFrame:
    """Return origin-level horizon scores without copying labels into predictions."""

    horizon = validate_horizon(horizon_sessions)
    trained_horizon = getattr(model, "horizon_sessions", horizon)
    if trained_horizon != horizon:
        raise HorizonContractError(
            f"model horizon {trained_horizon} does not match prediction horizon {horizon}"
        )
    missing = {"ticker", "origin"}.difference(features.columns)
    if missing:
        raise HorizonContractError(f"prediction features missing keys: {sorted(missing)}")
    if features.duplicated(["ticker", "origin"]).any():
        raise HorizonContractError("prediction feature keys are duplicated")
    matrix = _prediction_arrays(features, model.feature_columns)
    prediction = np.asarray(
        model.booster.predict(matrix, num_iteration=model.boosting_rounds), dtype=float
    )
    if prediction.shape != (len(features),):
        raise HorizonContractError(
            f"LightGBM returned shape {prediction.shape}, expected {(len(features),)}"
        )
    result = features.loc[:, ["ticker", "origin"]].copy()
    result["origin"] = _as_date_series(result["origin"], "prediction origin")
    result.insert(0, "seed", model.seed)
    result.insert(0, "model", model_name)
    result["horizon"] = horizon
    result["score"] = prediction
    result["status"] = np.where(np.isfinite(prediction), "ok", "nonfinite")
    return (
        result.loc[:, ["model", "seed", "ticker", "origin", "horizon", "score", "status"]]
        .sort_values(["origin", "ticker"], kind="stable")
        .reset_index(drop=True)
    )


def _strict_datetime_index(values: Sequence[Any], name: str) -> pd.DatetimeIndex:
    try:
        result = pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="raise"))
        result = result.tz_convert(None).normalize()
    except (TypeError, ValueError) as exc:
        raise HorizonContractError(f"{name} must contain parseable timestamps") from exc
    if result.hasnans:
        raise HorizonContractError(f"{name} contains missing timestamps")
    if result.has_duplicates or not result.is_monotonic_increasing:
        raise HorizonContractError(f"{name} must be strictly increasing and unique")
    return result


def _validate_horizon_request(request: ForecastRequest, config: HorizonKronosConfig) -> None:
    if not request.ticker or not str(request.ticker).strip():
        raise HorizonContractError("ticker must be non-empty")
    if len(request.history) != config.context_sessions:
        raise HorizonContractError(
            f"history must contain exactly {config.context_sessions} completed sessions"
        )
    missing = sorted(set(INPUT_COLUMNS).difference(request.history.columns))
    if missing:
        raise HorizonContractError(f"history is missing required OHLCV columns: {missing}")
    history_dates = _strict_datetime_index(request.history_dates, "history_dates")
    forecast_dates = _strict_datetime_index(request.forecast_dates, "forecast_dates")
    if len(history_dates) != config.context_sessions or len(history_dates) != len(request.history):
        raise HorizonContractError("history_dates must align one-for-one with history")
    if len(forecast_dates) != config.horizon_sessions:
        raise HorizonContractError(
            f"forecast_dates must contain exactly {config.horizon_sessions} sessions"
        )
    origin = _as_timestamp(request.origin, "origin")
    if origin != history_dates[-1]:
        raise HorizonContractError("origin must equal the last completed history session")
    if forecast_dates[0] <= origin:
        raise HorizonContractError("all forecast timestamps must follow the origin")
    try:
        values = request.history[list(INPUT_COLUMNS)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise HorizonContractError("historical OHLCV must be numeric") from exc
    if not np.isfinite(values).all():
        raise HorizonContractError("historical OHLCV contains non-finite values")
    if (request.history[["open", "high", "low", "close"]].to_numpy(dtype=float) <= 0).any():
        raise HorizonContractError("historical prices must be positive")
    if (request.history["volume"].to_numpy(dtype=float) < 0).any():
        raise HorizonContractError("historical volume must be non-negative")


class HorizonKronosAdapter:
    """Point-path adapter for the declared horizons using official Kronos unchanged."""

    def __init__(
        self,
        predictor: OfficialPredictor,
        config: HorizonKronosConfig,
        *,
        seed_setter: Callable[[int], None] = set_global_inference_seed,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.predictor = predictor
        self.config = config
        self._seed_setter = seed_setter
        self._clock = clock
        predictor_limit = getattr(predictor, "max_context", KRONOS_MAX_CONTEXT_SESSIONS)
        try:
            predictor_limit = int(predictor_limit)
        except (TypeError, ValueError) as exc:
            raise HorizonContractError("official predictor max_context is invalid") from exc
        if predictor_limit > KRONOS_MAX_CONTEXT_SESSIONS:
            predictor_limit = KRONOS_MAX_CONTEXT_SESSIONS
        if config.context_sessions > predictor_limit:
            raise HorizonContractError(
                f"context {config.context_sessions} exceeds predictor max_context {predictor_limit}"
            )
        if config.horizon_sessions > predictor_limit:
            raise UnsupportedHorizonError(
                "horizon "
                f"{config.horizon_sessions} exceeds predictor output limit {predictor_limit}"
            )
        self.predictor_max_context = predictor_limit
        freeze_official_predictor(self.predictor)

    def predict_batch(
        self,
        requests: Iterable[ForecastRequest],
        *,
        split: str,
        return_samples: bool = False,
    ) -> pd.DataFrame:
        """Return a label-free point path; the public API cannot expose samples."""

        if return_samples:
            raise HorizonContractError(
                "the pinned official predictor averages samples internally and exposes only a "
                "point path"
            )
        if not split or not str(split).strip():
            raise HorizonContractError("split must be non-empty")
        ordered = list(requests)
        for request in ordered:
            _validate_horizon_request(request, self.config)
        ordered.sort(key=lambda item: (_as_timestamp(item.origin, "origin"), str(item.ticker)))
        keys = [
            ForecastOriginKey(str(item.ticker), _as_timestamp(item.origin, "origin"))
            for item in ordered
        ]
        canonical_keys = [key.canonical() for key in keys]
        if len(canonical_keys) != len(set(canonical_keys)):
            raise HorizonContractError("duplicate (ticker, origin) forecast requests")
        if not ordered:
            return pd.DataFrame(columns=KRONOS_PATH_COLUMNS)

        seed_split = f"{split}:h{self.config.horizon_sessions}"
        shard_id = logical_shard_identity(seed_split, self.config.seed, keys)
        effective_seed = derive_effective_seed(self.config.seed, shard_id)
        self._seed_setter(effective_seed)

        frames: list[pd.DataFrame] = []
        history_timestamps: list[pd.Series] = []
        forecast_timestamps: list[pd.Series] = []
        amount_sources: list[str] = []
        for request in ordered:
            columns = list(INPUT_COLUMNS)
            # This study is OHLCV-only.  If an unrelated ``amount`` column is
            # present, omit it and let upstream derive its documented OHLCV proxy.
            amount_sources.append("estimated_ohlcv_proxy_by_official_predictor")
            frames.append(request.history.loc[:, columns].copy())
            history_timestamps.append(
                pd.Series(_strict_datetime_index(request.history_dates, "history_dates"))
            )
            forecast_timestamps.append(
                pd.Series(_strict_datetime_index(request.forecast_dates, "forecast_dates"))
            )

        try:
            import torch

            inference_context = torch.inference_mode()
        except ImportError:  # pragma: no cover - injected CPU mocks need no torch
            inference_context = nullcontext()
        started = self._clock()
        with inference_context:
            predictions = self.predictor.predict_batch(
                df_list=frames,
                x_timestamp_list=history_timestamps,
                y_timestamp_list=forecast_timestamps,
                pred_len=self.config.horizon_sessions,
                T=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                sample_count=self.config.sample_count,
                verbose=False,
            )
        elapsed = self._clock() - started
        if not isinstance(predictions, (list, tuple)) or len(predictions) != len(ordered):
            raise HorizonContractError("official predictor returned the wrong batch cardinality")

        rows: list[dict[str, object]] = []
        for request, prediction, dates, amount_source in zip(
            ordered, predictions, forecast_timestamps, amount_sources, strict=True
        ):
            if not isinstance(prediction, pd.DataFrame) or "close" not in prediction:
                raise HorizonContractError("official predictor must return a DataFrame with close")
            actual_dates = _strict_datetime_index(prediction.index, "predicted forecast dates")
            expected_dates = pd.DatetimeIndex(dates)
            closes = pd.to_numeric(prediction["close"], errors="coerce").to_numpy(dtype=float)
            if len(closes) != self.config.horizon_sessions:
                raise HorizonContractError("official predictor returned the wrong forecast horizon")
            if not actual_dates.equals(expected_dates):
                raise HorizonContractError("official predictor changed forecast-date alignment")
            if not np.isfinite(closes).all() or (closes <= 0).any():
                raise HorizonContractError(
                    "official predictor returned non-finite/non-positive closes"
                )
            current_close = float(request.history["close"].iloc[-1])
            score = float(closes[-1] / current_close - 1.0)
            for step, (forecast_date, predicted_close) in enumerate(
                zip(expected_dates, closes, strict=True), start=1
            ):
                rows.append(
                    {
                        "model": "kronos",
                        "seed": self.config.seed,
                        "ticker": str(request.ticker),
                        "origin": _as_timestamp(request.origin, "origin"),
                        "target_horizon": self.config.horizon_sessions,
                        "forecast_step": step,
                        "input_end": _as_timestamp(request.origin, "origin"),
                        "forecast_date": forecast_date,
                        "predicted_close": float(predicted_close),
                        "current_close": current_close,
                        "score": score,
                        "split": str(split),
                        "logical_shard_id": shard_id,
                        "effective_seed": effective_seed,
                        "model_id": self.config.model_id,
                        "model_revision": self.config.model_revision,
                        "tokenizer_id": self.config.tokenizer_id,
                        "tokenizer_revision": self.config.tokenizer_revision,
                        "implementation_revision": self.config.implementation_revision,
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                        "top_k": self.config.top_k,
                        "sample_count": self.config.sample_count,
                        "samples_exposed": False,
                        "amount_source": amount_source,
                        "output_semantics": KRONOS_POINT_OUTPUT_SEMANTICS,
                        "batch_elapsed_seconds": float(elapsed),
                    }
                )
        result = pd.DataFrame.from_records(rows, columns=KRONOS_PATH_COLUMNS)
        validate_kronos_horizon_paths(result)
        return result


def validate_kronos_horizon_paths(frame: pd.DataFrame) -> None:
    """Validate complete, immutable, label-free generic Kronos point paths."""

    missing = set(KRONOS_PATH_COLUMNS).difference(frame.columns)
    if missing:
        raise HorizonContractError(f"Kronos paths missing columns: {sorted(missing)}")
    forbidden = _LABEL_COLUMN_NAMES.intersection(frame.columns)
    if forbidden:
        raise HorizonContractError(f"Kronos paths must remain label-free: {sorted(forbidden)}")
    if frame.empty:
        return
    keys = ["model", "seed", "ticker", "origin", "target_horizon", "forecast_step"]
    if frame.duplicated(keys).any():
        raise HorizonContractError("Kronos paths contain duplicate forecast-step keys")
    if not frame["model"].eq("kronos").all():
        raise HorizonContractError("Kronos path model provenance is mixed")
    if frame["samples_exposed"].astype(bool).any():
        raise HorizonContractError("official point paths cannot claim an exposed sample axis")
    if not frame["output_semantics"].eq(KRONOS_POINT_OUTPUT_SEMANTICS).all():
        raise HorizonContractError("Kronos output semantics are mixed or unsupported")

    working = frame.copy()
    working["origin"] = _as_date_series(working["origin"], "origin")
    working["input_end"] = _as_date_series(working["input_end"], "input_end")
    working["forecast_date"] = _as_date_series(working["forecast_date"], "forecast_date")
    if not working["input_end"].equals(working["origin"]):
        raise HorizonContractError("Kronos input_end must equal its forecast origin")
    for group_key, group in working.groupby(
        ["model", "seed", "ticker", "origin", "target_horizon"],
        sort=False,
        observed=True,
    ):
        raw_horizon = group_key[-1]
        try:
            numeric_horizon = float(raw_horizon)
        except (TypeError, ValueError) as exc:
            raise HorizonContractError("Kronos target_horizon is not an integer") from exc
        if not np.isfinite(numeric_horizon) or not numeric_horizon.is_integer():
            raise HorizonContractError("Kronos target_horizon is not an integer")
        horizon = validate_horizon(int(numeric_horizon))
        ordered = group.sort_values("forecast_step", kind="stable")
        steps = pd.to_numeric(ordered["forecast_step"], errors="coerce").tolist()
        if steps != list(range(1, horizon + 1)):
            raise HorizonContractError("Kronos point path is incomplete or out of order")
        dates = pd.DatetimeIndex(ordered["forecast_date"])
        if dates.has_duplicates or not dates.is_monotonic_increasing or dates[0] <= group_key[3]:
            raise HorizonContractError("Kronos forecast dates are not strictly after the origin")
        closes = pd.to_numeric(ordered["predicted_close"], errors="coerce").to_numpy(dtype=float)
        current = pd.to_numeric(ordered["current_close"], errors="coerce").to_numpy(dtype=float)
        scores = pd.to_numeric(ordered["score"], errors="coerce").to_numpy(dtype=float)
        if (
            not np.isfinite(closes).all()
            or not np.isfinite(current).all()
            or not np.isfinite(scores).all()
            or (closes <= 0).any()
            or (current <= 0).any()
        ):
            raise HorizonContractError("Kronos point path contains invalid numeric values")
        if not np.allclose(current, current[0], rtol=0.0, atol=0.0):
            raise HorizonContractError("Kronos point path current_close is inconsistent")
        expected_score = closes[-1] / current[0] - 1.0
        if not np.allclose(scores, expected_score, rtol=1e-12, atol=1e-12):
            raise HorizonContractError("Kronos point path score is inconsistent with its closes")


def collapse_kronos_horizon_scores(paths: pd.DataFrame) -> pd.DataFrame:
    """Collapse validated point paths to the LightGBM-compatible score schema."""

    validate_kronos_horizon_paths(paths)
    if paths.empty:
        return pd.DataFrame(
            columns=["model", "seed", "ticker", "origin", "horizon", "score", "status"]
        )
    scores = paths.loc[
        :, ["model", "seed", "ticker", "origin", "target_horizon", "score"]
    ].drop_duplicates()
    if scores.duplicated(["model", "seed", "ticker", "origin", "target_horizon"]).any():
        raise HorizonContractError("Kronos path rows contain inconsistent scores")
    scores = scores.rename(columns={"target_horizon": "horizon"})
    scores["status"] = "ok"
    return scores.sort_values(["horizon", "origin", "ticker"], kind="stable").reset_index(drop=True)


def validation_metrics_by_instrument(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Report validation coverage and errors separately for every instrument.

    Metrics are diagnostics, not additional model-selection objectives.  Missing
    or failed forecasts remain visible through ``eligible_labels``, ``scored``,
    and ``coverage``.  Constant-score Pearson/Spearman correlations are ``NA``.
    """

    prediction_required = {"model", "seed", "ticker", "origin", "horizon", "score", "status"}
    label_required = {"ticker", "origin", "horizon_sessions", "target"}
    missing_predictions = prediction_required.difference(predictions.columns)
    missing_labels = label_required.difference(labels.columns)
    if missing_predictions:
        raise HorizonContractError(f"predictions missing columns: {sorted(missing_predictions)}")
    if missing_labels:
        raise HorizonContractError(f"labels missing columns: {sorted(missing_labels)}")
    if predictions.empty:
        raise HorizonContractError("predictions are empty")
    if labels.empty:
        raise HorizonContractError("labels are empty")

    pred = predictions.loc[:, list(prediction_required)].copy()
    truth = labels.loc[:, list(label_required)].copy()
    pred["ticker"] = pred["ticker"].astype("string")
    truth["ticker"] = truth["ticker"].astype("string")
    pred["origin"] = _as_date_series(pred["origin"], "prediction origin")
    truth["origin"] = _as_date_series(truth["origin"], "label origin")
    pred["horizon"] = pd.to_numeric(pred["horizon"], errors="coerce")
    truth["horizon"] = pd.to_numeric(truth.pop("horizon_sessions"), errors="coerce")
    prediction_horizons = pred["horizon"].to_numpy(dtype=float)
    label_horizons = truth["horizon"].to_numpy(dtype=float)
    if (
        not np.isfinite(prediction_horizons).all()
        or not np.isfinite(label_horizons).all()
        or not np.equal(prediction_horizons, np.floor(prediction_horizons)).all()
        or not np.equal(label_horizons, np.floor(label_horizons)).all()
    ):
        raise HorizonContractError("prediction and label horizons must be finite integers")
    for horizon in pred["horizon"].unique():
        validate_horizon(int(horizon))
    for horizon in truth["horizon"].unique():
        validate_horizon(int(horizon))
    if "target_kind" in labels.columns:
        kinds = labels["target_kind"].astype("string")
        if kinds.isna().any() or not kinds.eq(TERMINAL_RETURN_TARGET_KIND).all():
            raise HorizonContractError(
                f"validation labels must use target_kind={TERMINAL_RETURN_TARGET_KIND!r}"
            )
    if pred.duplicated(["model", "seed", "ticker", "origin", "horizon"]).any():
        raise HorizonContractError("predictions contain duplicate full keys")
    if truth.duplicated(["ticker", "origin", "horizon"]).any():
        raise HorizonContractError("labels contain duplicate horizon keys")
    truth["target"] = pd.to_numeric(truth["target"], errors="coerce")
    if not np.isfinite(truth["target"].to_numpy(dtype=float)).all():
        raise HorizonContractError("validation labels must be finite and eligible")
    pred["score"] = pd.to_numeric(pred["score"], errors="coerce")
    finite = np.isfinite(pred["score"].to_numpy(dtype=float))
    okay = pred["status"].astype("string").eq("ok").to_numpy(dtype=bool)
    if not np.array_equal(finite, okay):
        raise HorizonContractError("prediction status and score finiteness disagree")

    model_runs = pred.loc[:, ["model", "seed"]].drop_duplicates()
    expected = (
        truth.assign(_join=1).merge(model_runs.assign(_join=1), on="_join").drop(columns="_join")
    )
    aligned = expected.merge(
        pred,
        on=["model", "seed", "ticker", "origin", "horizon"],
        how="left",
        validate="one_to_one",
    )

    records: list[dict[str, object]] = []
    group_columns = ["model", "seed", "horizon", "ticker"]
    for group_key, group in aligned.groupby(group_columns, sort=True, observed=True):
        available = group["status"].notna()
        scored = group["status"].eq("ok") & np.isfinite(group["score"])
        actual = group.loc[scored, "target"].to_numpy(dtype=float)
        estimate = group.loc[scored, "score"].to_numpy(dtype=float)
        errors = estimate - actual
        if len(errors):
            mae = float(np.mean(np.abs(errors)))
            rmse = float(np.sqrt(np.mean(np.square(errors))))
            bias = float(np.mean(errors))
            direction = float(np.mean(np.sign(estimate) == np.sign(actual)))
            pearson = (
                float(np.corrcoef(estimate, actual)[0, 1])
                if len(errors) >= 2 and np.ptp(estimate) > 0 and np.ptp(actual) > 0
                else np.nan
            )
            spearman = spearman_average_rank(estimate, actual)
        else:
            mae = rmse = bias = direction = pearson = spearman = np.nan
        records.append(
            {
                "model": group_key[0],
                "seed": int(group_key[1]),
                "horizon": int(group_key[2]),
                "ticker": str(group_key[3]),
                "eligible_labels": int(len(group)),
                "forecast_rows": int(available.sum()),
                "scored": int(scored.sum()),
                "coverage": float(scored.sum() / len(group)),
                "mae": mae,
                "rmse": rmse,
                "bias": bias,
                "pearson_correlation": pearson,
                "spearman_correlation": spearman,
                "directional_accuracy": direction,
                "first_origin": pd.Timestamp(group["origin"].min()),
                "last_origin": pd.Timestamp(group["origin"].max()),
            }
        )
    return (
        pd.DataFrame.from_records(records)
        .sort_values(group_columns, kind="stable")
        .reset_index(drop=True)
    )


# A descriptive alias for callers that read this metric as a report rather than
# as part of the validation stage.
per_instrument_validation_metrics = validation_metrics_by_instrument


__all__ = [
    "HORIZON_CANDIDATES",
    "KRONOS_EXPOSES_SAMPLE_PATHS",
    "KRONOS_MAX_CONTEXT_SESSIONS",
    "KRONOS_MAX_OUTPUT_SESSIONS",
    "KRONOS_PATH_COLUMNS",
    "KRONOS_POINT_OUTPUT_SEMANTICS",
    "MAX_SEARCH_CONFIGS_PER_HORIZON",
    "SUPPORTED_HORIZONS",
    "TERMINAL_RETURN_TARGET_KIND",
    "HorizonContractError",
    "HorizonKronosAdapter",
    "HorizonKronosConfig",
    "HorizonLightGBMModel",
    "HorizonSpec",
    "HorizonTuningResult",
    "KronosHorizonCapabilities",
    "UnsupportedHorizonError",
    "build_horizon_supervised_panel",
    "build_horizon_targets",
    "chronological_splits_by_horizon",
    "chronological_train_validation_split",
    "collapse_kronos_horizon_scores",
    "fit_final_horizon_lightgbm",
    "kronos_horizon_capabilities",
    "matured_labels",
    "per_instrument_validation_metrics",
    "predict_horizon_lightgbm",
    "tune_horizon_lightgbm",
    "validate_horizon",
    "validate_horizon_search_splits",
    "validate_horizons",
    "validate_kronos_horizon_paths",
    "validate_search_candidates",
]
