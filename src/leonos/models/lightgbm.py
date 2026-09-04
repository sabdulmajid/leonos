"""Pooled LightGBM baseline with a bounded validation-only search."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..artifacts import atomic_write_json
from ..evaluation import daily_rankic
from ..features import FEATURE_COLUMNS, FEATURE_SET_NAME, validate_feature_frame

PREDICTION_KEY_COLUMNS = ("model", "seed", "ticker", "origin", "horizon")
TRAIN_LABEL_END_MAX = pd.Timestamp("2023-12-31")
VALIDATION_ORIGIN_MIN = pd.Timestamp("2024-07-01")
VALIDATION_ORIGIN_MAX = pd.Timestamp("2024-12-31")
VALIDATION_LABEL_END_MAX = pd.Timestamp("2024-12-31")
FINAL_REFIT_LABEL_END_MAX = pd.Timestamp("2024-12-31")


@dataclass(frozen=True)
class CandidateConfig:
    """One intentionally small LightGBM configuration."""

    candidate_id: str
    learning_rate: float
    num_leaves: int
    min_data_in_leaf: int
    max_depth: int = -1
    feature_fraction: float = 1.0
    bagging_fraction: float = 1.0
    lambda_l1: float = 0.0
    lambda_l2: float = 1.0

    def complexity_key(self, best_iteration: int) -> tuple[float, int, int, str]:
        depth = self.max_depth if self.max_depth > 0 else self.num_leaves
        return (
            float(self.num_leaves * max(best_iteration, 1)),
            self.num_leaves,
            depth,
            self.candidate_id,
        )


DEFAULT_CANDIDATES = (
    CandidateConfig("l31_lr05", 0.05, 31, 20),
    CandidateConfig("l15_lr05", 0.05, 15, 20),
    CandidateConfig("l63_lr05", 0.05, 63, 20),
    CandidateConfig("l31_lr02", 0.02, 31, 20),
    CandidateConfig("l31_min50", 0.05, 31, 50),
    CandidateConfig(
        "l31_subsample", 0.05, 31, 20, feature_fraction=0.8, bagging_fraction=0.8
    ),
    CandidateConfig("l31_regularized", 0.05, 31, 20, lambda_l1=0.1, lambda_l2=5.0),
    CandidateConfig("l31_depth6", 0.05, 31, 20, max_depth=6),
)


@dataclass(frozen=True)
class SearchConfig:
    max_boost_rounds: int = 1_000
    early_stopping_rounds: int = 50
    minimum_daily_coverage: int = 3
    num_threads: int = 8
    seed: int = 42
    tie_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.max_boost_rounds < 1:
            raise ValueError("max_boost_rounds must be positive")
        if self.early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be positive")
        if not 1 <= self.num_threads <= 16:
            raise ValueError("num_threads must be capped between one and 16")
        if self.minimum_daily_coverage < 2:
            raise ValueError("minimum_daily_coverage must be at least two")
        if self.tie_tolerance < 0:
            raise ValueError("tie_tolerance cannot be negative")


DEFAULT_SEARCH_CONFIG = SearchConfig()


@dataclass(frozen=True)
class CandidateResult:
    candidate: CandidateConfig
    status: str
    best_iteration: int
    validation_mean_daily_rankic: float
    validation_mean_daily_mae: float
    fit_seconds: float
    error: str | None = None

    def flat_record(self) -> dict[str, object]:
        record: dict[str, object] = asdict(self.candidate)
        record.update(
            {
                "status": self.status,
                "best_iteration": self.best_iteration,
                "validation_mean_daily_rankic": self.validation_mean_daily_rankic,
                "validation_mean_daily_mae": self.validation_mean_daily_mae,
                "fit_seconds": self.fit_seconds,
                "error": self.error,
            }
        )
        return record


@dataclass
class TuningResult:
    selected: CandidateResult
    candidates: tuple[CandidateResult, ...]
    validation_booster: Any
    feature_columns: tuple[str, ...]
    search_config: SearchConfig

    def records(self) -> pd.DataFrame:
        return pd.DataFrame([candidate.flat_record() for candidate in self.candidates])


@dataclass
class LightGBMModel:
    booster: Any
    candidate: CandidateConfig
    feature_columns: tuple[str, ...]
    seed: int
    boosting_rounds: int
    training_rows: int
    training_label_end_max: pd.Timestamp
    fit_seconds: float


def _require_lightgbm() -> Any:
    try:
        import lightgbm as lgb
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "LightGBM is required for fitting; install the pinned project dependencies"
        ) from exc
    return lgb


def _as_dates(values: pd.Series, column: str) -> pd.Series:
    try:
        return pd.to_datetime(values, utc=True, errors="raise").dt.tz_convert(None).dt.normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {column} timestamps") from exc


def validate_search_splits(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    """Enforce the predeclared development and validation timing boundaries."""

    for name, frame in (("training", train), ("validation", validation)):
        missing = {"ticker", "origin", "label_end", "target"}.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} frame missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{name} frame is empty")
        if frame.duplicated(["ticker", "origin"]).any():
            raise ValueError(f"{name} frame has duplicate ticker/origin keys")
    train_label_end = _as_dates(train["label_end"], "training label_end")
    validation_origin = _as_dates(validation["origin"], "validation origin")
    validation_label_end = _as_dates(validation["label_end"], "validation label_end")
    if (train_label_end > TRAIN_LABEL_END_MAX).any():
        raise ValueError("development training labels must end by 2023-12-31")
    if (validation_origin < VALIDATION_ORIGIN_MIN).any() or (
        validation_origin > VALIDATION_ORIGIN_MAX
    ).any():
        raise ValueError("validation origins must fall in July-December 2024")
    if (validation_label_end > VALIDATION_LABEL_END_MAX).any():
        raise ValueError("validation labels must end by 2024-12-31")
    overlap = train[["ticker", "origin"]].merge(
        validation[["ticker", "origin"]], on=["ticker", "origin"], how="inner"
    )
    if not overlap.empty:
        raise ValueError("training and validation forecast keys overlap")


def validate_final_refit_frame(frame: pd.DataFrame) -> pd.Timestamp:
    missing = {"ticker", "origin", "label_end", "target"}.difference(frame.columns)
    if missing:
        raise ValueError(f"final refit frame missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("final refit frame is empty")
    label_end = _as_dates(frame["label_end"], "final refit label_end")
    if (label_end > FINAL_REFIT_LABEL_END_MAX).any():
        raise ValueError("final refit labels must end by 2024-12-31")
    return pd.Timestamp(label_end.max())


def _supervised_arrays(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    target_col: str = "target",
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    validate_feature_frame(frame, feature_columns)
    if target_col not in frame.columns:
        raise ValueError(f"supervised frame missing target {target_col!r}")
    target = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(target)
    if not keep.any():
        raise ValueError("supervised frame contains no finite targets")
    selected = frame.loc[keep].reset_index(drop=True)
    matrix = selected.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    if matrix.notna().sum(axis=1).eq(0).any():
        raise ValueError("one or more training rows have no finite predictive features")
    return matrix, target[keep], selected


def _prediction_arrays(
    frame: pd.DataFrame, feature_columns: Sequence[str]
) -> pd.DataFrame:
    validate_feature_frame(frame, feature_columns)
    matrix = frame.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    return matrix.replace([np.inf, -np.inf], np.nan)


def _candidate_params(
    candidate: CandidateConfig, config: SearchConfig, seed: int
) -> dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "None",
        "boosting_type": "gbdt",
        "learning_rate": candidate.learning_rate,
        "num_leaves": candidate.num_leaves,
        "min_data_in_leaf": candidate.min_data_in_leaf,
        "max_depth": candidate.max_depth,
        "feature_fraction": candidate.feature_fraction,
        "bagging_fraction": candidate.bagging_fraction,
        "bagging_freq": 1 if candidate.bagging_fraction < 1.0 else 0,
        "lambda_l1": candidate.lambda_l1,
        "lambda_l2": candidate.lambda_l2,
        "feature_pre_filter": False,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": config.num_threads,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "verbosity": -1,
    }


def _rankic_feval(origin: pd.Series, minimum_coverage: int):
    dates = _as_dates(origin, "validation origin").to_numpy()
    groups = [np.flatnonzero(dates == date) for date in np.unique(dates)]

    def evaluate(prediction: np.ndarray, dataset: Any) -> tuple[str, float, bool]:
        label = np.asarray(dataset.get_label(), dtype=float)
        correlations = []
        for indices in groups:
            if len(indices) < minimum_coverage:
                continue
            score = prediction[indices]
            actual = label[indices]
            score_rank = pd.Series(score).rank(method="average").to_numpy(dtype=float)
            label_rank = pd.Series(actual).rank(method="average").to_numpy(dtype=float)
            if np.ptp(score_rank) == 0.0 or np.ptp(label_rank) == 0.0:
                continue
            correlations.append(float(np.corrcoef(score_rank, label_rank)[0, 1]))
        # A valid experiment has many defined dates.  Returning -1 keeps the
        # upstream callback finite while the post-fit gate rejects no-defined-date
        # candidates explicitly.
        value = float(np.mean(correlations)) if correlations else -1.0
        return "mean_daily_rankic", value, True

    return evaluate


def select_candidate(
    results: Sequence[CandidateResult], *, tie_tolerance: float = 1e-6
) -> CandidateResult:
    """Select RankIC winner; close ties prefer deterministically lower capacity."""

    successful = [
        result
        for result in results
        if result.status == "ok" and np.isfinite(result.validation_mean_daily_rankic)
    ]
    if not successful:
        errors = "; ".join(
            f"{result.candidate.candidate_id}: {result.error or result.status}"
            for result in results
        )
        raise RuntimeError(f"all LightGBM candidates failed: {errors}")
    best_score = max(result.validation_mean_daily_rankic for result in successful)
    contenders = [
        result
        for result in successful
        if result.validation_mean_daily_rankic >= best_score - tie_tolerance
    ]
    return min(
        contenders,
        key=lambda result: result.candidate.complexity_key(result.best_iteration),
    )


def tune_lightgbm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    candidates: Sequence[CandidateConfig] = DEFAULT_CANDIDATES,
    config: SearchConfig = DEFAULT_SEARCH_CONFIG,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    target_col: str = "target",
) -> TuningResult:
    """Run at most twelve configurations and select by validation daily RankIC."""

    if not 1 <= len(candidates) <= 12:
        raise ValueError("LightGBM search must contain between one and twelve candidates")
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("LightGBM candidate identifiers must be unique")
    validate_search_splits(train, validation)
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
    feval = _rankic_feval(
        validation_rows["origin"], minimum_coverage=config.minimum_daily_coverage
    )
    outcomes: list[CandidateResult] = []
    boosters: dict[str, Any] = {}
    for candidate in candidates:
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
            validation_rankic = float(defined.mean())
            daily_mae = (
                scored.assign(abs_error=np.abs(scored["score"] - scored["target"]))
                .groupby("origin", observed=True)["abs_error"]
                .mean()
            )
            outcome = CandidateResult(
                candidate=candidate,
                status="ok",
                best_iteration=best_iteration,
                validation_mean_daily_rankic=validation_rankic,
                validation_mean_daily_mae=float(daily_mae.mean()),
                fit_seconds=float(time.perf_counter() - started),
            )
            boosters[candidate.candidate_id] = booster
        except Exception as exc:  # retain every failed candidate in the search record
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
    return TuningResult(
        selected=selected,
        candidates=tuple(outcomes),
        validation_booster=boosters[selected.candidate.candidate_id],
        feature_columns=columns,
        search_config=config,
    )


def fit_final_lightgbm(
    refit: pd.DataFrame,
    tuning: TuningResult,
    *,
    seed: int | None = None,
    target_col: str = "target",
) -> LightGBMModel:
    """Refit the frozen selected configuration once through 2024-12-31."""

    label_end_max = validate_final_refit_frame(refit)
    train_x, train_y, _ = _supervised_arrays(
        refit, tuning.feature_columns, target_col=target_col
    )
    lgb = _require_lightgbm()
    selected_seed = tuning.search_config.seed if seed is None else int(seed)
    dataset = lgb.Dataset(
        train_x,
        label=train_y,
        feature_name=list(tuning.feature_columns),
        free_raw_data=False,
    )
    rounds = int(tuning.selected.best_iteration)
    if rounds < 1:
        raise ValueError("selected candidate has no valid boosting round count")
    started = time.perf_counter()
    booster = lgb.train(
        _candidate_params(tuning.selected.candidate, tuning.search_config, selected_seed),
        dataset,
        num_boost_round=rounds,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    return LightGBMModel(
        booster=booster,
        candidate=tuning.selected.candidate,
        feature_columns=tuning.feature_columns,
        seed=selected_seed,
        boosting_rounds=rounds,
        training_rows=len(train_x),
        training_label_end_max=label_end_max,
        fit_seconds=float(time.perf_counter() - started),
    )


def predict_lightgbm(
    model: LightGBMModel,
    features: pd.DataFrame,
    *,
    model_name: str = "lightgbm",
    horizon: int = 10,
) -> pd.DataFrame:
    """Return immutable-artifact-shaped raw appreciation scores."""

    missing = {"ticker", "origin"}.difference(features.columns)
    if missing:
        raise ValueError(f"prediction features missing keys: {sorted(missing)}")
    if features.duplicated(["ticker", "origin"]).any():
        raise ValueError("prediction feature keys are duplicated")
    if horizon != 10:
        raise ValueError("version 1 prediction horizon is fixed at ten sessions")
    matrix = _prediction_arrays(features, model.feature_columns)
    prediction = np.asarray(
        model.booster.predict(matrix, num_iteration=model.boosting_rounds), dtype=float
    )
    if prediction.shape != (len(features),):
        raise ValueError(
            f"LightGBM returned shape {prediction.shape}, expected {(len(features),)}"
        )
    result = features.loc[:, ["ticker", "origin"]].copy()
    result["origin"] = _as_dates(result["origin"], "prediction origin")
    result.insert(0, "seed", model.seed)
    result.insert(0, "model", model_name)
    result["horizon"] = horizon
    result["score"] = prediction
    result["status"] = np.where(np.isfinite(prediction), "ok", "nonfinite")
    return result.loc[
        :, [*PREDICTION_KEY_COLUMNS, "score", "status"]
    ].sort_values(["origin", "ticker"], kind="stable").reset_index(drop=True)


def validate_prediction_artifact(predictions: pd.DataFrame) -> None:
    missing = set(PREDICTION_KEY_COLUMNS).union({"score", "status"}).difference(
        predictions.columns
    )
    if missing:
        raise ValueError(f"prediction artifact missing columns: {sorted(missing)}")
    if predictions.duplicated(list(PREDICTION_KEY_COLUMNS)).any():
        raise ValueError("prediction artifact contains duplicate full keys")
    successful = predictions["status"].eq("ok")
    finite = np.isfinite(pd.to_numeric(predictions["score"], errors="coerce"))
    if not (successful == finite).all():
        raise ValueError("prediction status and score finiteness disagree")


def write_prediction_artifact(
    predictions: pd.DataFrame, path: str | Path, *, overwrite: bool = False
) -> Path:
    """Atomically persist one prediction shard without silently overwriting it."""

    validate_prediction_artifact(predictions)
    destination = Path(path)
    if destination.suffix != ".parquet":
        raise ValueError("prediction artifacts must use a .parquet suffix")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"immutable prediction shard already exists: {destination}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".parquet", dir=destination.parent
    )
    os.close(fd)
    try:
        predictions.to_parquet(temporary_name, index=False)
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def read_prediction_artifacts(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Load resumable shards and fail on duplicated forecast identities."""

    if not paths:
        raise ValueError("no prediction shards supplied")
    frames = [pd.read_parquet(Path(path)) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    validate_prediction_artifact(combined)
    return combined.sort_values(list(PREDICTION_KEY_COLUMNS), kind="stable").reset_index(
        drop=True
    )


def save_tuning_artifacts(tuning: TuningResult, directory: str | Path) -> dict[str, Path]:
    """Persist all search candidates plus the selected validation model."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    records_path = root / "candidates.parquet"
    fd, temporary_name = tempfile.mkstemp(
        prefix=".candidates.", suffix=".parquet", dir=root
    )
    os.close(fd)
    try:
        tuning.records().to_parquet(temporary_name, index=False)
        os.replace(temporary_name, records_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    model_path = root / "validation_model.txt"
    temporary_model = root / ".validation_model.txt.tmp"
    tuning.validation_booster.save_model(str(temporary_model))
    os.replace(temporary_model, model_path)
    metadata_path = atomic_write_json(
        root / "selection.json",
        {
            "feature_set": FEATURE_SET_NAME,
            "feature_columns": list(tuning.feature_columns),
            "search_config": asdict(tuning.search_config),
            "selected": tuning.selected.flat_record(),
        },
    )
    return {
        "candidates": records_path,
        "model": model_path,
        "selection": metadata_path,
    }


def save_final_model(model: LightGBMModel, directory: str | Path) -> dict[str, Path]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "model.txt"
    temporary_model = root / ".model.txt.tmp"
    model.booster.save_model(str(temporary_model))
    os.replace(temporary_model, model_path)
    metadata_path = atomic_write_json(
        root / "model.json",
        {
            "candidate": asdict(model.candidate),
            "feature_set": FEATURE_SET_NAME,
            "feature_columns": list(model.feature_columns),
            "seed": model.seed,
            "boosting_rounds": model.boosting_rounds,
            "training_rows": model.training_rows,
            "training_label_end_max": model.training_label_end_max.isoformat(),
            "fit_seconds": model.fit_seconds,
        },
    )
    return {"model": model_path, "metadata": metadata_path}


def load_final_model(directory: str | Path) -> LightGBMModel:
    root = Path(directory)
    metadata = json.loads((root / "model.json").read_text(encoding="utf-8"))
    lgb = _require_lightgbm()
    booster = lgb.Booster(model_file=str(root / "model.txt"))
    return LightGBMModel(
        booster=booster,
        candidate=CandidateConfig(**metadata["candidate"]),
        feature_columns=tuple(metadata["feature_columns"]),
        seed=int(metadata["seed"]),
        boosting_rounds=int(metadata["boosting_rounds"]),
        training_rows=int(metadata["training_rows"]),
        training_label_end_max=pd.Timestamp(metadata["training_label_end_max"]),
        fit_seconds=float(metadata["fit_seconds"]),
    )


__all__ = [
    "CandidateConfig",
    "CandidateResult",
    "DEFAULT_CANDIDATES",
    "DEFAULT_SEARCH_CONFIG",
    "LightGBMModel",
    "PREDICTION_KEY_COLUMNS",
    "SearchConfig",
    "TuningResult",
    "fit_final_lightgbm",
    "load_final_model",
    "predict_lightgbm",
    "read_prediction_artifacts",
    "save_final_model",
    "save_tuning_artifacts",
    "select_candidate",
    "tune_lightgbm",
    "validate_final_refit_frame",
    "validate_prediction_artifact",
    "validate_search_splits",
    "write_prediction_artifact",
]
