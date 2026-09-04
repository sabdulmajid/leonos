"""Thin, resumable CPU preparation and LightGBM experiment pipeline."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .artifacts import atomic_write_json, git_state, runtime_environment, stable_hash
from .data import (
    CORE_COLUMNS,
    DEFAULT_DOCUMENTED_SPLITS,
    DataIntegrityError,
    apply_quality_policy,
    audit_daily_panel,
    exchange_sessions,
    load_daily_panel,
    load_manifest,
    normalize_daily_bars,
    to_canonical_bars,
    verify_manifest_files,
    write_audit_reports,
)
from .features import FEATURE_COLUMNS, build_ohlcv_features, feature_manifest
from .models.lightgbm import (
    DEFAULT_CANDIDATES,
    CandidateConfig,
    LightGBMModel,
    SearchConfig,
    fit_final_lightgbm,
    predict_lightgbm,
    read_prediction_artifacts,
    save_final_model,
    save_tuning_artifacts,
    tune_lightgbm,
    write_prediction_artifact,
)
from .targets import SplitSpec, TargetSpec, apply_split, build_targets

PREPARE_SCHEMA = "leonos.prepare.v1"
BASELINE_SCHEMA = "leonos.lightgbm_run.v1"

_PREPARED_PARQUETS = (
    "accepted_bars",
    "calendar",
    "exclusions",
    "labels_all",
    "labels_development",
    "labels_validation",
    "labels_refit",
    "labels_test",
    "features",
)


def _path(config: Mapping[str, Any], key: str) -> Path:
    try:
        value = config["paths"][key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"configuration is missing paths.{key}") from exc
    return Path(str(value))


def _config_hash(config: Mapping[str, Any]) -> str:
    metadata = config.get("_meta", {})
    if isinstance(metadata, Mapping) and metadata.get("sha256"):
        return str(metadata["sha256"])
    payload = {key: value for key, value in config.items() if key != "_meta"}
    return stable_hash(payload)


def _implementation_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _prepare_implementation_hash() -> str:
    package = Path(__file__).resolve().parent
    return _implementation_hash(
        [package / name for name in ("pipeline.py", "data.py", "targets.py", "features.py")]
    )


def _baseline_implementation_hash() -> str:
    package = Path(__file__).resolve().parent
    return _implementation_hash(
        [
            package / "pipeline.py",
            package / "features.py",
            package / "evaluation.py",
            package / "models" / "lightgbm.py",
        ]
    )


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


def _prepared_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    root = _path(config, "prepared_data")
    summary = _path(config, "summaries")
    return {
        "accepted_bars": root / "accepted_bars.parquet",
        "calendar": root / "sessions.parquet",
        "exclusions": root / "exclusions.parquet",
        "normalization": root / "normalization.json",
        "raw_audit_json": root / "audit_raw.json",
        "raw_audit_markdown": root / "audit_raw.md",
        "accepted_audit_json": root / "audit_accepted.json",
        "accepted_audit_markdown": root / "audit_accepted.md",
        "feature_manifest": root / "feature_manifest.json",
        "labels_all": root / "labels" / "all.parquet",
        "labels_development": root / "labels" / "development.parquet",
        "labels_validation": root / "labels" / "validation.parquet",
        "labels_refit": root / "labels" / "refit.parquet",
        "labels_test": root / "labels" / "test.parquet",
        "features": root / "features.parquet",
        "run_metadata": root / "prepare_run.json",
        "summary": summary / "preparation.json",
        "summary_audit_json": summary / "data_acceptance.json",
        "summary_audit_markdown": summary / "data_acceptance.md",
        "summary_raw_audit_json": summary / "data_audit_raw.json",
    }


def _manifest_signature(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    source_files = [
        {
            "path": item.get("path"),
            "sha256": item.get("sha256"),
            "size_bytes": item.get("size_bytes"),
        }
        for item in manifest.get("files", [])
        if item.get("kind") == "parquet"
    ]
    return stable_hash(
        {
            "schema": PREPARE_SCHEMA,
            "config_hash": _config_hash(config),
            "implementation_hash": _prepare_implementation_hash(),
            "dataset": manifest.get("dataset"),
            "source_files": source_files,
        }
    )


def _parquet_row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def _prepared_run_is_complete(
    marker: Mapping[str, Any],
    signature: str,
    paths: Mapping[str, Path],
) -> bool:
    if marker.get("status") != "complete" or marker.get("run_signature") != signature:
        return False
    row_counts = marker.get("row_counts", {})
    try:
        for key in _PREPARED_PARQUETS:
            path = paths[key]
            if not path.is_file() or _parquet_row_count(path) != int(row_counts[key]):
                return False
        for key in (
            "normalization",
            "raw_audit_json",
            "raw_audit_markdown",
            "accepted_audit_json",
            "accepted_audit_markdown",
            "feature_manifest",
            "summary",
            "summary_audit_json",
            "summary_audit_markdown",
            "summary_raw_audit_json",
        ):
            if not paths[key].is_file():
                return False
    except (KeyError, OSError, ValueError):
        return False
    return True


def _documented_split_settings(
    config: Mapping[str, Any],
) -> tuple[Sequence[Mapping[str, Any]], bool]:
    data_config = config.get("data", {})
    if not isinstance(data_config, Mapping):
        raise ValueError("configuration data section must be a mapping")
    events = data_config.get("documented_splits", DEFAULT_DOCUMENTED_SPLITS)
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise ValueError("data.documented_splits must be a sequence")
    require = bool(data_config.get("require_adjustment_check", True))
    return events, require


def _target_spec(config: Mapping[str, Any]) -> TargetSpec:
    try:
        forecast = config["forecast"]
        return TargetSpec(
            context_sessions=int(forecast["context_sessions"]),
            horizon_sessions=int(forecast["horizon_sessions"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid forecast context/horizon configuration") from exc


def _split_specs(config: Mapping[str, Any]) -> dict[str, SplitSpec]:
    try:
        split = config["splits"]
        return {
            "development": SplitSpec(
                "development", label_end_max=split["development_label_end_max"]
            ),
            "validation": SplitSpec(
                "validation",
                origin_start=split["validation_origin_min"],
                origin_end=split["validation_origin_max"],
                label_end_max=split["validation_label_end_max"],
            ),
            "refit": SplitSpec(
                "refit", label_end_max=split["final_refit_label_end_max"]
            ),
            "test": SplitSpec(
                "test",
                origin_start=split["test_origin_min"],
                origin_end=split.get("test_origin_max"),
                label_end_max=split.get("test_label_end_max"),
            ),
        }
    except (KeyError, TypeError) as exc:
        raise ValueError("configuration is missing required chronological split bounds") from exc


def _coverage(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()) if len(frame) else 0,
        "origin_min": frame["origin"].min().date().isoformat() if len(frame) else None,
        "origin_max": frame["origin"].max().date().isoformat() if len(frame) else None,
        "label_end_max": frame["label_end"].max().date().isoformat()
        if len(frame)
        else None,
    }


def _json_paths(paths: Mapping[str, Path]) -> dict[str, str]:
    return {key: str(value) for key, value in paths.items()}


def prepare_data(config: Mapping[str, Any]) -> dict[str, str]:
    """Audit, accept, target, and feature the pinned daily panel once."""

    paths = _prepared_paths(config)
    raw_root = _path(config, "raw_data")
    manifest_path = raw_root / "manifest.json"
    manifest = load_manifest(manifest_path)
    declared_revision = str(config.get("sources", {}).get("dataset", {}).get("revision", ""))
    recorded_revision = str(manifest["dataset"]["revision"])
    if declared_revision and declared_revision != recorded_revision:
        raise DataIntegrityError(
            f"configured dataset revision {declared_revision} != manifest {recorded_revision}"
        )
    data_config = config.get("data", {})
    verify_hashes = bool(data_config.get("verify_hashes", True))
    verify_manifest_files(raw_root, manifest, verify_hashes=verify_hashes)
    signature = _manifest_signature(config, manifest)
    if paths["run_metadata"].is_file():
        import json

        marker = json.loads(paths["run_metadata"].read_text(encoding="utf-8"))
        if _prepared_run_is_complete(marker, signature, paths):
            return _json_paths(paths)

    started = time.perf_counter()
    started_at = datetime.now(UTC)
    # Version 1 deliberately ignores the source's published indicators, joins,
    # and text fields. Only native daily OHLCV/audit columns enter preparation.
    raw = load_daily_panel(
        raw_root,
        manifest_path=manifest_path,
        columns=CORE_COLUMNS,
        verify_hashes=False,
    )
    normalized, normalization = normalize_daily_bars(
        raw,
        drop_identical_duplicates=bool(
            data_config.get("drop_identical_duplicates", True)
        ),
    )
    if normalized.empty:
        raise DataIntegrityError("normalized daily panel is empty")
    calendar = exchange_sessions(normalized["datetime"].min(), normalized["datetime"].max())
    target_spec = _target_spec(config)
    documented_splits, require_adjustment_check = _documented_split_settings(config)

    raw_audit = audit_daily_panel(
        normalized,
        calendar_sessions=calendar,
        documented_splits=documented_splits,
        context_sessions=target_spec.context_sessions,
        horizon_sessions=target_spec.horizon_sessions,
        require_adjustment_check=require_adjustment_check,
    )
    write_audit_reports(
        raw_audit, paths["raw_audit_json"], paths["raw_audit_markdown"]
    )
    atomic_write_json(paths["summary_raw_audit_json"], raw_audit)

    canonical = to_canonical_bars(normalized)
    accepted, exclusions = apply_quality_policy(
        canonical,
        calendar,
        fatal_ohlc=str(data_config.get("fatal_ohlc", "raise")),
    )
    accepted_audit = audit_daily_panel(
        accepted,
        calendar_sessions=calendar,
        documented_splits=documented_splits,
        context_sessions=target_spec.context_sessions,
        horizon_sessions=target_spec.horizon_sessions,
        require_adjustment_check=require_adjustment_check,
    )
    write_audit_reports(
        accepted_audit,
        paths["accepted_audit_json"],
        paths["accepted_audit_markdown"],
    )
    write_audit_reports(
        accepted_audit,
        paths["summary_audit_json"],
        paths["summary_audit_markdown"],
    )
    if not bool(accepted_audit.get("acceptance", {}).get("accepted")):
        codes = [
            item.get("code")
            for item in accepted_audit.get("findings", [])
            if item.get("severity") == "error"
        ]
        raise DataIntegrityError(f"post-policy daily panel failed acceptance: {codes}")

    # Native ``datetime`` remains untouched for provenance. The Leonos ``session``
    # alias is a timezone-naive exchange date, which is also the contract consumed
    # by the Qlib file adapter and Kronos request builder.
    standardized = accepted.copy()
    standardized["ticker"] = standardized["ticker"].astype("string")
    standardized["session"] = (
        pd.to_datetime(standardized["session"], utc=True, errors="raise")
        .dt.tz_convert(None)
        .dt.normalize()
    )

    labels_all = build_targets(standardized, calendar, spec=target_spec)
    split_specs = _split_specs(config)
    labels = {
        name: apply_split(labels_all, split_spec)
        for name, split_spec in split_specs.items()
    }
    empty_splits = [name for name, frame in labels.items() if frame.empty]
    if empty_splits:
        raise DataIntegrityError(f"accepted panel produced empty target splits: {empty_splits}")
    refit_keys = labels["refit"][["ticker", "origin"]]
    for name in ("development", "validation"):
        missing_refit = labels[name][["ticker", "origin"]].merge(
            refit_keys,
            on=["ticker", "origin"],
            how="left",
            indicator=True,
        )
        if missing_refit["_merge"].eq("left_only").any():
            raise DataIntegrityError(f"{name} keys are missing from the declared refit set")

    features = build_ohlcv_features(
        standardized,
        calendar,
        keys=labels_all[["ticker", "origin"]],
        context_sessions=target_spec.context_sessions,
    )
    feature_keys = features[["ticker", "origin"]]
    feature_alignment = labels_all[["ticker", "origin"]].merge(
        feature_keys,
        on=["ticker", "origin"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    if feature_alignment["_merge"].eq("left_only").any():
        missing_count = int(feature_alignment["_merge"].eq("left_only").sum())
        raise DataIntegrityError(f"{missing_count} labeled origins lack causal OHLCV features")

    _atomic_write_parquet(standardized, paths["accepted_bars"])
    _atomic_write_parquet(pd.DataFrame({"session": calendar}), paths["calendar"])
    _atomic_write_parquet(exclusions, paths["exclusions"])
    atomic_write_json(paths["normalization"], normalization)
    atomic_write_json(paths["feature_manifest"], feature_manifest())
    _atomic_write_parquet(labels_all, paths["labels_all"])
    _atomic_write_parquet(labels["development"], paths["labels_development"])
    _atomic_write_parquet(labels["validation"], paths["labels_validation"])
    _atomic_write_parquet(labels["refit"], paths["labels_refit"])
    _atomic_write_parquet(labels["test"], paths["labels_test"])
    _atomic_write_parquet(features, paths["features"])

    reason_counts = (
        exclusions.groupby("reason", observed=True).size().astype(int).to_dict()
        if len(exclusions)
        else {}
    )
    row_counts = {key: _parquet_row_count(paths[key]) for key in _PREPARED_PARQUETS}
    summary: dict[str, Any] = {
        "schema_version": PREPARE_SCHEMA,
        "status": "complete",
        "run_signature": signature,
        "config_hash": _config_hash(config),
        "implementation_hash": _prepare_implementation_hash(),
        "dataset_revision": recorded_revision,
        "source_revisions": config.get("sources", {}),
        "git": git_state(),
        "environment": runtime_environment(),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "raw_rows": int(len(raw)),
        "normalized_rows": int(len(normalized)),
        "accepted_rows": int(len(accepted)),
        "excluded_unique_rows": int(exclusions["row_index"].nunique())
        if len(exclusions)
        else 0,
        "exclusion_ledger_rows": int(len(exclusions)),
        "exclusions_by_reason": reason_counts,
        "accepted_tickers": int(accepted["ticker"].nunique()),
        "calendar_sessions": int(len(calendar)),
        "feature_count": len(FEATURE_COLUMNS),
        "split_coverage": {
            "all": _coverage(labels_all),
            **{name: _coverage(frame) for name, frame in labels.items()},
        },
        "row_counts": row_counts,
        "elapsed_seconds": float(time.perf_counter() - started),
        "artifacts": {key: str(value) for key, value in paths.items()},
    }
    atomic_write_json(paths["summary"], summary)
    atomic_write_json(paths["run_metadata"], summary)
    return _json_paths(paths)


def load_complete_preparation(
    config: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Load the exact prepared-path map after checking its provenance gate."""

    import json

    paths = _prepared_paths(config)
    if not paths["run_metadata"].is_file():
        raise FileNotFoundError("prepared artifacts are absent; run prepare_data first")
    marker = json.loads(paths["run_metadata"].read_text(encoding="utf-8"))
    raw_root = _path(config, "raw_data")
    manifest = load_manifest(raw_root / "manifest.json")
    signature = _manifest_signature(config, manifest)
    if not _prepared_run_is_complete(marker, signature, paths):
        raise DataIntegrityError("prepared artifacts are stale or incomplete; rerun prepare_data")
    return paths, marker


def _merge_supervised(labels: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if labels.duplicated(["ticker", "origin"]).any():
        raise DataIntegrityError("labels contain duplicate ticker/origin keys")
    if features.duplicated(["ticker", "origin"]).any():
        raise DataIntegrityError("features contain duplicate ticker/origin keys")
    merged = labels.merge(
        features[["ticker", "origin", *FEATURE_COLUMNS]],
        on=["ticker", "origin"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing = merged["_merge"].eq("left_only")
    if missing.any():
        raise DataIntegrityError(f"{int(missing.sum())} labels lack cached features")
    return merged.drop(columns="_merge")


def _candidate_configs(config: Mapping[str, Any]) -> tuple[CandidateConfig, ...]:
    lightgbm = config.get("lightgbm", {})
    configured = lightgbm.get("candidates")
    if configured is None:
        candidates = DEFAULT_CANDIDATES
    else:
        if not isinstance(configured, Sequence) or isinstance(configured, (str, bytes)):
            raise ValueError("lightgbm.candidates must be a sequence of mappings")
        candidates = tuple(CandidateConfig(**dict(candidate)) for candidate in configured)
    maximum = min(int(lightgbm.get("max_search_configs", 12)), 12)
    if maximum < 1 or len(candidates) > maximum:
        raise ValueError(
            f"configured {len(candidates)} LightGBM candidates exceeds cap {maximum}"
        )
    return tuple(candidates)


def _search_config(config: Mapping[str, Any], seed: int) -> SearchConfig:
    lightgbm = config.get("lightgbm", {})
    evaluation = config.get("evaluation", {})
    return SearchConfig(
        max_boost_rounds=int(lightgbm.get("max_rounds", 1_000)),
        early_stopping_rounds=int(lightgbm.get("early_stopping_rounds", 50)),
        minimum_daily_coverage=int(evaluation.get("minimum_daily_coverage", 3)),
        num_threads=int(lightgbm.get("num_threads", 8)),
        seed=int(seed),
        tie_tolerance=float(lightgbm.get("tie_tolerance", 1e-6)),
    )


def _baseline_signature(
    config: Mapping[str, Any], prepare_marker: Mapping[str, Any], seed: int
) -> str:
    return stable_hash(
        {
            "schema": BASELINE_SCHEMA,
            "config_hash": _config_hash(config),
            "prepare_signature": prepare_marker["run_signature"],
            "implementation_hash": _baseline_implementation_hash(),
            "seed": int(seed),
            "candidates": [asdict(candidate) for candidate in _candidate_configs(config)],
        }
    )


def _baseline_paths(config: Mapping[str, Any], seed: int, signature: str) -> dict[str, Path]:
    del signature
    artifact_root = _path(config, "artifacts")
    run_root = artifact_root / "models" / "lightgbm" / f"seed={seed}"
    summary_root = _path(config, "summaries")
    return {
        "run_root": run_root,
        "tuning_root": run_root / "tuning",
        "final_model_root": run_root / "final",
        "validation_predictions": artifact_root
        / "predictions"
        / "lightgbm"
        / "validation"
        / f"seed={seed}"
        / "predictions.parquet",
        "test_predictions": artifact_root
        / "predictions"
        / "lightgbm"
        / "test"
        / f"seed={seed}"
        / "predictions.parquet",
        "run_metadata": run_root / "run.json",
        "latest": run_root / "latest.json",
        "summary": summary_root / f"lightgbm_seed_{seed}.json",
        "summary_candidates": summary_root / f"lightgbm_candidates_seed_{seed}.parquet",
    }


def _baseline_run_is_complete(
    marker: Mapping[str, Any], signature: str, paths: Mapping[str, Path]
) -> bool:
    if marker.get("status") != "complete" or marker.get("run_signature") != signature:
        return False
    required = (
        paths["validation_predictions"],
        paths["test_predictions"],
        paths["tuning_root"] / "candidates.parquet",
        paths["tuning_root"] / "selection.json",
        paths["tuning_root"] / "validation_model.txt",
        paths["final_model_root"] / "model.txt",
        paths["final_model_root"] / "model.json",
        paths["latest"],
        paths["summary"],
        paths["summary_candidates"],
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        validation = read_prediction_artifacts([paths["validation_predictions"]])
        test = read_prediction_artifacts([paths["test_predictions"]])
        return len(validation) == int(marker["row_counts"]["validation_predictions"]) and len(
            test
        ) == int(marker["row_counts"]["test_predictions"])
    except (KeyError, OSError, ValueError):
        return False


def _write_immutable_predictions(frame: pd.DataFrame, path: Path) -> Path:
    if not path.exists():
        return write_prediction_artifact(frame, path)
    existing = read_prediction_artifacts([path])
    expected = frame.sort_values(
        ["model", "seed", "ticker", "origin", "horizon"], kind="stable"
    ).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(existing, expected, check_exact=True)
    except AssertionError as exc:
        raise DataIntegrityError(
            f"immutable prediction artifact differs from rerun output: {path}"
        ) from exc
    return path


def fit_baseline(config: Mapping[str, Any], seed: int = 42) -> dict[str, str]:
    """Tune on validation, refit through 2024, and persist CPU predictions."""

    prepared, prepare_marker = load_complete_preparation(config)
    signature = _baseline_signature(config, prepare_marker, seed)
    paths = _baseline_paths(config, seed, signature)
    if paths["run_metadata"].is_file():
        import json

        marker = json.loads(paths["run_metadata"].read_text(encoding="utf-8"))
        if _baseline_run_is_complete(marker, signature, paths):
            return _json_paths(paths)
        if marker.get("run_signature") != signature:
            raise DataIntegrityError(
                "existing fixed-path LightGBM artifacts use a different run signature; "
                "preserve them and select a new artifact root or seed"
            )

    started = time.perf_counter()
    features = pd.read_parquet(prepared["features"])
    labels = {
        name: pd.read_parquet(prepared[f"labels_{name}"])
        for name in ("development", "validation", "refit", "test")
    }
    supervised = {
        name: _merge_supervised(frame, features) for name, frame in labels.items()
    }
    search_config = _search_config(config, seed)
    tuning_started = time.perf_counter()
    tuning = tune_lightgbm(
        supervised["development"],
        supervised["validation"],
        candidates=_candidate_configs(config),
        config=search_config,
    )
    tuning_seconds = float(time.perf_counter() - tuning_started)
    save_tuning_artifacts(tuning, paths["tuning_root"])

    validation_model = LightGBMModel(
        booster=tuning.validation_booster,
        candidate=tuning.selected.candidate,
        feature_columns=tuning.feature_columns,
        seed=int(seed),
        boosting_rounds=tuning.selected.best_iteration,
        training_rows=len(supervised["development"]),
        training_label_end_max=pd.Timestamp(
            supervised["development"]["label_end"].max()
        ),
        fit_seconds=tuning.selected.fit_seconds,
    )
    validation_prediction_started = time.perf_counter()
    validation_predictions = predict_lightgbm(
        validation_model, supervised["validation"]
    )
    validation_prediction_seconds = float(
        time.perf_counter() - validation_prediction_started
    )
    _write_immutable_predictions(
        validation_predictions, paths["validation_predictions"]
    )

    final_model = fit_final_lightgbm(supervised["refit"], tuning, seed=seed)
    save_final_model(final_model, paths["final_model_root"])
    test_prediction_started = time.perf_counter()
    test_predictions = predict_lightgbm(final_model, supervised["test"])
    test_prediction_seconds = float(time.perf_counter() - test_prediction_started)
    _write_immutable_predictions(test_predictions, paths["test_predictions"])
    _atomic_write_parquet(tuning.records(), paths["summary_candidates"])

    selected = tuning.selected.flat_record()
    metadata: dict[str, Any] = {
        "schema_version": BASELINE_SCHEMA,
        "status": "complete",
        "run_signature": signature,
        "config_hash": _config_hash(config),
        "prepare_signature": prepare_marker["run_signature"],
        "implementation_hash": _baseline_implementation_hash(),
        "seed": int(seed),
        "git": git_state(),
        "environment": runtime_environment(),
        "source_revisions": config.get("sources", {}),
        "feature_set": feature_manifest(),
        "search": {
            "candidate_count": len(tuning.candidates),
            "configuration": asdict(search_config),
            "selected": selected,
            "all_candidates_path": str(paths["tuning_root"] / "candidates.parquet"),
        },
        "split_coverage": {name: _coverage(frame) for name, frame in labels.items()},
        "row_counts": {
            "development": len(supervised["development"]),
            "validation": len(supervised["validation"]),
            "refit": len(supervised["refit"]),
            "test": len(supervised["test"]),
            "validation_predictions": len(validation_predictions),
            "test_predictions": len(test_predictions),
        },
        "prediction_coverage": {
            "validation_finite": int(np.isfinite(validation_predictions["score"]).sum()),
            "test_finite": int(np.isfinite(test_predictions["score"]).sum()),
        },
        "timing_seconds": {
            "tuning": tuning_seconds,
            "final_refit": final_model.fit_seconds,
            "validation_prediction": validation_prediction_seconds,
            "test_prediction": test_prediction_seconds,
            "total": float(time.perf_counter() - started),
        },
        "artifacts": {key: str(value) for key, value in paths.items()},
    }
    atomic_write_json(paths["summary"], metadata)
    atomic_write_json(
        paths["latest"],
        {
            "schema_version": BASELINE_SCHEMA,
            "seed": int(seed),
            "run_signature": signature,
            "run_metadata": str(paths["run_metadata"]),
        },
    )
    # The run marker is deliberately last: its presence means every referenced
    # model, prediction, summary, and pointer is durable.
    atomic_write_json(paths["run_metadata"], metadata)
    return _json_paths(paths)


__all__ = ["fit_baseline", "load_complete_preparation", "prepare_data"]
