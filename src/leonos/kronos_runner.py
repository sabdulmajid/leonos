"""Asset resolution and efficient, resumable Kronos inference orchestration."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import atomic_write_json, git_state, runtime_environment, stable_hash
from .models.kronos import (
    ForecastRequest,
    KronosAdapter,
    KronosInferenceConfig,
    iter_completed_origin_keys,
    load_official_predictor,
    prediction_shard_path,
    validate_prediction_frame,
    write_prediction_shard,
)
from .pipeline import load_complete_preparation


@dataclass(frozen=True)
class KronosAssets:
    source_root: Path
    model_snapshot: Path
    tokenizer_snapshot: Path


class RunPlanMismatchError(RuntimeError):
    """A worker attempted to enter an existing namespace with a different plan."""


def frozen_kronos_execution_plan(
    config: Mapping[str, Any],
    *,
    batch_size: int | None = None,
    num_shards: int | None = None,
) -> tuple[int, int]:
    """Return the canonical YAML plan and reject differing command-line overrides."""

    try:
        inference = config["forecast"]["kronos"]
        frozen_batch_size = int(inference["batch_size"])
        frozen_num_shards = int(inference["num_shards"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "forecast.kronos.batch_size and num_shards must be frozen in config"
        ) from exc
    if frozen_batch_size < 1 or frozen_num_shards < 1:
        raise ValueError("frozen Kronos batch_size and num_shards must be positive")
    if batch_size is not None and int(batch_size) != frozen_batch_size:
        raise ValueError("--batch-size must equal frozen forecast.kronos.batch_size")
    if num_shards is not None and int(num_shards) != frozen_num_shards:
        raise ValueError("--num-shards must equal frozen forecast.kronos.num_shards")
    return frozen_batch_size, frozen_num_shards


def _checked_run(arguments: Sequence[str], *, cwd: Path | None = None) -> None:
    subprocess.run(list(arguments), cwd=cwd, check=True)


def _ensure_source_checkout(url: str, revision: str, root: Path) -> Path:
    destination = root / revision
    if destination.exists():
        actual = subprocess.run(
            ["git", "-C", str(destination), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual != revision:
            raise RuntimeError(f"Kronos checkout revision mismatch at {destination}")
        return destination

    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".kronos-source-", dir=root))
    try:
        _checked_run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(temporary)])
        _checked_run(["git", "checkout", "--detach", revision], cwd=temporary)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def ensure_kronos_assets(config: dict[str, Any]) -> KronosAssets:
    """Resolve exact code and model snapshots; never follows a mutable revision."""
    sources = config["sources"]
    checkpoint_root = Path(config["paths"]["checkpoints"])
    code = sources["kronos_code"]
    source_root = _ensure_source_checkout(
        str(code["url"]), str(code["revision"]), checkpoint_root / "sources" / "Kronos"
    )
    from huggingface_hub import snapshot_download

    cache = checkpoint_root / "huggingface"
    model = sources["kronos_model"]
    tokenizer = sources["kronos_tokenizer"]
    model_path = Path(
        snapshot_download(
            repo_id=str(model["repo_id"]),
            revision=str(model["revision"]),
            cache_dir=cache,
            allow_patterns=["config.json", "model.safetensors", "README.md"],
        )
    )
    tokenizer_path = Path(
        snapshot_download(
            repo_id=str(tokenizer["repo_id"]),
            revision=str(tokenizer["revision"]),
            cache_dir=cache,
            allow_patterns=["config.json", "model.safetensors", "README.md"],
        )
    )
    return KronosAssets(source_root, model_path, tokenizer_path)


def iter_requests(
    bars: pd.DataFrame,
    calendar: Sequence[object],
    keys: pd.DataFrame,
    *,
    columns: Sequence[str] = ("open", "high", "low", "close", "volume"),
    context_sessions: int = 90,
    horizon_sessions: int = 10,
) -> Iterator[ForecastRequest]:
    """Build contexts once per ticker and yield in frozen origin/ticker order."""
    required_bars = {"ticker", "session", *columns}
    missing = required_bars.difference(bars.columns)
    if missing:
        raise ValueError(f"bars missing request columns: {sorted(missing)}")
    if not {"ticker", "origin"}.issubset(keys.columns):
        raise ValueError("forecast keys must contain ticker and origin")
    if keys.duplicated(["ticker", "origin"]).any():
        raise ValueError("forecast keys are duplicated")
    sessions = pd.DatetimeIndex(pd.to_datetime(calendar, utc=True)).tz_convert(None).normalize()
    if sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise ValueError("calendar must be strictly increasing and unique")
    positions = {timestamp: index for index, timestamp in enumerate(sessions)}
    frame = bars.loc[:, ["ticker", "session", *columns]].copy()
    frame["session"] = pd.to_datetime(frame["session"], utc=True).dt.tz_convert(None).dt.normalize()
    if frame.duplicated(["ticker", "session"]).any():
        raise ValueError("bars contain duplicated ticker/session keys")
    by_ticker: dict[str, pd.DataFrame] = {}
    for ticker, group in frame.groupby("ticker", sort=True, observed=True):
        indexed = group.set_index("session").loc[:, list(columns)].reindex(sessions)
        by_ticker[str(ticker)] = indexed

    ordered = keys.loc[:, ["ticker", "origin"]].copy()
    ordered["ticker"] = ordered["ticker"].astype(str)
    ordered["origin"] = (
        pd.to_datetime(ordered["origin"], utc=True).dt.tz_convert(None).dt.normalize()
    )
    ordered = ordered.sort_values(["origin", "ticker"], kind="stable")
    for row in ordered.itertuples(index=False):
        ticker, origin = str(row.ticker), pd.Timestamp(row.origin)
        if ticker not in by_ticker or origin not in positions:
            raise ValueError(f"unknown forecast key: {(ticker, origin)}")
        position = positions[origin]
        if position < context_sessions - 1 or position + horizon_sessions >= len(sessions):
            raise ValueError(f"forecast key lacks calendar context/horizon: {(ticker, origin)}")
        history_dates = sessions[position - context_sessions + 1 : position + 1]
        future_dates = sessions[position + 1 : position + horizon_sessions + 1]
        history = by_ticker[ticker].loc[history_dates].reset_index(drop=True)
        values = history.loc[:, list(columns)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"forecast context has missing/nonfinite values: {(ticker, origin)}")
        yield ForecastRequest(ticker, origin, history, history_dates, future_dates)


def chunked(values: Iterable[Any], size: int) -> Iterator[list[Any]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _origin_keys_from_paths(paths: Sequence[Path]) -> set[tuple[str, int, str, pd.Timestamp]]:
    return set(iter_completed_origin_keys(paths)) if paths else set()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _resume_command(
    config: Mapping[str, Any],
    *,
    split: str,
    seed: int,
    device: str,
    batch_size: int,
    shard_index: int,
    num_shards: int,
    limit: int | None,
    run_name: str | None,
) -> str:
    config_path = str(config.get("_meta", {}).get("path", "configs/base.yaml"))
    command = [".venv/bin/leonos", "--config", config_path]
    if run_name == "validation-smoke":
        command.extend(
            [
                "smoke",
                "--device",
                device,
                "--batch-size",
                str(batch_size),
                "--limit",
                str(limit),
                "--seed",
                str(seed),
            ]
        )
    else:
        command.extend(
            [
                "predict",
                "--model",
                "kronos",
                "--split",
                split,
                "--seed",
                str(seed),
                "--device",
                device,
                "--batch-size",
                str(batch_size),
                "--shard-index",
                str(shard_index),
                "--num-shards",
                str(num_shards),
            ]
        )
    return shlex.join(command)


def _reset_gpu_peak_memory(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.device(device))
    except (ImportError, RuntimeError, ValueError):
        # A requested-but-unavailable device will fail during model loading; the
        # worker failure marker should still be writable with null memory values.
        return


def _gpu_peak_memory(device: str) -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }
    if not str(device).startswith("cuda"):
        return result
    try:
        import torch

        if torch.cuda.is_available():
            torch_device = torch.device(device)
            result["peak_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(torch_device)
            )
            result["peak_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(torch_device)
            )
    except (ImportError, RuntimeError, ValueError):
        pass
    return result


def _gpu_total_memory(device: str) -> int | None:
    if not str(device).startswith("cuda"):
        return None
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(torch.device(device)).total_memory)
    except (ImportError, RuntimeError, ValueError):
        pass
    return None


def select_safe_batch_size(
    records: Sequence[dict[str, Any]], *, max_memory_fraction: float = 0.85
) -> int:
    """Select highest measured throughput among successful memory-safe batches."""

    if not 0.0 < max_memory_fraction <= 1.0:
        raise ValueError("max_memory_fraction must be in (0, 1]")
    eligible = []
    for record in records:
        if record.get("status") != "ok":
            continue
        peak = record.get("peak_reserved_bytes")
        total = record.get("total_memory_bytes")
        if peak is not None and total is not None and peak / total > max_memory_fraction:
            continue
        eligible.append(record)
    if not eligible:
        raise RuntimeError("no successful memory-safe Kronos batch was measured")
    selected = max(
        eligible,
        key=lambda item: (float(item["origins_per_second"]), -int(item["batch_size"])),
    )
    return int(selected["batch_size"])


def _synchronize_gpu(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(device))
    except (ImportError, RuntimeError, ValueError):
        return


def benchmark_kronos_batches(
    config: dict[str, Any],
    *,
    device: str = "cuda:0",
    seed: int = 42,
    batch_sizes: Sequence[int] = (4, 8, 16, 32, 64),
    max_memory_fraction: float = 0.85,
) -> dict[str, Any]:
    """Brief validation-side sweep after a successful real-data smoke test.

    Benchmark forecasts are deliberately discarded and cannot enter evaluation.
    The recommendation is frozen in YAML before the canonical test run.
    """

    sizes = tuple(dict.fromkeys(int(value) for value in batch_sizes))
    if not sizes or any(value < 1 for value in sizes):
        raise ValueError("batch_sizes must contain positive integers")
    smoke_root = (
        Path(config["paths"]["artifacts"])
        / "predictions"
        / "kronos"
        / "validation-smoke"
        / f"seed={seed}"
    )
    smoke_workers = sorted(smoke_root.glob("worker-*-of-*.json"))
    if not smoke_workers:
        raise RuntimeError("run the real-data Kronos validation smoke before benchmarking")
    smoke_states = [json.loads(path.read_text(encoding="utf-8")) for path in smoke_workers]
    if not all(state.get("status") == "complete" for state in smoke_states):
        raise RuntimeError("Kronos validation smoke is not complete")

    started_at = _utc_now()
    started = time.perf_counter()
    prepared, preparation = load_complete_preparation(config)
    labels = pd.read_parquet(prepared["labels_validation"])
    bars = pd.read_parquet(prepared["accepted_bars"])
    calendar = pd.read_parquet(prepared["calendar"])["session"]
    keys = labels.loc[:, ["ticker", "origin"]].sort_values(
        ["origin", "ticker"], kind="stable"
    )
    requests = list(
        iter_requests(
            bars,
            calendar,
            keys.iloc[: max(sizes)],
            context_sessions=int(config["forecast"]["context_sessions"]),
            horizon_sessions=int(config["forecast"]["horizon_sessions"]),
        )
    )
    if len(requests) < max(sizes):
        raise RuntimeError("validation split is too small for the requested batch sweep")

    inference = config["forecast"]["kronos"]
    adapter_config = KronosInferenceConfig(
        seed=seed,
        temperature=float(inference["temperature"]),
        top_p=float(inference["top_p"]),
        top_k=0 if inference["top_k"] is None else int(inference["top_k"]),
        sample_count=int(inference["sample_count"]),
        model_revision=str(config["sources"]["kronos_model"]["revision"]),
        tokenizer_revision=str(config["sources"]["kronos_tokenizer"]["revision"]),
        implementation_revision=str(config["sources"]["kronos_code"]["revision"]),
    )
    assets = ensure_kronos_assets(config)
    predictor = load_official_predictor(
        source_root=assets.source_root,
        model_snapshot=assets.model_snapshot,
        tokenizer_snapshot=assets.tokenizer_snapshot,
        device=device,
        config=adapter_config,
    )
    adapter = KronosAdapter(predictor, adapter_config)
    adapter.predict_batch(requests[:1], split="validation-batch-warmup")
    _synchronize_gpu(device)
    total_memory = _gpu_total_memory(device)
    records: list[dict[str, Any]] = []
    for size in sizes:
        _reset_gpu_peak_memory(device)
        _synchronize_gpu(device)
        batch_started = time.perf_counter()
        try:
            frame = adapter.predict_batch(
                requests[:size], split=f"validation-batch-benchmark-{size}"
            )
            _synchronize_gpu(device)
            elapsed = time.perf_counter() - batch_started
            if len(frame) != size * int(config["forecast"]["horizon_sessions"]):
                raise RuntimeError("benchmark forecast cardinality mismatch")
            memory = _gpu_peak_memory(device)
            records.append(
                {
                    "batch_size": size,
                    "status": "ok",
                    "elapsed_seconds": float(elapsed),
                    "origins_per_second": float(size / elapsed),
                    "total_memory_bytes": total_memory,
                    **memory,
                }
            )
        except RuntimeError as exc:
            memory = _gpu_peak_memory(device)
            records.append(
                {
                    "batch_size": size,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": float(time.perf_counter() - batch_started),
                    "origins_per_second": 0.0,
                    "total_memory_bytes": total_memory,
                    **memory,
                }
            )
            if "out of memory" in str(exc).lower():
                break
            raise
    recommended = select_safe_batch_size(records, max_memory_fraction=max_memory_fraction)
    result = {
        "schema_version": "leonos.kronos_batch_benchmark.v1",
        "status": "complete",
        "seed": int(seed),
        "device": device,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "max_memory_fraction": float(max_memory_fraction),
        "recommended_batch_size": recommended,
        "prepare_signature": preparation["run_signature"],
        "dataset_revision": preparation["dataset_revision"],
        "config_hash": config.get("_meta", {}).get("sha256") or stable_hash(config),
        "git": git_state(),
        "environment": runtime_environment(),
        "source_revisions": config.get("sources", {}),
        "records": records,
        "note": "validation-side throughput only; benchmark predictions were discarded",
    }
    destination = Path(config["paths"]["summaries"]) / f"kronos_batch_seed_{seed}.json"
    atomic_write_json(destination, result)
    result["summary_path"] = str(destination)
    return result


def forecast_key_set_metadata(keys: pd.DataFrame) -> dict[str, Any]:
    if not {"ticker", "origin"}.issubset(keys.columns):
        raise ValueError("forecast keys must contain ticker and origin")
    clean = keys.loc[:, ["ticker", "origin"]].copy()
    clean["ticker"] = clean["ticker"].astype(str)
    clean["origin"] = (
        pd.to_datetime(clean["origin"], utc=True, errors="raise")
        .dt.tz_convert(None)
        .dt.normalize()
    )
    if clean.duplicated(["ticker", "origin"]).any():
        raise ValueError("eligible forecast keys are duplicated")
    clean = clean.sort_values(["origin", "ticker"], kind="stable")
    records = [
        {"ticker": str(row.ticker), "origin": pd.Timestamp(row.origin).isoformat()}
        for row in clean.itertuples(index=False)
    ]
    return {"count": len(records), "sha256": stable_hash(records)}


def _ensure_run_plan(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Atomically publish or exactly validate one immutable multi-worker plan."""

    document = {
        "schema_version": "leonos.kronos_run_plan.v1",
        "run_signature": stable_hash(payload),
        "plan": payload,
    }

    def read_existing() -> dict[str, Any]:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunPlanMismatchError(f"invalid existing Kronos run plan: {path}") from exc
        if existing != document:
            raise RunPlanMismatchError(
                "Kronos namespace already has a different immutable run plan"
            )
        return existing

    if path.exists():
        return read_existing()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            return read_existing()
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
        return document
    finally:
        temporary_path.unlink(missing_ok=True)


def _assigned_origin_keys(
    keys: pd.DataFrame,
    *,
    seed: int,
    batch_size: int,
    shard_index: int,
    num_shards: int,
) -> set[tuple[str, int, str, pd.Timestamp]]:
    assigned: set[tuple[str, int, str, pd.Timestamp]] = set()
    for batch_number, start in enumerate(range(0, len(keys), batch_size)):
        if batch_number % num_shards != shard_index:
            continue
        batch = keys.iloc[start : start + batch_size]
        assigned.update(
            (
                "kronos",
                int(seed),
                str(row.ticker),
                pd.Timestamp(row.origin),
            )
            for row in batch.itertuples(index=False)
        )
    return assigned


def _merge_gpu_memory(
    previous: Mapping[str, Any] | None, current: Mapping[str, Any]
) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
        values = [
            int(value)
            for value in (
                (previous or {}).get(name),
                current.get(name),
            )
            if value is not None
        ]
        result[name] = max(values) if values else None
    return result


def _validate_existing_worker_identity(
    state: Mapping[str, Any],
    *,
    split: str,
    run_name: str,
    seed: int,
    batch_size: int,
    shard_index: int,
    num_shards: int,
    run_plan_signature: str,
) -> None:
    expected = {
        "split": split,
        "run_name": run_name,
        "seed": int(seed),
        "batch_size": int(batch_size),
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "run_plan_signature": run_plan_signature,
    }
    wrong = [name for name, value in expected.items() if state.get(name) != value]
    if wrong:
        raise RunPlanMismatchError(
            f"existing Kronos worker state has mismatched identity: {sorted(wrong)}"
        )


def _complete_worker_is_safe(
    state: Mapping[str, Any],
    *,
    output_root: Path,
    assigned_keys: set[tuple[str, int, str, pd.Timestamp]],
    assigned_origins: int,
) -> bool:
    if state.get("status") != "complete":
        return False
    if int(state.get("assigned_origins", -1)) != assigned_origins or int(
        state.get("completed_origins", -1)
    ) != assigned_origins:
        raise RunPlanMismatchError("complete Kronos worker has inconsistent origin counts")
    required_runtime = {
        "first_started_at_utc",
        "finished_at_utc",
        "attempt_count",
        "cumulative_elapsed_seconds",
        "resume_command",
    }
    if not required_runtime.issubset(state):
        return False
    listed = {Path(str(value)).resolve() for value in state.get("artifacts", [])}
    if any(path.parent != output_root.resolve() or not path.is_file() for path in listed):
        raise RunPlanMismatchError("complete Kronos worker lists invalid shard paths")
    listed_keys = _origin_keys_from_paths(sorted(listed))
    if listed_keys != assigned_keys:
        raise RunPlanMismatchError("complete Kronos worker shards do not match its assignment")
    return True


def run_kronos_predictions(
    config: dict[str, Any],
    *,
    split: str,
    seed: int,
    device: str,
    batch_size: int,
    shard_index: int = 0,
    num_shards: int = 1,
    limit: int | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Run canonical logical batches assigned round-robin to this worker."""
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if run_name not in {None, split, "validation-smoke"}:
        raise ValueError("unsupported Kronos run namespace")
    if run_name == "validation-smoke" and limit is None:
        raise ValueError("the validation smoke namespace requires an explicit limit")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        if run_name is None or run_name == split:
            raise ValueError("limit is allowed only in a separate named smoke namespace")
    if run_name is None or run_name == split:
        frozen_kronos_execution_plan(
            config, batch_size=batch_size, num_shards=num_shards
        )
    started_at_utc = _utc_now()
    started_counter = time.perf_counter()
    prepared, preparation = load_complete_preparation(config)
    label_name = {
        "val": "validation",
        "validation": "validation",
        "test": "test",
    }.get(split, split)
    label_path_key = f"labels_{label_name}"
    if label_path_key not in prepared:
        raise ValueError(f"unsupported prepared forecast split: {split!r}")
    labels = pd.read_parquet(prepared[label_path_key])
    bars = pd.read_parquet(prepared["accepted_bars"])
    calendar = pd.read_parquet(prepared["calendar"])["session"]
    eligible_keys = labels.loc[:, ["ticker", "origin"]].sort_values(
        ["origin", "ticker"], kind="stable"
    )
    eligible_key_metadata = forecast_key_set_metadata(eligible_keys)
    keys = eligible_keys
    if limit is not None:
        keys = keys.iloc[:limit]
    selected_key_metadata = forecast_key_set_metadata(keys)

    inference = config["forecast"]["kronos"]
    adapter_config = KronosInferenceConfig(
        seed=seed,
        temperature=float(inference["temperature"]),
        top_p=float(inference["top_p"]),
        top_k=0 if inference["top_k"] is None else int(inference["top_k"]),
        sample_count=int(inference["sample_count"]),
        model_revision=str(config["sources"]["kronos_model"]["revision"]),
        tokenizer_revision=str(config["sources"]["kronos_tokenizer"]["revision"]),
        implementation_revision=str(config["sources"]["kronos_code"]["revision"]),
    )
    output_root = (
        Path(config["paths"]["artifacts"])
        / "predictions"
        / "kronos"
        / (run_name or split)
        / f"seed={seed}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    config_hash = config.get("_meta", {}).get("sha256") or stable_hash(config)
    run_plan_payload = {
        "split": split,
        "run_name": run_name or split,
        "seed": int(seed),
        "batch_size": int(batch_size),
        "num_shards": int(num_shards),
        "limit": limit,
        "eligible_key_count": eligible_key_metadata["count"],
        "eligible_key_sha256": eligible_key_metadata["sha256"],
        "selected_key_count": selected_key_metadata["count"],
        "selected_key_sha256": selected_key_metadata["sha256"],
        "sampling": asdict(adapter_config),
        "config_hash": config_hash,
        "prepare_signature": str(preparation["run_signature"]),
        "dataset_revision": str(preparation["dataset_revision"]),
    }
    run_plan_path = output_root / "run-plan.json"
    run_plan = _ensure_run_plan(run_plan_path, run_plan_payload)
    existing_paths = sorted(output_root.glob("*.parquet"))
    completed = _origin_keys_from_paths(existing_paths)
    total_batches = (len(keys) + batch_size - 1) // batch_size
    assigned_batch_numbers = range(shard_index, total_batches, num_shards)
    assigned_origins = sum(
        min(batch_size, len(keys) - batch_number * batch_size)
        for batch_number in assigned_batch_numbers
    )
    assigned_keys = _assigned_origin_keys(
        keys,
        seed=seed,
        batch_size=batch_size,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    if len(assigned_keys) != assigned_origins:
        raise RuntimeError("Kronos worker assignment cardinality mismatch")
    manifest_path = output_root / f"worker-{shard_index:02d}-of-{num_shards:02d}.json"
    resume_command = _resume_command(
        config,
        split=split,
        seed=seed,
        device=device,
        batch_size=batch_size,
        shard_index=shard_index,
        num_shards=num_shards,
        limit=limit,
        run_name=run_name,
    )
    previous_state: dict[str, Any] | None = None
    if manifest_path.is_file():
        try:
            previous_state = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunPlanMismatchError(
                f"invalid existing Kronos worker state: {manifest_path}"
            ) from exc
        _validate_existing_worker_identity(
            previous_state,
            split=split,
            run_name=run_name or split,
            seed=seed,
            batch_size=batch_size,
            shard_index=shard_index,
            num_shards=num_shards,
            run_plan_signature=str(run_plan["run_signature"]),
        )
        if _complete_worker_is_safe(
            previous_state,
            output_root=output_root,
            assigned_keys=assigned_keys,
            assigned_origins=assigned_origins,
        ):
            return previous_state

    previous_elapsed = float(
        (previous_state or {}).get(
            "cumulative_elapsed_seconds", (previous_state or {}).get("elapsed_seconds", 0.0)
        )
    )
    if not np.isfinite(previous_elapsed) or previous_elapsed < 0:
        raise RunPlanMismatchError("existing Kronos worker has invalid elapsed time")
    prior_attempts = (previous_state or {}).get(
        "attempt_count", 1 if previous_state is not None else 0
    )
    attempt_count = int(prior_attempts) + 1
    if attempt_count < 1:
        raise RunPlanMismatchError("existing Kronos worker has invalid attempt count")
    first_started_at_utc = str(
        (previous_state or {}).get(
            "first_started_at_utc",
            (previous_state or {}).get("started_at_utc", started_at_utc),
        )
    )
    _reset_gpu_peak_memory(device)
    initial_gpu_memory = _merge_gpu_memory(
        (previous_state or {}).get("gpu_memory"), _gpu_peak_memory(device)
    )
    state: dict[str, Any] = {
        "schema_version": "leonos.kronos_worker.v1",
        "status": "running",
        "split": split,
        "run_name": run_name or split,
        "seed": seed,
        "device": device,
        "batch_size": batch_size,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "eligible_origins": len(keys),
        "assigned_origins": assigned_origins,
        "completed_origins": 0,
        "prepare_signature": str(preparation["run_signature"]),
        "dataset_revision": str(preparation["dataset_revision"]),
        "run_plan_signature": str(run_plan["run_signature"]),
        "run_plan_path": str(run_plan_path),
        "first_started_at_utc": first_started_at_utc,
        "started_at_utc": first_started_at_utc,
        "attempt_started_at_utc": started_at_utc,
        "attempt_count": attempt_count,
        "finished_at_utc": None,
        "attempt_elapsed_seconds": 0.0,
        "cumulative_elapsed_seconds": previous_elapsed,
        "elapsed_seconds": previous_elapsed,
        "gpu_memory": initial_gpu_memory,
        "resume_command": resume_command,
        "git": git_state(),
        "environment": runtime_environment(),
        "config_hash": config_hash,
        "source_revisions": config.get("sources", {}),
        "artifacts": [],
    }
    atomic_write_json(manifest_path, state)

    def update_runtime() -> None:
        attempt_elapsed = float(time.perf_counter() - started_counter)
        cumulative_elapsed = previous_elapsed + attempt_elapsed
        state["attempt_elapsed_seconds"] = attempt_elapsed
        state["cumulative_elapsed_seconds"] = cumulative_elapsed
        state["elapsed_seconds"] = cumulative_elapsed
        state["gpu_memory"] = _merge_gpu_memory(
            state.get("gpu_memory"), _gpu_peak_memory(device)
        )

    try:
        assets = ensure_kronos_assets(config)
        predictor = load_official_predictor(
            source_root=assets.source_root,
            model_snapshot=assets.model_snapshot,
            tokenizer_snapshot=assets.tokenizer_snapshot,
            device=device,
            config=adapter_config,
        )
        adapter = KronosAdapter(predictor, adapter_config)
        requests = iter_requests(bars, calendar, keys)
        for batch_number, batch in enumerate(chunked(requests, batch_size)):
            if batch_number % num_shards != shard_index:
                continue
            path = prediction_shard_path(output_root, split=split, seed=seed, requests=batch)
            expected = {
                ("kronos", seed, str(request.ticker), pd.Timestamp(request.origin))
                for request in batch
            }
            already = expected.intersection(completed)
            if path.exists():
                frame = pd.read_parquet(path)
                validate_prediction_frame(frame)
                if already != expected:
                    raise RuntimeError(
                        f"partial/colliding resume state for logical batch {batch_number}"
                    )
            elif already:
                raise RuntimeError(f"origins completed under a different shard identity: {already}")
            else:
                frame = adapter.predict_batch(batch, split=split)
                write_prediction_shard(frame, path)
                completed.update(expected)
            state["completed_origins"] += len(batch)
            state["artifacts"].append(str(path))
            update_runtime()
            atomic_write_json(manifest_path, state)
            print(
                f"worker {shard_index}/{num_shards}: "
                f"{state['completed_origins']}/{state['assigned_origins']} origins",
                flush=True,
            )
        state["status"] = "complete"
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["finished_at_utc"] = _utc_now()
        update_runtime()
        atomic_write_json(manifest_path, state)
        raise
    state["finished_at_utc"] = _utc_now()
    update_runtime()
    atomic_write_json(manifest_path, state)
    return state


def collapse_kronos_scores(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Validate long-form paths and return one saved score per forecast origin."""
    if not paths:
        raise ValueError("no Kronos prediction shards supplied")
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        validate_prediction_frame(frame)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["model", "seed", "ticker", "origin", "horizon"]).any():
        raise ValueError("duplicate Kronos forecast keys across shards")
    scores = combined.loc[:, ["model", "seed", "ticker", "origin", "score"]].drop_duplicates()
    if scores.duplicated(["model", "seed", "ticker", "origin"]).any():
        raise ValueError("inconsistent Kronos scores across horizon rows")
    # The validated ten-row path collapses to the same origin-level schema as
    # LightGBM. ``horizon`` describes the declared target, not a path-row number.
    scores["horizon"] = 10
    scores["status"] = "ok"
    return scores.sort_values(["origin", "ticker"], kind="stable").reset_index(drop=True)


__all__ = [
    "KronosAssets",
    "RunPlanMismatchError",
    "benchmark_kronos_batches",
    "chunked",
    "collapse_kronos_scores",
    "ensure_kronos_assets",
    "forecast_key_set_metadata",
    "frozen_kronos_execution_plan",
    "iter_requests",
    "run_kronos_predictions",
    "select_safe_batch_size",
]
