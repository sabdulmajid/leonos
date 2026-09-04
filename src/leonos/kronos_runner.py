"""Asset resolution and efficient, resumable Kronos inference orchestration."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable, Iterator, Sequence
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


def _key_set_metadata(keys: pd.DataFrame) -> dict[str, Any]:
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
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        if run_name is None or run_name == split:
            raise ValueError("limit is allowed only in a separate named smoke namespace")
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
    eligible_key_metadata = _key_set_metadata(eligible_keys)
    keys = eligible_keys
    if limit is not None:
        keys = keys.iloc[:limit]
    selected_key_metadata = _key_set_metadata(keys)

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
    manifest_path = output_root / f"worker-{shard_index:02d}-of-{num_shards:02d}.json"
    _reset_gpu_peak_memory(device)
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
        "started_at_utc": started_at_utc,
        "finished_at_utc": None,
        "elapsed_seconds": 0.0,
        "gpu_memory": _gpu_peak_memory(device),
        "git": git_state(),
        "environment": runtime_environment(),
        "config_hash": config_hash,
        "source_revisions": config.get("sources", {}),
        "artifacts": [],
    }
    atomic_write_json(manifest_path, state)
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
            state["elapsed_seconds"] = float(time.perf_counter() - started_counter)
            state["gpu_memory"] = _gpu_peak_memory(device)
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
        state["elapsed_seconds"] = float(time.perf_counter() - started_counter)
        state["gpu_memory"] = _gpu_peak_memory(device)
        atomic_write_json(manifest_path, state)
        raise
    state["finished_at_utc"] = _utc_now()
    state["elapsed_seconds"] = float(time.perf_counter() - started_counter)
    state["gpu_memory"] = _gpu_peak_memory(device)
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
    "chunked",
    "collapse_kronos_scores",
    "ensure_kronos_assets",
    "iter_requests",
    "run_kronos_predictions",
]
