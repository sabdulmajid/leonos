from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import leonos.kronos_runner as runner
import leonos.pipeline as pipeline
from leonos.data import EXPECTED_PARQUETS, MANIFEST_SCHEMA, DataIntegrityError
from leonos.kronos_runner import chunked, iter_requests, run_kronos_predictions
from leonos.models.kronos import KronosInferenceConfig


def _runner_config(root: Path) -> dict[str, object]:
    inference = KronosInferenceConfig()
    return {
        "paths": {
            "raw_data": str(root / "raw"),
            "prepared_data": str(root / "prepared"),
            "artifacts": str(root / "artifacts"),
            "summaries": str(root / "summaries"),
        },
        "sources": {
            "dataset": {"revision": "b" * 40},
            "kronos_code": {"revision": inference.implementation_revision},
            "kronos_model": {"revision": inference.model_revision},
            "kronos_tokenizer": {"revision": inference.tokenizer_revision},
        },
        "forecast": {
            "context_sessions": 90,
            "horizon_sessions": 10,
            "kronos": {
                "temperature": inference.temperature,
                "top_p": inference.top_p,
                "top_k": None,
                "sample_count": inference.sample_count,
            },
        },
    }


def _write_prepare_marker(config: dict[str, object]) -> dict[str, object]:
    raw = Path(config["paths"]["raw_data"])
    prepared = Path(config["paths"]["prepared_data"])
    raw.mkdir(parents=True)
    prepared.mkdir(parents=True)
    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset": {
            "repo_id": "twelvedata/financial-world-model",
            "repo_type": "dataset",
            "revision": "b" * 40,
            "config": "bars_1day",
        },
        "expected_parquets": EXPECTED_PARQUETS,
        "files": [],
    }
    (raw / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    signature = pipeline._manifest_signature(config, manifest)
    marker = {
        "status": "complete",
        "run_signature": signature,
        "dataset_revision": "b" * 40,
        "row_counts": {},
    }
    (prepared / "prepare_run.json").write_text(json.dumps(marker), encoding="utf-8")
    return manifest


def test_request_builder_preserves_origin_ticker_alignment() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=110)
    bars = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "session": session,
                "open": base,
                "high": base + 2,
                "low": base - 2,
                "close": base + 1,
                "volume": 1_000 + index,
            }
            for ticker, offset in (("MSFT", 200), ("AAPL", 100))
            for index, session in enumerate(calendar)
            for base in [offset + index]
        ]
    )
    keys = pd.DataFrame({"ticker": ["MSFT", "AAPL"], "origin": [calendar[91], calendar[90]]})
    requests = list(iter_requests(bars, calendar, keys))
    assert [(r.ticker, r.origin) for r in requests] == [
        ("AAPL", calendar[90]),
        ("MSFT", calendar[91]),
    ]
    assert requests[0].history.iloc[-1]["close"] == pytest.approx(191)
    assert requests[0].history_dates[-1] == requests[0].origin
    assert requests[0].forecast_dates[0] == calendar[91]
    assert len(requests[0].forecast_dates) == 10


def test_request_builder_rejects_future_or_missing_context() -> None:
    calendar = pd.bdate_range("2024-01-02", periods=101)
    bars = pd.DataFrame(
        {
            "ticker": "A",
            "session": calendar,
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.0,
            "volume": 1.0,
        }
    )
    bars.loc[bars["session"] == calendar[50], "close"] = np.nan
    keys = pd.DataFrame({"ticker": ["A"], "origin": [calendar[90]]})
    with pytest.raises(ValueError, match="missing/nonfinite"):
        list(iter_requests(bars, calendar, keys))


def test_chunking_is_deterministic() -> None:
    assert list(chunked(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]


def test_runner_refuses_configuration_stale_preparation(tmp_path: Path) -> None:
    config = _runner_config(tmp_path)
    _write_prepare_marker(config)
    changed = copy.deepcopy(config)
    changed["forecast"]["context_sessions"] = 91

    with pytest.raises(DataIntegrityError, match="stale or incomplete"):
        run_kronos_predictions(
            changed, split="validation", seed=42, device="cpu", batch_size=1
        )
    assert not Path(config["paths"]["artifacts"]).exists()


def test_runner_refuses_dataset_stale_preparation(tmp_path: Path) -> None:
    config = _runner_config(tmp_path)
    manifest = _write_prepare_marker(config)
    manifest["dataset"]["revision"] = "c" * 40
    manifest_path = Path(config["paths"]["raw_data"]) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataIntegrityError, match="stale or incomplete"):
        run_kronos_predictions(
            config, split="validation", seed=42, device="cpu", batch_size=1
        )
    assert not Path(config["paths"]["artifacts"]).exists()


def _stub_complete_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], dict[str, object]]:
    config = _runner_config(tmp_path)
    prepared = Path(config["paths"]["prepared_data"])
    prepared.mkdir(parents=True)
    labels = prepared / "validation.parquet"
    bars = prepared / "bars.parquet"
    calendar = prepared / "calendar.parquet"
    pd.DataFrame(
        {
            "ticker": pd.Series(dtype="string"),
            "origin": pd.Series(dtype="datetime64[ns]"),
        }
    ).to_parquet(labels, index=False)
    pd.DataFrame(
        columns=["ticker", "session", "open", "high", "low", "close", "volume"]
    ).to_parquet(bars, index=False)
    pd.DataFrame({"session": [pd.Timestamp("2025-01-02")]}).to_parquet(
        calendar, index=False
    )
    paths = {
        "labels_validation": labels,
        "accepted_bars": bars,
        "calendar": calendar,
    }
    marker = {"run_signature": "prepare-signature", "dataset_revision": "b" * 40}
    monkeypatch.setattr(
        runner, "load_complete_preparation", lambda supplied: (paths, marker)
    )
    monkeypatch.setattr(
        runner,
        "load_official_predictor",
        lambda **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(runner, "KronosAdapter", lambda predictor, spec: SimpleNamespace())
    return config, marker


def test_worker_state_records_preparation_timing_revisions_and_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, marker = _stub_complete_preparation(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner,
        "ensure_kronos_assets",
        lambda supplied: SimpleNamespace(
            source_root=tmp_path, model_snapshot=tmp_path, tokenizer_snapshot=tmp_path
        ),
    )

    state = run_kronos_predictions(
        config, split="validation", seed=42, device="cpu", batch_size=1
    )

    assert state["status"] == "complete"
    assert state["prepare_signature"] == marker["run_signature"]
    assert state["dataset_revision"] == marker["dataset_revision"]
    assert pd.Timestamp(state["finished_at_utc"]) >= pd.Timestamp(state["started_at_utc"])
    assert state["elapsed_seconds"] >= 0.0
    assert state["gpu_memory"] == {
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }
    assert state["source_revisions"] == config["sources"]
    plan = json.loads(Path(state["run_plan_path"]).read_text(encoding="utf-8"))
    assert state["run_plan_signature"] == plan["run_signature"]
    assert plan["plan"]["eligible_key_count"] == 0
    assert plan["plan"]["selected_key_count"] == 0
    assert plan["plan"]["batch_size"] == 1
    assert plan["plan"]["num_shards"] == 1


@pytest.mark.parametrize(("batch_size", "num_shards"), [(2, 1), (1, 2)])
def test_existing_namespace_rejects_changed_worker_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    num_shards: int,
) -> None:
    config, _ = _stub_complete_preparation(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner,
        "ensure_kronos_assets",
        lambda supplied: SimpleNamespace(
            source_root=tmp_path, model_snapshot=tmp_path, tokenizer_snapshot=tmp_path
        ),
    )
    run_kronos_predictions(
        config, split="validation", seed=42, device="cpu", batch_size=1
    )

    with pytest.raises(runner.RunPlanMismatchError, match="different immutable run plan"):
        run_kronos_predictions(
            config,
            split="validation",
            seed=42,
            device="cpu",
            batch_size=batch_size,
            num_shards=num_shards,
        )


def test_limit_requires_separate_smoke_namespace(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="separate named smoke namespace"):
        run_kronos_predictions(
            _runner_config(tmp_path),
            split="validation",
            seed=42,
            device="cpu",
            batch_size=1,
            limit=1,
        )


def test_worker_failure_state_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _stub_complete_preparation(tmp_path, monkeypatch)

    def fail_assets(supplied: object) -> None:
        raise RuntimeError("checkpoint unavailable")

    monkeypatch.setattr(runner, "ensure_kronos_assets", fail_assets)
    with pytest.raises(RuntimeError, match="checkpoint unavailable"):
        run_kronos_predictions(
            config, split="validation", seed=42, device="cpu", batch_size=1
        )

    manifest = (
        Path(config["paths"]["artifacts"])
        / "predictions/kronos/validation/seed=42/worker-00-of-01.json"
    )
    failed = json.loads(manifest.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["finished_at_utc"]
    assert failed["elapsed_seconds"] >= 0.0
    assert failed["error"] == "RuntimeError: checkpoint unavailable"


def test_collapsed_scores_keep_evaluation_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    origin = pd.Timestamp("2025-01-02")
    frame = pd.DataFrame(
        {
            "model": "kronos",
            "seed": 42,
            "ticker": "AAPL",
            "origin": origin,
            "horizon": range(1, 11),
            "score": 0.02,
        }
    )
    monkeypatch.setattr(runner.pd, "read_parquet", lambda path: frame)
    monkeypatch.setattr(runner, "validate_prediction_frame", lambda supplied: None)

    collapsed = runner.collapse_kronos_scores(["one.parquet"])

    assert collapsed.to_dict("records") == [
        {
            "model": "kronos",
            "seed": 42,
            "ticker": "AAPL",
            "origin": origin,
            "score": 0.02,
            "horizon": 10,
            "status": "ok",
        }
    ]
