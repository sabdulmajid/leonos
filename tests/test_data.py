from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from leonos.data import (
    DOWNLOAD_PATTERNS,
    DataIntegrityError,
    apply_quality_policy,
    audit_adjustments,
    audit_daily_panel,
    build_manifest,
    exchange_sessions,
    fetch_daily_snapshot,
    load_daily_panel,
    normalize_daily_bars,
    to_canonical_bars,
)

REVISION = "a" * 40


def _bars(
    sessions: pd.DatetimeIndex,
    *,
    symbol: str = "XYZ",
    source_split: str | None = None,
) -> pd.DataFrame:
    local = pd.DatetimeIndex(sessions).tz_localize("America/New_York")
    close = 100.0 + np.arange(len(local), dtype=float) / 10.0
    frame = pd.DataFrame(
        {
            "datetime": local,
            "symbol": symbol,
            "timeframe": "1day",
            "open": close - 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(len(local), 1_000_000, dtype=np.int64),
            "close_adj": close,
        }
    )
    if source_split is not None:
        frame["source_split"] = source_split
    return frame


def _write_snapshot(root: Path) -> None:
    split_dates = {
        "train": pd.date_range("2023-12-27", periods=3, freq="B"),
        "val": pd.date_range("2024-01-02", periods=3, freq="B"),
        "test": pd.date_range("2025-01-02", periods=3, freq="B"),
    }
    bars_dir = root / "bars_1day"
    bars_dir.mkdir(parents=True, exist_ok=True)
    for split, dates in split_dates.items():
        _bars(dates).to_parquet(bars_dir / f"{split}.parquet", index=False)
    (root / "README.md").write_text("fixture\n", encoding="utf-8")


def test_manifest_binds_exact_files_schema_counts_and_dates(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    manifest = build_manifest(
        tmp_path,
        revision=REVISION,
        retrieved_at_utc="2026-01-01T00:00:00+00:00",
    )

    parquet = {item["split"]: item for item in manifest["files"] if item["kind"] == "parquet"}
    assert set(parquet) == {"train", "val", "test"}
    assert parquet["train"]["row_count"] == 3
    assert parquet["train"]["sha256"]
    assert {field["name"] for field in parquet["train"]["schema"]}.issuperset(
        {"datetime", "symbol", "open", "close", "volume"}
    )
    assert parquet["train"]["date_min"].startswith("2023-12-27")

    loaded = load_daily_panel(tmp_path)
    assert len(loaded) == 9
    assert set(loaded["source_split"]) == {"train", "val", "test"}
    canonical = to_canonical_bars(loaded)
    assert canonical["ticker"].equals(canonical["symbol"])
    assert canonical["session"].equals(canonical["datetime"])


def test_exchange_calendar_expands_to_early_dataset_history() -> None:
    sessions = exchange_sessions("1970-01-01", "1970-01-07")
    assert sessions.strftime("%Y-%m-%d").tolist() == [
        "1970-01-02",
        "1970-01-05",
        "1970-01-06",
        "1970-01-07",
    ]


def test_fetch_resolves_once_and_reuses_manifest_without_main_lookup(tmp_path: Path) -> None:
    class FakeApi:
        calls = 0

        def dataset_info(self, *, repo_id: str) -> SimpleNamespace:
            assert repo_id == "twelvedata/financial-world-model"
            self.calls += 1
            return SimpleNamespace(sha=REVISION)

    api = FakeApi()
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        _write_snapshot(Path(str(kwargs["local_dir"])))
        return str(kwargs["local_dir"])

    manifest_path = fetch_daily_snapshot(tmp_path, api=api, snapshot_download_fn=download)
    assert manifest_path.is_file()
    assert api.calls == 1
    assert calls[0]["revision"] == REVISION
    assert calls[0]["allow_patterns"] == list(DOWNLOAD_PATTERNS)

    class BombApi:
        def dataset_info(self, **_: object) -> None:
            raise AssertionError("existing manifest must prevent resolving main")

    def bomb_download(**_: object) -> str:
        raise AssertionError("existing manifest must prevent another download")

    assert (
        fetch_daily_snapshot(tmp_path, api=BombApi(), snapshot_download_fn=bomb_download)
        == manifest_path
    )


def test_normalization_rejects_conflicts_and_logs_identical_deduplication() -> None:
    frame = _bars(pd.date_range("2024-01-02", periods=3, freq="B"))
    duplicate = pd.concat([frame, frame.iloc[[1]]], ignore_index=True)
    with pytest.raises(DataIntegrityError, match="identical duplicates"):
        normalize_daily_bars(duplicate)

    cleaned, log = normalize_daily_bars(duplicate, drop_identical_duplicates=True)
    assert len(cleaned) == len(frame)
    assert log["rows_removed"] == 1
    assert log["identical_duplicate_keys"] == 1
    assert log["no_market_values_filled"] is True

    conflict = pd.concat([frame, frame.iloc[[1]].assign(close=999.0)], ignore_index=True)
    with pytest.raises(DataIntegrityError, match="conflicting duplicate"):
        normalize_daily_bars(conflict, drop_identical_duplicates=True)


def test_quality_policy_logs_only_predeclared_row_exclusions() -> None:
    sessions = pd.date_range("2024-01-02", periods=3, freq="B")
    canonical = to_canonical_bars(_bars(sessions))
    canonical.loc[1, "volume"] = 0
    accepted, exclusions = apply_quality_policy(canonical, sessions[:2])

    assert accepted["ticker"].tolist() == ["XYZ"]
    assert exclusions["reason"].tolist() == [
        "nonpositive_or_nonfinite_volume",
        "non_exchange_session",
    ]
    assert len(canonical) == 3  # input was not mutated or truncated

    fatal = canonical.copy()
    fatal.loc[0, "high"] = fatal.loc[0, "low"] - 1
    with pytest.raises(DataIntegrityError, match="fatal OHLC"):
        apply_quality_policy(fatal, sessions)


def test_calendar_eligibility_does_not_compress_across_missing_bar() -> None:
    calendar = pd.date_range("2024-01-02", periods=106, freq="B")
    full = _bars(calendar)
    full_report = audit_daily_panel(
        full,
        calendar_sessions=calendar,
        documented_splits=[],
        require_adjustment_check=False,
    )
    assert full_report["origin_eligibility"]["all/XYZ"]["usable_context_and_label"] == 7

    # Removing position 95 still leaves ten later *rows* after position 94,
    # but the exact ten-session label is incomplete and must not be compressed.
    missing = _bars(calendar.delete(95))
    missing_report = audit_daily_panel(
        missing,
        calendar_sessions=calendar,
        documented_splits=[],
        require_adjustment_check=False,
    )
    assert missing_report["origin_eligibility"]["all/XYZ"]["usable_context_and_label"] == 0
    assert missing_report["coverage_by_ticker"]["XYZ"]["missing_exchange_sessions"] == 1
    assert any(x["code"] == "ticker_missing_sessions" for x in missing_report["findings"])


def test_split_check_accepts_retroactively_adjusted_prices_and_rejects_drop() -> None:
    sessions = pd.DatetimeIndex(["2024-06-06", "2024-06-07", "2024-06-10", "2024-06-11"])
    frame = _bars(sessions, symbol="NVDA")
    event = {
        "symbol": "NVDA",
        "effective_session": "2024-06-10",
        "ratio_new_to_old": 10.0,
        "source": "https://investor.nvidia.com/example",
    }
    report, findings = audit_adjustments(frame, [event])
    assert report["status"] == "pass"
    assert report["close_adj_exactly_equals_close"] is True
    assert not any(item["severity"] == "error" for item in findings)

    broken = frame.copy()
    post = broken["datetime"].dt.date >= pd.Timestamp("2024-06-10").date()
    for column in ("open", "high", "low", "close", "close_adj"):
        broken.loc[post, column] /= 10.0
    report, findings = audit_adjustments(broken, [event])
    assert report["status"] == "fail"
    assert any(item["code"] == "documented_split_price_discontinuity" for item in findings)


def test_audit_flags_nonfinite_bad_candle_zero_volume_and_duplicate() -> None:
    calendar = pd.date_range("2024-01-02", periods=3, freq="B")
    frame = _bars(calendar)
    bad = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    bad.loc[1, "high"] = bad.loc[1, "low"] - 1.0
    bad.loc[2, "close"] = np.inf
    bad.loc[2, "volume"] = 0
    report = audit_daily_panel(
        bad,
        calendar_sessions=calendar,
        documented_splits=[],
        require_adjustment_check=False,
    )
    codes = {item["code"] for item in report["findings"]}
    assert report["acceptance"]["accepted"] is False
    assert {
        "identical_duplicate_symbol_session",
        "candle_inconsistent",
        "nonfinite_close",
        "zero_volume",
    }.issubset(codes)
