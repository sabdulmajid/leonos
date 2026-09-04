"""Pinned Twelve Data daily-bar retrieval and acceptance auditing.

The raw snapshot is immutable: the first fetch resolves one Hugging Face commit,
writes a manifest, and every later operation follows that manifest.  Auditing is
deliberately non-reconstructive: it never fills a missing bar or guesses a
corporate-action factor.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATASET_REPO_ID = "twelvedata/financial-world-model"
DATASET_CONFIG = "bars_1day"
SOURCE_TIMEZONE = "America/New_York"
EXPECTED_PARQUETS: dict[str, str] = {
    "train": "bars_1day/train.parquet",
    "val": "bars_1day/val.parquet",
    "test": "bars_1day/test.parquet",
}
DOWNLOAD_PATTERNS: tuple[str, ...] = (
    *EXPECTED_PARQUETS.values(),
    "README.md",
    "LICENSE",
    "LICENSE.*",
)
CORE_COLUMNS: tuple[str, ...] = (
    "datetime",
    "symbol",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_adj",
)
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "close_adj")
KEY_COLUMNS: tuple[str, ...] = ("symbol", "datetime")
MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA = "leonos.daily_snapshot.v1"
AUDIT_SCHEMA = "leonos.data_audit.v1"
_OID = re.compile(r"^[0-9a-f]{40,64}$")

# Independent issuer sources, not inferred from the downloaded price series.
# ``effective_session`` is the first session trading on a split-adjusted basis.
DEFAULT_DOCUMENTED_SPLITS: tuple[dict[str, Any], ...] = (
    {
        "symbol": "AAPL",
        "effective_session": "2020-08-31",
        "ratio_new_to_old": 4.0,
        "source": "https://investor.apple.com/dividend-history/default.aspx",
    },
    {
        "symbol": "NVDA",
        "effective_session": "2024-06-10",
        "ratio_new_to_old": 10.0,
        "source": (
            "https://investor.nvidia.com/news/press-release-details/2024/"
            "NVIDIA-Announces-Financial-Results-for-First-Quarter-Fiscal-2025/"
            "default.aspx"
        ),
    },
    {
        "symbol": "AVGO",
        "effective_session": "2024-07-15",
        "ratio_new_to_old": 10.0,
        "source": (
            "https://investors.broadcom.com/news-releases/news-release-details/"
            "broadcom-inc-announces-second-quarter-fiscal-year-2024-financial"
        ),
    },
)


class DataIntegrityError(ValueError):
    """Raised when raw data cannot be made deterministic without guessing."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    tmp.replace(path)


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading a file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def validate_immutable_revision(revision: str) -> str:
    """Reject branch names and tags; snapshots must use a full immutable OID."""

    revision = str(revision).strip().lower()
    if not _OID.fullmatch(revision):
        raise ValueError(
            "dataset revision must be a 40-64 character lowercase hexadecimal commit OID"
        )
    return revision


def resolve_dataset_revision(repo_id: str = DATASET_REPO_ID, *, api: Any | None = None) -> str:
    """Resolve the dataset's current commit once; callers persist the result."""

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    revision = getattr(api.dataset_info(repo_id=repo_id), "sha", None)
    if not revision:
        raise RuntimeError(f"Hugging Face returned no commit SHA for {repo_id}")
    return validate_immutable_revision(str(revision))


def _arrow_schema(path: Path) -> list[dict[str, Any]]:
    schema = pq.ParquetFile(path).schema_arrow
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in schema
    ]


def _parquet_summary(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    missing = {"datetime", "symbol"}.difference(names)
    if missing:
        raise DataIntegrityError(f"{path} is missing manifest columns: {sorted(missing)}")
    dates = pd.to_datetime(
        pq.read_table(path, columns=["datetime"]).column("datetime").to_pandas(),
        errors="coerce",
    )
    if dates.isna().any():
        raise DataIntegrityError(f"{path} contains unparseable datetimes")
    return {
        "row_count": int(parquet.metadata.num_rows),
        "row_groups": int(parquet.metadata.num_row_groups),
        "schema": _arrow_schema(path),
        "date_min": dates.min().isoformat() if len(dates) else None,
        "date_max": dates.max().isoformat() if len(dates) else None,
    }


def build_manifest(
    snapshot_root: str | Path,
    *,
    revision: str,
    repo_id: str = DATASET_REPO_ID,
    retrieved_at_utc: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect the three expected daily parquet files and write their manifest."""

    root = Path(snapshot_root)
    revision = validate_immutable_revision(revision)
    files: list[dict[str, Any]] = []
    for split, relative in EXPECTED_PARQUETS.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"expected pinned dataset file is missing: {path}")
        # Opening the footer and datetime column proves the file parses; hashing
        # binds all other columns without loading the whole table.
        item = {
            "path": relative,
            "kind": "parquet",
            "split": split,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        item.update(_parquet_summary(path))
        files.append(item)

    metadata_paths: list[Path] = []
    readme = root / "README.md"
    if readme.is_file():
        metadata_paths.append(readme)
    metadata_paths.extend(sorted(p for p in root.glob("LICENSE*") if p.is_file()))
    for path in metadata_paths:
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "kind": "metadata",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset": {
            "repo_id": repo_id,
            "repo_type": "dataset",
            "revision": revision,
            "config": DATASET_CONFIG,
        },
        "retrieved_at_utc": retrieved_at_utc or _utc_now(),
        "expected_parquets": dict(EXPECTED_PARQUETS),
        "files": files,
    }
    destination = Path(output_path) if output_path else root / MANIFEST_NAME
    _atomic_json(destination, manifest)
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a Leonos snapshot manifest."""

    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise DataIntegrityError(f"unsupported manifest schema in {path}")
    dataset = manifest.get("dataset", {})
    validate_immutable_revision(str(dataset.get("revision", "")))
    if manifest.get("expected_parquets") != EXPECTED_PARQUETS:
        raise DataIntegrityError("manifest does not declare exactly the daily split files")
    return manifest


def verify_manifest_files(
    snapshot_root: str | Path,
    manifest: Mapping[str, Any],
    *,
    verify_hashes: bool = True,
) -> None:
    """Verify expected paths, sizes, hashes, and parseability against a manifest."""

    root = Path(snapshot_root)
    entries = {str(entry["path"]): entry for entry in manifest.get("files", [])}
    for relative in EXPECTED_PARQUETS.values():
        entry = entries.get(relative)
        if entry is None:
            raise DataIntegrityError(f"manifest has no entry for {relative}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(entry["size_bytes"]):
            raise DataIntegrityError(f"size mismatch for {relative}")
        if verify_hashes and sha256_file(path) != entry["sha256"]:
            raise DataIntegrityError(f"SHA-256 mismatch for {relative}")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != int(entry["row_count"]):
            raise DataIntegrityError(f"row-count mismatch for {relative}")


def fetch_daily_snapshot(
    snapshot_root: str | Path,
    *,
    revision: str | None = None,
    repo_id: str = DATASET_REPO_ID,
    api: Any | None = None,
    snapshot_download_fn: Callable[..., str] | None = None,
    verify_existing_hashes: bool = True,
) -> Path:
    """Download only pinned daily bars and small metadata, then write a manifest.

    If a manifest already exists it is authoritative: ``main`` is not resolved
    again.  To intentionally select another revision, use a fresh snapshot root.
    """

    root = Path(snapshot_root)
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_file():
        manifest = load_manifest(manifest_path)
        recorded = manifest["dataset"]
        if recorded["repo_id"] != repo_id:
            raise DataIntegrityError(
                f"existing manifest is for {recorded['repo_id']}, not {repo_id}"
            )
        if revision is not None and validate_immutable_revision(revision) != recorded["revision"]:
            raise DataIntegrityError("requested revision differs from the existing manifest")
        verify_manifest_files(root, manifest, verify_hashes=verify_existing_hashes)
        return manifest_path

    pinned = (
        resolve_dataset_revision(repo_id, api=api)
        if revision is None
        else validate_immutable_revision(revision)
    )
    if snapshot_download_fn is None:
        from huggingface_hub import snapshot_download

        snapshot_download_fn = snapshot_download
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download_fn(
        repo_id=repo_id,
        repo_type="dataset",
        revision=pinned,
        allow_patterns=list(DOWNLOAD_PATTERNS),
        local_dir=str(root),
    )
    build_manifest(root, revision=pinned, repo_id=repo_id)
    return manifest_path


def load_daily_panel(
    snapshot_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    columns: Sequence[str] | None = None,
    verify_hashes: bool = True,
) -> pd.DataFrame:
    """Read only files named by the pinned manifest and attach source split."""

    root = Path(snapshot_root)
    path = Path(manifest_path) if manifest_path else root / MANIFEST_NAME
    manifest = load_manifest(path)
    verify_manifest_files(root, manifest, verify_hashes=verify_hashes)
    frames: list[pd.DataFrame] = []
    for split, relative in EXPECTED_PARQUETS.items():
        frame = pd.read_parquet(root / relative, columns=list(columns) if columns else None)
        if "source_split" in frame.columns:
            raise DataIntegrityError("source parquet unexpectedly contains reserved source_split")
        frame["source_split"] = split
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, copy=False)


def to_canonical_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Return downstream key aliases while retaining all native source columns."""

    missing = {"symbol", "datetime"}.difference(frame.columns)
    if missing:
        raise ValueError(f"bars missing source keys: {sorted(missing)}")
    if {"ticker", "session"}.intersection(frame.columns):
        raise ValueError("ticker/session are reserved Leonos aliases")
    out = frame.copy()
    out["ticker"] = out["symbol"]
    out["session"] = out["datetime"]
    return out


def apply_quality_policy(
    canonical_bars: pd.DataFrame,
    calendar_sessions: Sequence[object],
    *,
    fatal_ohlc: str = "raise",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the predeclared row policy and return bars plus an exclusion ledger.

    Non-exchange rows and nonpositive/nonfinite volume cannot enter either model
    and are deterministically excluded. Invalid prices or candle envelopes are
    fatal by default; ``fatal_ohlc="exclude"`` is available only for an explicit,
    documented scope decision. The input frame and raw parquet files are untouched.
    """

    if fatal_ohlc not in {"raise", "exclude"}:
        raise ValueError("fatal_ohlc must be 'raise' or 'exclude'")
    required = {"ticker", "session", "open", "high", "low", "close", "volume"}
    missing = required.difference(canonical_bars.columns)
    if missing:
        raise DataIntegrityError(f"canonical bars missing columns: {sorted(missing)}")
    frame = canonical_bars.copy()
    session = _session_series(frame["session"])
    if session.isna().any() or frame["ticker"].astype("string").isna().any():
        raise DataIntegrityError("ticker/session keys must be finite before quality policy")
    if frame.assign(_session=session).duplicated(["ticker", "_session"]).any():
        raise DataIntegrityError("duplicate ticker/session keys must be resolved before policy")

    calendar = _canonical_session_index(calendar_sessions)
    non_exchange = ~session.isin(calendar)
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    invalid_volume = ~np.isfinite(volume) | (volume <= 0)
    price_columns = [column for column in PRICE_COLUMNS if column in frame.columns]
    prices = {column: pd.to_numeric(frame[column], errors="coerce") for column in price_columns}
    invalid_price = pd.Series(False, index=frame.index)
    for values in prices.values():
        invalid_price |= ~np.isfinite(values) | (values <= 0)
    candle = (
        (prices["high"] < prices["low"])
        | (prices["high"] < prices["open"])
        | (prices["high"] < prices["close"])
        | (prices["low"] > prices["open"])
        | (prices["low"] > prices["close"])
    )
    fatal = invalid_price | candle
    if fatal.any() and fatal_ohlc == "raise":
        keys = frame.loc[fatal, ["ticker", "session"]].head(5).to_dict("records")
        raise DataIntegrityError(
            f"fatal OHLC rows require an explicit scope decision; count={int(fatal.sum())}, "
            f"sample={keys}"
        )

    reasons: list[dict[str, Any]] = []
    masks: list[tuple[str, pd.Series]] = [
        ("non_exchange_session", non_exchange),
        ("nonpositive_or_nonfinite_volume", invalid_volume),
    ]
    if fatal_ohlc == "exclude":
        masks.extend(
            [
                ("nonpositive_or_nonfinite_price", invalid_price),
                ("candle_inconsistent", candle),
            ]
        )
    key_columns = [column for column in ("ticker", "session", "source_split") if column in frame]
    for reason, mask in masks:
        for row_index, row in frame.loc[mask, key_columns].iterrows():
            record = {"row_index": int(row_index), "reason": reason}
            for column in key_columns:
                value = row[column]
                record[column] = (
                    pd.Timestamp(value).isoformat() if column == "session" else str(value)
                )
            reasons.append(record)
    exclusions = pd.DataFrame.from_records(
        reasons, columns=["row_index", *key_columns, "reason"]
    ).sort_values(["row_index", "reason"], kind="stable", ignore_index=True)
    excluded = non_exchange | invalid_volume | (fatal if fatal_ohlc == "exclude" else False)
    accepted = frame.loc[~excluded].reset_index(drop=True)
    return accepted, exclusions


def _parse_datetimes(values: pd.Series) -> tuple[pd.Series, str | None, bool]:
    parsed = pd.to_datetime(values, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        source_tz = str(parsed.dt.tz)
        return parsed.dt.tz_convert(SOURCE_TIMEZONE), source_tz, False
    if pd.api.types.is_datetime64_dtype(parsed.dtype):
        return parsed.dt.tz_localize(SOURCE_TIMEZONE), None, True
    # Mixed-offset objects are converted through UTC. This keeps the instant but
    # is reported as a noncanonical source representation by the caller.
    converted = pd.to_datetime(values, errors="coerce", utc=True)
    return converted.dt.tz_convert(SOURCE_TIMEZONE), "mixed-or-object", False


def _all_rows_identical(group: pd.DataFrame, columns: Sequence[str]) -> bool:
    first = group.iloc[0]
    for _, row in group.iloc[1:].iterrows():
        for column in columns:
            left, right = first[column], row[column]
            if pd.isna(left) and pd.isna(right):
                continue
            equal = left == right
            if pd.isna(equal) or not bool(equal):
                return False
    return True


def normalize_daily_bars(
    frame: pd.DataFrame, *, drop_identical_duplicates: bool = False
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply deterministic types/order/symbol normalization without filling bars.

    Conflicting duplicate keys always raise. Identical duplicates are removable
    only when explicitly requested, and the returned log records the exact rule.
    """

    missing = set(CORE_COLUMNS).difference(frame.columns)
    if missing:
        raise DataIntegrityError(f"daily bars missing required columns: {sorted(missing)}")
    out = frame.copy()
    original_symbols = out["symbol"].astype("string")
    out["symbol"] = original_symbols.str.strip().str.upper()
    symbol_changes = int((original_symbols != out["symbol"]).fillna(False).sum())
    parsed, original_tz, localized_naive = _parse_datetimes(out["datetime"])
    out["datetime"] = parsed
    for column in (*PRICE_COLUMNS, "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["timeframe"] = out["timeframe"].astype("string").str.strip().str.lower()
    out = out.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)

    duplicate = out.duplicated(list(KEY_COLUMNS), keep=False)
    identical_keys: list[dict[str, str]] = []
    conflicting_keys: list[dict[str, str]] = []
    compare_columns = [c for c in out.columns if c not in KEY_COLUMNS]
    if duplicate.any():
        for (symbol, timestamp), group in out.loc[duplicate].groupby(
            list(KEY_COLUMNS), sort=True, dropna=False
        ):
            key = {"symbol": str(symbol), "datetime": str(timestamp)}
            if _all_rows_identical(group, compare_columns):
                identical_keys.append(key)
            else:
                conflicting_keys.append(key)
    if conflicting_keys:
        raise DataIntegrityError(
            f"conflicting duplicate ticker/session keys: {conflicting_keys[:5]}"
        )
    if identical_keys and not drop_identical_duplicates:
        raise DataIntegrityError(
            "identical duplicates found; rerun with drop_identical_duplicates=True "
            f"to apply the logged keep-first rule: {identical_keys[:5]}"
        )
    before = len(out)
    if identical_keys:
        out = out.drop_duplicates(list(KEY_COLUMNS), keep="first").reset_index(drop=True)
    log = {
        "symbol_normalization": "strip_then_uppercase",
        "symbol_values_changed": symbol_changes,
        "datetime_source_timezone": original_tz,
        "naive_datetimes_localized_to": SOURCE_TIMEZONE if localized_naive else None,
        "sorting": [*KEY_COLUMNS],
        "identical_duplicate_policy": "keep_first_after_stable_key_sort"
        if drop_identical_duplicates
        else "reject",
        "identical_duplicate_keys": len(identical_keys),
        "rows_removed": before - len(out),
        "no_market_values_filled": True,
    }
    return out, log


def exchange_sessions(
    start: object, end: object, *, calendar_name: str = "XNYS"
) -> pd.DatetimeIndex:
    """Resolve official exchange sessions; never substitute weekday dates."""

    try:
        import exchange_calendars as xcals
    except ImportError as exc:  # pragma: no cover - exercised in minimal envs
        raise RuntimeError(
            "exchange-calendars is required for production session auditing"
        ) from exc
    start_date = pd.Timestamp(start).date().isoformat()
    end_date = pd.Timestamp(end).date().isoformat()
    # exchange-calendars otherwise materializes only its default recent window,
    # while the daily source reaches back to 1970.
    calendar = xcals.get_calendar(calendar_name, start=start_date, end=end_date)
    sessions = calendar.sessions
    sessions = sessions[
        (sessions >= pd.Timestamp(start_date)) & (sessions <= pd.Timestamp(end_date))
    ]
    return _canonical_session_index(sessions)


def _canonical_session_index(values: Iterable[object]) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(list(values), errors="raise", utc=True)
    result = pd.DatetimeIndex(parsed).tz_convert(None).normalize().unique().sort_values()
    if result.has_duplicates:
        raise ValueError("calendar contains duplicate sessions")
    return result


def _session_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()


def _json_number(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def audit_adjustments(
    frame: pd.DataFrame,
    documented_splits: Sequence[Mapping[str, Any]] | None,
    *,
    log_tolerance: float = 0.30,
    require_check: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Check price continuity around independently documented stock splits.

    Each split mapping requires ``symbol``, ``effective_session`` and
    ``ratio_new_to_old``. The source snapshot presents all price columns on a
    retroactively split-consistent basis, so a documented split must *not* leave
    the declared ratio-sized price jump. Ratios based on ``close_adj / close``
    are reported only as diagnostics; they are never applied to OHLCV or volume.
    """

    findings: list[dict[str, Any]] = []
    close = pd.to_numeric(frame["close"], errors="coerce")
    adjusted = pd.to_numeric(frame["close_adj"], errors="coerce")
    diagnostic_factor = adjusted / close
    finite_factor = diagnostic_factor[np.isfinite(diagnostic_factor)]
    close_alias = bool(
        len(frame)
        and np.array_equal(
            close.to_numpy(dtype=np.float64, na_value=np.nan),
            adjusted.to_numpy(dtype=np.float64, na_value=np.nan),
            equal_nan=True,
        )
    )
    report: dict[str, Any] = {
        "close_adj_over_close_diagnostic_only": {
            "count": int(len(finite_factor)),
            "min": _json_number(finite_factor.min()) if len(finite_factor) else None,
            "max": _json_number(finite_factor.max()) if len(finite_factor) else None,
            "equal_to_one_count": int(np.isclose(finite_factor, 1.0).sum()),
            "warning": (
                "This ratio is not assumed to be a valid split factor; it may include "
                "dividend adjustment and must not be inverted into volume."
            ),
        },
        "close_adj_exactly_equals_close": close_alias,
        "published_dividend_adjustment_claim_supported": False,
        "observed_price_convention": (
            "source OHLC prices are tested as retroactively split-consistent; "
            "close_adj is an exact close alias in this snapshot"
            if close_alias
            else "close_adj differs from close and requires separate investigation"
        ),
        "documented_split_checks": [],
        "channel_policy": {
            "model_prices": "use the source OHLC channels together without substitution",
            "portfolio_returns": "adjusted-price convention; dividends are not added separately",
            "close_adj_with_other_ohl": "not used; close_adj carries no independent evidence here",
            "volume": "source volume preserved exactly; no factor inversion",
        },
    }
    if close_alias:
        findings.append(
            {
                "severity": "warning",
                "code": "close_adj_is_close_alias",
                "count": int(len(frame)),
                "detail": (
                    "the collection fallback sets close_adj=close when the API omits it; "
                    "the card's dividend-adjusted claim is unsupported by a distinct series"
                ),
            }
        )
    events = list(documented_splits or [])
    if require_check and not events:
        findings.append(
            {
                "severity": "error",
                "code": "adjustment_check_missing",
                "count": 1,
                "detail": "at least one independently documented split check is required",
            }
        )
    sessions = _session_series(frame["datetime"])
    symbols = frame["symbol"].astype("string")
    for event in events:
        symbol = str(event["symbol"]).strip().upper()
        effective = _canonical_session_index([event["effective_session"]])[0]
        ratio = float(event["ratio_new_to_old"])
        if not math.isfinite(ratio) or ratio <= 1.0:
            raise ValueError(f"invalid split ratio for {symbol}: {ratio}")
        rows = frame.loc[symbols.eq(symbol)].copy()
        rows["_session"] = sessions.loc[rows.index]
        rows = rows.sort_values("_session", kind="stable")
        prior = rows.loc[rows["_session"] < effective].tail(1)
        current = rows.loc[rows["_session"] == effective].head(1)
        result: dict[str, Any] = {
            "symbol": symbol,
            "effective_session": effective.date().isoformat(),
            "ratio_new_to_old": ratio,
            "source": event.get("source"),
        }
        if prior.empty or current.empty:
            result["status"] = "missing_rows"
            findings.append(
                {
                    "severity": "error",
                    "code": "documented_split_rows_missing",
                    "count": 1,
                    "detail": f"{symbol} lacks pre/effective rows for {effective.date()}",
                }
            )
            report["documented_split_checks"].append(result)
            continue
        pre = prior.iloc[0]
        post = current.iloc[0]
        raw_ratio = float(pre["close"]) / float(post["close"])
        adj_ratio = float(pre["close_adj"]) / float(post["close_adj"])
        price_continuous = bool(abs(math.log(raw_ratio)) <= log_tolerance)
        unadjusted_drop = bool(abs(math.log(raw_ratio / ratio)) <= log_tolerance)
        adjusted_continuous = bool(abs(math.log(adj_ratio)) <= log_tolerance)
        result.update(
            {
                "prior_session": pd.Timestamp(pre["_session"]).date().isoformat(),
                "raw_pre_close_over_effective_close": raw_ratio,
                "adjusted_pre_close_over_effective_close": adj_ratio,
                "source_close_is_split_consistent": price_continuous,
                "unadjusted_ratio_sized_drop_detected": unadjusted_drop,
                "adjusted_close_is_continuous": adjusted_continuous,
                "pre_volume": _json_number(pre["volume"]),
                "effective_volume": _json_number(post["volume"]),
                "status": "pass"
                if price_continuous and adjusted_continuous and not unadjusted_drop
                else "fail",
            }
        )
        if not price_continuous or unadjusted_drop:
            findings.append(
                {
                    "severity": "error",
                    "code": "documented_split_price_discontinuity",
                    "count": 1,
                    "detail": (
                        f"{symbol} source prices are not continuous across the documented "
                        f"{ratio}:1 split"
                    ),
                }
            )
        if not adjusted_continuous:
            findings.append(
                {
                    "severity": "error",
                    "code": "documented_split_adjusted_discontinuity",
                    "count": 1,
                    "detail": f"{symbol} close_adj remains discontinuous across documented split",
                }
            )
        report["documented_split_checks"].append(result)
    report["status"] = (
        "pass"
        if events and all(x.get("status") == "pass" for x in report["documented_split_checks"])
        else "fail"
        if events or require_check
        else "not_checked"
    )
    return report, findings


def audit_daily_panel(
    frame: pd.DataFrame,
    *,
    calendar_sessions: Sequence[object] | None = None,
    documented_splits: Sequence[Mapping[str, Any]] | None = DEFAULT_DOCUMENTED_SPLITS,
    context_sessions: int = 90,
    horizon_sessions: int = 10,
    require_adjustment_check: bool = True,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Produce a JSON-safe, bounded acceptance audit for a daily panel."""

    if context_sessions < 1 or horizon_sessions < 1:
        raise ValueError("context_sessions and horizon_sessions must be positive")
    findings: list[dict[str, Any]] = []

    def finding(severity: str, code: str, count: int, detail: str) -> None:
        if count:
            findings.append(
                {"severity": severity, "code": code, "count": int(count), "detail": detail}
            )

    missing = sorted(set(CORE_COLUMNS).difference(frame.columns))
    report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "created_at_utc": created_at_utc or _utc_now(),
        "configuration": {
            "calendar": "XNYS",
            "context_sessions": context_sessions,
            "horizon_sessions": horizon_sessions,
            "source_timezone": SOURCE_TIMEZONE,
        },
        "summary": {"rows": int(len(frame)), "required_columns_missing": missing},
        "findings": findings,
    }
    if missing:
        finding("error", "required_columns_missing", len(missing), ", ".join(missing))
        report["acceptance"] = {"accepted": False, "error_count": 1, "warning_count": 0}
        return report

    work = frame.copy()
    work["symbol"] = work["symbol"].astype("string")
    parsed, original_tz, localized_naive = _parse_datetimes(work["datetime"])
    invalid_datetime = int(parsed.isna().sum())
    finding("error", "invalid_datetime", invalid_datetime, "unparseable source timestamps")
    if localized_naive:
        finding(
            "error",
            "timezone_naive",
            len(work),
            f"source card promises tz-aware {SOURCE_TIMEZONE} timestamps",
        )
    elif original_tz != SOURCE_TIMEZONE:
        finding(
            "error",
            "unexpected_timezone",
            len(work),
            f"observed {original_tz!r}; expected {SOURCE_TIMEZONE!r}",
        )
    work["datetime"] = parsed
    work["_session"] = _session_series(parsed)
    report["timestamps"] = {
        "observed_timezone": original_tz,
        "timezone_expected": SOURCE_TIMEZONE,
        "session_date_rule": "calendar date in America/New_York",
        "source_bar_timestamp_semantics": "interval start",
        "information_available": "daily OHLCV is treated as known only after that session closes",
        "distinct_local_times": sorted(
            {timestamp.strftime("%H:%M:%S") for timestamp in parsed.dropna().head(10000)}
        ),
    }
    wrong_timeframe = int(work["timeframe"].astype("string").str.lower().ne("1day").sum())
    finding("error", "non_daily_rows", wrong_timeframe, "only timeframe=1day is permitted")
    null_symbols = int(work["symbol"].isna().sum())
    finding("error", "missing_symbol", null_symbols, "symbol keys may not be null")

    duplicate = work.duplicated(["symbol", "_session"], keep=False)
    identical_duplicate_keys = 0
    conflicting_duplicate_keys = 0
    if duplicate.any():
        compare_columns = [c for c in work.columns if c not in {"symbol", "_session"}]
        for _, group in work.loc[duplicate].groupby(
            ["symbol", "_session"], sort=True, dropna=False
        ):
            if _all_rows_identical(group, compare_columns):
                identical_duplicate_keys += 1
            else:
                conflicting_duplicate_keys += 1
    duplicate_keys = identical_duplicate_keys + conflicting_duplicate_keys
    finding(
        "error",
        "conflicting_duplicate_symbol_session",
        conflicting_duplicate_keys,
        "same key has different source values; deterministic repair is not allowed",
    )
    finding(
        "error",
        "identical_duplicate_symbol_session",
        identical_duplicate_keys,
        "explicitly run normalize_daily_bars(..., drop_identical_duplicates=True)",
    )

    numeric: dict[str, pd.Series] = {}
    for column in (*PRICE_COLUMNS, "volume"):
        numeric[column] = pd.to_numeric(work[column], errors="coerce")
        invalid = int((~np.isfinite(numeric[column])).sum())
        finding("error", f"nonfinite_{column}", invalid, f"{column} must be finite")
    for column in PRICE_COLUMNS:
        nonpositive = int((numeric[column] <= 0).sum())
        finding("error", f"nonpositive_{column}", nonpositive, f"{column} must be positive")
    negative_volume = int((numeric["volume"] < 0).sum())
    zero_volume = int((numeric["volume"] == 0).sum())
    finding("error", "negative_volume", negative_volume, "volume must not be negative")
    finding("error", "zero_volume", zero_volume, "zero-volume daily bars require exclusion/review")
    candle_bad = (
        (numeric["high"] < numeric["low"])
        | (numeric["high"] < numeric["open"])
        | (numeric["high"] < numeric["close"])
        | (numeric["low"] > numeric["open"])
        | (numeric["low"] > numeric["close"])
    )
    finding("error", "candle_inconsistent", int(candle_bad.sum()), "raw OHLC envelope violation")
    report["quality_counts"] = {
        "duplicate_keys": duplicate_keys,
        "identical_duplicate_keys": identical_duplicate_keys,
        "conflicting_duplicate_keys": conflicting_duplicate_keys,
        "zero_volume_rows": zero_volume,
        "candle_inconsistent_rows": int(candle_bad.sum()),
        "nonfinite_by_column": {
            column: int((~np.isfinite(values)).sum()) for column, values in numeric.items()
        },
        "nonpositive_by_price_column": {
            column: int((values <= 0).sum())
            for column, values in numeric.items()
            if column != "volume"
        },
    }

    valid_dates = work["_session"].dropna()
    if calendar_sessions is None and len(valid_dates):
        try:
            calendar = exchange_sessions(valid_dates.min(), valid_dates.max())
        except RuntimeError as exc:
            calendar = pd.DatetimeIndex([])
            finding("error", "calendar_unavailable", 1, str(exc))
    else:
        calendar = _canonical_session_index([] if calendar_sessions is None else calendar_sessions)
    calendar_set = set(calendar)
    observed_set = set(valid_dates)
    unexpected = sorted(observed_set.difference(calendar_set)) if len(calendar) else []
    finding(
        "error",
        "non_exchange_sessions",
        len(unexpected),
        f"sample={[_date_string(x) for x in unexpected[:5]]}",
    )
    report["calendar"] = {
        "session_count": int(len(calendar)),
        "date_min": _date_string(calendar.min()) if len(calendar) else None,
        "date_max": _date_string(calendar.max()) if len(calendar) else None,
        "unexpected_observed_sessions": [_date_string(x) for x in unexpected[:20]],
        "weekday_fallback_used": False,
    }

    source_split = (
        work["source_split"].astype("string")
        if "source_split" in work.columns
        else pd.Series("all", index=work.index, dtype="string")
    )
    work["_source_split"] = source_split
    coverage: dict[str, Any] = {}
    eligibility: dict[str, Any] = {}
    global_min = min(observed_set) if observed_set else None
    global_max = max(observed_set) if observed_set else None
    per_ticker: dict[str, Any] = {}
    for symbol, group in work.groupby("symbol", sort=True, observed=True):
        symbol_sessions = set(group["_session"].dropna())
        first = min(symbol_sessions) if symbol_sessions else None
        last = max(symbol_sessions) if symbol_sessions else None
        if first is not None and last is not None and len(calendar):
            observed_positions = calendar.get_indexer(pd.DatetimeIndex(sorted(symbol_sessions)))
            observed_positions = observed_positions[observed_positions >= 0]
            present = np.zeros(len(calendar), dtype=bool)
            present[observed_positions] = True
            first_position = int(observed_positions.min())
            last_position = int(observed_positions.max())
            missing_positions = np.flatnonzero(~present[first_position : last_position + 1])
            missing_positions += first_position
            missing_sessions = calendar[missing_positions].tolist()

            cumulative = np.concatenate(([0], np.cumsum(present, dtype=np.int64)))
            positions = np.arange(len(calendar), dtype=np.int64)
            context_complete = np.zeros(len(calendar), dtype=bool)
            context_possible = positions >= context_sessions - 1
            context_pos = positions[context_possible]
            context_complete[context_pos] = (
                cumulative[context_pos + 1] - cumulative[context_pos - context_sessions + 1]
                == context_sessions
            )
            label_complete = np.zeros(len(calendar), dtype=bool)
            label_possible = positions + horizon_sessions < len(calendar)
            label_pos = positions[label_possible]
            label_complete[label_pos] = (
                cumulative[label_pos + horizon_sessions + 1] - cumulative[label_pos + 1]
                == horizon_sessions
            )
        else:
            missing_sessions = []
            context_complete = np.zeros(len(calendar), dtype=bool)
            label_complete = np.zeros(len(calendar), dtype=bool)
        per_ticker[str(symbol)] = {
            "rows": int(len(group)),
            "first_session": _date_string(first),
            "last_session": _date_string(last),
            "missing_exchange_sessions": len(missing_sessions),
            "missing_session_sample": [_date_string(x) for x in missing_sessions[:20]],
            "starts_after_panel": bool(first and global_min and first > global_min),
            "ends_before_panel": bool(last and global_max and last < global_max),
            "by_split": {},
        }
        if missing_sessions:
            finding(
                "warning",
                "ticker_missing_sessions",
                len(missing_sessions),
                f"{symbol}: missing within observed lifetime; no fill will be applied",
            )

        for split, split_group in group.groupby("_source_split", sort=True, observed=True):
            key = f"{split}/{symbol}"
            split_sessions = split_group["_session"].dropna()
            per_ticker[str(symbol)]["by_split"][str(split)] = {
                "rows": int(len(split_group)),
                "first_session": _date_string(split_sessions.min())
                if len(split_sessions)
                else None,
                "last_session": _date_string(split_sessions.max()) if len(split_sessions) else None,
            }
            split_positions = calendar.get_indexer(
                pd.DatetimeIndex(split_sessions.drop_duplicates().sort_values())
            )
            split_positions = split_positions[split_positions >= 0]
            context_count = int(context_complete[split_positions].sum())
            label_count = int(label_complete[split_positions].sum())
            both_count = int(
                (context_complete[split_positions] & label_complete[split_positions]).sum()
            )
            eligibility[key] = {
                "origins": int(split_sessions.nunique()),
                "usable_90_session_contexts": context_count,
                "complete_10_session_labels": label_count,
                "usable_context_and_label": both_count,
            }

    for split, group in work.groupby("_source_split", sort=True, observed=True):
        dates = group["_session"].dropna()
        coverage[str(split)] = {
            "rows": int(len(group)),
            "ticker_count": int(group["symbol"].nunique()),
            "date_min": _date_string(dates.min()) if len(dates) else None,
            "date_max": _date_string(dates.max()) if len(dates) else None,
        }
    report["coverage_by_split"] = coverage
    report["coverage_by_ticker"] = per_ticker
    report["origin_eligibility"] = eligibility
    report["coverage_boundary_interpretation"] = (
        "Starts/ends are reported, but bars alone cannot distinguish IPO, rename, or delisting."
    )

    if "source_split" in work.columns:
        bounds = {
            "train": (None, pd.Timestamp("2023-12-31")),
            "val": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
            "test": (pd.Timestamp("2025-01-01"), None),
        }
        for split, (lower, upper) in bounds.items():
            mask = work["_source_split"].eq(split)
            dates = work.loc[mask, "_session"]
            bad = pd.Series(False, index=dates.index)
            if lower is not None:
                bad |= dates < lower
            if upper is not None:
                bad |= dates > upper
            finding(
                "error",
                "source_split_boundary_violation",
                int(bad.sum()),
                f"{split} rows fall outside published source boundaries",
            )

    adjustment_report, adjustment_findings = audit_adjustments(
        work,
        documented_splits,
        require_check=require_adjustment_check,
    )
    findings.extend(adjustment_findings)
    report["adjustments"] = adjustment_report
    report["summary"].update(
        {
            "ticker_count": int(work["symbol"].nunique()),
            "date_min": _date_string(valid_dates.min()) if len(valid_dates) else None,
            "date_max": _date_string(valid_dates.max()) if len(valid_dates) else None,
        }
    )
    errors = sum(item["severity"] == "error" for item in findings)
    warnings = sum(item["severity"] == "warning" for item in findings)
    report["acceptance"] = {
        "accepted": errors == 0,
        "error_count": errors,
        "warning_count": warnings,
    }
    return report


def _date_string(value: object | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def render_audit_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human companion to the machine-readable audit."""

    acceptance = report.get("acceptance", {})
    summary = report.get("summary", {})
    lines = [
        "# Daily data acceptance",
        "",
        f"Status: **{'PASS' if acceptance.get('accepted') else 'FAIL'}**",
        "",
        (
            f"Rows: {summary.get('rows', 0):,}; tickers: "
            f"{summary.get('ticker_count', 0)}; dates: {summary.get('date_min')} to "
            f"{summary.get('date_max')}."
        ),
        "",
        "Daily timestamps denote interval starts; each bar is usable only after its "
        "session close. Missing market bars were not filled.",
        "",
        "## Findings",
        "",
    ]
    findings = report.get("findings", [])
    if not findings:
        lines.append("No findings.")
    else:
        for item in findings:
            lines.append(
                f"- {str(item['severity']).upper()} `{item['code']}` "
                f"({item['count']}): {item['detail']}"
            )
    lines.extend(
        [
            "",
            "## Adjustment basis",
            "",
            "Documented split checks: "
            f"{report.get('adjustments', {}).get('status', 'not_checked')}. "
            "`close_adj / close` is diagnostic only and was not applied to OHLCV or volume.",
            "",
        ]
    )
    return "\n".join(lines)


def write_audit_reports(
    report: Mapping[str, Any], json_path: str | Path, markdown_path: str | Path
) -> None:
    """Atomically persist the audit JSON plus a concise Markdown summary."""

    _atomic_json(Path(json_path), report)
    markdown_path = Path(markdown_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=markdown_path.parent,
        prefix=f".{markdown_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp = Path(handle.name)
        handle.write(render_audit_markdown(report))
    tmp.replace(markdown_path)
