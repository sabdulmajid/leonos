"""Frozen Kronos-base inference adapter and resumable prediction shards.

The adapter deliberately accepts *future timestamps*, never future market values.
When ``amount`` is absent, it is left absent so the pinned official predictor applies
its documented proxy ``volume * mean(open, high, low, close)``.  That value is an
estimated turnover proxy derived entirely from OHLCV, not observed dollar turnover.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

KRONOS_IMPLEMENTATION_URL = "https://github.com/shiyu-coder/Kronos"
KRONOS_IMPLEMENTATION_REVISION = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
KRONOS_MODEL_ID = "NeoQuasar/Kronos-base"
KRONOS_MODEL_REVISION = "2b554741eca47781b64468546e77fef3e85130e6"
KRONOS_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
KRONOS_TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"

MODEL_NAME = "kronos"
CONTEXT_SESSIONS = 90
FORECAST_SESSIONS = 10
PRICE_COLUMNS = ("open", "high", "low", "close")
INPUT_COLUMNS = (*PRICE_COLUMNS, "volume")

# The official implementation treats top_k=0 as disabled.
INVESTMENT_TEMPERATURE = 0.6
INVESTMENT_TOP_P = 0.9
INVESTMENT_TOP_K = 0
INVESTMENT_SAMPLE_COUNT = 10

AMOUNT_PROXY_POLICY = (
    "official predictor estimate: volume * mean(open, high, low, close); "
    "derived OHLCV turnover proxy, not observed dollar turnover"
)
OUTPUT_SEMANTICS = (
    "official predictor averages decoded sampled trajectories; output is a point "
    "path, not a calibrated quantile distribution"
)

PREDICTION_KEY_COLUMNS = ("model", "seed", "ticker", "origin", "horizon")
PREDICTION_COLUMNS = (
    *PREDICTION_KEY_COLUMNS,
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
    "amount_source",
    "output_semantics",
    "batch_elapsed_seconds",
)
_LABEL_COLUMN_NAMES = frozenset(
    {"label", "labels", "target", "targets", "actual", "actual_y", "realized", "y"}
)


class KronosContractError(ValueError):
    """The request, configuration, or official prediction violated the contract."""


class PredictionShardError(ValueError):
    """A prediction shard is malformed, incomplete, or duplicates another shard."""


class OfficialPredictor(Protocol):
    """Small surface used from the pinned upstream ``KronosPredictor``."""

    model: Any
    tokenizer: Any

    def predict_batch(
        self,
        *,
        df_list: list[pd.DataFrame],
        x_timestamp_list: list[pd.Series],
        y_timestamp_list: list[pd.Series],
        pred_len: int,
        T: float,
        top_k: int,
        top_p: float,
        sample_count: int,
        verbose: bool,
    ) -> list[pd.DataFrame]: ...


@dataclass(frozen=True)
class KronosInferenceConfig:
    """The predeclared Leonos v1 investment-inference contract."""

    seed: int = 42
    context_sessions: int = CONTEXT_SESSIONS
    forecast_sessions: int = FORECAST_SESSIONS
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
        fixed = {
            "context_sessions": (self.context_sessions, CONTEXT_SESSIONS),
            "forecast_sessions": (self.forecast_sessions, FORECAST_SESSIONS),
            "temperature": (self.temperature, INVESTMENT_TEMPERATURE),
            "top_p": (self.top_p, INVESTMENT_TOP_P),
            "top_k": (self.top_k, INVESTMENT_TOP_K),
            "sample_count": (self.sample_count, INVESTMENT_SAMPLE_COUNT),
            "model_id": (self.model_id, KRONOS_MODEL_ID),
            "tokenizer_id": (self.tokenizer_id, KRONOS_TOKENIZER_ID),
        }
        changed = [name for name, (actual, expected) in fixed.items() if actual != expected]
        if changed:
            raise KronosContractError(f"Leonos v1 Kronos settings are fixed; changed: {changed}")
        if self.seed < 0:
            raise KronosContractError("seed must be non-negative")
        for name, revision in (
            ("model_revision", self.model_revision),
            ("tokenizer_revision", self.tokenizer_revision),
            ("implementation_revision", self.implementation_revision),
        ):
            if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                raise KronosContractError(f"{name} must be a full lowercase 40-hex revision")


@dataclass(frozen=True)
class ForecastOriginKey:
    """Identity of one forecast origin before its ten horizon rows are expanded."""

    ticker: str
    origin: pd.Timestamp

    def canonical(self) -> tuple[str, str]:
        return (pd.Timestamp(self.origin).isoformat(), str(self.ticker))


@dataclass(frozen=True)
class ForecastRequest:
    """Past OHLCV plus future calendar only; realized future values are impossible."""

    ticker: str
    origin: pd.Timestamp
    history: pd.DataFrame
    history_dates: Sequence[Any]
    forecast_dates: Sequence[Any]

    @property
    def key(self) -> ForecastOriginKey:
        return ForecastOriginKey(self.ticker, pd.Timestamp(self.origin))


def _datetime_index(values: Sequence[Any], name: str) -> pd.DatetimeIndex:
    try:
        result = pd.DatetimeIndex(pd.to_datetime(values))
    except (TypeError, ValueError) as exc:
        raise KronosContractError(f"{name} must contain parseable timestamps") from exc
    if result.hasnans:
        raise KronosContractError(f"{name} contains missing timestamps")
    if result.has_duplicates or not result.is_monotonic_increasing:
        raise KronosContractError(f"{name} must be strictly increasing and unique")
    return result


def validate_request(request: ForecastRequest) -> None:
    """Validate shape, chronology, and finite historical OHLCV values."""

    if not request.ticker or not str(request.ticker).strip():
        raise KronosContractError("ticker must be non-empty")
    if len(request.history) != CONTEXT_SESSIONS:
        raise KronosContractError(
            f"history must contain exactly {CONTEXT_SESSIONS} completed sessions"
        )
    missing = sorted(set(INPUT_COLUMNS).difference(request.history.columns))
    if missing:
        raise KronosContractError(f"history is missing required OHLCV columns: {missing}")

    history_dates = _datetime_index(request.history_dates, "history_dates")
    forecast_dates = _datetime_index(request.forecast_dates, "forecast_dates")
    if len(history_dates) != CONTEXT_SESSIONS or len(history_dates) != len(request.history):
        raise KronosContractError("history_dates must align one-for-one with history")
    if len(forecast_dates) != FORECAST_SESSIONS:
        raise KronosContractError(
            f"forecast_dates must contain exactly {FORECAST_SESSIONS} exchange sessions"
        )
    origin = pd.Timestamp(request.origin)
    if origin != history_dates[-1]:
        raise KronosContractError("origin must equal the last completed history session")
    if forecast_dates[0] <= origin:
        raise KronosContractError("all forecast timestamps must be after the origin")

    numeric_columns = list(INPUT_COLUMNS)
    if "amount" in request.history:
        numeric_columns.append("amount")
    try:
        values = request.history[numeric_columns].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise KronosContractError("historical OHLCV must be numeric") from exc
    if not np.isfinite(values).all():
        raise KronosContractError("historical OHLCV contains non-finite values")
    if (request.history[list(PRICE_COLUMNS)].to_numpy(dtype=np.float64) <= 0).any():
        raise KronosContractError("historical prices must be positive")
    if (request.history["volume"].to_numpy(dtype=np.float64) < 0).any():
        raise KronosContractError("historical volume must be non-negative")
    if (
        "amount" in request.history
        and (request.history["amount"].to_numpy(dtype=np.float64) < 0).any()
    ):
        raise KronosContractError("historical amount must be non-negative")


def canonicalize_requests(requests: Iterable[ForecastRequest]) -> list[ForecastRequest]:
    """Validate and put requests in the frozen origin-then-ticker batch order."""

    ordered = list(requests)
    for request in ordered:
        validate_request(request)
    ordered.sort(key=lambda request: request.key.canonical())
    keys = [request.key.canonical() for request in ordered]
    if len(keys) != len(set(keys)):
        raise KronosContractError("duplicate (ticker, origin) forecast requests")
    return ordered


def logical_shard_identity(split: str, seed: int, ordered_keys: Sequence[ForecastOriginKey]) -> str:
    """Hash a canonical key slice; GPU assignment is intentionally absent."""

    if not split or not split.strip():
        raise KronosContractError("split must be non-empty")
    canonical_keys = sorted(key.canonical() for key in ordered_keys)
    if len(canonical_keys) != len(set(canonical_keys)):
        raise KronosContractError("logical shard contains duplicate origin keys")
    payload = {
        "contract": "leonos-kronos-shard-v1",
        "split": split,
        "seed": int(seed),
        "keys": canonical_keys,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


def derive_effective_seed(seed: int, logical_shard_id: str) -> int:
    """Derive a stable per-logical-batch seed, independent of process/GPU."""

    payload = f"leonos-kronos-seed-v1\0{int(seed)}\0{logical_shard_id}".encode()
    # Keep within the range accepted by NumPy and all supported torch generators.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32 - 1)


def set_global_inference_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch immediately before one logical batch."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - production dependency error
        raise RuntimeError("torch is required for Kronos inference") from exc
    torch.manual_seed(seed)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


def _freeze_module(module: Any, name: str) -> None:
    if module is None:
        raise KronosContractError(f"official predictor has no {name}")
    if not callable(getattr(module, "eval", None)):
        raise KronosContractError(f"official predictor {name} has no eval()")
    module.eval()
    requires_grad = getattr(module, "requires_grad_", None)
    if callable(requires_grad):
        requires_grad(False)
    else:
        parameters = getattr(module, "parameters", None)
        if not callable(parameters):
            raise KronosContractError(f"official predictor {name} cannot be frozen")
        for parameter in parameters():
            parameter.requires_grad_(False)


def freeze_official_predictor(predictor: OfficialPredictor) -> None:
    """Put both released modules in evaluation mode and disable gradients."""

    _freeze_module(getattr(predictor, "model", None), "model")
    _freeze_module(getattr(predictor, "tokenizer", None), "tokenizer")


def _verify_checkout_revision(source_root: Path, expected_revision: str) -> None:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise KronosContractError(
            "official implementation must be an inspectable git checkout"
        ) from exc
    actual = result.stdout.strip()
    if actual != expected_revision:
        raise KronosContractError(
            f"official implementation revision mismatch: expected {expected_revision}, got {actual}"
        )
    dirty = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise KronosContractError("official implementation checkout has tracked modifications")


def _verify_hf_snapshot(snapshot: Path, expected_revision: str, name: str) -> Path:
    resolved = snapshot.expanduser().resolve(strict=True)
    if not resolved.is_dir() or resolved.name != expected_revision:
        raise KronosContractError(
            f"{name} must be a local immutable Hugging Face snapshot ending in "
            f"{expected_revision}; got {resolved}"
        )
    for required in ("config.json", "model.safetensors"):
        if not (resolved / required).is_file():
            raise KronosContractError(f"{name} snapshot is missing {required}")
    return resolved


def _module_belongs_to(module: Any, root: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return False
    try:
        Path(module_file).resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


@contextmanager
def _official_model_package(source_root: Path) -> Iterator[Any]:
    """Import upstream's hard-coded top-level ``model`` package without leaking it."""

    package_dir = source_root / "model"
    if not (package_dir / "__init__.py").is_file():
        raise KronosContractError(f"official model package not found at {package_dir}")

    module_names = {name for name in sys.modules if name == "model" or name.startswith("model.")}
    previous_modules = {name: sys.modules[name] for name in module_names}
    alien_modules = [
        name
        for name, module in previous_modules.items()
        if not _module_belongs_to(module, package_dir)
    ]
    if alien_modules:
        raise KronosContractError(
            "cannot safely import official Kronos because unrelated modules already occupy "
            f"its top-level 'model' namespace: {sorted(alien_modules)}"
        )

    source_text = str(source_root)
    sys.path.insert(0, source_text)
    importlib.invalidate_caches()
    try:
        module = importlib.import_module("model")
        if not _module_belongs_to(module, package_dir):
            raise KronosContractError("imported 'model' did not resolve to pinned Kronos source")
        yield module
    finally:
        try:
            sys.path.remove(source_text)
        except ValueError:  # pragma: no cover - defensive against third-party path mutation
            pass
        for name in list(sys.modules):
            if name == "model" or name.startswith("model."):
                if name in previous_modules:
                    sys.modules[name] = previous_modules[name]
                else:
                    sys.modules.pop(name, None)


def load_official_predictor(
    *,
    source_root: Path,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    device: str,
    config: KronosInferenceConfig | None = None,
) -> OfficialPredictor:
    """Load only pinned local code/checkpoints; this function performs no download."""

    config = config or KronosInferenceConfig()
    source_root = Path(source_root).expanduser().resolve(strict=True)
    _verify_checkout_revision(source_root, config.implementation_revision)
    model_snapshot = _verify_hf_snapshot(model_snapshot, config.model_revision, "model")
    tokenizer_snapshot = _verify_hf_snapshot(
        tokenizer_snapshot, config.tokenizer_revision, "tokenizer"
    )

    with _official_model_package(source_root) as module:
        tokenizer = module.KronosTokenizer.from_pretrained(str(tokenizer_snapshot))
        model = module.Kronos.from_pretrained(str(model_snapshot))
        predictor = module.KronosPredictor(model, tokenizer, device=device, max_context=512)
    freeze_official_predictor(predictor)
    return predictor


class KronosAdapter:
    """Enforce Leonos timing/sampling rules around the official predictor."""

    def __init__(
        self,
        predictor: OfficialPredictor,
        config: KronosInferenceConfig | None = None,
        *,
        seed_setter: Callable[[int], None] = set_global_inference_seed,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.predictor = predictor
        self.config = config or KronosInferenceConfig()
        self._seed_setter = seed_setter
        self._clock = clock
        freeze_official_predictor(self.predictor)

    def predict_batch(self, requests: Iterable[ForecastRequest], *, split: str) -> pd.DataFrame:
        """Return long-form forecast rows with no realized labels attached."""

        ordered = canonicalize_requests(requests)
        if not ordered:
            return pd.DataFrame(columns=PREDICTION_COLUMNS)
        shard_id = logical_shard_identity(split, self.config.seed, [r.key for r in ordered])
        effective_seed = derive_effective_seed(self.config.seed, shard_id)
        self._seed_setter(effective_seed)

        frames: list[pd.DataFrame] = []
        history_timestamp_list: list[pd.Series] = []
        forecast_timestamp_list: list[pd.Series] = []
        amount_sources: list[str] = []
        for request in ordered:
            columns = list(INPUT_COLUMNS)
            if "amount" in request.history:
                columns.append("amount")
                amount_sources.append("source_provided")
            else:
                amount_sources.append("estimated_ohlcv_proxy_by_official_predictor")
            # Copy and strip all unrelated columns so labels can never reach inference.
            frames.append(request.history.loc[:, columns].copy())
            history_timestamp_list.append(
                pd.Series(_datetime_index(request.history_dates, "history_dates"))
            )
            forecast_timestamp_list.append(
                pd.Series(_datetime_index(request.forecast_dates, "forecast_dates"))
            )

        try:
            import torch

            inference_context = torch.inference_mode()
        except ImportError:  # pragma: no cover - mocks can exercise without torch installed
            inference_context = nullcontext()
        started = self._clock()
        with inference_context:
            predictions = self.predictor.predict_batch(
                df_list=frames,
                x_timestamp_list=history_timestamp_list,
                y_timestamp_list=forecast_timestamp_list,
                pred_len=self.config.forecast_sessions,
                T=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                sample_count=self.config.sample_count,
                verbose=False,
            )
        elapsed = self._clock() - started
        if not isinstance(predictions, (list, tuple)) or len(predictions) != len(ordered):
            raise KronosContractError("official predictor returned the wrong batch cardinality")

        rows: list[dict[str, Any]] = []
        for request, prediction, future_dates, amount_source in zip(
            ordered, predictions, forecast_timestamp_list, amount_sources, strict=True
        ):
            if not isinstance(prediction, pd.DataFrame) or "close" not in prediction:
                raise KronosContractError("official predictor must return a DataFrame with close")
            expected_dates = pd.DatetimeIndex(future_dates)
            actual_dates = _datetime_index(prediction.index, "predicted forecast dates")
            closes = prediction["close"].to_numpy(dtype=np.float64)
            if len(prediction) != FORECAST_SESSIONS or len(closes) != FORECAST_SESSIONS:
                raise KronosContractError("official predictor returned the wrong forecast horizon")
            if not actual_dates.equals(expected_dates):
                raise KronosContractError("official predictor changed forecast-date alignment")
            if not np.isfinite(closes).all() or (closes <= 0).any():
                raise KronosContractError(
                    "official predictor returned non-finite/non-positive closes"
                )

            current_close = float(request.history["close"].iloc[-1])
            score = float(closes.mean() / current_close - 1.0)
            if not np.isfinite(score):
                raise KronosContractError("derived Kronos score is non-finite")
            origin = pd.Timestamp(request.origin)
            for horizon, (forecast_date, predicted_close) in enumerate(
                zip(expected_dates, closes, strict=True), start=1
            ):
                rows.append(
                    {
                        "model": MODEL_NAME,
                        "seed": self.config.seed,
                        "ticker": str(request.ticker),
                        "origin": origin,
                        "horizon": horizon,
                        "input_end": origin,
                        "forecast_date": forecast_date,
                        "predicted_close": float(predicted_close),
                        "current_close": current_close,
                        "score": score,
                        "split": split,
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
                        "amount_source": amount_source,
                        "output_semantics": OUTPUT_SEMANTICS,
                        "batch_elapsed_seconds": float(elapsed),
                    }
                )
        result = pd.DataFrame.from_records(rows, columns=PREDICTION_COLUMNS)
        validate_prediction_frame(result)
        return result


def validate_prediction_frame(frame: pd.DataFrame) -> None:
    """Validate one complete, label-free, long-form Kronos prediction shard."""

    label_columns = _LABEL_COLUMN_NAMES.intersection(map(str.lower, frame.columns))
    if label_columns:
        raise PredictionShardError(
            f"realized labels must be stored separately; found columns {sorted(label_columns)}"
        )
    missing = sorted(set(PREDICTION_COLUMNS).difference(frame.columns))
    if missing:
        raise PredictionShardError(f"prediction frame is missing columns: {missing}")
    if frame.empty:
        raise PredictionShardError("prediction shards must not be empty")
    if frame.duplicated(list(PREDICTION_KEY_COLUMNS)).any():
        raise PredictionShardError("duplicate prediction key within shard")
    if set(frame["model"].unique()) != {MODEL_NAME}:
        raise PredictionShardError("Kronos shard contains a different model name")
    for column in ("predicted_close", "current_close", "score", "batch_elapsed_seconds"):
        if not np.isfinite(frame[column].to_numpy(dtype=np.float64)).all():
            raise PredictionShardError(f"{column} contains non-finite values")
    if (frame[["predicted_close", "current_close"]].to_numpy(dtype=np.float64) <= 0).any():
        raise PredictionShardError("predicted_close and current_close must be positive")
    if (frame["batch_elapsed_seconds"].to_numpy(dtype=np.float64) < 0).any():
        raise PredictionShardError("batch_elapsed_seconds must be non-negative")

    singleton_columns = (
        "seed",
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
        "batch_elapsed_seconds",
    )
    if any(frame[column].nunique(dropna=False) != 1 for column in singleton_columns):
        raise PredictionShardError("one file must contain exactly one immutable logical shard")
    if float(frame["temperature"].iloc[0]) != INVESTMENT_TEMPERATURE:
        raise PredictionShardError("unexpected Kronos temperature")
    if float(frame["top_p"].iloc[0]) != INVESTMENT_TOP_P:
        raise PredictionShardError("unexpected Kronos top_p")
    if int(frame["top_k"].iloc[0]) != INVESTMENT_TOP_K:
        raise PredictionShardError("unexpected Kronos top_k")
    if int(frame["sample_count"].iloc[0]) != INVESTMENT_SAMPLE_COUNT:
        raise PredictionShardError("unexpected Kronos sample_count")
    expected_metadata = {
        "model_id": KRONOS_MODEL_ID,
        "model_revision": KRONOS_MODEL_REVISION,
        "tokenizer_id": KRONOS_TOKENIZER_ID,
        "tokenizer_revision": KRONOS_TOKENIZER_REVISION,
        "implementation_revision": KRONOS_IMPLEMENTATION_REVISION,
        "output_semantics": OUTPUT_SEMANTICS,
    }
    for column, expected in expected_metadata.items():
        if set(frame[column].astype(str).unique()) != {expected}:
            raise PredictionShardError(f"unexpected Kronos {column}")
    if not set(frame["amount_source"]).issubset(
        {"source_provided", "estimated_ohlcv_proxy_by_official_predictor"}
    ):
        raise PredictionShardError("unknown amount_source")

    origin_group_columns = ["model", "seed", "ticker", "origin"]
    expected_horizons = tuple(range(1, FORECAST_SESSIONS + 1))
    for _, group in frame.groupby(origin_group_columns, sort=False, dropna=False):
        horizons = tuple(sorted(int(value) for value in group["horizon"]))
        if horizons != expected_horizons:
            raise PredictionShardError(
                f"each origin must contain horizons 1..{FORECAST_SESSIONS}; got {horizons}"
            )
        if group["logical_shard_id"].nunique(dropna=False) != 1:
            raise PredictionShardError("one origin spans multiple logical shard identities")
        if group["score"].nunique(dropna=False) != 1:
            raise PredictionShardError("score must be constant across an origin's horizons")
        if group["current_close"].nunique(dropna=False) != 1:
            raise PredictionShardError("current_close must be constant across an origin's horizons")
        origin = pd.Timestamp(group["origin"].iloc[0])
        input_ends = pd.DatetimeIndex(pd.to_datetime(group["input_end"]))
        if not (input_ends == origin).all():
            raise PredictionShardError("input_end must equal the information-date origin")
        horizon_order = group.sort_values("horizon")
        forecast_dates = _datetime_index(horizon_order["forecast_date"], "persisted forecast dates")
        if forecast_dates[0] <= origin:
            raise PredictionShardError("persisted forecast dates must follow the origin")
        reconstructed_score = (
            group["predicted_close"].to_numpy(dtype=np.float64).mean()
            / float(group["current_close"].iloc[0])
            - 1.0
        )
        if not np.isclose(reconstructed_score, float(group["score"].iloc[0]), atol=1e-12):
            raise PredictionShardError("stored score does not match the predicted close path")

    origin_rows = frame.loc[:, ["ticker", "origin"]].drop_duplicates()
    keys = [
        ForecastOriginKey(str(row.ticker), pd.Timestamp(row.origin))
        for row in origin_rows.itertuples(index=False)
    ]
    expected_shard_id = logical_shard_identity(
        str(frame["split"].iloc[0]), int(frame["seed"].iloc[0]), keys
    )
    if str(frame["logical_shard_id"].iloc[0]) != expected_shard_id:
        raise PredictionShardError("logical_shard_id does not match the persisted origin keys")
    expected_effective_seed = derive_effective_seed(int(frame["seed"].iloc[0]), expected_shard_id)
    if int(frame["effective_seed"].iloc[0]) != expected_effective_seed:
        raise PredictionShardError("effective_seed does not match the logical shard")


def write_prediction_shard(frame: pd.DataFrame, path: Path) -> Path:
    """Validate then atomically replace a Parquet shard on the same filesystem."""

    validate_prediction_frame(frame)
    path = Path(path)
    if path.suffix != ".parquet":
        raise PredictionShardError("prediction shard path must end in .parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp.parquet"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.loc[:, PREDICTION_COLUMNS].to_parquet(temporary_path, index=False)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass  # Directory fsync is not available on every supported filesystem.
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def iter_completed_origin_keys(
    paths: Iterable[Path],
) -> Iterator[tuple[str, int, str, pd.Timestamp]]:
    """Yield complete origins and reject duplicated/corrupt resume state."""

    seen_forecast_keys: set[tuple[str, int, str, pd.Timestamp, int]] = set()
    seen_origin_keys: set[tuple[str, int, str, pd.Timestamp]] = set()
    for path in sorted(map(Path, paths)):
        frame = pd.read_parquet(path)
        validate_prediction_frame(frame)
        for row in frame.loc[:, PREDICTION_KEY_COLUMNS].itertuples(index=False, name=None):
            key = (str(row[0]), int(row[1]), str(row[2]), pd.Timestamp(row[3]), int(row[4]))
            if key in seen_forecast_keys:
                raise PredictionShardError(f"duplicate forecast key across shards: {key}")
            seen_forecast_keys.add(key)
        for row in (
            frame.loc[:, PREDICTION_KEY_COLUMNS[:-1]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ):
            origin_key = (str(row[0]), int(row[1]), str(row[2]), pd.Timestamp(row[3]))
            if origin_key in seen_origin_keys:
                raise PredictionShardError(
                    f"origin split across/duplicated in shards: {origin_key}"
                )
            seen_origin_keys.add(origin_key)
            yield origin_key


def pending_requests(
    requests: Iterable[ForecastRequest],
    *,
    seed: int,
    completed_origin_keys: Iterable[tuple[str, int, str, pd.Timestamp]],
) -> list[ForecastRequest]:
    """Return canonically ordered origins not present in validated complete shards."""

    completed = {
        (str(model), int(done_seed), str(ticker), pd.Timestamp(origin))
        for model, done_seed, ticker, origin in completed_origin_keys
    }
    ordered = canonicalize_requests(requests)
    return [
        request
        for request in ordered
        if (MODEL_NAME, int(seed), str(request.ticker), pd.Timestamp(request.origin))
        not in completed
    ]


def prediction_shard_path(
    directory: Path, *, split: str, seed: int, requests: Iterable[ForecastRequest]
) -> Path:
    """Create a deterministic filename from the same canonical logical identity."""

    ordered = canonicalize_requests(requests)
    if not ordered:
        raise KronosContractError("cannot name an empty prediction shard")
    shard_id = logical_shard_identity(split, seed, [request.key for request in ordered])
    return Path(directory) / f"kronos-{split}-seed{seed}-{shard_id}.parquet"
