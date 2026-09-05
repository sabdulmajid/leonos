"""Public daily market-data adapters for the small Leonos comparison universe.

This module is intentionally separate from :mod:`leonos.data`, which owns the
immutable research dataset.  It is for public, point-in-time reference data and
never reads brokerage exports, positions, balances, or other private inputs.

Two details are deliberately explicit:

* a vendor's daily timestamp normally denotes the *start* of a session; the
  corresponding completed-session close is derived from an exchange calendar;
* unadjusted/split-adjusted OHLC and distribution-adjusted close are not treated
  as interchangeable, and a missing bar is never forward-filled.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

MARKET_SNAPSHOT_SCHEMA = "leonos.public_market_snapshot.v1"
DEFAULT_PUBLIC_CACHE_ROOT = Path("data/public_market")
YAHOO_CHART_ENDPOINT = "https://query1.finance.yahoo.com/v8/finance/chart"
TWELVE_DATA_ENDPOINT = "https://api.twelvedata.com/time_series"
BANK_OF_CANADA_ENDPOINT = "https://www.bankofcanada.ca/valet/observations"

BAR_COLUMNS: tuple[str, ...] = (
    "symbol",
    "provider",
    "vendor_symbol",
    "session",
    "vendor_timestamp_utc",
    "session_open_utc",
    "session_close_utc",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "currency",
    "price_basis",
    "is_complete",
    "is_eligible",
    "quality",
)
ACTION_COLUMNS: tuple[str, ...] = (
    "symbol",
    "provider",
    "action_date",
    "action_timestamp_utc",
    "action_type",
    "amount",
    "numerator",
    "denominator",
)
FX_COLUMNS: tuple[str, ...] = (
    "symbol",
    "provider",
    "date",
    "published_at_utc",
    "base_currency",
    "quote_currency",
    "value",
    "measure",
    "source_series",
    "is_complete",
    "quality",
)

AssetKind = Literal["stock", "etf", "commodity_trust", "fx"]
ShariaStatus = Literal["issuer_mandate", "not_applicable", "not_issuer_verified"]
JsonTransport = Callable[[str, float], bytes]


class MarketDataError(RuntimeError):
    """Base class for a public market-data retrieval or validation error."""


class UnsupportedInstrumentError(MarketDataError):
    """Raised for symbols outside the explicitly reviewed public universe."""


class SourceUnavailableError(MarketDataError):
    """Raised when a configured public source cannot satisfy a request."""


class MarketDataIntegrityError(MarketDataError):
    """Raised when a response is structurally inconsistent with its request."""


@dataclass(frozen=True)
class VendorMapping:
    """A reviewed vendor identifier, optionally constrained by an exchange MIC."""

    provider: str
    symbol: str
    mic_code: str | None = None


@dataclass(frozen=True)
class InstrumentSpec:
    """Public reference metadata; it is not a recommendation or a live listing feed."""

    symbol: str
    name: str
    asset_kind: AssetKind
    currency: str
    exchange: str | None
    mic_code: str | None
    calendar: str | None
    timezone: str
    inception_date: date | None
    inception_basis: str
    distribution_policy: str
    sharia_status: ShariaStatus
    sharia_basis: str
    vendor_mappings: tuple[VendorMapping, ...]
    official_urls: tuple[str, ...]

    def vendor_mapping(self, provider: str) -> VendorMapping:
        """Return the reviewed mapping for ``provider`` (case-insensitive)."""

        provider = provider.casefold()
        for mapping in self.vendor_mappings:
            if mapping.provider.casefold() == provider:
                return mapping
        raise SourceUnavailableError(f"{self.symbol} has no reviewed {provider} mapping")


# Inception dates use the semantic stated in ``inception_basis``.  In particular,
# SPUS and SPWO use SEC commencement-of-operations evidence because their current
# issuer web tables contain apparent copy errors (documented in the methodology).
INSTRUMENTS: dict[str, InstrumentSpec] = {
    "SPUS": InstrumentSpec(
        symbol="SPUS",
        name="SP Funds S&P 500 Sharia Industry Exclusions ETF",
        asset_kind="etf",
        currency="USD",
        exchange="NYSE Arca",
        mic_code="ARCX",
        calendar="XNYS",
        timezone="America/New_York",
        inception_date=date(2019, 12, 17),
        inception_basis="SEC-reported commencement of operations",
        distribution_policy="Monthly income, if any; capital gains at least annually",
        sharia_status="issuer_mandate",
        sharia_basis="Issuer says the index/fund follows AAOIFI Sharia guidelines",
        vendor_mappings=(
            VendorMapping("yahoo", "SPUS"),
            VendorMapping("twelve_data", "SPUS", "ARCX"),
        ),
        official_urls=(
            "https://www.sp-funds.com/spus/",
            "https://www.sec.gov/Archives/edgar/data/1742912/000199937126007133/"
            "spus-497k_033026.htm",
        ),
    ),
    "HLAL": InstrumentSpec(
        symbol="HLAL",
        name="Wahed FTSE USA Shariah ETF",
        asset_kind="etf",
        currency="USD",
        exchange="Nasdaq",
        mic_code="XNAS",
        calendar="XNAS",
        timezone="America/New_York",
        inception_date=date(2019, 7, 16),
        inception_basis="Issuer-reported fund inception/listing date",
        distribution_policy="Annual (issuer-reported frequency)",
        sharia_status="issuer_mandate",
        sharia_basis="Issuer says FTSE/Yasaar certified the tracked index Sharia-compliant",
        vendor_mappings=(
            VendorMapping("yahoo", "HLAL"),
            VendorMapping("twelve_data", "HLAL", "XNAS"),
        ),
        official_urls=(
            "https://www.wahed.com/hlal",
            "https://www.sec.gov/Archives/edgar/data/1683471/000089418925008945/"
            "ck0001683471-20250925.htm",
        ),
    ),
    "SPTE": InstrumentSpec(
        symbol="SPTE",
        name="SP Funds S&P Global Technology ETF",
        asset_kind="etf",
        currency="USD",
        exchange="NYSE Arca",
        mic_code="ARCX",
        calendar="XNYS",
        timezone="America/New_York",
        inception_date=date(2023, 11, 30),
        inception_basis="SEC-reported commencement of operations",
        distribution_policy="Net investment income at least monthly; realized gains annually",
        sharia_status="issuer_mandate",
        sharia_basis="Issuer says the tracked technology index is Sharia-screened",
        vendor_mappings=(
            VendorMapping("yahoo", "SPTE"),
            VendorMapping("twelve_data", "SPTE", "ARCX"),
        ),
        official_urls=(
            "https://www.sp-funds.com/spte/",
            "https://www.sec.gov/Archives/edgar/data/1989916/000199937126004583/"
            "spte-497k_022726.htm",
        ),
    ),
    "SPWO": InstrumentSpec(
        symbol="SPWO",
        name="SP Funds S&P World (ex-US) ETF",
        asset_kind="etf",
        currency="USD",
        exchange="NYSE Arca",
        mic_code="ARCX",
        calendar="XNYS",
        timezone="America/New_York",
        inception_date=date(2023, 12, 19),
        inception_basis="SEC-reported commencement of operations",
        distribution_policy="Net investment income at least monthly; realized gains annually",
        sharia_status="issuer_mandate",
        sharia_basis="Issuer says the tracked world index follows AAOIFI Sharia principles",
        vendor_mappings=(
            VendorMapping("yahoo", "SPWO"),
            VendorMapping("twelve_data", "SPWO", "ARCX"),
        ),
        official_urls=(
            "https://www.sp-funds.com/spwo/",
            "https://www.sec.gov/Archives/edgar/data/1989916/000199937126004449/"
            "spfunds-485bpos_022626.htm",
        ),
    ),
    "MU": InstrumentSpec(
        symbol="MU",
        name="Micron Technology, Inc. common stock",
        asset_kind="stock",
        currency="USD",
        exchange="Nasdaq Global Select Market",
        mic_code="XNAS",
        calendar="XNAS",
        timezone="America/New_York",
        inception_date=None,
        inception_basis="Not applicable: company stock, not a fund inception field",
        distribution_policy="Board-declared quarterly cash dividends currently observed",
        sharia_status="not_issuer_verified",
        sharia_basis="Micron does not make an issuer Sharia-compliance mandate",
        vendor_mappings=(
            VendorMapping("yahoo", "MU"),
            VendorMapping("twelve_data", "MU", "XNAS"),
        ),
        official_urls=(
            "https://investors.micron.com/financials/quarterly-results/",
            "https://www.sec.gov/edgar/browse/?CIK=0000723125",
        ),
    ),
    "GOOGL": InstrumentSpec(
        symbol="GOOGL",
        name="Alphabet Inc. Class A common stock",
        asset_kind="stock",
        currency="USD",
        exchange="Nasdaq Global Select Market",
        mic_code="XNAS",
        calendar="XNAS",
        timezone="America/New_York",
        inception_date=None,
        inception_basis="Not applicable: company stock, not a fund inception field",
        distribution_policy="Board-declared quarterly cash dividends currently observed",
        sharia_status="not_issuer_verified",
        sharia_basis="Alphabet does not make an issuer Sharia-compliance mandate",
        vendor_mappings=(
            VendorMapping("yahoo", "GOOGL"),
            VendorMapping("twelve_data", "GOOGL", "XNAS"),
        ),
        official_urls=(
            "https://abc.xyz/investor/",
            "https://www.sec.gov/edgar/browse/?CIK=0001652044",
        ),
    ),
    "TSM": InstrumentSpec(
        symbol="TSM",
        name="Taiwan Semiconductor Manufacturing Company Limited ADS",
        asset_kind="stock",
        currency="USD",
        exchange="New York Stock Exchange",
        mic_code="XNYS",
        calendar="XNYS",
        timezone="America/New_York",
        inception_date=None,
        inception_basis="Not applicable: each listed ADS represents five common shares",
        distribution_policy="Issuer-declared cash dividends; ADS depositary deductions may apply",
        sharia_status="not_issuer_verified",
        sharia_basis="TSMC does not make an issuer Sharia-compliance mandate",
        vendor_mappings=(
            VendorMapping("yahoo", "TSM"),
            VendorMapping("twelve_data", "TSM", "XNYS"),
        ),
        official_urls=(
            "https://investor.tsmc.com/english/",
            "https://www.sec.gov/edgar/browse/?CIK=0001046179",
        ),
    ),
    "ISRG": InstrumentSpec(
        symbol="ISRG",
        name="Intuitive Surgical, Inc. common stock",
        asset_kind="stock",
        currency="USD",
        exchange="Nasdaq Global Select Market",
        mic_code="XNAS",
        calendar="XNAS",
        timezone="America/New_York",
        inception_date=None,
        inception_basis="Not applicable: company stock, not a fund inception field",
        distribution_policy="No regular cash dividend currently declared",
        sharia_status="not_issuer_verified",
        sharia_basis="Intuitive Surgical does not make an issuer Sharia-compliance mandate",
        vendor_mappings=(
            VendorMapping("yahoo", "ISRG"),
            VendorMapping("twelve_data", "ISRG", "XNAS"),
        ),
        official_urls=(
            "https://isrg.intuitive.com/",
            "https://www.sec.gov/edgar/browse/?CIK=0001035267",
        ),
    ),
    "VRTX": InstrumentSpec(
        symbol="VRTX",
        name="Vertex Pharmaceuticals Incorporated common stock",
        asset_kind="stock",
        currency="USD",
        exchange="Nasdaq Global Select Market",
        mic_code="XNAS",
        calendar="XNAS",
        timezone="America/New_York",
        inception_date=None,
        inception_basis="Not applicable: company stock, not a fund inception field",
        distribution_policy="No regular cash dividend currently declared",
        sharia_status="not_issuer_verified",
        sharia_basis="Vertex does not make an issuer Sharia-compliance mandate",
        vendor_mappings=(
            VendorMapping("yahoo", "VRTX"),
            VendorMapping("twelve_data", "VRTX", "XNAS"),
        ),
        official_urls=(
            "https://investors.vrtx.com/",
            "https://www.sec.gov/edgar/browse/?CIK=0000875320",
        ),
    ),
    "WSHR": InstrumentSpec(
        symbol="WSHR",
        name="Wealthsimple Shariah World Equity Index ETF",
        asset_kind="etf",
        currency="CAD",
        exchange="Cboe Canada (formerly NEO Exchange)",
        mic_code="NEOE",
        # exchange_calendars has no Cboe Canada calendar.  Cboe Canada and TSX
        # share the relevant regular session/holiday schedule for daily bars.
        calendar="XTSE",
        timezone="America/Toronto",
        inception_date=date(2021, 5, 12),
        inception_basis="Cboe Canada listing date",
        distribution_policy="Quarterly cash distributions currently observed",
        sharia_status="issuer_mandate",
        sharia_basis="Issuer says fund/index were certified by Ratings Intelligence Partners",
        vendor_mappings=(
            VendorMapping("yahoo", "WSHR.NE"),
            VendorMapping("twelve_data", "WSHR", "NEOE"),
        ),
        official_urls=(
            "https://www.cboe.com/markets/ca/equities/securities/WSHR/",
            "https://help.wealthsimple.com/hc/en-ca/articles/1500011334461-"
            "Stocks-held-in-the-Wealthsimple-Shariah-World-Equity-Index-ETF-WSHR",
        ),
    ),
    "GLDM": InstrumentSpec(
        symbol="GLDM",
        name="SPDR Gold MiniShares Trust",
        asset_kind="commodity_trust",
        currency="USD",
        exchange="NYSE Arca",
        mic_code="ARCX",
        calendar="XNYS",
        timezone="America/New_York",
        inception_date=date(2018, 6, 25),
        inception_basis="Sponsor-reported fund inception date",
        distribution_policy="No income; trust sells gold as needed to pay ongoing expenses",
        sharia_status="not_issuer_verified",
        sharia_basis=(
            "Sponsor describes physical gold exposure but does not make a Sharia certification; "
            "do not infer one from the asset alone"
        ),
        vendor_mappings=(
            VendorMapping("yahoo", "GLDM"),
            VendorMapping("twelve_data", "GLDM", "ARCX"),
        ),
        official_urls=(
            "https://www.ssga.com/us/en/individual/etfs/spdr-gold-minishares-gldm",
            "https://www.sec.gov/Archives/edgar/data/1618181/000143774925036313/"
            "gldm20250930_10k.htm",
        ),
    ),
    "UMMA": InstrumentSpec(
        symbol="UMMA",
        name="Wahed Dow Jones Islamic World ETF",
        asset_kind="etf",
        currency="USD",
        exchange="Nasdaq",
        mic_code="XNAS",
        calendar="XNAS",
        timezone="America/New_York",
        inception_date=date(2022, 1, 7),
        inception_basis="Issuer-reported inception date",
        distribution_policy="Issuer-declared distributions with published purification data",
        sharia_status="issuer_mandate",
        sharia_basis="Issuer provides a Sharia certificate and periodic Sharia audit reports",
        vendor_mappings=(
            VendorMapping("yahoo", "UMMA"),
            VendorMapping("twelve_data", "UMMA", "XNAS"),
        ),
        official_urls=(
            "https://www.wahed.com/umma",
            "https://www.sec.gov/Archives/edgar/data/1683471/000089418926020931/"
            "ummasummary.htm",
        ),
    ),
    "MNZL": InstrumentSpec(
        symbol="MNZL",
        name="Manzil Russell Halal USA Broad Market ETF",
        asset_kind="etf",
        currency="USD",
        exchange="Nasdaq",
        mic_code="XNAS",
        calendar="XNAS",
        timezone="America/New_York",
        inception_date=date(2025, 11, 18),
        inception_basis="SEC shareholder-report inception date",
        distribution_policy="Issuer-declared distributions with purification disclosure",
        sharia_status="issuer_mandate",
        sharia_basis="Issuer provides a Sharia certificate and IdealRatings-screened index mandate",
        vendor_mappings=(
            VendorMapping("yahoo", "MNZL"),
            VendorMapping("twelve_data", "MNZL", "XNAS"),
        ),
        official_urls=(
            "https://manzilfunds.com/",
            "https://www.sec.gov/Archives/edgar/data/1592900/000159290026001525/"
            "ck0001592900-20260131.htm",
        ),
    ),
    "SPRE": InstrumentSpec(
        symbol="SPRE",
        name="SP Funds S&P Global REIT Sharia ETF",
        asset_kind="etf",
        currency="USD",
        exchange="NYSE Arca",
        mic_code="ARCX",
        calendar="XNYS",
        timezone="America/New_York",
        inception_date=date(2020, 12, 29),
        inception_basis="Issuer-reported inception date",
        distribution_policy="Issuer-declared monthly income and annual capital-gain distributions",
        sharia_status="issuer_mandate",
        sharia_basis="Issuer says the tracked global REIT index is Sharia-screened",
        vendor_mappings=(
            VendorMapping("yahoo", "SPRE"),
            VendorMapping("twelve_data", "SPRE", "ARCX"),
        ),
        official_urls=(
            "https://www.sp-funds.com/spre/",
            "https://www.sec.gov/edgar/browse/?CIK=0001742912",
        ),
    ),
    "USDCAD": InstrumentSpec(
        symbol="USDCAD",
        name="U.S. dollar in Canadian dollars",
        asset_kind="fx",
        currency="CAD",
        exchange=None,
        mic_code=None,
        calendar=None,
        timezone="America/Toronto",
        inception_date=None,
        inception_basis="Not applicable",
        distribution_policy="Not applicable",
        sharia_status="not_applicable",
        sharia_basis="Reference exchange rate, not a security",
        vendor_mappings=(
            VendorMapping("bank_of_canada", "FXUSDCAD"),
            VendorMapping("yahoo", "USDCAD=X"),
            VendorMapping("twelve_data", "USD/CAD"),
        ),
        official_urls=("https://www.bankofcanada.ca/rates/exchange/daily-exchange-rates/",),
    ),
    "CADEUR": InstrumentSpec(
        symbol="CADEUR",
        name="Canadian dollar in euros",
        asset_kind="fx",
        currency="EUR",
        exchange=None,
        mic_code=None,
        calendar=None,
        timezone="America/Toronto",
        inception_date=None,
        inception_basis="Not applicable",
        distribution_policy="Not applicable",
        sharia_status="not_applicable",
        sharia_basis="Reference exchange rate, not a security",
        vendor_mappings=(
            # The Bank publishes EUR/CAD.  CADEUR is its explicitly marked reciprocal.
            VendorMapping("bank_of_canada", "FXEURCAD"),
            VendorMapping("yahoo", "CADEUR=X"),
            VendorMapping("twelve_data", "CAD/EUR"),
        ),
        official_urls=("https://www.bankofcanada.ca/rates/exchange/daily-exchange-rates/",),
    ),
}


def instrument(symbol: str) -> InstrumentSpec:
    """Return reviewed public metadata for a canonical symbol."""

    canonical = str(symbol).strip().upper().replace("/", "")
    try:
        return INSTRUMENTS[canonical]
    except KeyError as exc:
        raise UnsupportedInstrumentError(f"unsupported public instrument: {symbol!r}") from exc


def _utc_timestamp(value: datetime | pd.Timestamp | None = None) -> pd.Timestamp:
    stamp = pd.Timestamp(datetime.now(UTC) if value is None else value)
    if stamp.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return stamp.tz_convert("UTC")


def _default_transport(url: str, timeout: float) -> bytes:
    request = Request(url, headers={"User-Agent": "leonos-public-market-data/0.1"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS endpoints
        return response.read()


def _json_response(url: str, *, timeout: float, transport: JsonTransport) -> Mapping[str, Any]:
    try:
        raw = transport(url, timeout)
        payload = json.loads(raw)
    except Exception as exc:
        raise SourceUnavailableError(f"could not retrieve/parse {url}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise MarketDataIntegrityError(f"JSON response from {url} is not an object")
    return payload


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


@dataclass
class DailyBarResult:
    """Normalized daily bars plus source provenance and provider diagnostics."""

    spec: InstrumentSpec
    provider: str
    source_url: str
    retrieved_at_utc: pd.Timestamp
    price_basis: str
    bars: pd.DataFrame
    actions: pd.DataFrame = field(default_factory=lambda: _empty_frame(ACTION_COLUMNS))
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()

    @property
    def eligible_bars(self) -> pd.DataFrame:
        """Return complete, structurally valid bars without filling any gaps."""

        if self.bars.empty:
            return self.bars.copy()
        return self.bars.loc[self.bars["is_eligible"].astype(bool)].copy()

    @property
    def latest_eligible(self) -> pd.Series | None:
        eligible = self.eligible_bars
        if eligible.empty:
            return None
        return eligible.sort_values("session").iloc[-1]

    @property
    def missing_completed_sessions(self) -> tuple[date, ...]:
        """Return omitted exchange sessions between the first and last returned rows.

        A missing-valued row is not an omitted session: it remains in ``bars`` and
        is identified by its quality flags.  A missing leading/trailing requested
        session cannot be inferred from a result alone; callers needing an exact
        endpoint should also use :meth:`PublicMarketDataAdapter.fetch_daily`'s
        ``required_session`` argument.
        """

        if self.bars.empty or self.spec.calendar is None:
            return ()
        calendar = xcals.get_calendar(self.spec.calendar)
        first = pd.Timestamp(self.bars["session"].min())
        last = pd.Timestamp(self.bars["session"].max())
        observed = set(self.bars["session"].dt.date)
        return tuple(
            label.date()
            for label in calendar.sessions_in_range(first, last)
            if calendar.session_close(label) <= self.retrieved_at_utc
            and label.date() not in observed
        )


@dataclass
class FXReferenceResult:
    """Daily indicative FX observations; these are averages, not executable closes."""

    spec: InstrumentSpec
    provider: str
    source_url: str
    retrieved_at_utc: pd.Timestamp
    observations: pd.DataFrame
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()

    @property
    def complete_observations(self) -> pd.DataFrame:
        if self.observations.empty:
            return self.observations.copy()
        return self.observations.loc[self.observations["is_complete"].astype(bool)].copy()


def _session_bounds(
    spec: InstrumentSpec, session_date: date
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if spec.calendar is None:
        return None
    calendar = xcals.get_calendar(spec.calendar)
    label = pd.Timestamp(session_date)
    if not calendar.is_session(label):
        return None
    return calendar.session_open(label), calendar.session_close(label)


def _float_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _quality_for_bar(
    *,
    spec: InstrumentSpec,
    values: Mapping[str, float],
    session_known: bool,
    is_complete: bool,
) -> tuple[str, bool]:
    issues: list[str] = []
    required = ["open", "high", "low", "close"]
    if spec.asset_kind in {"stock", "etf", "commodity_trust"}:
        required.append("volume")
    missing = [name for name in required if not math.isfinite(values[name])]
    if missing:
        issues.append("missing:" + ",".join(missing))
    if not session_known:
        issues.append("not_exchange_session")
    if not is_complete:
        issues.append("session_not_complete")

    o, h, low, close = (values[name] for name in ("open", "high", "low", "close"))
    if all(math.isfinite(value) for value in (o, h, low, close)):
        if min(o, h, low, close) <= 0:
            issues.append("nonpositive_price")
        if h < low or h < max(o, close) or low > min(o, close):
            issues.append("invalid_ohlc")
    volume = values["volume"]
    if math.isfinite(volume) and volume < 0:
        issues.append("negative_volume")
    return ("ok" if not issues else "|".join(issues), not issues)


def _normalise_bar_rows(
    *,
    spec: InstrumentSpec,
    provider: str,
    vendor_symbol: str,
    rows: Sequence[Mapping[str, Any]],
    as_of_utc: pd.Timestamp,
    price_basis: str,
) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        vendor_stamp = pd.Timestamp(row["vendor_timestamp_utc"])
        if vendor_stamp.tzinfo is None:
            raise MarketDataIntegrityError("vendor timestamps must be timezone-aware")
        vendor_stamp = vendor_stamp.tz_convert("UTC")
        session_date = row.get("session")
        if isinstance(session_date, str):
            session_date = date.fromisoformat(session_date)
        if not isinstance(session_date, date):
            session_date = vendor_stamp.tz_convert(spec.timezone).date()
        bounds = _session_bounds(spec, session_date)
        open_utc, close_utc = bounds if bounds else (pd.NaT, pd.NaT)
        complete = bool(bounds and as_of_utc >= close_utc)
        values = {
            name: _float_or_nan(row.get(name))
            for name in ("open", "high", "low", "close", "adj_close", "volume")
        }
        quality, eligible = _quality_for_bar(
            spec=spec,
            values=values,
            session_known=bounds is not None,
            is_complete=complete,
        )
        normalized.append(
            {
                "symbol": spec.symbol,
                "provider": provider,
                "vendor_symbol": vendor_symbol,
                "session": pd.Timestamp(session_date),
                "vendor_timestamp_utc": vendor_stamp,
                "session_open_utc": open_utc,
                "session_close_utc": close_utc,
                **values,
                "currency": spec.currency,
                "price_basis": price_basis,
                "is_complete": complete,
                "is_eligible": eligible,
                "quality": quality,
            }
        )
    if not normalized:
        return _empty_frame(BAR_COLUMNS)
    frame = (
        pd.DataFrame(normalized, columns=BAR_COLUMNS)
        .sort_values("session")
        .reset_index(drop=True)
    )
    if frame["session"].duplicated().any():
        duplicates = frame.loc[frame["session"].duplicated(False), "session"].dt.date.tolist()
        raise MarketDataIntegrityError(f"duplicate sessions from {provider}: {duplicates}")
    return frame


def parse_yahoo_chart(
    payload: Mapping[str, Any],
    spec: InstrumentSpec,
    *,
    retrieved_at_utc: datetime | pd.Timestamp,
    source_url: str = YAHOO_CHART_ENDPOINT,
) -> DailyBarResult:
    """Parse one Yahoo chart response without repairing incomplete bars."""

    retrieved = _utc_timestamp(retrieved_at_utc)
    chart = payload.get("chart")
    if not isinstance(chart, Mapping):
        raise MarketDataIntegrityError("Yahoo response has no chart object")
    if chart.get("error"):
        raise SourceUnavailableError(f"Yahoo chart error: {chart['error']}")
    results = chart.get("result")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], Mapping):
        raise MarketDataIntegrityError("Yahoo response must contain exactly one chart result")
    result = results[0]
    meta = result.get("meta")
    if not isinstance(meta, Mapping):
        raise MarketDataIntegrityError("Yahoo response has no metadata")
    mapping = spec.vendor_mapping("yahoo")
    if str(meta.get("symbol", "")).upper() != mapping.symbol.upper():
        raise MarketDataIntegrityError(
            f"Yahoo returned symbol {meta.get('symbol')!r}, expected {mapping.symbol!r}"
        )
    if meta.get("currency") and str(meta["currency"]).upper() != spec.currency:
        raise MarketDataIntegrityError(
            f"Yahoo returned currency {meta['currency']!r}, expected {spec.currency!r}"
        )

    timestamps = result.get("timestamp", [])
    indicators = result.get("indicators")
    if not isinstance(timestamps, list) or not isinstance(indicators, Mapping):
        raise MarketDataIntegrityError("Yahoo chart result has malformed timestamps/indicators")
    quotes = indicators.get("quote")
    if not isinstance(quotes, list) or len(quotes) != 1 or not isinstance(quotes[0], Mapping):
        raise MarketDataIntegrityError("Yahoo chart result has no quote array")
    quote_values = quotes[0]
    adjusted_blocks = indicators.get("adjclose", [])
    adjusted = (
        adjusted_blocks[0].get("adjclose", [])
        if isinstance(adjusted_blocks, list)
        and adjusted_blocks
        and isinstance(adjusted_blocks[0], Mapping)
        else []
    )

    def series(name: str) -> list[Any]:
        value = quote_values.get(name, [])
        if not isinstance(value, list):
            raise MarketDataIntegrityError(f"Yahoo {name} indicator is not an array")
        if len(value) != len(timestamps):
            raise MarketDataIntegrityError(
                f"Yahoo {name} has {len(value)} values for {len(timestamps)} timestamps"
            )
        return value

    arrays = {name: series(name) for name in ("open", "high", "low", "close", "volume")}
    if adjusted and len(adjusted) != len(timestamps):
        raise MarketDataIntegrityError("Yahoo adjusted-close length differs from timestamps")
    rows: list[dict[str, Any]] = []
    for index, unix_seconds in enumerate(timestamps):
        stamp = pd.Timestamp(int(unix_seconds), unit="s", tz="UTC")
        rows.append(
            {
                "vendor_timestamp_utc": stamp,
                **{name: values[index] for name, values in arrays.items()},
                "adj_close": adjusted[index] if adjusted else None,
            }
        )
    price_basis = (
        "Yahoo OHLC/close are split-adjusted; adj_close additionally adjusts for cash "
        "distributions and capital gains"
    )
    bars = _normalise_bar_rows(
        spec=spec,
        provider="yahoo",
        vendor_symbol=mapping.symbol,
        rows=rows,
        as_of_utc=retrieved,
        price_basis=price_basis,
    )

    actions: list[dict[str, Any]] = []
    events = result.get("events", {})
    if isinstance(events, Mapping):
        for event_type, records in events.items():
            if not isinstance(records, Mapping):
                continue
            for record in records.values():
                if not isinstance(record, Mapping) or record.get("date") is None:
                    continue
                stamp = pd.Timestamp(int(record["date"]), unit="s", tz="UTC")
                local_date = stamp.tz_convert(spec.timezone).date()
                actions.append(
                    {
                        "symbol": spec.symbol,
                        "provider": "yahoo",
                        "action_date": pd.Timestamp(local_date),
                        "action_timestamp_utc": stamp,
                        "action_type": str(event_type),
                        "amount": _float_or_nan(record.get("amount")),
                        "numerator": _float_or_nan(record.get("numerator")),
                        "denominator": _float_or_nan(record.get("denominator")),
                    }
                )
    action_frame = _empty_frame(ACTION_COLUMNS)
    if actions:
        action_frame = (
            pd.DataFrame(actions, columns=ACTION_COLUMNS)
            .sort_values("action_date")
            .reset_index(drop=True)
        )
    metadata = {
        key: meta.get(key)
        for key in (
            "symbol",
            "currency",
            "exchangeName",
            "fullExchangeName",
            "instrumentType",
            "firstTradeDate",
            "regularMarketTime",
            "regularMarketPrice",
            "exchangeTimezoneName",
            "dataGranularity",
        )
        if key in meta
    }
    diagnostics = tuple(
        f"{row.session.date()}: {row.quality}"
        for row in bars.itertuples()
        if row.quality != "ok"
    )
    return DailyBarResult(
        spec=spec,
        provider="yahoo",
        source_url=source_url,
        retrieved_at_utc=retrieved,
        price_basis=price_basis,
        bars=bars,
        actions=action_frame,
        provider_metadata=metadata,
        diagnostics=diagnostics,
    )


class DailyBarSource(Protocol):
    name: str

    def supports(self, spec: InstrumentSpec) -> bool: ...

    def fetch(
        self,
        spec: InstrumentSpec,
        *,
        start: date,
        end: date,
        as_of: datetime | pd.Timestamp | None = None,
    ) -> DailyBarResult: ...


class YahooDailySource:
    """Local-use adapter for Yahoo's public chart response.

    Yahoo's endpoint is unversioned as a data release and its help page prohibits
    redistribution.  Cache snapshots locally, retain retrieval time, and do not
    treat this source as execution-grade market data.
    """

    name = "yahoo"

    def __init__(self, *, timeout: float = 20.0, transport: JsonTransport | None = None):
        self.timeout = timeout
        self.transport = transport or _default_transport

    def supports(self, spec: InstrumentSpec) -> bool:
        try:
            spec.vendor_mapping(self.name)
        except SourceUnavailableError:
            return False
        return spec.asset_kind != "fx"

    def fetch(
        self,
        spec: InstrumentSpec,
        *,
        start: date,
        end: date,
        as_of: datetime | pd.Timestamp | None = None,
    ) -> DailyBarResult:
        if not self.supports(spec):
            raise SourceUnavailableError(f"Yahoo daily OHLCV is not configured for {spec.symbol}")
        if end < start:
            raise ValueError("end must be on or after start")
        mapping = spec.vendor_mapping(self.name)
        # Yahoo period2 is exclusive.  UTC midnights safely bracket all reviewed
        # North American daily sessions; returned dates are resolved in exchange time.
        period1 = int(datetime.combine(start, time.min, tzinfo=UTC).timestamp())
        period2 = int(datetime.combine(end + timedelta(days=1), time.min, tzinfo=UTC).timestamp())
        params = {
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "events": "div,splits,capitalGains",
            "includeAdjustedClose": "true",
        }
        url = f"{YAHOO_CHART_ENDPOINT}/{quote(mapping.symbol, safe='')}?{urlencode(params)}"
        retrieved = _utc_timestamp(as_of)
        payload = _json_response(url, timeout=self.timeout, transport=self.transport)
        return parse_yahoo_chart(payload, spec, retrieved_at_utc=retrieved, source_url=url)


def parse_twelve_data(
    payload: Mapping[str, Any],
    spec: InstrumentSpec,
    *,
    retrieved_at_utc: datetime | pd.Timestamp,
    source_url: str = TWELVE_DATA_ENDPOINT,
) -> DailyBarResult:
    """Parse Twelve Data's split-adjusted daily response."""

    if payload.get("status") == "error" or payload.get("code"):
        raise SourceUnavailableError(f"Twelve Data error: {payload.get('message', payload)}")
    meta = payload.get("meta")
    values = payload.get("values")
    if not isinstance(meta, Mapping) or not isinstance(values, list):
        raise MarketDataIntegrityError("Twelve Data response lacks meta/values")
    mapping = spec.vendor_mapping("twelve_data")
    if str(meta.get("symbol", "")).upper() != mapping.symbol.upper():
        raise MarketDataIntegrityError("Twelve Data returned an unexpected symbol")
    if meta.get("currency") and str(meta["currency"]).upper() != spec.currency:
        raise MarketDataIntegrityError("Twelve Data returned an unexpected currency")
    returned_mic = str(meta.get("mic_code", "")).upper()
    if mapping.mic_code and returned_mic and returned_mic != mapping.mic_code:
        raise MarketDataIntegrityError(
            f"Twelve Data returned MIC {returned_mic!r}, expected {mapping.mic_code!r}"
        )
    rows: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping) or not value.get("datetime"):
            raise MarketDataIntegrityError("malformed Twelve Data daily value")
        session_date = date.fromisoformat(str(value["datetime"])[:10])
        bounds = _session_bounds(spec, session_date)
        vendor_stamp = bounds[0] if bounds else pd.Timestamp(
            datetime.combine(session_date, time.min, tzinfo=ZoneInfo(spec.timezone))
        ).tz_convert("UTC")
        rows.append(
            {
                "session": session_date,
                "vendor_timestamp_utc": vendor_stamp,
                "open": value.get("open"),
                "high": value.get("high"),
                "low": value.get("low"),
                "close": value.get("close"),
                "adj_close": None,
                "volume": value.get("volume"),
            }
        )
    retrieved = _utc_timestamp(retrieved_at_utc)
    price_basis = (
        "Twelve Data adjust=splits: all OHLC are split-adjusted; no separate "
        "distribution-adjusted close is returned"
    )
    bars = _normalise_bar_rows(
        spec=spec,
        provider="twelve_data",
        vendor_symbol=mapping.symbol,
        rows=rows,
        as_of_utc=retrieved,
        price_basis=price_basis,
    )
    diagnostics = tuple(
        f"{row.session.date()}: {row.quality}"
        for row in bars.itertuples()
        if row.quality != "ok"
    )
    return DailyBarResult(
        spec=spec,
        provider="twelve_data",
        source_url=source_url,
        retrieved_at_utc=retrieved,
        price_basis=price_basis,
        bars=bars,
        provider_metadata=dict(meta),
        diagnostics=diagnostics,
    )


class TwelveDataDailySource:
    """Documented-key fallback using an exchange-constrained symbol mapping."""

    name = "twelve_data"

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 20.0,
        transport: JsonTransport | None = None,
    ):
        if not api_key.strip():
            raise ValueError("a non-empty Twelve Data API key is required")
        self._api_key = api_key.strip()
        self.timeout = timeout
        self.transport = transport or _default_transport

    def supports(self, spec: InstrumentSpec) -> bool:
        try:
            spec.vendor_mapping(self.name)
        except SourceUnavailableError:
            return False
        return spec.asset_kind != "fx"

    def fetch(
        self,
        spec: InstrumentSpec,
        *,
        start: date,
        end: date,
        as_of: datetime | pd.Timestamp | None = None,
    ) -> DailyBarResult:
        if not self.supports(spec):
            raise SourceUnavailableError(
                f"Twelve Data daily OHLCV is not configured for {spec.symbol}"
            )
        if end < start:
            raise ValueError("end must be on or after start")
        mapping = spec.vendor_mapping(self.name)
        public_params: dict[str, Any] = {
            "symbol": mapping.symbol,
            "interval": "1day",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "order": "asc",
            "adjust": "splits",
            "outputsize": 5000,
        }
        if mapping.mic_code:
            public_params["mic_code"] = mapping.mic_code
        actual_params = {**public_params, "apikey": self._api_key}
        url = f"{TWELVE_DATA_ENDPOINT}?{urlencode(actual_params)}"
        public_url = f"{TWELVE_DATA_ENDPOINT}?{urlencode(public_params)}&apikey=%3Credacted%3E"
        retrieved = _utc_timestamp(as_of)
        payload = _json_response(url, timeout=self.timeout, transport=self.transport)
        return parse_twelve_data(
            payload,
            spec,
            retrieved_at_utc=retrieved,
            source_url=public_url,
        )


def parse_bank_of_canada_fx(
    payload: Mapping[str, Any],
    spec: InstrumentSpec,
    *,
    retrieved_at_utc: datetime | pd.Timestamp,
    source_url: str = BANK_OF_CANADA_ENDPOINT,
) -> FXReferenceResult:
    """Parse Bank of Canada daily averages, inverting EUR/CAD for ``CADEUR``."""

    if spec.symbol not in {"USDCAD", "CADEUR"}:
        raise UnsupportedInstrumentError("Bank of Canada parser supports USDCAD and CADEUR")
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise MarketDataIntegrityError("Bank of Canada response has no observations array")
    mapping = spec.vendor_mapping("bank_of_canada")
    retrieved = _utc_timestamp(retrieved_at_utc)
    rows: list[dict[str, Any]] = []
    for observation in observations:
        if not isinstance(observation, Mapping) or not observation.get("d"):
            raise MarketDataIntegrityError("malformed Bank of Canada observation")
        observed_date = date.fromisoformat(str(observation["d"]))
        source_value = observation.get(mapping.symbol)
        raw_value = source_value.get("v") if isinstance(source_value, Mapping) else None
        value = _float_or_nan(raw_value)
        quality = "ok"
        if not math.isfinite(value) or value <= 0:
            quality = "missing_or_nonpositive_value"
            value = math.nan
        elif spec.symbol == "CADEUR":
            value = 1.0 / value
        publication_local = datetime.combine(
            observed_date, time(16, 30), tzinfo=ZoneInfo("America/Toronto")
        )
        published_at = pd.Timestamp(publication_local).tz_convert("UTC")
        complete = bool(quality == "ok" and retrieved >= published_at)
        if quality == "ok" and not complete:
            quality = "publication_not_complete"
        base, quote_currency = ("USD", "CAD") if spec.symbol == "USDCAD" else ("CAD", "EUR")
        rows.append(
            {
                "symbol": spec.symbol,
                "provider": "bank_of_canada",
                "date": pd.Timestamp(observed_date),
                "published_at_utc": published_at,
                "base_currency": base,
                "quote_currency": quote_currency,
                "value": value,
                "measure": (
                    "daily_average"
                    if spec.symbol == "USDCAD"
                    else "reciprocal_of_daily_average_EURCAD"
                ),
                "source_series": mapping.symbol,
                "is_complete": complete,
                "quality": quality,
            }
        )
    frame = (
        pd.DataFrame(rows, columns=FX_COLUMNS).sort_values("date").reset_index(drop=True)
        if rows
        else _empty_frame(FX_COLUMNS)
    )
    details = payload.get("seriesDetail", {})
    diagnostics = tuple(
        f"{row.date.date()}: {row.quality}"
        for row in frame.itertuples()
        if row.quality != "ok"
    )
    return FXReferenceResult(
        spec=spec,
        provider="bank_of_canada",
        source_url=source_url,
        retrieved_at_utc=retrieved,
        observations=frame,
        provider_metadata={
            "series_detail": details.get(mapping.symbol) if isinstance(details, Mapping) else None,
            "derived_by_reciprocal": spec.symbol == "CADEUR",
            "api_version": "1.0.1",
        },
        diagnostics=diagnostics,
    )


class BankOfCanadaFXSource:
    """No-key official source for CAD reference rates published once per business day."""

    name = "bank_of_canada"

    def __init__(self, *, timeout: float = 20.0, transport: JsonTransport | None = None):
        self.timeout = timeout
        self.transport = transport or _default_transport

    def fetch(
        self,
        spec: InstrumentSpec,
        *,
        start: date,
        end: date,
        as_of: datetime | pd.Timestamp | None = None,
    ) -> FXReferenceResult:
        if spec.asset_kind != "fx":
            raise SourceUnavailableError(f"{spec.symbol} is not an FX reference series")
        if end < start:
            raise ValueError("end must be on or after start")
        mapping = spec.vendor_mapping(self.name)
        params = {"start_date": start.isoformat(), "end_date": end.isoformat()}
        url = f"{BANK_OF_CANADA_ENDPOINT}/{mapping.symbol}/json?{urlencode(params)}"
        retrieved = _utc_timestamp(as_of)
        payload = _json_response(url, timeout=self.timeout, transport=self.transport)
        return parse_bank_of_canada_fx(
            payload,
            spec,
            retrieved_at_utc=retrieved,
            source_url=url,
        )


class PublicMarketDataAdapter:
    """Try reviewed daily sources in order without silently blending vendors."""

    def __init__(
        self,
        daily_sources: Sequence[DailyBarSource] | None = None,
        *,
        fx_source: BankOfCanadaFXSource | None = None,
    ):
        self.daily_sources = tuple(daily_sources or (YahooDailySource(),))
        self.fx_source = fx_source or BankOfCanadaFXSource()

    def fetch_daily(
        self,
        symbol: str,
        *,
        start: date,
        end: date,
        as_of: datetime | pd.Timestamp | None = None,
        required_session: date | None = None,
    ) -> DailyBarResult:
        """Fetch one source's bars, optionally requiring a usable end session.

        A failed or stale source may fall through to the next configured source.
        Results from different providers are never spliced together because their
        adjustment and consolidated-volume semantics can differ.
        """

        spec = instrument(symbol)
        if spec.asset_kind == "fx":
            raise UnsupportedInstrumentError("use fetch_fx_reference for FX observations")
        failures: list[str] = []
        for source in self.daily_sources:
            if not source.supports(spec):
                continue
            try:
                result = source.fetch(spec, start=start, end=end, as_of=as_of)
                if required_session is not None:
                    eligible_dates = set(result.eligible_bars["session"].dt.date)
                    if required_session not in eligible_dates:
                        raise SourceUnavailableError(
                            f"{source.name} has no eligible {required_session} bar"
                        )
                return result
            except (MarketDataError, OSError, ValueError) as exc:
                failures.append(f"{source.name}: {exc}")
        detail = "; ".join(failures) or "no configured source supports it"
        raise SourceUnavailableError(f"unable to fetch {spec.symbol}: {detail}")

    def fetch_fx_reference(
        self,
        symbol: str,
        *,
        start: date,
        end: date,
        as_of: datetime | pd.Timestamp | None = None,
    ) -> FXReferenceResult:
        spec = instrument(symbol)
        return self.fx_source.fetch(spec, start=start, end=end, as_of=as_of)


def _safe_cache_root(root: str | Path) -> Path:
    path = Path(root)
    # Both names are ignored by the repository.  Requiring an actual path
    # component prevents an accidental snapshot beside source code/tests.
    if not {"data", "artifacts"}.intersection(path.parts):
        raise ValueError("public market cache must be under a data/ or artifacts/ directory")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def save_public_snapshot(
    result: DailyBarResult | FXReferenceResult,
    *,
    cache_root: str | Path = DEFAULT_PUBLIC_CACHE_ROOT,
) -> Path:
    """Atomically persist normalized public data and a provenance manifest.

    The destination is constrained to an ignored ``data/`` or ``artifacts/``
    directory.  Source URLs in results never contain an unredacted API key.
    """

    root = _safe_cache_root(cache_root)
    stamp = result.retrieved_at_utc.strftime("%Y%m%dT%H%M%SZ")
    request_id = hashlib.sha256(result.source_url.encode()).hexdigest()[:10]
    destination = root / result.provider / result.spec.symbol / f"{stamp}-{request_id}"
    destination.mkdir(parents=True, exist_ok=True)
    frame = result.bars if isinstance(result, DailyBarResult) else result.observations
    data_name = "bars.parquet" if isinstance(result, DailyBarResult) else "fx.parquet"
    data_path = destination / data_name
    with tempfile.NamedTemporaryFile(dir=destination, suffix=".parquet", delete=False) as handle:
        temporary_data = Path(handle.name)
    try:
        frame.to_parquet(temporary_data, index=False)
        temporary_data.replace(data_path)
    finally:
        temporary_data.unlink(missing_ok=True)

    files = [
        {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": _sha256(data_path),
            "rows": int(len(frame)),
        }
    ]
    if isinstance(result, DailyBarResult) and not result.actions.empty:
        actions_path = destination / "actions.parquet"
        with tempfile.NamedTemporaryFile(
            dir=destination, suffix=".parquet", delete=False
        ) as handle:
            temporary_actions = Path(handle.name)
        try:
            result.actions.to_parquet(temporary_actions, index=False)
            temporary_actions.replace(actions_path)
        finally:
            temporary_actions.unlink(missing_ok=True)
        files.append(
            {
                "path": actions_path.name,
                "bytes": actions_path.stat().st_size,
                "sha256": _sha256(actions_path),
                "rows": int(len(result.actions)),
            }
        )
    manifest = {
        "schema_version": MARKET_SNAPSHOT_SCHEMA,
        "symbol": result.spec.symbol,
        "instrument": _jsonable(asdict(result.spec)),
        "provider": result.provider,
        "source_url": result.source_url,
        "retrieved_at_utc": result.retrieved_at_utc.isoformat(),
        "provider_metadata": _jsonable(result.provider_metadata),
        "diagnostics": list(result.diagnostics),
        "files": files,
    }
    if isinstance(result, DailyBarResult):
        manifest["price_basis"] = result.price_basis
        manifest["missing_completed_sessions"] = [
            missing.isoformat() for missing in result.missing_completed_sessions
        ]
    manifest_path = destination / "manifest.json"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination, suffix=".json", delete=False
    ) as handle:
        temporary_manifest = Path(handle.name)
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary_manifest.replace(manifest_path)
    return manifest_path


__all__ = [
    "ACTION_COLUMNS",
    "BANK_OF_CANADA_ENDPOINT",
    "BAR_COLUMNS",
    "BankOfCanadaFXSource",
    "DailyBarResult",
    "DEFAULT_PUBLIC_CACHE_ROOT",
    "FXReferenceResult",
    "INSTRUMENTS",
    "InstrumentSpec",
    "MARKET_SNAPSHOT_SCHEMA",
    "MarketDataError",
    "MarketDataIntegrityError",
    "PublicMarketDataAdapter",
    "SourceUnavailableError",
    "TwelveDataDailySource",
    "UnsupportedInstrumentError",
    "VendorMapping",
    "YahooDailySource",
    "instrument",
    "parse_bank_of_canada_fx",
    "parse_twelve_data",
    "parse_yahoo_chart",
    "save_public_snapshot",
]
