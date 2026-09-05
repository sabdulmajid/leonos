from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from leonos.market_data import (
    BAR_COLUMNS,
    BankOfCanadaFXSource,
    MarketDataIntegrityError,
    PublicMarketDataAdapter,
    SourceUnavailableError,
    TwelveDataDailySource,
    YahooDailySource,
    instrument,
    parse_bank_of_canada_fx,
    parse_twelve_data,
    parse_yahoo_chart,
    save_public_snapshot,
)

AS_OF = datetime(2026, 9, 5, 15, 0, tzinfo=UTC)


def _yahoo_payload(
    *,
    vendor_symbol: str = "WSHR.NE",
    currency: str = "CAD",
    timestamps: list[int] | None = None,
    opens: list[float | None] | None = None,
    highs: list[float | None] | None = None,
    lows: list[float | None] | None = None,
    closes: list[float | None] | None = None,
    adjusted: list[float | None] | None = None,
    volumes: list[int | None] | None = None,
) -> dict[str, object]:
    timestamps = timestamps or [1788442200, 1788528600]  # Sep 3/4 09:30 ET
    opens = opens or [35.9, 36.0]
    highs = highs or [36.0, 36.1]
    lows = lows or [35.7, 35.8]
    closes = closes if closes is not None else [35.75, None]
    adjusted = adjusted if adjusted is not None else [35.60, None]
    volumes = volumes if volumes is not None else [10_322, 13_095]
    return {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {
                        "symbol": vendor_symbol,
                        "currency": currency,
                        "exchangeName": "NEO",
                        "fullExchangeName": "Cboe CA",
                        "exchangeTimezoneName": "America/Toronto",
                        "firstTradeDate": 1620826200,
                        "regularMarketTime": 1788551970,
                        "regularMarketPrice": 35.68,
                        "dataGranularity": "1d",
                    },
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [
                            {
                                "open": opens,
                                "high": highs,
                                "low": lows,
                                "close": closes,
                                "volume": volumes,
                            }
                        ],
                        "adjclose": [{"adjclose": adjusted}],
                    },
                    "events": {
                        "dividends": {
                            "1787923800": {"date": 1787923800, "amount": 0.21}
                        }
                    },
                }
            ],
        }
    }


def _twelve_payload() -> dict[str, object]:
    return {
        "status": "ok",
        "meta": {
            "symbol": "WSHR",
            "currency": "CAD",
            "exchange": "NEO",
            "mic_code": "NEOE",
            "exchange_timezone": "America/Toronto",
            "interval": "1day",
        },
        "values": [
            {
                "datetime": "2026-09-04",
                "open": "35.90",
                "high": "36.10",
                "low": "35.60",
                "close": "35.68",
                "volume": "14000",
            }
        ],
    }


def test_reviewed_catalog_maps_wshr_and_flags_unverified_securities() -> None:
    wshr = instrument("wshr")
    assert wshr.exchange == "Cboe Canada (formerly NEO Exchange)"
    assert wshr.currency == "CAD"
    assert wshr.mic_code == "NEOE"
    assert wshr.inception_date == date(2021, 5, 12)
    assert wshr.vendor_mapping("yahoo").symbol == "WSHR.NE"
    assert wshr.vendor_mapping("twelve_data").symbol == "WSHR"
    assert wshr.vendor_mapping("twelve_data").mic_code == "NEOE"

    assert instrument("SPUS").inception_date == date(2019, 12, 17)
    assert instrument("SPTE").inception_date == date(2023, 11, 30)
    assert instrument("SPWO").inception_date == date(2023, 12, 19)
    assert instrument("GLDM").sharia_status == "not_issuer_verified"
    assert instrument("MU").sharia_status == "not_issuer_verified"


def test_yahoo_parser_uses_completed_session_close_and_never_fills_missing_bar() -> None:
    result = parse_yahoo_chart(
        _yahoo_payload(),
        instrument("WSHR"),
        retrieved_at_utc=AS_OF,
        source_url="https://example.test/wshr",
    )

    assert tuple(result.bars.columns) == BAR_COLUMNS
    assert result.bars["session"].dt.date.tolist() == [date(2026, 9, 3), date(2026, 9, 4)]
    assert result.bars.loc[0, "vendor_timestamp_utc"] == pd.Timestamp("2026-09-03T13:30Z")
    assert result.bars.loc[0, "session_close_utc"] == pd.Timestamp("2026-09-03T20:00Z")
    assert bool(result.bars.loc[0, "is_eligible"])
    assert pd.isna(result.bars.loc[1, "close"])
    assert not bool(result.bars.loc[1, "is_eligible"])
    assert "missing:close" in result.bars.loc[1, "quality"]
    assert result.eligible_bars["session"].dt.date.tolist() == [date(2026, 9, 3)]
    assert len(result.actions) == 1
    assert result.actions.loc[0, "action_type"] == "dividends"


def test_yahoo_parser_keeps_close_and_distribution_adjusted_close_distinct() -> None:
    payload = _yahoo_payload(
        vendor_symbol="SPUS",
        currency="USD",
        timestamps=[1788528600],
        opens=[100.0],
        highs=[102.0],
        lows=[99.0],
        closes=[101.0],
        adjusted=[97.5],
        volumes=[1000],
    )
    result = parse_yahoo_chart(payload, instrument("SPUS"), retrieved_at_utc=AS_OF)

    assert result.bars.loc[0, "close"] == 101.0
    assert result.bars.loc[0, "adj_close"] == 97.5
    assert "cash distributions" in result.price_basis
    assert bool(result.bars.loc[0, "is_eligible"])


def test_yahoo_parser_flags_invalid_ohlc_instead_of_silently_accepting_it() -> None:
    payload = _yahoo_payload(
        timestamps=[1788442200],
        opens=[35.8],
        highs=[35.9],
        lows=[35.7],
        closes=[35.5],
        adjusted=[35.5],
        volumes=[200],
    )
    result = parse_yahoo_chart(payload, instrument("WSHR"), retrieved_at_utc=AS_OF)

    assert result.bars.loc[0, "quality"] == "invalid_ohlc"
    assert result.eligible_bars.empty


def test_yahoo_parser_rejects_wrong_currency_or_series_lengths() -> None:
    with pytest.raises(MarketDataIntegrityError, match="currency"):
        parse_yahoo_chart(
            _yahoo_payload(currency="USD"), instrument("WSHR"), retrieved_at_utc=AS_OF
        )

    payload = _yahoo_payload()
    payload["chart"]["result"][0]["indicators"]["quote"][0]["open"] = [35.9]  # type: ignore[index]
    with pytest.raises(MarketDataIntegrityError, match="values for"):
        parse_yahoo_chart(payload, instrument("WSHR"), retrieved_at_utc=AS_OF)


def test_required_session_falls_back_without_blending_providers() -> None:
    yahoo = YahooDailySource(transport=lambda _url, _timeout: json.dumps(_yahoo_payload()).encode())
    twelve = TwelveDataDailySource(
        "test-key",
        transport=lambda _url, _timeout: json.dumps(_twelve_payload()).encode(),
    )
    adapter = PublicMarketDataAdapter((yahoo, twelve))
    result = adapter.fetch_daily(
        "WSHR",
        start=date(2026, 9, 3),
        end=date(2026, 9, 4),
        as_of=AS_OF,
        required_session=date(2026, 9, 4),
    )

    assert result.provider == "twelve_data"
    assert result.eligible_bars["session"].dt.date.tolist() == [date(2026, 9, 4)]
    assert result.bars["adj_close"].isna().all()
    assert "adjust=splits" in result.price_basis
    assert "test-key" not in result.source_url
    assert result.bars["provider"].unique().tolist() == ["twelve_data"]


def test_adapter_reports_all_source_failures() -> None:
    yahoo = YahooDailySource(transport=lambda _url, _timeout: b"not-json")
    adapter = PublicMarketDataAdapter((yahoo,))
    with pytest.raises(SourceUnavailableError, match="yahoo"):
        adapter.fetch_daily("SPUS", start=date(2026, 9, 4), end=date(2026, 9, 4))


def test_twelve_parser_checks_exchange_mic() -> None:
    payload = _twelve_payload()
    payload["meta"]["mic_code"] = "XTSE"  # type: ignore[index]
    with pytest.raises(MarketDataIntegrityError, match="MIC"):
        parse_twelve_data(payload, instrument("WSHR"), retrieved_at_utc=AS_OF)


def test_bank_of_canada_marks_daily_average_and_explicit_reciprocal() -> None:
    payload = {
        "seriesDetail": {
            "FXUSDCAD": {"label": "USD/CAD"},
            "FXEURCAD": {"label": "EUR/CAD"},
        },
        "observations": [
            {"d": "2026-09-04", "FXUSDCAD": {"v": "1.3840"}, "FXEURCAD": {"v": "1.6075"}}
        ],
    }
    usd = parse_bank_of_canada_fx(payload, instrument("USDCAD"), retrieved_at_utc=AS_OF)
    eur = parse_bank_of_canada_fx(payload, instrument("CADEUR"), retrieved_at_utc=AS_OF)

    assert usd.observations.loc[0, "value"] == pytest.approx(1.3840)
    assert usd.observations.loc[0, "measure"] == "daily_average"
    assert eur.observations.loc[0, "value"] == pytest.approx(1 / 1.6075)
    assert eur.observations.loc[0, "measure"] == "reciprocal_of_daily_average_EURCAD"
    assert eur.observations.loc[0, "published_at_utc"] == pd.Timestamp("2026-09-04T20:30Z")
    assert bool(eur.observations.loc[0, "is_complete"])


def test_bank_of_canada_source_constructs_no_key_official_url() -> None:
    payload = {
        "seriesDetail": {"FXUSDCAD": {"label": "USD/CAD"}},
        "observations": [{"d": "2026-09-04", "FXUSDCAD": {"v": "1.3840"}}],
    }
    requested: list[str] = []

    def transport(url: str, _timeout: float) -> bytes:
        requested.append(url)
        return json.dumps(payload).encode()

    source = BankOfCanadaFXSource(transport=transport)
    result = source.fetch(
        instrument("USDCAD"), start=date(2026, 9, 4), end=date(2026, 9, 4), as_of=AS_OF
    )
    assert requested == [result.source_url]
    assert "/FXUSDCAD/json?" in result.source_url
    assert "start_date=2026-09-04" in result.source_url


def test_public_snapshot_is_provenanced_and_restricted_to_ignored_roots(tmp_path: Path) -> None:
    result = parse_yahoo_chart(
        _yahoo_payload(), instrument("WSHR"), retrieved_at_utc=AS_OF, source_url="public-url"
    )
    manifest_path = save_public_snapshot(result, cache_root=tmp_path / "data" / "public_market")
    manifest = json.loads(manifest_path.read_text())

    assert manifest["schema_version"] == "leonos.public_market_snapshot.v1"
    assert manifest["symbol"] == "WSHR"
    assert manifest["provider"] == "yahoo"
    assert manifest["files"][0]["sha256"]
    assert (manifest_path.parent / "bars.parquet").is_file()
    assert (manifest_path.parent / "actions.parquet").is_file()

    with pytest.raises(ValueError, match="data/ or artifacts"):
        save_public_snapshot(result, cache_root=tmp_path / "public_market")
