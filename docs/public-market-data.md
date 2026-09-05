# Public daily-market-data methodology

Verified 2026-09-05. This note covers only public reference data. It does not
use brokerage exports, positions, quantities, balances, or any other private
input, and it is not investment or Sharia advice.

The reusable implementation is `leonos.market_data`. It is separate from the
immutable v1 research dataset and does not change that experiment's universe,
models, inputs, or results.

## Reviewed security master

| Canonical symbol | Listing and currency | Date stored, with semantics | Distribution semantics | Sharia status used by the adapter |
| --- | --- | --- | --- | --- |
| SPUS | NYSE Arca, USD | 2019-12-17, SEC commencement/performance inception | Current issuer history is monthly; prospectus permits income and capital-gain distributions | `issuer_mandate`: current SEC prospectus describes a Sharia-screened index and pre-screening of investments |
| HLAL | Nasdaq, USD | 2019-07-16, issuer inception/listing | Annual frequency reported by issuer; purification is a separate quarterly disclosure | `issuer_mandate`: FTSE rules plus ongoing/annual review by Yasaar described in the prospectus |
| SPTE | NYSE Arca, USD | 2023-11-30, SEC commencement/performance inception | Net investment income at least monthly; realized gains at least annually | `issuer_mandate`: tracked index contains GICS Information Technology constituents passing Sharia screens |
| SPWO | NYSE Arca, USD | 2023-12-19, SEC commencement of operations | Net investment income at least monthly; realized gains at least annually | `issuer_mandate`: prospectus describes AAOIFI-based screens for the tracked world ex-US index |
| MU | Nasdaq Global Select Market, USD | `NA`: company stock, not a fund-inception field | Dividends are board-declared, not guaranteed; latest official quarterly-results page reports the current declaration | `not_issuer_verified`: do not infer a current Sharia screen from its presence in an ETF |
| WSHR | Cboe Canada, CAD | 2021-05-12, Cboe listing date (not relabelled as a legal fund-inception date) | Quarterly cash distributions currently reported | `issuer_mandate`: Wealthsimple says the fund/index are certified by Ratings Intelligence Partners and publishes quarterly purification information |
| GLDM | NYSE Arca, USD | 2018-06-25 sponsor fund inception; first listing was 2018-06-26 | No income; the trust regularly sells gold to pay expenses, reducing gold represented per share | `not_issuer_verified`: current sponsor and SEC materials reviewed do not state a GLDM Sharia certification. This is a verification flag, not a finding that GLDM is non-compliant |

SP Funds' current web table incorrectly shows 2020-12-29 for SPUS. Its March
30, 2026 SEC summary prospectus reports since-inception performance from
2019-12-17, consistent with older SEC commencement evidence. The current SPWO
web page also contains copied SPUS fields; the February 26, 2026 SEC filing is
used instead. These apparent website errors are why every date has an explicit
semantic basis in the code.

## WSHR identifier resolution

WSHR is a CAD ETF listed on Cboe Canada, formerly NEO Exchange. Cboe reports
currency CAD and date listed 2021-05-12. The reviewed vendor identifiers are:

- Yahoo: `WSHR.NE`. The returned metadata says `exchangeName=NEO`,
  `fullExchangeName=Cboe CA`, currency CAD. `WSHR.TO` is not the reviewed
  mapping and the chart endpoint returned 404 during verification.
- Twelve Data: `WSHR`, constrained to MIC `NEOE`. The public symbol-search
  response reports exchange NEO, MIC NEOE, America/Toronto, ETF, Canada, CAD:
  <https://api.twelvedata.com/symbol_search?symbol=WSHR&outputsize=30>.
- No reviewed Stooq mapping was found. It is not treated as a fallback.

`exchange_calendars` 4.13.2 has no separate Cboe Canada calendar. The adapter
uses `XTSE` as an explicit daily-session proxy because the regular Canadian
equity session and holiday schedule are shared for the dates in scope. This is
not a claim that Cboe and TSX are the same exchange, and exchange-calendar
changes should be rechecked before relying on a future close cutoff.

## Bar, adjustment, and fallback semantics

Yahoo is the default no-key public reference source. The adapter calls its
unversioned chart endpoint, retains the exact request and retrieval time, and
parses corporate-action records separately. Yahoo documents historical `Close`
as split-adjusted and `Adjusted close` as further adjusted for cash dividends
and capital-gain distributions:

- <https://help.yahoo.com/kb/SLN28256.html>
- <https://help.yahoo.com/kb/SLN2310.html>

The two columns remain distinct as `close` and `adj_close`; neither is silently
substituted for the other. This matters for the distributing ETFs. Yahoo's daily
Unix timestamp denotes the session start in these responses, not the moment the
finished candle became knowable. The adapter resolves its local exchange date,
uses the exchange calendar to add `session_open_utc` and `session_close_utc`,
and makes the row eligible only after that regular-session close.

Missing values remain missing. The adapter never forward-fills, backward-fills,
or reconstructs OHLCV. It rejects duplicate sessions and flags missing required
fields, non-sessions, incomplete sessions, nonpositive prices, negative volume,
and candles whose high/low envelope does not contain open and close. The result
also exposes omitted completed sessions between its first and last returned rows;
`required_session` separately enforces a requested endpoint without synthesizing
a row.

Twelve Data is an opt-in, API-key-backed fallback. It is instantiated only with
a caller-supplied key, constrains WSHR to MIC NEOE, requests
`adjust=splits`, redacts the key from stored provenance, and returns no separate
distribution-adjusted close. Its documented ETF time-series endpoint is:
<https://twelvedata.com/docs/etfs>. A `required_session` can cause the adapter to
try the next configured source, but results from providers are never spliced:
volume consolidation and adjustment policies need not match.

Yahoo's endpoint is unversioned and not execution-grade or an official exchange
record. Its help pages restrict redistribution. Snapshots are therefore local
only under ignored `data/` or `artifacts/` paths, with Parquet data, a JSON
manifest, hashes, source URL, retrieval time, metadata, and diagnostics.

## Completed-session check on 2026-09-05

The following public response was retrieved at
2026-09-05T15:18:09.973247Z. The relevant U.S. and Canadian regular sessions on
September 3 and 4 closed at 20:00:00Z while Eastern Daylight Time was in effect.
`regularMarketTime` is separately reported because it is provider quote
metadata, not the normalized daily-bar completion timestamp.

| Symbol | Currency | Latest eligible daily session and close | Yahoo `regularMarketTime`, price | Qualification |
| --- | --- | --- | --- | --- |
| SPUS | USD | 2026-09-04, 59.0299987793 | 2026-09-04T20:00:00Z, 59.03 | eligible |
| HLAL | USD | 2026-09-04, 73.2570037842 | 2026-09-04T20:00:00Z, 73.257 | eligible |
| SPTE | USD | 2026-09-04, 48.1049995422 | 2026-09-04T19:59:59Z, 48.105 | eligible |
| SPWO | USD | 2026-09-04, 33.7799987793 | 2026-09-04T20:00:00Z, 33.78 | eligible |
| MU | USD | 2026-09-04, 1016.5900268555 | 2026-09-04T20:00:01Z, 1016.59 | eligible; one-second metadata difference is retained |
| WSHR | CAD | 2026-09-03, 35.75 | 2026-09-04T19:59:30Z, 35.68 | September 4 quote metadata exists, but the returned daily candle has no close and is ineligible |
| GLDM | USD | 2026-09-04, 87.7300033569 | 2026-09-04T20:00:00Z, 87.73 | optional instrument; eligible |

The nine-row WSHR check had seven eligible rows: 2026-08-27 failed the OHLC
envelope check by one cent and 2026-09-04 lacked a close. A separate full-history
request retrieved at 2026-09-05T15:27:33.457598Z returned 1,335 sessions from
2021-05-12 through 2026-09-04; 1,199 were eligible, 134 had inconsistent OHLC
envelopes, and two had missing required values. This observed vendor quality
issue makes a MIC-constrained fallback and per-session validation particularly
important for WSHR. It also means September 4 must not be manufactured from the
metadata quote or the prior close.

## CAD FX reference rates

FX is fetched only when a conversion reference is needed. The Bank of Canada's
Valet API is official, no-key, documented as API version 1.0.1, and publishes
daily averages once each business day by 16:30 Eastern:

- <https://www.bankofcanada.ca/valet/docs/>
- <https://www.bankofcanada.ca/rates/exchange/daily-exchange-rates/>

At the same 2026-09-05T15:18:09.973247Z retrieval, the latest complete
observation was dated 2026-09-04:

| Pair | Value | Source series and transformation | Conservative available-by cutoff |
| --- | ---: | --- | --- |
| USDCAD | 1.3840 CAD per USD | `FXUSDCAD`, direct | 2026-09-04T20:30:00Z |
| CADEUR | 0.6220839813374806 EUR per CAD | reciprocal of official `FXEURCAD=1.6075` | 2026-09-04T20:30:00Z |

The 16:30 timestamp is a conservative completion cutoff derived from "by
16:30", not an observed exact publication tick. These are indicative daily
averages, not closes, broker conversion rates, spreads, or executable quotes.

## Current issuer holdings and overlap evidence

The three SP Funds CSVs and issuer-linked HLAL sheet below contained holdings as
of 2026-09-04 when retrieved on 2026-09-05. Always filter every input to its
latest `Date`: the HLAL sheet retained one older cash row dated 2026-08-04.

- SPUS: <https://www.sp-funds.com/wp-content/uploads/data/TidalFG_Holdings_SPUS.csv>
- SPTE: <https://www.sp-funds.com/wp-content/uploads/data/TidalFG_Holdings_SPTE.csv>
- SPWO: <https://www.sp-funds.com/wp-content/uploads/data/TidalFG_Holdings_SPWO.csv>
- HLAL: <https://docs.google.com/spreadsheets/d/1UC1Bk67bGuYsos_i8y_HQpNoHpVHAvqf71MbgrafJOQ/export?format=csv&gid=0>
- WSHR quarterly disclosure, as of 2026-06-30:
  <https://www.mackenzieinvestments.com/content/dam/mackenzie/en/qpds/qpd-q1-wshr-en.pdf>

| Fund and holdings date | Largest reported positions | MU weight | Issuer sector evidence located |
| --- | --- | ---: | --- |
| SPUS, 2026-09-04 | NVDA 14.21%, AAPL 12.38%, MSFT 9.73%, GOOGL 5.15%, AVGO 4.33% | 2.76% | No same-day issuer sector total. Current SEC prospectus reports 29.38% Information Technology as of 2025-11-30, which must not be relabelled current |
| HLAL, 2026-09-04 | NVDA 13.11%, AAPL 11.90%, MSFT 9.50%, GOOGL 5.00%, AVGO 4.17% | 2.71% | Q2 issuer factsheet reports 68.0% ICB Technology as of 2026-06-30 |
| SPTE, 2026-09-04 | NVDA 11.82%, AAPL 11.57%, TSM 11.55%, MSFT 9.04%, ASML 5.89% | 2.69% | Mandate/index constituents are GICS Information Technology; this is stronger than a stale allocation estimate but is not a claim that cash is technology |
| SPWO, 2026-09-04 | TSM 19.97%, Samsung Electronics 4.26%, ASML 3.16%, Alibaba 2.95%, SK hynix 2.84% | absent from current file | No same-day issuer sector total located; do not derive one from company names alone |
| WSHR, 2026-06-30 | Intertek 1.1%, Coca-Cola 1.0%, Cisco 1.0%, Wesfarmers 0.9%, Air Liquide 0.9% | not in disclosed top 25; full-list absence not established | Official quarterly disclosure reports 11.1% Information Technology |

Supporting Q2 2026 issuer factsheets:

- <https://www.sp-funds.com/wp-content/uploads/SPUS-Factsheet-2026-Q2.pdf>
- <https://www.sp-funds.com/wp-content/uploads/SPTE-Factsheet-2026-Q2.pdf>
- <https://www.sp-funds.com/wp-content/uploads/SPWO-Factsheet-2026-Q2.pdf>
- <https://cdn.prod.website-files.com/692951a60766e54470be6c6e/6a4788ad92ab9eb55942bb65_HLAL%20ETF%20Factsheet%20Q2%202026.pdf>

For the four same-date full files, the overlap coefficient below is
`sum(min(weight_a, weight_b))` over common exact `StockTicker` identifiers after
dropping zero weights and `Cash & Other`. Values are percentage points of NAV.

| Pair | Common exact identifiers | Overlap coefficient |
| --- | ---: | ---: |
| SPUS–HLAL | 146 | 83.63 pp |
| SPUS–SPTE | 59 | 55.49 pp |
| SPTE–HLAL | 44 | 51.84 pp |
| SPTE–SPWO | 36 | 26.35 pp |
| SPUS–SPWO | 0 | 0.00 pp |
| SPWO–HLAL | 1 (`SCCO`) | 0.05 pp |

This is reproducible identifier overlap using the files' published, rounded
weights, not total economic overlap. It
undercounts equivalent ADR/local listings, dual share classes, and issuers whose
vendors use different identifiers. WSHR is excluded because its freshest public
official disclosure supplies only the top 25, not a same-date full portfolio.
Wealthsimple's help article was updated 2026-08-30 but explicitly labels its full
index list "as of September 2024"; it is not substituted for current holdings:
<https://help.wealthsimple.com/hc/en-ca/articles/1500011334461-Stocks-held-in-the-Wealthsimple-Shariah-World-Equity-Index-ETF-WSHR>.

## Wealthsimple self-directed costs verified 2026-09-05

The controlling public fee schedule does not display a revision/effective date,
so the access date is recorded. The USD help article displayed revision
2026-09-04 10:52.

| Arrangement | Current public terms |
| --- | --- |
| Listed Canadian and U.S. securities | $0 commission. Bid/ask spread, market impact, taxes, fund expenses, and other applicable charges are not thereby zero |
| U.S.-listed fill from a CAD account | 1.5% currency-conversion fee applied to the WSII Corporate Exchange Rate on each applicable filled buy/sell. That corporate live rate already includes a variable market-condition spread |
| U.S.-listed fill from a USD account | No conversion fee on the securities trade itself. Core subscription is CAD $10 plus tax per month after a 30-day trial; Premium and Generation clients can opt in free |
| Cash conversion between CAD and USD accounts | Under $10,000: 1.5%; $10,000–$24,999.99: 1.0%; $25,000–$99,999.99: 0.5%; $100,000+: 0%, each applied to the Corporate Exchange Rate. Direct USD funding from another Canadian institution is supported without an FX fee |
| Direct Wealthsimple physical-gold buy/sell | 1% on each order based on bullion value. Help material says order spread covers trading, storage, and operations but does not quantify a separate spread; CAD quote, no conversion transaction; separate storage fee is $0 |
| Delivery of direct gold | Non-registered accounts only: 11% for a 0.1 oz coin and 2.25% for a 1 oz coin, including minting, insurance, and delivery; maximum 5 oz per transaction; disposition/withdrawal and no Wealthsimple buyback are disclosed limitations |

Official sources and visible revisions:

- Trade fee schedule, accessed 2026-09-05:
  <https://www.wealthsimple.com/en-ca/legal/fees/trade>
- USD accounts, updated 2026-09-04 10:52:
  <https://help.wealthsimple.com/hc/en-ca/articles/4414660979355-Upgrade-to-USD-accounts-for-stock-and-crypto-trading>
- CAD/USD conversion details, updated 2026-08-28:
  <https://help.wealthsimple.com/hc/en-ca/articles/4415548242971-Convert-funds-between-CAD-and-USD>
- Direct physical gold, updated 2026-08-27 17:43:
  <https://help.wealthsimple.com/hc/en-ca/articles/42432335559707-Buy-and-sell-physical-gold-with-Wealthsimple>
- Physical redemption, updated 2026-08-07 20:41:
  <https://help.wealthsimple.com/hc/en-ca/articles/41866506703771-Convert-digital-gold-to-physical-gold>

Direct gold and GLDM are different legal/economic products. Wealthsimple says
direct gold gives fractional ownership of bullion held in segregated
program-level storage. GLDM shares are exchange-traded trust interests; ordinary
shareholders have no individual physical-redemption entitlement, pay the trust's
0.10% gross expense ratio, and the trust sells gold for expenses. Neither the
broker's physical-backing claim nor gold's asset class establishes GLDM's Sharia
status.

## Primary public documents and revisions

- SPUS SEC summary prospectus dated 2026-03-30, accession
  `0001999371-26-007133`:
  <https://www.sec.gov/Archives/edgar/data/1742912/000199937126007133/spus-497k_033026.htm>
- HLAL SEC prospectus dated 2025-09-30, accession
  `0000894189-25-008945`:
  <https://www.sec.gov/Archives/edgar/data/1683471/000089418925008945/ck0001683471-20250925.htm>
- SPTE SEC summary prospectus dated 2026-02-27, accession
  `0001999371-26-004583`:
  <https://www.sec.gov/Archives/edgar/data/1989916/000199937126004583/spte-497k_022726.htm>
- SPWO/SP Funds SEC post-effective amendment filed 2026-02-26, accession
  `0001999371-26-004449`:
  <https://www.sec.gov/Archives/edgar/data/1989916/000199937126004449/spfunds-485bpos_022626.htm>
- MU SEC Form 10-Q for quarter ended 2026-02-26, accession
  `0000723125-26-000006`, plus current issuer quarterly-results page:
  <https://www.sec.gov/Archives/edgar/data/723125/000072312526000006/mu-20260226.htm> and
  <https://investors.micron.com/financials/quarterly-results/>
- WSHR Cboe listing record, no visible revision date, accessed 2026-09-05:
  <https://www.cboe.com/markets/ca/equities/securities/WSHR/>
- WSHR June 2026 distribution release:
  <https://www.mackenzieinvestments.com/en/media-centre/press-releases/_2026/2026-june-15-mackenzie-investments-announces-may-2026-distributions-for-its-exchange-traded-funds>
- GLDM sponsor page with fund/listing information as of 2026-09-05 and market
  price as of 2026-09-04:
  <https://www.ssga.com/us/en/individual/etfs/spdr-gold-minishares-gldm>
- GLDM SEC Form 10-K for fiscal year ended 2025-09-30:
  <https://www.sec.gov/Archives/edgar/data/1618181/000143774925036313/gldm20250930_10k.htm>

All web facts and fees can change. Re-fetch authoritative pages and record a new
retrieval timestamp before a later use; do not silently treat this verification
date as a perpetual guarantee.
