# Daily data acceptance

Status: **FAIL**

Rows: 515,857; tickers: 51; dates: 1970-01-02 to 2026-09-03.

Daily timestamps denote interval starts; each bar is usable only after its session close. Missing market bars were not filled.

## Findings

- ERROR `zero_volume` (77): zero-volume daily bars require exclusion/review
- ERROR `non_exchange_sessions` (1): sample=['1985-09-27']
- WARNING `ticker_missing_sessions` (2): AAPL: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): ADBE: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (117): AMD: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): AMZN: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): BA: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (3298): BAC: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): BRK.B: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): CAT: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): COP: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): COST: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): CSCO: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): CVX: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): DIS: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): GE: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): GOOGL: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): GS: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): HD: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): JNJ: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): JPM: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): KO: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): LIN: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): LLY: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): MA: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): MCD: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): MRK: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): MS: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): MSFT: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): NFLX: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): NKE: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): NVDA: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): ORCL: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): PEP: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): PFE: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): PG: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): RTX: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): SBUX: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): T: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (880): TMO: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1255): UNH: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): VZ: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): WFC: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): WMT: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): XOM: missing within observed lifetime; no fill will be applied
- WARNING `close_adj_is_close_alias` (515857): the collection fallback sets close_adj=close when the API omits it; the card's dividend-adjusted claim is unsupported by a distinct series

## Adjustment basis

Documented split checks: pass. `close_adj / close` is diagnostic only and was not applied to OHLCV or volume.
