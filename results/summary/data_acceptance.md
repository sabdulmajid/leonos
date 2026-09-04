# Daily data acceptance

Status: **PASS**

Rows: 515,779; tickers: 51; dates: 1970-01-02 to 2026-09-03.

Daily timestamps denote interval starts; each bar is usable only after its session close. Missing market bars were not filled.

## Findings

- WARNING `ticker_missing_sessions` (3): AAPL: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): ADBE: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (3): AMD: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): AMZN: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): BA: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (3304): BAC: missing within observed lifetime; no fill will be applied
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
- WARNING `ticker_missing_sessions` (3): LLY: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): MA: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): MCD: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): MRK: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): MS: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): MSFT: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): NFLX: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): NKE: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): NVDA: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): ORCL: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): PEP: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (3): PFE: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): PG: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (4): RTX: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): SBUX: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): T: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): TMO: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (1): UNH: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): VZ: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (9): WFC: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (54): WMT: missing within observed lifetime; no fill will be applied
- WARNING `ticker_missing_sessions` (2): XOM: missing within observed lifetime; no fill will be applied
- WARNING `close_adj_is_close_alias` (515779): the collection fallback sets close_adj=close when the API omits it; the card's dividend-adjusted claim is unsupported by a distinct series

## Adjustment basis

Documented split checks: pass. `close_adj / close` is diagnostic only and was not applied to OHLCV or volume.
