# Retail interpretation and post-v1 research boundary

This note explains what the completed experiment simulates and defines the safe
boundary for a slower-turnover, retail-oriented follow-up. It is educational
research, not individualized investment advice or a claim of guaranteed returns.

## Exact timing in v1

A daily source timestamp is a session label, not an intraday observation time.
The OHLCV row for session t is treated as unknown until the regular session has
closed. The score is then indexed by t. Pinned Qlib deliberately looks back one
trading step, so the first eligible execution is the stored daily open for session
t+1.

There is no 9:00 a.m. or pre-market execution, no live order submission, and no
intraday quote. Economically, the assumption is intended to approximate the next
regular-session opening, normally 9:30 a.m. Eastern Time. The repository has not
independently established that the vendor's daily open is the official opening
auction print. A real market-on-open order may receive a different price, be
partially filled, or be rejected, and broker/exchange cutoffs apply.

The strategy evaluates scores daily but does not necessarily round-trip daily. It
can replace at most one holding on a session and normally cannot sell a holding
before five exchange sessions. Qlib decides a proposed replacement before the
minimum-hold check; when a sale is suppressed it can hold six names, including
at the evaluation endpoint. This behavior is preserved rather than described as
continuous equal weighting.

## What five basis points means

One basis point is 0.01%. Five basis points is 0.05% of executed notional on each
side:

- buying USD 1,000 costs USD 0.50;
- later selling USD 1,000 costs another USD 0.50;
- the simple unchanged-notional round trip is therefore approximately USD 1.00,
  or 0.10%.

The v1 value is an illustrative all-in proportional friction. It is not a quoted
broker commission. A retail implementation must separately model commissions,
bid/ask spread, opening-auction slippage, regulatory fees, taxes, partial fills,
and any currency conversion. Zero-commission advertising does not imply
zero-friction execution.

Production Qlib used USD 1,000,000 and one-share lots. The displayed USD 100
wealth paths are index normalizations; they are not literal USD 100 trade
instructions. A small-account implementation would need broker-supported
fractional shares or a fund/ETF implementation.

## What the two models actually know

LightGBM is the locally supervised competitor. One pooled model learns from
468,228 historical stock-date examples whose labels end by 2023-12-31. Every
example comes from one of the same 51 tickers, but ticker identity is not a
feature. Each prediction row has 61 causal OHLCV/calendar features: candlestick
shape, recent returns, price versus moving averages, volatility, rolling
high/low/rank, volume change, price-volume correlation, and calendar cycles. The
largest rolling horizon is 60 sessions, the maximum effective lookback is 61
bars, and eligibility still requires the full 90-session context. It receives no
news, fundamentals, macro data, sector, ticker ID, genuine VWAP, or cross-stock
inputs.

Eight declared tree configurations were compared on July–December 2024 only,
using mean daily cross-sectional RankIC for early stopping and selection. The
selected seed-42 model was only one boosting iteration, then refit on every
eligible example whose label ended by 2024-12-31. That simplicity and its many
tied test scores are evidence of weak model discrimination, not hidden
sophistication.

Kronos is the generic zero-shot competitor. Leonos never trains or fine-tunes it.
The pinned released model and tokenizer ingest exactly 90 same-stock OHLCV
sessions plus past and future calendar timestamps. Because observed amount is
absent, the upstream predictor derives an amount proxy as volume multiplied by
mean OHLC; this adds no independent data. Kronos autoregressively samples ten
ten-session candle paths and averages them into one predicted path. Its score is
the predicted ten-close average divided by the current close, minus one. The ten
samples are inference randomness, not ten possible economic futures with
calibrated probabilities. Kronos's reported pretraining cutoff and market corpus
are author claims; exact checkpoint membership is not independently auditable.

The comparison is therefore practical, not causal. LightGBM was supervised on
this panel and target; Kronos brings generic pretrained representations. A
difference cannot be attributed to pretraining alone.

## Universe and data reality

All 51 equities in the fixed publisher-supplied panel are evaluated; there is no
ten-stock restriction. The exact symbols are AAPL, ABBV, ADBE, AMD, AMZN, AVGO,
BA, BAC,
BLK, BRK.B, CAT, COP, COST, CRM, CSCO, CVX, DIS, GE, GOOGL, GS, HD, JNJ, JPM,
KO, LIN, LLY, MA, MCD, META, MRK, MS, MSFT, NFLX, NKE, NVDA, ORCL, PEP, PFE,
PG, RTX, SBUX, T, TMO, TMUS, TSLA, UNH, V, VZ, WFC, WMT, and XOM.

The publisher describes this as a large-cap, cross-sector U.S. basket, but its
inclusion rule is undocumented here. It is not a dynamically reconstructed top
51 by market capitalization or a daily top-movers screen. Because it is a fixed
present-day surviving basket, it has survivorship and selection limitations.

The publisher describes the immutable snapshot as Twelve Data-derived daily
bars. Leonos records exact file hashes and schema, removes 77 nonpositive-volume
rows and one
non-exchange-calendar row under a deterministic rule, and accepts 515,779 rows.
It finds no duplicate keys, non-finite prices, or inconsistent candle envelopes;
missing sessions are never filled, and incomplete contexts/labels are excluded.

That establishes internal consistency and reproducibility, not exchange-level
authentication or a point-in-time historical vintage. The source's adjusted close
equals close throughout, dividends are excluded, and only representative split
checks were independently reconciled.

## Seeds are not market futures

Kronos sampling is stochastic, and LightGBM training/validation selection also
uses a seed. Seeds 42, 43, and 44 measure sensitivity to those implementation
choices on the same realized 2025–2026 market history. Ten thousand model seeds
would not create ten thousand independent histories or establish the probability
of future profit.

The leonos scenario command instead performs a paired, circular moving-block
bootstrap of completed daily RankIC rows and net-return ledgers. The configured
run makes 100,000 date-block draws and evaluates each draw under three
forecast seeds and
three cost cases, producing 900,000 scenario-path evaluations. Those are not
900,000 independent markets. Twenty-session blocks preserve some local serial
dependence, and the exact same sampled blocks are applied to Kronos, LightGBM,
and the equal-weight reference. It reports terminal-value and drawdown
distributions, RankIC-difference intervals, and paired resample frequencies.

This is a conditional historical stress analysis. It can show how much the result
depends on which observed market blocks recur or disappear. It does not invent
new corporate events, retrain models, recompute orders when block boundaries are
joined, or produce calibrated probabilities for the future.

Circular sampling gives every observed date equal starting weight and removes the
finite-sample edge bias seen in the frozen non-circular v1 interval. It can wrap a
block from the historical endpoint to the beginning, which is itself a modeling
choice. Block-length and method sensitivity matter more than increasing a
replicate counter after Monte Carlo error is already small.

## Frozen v1 versus a retail-oriented v2

The v1 result remains frozen. Any slower strategy designed after seeing its test
period is exploratory and cannot replace the headline comparison.

A defensible retail v2 should be declared before its next untouched evaluation:

- rebalance on the last completed exchange session of each month;
- trade no earlier than the next regular-session open;
- rank the same fixed eligible universe and hold at most ten names, retaining
  its survivor-basket limitation unless point-in-time membership is newly sourced;
- equal-weight selected names, cap each name at 10% of invested capital, and keep
  5% cash;
- optionally require both close above SMA200 and SMA50 above SMA200 as a causal
  trend filter, leaving filtered allocations in cash;
- use fractional shares only when the selected broker explicitly supports them;
- avoid trivial turnover with a declared minimum order size;
- test several predeclared realistic cost/slippage cases rather than choosing the
  one that looks best;
- compare against a low-turnover equal-weight or broad-market reference;
- report taxes as unmodeled unless account jurisdiction and tax status are known.

The 50/200-day rule is a trend filter, not a law of markets. It can reduce exposure
in sustained declines, but it reacts slowly and can repeatedly buy and sell during
sideways markets. Momentum has historical research support over some assets and
periods; it has no guarantee of working in a future stock, month, or regime.

## What slower strategies are actually trying to do

There is no strategy that will win no matter what. A useful retail comparison is
a ladder of increasingly active hypotheses:

- diversified buy-and-hold accepts market risk and minimizes decisions and
  turnover; it can still suffer long, deep drawdowns;
- a 50/200-day trend rule holds risk only when the shorter trend is above the
  longer trend; it trades slowly but is late at turning points and can whipsaw;
- cross-sectional momentum ranks stocks by a predeclared trailing return, often
  skipping the most recent month, and rebalances monthly; it can crash when prior
  losers rebound abruptly;
- monthly model ranking uses Kronos or LightGBM only as another ranking signal,
  with the same next-open timing, diversification, and cost accounting;
- volatility scaling, position caps, and cash limits control concentration and
  path risk; they do not create predictive skill.

Value and quality factors can be reasonable additional research comparators, but
they require point-in-time fundamentals and a separate provenance audit. Adding
them to the current OHLCV-only experiment after seeing test results would change
the question and contaminate the frozen test, so they belong in a predeclared v2
with a new untouched or prospective evaluation period.

## Requirements before a today signal

The pinned research snapshot is not a live feed. A genuine current paper signal
requires:

1. a named, licensed real-time or end-of-day data source with adjustment policy;
2. the exact as-of timestamp and confirmation that every input bar was public by
   then;
3. a refreshed data-quality audit and model-age disclosure;
4. broker capabilities, order types, fractional-share rules, fees, jurisdiction,
   account size, and risk constraints;
5. paper trading with recorded quotes, submitted orders, fills, rejects, and
   slippage before any capital is exposed.

Until those inputs exist, Leonos must not answer what should I buy today from the
historical snapshot or imply that its simulated opening fills are available to a
retail account.
