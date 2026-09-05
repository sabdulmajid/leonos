# Leonos post-v1 scenario analysis

This analysis reuses completed forecasts and ledgers. It does not retrain either
model or create new market forecasts. The canonical run drew 100,000 shared,
paired circular moving-block paths of 409 sessions and evaluated each under three
forecast seeds and three transaction-cost settings: 900,000 scenario-case rows.
It finished on CPU in 18.7 seconds with 643 MiB peak resident memory.

Run signature:
`ae0ccb181819c07db4091bca453a9f5f3a88117df450245c96ee710c41780ce6`.
The machine-readable summary is
[`results/summary/scenarios/ae0ccb18…json`](../results/summary/scenarios/ae0ccb181819c07db4091bca453a9f5f3a88117df450245c96ee710c41780ce6.json).

## What happened to a normalized USD 100 at five basis points per side

The observed column is the one realized 2025-01-03 through 2026-08-21 sequence.
The median and range come from reordering/resampling that same history in
20-session blocks. They are conditional historical stress results, not forecasts
or calibrated future probabilities.

| Forecast seed | Kronos observed | Kronos resampled median [95% range] | LightGBM observed | LightGBM resampled median [95% range] | Equal-weight observed | Resampled share K > L |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | $149.99 | $144.76 [$64.35, $420.54] | $207.99 | $204.11 [$86.21, $556.99] | $128.99 | 28.63% |
| 43 | $135.02 | $134.31 [$63.34, $298.61] | $226.09 | $223.80 [$90.15, $593.05] | $128.99 | 10.05% |
| 44 | $153.98 | $152.29 [$77.13, $322.03] | $151.42 | $150.83 [$81.95, $287.78] | $128.99 | 50.93% |

These are index normalizations of a USD 1,000,000 whole-share backtest. They do
not mean that the same portfolio could literally be bought with USD 100; all
initial selected shares cost more than the approximately USD 19 target allocation
per name. A literal small account would require broker-supported fractional
shares and a separately tested execution path.

The resampled fraction ending below the initial USD 100 was 20.38%, 22.45%, and
11.64% for Kronos across seeds 42–44; for LightGBM it was 5.49%, 4.07%, and 9.58%.
The equal-weight reference's value was 4.85%. Those are descriptions of this
resampling design, not estimated chances of future loss.

## Ranking evidence remains inconclusive

| Seed | Observed mean RankIC difference (K−L) | Circular-block 95% interval | Resampled share Δ > 0 |
| ---: | ---: | ---: | ---: |
| 42 | -0.0056 | [-0.0693, 0.0556] | 43.64% |
| 43 | -0.0036 | [-0.0585, 0.0502] | 45.25% |
| 44 | -0.0089 | [-0.0731, 0.0523] | 39.74% |

Every interval contains zero. More Monte Carlo draws reduce numerical noise in
the resampling calculation; they do not add independent market dates, remove
sampling uncertainty from a 409-date test, or prove that LightGBM has persistent
predictive skill.

## Cost sensitivity exposes path dependence

| Seed | Kronos ending value at 0 / 5 / 15 bps | LightGBM ending value at 0 / 5 / 15 bps |
| ---: | ---: | ---: |
| 42 | $155.86 / $149.99 / $139.03 | $214.89 / $207.99 / $195.08 |
| 43 | $95.17 / $135.02 / $89.29 | $235.52 / $226.09 / $208.21 |
| 44 | $159.66 / $153.98 / $163.22 | $156.10 / $151.42 / $142.35 |

The primary seed behaves monotonically, but two Kronos sensitivity-seed paths do
not. This is not a claim that fees improve performance. Qlib recomputes
whole-share sizing and later cash/positions for every cost setting; a small early
cash change can alter a concentrated, path-dependent portfolio. That instability
is another reason not to treat the absolute returns as a deployable expectation.
A retail v2 should use fractional target weights where available and separate
execution-cost subtraction from stock-selection sensitivity.

## Limits of the resampled account paths

RankIC rows are stateless and pair cleanly by information date. Portfolio returns
are not: a sampled block begins with the holdings inherited from its location in
the original run. At joined block boundaries Leonos does not reconstruct holdings,
orders, or costs. The wealth distributions therefore stress the saved realized
return stream; they are not a full counterfactual rerun of the trading engine.

The defensible conclusion is unchanged: LightGBM had the stronger realized
portfolio in two seeds, Kronos narrowly led in one, and neither model demonstrated
a statistically resolved cross-sectional ranking advantage. The broad
equal-weight basket had much lower turnover, concentration, and resampled
drawdown than either active strategy.
