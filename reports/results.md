# Leonos v1 results

> **Portfolio correction in progress:** independent reruns found unstable Qlib
> ordering at exact LightGBM score ties. All portfolio values in the table below
> are withdrawn pending a deterministic saved-score rerun. RankIC, MAE, coverage,
> bootstrap intervals, and runtime values remain valid.

Primary seed: 42. Test span: 2025-01-03 through 2026-08-21 (signals trade at the next session open). Costs are 5 bps per side; cash return is zero.

The ranking evidence is inconclusive because the paired 95% interval contains zero: mean daily RankIC difference (Kronos − LightGBM) was -0.0056 with paired moving-block 95% CI [-0.0746, 0.0496] across 409 dates. LightGBM also produced the higher 5-bps net return. Seed sensitivity is incomplete (1/3 declared seeds). The RankIC-difference sign changes across calendar-year segments.

| Model | Coverage | Mean RankIC | Paired Δ RankIC (95% CI) | MAE (bp) | Net return | Net Sharpe | Max drawdown | Turnover | Costs | Inference seconds | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| kronos | 100.00% | -0.0032 | -0.0056 [-0.0746, 0.0496] | 403.7 | 49.99% | 0.84 | -36.91% | 72.40 | $47,777 | 862.0 | 1.19 GiB |
| lightgbm | 100.00% | 0.0025 | reference | 290.6 | 54.33% | 1.02 | -22.46% | 70.75 | $42,855 | 0.0 | NA |

The zero-score reference has RankIC `NA` and MAE 292.1 bp. The 95%-invested equal-weight buy-and-hold reference returned 28.99% net, with Sharpe 1.20 and maximum drawdown -16.36%.

Cost sensitivities at 0 and 15 bps and block-bootstrap sensitivities at 10 and 40 sessions are retained in the saved evaluation artifacts. Remaining positions are marked at the last close and are not forcibly liquidated.

## Limits

This is a finite test on a fixed surviving-stock basket using a present-day historical snapshot, not a point-in-time universe. The checkpoint's June 2024 pretraining cutoff is author-reported rather than independently audited. Kronos and LightGBM differ in architecture and training history, so this does not isolate pretraining causally. The portfolio is a split-adjusted price-return simulation, not literal historical share accounting; dividends are excluded. Daily next-open fills and fixed proportional costs simplify execution and say nothing about live fill quality or capacity.
