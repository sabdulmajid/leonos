# Leonos agent guide

## Fixed scope

Version 1 compares exactly two substantive competitors on the accepted fixed
U.S.-equity panel: frozen `Kronos-base` and one pooled LightGBM. Use only source
daily OHLCV and known calendar fields. Zero-score and equal-weight buy-and-hold are
references, not new modeling workstreams. Do not add crypto, news, fundamentals,
new architectures, distillation, intraday data, or live execution.

## Commands

```bash
uv sync --extra dev
uv run pytest
uv run leonos --help
uv run leonos data fetch
uv run leonos data audit
uv run leonos prepare
uv run leonos baseline fit
uv run leonos smoke
uv run leonos predict --model kronos --split test --seed 42
uv run leonos predict --model lightgbm --split test --seed 42
uv run leonos evaluate
uv run leonos report
```

Treat `configs/base.yaml` as the sole experiment configuration. Keep public
interfaces typed and artifacts in Parquet/JSON. CPU CI must use small fixtures and
mocks: never download the dataset or checkpoints during tests.

## Timing and data invariants

- Context is the same ticker's last 90 completed exchange sessions through `t`.
- The label is `mean(close[t+1:t+10]) / close[t] - 1`; it is not a directly
  executable holding-period return.
- Origins, input end, ten future exchange dates, label end, and earliest execution
  date are explicit. Missing bars are never forward-filled into model inputs.
- Signals formed after close `t` may first trade at open `t+1`. Qlib
  `TopkDropoutStrategy` already asks for the previous step's signal; do not shift it
  twice.
- Predictions and labels are immutable, separate artifacts. Future-price mutation
  must not change inputs, predictions, or orders formed earlier.
- Data-dependent transforms fit on development training data only. Development
  labels end by 2023-12-31; validation origins are July-December 2024 with label
  ends by 2024-12-31. Test begins in 2025.

## Ownership and safe integration

The lead orchestrator owns shared config, CLI, protocol, portfolio integration,
Git history, jobs, and reports. Bounded agents may own data, baseline/evaluation,
or Kronos modules and non-overlapping tests. Preserve unrelated work, inspect diffs
before commits, and never commit data, weights, secrets, caches, or machine paths.

## Evidence rules

Never tune on test, choose the best stochastic seed, discard finite inconvenient
forecasts, or present synthetic fixtures as market evidence. Constant-score RankIC
is `NA`. Compare models only on common eligible observations and disclose coverage.
Invalidate affected runs after correctness bugs. Report unsupported metrics as
`NA`, not estimates. Qlib is evaluation software, not proof that source data are
unbiased. Do not claim checkpoint provenance is independently audited.

Do not interfere with other cluster jobs. Long jobs need immutable config, commit
and dirty-state metadata, a real process ID, progress/failure artifacts, and an
exact resume command. Check them only at useful intervals.
