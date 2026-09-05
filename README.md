# Leonos

Leonos asks one question: on a fixed basket of U.S. equities, does frozen
`NeoQuasar/Kronos-base` rank future ten-session average-price appreciation better
than a pooled LightGBM model, and does any ranking advantage improve the same
long-only portfolio after transaction costs?

**Status (2026-09-04): the primary seed-42 comparison is complete and
inconclusive.** Across 409 test dates and 20,859 common observations, Kronos's
mean daily RankIC was -0.0032 versus 0.0025 for LightGBM. The paired difference
was -0.0056 (95% moving-block-bootstrap CI [-0.0746, 0.0496]). With deterministic
portfolio-only tie handling and five basis points per side, LightGBM returned
107.99% net versus 49.99% for Kronos, with higher Sharpe and shallower drawdown.
See [the results](reports/results.md); the final declared sensitivity seed remains
in progress.

The pinned snapshot passed the post-policy audit with 515,779 daily bars and all
51 equities. Both models covered every one of the 20,859 eligible test origins.
The real-data Kronos smoke passed 2/2 origins, and the complete seed-42 inference
finished on two GPUs in 14.4 minutes.
The repository is an independent application of published components to a
different equity panel, not an exact reproduction of the Kronos paper. A negative
or inconclusive result is valid.

## Reproducible path

```bash
python -m venv .venv
.venv/bin/pip install uv
.venv/bin/uv sync --extra dev
.venv/bin/pytest
.venv/bin/leonos data fetch
.venv/bin/leonos data audit
.venv/bin/leonos prepare
.venv/bin/leonos baseline fit
```

GPU extras are intentionally separate from the CPU gate:

```bash
.venv/bin/uv sync --extra dev --extra kronos --extra qlib
.venv/bin/leonos smoke
.venv/bin/leonos benchmark-kronos
```

Building pinned Qlib from source requires Python development headers (`Python.h`).
Freeze the measured batch recommendation in `configs/base.yaml` before canonical
test prediction; the run plan then rejects batch/worker changes during resume.

Large market rows, checkpoints, and forecast shards live under ignored `data/`,
`checkpoints/`, and `artifacts/`. Small manifests, aggregate results, and reports
are committed. See [the protocol](docs/protocol.md) and
[current status](reports/status.md).

## Research integrity

Forecasts are indexed by their information date after the close. Labels are stored
separately and joined only for evaluation. LightGBM selection uses validation only;
Kronos remains frozen. Test results never select models, exclusions, portfolio
settings, sampling parameters, or reported periods.
