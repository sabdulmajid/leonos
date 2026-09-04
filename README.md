# Leonos

Leonos asks one question: on a fixed basket of U.S. equities, does frozen
`NeoQuasar/Kronos-base` rank future ten-session average-price appreciation better
than a pooled LightGBM model, and does any ranking advantage improve the same
long-only portfolio after transaction costs?

**Status (2026-09-04): M0 foundation in progress. No comparative result exists yet.**
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
```

Large market rows, checkpoints, and forecast shards live under ignored `data/`,
`checkpoints/`, and `artifacts/`. Small manifests, aggregate results, and reports
are committed. See [the protocol](docs/protocol.md) and
[current status](reports/status.md).

## Research integrity

Forecasts are indexed by their information date after the close. Labels are stored
separately and joined only for evaluation. LightGBM selection uses validation only;
Kronos remains frozen. Test results never select models, exclusions, portfolio
settings, sampling parameters, or reported periods.
