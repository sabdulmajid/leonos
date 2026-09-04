# Status

- Milestone: M1 CPU vertical slice complete (2026-09-04).
- Completed evidence: pinned data accepted (515,779 rows, 51 tickers); exact splits
  and 61 causal features frozen; LightGBM validation mean daily RankIC 0.0650 on
  118 dates; 20,859/20,859 finite test forecasts saved without test inspection.
- Active Leonos jobs: none. Both GPUs remain occupied by pre-existing TimesFM jobs
  and have not been touched by Leonos.
- Kronos forecasts: 0 / 20,859 test origins.
- Blocker: no Leonos GPU allocation is currently free; CPU work continues.
- Next command: `.venv/bin/pytest -q && .venv/bin/ruff check .`.
