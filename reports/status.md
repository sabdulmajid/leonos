# Status

- Milestone: M1 CPU vertical slice complete (2026-09-04).
- Completed evidence: pinned data accepted (515,779 rows, 51 tickers); exact splits
  and 61 causal features frozen; LightGBM validation mean daily RankIC 0.0650 on
  118 dates; 20,859/20,859 finite test forecasts saved without test inspection;
  88 CPU/Qlib tests pass.
- Active Leonos jobs: none. The pre-existing TimesFM jobs ended; both user-provided
  GPUs were idle at the latest inventory and have not yet been touched by Leonos.
- Kronos forecasts: 0 / 20,859 test origins.
- Blocker: GitHub accepted the integration-branch push, but draft-PR creation was
  rejected with HTTP 403 (`Resource not accessible by personal access token`).
- Next command: `.venv/bin/pytest -q && .venv/bin/ruff check .`, then push and run
  the two-origin real-data Kronos validation smoke.
