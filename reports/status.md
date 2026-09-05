# Status

- Milestone: M2 model integration complete (2026-09-04).
- Completed evidence: pinned data accepted (515,779 rows, 51 tickers); exact splits
  and 61 causal features frozen; LightGBM validation mean daily RankIC 0.0650 on
  118 dates; 20,859/20,859 finite test forecasts saved without test inspection;
  88 CPU/Qlib tests pass; Kronos validation smoke produced 2/2 aligned finite
  forecasts; validation-only throughput selected batch 16 (14.57 origins/s,
  3.72 GiB peak reserved).
- Active Leonos jobs: seed-42 worker 0/2, exec session `20928`, PID `739344`,
  `cuda:0`; worker 1/2, exec session `73828`, PID `739342`, `cuda:1`.
- Kronos forecasts: 192 / 20,859 test origins at the initial health check.
- Blocker: GitHub accepted the integration-branch push, but draft-PR creation was
  rejected with HTTP 403 (`Resource not accessible by personal access token`).
- Next command: after a meaningful interval, read the two worker manifests once;
  each manifest also contains its exact resume command.
