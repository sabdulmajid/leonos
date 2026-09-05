# Status

- Milestone: M3 primary comparison complete; M4 sensitivity active (2026-09-04).
- Completed evidence: pinned data accepted (515,779 rows, 51 tickers); exact splits
  and 61 causal features frozen; LightGBM validation mean daily RankIC 0.0650 on
  118 dates; 20,859/20,859 finite test forecasts saved without test inspection;
  88 CPU/Qlib tests pass; Kronos validation smoke produced 2/2 aligned finite
  forecasts; validation-only throughput selected batch 16 (14.57 origins/s,
  3.72 GiB peak reserved); seed-42 Kronos completed all 20,859 test origins
  (208,590 finite horizon rows, no duplicate keys) in 14.4 minutes wall time;
  paired test delta RankIC -0.0056 with 95% CI [-0.0746, 0.0496]; at five bps,
  Kronos returned 49.99% net versus LightGBM's 54.33%.
- Active Leonos jobs: none.
- Kronos forecasts: 20,859 / 20,859 seed-42 test origins complete.
- Blocker: GitHub accepted the integration-branch push, but draft-PR creation was
  rejected with HTTP 403 (`Resource not accessible by personal access token`).
- Next command: launch the two frozen seed-43 Kronos workers from a clean commit.
