# Status

- Milestone: M3 signal comparison complete; portfolio correction and M4
  sensitivity active (2026-09-04).
- Completed evidence: pinned data accepted (515,779 rows, 51 tickers); exact splits
  and 61 causal features frozen; LightGBM validation mean daily RankIC 0.0650 on
  118 dates; 20,859/20,859 finite test forecasts saved without test inspection;
  88 CPU/Qlib tests pass; Kronos validation smoke produced 2/2 aligned finite
  forecasts; validation-only throughput selected batch 16 (14.57 origins/s,
  3.72 GiB peak reserved); seed-42 Kronos completed all 20,859 test origins
  (208,590 finite horizon rows, no duplicate keys) in 14.4 minutes wall time;
  paired test delta RankIC -0.0056 with 95% CI [-0.0746, 0.0496]. Seed-42
  portfolio values are invalidated after exact LightGBM ties exposed unstable
  Qlib top-k ordering; deterministic tie handling and a saved-score rerun are
  active, with forecasts and signal metrics unaffected.
- Active Leonos jobs: none.
- Kronos forecasts: seed 42 complete (20,859 / 20,859); seed 43 complete
  (20,859 / 20,859, with 208,590 finite horizon rows and no duplicate keys).
- Blocker: GitHub accepted the integration-branch push, but draft-PR creation was
  rejected with HTTP 403 (`Resource not accessible by personal access token`).
- Next command: commit the expanded report renderer, then regenerate saved-score
  evaluations for seeds 42 and 43 from a clean commit.
