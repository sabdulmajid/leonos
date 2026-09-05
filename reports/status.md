# Status

- Milestone: M4 robustness and delivery review active (2026-09-04).
- Completed evidence: pinned data accepted (515,779 rows, 51 tickers); exact splits
  and 61 causal features frozen; LightGBM validation mean daily RankIC 0.0650 on
  118 dates; 20,859/20,859 finite test forecasts saved without test inspection;
  88 CPU/Qlib tests pass; Kronos validation smoke produced 2/2 aligned finite
  forecasts; validation-only throughput selected batch 16 (14.57 origins/s,
  3.72 GiB peak reserved); seed-42 Kronos completed all 20,859 test origins
  (208,590 finite horizon rows, no duplicate keys) in 14.4 minutes wall time;
  paired test delta RankIC -0.0056 with 95% CI [-0.0746, 0.0496]; deterministic
  saved-score portfolio reruns reproduce exactly, with seed-42 five-bps returns
  of 49.99% for Kronos and 107.99% for LightGBM. Seed 43 also favors LightGBM
  (delta RankIC -0.0036; five-bps returns 35.02% versus 126.09%).
- Active Leonos jobs: none.
- Kronos forecasts: seed 42 complete (20,859 / 20,859); seed 43 complete
  (20,859 / 20,859, with 208,590 finite horizon rows and no duplicate keys).
- Seed-44 Kronos forecasts: 20,859 / 20,859 complete, with 208,590 finite
  horizon rows and no duplicate keys. All three RankIC differences are negative
  with confidence intervals containing zero; portfolio winners are not stable
  across seeds (LightGBM wins 42/43, Kronos narrowly wins 44).
- Blocker: GitHub accepted the integration-branch push, but draft-PR creation was
  rejected with HTTP 403 (`Resource not accessible by personal access token`).
- Clean-clone gate: pushed commit `4f5b02d` synced from the lockfile without
  data/model downloads; 87 tests passed, 3 optional-dependency tests skipped,
  and CLI help passed.
- Next command: regenerate the all-seed report, complete independent claim review,
  and push the final delivery commit.
