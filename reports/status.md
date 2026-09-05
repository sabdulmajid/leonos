# Status

- Milestone: frozen M4 plus post-v1 retail interpretation complete (2026-09-05).
- Completed evidence: all three 20,859-origin model runs and v1 evaluation remain
  frozen. A CPU-only analysis added 100,000 shared 20-session block draws across
  three forecast seeds and three cost cases (900,000 scenario evaluations). Every
  resampled RankIC-difference interval still contains zero; LightGBM's portfolio
  leads in seeds 42/43 and Kronos narrowly leads in seed 44.
- Verification: 100 tests and Ruff pass. The full scenario run took 18.6 seconds,
  used 643 MiB peak RAM, and records clean commit `5359261` plus input/output
  hashes under run signature `07db337a…`.
- Active Leonos jobs: none. Forecast counts: 20,859 / 20,859 for seeds 42, 43,
  and 44; no GPU work was repeated.
- Blocker: GitHub accepts branch pushes, but draft-PR creation remains HTTP 403
  (`Resource not accessible by personal access token`).
- Next command: none for v1. A live/paper v2 needs a named data source, broker
  capabilities, account/jurisdiction constraints, and a new untouched period.
