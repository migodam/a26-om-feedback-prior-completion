# A26 data and budget implementation handoff

Status: implemented; static parse PASS; runtime tests NOT_RUN by worker.

- `load_split(data_root,role)` reads existing A25 sibling data only. It checks the public manifest, geometry, exact 64/16-parent whitelist, all eight condition rows and metadata before observations or chi targets. Legacy A25 TV `initial_mass/lambda/initial_wall/initializer_id` remain optional additional baseline fields. Target currents, clean measurements and noise seeds are never indexed.
- `online_from_split(bundle)` copies only aligned online observations and metadata. It strips all truth, target currents and legacy TV baseline fields. `noise_group` is explicit `NOISE_FREE` or `NOISY`.
- `BudgetLedger` and `charged_job` are an independent A26 copy of A25 accounting; no original module mutation. CPU 7200 / GPU 14400 hard caps and zero new labels are enforced. Historical A26 receipts scan once at job start. Failures, interruptions, immutable terminal receipts, unknown interruptions, foreign locks and shared Windows GPU lock rules remain enforced.
- Future `load_locked_split` requires an explicit A26 six-checkpoint freeze and parent-authored primary/repeat PASS records. It durably consumes one A26 cohort and one opening per split before any locked read. Successful test paths mock the reader. No historical locked input has been opened.
- Fourteen safety regression cases are in `tests/test_a26_data_budget.py`. Root must run them through its metered driver, then check actual primary archives there. Worker ran AST parsing only.

Artifacts: `SUMMARY.json` and append-only `COMMAND_COST.jsonl` in this directory. No remote operation, training, new label generation, SHA check, or changes outside owned files and this delegated artifact directory.
