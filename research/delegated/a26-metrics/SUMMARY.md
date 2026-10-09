# A26 offline metric and real-evidence plot handoff

Status: implemented; static parse PASS. Numeric tests, Torch SSIM parity, real rendering and visual review are NOT_RUN by worker.

`image_metrics(predchi,truthchi,bgchi,grid=24)` returns delta/full/component NRMSE, the training-equivalent 7×7 valid SSIM definition, support IoU, exact support-boundary F1, passivity and box feasibility. Real/imag denominators are delta component norms. Zero denominators raise explicitly. The support threshold `.25*max(abs(truthdelta))` and edges use truth solely for offline evaluation, never encoder inputs or features.

`aggregate_rows(rows,metrics,by)` supports the root evaluator's raw→parent→family pipeline. Eight distinct cases are required per raw parent; all four validation parents are required per family. Conditions average within parent and parents then receive equal weights. Missing/failed evidence remains `None` with separate descriptive-only available means.

`paired_statistics(parentrows,config)` supplies fixed 2000 independent-parent bootstrap resamples for N2/N1, N2/N3, N1/N3 and each NN/own BASE_N*. Each comparison has `all_families`, `complex_families`, `per_family` metric dictionaries and `complex_mean_gain`, `complex_ci95`, `complex_coverage_complete`, `family_mean_gains`, `positive_complex_families` aliases. The complete 16-parent validation whitelist is mandatory, with 12 parents in the exact three complex families. No scientific gate is decided here.

`render_figures(root,config=None,book=None,*,seed=None)` requires a paid book; aliases `render` and `generate_figures`. It uses only actual `PREDICTIONS.npz`, metric CSV, paired-statistic JSON, training JSON histories and warm/fresh runtime CSV. Outputs eight figure types as PNG/SVG plus plotted CSV values and `FIGURE_MANIFEST.json` under `results/a26/evaluation/<seed>/figures`. Missing sources remain NOT_RUN. Reconstruction cases are the first validation parent per family, both backgrounds, amplitude1.2 at20dB, selected by metadata. Timing is complete raw-to-image inference and excludes offline full audits; fresh timing is explicitly same-process with uncontrolled OS cache.

No actual or locked payload was read by the worker; no synthetic result figure was generated. Prior data/budget/test files remained frozen. Costs, including a corrected evaluator filename-read failure, are in `COMMAND_COST.jsonl`.
