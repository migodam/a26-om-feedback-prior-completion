# A26 encoder and decoder implementation handoff

Status: implementation complete; static parsing PASS; numerical tests NOT_RUN pending the parent metered CPU job. No GPU execution, source-physics generation, new full-forward labels, SHA256 checks, or locked-file access occurred.

## Interfaces

- `FrozenEncoder(background_path, config, book, device="cpu")` and `load_encoders(data_root, config, book, device="cpu")`. The loader accepts an A25 repository root, data root, or background directory. `.reference` is the reused A25 `TorchReference`; `.models` maps `L` and `R` to the reused exact paired models. `.bind_book(book)` rebinds the active accounting ledger.
- The original A25 config is resolved as `background_path.parents[2]/configs/a25/nn_admitted.json`. A copied cache may provide `a25_source_config_path` explicitly. A26 protocol/source-commit fields are recorded separately and cannot replace original cache-construction provenance or change runtime tolerances.
- `.encode(observation, sigma, method)` accepts raw complex128 `[q,s]` or `[B,q,s]` only. It returns detached Torch tensors: `features` float32 `[B,12,g,g]`, `base_mass` and `initial_mass` float64 `[B,2g²]`, `beta` float64 `[B]`, and JSON-compatible diagnostics. Single observations preserve the batch dimension. Target dictionaries are rejected.
- `.save_diagnostics(destination)` returns `{"arrays": path, "metrics": path}`. An `.npz` destination is used directly; a directory gets `<background>_encoder_diagnostics.npz` and `.metrics.json`. Arrays include the real-mass background J spectrum/right singular vectors, full C, I-C, diagonal confidence, and optional Db spectrum. Db preparation is lazy; pre-encode/N3 exports truthfully omit the pending Db spectrum. Call again after N1/N2 if the complete spectrum is required.
- `ResidualDecoder(width=32, *, real_lower=-.5, real_upper=4., imag_lower=0., imag_upper=.5)` accepts `forward(normalized_features, base_mass, material_scale, volume=1/576)` and returns float64 real mass. `predict_mass` has the same signature; `predict_chi` returns complex128 physical chi `[B,g²]`. Normalizer statistics stay external. `project_mass` is also public. Width16 is the only allowed fallback.

## Frozen math and cost behavior

Common initialization is the boxed known-background material minus the saved background step times its ordinary L gradient. N1/N3 use the same one-step L physical base; N2 uses the TASK_R32 exact paired gradient. BP is the physical material increment `mass_unpack(-step*g_background, volume)` before box. N1/N2 share rank16 Db O and source-shared materialization with the original dark-denominator guard. Channels follow the protocol exactly and use source-mean field/w power; canonical current is never recovered.

N1/N3 never call R to produce beta. Their beta is NaN with `NO_FEEDBACK_NOT_COMPUTED`; positive-sigma group is `NO_FEEDBACK`, sigma0 is `NOISE_FREE`. N2 computes and charges matched L/R data at m0, returns `COMPUTED_FOR_N2`, and uses `NOISY` / `NOISE_FREE`; sigma0 beta remains NaN. The parent may join N2 beta to other methods by case_id as an offline regime diagnostic. Beta is never a CNN channel. Diagnostics expose actual operation counter deltas, core margins, and explicit relative forward residuals.

The encoder validates portable geometry/material/closure descriptors against the original A25 config, fixed known-background field/data/Db equations, cached J against Db*eb, saved data scale, and saved spectral-step rule. R core and forward residual guards retain original tolerances. A zero RHS requires an exactly zero residual and receives zero diagnostic residual; no floor or fallback is introduced.

Confidence uses stable relative spectral weights `t²/(t²+.05²)` with `t=s/smax`. The full weak complement retains the omitted/null coordinates. Scalar noise whitening leaves relative weights unchanged; noiseless normalization uses background RMS. This is numerical confidence, not calibrated statistical recoverability. The decoder has five Conv2d layers, GELU after the first four, dilations1/2/4/2/1, a zero-initialized two-channel head, shared physical RMS residual scaling, and identical box projection. Width32 parameter count is mathematically 31,810; runtime assertion is pending.

## Verification handoff

`tests/test_a26_encoder_decoder.py` contains nine tiny CPU tests under `get_ledger()`: original provenance/firewall; wrong J/changed guard rejection; identical N1/N3 base and prohibited R/current calls; N2 gradient against autograd and beta; confidence/nullspace/noise/export; explicit zero-RHS and dark denominator failure; source permutation and background-only loading; lazy unsafe O-rank failure; decoder topology/physical scaling/box/backward. Fixtures are finite analytic arrays; no full-wave builder or split archive is used.

Static command receipt: `research/delegated/a26-encoder/STATIC_COMMAND_COST.json`; AST parse wall 0.007950083s, process CPU 0.007405s, GPU0s. This receipt covers only the parsing command; no numerical or scientific result is asserted.
