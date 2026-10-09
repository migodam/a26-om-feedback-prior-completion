# A26 cost and runtime evidence

Resource accounting uses accepted terminal A26 job receipts. Nested span durations remain action evidence and are never summed into the budget. CPU jobs charge max(inclusive wall, process CPU); CUDA jobs charge process CPU and the entire occupied GPU wall. Failed, interrupted and budget-stopped attempts remain charged. Static-worker command evidence is retained in the aggregate ledger but is not added a second time to accepted totals.

## Resource snapshot

| Quantity | Recorded seconds/value |
| --- | --- |
| CPU limit | 7200 |
| GPU limit | 14400 |
| Accepted terminal CPU charge | 1094.7877 |
| Accepted terminal GPU occupation | 929.7002 |
| Active job | a26-scientific-review-001 |
| Active status | ACTIVE_CUTOFF — partial, nonterminal |
| Active cutoff UTC | 2026-10-09T04:18:20Z |
| Active partial CPU charge | 0.43067404 |
| Active partial GPU occupation | 0 |
| Known CPU through cutoff | 1095.2184 |
| Known GPU through cutoff | 929.7002 |
| CPU allowance remaining at cutoff | 6104.7816 |
| GPU allowance remaining at cutoff | 13470.3 |
| Other unresolved job tails | [] |
| Durable new-label reservations | 0 |

This cutoff is not the reporting job's terminal receipt. Report writing, figure rendering or later work after the sampled boundary remains outside this partial row until an accepted terminal receipt is available. An unknown interrupted tail keeps the total incomplete.

## Accepted attempts, including failures

| Job | Status | Device | Inclusive wall s | Process CPU s | CPU charge s | GPU occupation s | Recorded error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a26-data-audit-001 | COMPLETE | cpu | 0.3050318 | 0.09375 | 0.3050318 | 0 | NOT_RUN |
| a26-data-audit-001--controller | COMPLETE | cpu | 0.1106714 | 0.03125 | 0.1106714 | 0 | NOT_RUN |
| a26-feature-cache-001 | COMPLETE | cuda | 53.601417 | 42.28125 | 42.28125 | 53.601417 | NOT_RUN |
| a26-feature-cache-001--controller | COMPLETE | cuda | 1.399626 | 1.1875 | 1.1875 | 1.399626 | NOT_RUN |
| a26-local-report-001 | COMPLETE | cpu | 12.019018 | 11.546303 | 12.019018 | 0 | NOT_RUN |
| a26-primary-evaluation-001 | COMPLETE | cuda | 206.07951 | 183.82812 | 183.82812 | 206.07951 | NOT_RUN |
| a26-primary-evaluation-001--controller | COMPLETE | cuda | 1.5881305 | 2.15625 | 2.15625 | 1.5881305 | NOT_RUN |
| a26-primary-runtime-001 | COMPLETE | cuda | 77.841394 | 61.78125 | 61.78125 | 77.841394 | NOT_RUN |
| a26-primary-runtime-001--controller | COMPLETE | cuda | 1.6401027 | 1.6875 | 1.6875 | 1.6401027 | NOT_RUN |
| a26-primary-training-001 | COMPLETE | cuda | 546.91826 | 494.125 | 494.125 | 546.91826 | NOT_RUN |
| a26-primary-training-001--controller | COMPLETE | cuda | 1.7900797 | 2.5 | 2.5 | 1.7900797 | NOT_RUN |
| a26-process-cold-N1-001 | COMPLETE | cuda | 5.246362 | 3.046875 | 3.046875 | 5.246362 | NOT_RUN |
| a26-process-cold-N1-001--controller | COMPLETE | cuda | 1.6541548 | 1.421875 | 1.421875 | 1.6541548 | NOT_RUN |
| a26-process-cold-N2-001 | COMPLETE | cuda | 4.9869753 | 3.0625 | 3.0625 | 4.9869753 | NOT_RUN |
| a26-process-cold-N2-001--controller | COMPLETE | cuda | 1.5460457 | 1.5 | 1.5 | 1.5460457 | NOT_RUN |
| a26-process-cold-N3-001 | INTERRUPTED | cuda | 0.2275486 | 0.140625 | 0.140625 | 0.2275486 | NOT_RUN |
| a26-process-cold-N3-001--controller | COMPLETE | cuda | 0 | 0.015625 | 0.015625 | 0 | NOT_RUN |
| a26-process-cold-N3-002 | COMPLETE | cuda | 4.2219398 | 2.4375 | 2.4375 | 4.2219398 | NOT_RUN |
| a26-process-cold-N3-002--controller | COMPLETE | cuda | 1.6764791 | 1.53125 | 1.53125 | 1.6764791 | NOT_RUN |
| a26-repro-export-001 | COMPLETE | cpu | 0.6080329 | 0.5 | 0.6080329 | 0 | NOT_RUN |
| a26-repro-export-001--controller | COMPLETE | cpu | 0.2143256 | 0.046875 | 0.2143256 | 0 | NOT_RUN |
| a26-runtime-mismatch-001 | FAILED | cpu | 1 | 1 | 1 | 0 | NOT_RUN |
| a26-tests-001 | FAILED | cpu | 16.04374 | 7.0625 | 16.04374 | 0 | RuntimeError: UNIT_TESTS_FAILED:{"tests": 92, "errors": 50, "failures": 0, "passed": false} |
| a26-tests-001--controller | COMPLETE | cpu | 1.6076031 | 0.96875 | 1.6076031 | 0 | NOT_RUN |
| a26-tests-002 | FAILED | cpu | 15.229222 | 8.125 | 15.229222 | 0 | RuntimeError: UNIT_TESTS_FAILED:{"tests": 108, "errors": 4, "failures": 0, "passed": false} |
| a26-tests-002--controller | COMPLETE | cpu | 1.8506722 | 1.25 | 1.8506722 | 0 | NOT_RUN |
| a26-tests-003 | COMPLETE | cpu | 12.893155 | 10.375 | 12.893155 | 0 | NOT_RUN |
| a26-tests-003--controller | COMPLETE | cpu | 2.7565316 | 1.203125 | 2.7565316 | 0 | NOT_RUN |
| a26-tests-final-001 | COMPLETE | cpu | 12.809896 | 10.0625 | 12.809896 | 0 | NOT_RUN |
| a26-tests-final-001--controller | COMPLETE | cpu | 3.0349564 | 1.265625 | 3.0349564 | 0 | NOT_RUN |
| a26-training-benchmark-001 | FAILED | cuda | 3.5015058 | 2.59375 | 2.59375 | 3.5015058 | RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility |
| a26-training-benchmark-001--controller | COMPLETE | cuda | 1.5869444 | 1.5 | 1.5 | 1.5869444 | NOT_RUN |
| a26-training-benchmark-002 | COMPLETE | cuda | 12.527467 | 11.140625 | 11.140625 | 12.527467 | NOT_RUN |
| a26-training-benchmark-002--controller | COMPLETE | cuda | 1.6662608 | 1.578125 | 1.578125 | 1.6662608 | NOT_RUN |
| a26-transport-preflight-1b1ce7232bbb | COMPLETE | cpu | 5.6536069 | 0.068301 | 5.6536069 | 0 | NOT_RUN |
| a26-transport-pull-0813ce386495 | COMPLETE | cpu | 11.971132 | 0.172769 | 11.971132 | 0 | NOT_RUN |
| a26-transport-pull-81e0d9ab5373 | COMPLETE | cpu | 8.6852525 | 0.207677 | 8.6852525 | 0 | NOT_RUN |
| a26-transport-pull-8d076d6daef3 | COMPLETE | cpu | 5.1335296 | 0.071506 | 5.1335296 | 0 | NOT_RUN |
| a26-transport-pull-a9078a4fd576 | COMPLETE | cpu | 10.097501 | 0.275757 | 10.097501 | 0 | NOT_RUN |
| a26-transport-pull-bb24d8ddb827 | COMPLETE | cpu | 1.9092531 | 0.067035 | 1.9092531 | 0 | NOT_RUN |
| a26-transport-pull-c6b966bc7758 | COMPLETE | cpu | 1.9153548 | 0.059536 | 1.9153548 | 0 | NOT_RUN |
| a26-transport-pull-e61e9fc755b8 | COMPLETE | cpu | 11.732704 | 0.225053 | 11.732704 | 0 | NOT_RUN |
| a26-transport-pull-f8cb51ad384c | COMPLETE | cpu | 14.266621 | 0.37426 | 14.266621 | 0 | NOT_RUN |
| a26-transport-push-2b3adb976a16 | COMPLETE | cpu | 2.2105165 | 0.084553 | 2.2105165 | 0 | NOT_RUN |
| a26-transport-push-41012446afc2 | COMPLETE | cpu | 2.4629064 | 0.195626 | 2.4629064 | 0 | NOT_RUN |
| a26-transport-push-4381f38e7a36 | COMPLETE | cpu | 2.1285401 | 0.088361 | 2.1285401 | 0 | NOT_RUN |
| a26-transport-push-4f861818b231 | COMPLETE | cpu | 2.1321938 | 0.106053 | 2.1321938 | 0 | NOT_RUN |
| a26-transport-push-7ee89f67650c | COMPLETE | cpu | 2.2633277 | 0.133369 | 2.2633277 | 0 | NOT_RUN |
| a26-transport-push-92ed821b9222 | COMPLETE | cpu | 2.3395829 | 0.127801 | 2.3395829 | 0 | NOT_RUN |
| a26-transport-push-9753528f2fec | COMPLETE | cpu | 2.5636699 | 0.239279 | 2.5636699 | 0 | NOT_RUN |
| a26-transport-push-a7f021fbb391 | COMPLETE | cpu | 2.236359 | 0.103773 | 2.236359 | 0 | NOT_RUN |
| a26-transport-run-1116d9c1eae2 | COMPLETE | cpu | 22.654048 | 0.064592 | 22.654048 | 0 | NOT_RUN |
| a26-transport-run-1317a4015db0 | COMPLETE | cpu | 21.950161 | 0.20995 | 21.950161 | 0 | NOT_RUN |
| a26-transport-run-1cae92177571 | COMPLETE | cuda | 213.60139 | 0.094454 | 0.094454 | 0 | NOT_RUN |
| a26-transport-run-23a6475c5266 | COMPLETE | cuda | 5.4045077 | 0.161953 | 0.161953 | 0 | NOT_RUN |
| a26-transport-run-2e544609b70b | COMPLETE | cuda | 19.575241 | 0.111776 | 0.111776 | 0 | NOT_RUN |
| a26-transport-run-2fd6a3ab7c86 | COMPLETE | cuda | 10.407691 | 0.082892 | 0.082892 | 0 | NOT_RUN |
| a26-transport-run-35849a6239e3 | COMPLETE | cuda | 12.997008 | 0.160233 | 0.160233 | 0 | NOT_RUN |
| a26-transport-run-4e1252dba60d | COMPLETE | cuda | 85.396715 | 0.112265 | 0.112265 | 0 | NOT_RUN |
| a26-transport-run-6677f0f3d238 | COMPLETE | cpu | 6.676397 | 0.1991 | 6.676397 | 0 | NOT_RUN |
| a26-transport-run-6a2133338fd5 | COMPLETE | cuda | 11.71176 | 0.168075 | 0.168075 | 0 | NOT_RUN |
| a26-transport-run-8eeddc0bca84 | COMPLETE | cuda | 554.55584 | 0.126091 | 0.126091 | 0 | NOT_RUN |
| a26-transport-run-91d09bf6f00c | COMPLETE | cuda | 11.96592 | 0.167817 | 0.167817 | 0 | NOT_RUN |
| a26-transport-run-91f7cae19931 | COMPLETE | cuda | 60.66347 | 0.082406 | 0.082406 | 0 | NOT_RUN |
| a26-transport-run-a0518bb66d99 | COMPLETE | cpu | 5.903254 | 0.06921 | 5.903254 | 0 | NOT_RUN |
| a26-transport-run-b51bda722882 | COMPLETE | cpu | 21.397825 | 0.089045 | 21.397825 | 0 | NOT_RUN |
| a26-transport-run-d54505df3b5d | COMPLETE | cpu | 23.237522 | 0.061348 | 23.237522 | 0 | NOT_RUN |
| a26-worktree-setup-001 | COMPLETE | cpu | 2 | 2 | 2 | 0 | NOT_RUN |

## Cost-only architecture/update selection

Recorded status: COST_SELECTED; width: 32; U: 2000; jobs: 3; basis: cost_only_no_quality_metric. Safety factor: 1.3. Final reserves from the selection: CPU 1800 s, GPU 900 s. These are cost-admission records, not accuracy results.

| Arm | Width | Device | Seed | Warm updates | Measured updates | Max update wall s | Max update CPU s | Full validation cases | Full validation wall s | Full validation CPU s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N1 | 32 | cuda | 261009 | 1 | 10 | 0.0921289 | 0.09375 | 128 | 1.6400713 | 1.5625 |
| N2 | 32 | cuda | 261009 | 1 | 10 | 0.1016245 | 0.09375 | 128 | 1.6368473 | 1.546875 |
| N3 | 32 | cuda | 261009 | 1 | 10 | 0.0932329 | 0.09375 | 128 | 1.6727236 | 1.578125 |

| Candidate U | Validations/arm | Projected CPU s | Projected GPU s | CPU with reserve s | GPU with reserve s | Fits recorded allowance |
| --- | --- | --- | --- | --- | --- | --- |
| 2000 | 20 | 853.89688 | 875.73474 | 2653.8969 | 1775.7347 | TRUE |
| 1000 | 10 | 427.33438 | 438.3072 | 2227.3344 | 1338.3072 | TRUE |
| 500 | 5 | 214.05312 | 219.59343 | 2014.0531 | 1119.5934 | TRUE |
| 250 | 3 | 110.45938 | 113.45382 | 1910.4594 | 1013.4538 | TRUE |
| 100 | 1 | 43.428125 | 44.622418 | 1843.4281 | 944.62242 | TRUE |

## Actual inference runtimes

Runtime status: MEASURED_RAW_MEASUREMENT_TO_IMAGE. The scope is raw measurement → encoder → transfer → CNN → material box. Offline full-physics audits are excluded from full-inference timing and remain paid separately. Cache-only CNN microbenchmarks above cannot establish full-inference speed. Fresh encoder/checkpoint timings in an existing process and actual fresh-interpreter records remain separate; OS disk cache is uncontrolled.

| Measured scope | Arm | Status | Recorded samples | Background if specified | Median/individual s | Minimum s | Maximum s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RESIDENT_WARM_COUNTED_NATIVE | N1 | RECORDED | 48 | NOT_RUN | 0.04082735 | 0.0224831 | 0.0483271 |
| RESIDENT_WARM_COUNTED_NATIVE | N2 | RECORDED | 48 | NOT_RUN | 0.1260843 | 0.0435951 | 0.1654324 |
| RESIDENT_WARM_COUNTED_NATIVE | N3 | RECORDED | 48 | NOT_RUN | 0.0385077 | 0.0184577 | 0.0462558 |
| INSTRUMENTED_RESIDENT_WARM | N1 | RECORDED | 48 | NOT_RUN | 0.05888995 | 0.0381011 | 0.0754573 |
| INSTRUMENTED_RESIDENT_WARM | N2 | RECORDED | 48 | NOT_RUN | 0.157803 | 0.074177 | 0.2119345 |
| INSTRUMENTED_RESIDENT_WARM | N3 | RECORDED | 48 | NOT_RUN | 0.0526211 | 0.0324458 | 0.0682884 |
| FRESH_ENCODER_CHECKPOINT_SAME_PROCESS | N1 | RECORDED | 2 | NOT_RUN | 1.1411857 | 1.0011151 | 1.2812564 |
| FRESH_ENCODER_CHECKPOINT_SAME_PROCESS | N2 | RECORDED | 2 | NOT_RUN | 1.0532414 | 1.0409122 | 1.0655707 |
| FRESH_ENCODER_CHECKPOINT_SAME_PROCESS | N3 | RECORDED | 2 | NOT_RUN | 0.9976109 | 0.9832496 | 1.0119722 |
| ACTUAL_FRESH_INTERPRETER | N1 | ACTUAL_FRESH_INTERPRETER | 1 | weak | 4.1931959 | NOT_RUN | NOT_RUN |
| NEW_ENCODER_IN_ALREADY_STARTED_INTERPRETER | N1 | ACTUAL_FRESH_INTERPRETER | 1 | ring | 1.0215122 | NOT_RUN | NOT_RUN |
| ACTUAL_FRESH_INTERPRETER | N2 | ACTUAL_FRESH_INTERPRETER | 1 | weak | 3.8322692 | NOT_RUN | NOT_RUN |
| NEW_ENCODER_IN_ALREADY_STARTED_INTERPRETER | N2 | ACTUAL_FRESH_INTERPRETER | 1 | ring | 1.1323902 | NOT_RUN | NOT_RUN |
| ACTUAL_FRESH_INTERPRETER | N3 | ACTUAL_FRESH_INTERPRETER | 1 | weak | 3.1338465 | NOT_RUN | NOT_RUN |
| NEW_ENCODER_IN_ALREADY_STARTED_INTERPRETER | N3 | ACTUAL_FRESH_INTERPRETER | 1 | ring | 1.0667992 | NOT_RUN | NOT_RUN |

Aggregate action/failure ledger: `COST_LEDGER.jsonl` (41279 provenance-wrapped records; 41185 job actions). Scientific quality/speed noninferiority, expansion and locked-opening decisions are owned by the parent.

## Artifact sources

| Path | Source status | Records |
| --- | --- | --- |
| results/a26/learning/261009/N1/training_manifest.json | RECORDED | NOT_RUN |
| results/a26/learning/261009/N1/raw_loss.json | RECORDED | 2000 |
| results/a26/learning/261009/N1/validation_history.json | RECORDED | 20 |
| results/a26/learning/261009/N2/training_manifest.json | RECORDED | NOT_RUN |
| results/a26/learning/261009/N2/raw_loss.json | RECORDED | 2000 |
| results/a26/learning/261009/N2/validation_history.json | RECORDED | 20 |
| results/a26/learning/261009/N3/training_manifest.json | RECORDED | NOT_RUN |
| results/a26/learning/261009/N3/raw_loss.json | RECORDED | 2000 |
| results/a26/learning/261009/N3/validation_history.json | RECORDED | 20 |
| results/a26/learning/COST_SELECTED_UPDATES.json | RECORDED | NOT_RUN |
| results/a26/learning/BENCHMARK_WIDTH32.json | RECORDED | NOT_RUN |
| results/a26/learning/BENCHMARK_WIDTH16.json | NOT_RUN | NOT_RUN |
| results/a26/evaluation/261009/EVALUATION_SUMMARY.json | RECORDED | NOT_RUN |
| results/a26/evaluation/261009/PER_CASE_METRICS.csv | RECORDED | 1024 |
| results/a26/evaluation/261009/PER_PARENT_METRICS.csv | RECORDED | 128 |
| results/a26/evaluation/261009/PER_FAMILY_METRICS.csv | RECORDED | 32 |
| results/a26/evaluation/261009/PAIRED_STATISTICS.json | RECORDED | NOT_RUN |
| results/a26/evaluation/261009/PHYSICS_DIAGNOSTICS.json | RECORDED | 112 |
| results/a26/evaluation/261009/FAILURES.json | RECORDED | 0 |
| results/a26/evaluation/261009/RUNTIME_SUMMARY.json | RECORDED | NOT_RUN |
| results/a26/evaluation/261009/WARM_RUNTIME.csv | RECORDED | 144 |
| results/a26/evaluation/261009/FRESH_RUNTIME.csv | RECORDED | 6 |
| results/a26/evaluation/261009/INSTRUMENTED_RUNTIME.csv | RECORDED | 144 |
| results/a26/evaluation/261009/PROCESS_COLD_N1.json | RECORDED | NOT_RUN |
| results/a26/evaluation/261009/PROCESS_COLD_N2.json | RECORDED | NOT_RUN |
| results/a26/evaluation/261009/PROCESS_COLD_N3.json | RECORDED | NOT_RUN |
| results/a26/cache/weak_BACKGROUND_DIAGNOSTICS.metrics.json | RECORDED | NOT_RUN |
| results/a26/cache/ring_BACKGROUND_DIAGNOSTICS.metrics.json | RECORDED | NOT_RUN |
| results/a26/cache/train_N1_DIAGNOSTICS.json | RECORDED | 128 |
| results/a26/cache/train_N2_DIAGNOSTICS.json | RECORDED | 128 |
| results/a26/cache/train_N3_DIAGNOSTICS.json | RECORDED | 128 |
| results/a26/cache/validation_N1_DIAGNOSTICS.json | RECORDED | 32 |
| results/a26/cache/validation_N2_DIAGNOSTICS.json | RECORDED | 32 |
| results/a26/cache/validation_N3_DIAGNOSTICS.json | RECORDED | 32 |
