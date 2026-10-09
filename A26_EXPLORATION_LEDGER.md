# A26 exploration ledger — seed 261009

These are factual records of frozen requirements and available attempts. Hypotheses label the registered purpose of an activity; they are not findings. Every result and decision remains tied to its source. Parent gate artifact: `results/a26/evaluation/261009/GATE_DECISION.json`. Missing primary/repeat stages are NOT_RUN, and this report never opens locked cohorts.

| Activity | Hypothesis/purpose | Evidence | Change | Development scope | Recorded result | Decision owner/basis | Cost evidence | Validation eligibility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen feature extraction | Separate explicit physics and observation features under the frozen protocol | results/a26/cache/{train,validation}_N*_DIAGNOSTICS.json | Frozen N1/N2/N3 feature definitions | TRAIN plus historically exposed validation | See recorded diagnostic coverage; missing entries are NOT_RUN | Parent-owned GATE_DECISION.json | Accepted feature-job receipts and action counters | Development only; no locked arrays read by this report |
| Actual TRAIN microbenchmark | Three mandatory arms fit the frozen resource allowance | results/a26/learning/BENCHMARK_WIDTH*.json; COST_SELECTED_UPDATES.json | Predeclared width32; width16 only after cost failure | One actual TRAIN batch, warm1 + measured10 updates/arm | COST_SELECTED | cost_only_no_quality_metric | Measured update and full128-row validation costs; safety1.3 plus reserves | Cost admission only; benchmark weights discarded |
| N1 training seed 261009 | Frozen learned increment may improve its own physical base | results/a26/learning/261009/N1/training_manifest.json | Identical architecture, initialization, loss and parent/condition schedule | 64 TRAIN parents; checkpoint selection on all16 exposed validation parents | TRAINED_EXPOSED_VALIDATION_SELECTED | Checkpoint record only; parent reviews science | Recorded scientific updates: 2000 | Exposed development; not blind confirmation |
| N2 training seed 261009 | Frozen learned increment may improve its own physical base | results/a26/learning/261009/N2/training_manifest.json | Identical architecture, initialization, loss and parent/condition schedule | 64 TRAIN parents; checkpoint selection on all16 exposed validation parents | TRAINED_EXPOSED_VALIDATION_SELECTED | Checkpoint record only; parent reviews science | Recorded scientific updates: 2000 | Exposed development; not blind confirmation |
| N3 training seed 261009 | Frozen learned increment may improve its own physical base | results/a26/learning/261009/N3/training_manifest.json | Identical architecture, initialization, loss and parent/condition schedule | 64 TRAIN parents; checkpoint selection on all16 exposed validation parents | TRAINED_EXPOSED_VALIDATION_SELECTED | Checkpoint record only; parent reviews science | Recorded scientific updates: 2000 | Exposed development; not blind confirmation |
| Independent predicted-image evaluation | Image metrics and exact full-physics residuals may expose model/prior failures | results/a26/evaluation/261009/EVALUATION_SUMMARY.json; PER_*_METRICS.csv; PHYSICS_DIAGNOSTICS.json | Offline full audits at predictions; no new truth labels | All128 exposed observations; recorded typical-state diagnostics | COMPLETE_EXPOSED_DEVELOPMENT_EVALUATION | Parent-owned GATE_DECISION.json | Accepted evaluation receipts | Historically exposed validation; failures retained |
| Full inference runtime | Complete raw-measurement inference cost may differ across arms | results/a26/evaluation/261009/WARM_RUNTIME.csv; FRESH_RUNTIME.csv; PROCESS_COLD_N*.json | Recorded warm, instrumented, same-process fresh and fresh-interpreter scopes | Frozen metadata-selected cases; no quality-selected timing cases | MEASURED_RAW_MEASUREMENT_TO_IMAGE | Parent owns quality/speed gate | Actual timing plus separately paid offline audits | Runtime evidence only; OS disk cache uncontrolled |
| a26-process-cold-N3-001 | Operational attempt as recorded by its job metadata | results/a26/jobs/a26-process-cold-N3-001/job_receipt.json | NOT_RUN | Accepted attempt receipt | INTERRUPTED: None | Attempt terminated; no scientific inference | CPU 0.140625 s; GPU 0.2275486 s | Failure is retained; retries do not erase its cost |
| a26-runtime-mismatch-001 | Operational attempt as recorded by its job metadata | results/a26/jobs/a26-runtime-mismatch-001/job_receipt.json | NOT_RUN | Accepted attempt receipt | FAILED: None | Attempt terminated; no scientific inference | CPU 1 s; GPU 0 s | Failure is retained; retries do not erase its cost |
| a26-tests-001 | Operational attempt as recorded by its job metadata | results/a26/jobs/a26-tests-001/job_receipt.json | tests | Accepted attempt receipt | FAILED: RuntimeError: UNIT_TESTS_FAILED:{"tests": 92, "errors": 50, "failures": 0, "passed": false} | Attempt terminated; no scientific inference | CPU 16.04374 s; GPU 0 s | Failure is retained; retries do not erase its cost |
| a26-tests-002 | Operational attempt as recorded by its job metadata | results/a26/jobs/a26-tests-002/job_receipt.json | tests | Accepted attempt receipt | FAILED: RuntimeError: UNIT_TESTS_FAILED:{"tests": 108, "errors": 4, "failures": 0, "passed": false} | Attempt terminated; no scientific inference | CPU 15.229222 s; GPU 0 s | Failure is retained; retries do not erase its cost |
| a26-training-benchmark-001 | Operational attempt as recorded by its job metadata | results/a26/jobs/a26-training-benchmark-001/job_receipt.json | benchmark | Accepted attempt receipt | FAILED: RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility | Attempt terminated; no scientific inference | CPU 2.59375 s; GPU 3.5015058 s | Failure is retained; retries do not erase its cost |

## Actual evaluation failures

NOT_RUN — required actual records are absent.

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
