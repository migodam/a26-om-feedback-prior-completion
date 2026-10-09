# Reproducing the A26 frozen-encoder pilot

The public inputs are existing TRAIN and exposed VALIDATION observations/labels
repackaged from A25. No new simulation labels or locked files are needed. The
three trained CNNs and raw predictions are genuine experiment outputs.

## Public files and installation

Download the assets from the [A26 release](https://github.com/migodam/a26-om-feedback-prior-completion/releases/tag/a26-v0.1.0):

- `A26_REPRO_INPUTS.zip`: extract at the repository root; creates `data/`, the
  original `configs/a25/nn_admitted.json`, and `REPRO_INPUT_AUDIT.json`. The
  background Jacobians belong to the known background only. No current teacher
  or unopened locked payload is included.
- `A26_CHECKPOINTS.zip`: extract at the repository root; places all three
  best/middle/final checkpoints in `results/a26/learning/261009/`.
- `A26_RAW_EVIDENCE.zip`: feature caches, observation-only arrays, prediction
  arrays, full per-action ledgers and run logs. Images/metrics/configs/receipts
  are also readable directly in Git. Nested action timing is not summed into
  resource totals.

Use Python 3.10+, install NumPy/SciPy/Matplotlib and an appropriate existing
PyTorch installation. The original GPU experiment used PyTorch 2.6.0+cu124,
NumPy 2.2.6, SciPy 1.15.3 on Windows with an RTX4060 Laptop 8GB. A fresh local
environment can use `pip install -r requirements-a26.txt`. CUDA installation
is outside this protocol; choose a PyTorch wheel compatible with your system.

## Inspect without opening the locked cohort

```sh
python -m json.tool configs/a26/pilot.json
python -m json.tool GATE_DECISION.json
```

The original gate failed, so a second seed and locked ID/OOD evaluation are
NOT_RUN. The checked data loader validates real paths/splits/parent IDs before
payload loading. Do not rename, expose or manufacture locked data to bypass it.

## Fresh computation

Run from the repository root with the module search path set:

```sh
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -B tools/a26_driver.py --root . --data-root data --stage audit --job reproduce-audit-001 --device cpu
python -B tools/a26_driver.py --root . --data-root data --stage tests --job reproduce-tests-001 --device cpu
```

The all-source test suite includes three optional Clarabel tests inherited
from A25; if Clarabel 0.11.1 is not installed they are explicitly skipped.
These do not certify an unrun TV trajectory in A26.

The experiment's accepted results are immutable. To recompute rather than
inspect, use a separate fresh working copy, keep `data/` and configurations,
and move the archived `results/a26/` into a separate historical directory.
The driver refuses to overwrite primary feature/evaluation artifacts. Existing
budget receipts enforce remaining A26 limits; never reset them in the same
campaign. A separately named replication has its own explicit resource cap.

```sh
python -B tools/a26_driver.py --root . --data-root data --stage cache --job reproduce-cache-001 --device cuda
python -B tools/a26_driver.py --root . --data-root data --stage benchmark --job reproduce-benchmark-001 --device cuda
python -B tools/a26_driver.py --root . --data-root data --stage train --job reproduce-train-001 --device cuda
python -B tools/a26_driver.py --root . --data-root data --stage evaluate --job reproduce-evaluate-001 --device cuda
python -B tools/a26_driver.py --root . --data-root data --stage timing --job reproduce-timing-001 --device cuda
python -B tools/a26_driver.py --root . --data-root data --stage report --job reproduce-report-001 --device cpu
```

The benchmark cost selector can choose fewer updates on a slower machine;
this is a cost adaptation, not an exact 2000-update replication. The original
`COST_SELECTED_UPDATES.json`, histories and manifests preserve the actual
2000-update selection. All arms must receive identical updates and sampling.
The driver uses the shared GPU lock; on another host explicitly configure a
local safe lock path rather than claim to hold XINAN's lock.

## Fresh interpreter latency

Each command below starts a separate process. Disk cache is uncontrolled; no
disk-cold performance claim is made. The first background's image includes
interpreter import, background preparation, checkpoint loading and inference.

```sh
python -B tools/a26_driver.py --root . --data-root data --stage process-cold --config configs/a26/process_cold_N1.json --job reproduce-cold-N1-001 --device cuda
python -B tools/a26_driver.py --root . --data-root data --stage process-cold --config configs/a26/process_cold_N2.json --job reproduce-cold-N2-001 --device cuda
python -B tools/a26_driver.py --root . --data-root data --stage process-cold --config configs/a26/process_cold_N3.json --job reproduce-cold-N3-001 --device cuda
```

Raw-to-image timing includes the encoder. Offline predicted-image full-Maxwell
audits are separately priced; cache-only CNN speed is never total inference.
Source/Git commits preserve provenance. SHA256 checks are deliberately absent.

