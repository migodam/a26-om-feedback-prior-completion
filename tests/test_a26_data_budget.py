"""A26 boundary regressions. Execute only inside the root metered test driver.

All archives are synthetic. Locked-loader success tests mock the array reader;
no historical locked file is ever opened by this module.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from a24.contracts import ForbiddenAccess
from a26.budget import (BudgetExceeded, BudgetLedger, budget_history, charged_job,
                        owned_file_lock, write_json, SHARED_GPU_LOCK)
from a26.data import (FAMILIES, ONLINE_KEYS, PROTOCOL_ID, SOURCE_MANIFEST_ID, SPLIT_PARENT_IDS,
                      load_split, online_from_split, load_locked_split, validate_locked_authorization)


def fixture(role="train"):
    ids = SPLIT_PARENT_IDS[role]
    geometry = {"grid": 24, "sources": 8, "receivers": 32, "source_axis": -1,
                "precision": "complex128/float64", "volume": 1. / 576,
                "current_normalization": "sqrt(cell_area)*physical_current", "wavenumber": 2*np.pi,
                "points": np.zeros((576, 2)).tolist(), "source_points": np.zeros((8, 2)).tolist(),
                "receiver_points": np.zeros((32, 2)).tolist()}
    parents, rows = [], []
    for offset, pid in enumerate(ids):
        family = FAMILIES[offset // (len(ids)//4)]
        parents.append(dict(parent_id=pid, split=role, family=family, ood_type="ID", amplitudes=[.3, 1.2]))
        for background in ("weak", "ring"):
            for ai, amplitude in enumerate((.3, 1.2)):
                for ni in (0, 1):
                    rows.append(dict(parent_id=pid, split=role, family=family, background_id=background,
                                     ood_type="ID", amplitude=amplitude, noise_db=np.nan if ni == 0 else 20.,
                                     sigma=0. if ni == 0 else .001, status="OK",
                                     variant_id=f"{background}:a{ai}:n{ni}",
                                     case_id=f"p{pid}_{background}_a{ai}_"+("clean" if ni == 0 else "snr20")))
    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    m = len(rows)
    arrays.update(manifest_id=np.asarray(SOURCE_MANIFEST_ID), geometry_descriptor=np.asarray(json.dumps(geometry)),
                  data=np.zeros((m, 32, 8), np.complex128), chi=np.zeros((m, 576), np.complex128),
                  initial_mass=np.zeros((m, 1152), np.float64), **{"lambda": np.ones(m, np.float64)},
                  initial_wall=np.zeros(m, np.float64), initializer_id=np.asarray("A25-legacy-TV-fixture"),
                  current=np.zeros(1), clean=np.zeros(1), noise_seed=np.zeros((m, 6), np.int64))
    manifest = dict(manifest_id=SOURCE_MANIFEST_ID, master_seed=251008, families=list(FAMILIES),
                    backgrounds=["weak", "ring"], amplitudes=[.3, 1.2], noise_snr_db=[None, 20],
                    splits={role: list(ids)}, parents=parents)
    return manifest, geometry, arrays


class TracedArchive:
    def __init__(self, arrays):
        self.arrays, self.files, self.accesses = arrays, list(arrays), []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __getitem__(self, key):
        self.accesses.append(key)
        if key in {"current", "clean", "noise_seed"}:
            raise AssertionError("excluded source payload was indexed: " + key)
        return self.arrays[key]


@contextmanager
def traced_fixture(role="train", changed=None):
    manifest, geometry, arrays = fixture(role)
    if changed:
        arrays.update(changed)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        write_json(path/"manifest.json", manifest)
        write_json(path/"geometry.json", geometry)
        traced = TracedArchive(arrays)
        with patch("a26.data.np.load", return_value=traced):
            yield path, traced


def authorization():
    freeze = {"protocol_id": PROTOCOL_ID, "status": "FROZEN", "model_selection_frozen": True,
              "width": 32, "updates": 100,
              "checkpoints": [dict(method=method, seed=seed, checkpoint=f"results/a26/{seed}/{method}/best.pt",
                                   width=32, completed_updates=100, status="TRAINED_VALIDATION_SELECTED")
                              for seed in (261009, 261010) for method in ("N1", "N2", "N3")]}
    gate = {"protocol_id": PROTOCOL_ID, "primary_gate": "PASS", "repeat_gate": "PASS",
            "repeat_independently_confirmed": True, "data_complete": True}
    return freeze, gate


class DataFirewallTests(unittest.TestCase):
    def test_exact_coverage_and_truth_access_order(self):
        for role, count in (("train", 512), ("validation", 128)):
            with self.subTest(role=role), traced_fixture(role) as (root, archive):
                arrays = load_split(root, role)
                self.assertEqual(arrays["data"].shape, (count, 32, 8))
                self.assertEqual(arrays["chi"].shape, (count, 576))
                self.assertEqual(archive.accesses[:2], ["split", "parent_id"])
                self.assertGreater(archive.accesses.index("chi"), archive.accesses.index("geometry_descriptor"))
                self.assertGreater(archive.accesses.index("data"), archive.accesses.index("status"))
                self.assertFalse({"current", "clean", "noise_seed"} & set(archive.accesses))
                self.assertEqual(set(arrays["noise_group"]), {"NOISE_FREE", "NOISY"})

    def test_unapproved_roles_do_not_touch_files_or_arrays(self):
        with patch("a26.data.np.load") as loader, patch("a26.data._read_json") as reader:
            for role in ("locked_id", "locked_ood", "evaluation", "online", "offline_generator"):
                with self.subTest(role=role), self.assertRaises(ForbiddenAccess):
                    load_split("/unopened/locked", role)
            loader.assert_not_called()
            reader.assert_not_called()

    def test_disguised_locked_split_fails_before_observation_and_truth(self):
        _, _, arrays = fixture()
        with traced_fixture(changed={"split": np.full(512, "locked_id")}) as (root, archive):
            with self.assertRaises(ForbiddenAccess):
                load_split(root, "train")
            self.assertEqual(archive.accesses, ["split", "parent_id"])
        parents = arrays["parent_id"].copy()
        parents[0] = 250081
        with traced_fixture(changed={"parent_id": parents}) as (root, archive):
            with self.assertRaises(ForbiddenAccess):
                load_split(root, "train")
            self.assertEqual(archive.accesses, ["split", "parent_id"])

    def test_parent_family_condition_noise_status_and_geometry_guards_precede_truth(self):
        _, geometry, arrays = fixture()
        family = arrays["family"].copy(); family[0] = "smooth"
        status = arrays["status"].astype("U30"); status[0] = "INITIALIZATION_FAILED"
        sigma = arrays["sigma"].copy(); sigma[0] = .1
        case = arrays["case_id"].copy(); case[0] = case[1]
        wrong_geometry = dict(geometry, volume=1.)
        for changed in ({"family": family}, {"status": status}, {"sigma": sigma}, {"case_id": case},
                        {"geometry_descriptor": np.asarray(json.dumps(wrong_geometry))}):
            with self.subTest(field=next(iter(changed))), traced_fixture(changed=changed) as (root, archive):
                with self.assertRaises((ForbiddenAccess, RuntimeError)):
                    load_split(root, "train")
                self.assertFalse({"data", "chi", "current"} & set(archive.accesses))

    def test_symlink_and_unknown_key_are_rejected_without_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"locked").mkdir()
            (root/"train.npz").symlink_to(root/"locked/locked_id.npz")
            with patch("a26.data.np.load") as loader, self.assertRaises(ForbiddenAccess):
                load_split(root, "train")
            loader.assert_not_called()
        with traced_fixture(changed={"hidden_truth": np.zeros(1)}) as (root, archive):
            with self.assertRaises(ForbiddenAccess):
                load_split(root, "train")
            self.assertEqual(archive.accesses, [])

    def test_online_copies_only_allowed_observation_fields_and_accepts_batch(self):
        _, _, arrays = fixture()
        arrays["chi"] = object()
        arrays["current"] = object()
        online = online_from_split(arrays)
        self.assertEqual(set(online), set(ONLINE_KEYS))
        self.assertFalse({"chi", "current", "clean", "initial_mass", "lambda", "initial_wall",
                          "initializer_id", "noise_seed"} & set(online))
        batch = {key: value[:8] if value.ndim else value for key, value in online.items()}
        self.assertEqual(online_from_split(batch)["data"].shape, (8, 32, 8))
        online["data"][0] = 1
        self.assertEqual(arrays["data"][0].sum(), 0)
        bad = dict(batch, data=np.zeros((8, 8, 32), np.complex128))
        with self.assertRaises(ValueError):
            online_from_split(bad)

    def test_future_locked_guard_and_once_claims_read_no_locked_payload(self):
        freeze, gate = authorization()
        with patch("a26.data.np.load") as loader:
            with self.assertRaises(ForbiddenAccess):
                load_locked_split("/unopened", "locked_id", a26_root="/unused", book=None,
                                  freeze_record=freeze, gate_record=dict(gate, repeat_gate="FAIL"))
            with self.assertRaises(ForbiddenAccess):
                load_locked_split("/unopened", "locked_id", a26_root="/unused", book=None,
                                  freeze_record=dict(freeze, protocol_id="A25"), gate_record=gate)
            loader.assert_not_called()
        self.assertEqual(len(validate_locked_authorization(freeze, gate)["checkpoints"]), 6)
        with tempfile.TemporaryDirectory() as directory:
            with charged_job(directory, "synthetic-once-guard") as book:
                with patch("a26.data._load_verified", return_value={}) as loader:
                    for role in ("locked_id", "locked_ood"):
                        load_locked_split("/unopened", role, a26_root=directory, book=book,
                                          freeze_record=freeze, gate_record=gate)
                    with self.assertRaises(ForbiddenAccess):
                        load_locked_split("/unopened", "locked_id", a26_root=directory, book=book,
                                          freeze_record=freeze, gate_record=gate)
                    self.assertEqual(loader.call_count, 2)
                self.assertEqual(book.reservations["a26_locked_cohort_open"], 1)
                self.assertEqual(book.reservations["a26_locked_id_open"], 1)
                self.assertEqual(book.reservations["a26_locked_ood_open"], 1)
            self.assertFalse((Path(directory)/"results/a25").exists())


class BudgetSafetyTests(unittest.TestCase):
    def test_a26_history_is_independent_and_failed_costs_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            for job, device, wall, process, gpu, status in (("failed", "cpu", 8., 3., 0., "FAILED"),
                                                           ("cuda", "cuda", 12., 2., 12., "COMPLETE")):
                write_json(Path(directory)/"results/a26/jobs"/job/"job_receipt.json",
                           dict(job=job, device=device, wall_seconds=wall, process_cpu_seconds=process,
                                gpu_occupation_seconds=gpu, status=status, durable_reservations={}))
            old = Path(directory)/"results/a25/jobs/old/job_receipt.json"
            write_json(old, {"process_cpu_seconds": 99999., "durable_reservations": {"new_full_forward_labels": 456}})
            saved = old.read_bytes()
            history = budget_history(directory)
            self.assertEqual(history["cpu_seconds"], 10.)
            self.assertEqual(history["gpu_seconds"], 12.)
            self.assertEqual(old.read_bytes(), saved)
            self.assertEqual(len(history["receipts"]), 2)

    def test_invalid_cuda_occupation_receipt_cannot_erase_gpu_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            write_json(Path(directory)/"results/a26/jobs/invalid/job_receipt.json",
                       dict(job="invalid", status="COMPLETE", device="cuda", wall_seconds=10.,
                            process_cpu_seconds=1., gpu_occupation_seconds=0., durable_reservations={}))
            with self.assertRaises(BudgetExceeded):
                budget_history(directory)

    def test_historical_scan_happens_once_at_job_start(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("a26.budget.budget_history", wraps=budget_history) as scan:
                with charged_job(directory, "once") as book:
                    for _ in range(4):
                        book.check()
                        with book.span("synthetic_operator"):
                            pass
                self.assertEqual(scan.call_count, 1)

    def test_zero_label_cap_is_hard_for_all_aliases_positive_limits_and_history(self):
        book = BudgetLedger(enforce=False)
        for category in ("new_labels", "new_clean_labels", "new_full_forward_labels"):
            with self.subTest(category=category), self.assertRaises(BudgetExceeded):
                book.reserve(category, 999999)
        self.assertFalse(book.reservations)
        with self.assertRaises(BudgetExceeded):
            BudgetLedger(prior_reservations={"new_labels": 1})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                with charged_job(directory, "illegal", config={"new_full_forward_labels_max": 1}):
                    pass
            write_json(Path(directory)/"results/a26/jobs/illegal/job_receipt.json",
                       dict(job="illegal", status="FAILED", device="cpu", wall_seconds=1., process_cpu_seconds=1.,
                            durable_reservations={"new_full_forward_labels": 1}))
            with self.assertRaises(BudgetExceeded):
                budget_history(directory)

    def test_terminal_receipts_immutable_and_unknown_interruption_blocks_next_job(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                with charged_job(directory, "failed"):
                    raise RuntimeError("synthetic failure")
            receipt = Path(directory)/"results/a26/jobs/failed/job_receipt.json"
            original = receipt.read_bytes()
            self.assertEqual(json.loads(original)["status"], "FAILED")
            with self.assertRaises(BudgetExceeded):
                with charged_job(directory, "failed"):
                    pass
            self.assertEqual(receipt.read_bytes(), original)
            write_json(Path(directory)/"results/a26/jobs/unknown/job_manifest.json", {"status": "STARTED"})
            with self.assertRaises(BudgetExceeded):
                with charged_job(directory, "next"):
                    pass
            self.assertFalse((Path(directory)/"results/a26/jobs/next").exists())

    def test_interrupt_receipt_and_foreign_locks_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(KeyboardInterrupt):
                with charged_job(directory, "interrupt"):
                    raise KeyboardInterrupt()
            receipt = Path(directory)/"results/a26/jobs/interrupt/job_receipt.json"
            self.assertEqual(json.loads(receipt.read_text())["status"], "INTERRUPTED")
            self.assertFalse(budget_history(directory)["unresolved_jobs"])
            path = Path(directory)/"foreign.lock"
            with owned_file_lock(path, owner_id="mine", job="unit") as own:
                ledger = BudgetLedger(enforce=False, metadata={"owner_id": "mine"})
                ledger._owned_locks = [own]
                path.write_bytes(b'{"owner_id":"foreign"}')
                with self.assertRaises(BudgetExceeded):
                    ledger.check()
            self.assertEqual(path.read_bytes(), b'{"owner_id":"foreign"}')
            self.assertEqual(SHARED_GPU_LOCK.as_posix(), "D:/AI/A20_OPM_IMAGING/runs/gpu.lock")

    def test_cpu_wall_and_monitor_accounting_and_caps(self):
        book = BudgetLedger(enforce=False)
        book.started_wall -= 5.
        self.assertGreaterEqual(book.receipt()["charged_cpu_seconds"], 5.)
        monitor = BudgetLedger(device="cuda", gpu_host_only=True, enforce=False)
        monitor.started_wall -= 100.
        receipt = monitor.receipt()
        self.assertEqual(receipt["gpu_occupation_seconds"], 0.)
        self.assertLess(receipt["charged_cpu_seconds"], 5.)
        for kwargs in ({"cpu_limit": 7201}, {"gpu_limit": 14401}, {"cpu_reserve": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BudgetLedger(**kwargs)


if __name__ == "__main__":
    unittest.main()
