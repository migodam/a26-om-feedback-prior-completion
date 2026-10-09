"""Synthetic cache-only checks; run by the parent's metered test driver."""
from contextlib import nullcontext, ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from a24.contracts import ForbiddenAccess
from a26 import training as T


CONFIG = {"protocol_id": "A26-FROZEN-ENCODER-20261009-v1", "source_commit": "fixture",
          "primary_seed": 261009, "repeat_seed": 261010, "updates_candidates": [2000, 1000, 500, 250, 100],
          "cost_safety_factor": 1.3, "validation_every": 100, "lr": .001, "weight_decay": .00001,
          "grad_clip": 1., "ssim_weight": .05, "final_cpu_reserve_seconds": 1800,
          "final_gpu_reserve_seconds": 900, "real_lower": -.5, "real_upper": 4.,
          "imag_lower": 0., "imag_upper": .5}


class Book:
    def __init__(self):
        self.counts = {}

    def span(self, name, **counts):
        for key, value in counts.items():
            self.counts[key] = self.counts.get(key, 0) + value
        return nullcontext()

    def check(self):
        pass

    def synchronize(self):
        pass


def fixture(role):
    parents = np.arange(250001, 250065) if role == "train" else np.arange(250065, 250081)
    per_family = len(parents) // 4
    parent_id = np.repeat(parents, 8)
    conditions = np.tile(np.arange(8), len(parents))
    count = len(parent_id)
    sigma = np.where(conditions % 2 == 0, 0., .01).astype(np.float64)
    metadata = {"case_id": np.array([f"{p}:condition{c}" for p, c in zip(parent_id, conditions)]),
                "parent_id": parent_id,
                "family": np.repeat(np.repeat(np.asarray(T.FAMILIES), per_family), 8),
                "background_id": np.where(conditions < 4, "weak", "ring"), "sigma": sigma,
                "noise_db": np.where(sigma == 0, np.nan, 20.).astype(np.float64),
                "split": np.full(count, role), "amplitude": np.where(conditions % 4 < 2, .3, 1.2),
                "variant_id": conditions}
    known = {"weak": np.full(T.CELLS, .1 + .01j, dtype=np.complex128),
             "ring": np.full(T.CELLS, .6 + .03j, dtype=np.complex128)}
    background = np.stack([known[name] for name in metadata["background_id"]])
    delta = np.repeat(np.linspace(.1, .2, len(parents)), 8)[:, None] * (1. + .2j)
    chi = np.asarray(background + delta, dtype=np.complex128)
    labels = {**metadata, "chi": chi}
    base = np.concatenate((background.real, background.imag), axis=1) / 24.
    datasets = {}
    for method, om_value in zip(T.METHODS, (0., 0., 3.)):
        features = np.zeros((count, T.CHANNELS, T.GRID, T.GRID), dtype=np.float32)
        features[:, 2] = om_value
        features[:, 8] = background.real.reshape(count, T.GRID, T.GRID).astype(np.float32)
        features[:, 9] = background.imag.reshape(count, T.GRID, T.GRID).astype(np.float32)
        cache = {**metadata, "features": features, "base_mass": base.copy(), "initial_mass": base.copy(),
                 "beta": (np.where(sigma == 0, np.nan, 1.).astype(np.float64)
                          if method == "N2" else np.full(count, np.nan, dtype=np.float64)),
                 "method": np.asarray(method), "cache_schema": np.asarray(T.CACHE_SCHEMA),
                 "protocol_id": np.asarray(CONFIG["protocol_id"])}
        datasets[method] = T.CachedDataset(cache, labels, role=role, background_materials=known)
    return datasets, labels, known


def benchmark_fixture(width=32, rate=.1, validation=2., device="cpu"):
    return {"width": width, "seed": 261009, "device": device, "rows": [
        {"method": method, "width": width, "batch_size": 4, "warmup_updates": 1, "measured_updates": 10,
         "validation_case_count": 128, "validation_parent_count": 16,
         "update_wall_seconds": rate, "update_cpu_seconds": rate,
         "validation_wall_seconds": validation, "validation_cpu_seconds": validation}
        for method in T.METHODS]}


class CacheTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        T.configure_torch(261009, "cpu")
        cls.train, cls.labels, cls.known = fixture("train")
        cls.validation, _, _ = fixture("validation")
        cls.normalizer = T.fit_normalizer(cls.train, config=CONFIG, book=Book())

    def test_locked_role_rejected_before_any_archive_open(self):
        with patch.object(T.np, "load") as load:
            with self.assertRaises(ForbiddenAccess):
                T.load_feature_cache("never-open.npz", role="locked_id")
            load.assert_not_called()

    def test_metadata_firewall_precedes_feature_payload(self):
        cache = dict(self.train["N1"].cache)
        cache["parent_id"] = np.full(512, 250081, dtype=np.int64)
        touched = []

        class Archive:
            files = list(cache)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def __getitem__(self, key):
                touched.append(key)
                if key in T.PAYLOAD_KEYS:
                    raise AssertionError("payload read before firewall")
                return cache[key]

        with patch.object(T.np, "load", return_value=Archive()):
            with self.assertRaises(ForbiddenAccess):
                T.load_feature_cache("metadata-only.npz", role="train", method="N1")
        self.assertFalse(set(touched) & set(T.PAYLOAD_KEYS))

    def test_feature_cache_rejects_truth_and_is_immutable(self):
        cache = self.train["N1"].cache
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "N1-train.npz"
            with self.assertRaises(ForbiddenAccess):
                T.write_feature_cache(path, {**cache, "chi": self.labels["chi"]}, method="N1", role="train", config=CONFIG)
            self.assertFalse(path.exists())
            T.write_feature_cache(path, cache, method="N1", role="train", config=CONFIG)
            result = T.load_feature_cache(path, role="train", method="N1", config=CONFIG)
            self.assertNotIn("chi", result)
            np.testing.assert_array_equal(result["features"], cache["features"])
            with self.assertRaises(ForbiddenAccess):
                T.write_feature_cache(path, cache, method="N1", role="train", config=CONFIG)

    def test_sampler_is_matched_balanced_and_covers_all_rows(self):
        samplers = [T.ParentSampler(self.train[method], 261009) for method in T.METHODS]
        for _ in range(2):
            seen = []
            for _ in range(128):
                rows = [sampler.next() for sampler in samplers]
                for other in rows[1:]:
                    np.testing.assert_array_equal(other, rows[0])
                self.assertEqual(len(set(self.train["N1"].cache["parent_id"][rows[0]])), 4)
                self.assertEqual(set(self.train["N1"].cache["family"][rows[0]]), set(T.FAMILIES))
                seen.extend(rows[0].tolist())
            self.assertEqual(sorted(seen), list(range(512)))

    def test_beta_is_only_computed_by_n2_and_never_in_noiseless_cases(self):
        for method in ("N1", "N3"):
            cache = dict(self.train[method].cache)
            T._validate_payload(cache, 512)
            cache["beta"] = np.where(cache["sigma"] == 0, np.nan, 1.).astype(np.float64)
            with self.assertRaises(ValueError):
                T._validate_payload(cache, 512)
        n2 = dict(self.train["N2"].cache)
        T._validate_payload(n2, 512)
        n2["beta"] = np.zeros(512, dtype=np.float64)
        with self.assertRaises(ValueError):
            T._validate_payload(n2, 512)

    def test_pooled_train_statistics_and_delta_scale(self):
        normalizer = self.normalizer
        self.assertAlmostEqual(float(normalizer.channel_mean.reshape(-1)[2]), 1.)
        self.assertAlmostEqual(float(normalizer.channel_std.reshape(-1)[2]), 2. ** .5, places=6)
        self.assertGreaterEqual(float(normalizer.channel_std.min()), .999999e-6)
        scale = np.sqrt(np.mean(np.abs(self.train["N1"].delta_chi) ** 2))
        self.assertAlmostEqual(float(normalizer.material_scale), scale, places=14)
        self.assertEqual(normalizer.provenance["pooled_feature_images"], 1536)
        with self.assertRaises(ForbiddenAccess):
            T.fit_normalizer(self.validation, config=CONFIG, book=Book())

    def test_identical_initialization_and_actual_cache_only_update(self):
        models = T.build_matched_models(config=CONFIG)
        self.assertEqual(sum(p.numel() for p in models["N1"].parameters()), 31810)
        for key, value in models["N1"].state_dict().items():
            for method in ("N2", "N3"):
                torch.testing.assert_close(value, models[method].state_dict()[key], rtol=0, atol=0)
        model = models["N1"]
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        rows = T.ParentSampler(self.train["N1"], 261009).next()
        batch = self.train["N1"].batch(rows)
        self.assertNotIn("data", batch)
        self.assertNotIn("current", batch)
        book = Book()
        components, gradient = T.one_update(model, T._optimizer(model, CONFIG), batch,
                                            normalizer=self.normalizer, config=CONFIG, book=book, phase="benchmark")
        self.assertTrue(np.isfinite(list(components.values())).all())
        self.assertGreater(gradient, 0.)
        self.assertTrue(any(not torch.equal(initial[key], value) for key, value in model.state_dict().items()))
        self.assertEqual(book.counts["benchmark_training_updates"], 1)
        self.assertNotIn("scientific_training_updates", book.counts)

    def test_delta_loss_has_exact_target_minimum(self):
        batch = self.train["N1"].batch([0, 8, 16, 24])
        loss, components = T.delta_loss(batch["chi"], batch, self.normalizer.material_scale)
        self.assertAlmostEqual(float(loss), 0., places=10)
        self.assertEqual(float(components["delta_mse"]), 0.)
        changed = batch["chi"] + .02 + .01j
        _, components = T.delta_loss(changed, batch, self.normalizer.material_scale)
        self.assertAlmostEqual(float(components["delta_mse"]), .0005, places=12)
        with self.assertRaises(ValueError):
            T.ssim_delta(changed.to(torch.complex64), batch["delta_chi"])

    def test_backward_cannot_call_maxwell_or_encoder_operations(self):
        from a25.torch_physics import TorchReference, TorchRelativeModel
        from a26.encoder import FrozenEncoder
        models = T.build_matched_models(config=CONFIG)
        batch = self.train["N1"].batch(T.ParentSampler(self.train["N1"], 261009).next())
        for key in ("features", "base_mass", "chi"):
            self.assertFalse(batch[key].requires_grad)
        with ExitStack() as stack:
            for cls, names in ((TorchReference, ("apply", "green", "observe", "apply_db", "solve_background")),
                               (TorchRelativeModel, ("forward", "jvp", "adjoint", "vjp_complex", "solve_A")),
                               (FrozenEncoder, ("encode",))):
                for name in names:
                    stack.enter_context(patch.object(cls, name, side_effect=AssertionError("PHYSICS_IN_DECODER_BACKWARD")))
            T.one_update(models["N1"], T._optimizer(models["N1"], CONFIG), batch,
                         normalizer=self.normalizer, config=CONFIG, book=Book(), phase="benchmark")

    def test_validation_and_prediction_preserve_complete_rows_and_dtypes(self):
        model = T.build_matched_models(config=CONFIG, width=16)["N1"]
        result = T.validate(model, self.validation["N1"], normalizer=self.normalizer, config=CONFIG, book=Book())
        self.assertEqual((result["parent_count"], result["case_count"]), (16, 128))
        self.assertAlmostEqual(result["parent_mean_delta_nrmse"], 1., places=12)
        self.assertEqual({row["conditions"] for row in result["parent_rows"]}, {8})
        prediction = T.predict_cached(model, self.normalizer, self.validation["N1"], batch=16)
        self.assertEqual(prediction["chi"].dtype, np.complex128)
        self.assertEqual(prediction["mass"].dtype, np.float64)
        np.testing.assert_array_equal(prediction["metadata"]["case_id"], self.validation["N1"].cache["case_id"])

    def test_cost_selection_accounts_for_final_validation_and_reserve(self):
        result = T.select_update_budget(benchmark_fixture(), config=CONFIG, cpu_available=1930., gpu_available=0.)
        self.assertEqual((result["updates"], result["width"], result["complete_jobs"]), (250, 32, 3))
        row = next(row for row in result["candidates"] if row["updates"] == 250)
        self.assertEqual(row["validation_events_per_arm"], 3)
        self.assertAlmostEqual(row["required_cpu_with_reserve"], 1800 + 1.3 * 3 * (250 * .1 + 3 * 2.))
        self.assertEqual(T._validation_steps(250), [100, 200, 250])

    def test_width_fallback_requires_prior_cost_failure(self):
        with self.assertRaises(ForbiddenAccess):
            T.select_update_budget(benchmark_fixture(width=16), config=CONFIG, cpu_available=7200., gpu_available=0.)
        failure = T.select_update_budget(benchmark_fixture(), config=CONFIG, cpu_available=1800., gpu_available=0.)
        self.assertEqual(failure["status"], "INSUFFICIENT_BUDGET")
        selected = T.select_update_budget(benchmark_fixture(width=16), config=CONFIG,
                                          cpu_available=7200., gpu_available=0., fallback_from=failure)
        self.assertEqual(selected["status"], "COST_SELECTED")
        self.assertEqual(selected["width"], 16)

    def test_checkpoint_weights_only_reload_preserves_prediction(self):
        model = T.build_matched_models(config=CONFIG, width=16)["N2"]
        batch = self.train["N2"].batch([0, 8, 16, 24])
        with torch.no_grad():
            expected = model(self.normalizer(batch["features"]), batch["base_mass"], self.normalizer.material_scale)
        metadata = {"method": "N2", "width": 16, "seed": 261009, "parameter_count": 8994,
                    "protocol_id": CONFIG["protocol_id"], "source_commit": "fixture", "updates": 100,
                    "volume": T.VOLUME, "grid": T.GRID,
                    **{key: CONFIG[key] for key in ("real_lower", "real_upper", "imag_lower", "imag_upper")}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "best.pt"
            T._save_checkpoint(path, model, self.normalizer, metadata=metadata, step=100,
                               kind="best", validation_score=.2, book=Book())
            loaded = T.load_trained_model(path, config=CONFIG)
            with torch.no_grad():
                actual = loaded.predict_mass(batch["features"], batch["base_mass"])
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            self.assertEqual(loaded.step, 100)
            self.assertEqual(loaded.predict_chi(batch["features"], batch["base_mass"]).dtype, torch.complex128)
            payload = torch.load(path, weights_only=True)
            self.assertFalse(payload["targets_in_checkpoint"])
            self.assertTrue(payload["exposed_validation_used_for_selection"])
            self.assertEqual(payload["metadata"]["updates"], 100)


if __name__ == "__main__":
    unittest.main()
