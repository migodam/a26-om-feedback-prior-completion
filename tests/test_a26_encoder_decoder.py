"""Tiny algebra/cache tests; run only in the parent's active CPU cost ledger.

The fixtures are finite arrays, not full-wave states or new labels.  These
tests never open dataset splits, locked arrays, or GPU devices.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from a24.contracts import ForbiddenAccess, NumericalFailure, get_ledger
from a25.torch_physics import mass_pack, mass_unpack
from a26.decoder import ResidualDecoder, project_mass
from a26.encoder import FEATURE_NAMES, FrozenEncoder, load_encoders


def _tiny_cache(root, *, background_id="weak", source_order=None):
    """Portable known-background equations without a source-physics builder."""
    grid, q, s = 6, 16, 3
    n, volume = grid ** 2, 1. / grid ** 2
    cfg = {"protocol_id": "A25-TINY-ENCODER-FIXTURE-v1", "source_commit": "original-a25-fixture",
           "selected_nn_closure": "TASK_R32", "master_seed": 251008,
           "grid": grid, "side": 1., "sources": s, "receivers": q,
           "source_radius": .75, "receiver_radius": .8, "wavenumber": 2 * math.pi,
           "background_weak": [.1, .04], "background_ring": [2., .1],
           "ring_radius_inner": .2, "ring_radius_outer": .35,
           "real_lower": -.5, "real_upper": 4., "imag_lower": 0., "imag_upper": .5,
           "feasibility_tolerance": 1e-10, "pole_relative_floor": 1e-10,
           "identity_tolerance": 1e-10, "backward_residual_tolerance": 1e-10,
           "core_relative_sigma_min": 1e-10, "core_condition_cap": 1e10,
           "orthogonal_rank_rtol": 1e-10}
    coordinate = np.linspace(-.5 + .5 / grid, .5 - .5 / grid, grid)
    points = np.stack(np.meshgrid(coordinate, coordinate, indexing="ij"), -1).reshape(n, 2)
    angle = np.linspace(0, 2 * math.pi, s, endpoint=False)
    tx = .75 * np.c_[np.cos(angle), np.sin(angle)]
    angle = np.linspace(0, 2 * math.pi, q, endpoint=False)
    rx = .8 * np.c_[np.cos(angle), np.sin(angle)]
    chi_b = np.full(n, .1 + .04j, dtype=np.complex128)
    if background_id == "ring":
        radius = np.linalg.norm(points, axis=1)
        chi_b[(radius >= .2) & (radius <= .35)] = 2. + .1j
    g = .02 * np.eye(n, dtype=np.complex128)
    incident = ((1 + .01 * np.arange(n)[:, None]) *
                np.exp(1j * np.arange(n)[:, None] * np.linspace(.02, .09, s)[None])).astype(np.complex128)
    eb = incident / (1 - .02 * chi_b[:, None])
    receiver = np.zeros((q, n), dtype=np.complex128)
    receiver[:, :q] = np.diag(np.linspace(.7, 1.3, q))
    db = receiver / (1 - .02 * chi_b[None])
    yb = receiver @ (chi_b[:, None] * eb)
    ug, vg = np.zeros((n, 32), dtype=np.complex128), np.zeros((n, 32), dtype=np.complex128)
    ug[:32, :] = np.diag(.02 / (1 - .02 * chi_b[:32]))
    vg[:32, :] = np.eye(32, dtype=np.complex128)
    if source_order is not None:
        tx, incident, eb, yb = (value[np.asarray(source_order)] if index == 0 else
                               value[:, np.asarray(source_order)]
                               for index, value in enumerate((tx, incident, eb, yb)))
    descriptor = {"adapter": "a24_2d_tm_a2",
                  "geometry": {"grid": grid, "side": 1., "n_tx": s, "n_rx": q,
                               "radius_tx": .75, "radius_rx": .8, "aperture": "full"},
                  "source_coordinates_in_order": tx.tolist(),
                  "receiver_coordinates_in_order": rx.tolist(), "frequency_hz": 299_792_458.,
                  "precision": "complex128", "material_law": "scalar_chi",
                  "current_normalization": "sqrt_cell_volume", "cell_volume": volume,
                  "background": np.c_[chi_b.real, chi_b.imag].tolist()}
    geometry = {"grid": grid, "points": points.tolist(), "source_points": tx.tolist(),
                "receiver_points": rx.tolist(), "sources": s, "receivers": q,
                "volume": volume, "wavenumber": 2 * math.pi, "precision": "complex128/float64",
                "source_axis": -1, "current_normalization": "sqrt(cell_area)*physical_current"}
    provenance = {"builder": "TASK_R32", "rank": 32, "probe_seed": 251545,
                  "source_commit": cfg["source_commit"], "protocol_id": cfg["protocol_id"],
                  "background_id": background_id, "background_descriptor": descriptor,
                  "basis_identity": "A25-frozen-background-" + background_id + "-TASK_R32",
                  "independent_material_probes": 4, "independent_receiver_probes": 8,
                  "truth_or_oracle_used": False}
    b = np.einsum("qn,ns->sqn", db, eb) / math.sqrt(volume)
    re = np.concatenate((b.real, b.imag), 1).reshape(2 * q * s, n)
    im = np.concatenate((-b.imag, b.real), 1).reshape(2 * q * s, n)
    jacobian = np.concatenate((re, im), 1)
    spectrum = np.linalg.svd(jacobian, compute_uv=False)
    norm2 = float(np.sum(abs(yb) ** 2))
    arrays = {"chi_b": chi_b, "eb": eb, "yb": yb, "Db": db, "G": g, "S": receiver,
              "incident": incident, "Ug": ug, "Vg": vg,
              "L_b": np.eye(n, dtype=np.complex128) - chi_b[:, None] * g,
              "volume": np.asarray(volume), "grid": np.asarray(grid), "points": points,
              "source_points": tx, "receiver_points": rx,
              "descriptor": np.asarray(json.dumps(descriptor)),
              "geometry_descriptor": np.asarray(json.dumps(geometry)),
              "closure_provenance": np.asarray(json.dumps(provenance)), "J": jacobian,
              "step": np.asarray(.8 * norm2 / spectrum[0] ** 2),
              "data_norm2": np.asarray(norm2), "beta": np.asarray(math.sqrt(norm2 / (q * s))),
              "initializer_id": np.asarray("TINY_KNOWN_BACKGROUND_FIXTURE")}
    config_path = root / "configs/a25/nn_admitted.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    path = root / "data/background" / (background_id + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path, arrays, cfg


class EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.book = get_ledger()
        if cls.book.device != "cpu":
            raise RuntimeError("A26_TINY_ENCODER_TESTS_REQUIRE_ACTIVE_CPU_LEDGER")
        torch.set_num_threads(1)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        with self.book.span("a26_tiny_portable_cache_fixture", a26_tiny_cache_fixtures=1):
            self.path, self.arrays, self.source_config = _tiny_cache(self.root)
        self.config = {"protocol_id": "A26-FROZEN-ENCODER-v1", "source_commit": "a25-public-artifact",
                       "observable_rank": 16, "confidence_relative_cut": .05, "closure": "TASK_R32"}

    def tearDown(self):
        self.directory.cleanup()

    def _encoder(self, path=None):
        return FrozenEncoder(path or self.path, self.config, self.book, "cpu")

    def _observation(self, encoder):
        return torch.stack((encoder.reference.yb * (1.12 + .025j),
                            encoder.reference.yb * (.93 - .01j)))

    def test_original_provenance_and_cache_firewall(self):
        with self.book.span("a26_original_provenance_and_firewall_test"):
            encoder = self._encoder()
            self.assertEqual(encoder.reference.config["protocol_id"], self.source_config["protocol_id"])
            self.assertEqual(encoder.reference.closure_provenance["source_commit"],
                             self.source_config["source_commit"])
            self.assertNotEqual(encoder.reference.config["source_commit"], self.config["source_commit"])
            self.assertIn("closure_provenance", encoder.reference.descriptor)
            with self.assertRaises(ForbiddenAccess):
                encoder.encode({"data": self.arrays["yb"], "chi": self.arrays["chi_b"]}, 0., "N1")
            changed = dict(self.arrays, chi=self.arrays["chi_b"])
            np.savez(self.path, **changed)
            with self.assertRaises(ForbiddenAccess):
                self._encoder()

    def test_wrong_background_j_and_changed_guard_are_rejected(self):
        with self.book.span("a26_background_and_guard_rejection_test"):
            changed = dict(self.arrays, J=self.arrays["J"] * 1.01)
            np.savez(self.path, **changed)
            with self.assertRaisesRegex(ValueError, "NOT_KNOWN_BACKGROUND"):
                self._encoder()
            np.savez(self.path, **self.arrays)
            changed_config = dict(self.config, core_condition_cap=1e12)
            with self.assertRaisesRegex(ValueError, "TOLERANCE_CHANGED"):
                FrozenEncoder(self.path, changed_config, self.book, "cpu")

    def test_L_bases_match_and_L_methods_never_run_R(self):
        with self.book.span("a26_L_equivalence_and_runtime_fairness_test"):
            encoder = self._encoder()
            self.assertFalse(encoder.observable_ready)
            observation = self._observation(encoder)
            with patch.object(encoder.models["R"], "forward", side_effect=AssertionError("R forbidden")), \
                 patch.object(encoder.reference, "canonical_current", side_effect=AssertionError("current forbidden")):
                n3 = encoder.encode(observation, [.02, 0.], "N3")
                self.assertFalse(encoder.observable_ready)
                n1 = encoder.encode(observation, [.02, 0.], "N1")
            torch.testing.assert_close(n1["base_mass"], n3["base_mass"], rtol=0., atol=0.)
            torch.testing.assert_close(n1["initial_mass"], n3["initial_mass"], rtol=0., atol=0.)
            keep = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]
            torch.testing.assert_close(n1["features"][:, keep], n3["features"][:, keep], rtol=0., atol=0.)
            self.assertGreater(float((n1["features"][:, 2:4] - n3["features"][:, 2:4]).abs().max()), 0.)
            self.assertEqual(n1["features"].shape, (2, 12, 6, 6))
            self.assertEqual(n1["features"].dtype, torch.float32)
            for result in (n1, n3):
                self.assertTrue(torch.isnan(result["beta"]).all())
                self.assertEqual(result["diagnostics"]["beta_status"], "NO_FEEDBACK_NOT_COMPUTED")
                self.assertEqual(result["diagnostics"]["beta_group"], ["NO_FEEDBACK", "NOISE_FREE"])
                self.assertEqual(result["diagnostics"]["physical_counts"].get("feedback_K_LU_factorizations", 0), 0)
                self.assertFalse(result["diagnostics"]["canonical_current_recovered"])
            self.assertEqual(n3["diagnostics"]["physical_counts"].get("a26_observation_SVDs", 0), 0)
            self.assertEqual(n1["diagnostics"]["physical_counts"]["a26_observation_SVDs"], 1)

    def test_N2_matched_gradient_OM_and_beta(self):
        with self.book.span("a26_N2_matched_gradient_OM_beta_test"):
            encoder = self._encoder()
            observation = self._observation(encoder)
            with patch.object(encoder.reference, "canonical_current", side_effect=AssertionError("current forbidden")):
                n1 = encoder.encode(observation, [.02, 0.], "N1")
                n2 = encoder.encode(observation, [.02, 0.], "N2")
            torch.testing.assert_close(n1["features"][:, 2:4], n2["features"][:, 2:4], rtol=0., atol=0.)
            mass = n2["initial_mass"].clone().requires_grad_()
            state = encoder.models["R"](mass, recover_current=False)
            objective = .5 * (state.data - observation).abs().square().sum() / encoder.reference.yb_norm2
            automatic = torch.autograd.grad(objective, mass)[0]
            n = encoder.reference.n
            encoded_gradient = n2["features"][:, 4:6].reshape(2, 2 * n).to(torch.float64)
            torch.testing.assert_close(encoded_gradient, automatic, rtol=2e-6, atol=1e-9)
            expected_base = encoder.reference.project(n2["initial_mass"] - encoder.reference.step * automatic)
            torch.testing.assert_close(n2["base_mass"], expected_base, rtol=1e-11, atol=1e-12)
            lst = encoder.models["L"](n2["initial_mass"], recover_current=False)
            expected_beta = torch.linalg.vector_norm(state.data[0].detach() - lst.data[0]) / (.02 * math.sqrt(48))
            torch.testing.assert_close(n2["beta"][0], expected_beta, rtol=1e-11, atol=1e-12)
            self.assertTrue(torch.isnan(n2["beta"][1]))
            self.assertEqual(n2["diagnostics"]["beta_group"], ["NOISY", "NOISE_FREE"])
            self.assertGreater(n2["diagnostics"]["physical_counts"]["feedback_K_LU_factorizations"], 0)

    def test_confidence_full_nullspace_noise_groups_and_export(self):
        with self.book.span("a26_soft_confidence_and_export_test"):
            encoder = self._encoder()
            c, weak = encoder.confidence, encoder.weak_operator
            identity = torch.eye(72, dtype=torch.float64)
            torch.testing.assert_close(c + weak, identity, rtol=0., atol=1e-14)
            torch.testing.assert_close(c, c.T, rtol=0., atol=1e-14)
            # Cells 16..35 are in the fixture's exact background-data nullspace.
            self.assertLess(float(c[:, 16:36].abs().max()), 1e-12)
            self.assertAlmostEqual(float(weak[20, 20]), 1., places=12)
            eigen = torch.linalg.eigvalsh(c)
            self.assertGreaterEqual(float(eigen.min()), -1e-12)
            self.assertLessEqual(float(eigen.max()), 1. + 1e-12)
            observation = self._observation(encoder)
            noiseless = encoder.encode(observation, 0., "N3")
            noisy = encoder.encode(observation, .1, "N3")
            torch.testing.assert_close(noiseless["features"], noisy["features"], rtol=0., atol=0.)
            self.assertEqual(noiseless["diagnostics"]["beta_group"], ["NOISE_FREE", "NOISE_FREE"])
            paths = encoder.save_diagnostics(self.root / "diagnostics")
            with np.load(paths["arrays"], allow_pickle=False) as saved:
                self.assertNotIn("Db_spectrum", saved.files)  # N3 leaves OM preparation pending.
                np.testing.assert_allclose(saved["confidence"] + saved["weak_operator"], np.eye(72), atol=1e-14)
                self.assertEqual(tuple(saved["feature_names"].tolist()), FEATURE_NAMES)
            metrics = json.loads(Path(paths["metrics"]).read_text(encoding="utf-8"))
            self.assertFalse(metrics["observable_prepared"])
            self.assertIn("NOT_STATISTICAL_RECOVERABILITY", metrics["confidence_interpretation"])

    def test_relative_residual_and_dark_denominator_fail(self):
        with self.book.span("a26_zero_rhs_residual_and_dark_denominator_test"):
            encoder = self._encoder()
            mass = mass_pack(encoder.reference.chi_b, encoder.volume)[None]
            state = encoder.models["L"](mass, recover_current=False)
            state.h.zero_()
            state.w.zero_()
            state.field = encoder.reference.eb[None]
            torch.testing.assert_close(encoder._forward_residual(state), torch.zeros(1, dtype=torch.float64))
            state.w[0, 0, 0] = 1.
            with self.assertRaisesRegex(NumericalFailure, "ZERO_RELATIVE_RHS"):
                encoder._forward_residual(state)
            field = encoder.reference.eb[None].clone()
            field[:, -1] = 0.
            with self.assertRaisesRegex(NumericalFailure, "UNSAFE_SHARED_MATERIAL_RATIO"):
                encoder.reference.material_ratio(torch.ones_like(field), field)

    def test_source_permutation_and_loader_are_background_only(self):
        with self.book.span("a26_source_symmetry_and_background_loader_test"):
            encoder = self._encoder()
            permutation = [2, 0, 1]
            permuted_path, _, _ = _tiny_cache(self.root / "permuted", source_order=permutation)
            permuted = self._encoder(permuted_path)
            observation = self._observation(encoder)
            for method in ("N1", "N2", "N3"):
                first = encoder.encode(observation, .02, method)
                second = permuted.encode(observation[:, :, permutation], .02, method)
                torch.testing.assert_close(first["features"], second["features"], rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(first["base_mass"], second["base_mass"], rtol=1e-11, atol=1e-12)
            _tiny_cache(self.root, background_id="ring")
            encoders = load_encoders(self.root / "data", self.config, self.book, "cpu")
            self.assertEqual(set(encoders), {"weak", "ring"})
            with self.assertRaises(ForbiddenAccess):
                load_encoders(self.root / "data/locked", self.config, self.book, "cpu")

    def test_unsafe_OM_rank_rejected_lazily(self):
        with self.book.span("a26_no_OM_ridge_or_fallback_test"):
            encoder = self._encoder()
            # Modify only this test's finite-array reference; N3 does not use Db SVD.
            encoder.reference.Db[-1] *= 1e-14
            with self.assertRaisesRegex(NumericalFailure, "OBSERVABLE_RANK_16"):
                encoder.prepare_observable()
            self.assertFalse(encoder.observable_ready)


class DecoderTests(unittest.TestCase):
    def test_topology_zero_head_physical_scale_box_and_backward(self):
        book = get_ledger()
        if book.device != "cpu":
            raise RuntimeError("A26_TINY_DECODER_TESTS_REQUIRE_ACTIVE_CPU_LEDGER")
        with book.span("a26_matched_decoder_physical_coordinates_test"):
            decoder = ResidualDecoder()
            convolutions = [layer for layer in decoder.body if isinstance(layer, nn.Conv2d)]
            self.assertEqual(len(convolutions), 5)
            self.assertEqual([layer.dilation[0] for layer in convolutions], [1, 2, 4, 2, 1])
            self.assertEqual(decoder.parameter_count, 31810)
            self.assertEqual(sum(isinstance(layer, nn.GELU) for layer in decoder.body), 4)
            features = torch.zeros((2, 12, 4, 4), dtype=torch.float32)
            volume, material_scale = 1. / 16, .2
            base_chi = torch.full((2, 16), .1 + .04j, dtype=torch.complex128)
            base = mass_pack(base_chi, volume)
            torch.testing.assert_close(decoder(features, base, material_scale, volume), base, rtol=0., atol=0.)
            with torch.no_grad():
                decoder.body[-1].bias.copy_(torch.tensor([.5, .25]))
            expected = base_chi + .1 + .05j
            torch.testing.assert_close(decoder.predict_chi(features, base, material_scale, volume), expected,
                                       rtol=1e-7, atol=1e-8)
            mass = decoder(features, base, material_scale, volume)
            loss = mass.square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(decoder.body[-1].bias.grad).all())
            self.assertGreater(float(decoder.body[-1].bias.grad.abs().sum()), 0.)
            outside = mass_pack(torch.full((2, 16), 8. - 2j, dtype=torch.complex128), volume)
            boxed = mass_unpack(project_mass(outside, volume), volume)
            torch.testing.assert_close(boxed, torch.full_like(boxed, 4. + 0.j))
            self.assertEqual(ResidualDecoder(width=16).width, 16)
            with self.assertRaises(ValueError):
                decoder(features, base, torch.ones(2), volume)
