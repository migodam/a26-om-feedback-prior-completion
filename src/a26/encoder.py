"""Lawful, fixed A26 observation-to-material features.

Only an already exported known-background cache and raw online measurement
enter this module.  The cached real MASS Jacobian is checked against Db and eb
before it is used for numerical confidence.  Its soft spectral operator is
not a calibrated statistical recoverability certificate.  No teacher/current
truth is accepted, no canonical current is recovered, and a failed R solve has
no fallback to L.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import torch

from a20.costs import write_json
from a24.contracts import ForbiddenAccess, NumericalFailure
from a25.torch_physics import (TorchReference, TorchRelativeModel, mass_pack,
                               mass_unpack, require_device)


FEATURE_NAMES = (
    "base_delta_real", "base_delta_imag", "O_or_BP_real", "O_or_BP_imag",
    "gradient_mass_real", "gradient_mass_imag", "source_mean_field_power",
    "source_mean_w_power", "background_real", "background_imag",
    "confidence_diag_real", "confidence_diag_imag",
)
_REFERENCE_ARRAYS = {"chi_b", "eb", "yb", "Db", "G", "S", "incident", "Ug", "Vg",
                     "L_b", "D_indices", "D_blocks", "source_points", "receiver_points",
                     "closure_provenance"}
_CACHE_KEYS = _REFERENCE_ARRAYS | {
    "descriptor", "geometry_descriptor", "volume", "grid", "points", "J", "step",
    "fixed_step", "data_norm2", "beta", "initializer_id",
}
_CACHE_REQUIRED = {"chi_b", "eb", "yb", "Db", "G", "S", "incident", "Ug", "Vg",
                   "descriptor", "geometry_descriptor", "closure_provenance",
                   "volume", "grid", "points", "source_points", "receiver_points", "J"}
_FROZEN_RUNTIME_KEYS = ("real_lower", "real_upper", "imag_lower", "imag_upper",
                        "feasibility_tolerance", "pole_relative_floor", "identity_tolerance",
                        "backward_residual_tolerance", "core_relative_sigma_min",
                        "core_condition_cap", "orthogonal_rank_rtol")


def _lawful_path(path):
    path = Path(path).expanduser().resolve()
    if ({part.lower() for part in path.parts}
            & {"locked", "locked_id", "locked_ood", "offline", "offline_eval", "oracle",
               "teacher", "labels"}) or any(word in path.name.lower()
                                            for word in ("oracle", "teacher", "full_gn")):
        raise ForbiddenAccess("A26_BACKGROUND_PATH_FORBIDDEN")
    return path


def _json_object(value, name):
    try:
        result = json.loads(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("A26_INVALID_" + name.upper() + "_JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("A26_" + name.upper() + "_OBJECT_REQUIRED")
    return result


def _diagnostic_values(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _diagnostic_values(item) for key, item in value.items()}
    return value


def _source_config_path(background_path, config):
    explicit = config.get("a25_source_config_path")
    path = (Path(explicit) if explicit else
            background_path.parents[2] / "configs/a25/nn_admitted.json")
    path = _lawful_path(path)
    if not path.is_file():
        raise ValueError("ORIGINAL_A25_CONFIG_REQUIRED:" + str(path))
    return path


def _validate_descriptors(arrays, descriptor, geometry, source_config, background_id):
    """Validate saved geometry and chi_b without constructing source physics."""
    grid, volume = int(arrays["grid"]), float(arrays["volume"])
    n = grid * grid
    side = float(source_config["side"])
    expected_geometry = {"grid": grid, "side": side,
                         "n_tx": int(source_config["sources"]),
                         "n_rx": int(source_config["receivers"]),
                         "radius_tx": float(source_config["source_radius"]),
                         "radius_rx": float(source_config["receiver_radius"]),
                         "aperture": "full"}
    if (grid != int(source_config["grid"]) or descriptor.get("geometry") != expected_geometry
            or descriptor.get("adapter") != "a24_2d_tm_a2"
            or descriptor.get("precision") != "complex128"
            or descriptor.get("material_law") != "scalar_chi"
            or descriptor.get("current_normalization") != "sqrt_cell_volume"
            or descriptor.get("cell_volume") != volume
            or not math.isclose(volume, (side / grid) ** 2, rel_tol=1e-14, abs_tol=0.)):
        raise ValueError("A26_ORIGINAL_BACKGROUND_GEOMETRY_MISMATCH")
    expected_frequency = float(source_config["wavenumber"]) * 299_792_458. / (2 * math.pi)
    if not math.isclose(float(descriptor.get("frequency_hz", float("nan"))),
                        expected_frequency, rel_tol=1e-14, abs_tol=0.):
        raise ValueError("A26_ORIGINAL_BACKGROUND_FREQUENCY_MISMATCH")
    for key, expected in (("grid", grid), ("volume", volume),
                          ("sources", source_config["sources"]),
                          ("receivers", source_config["receivers"]),
                          ("wavenumber", source_config["wavenumber"]),
                          ("precision", "complex128/float64"), ("source_axis", -1),
                          ("current_normalization", "sqrt(cell_area)*physical_current")):
        if geometry.get(key) != expected:
            raise ValueError("A26_PORTABLE_GEOMETRY_DESCRIPTOR_MISMATCH:" + key)
    for key, described in (("points", "points"),
                           ("source_points", "source_coordinates_in_order"),
                           ("receiver_points", "receiver_coordinates_in_order")):
        value = arrays[key]
        if value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError("A26_FINITE_FLOAT64_COORDINATES_REQUIRED:" + key)
        if not np.array_equal(value, np.asarray(geometry.get(key))):
            raise ValueError("A26_PORTABLE_COORDINATE_DESCRIPTOR_MISMATCH:" + key)
        if key != "points" and not np.array_equal(value, np.asarray(descriptor.get(described))):
            raise ValueError("A26_CACHE_APERTURE_ORDER_MISMATCH:" + key)
    if arrays["points"].shape != (n, 2):
        raise ValueError("A26_GRID_COORDINATE_SHAPE_MISMATCH")
    chi_b = arrays["chi_b"]
    if chi_b.dtype != np.complex128 or chi_b.shape != (n,):
        raise ValueError("A26_COMPLEX128_BACKGROUND_MATERIAL_REQUIRED")
    if not np.array_equal(np.c_[chi_b.real, chi_b.imag], np.asarray(descriptor.get("background"))):
        raise ValueError("A26_KNOWN_BACKGROUND_MATERIAL_DESCRIPTOR_MISMATCH")
    # The only material construction allowed here is the original known chi_b.
    expected_background = np.full(n, complex(*source_config["background_weak"]), np.complex128)
    if background_id == "ring":
        radius = np.linalg.norm(arrays["points"], axis=1)
        ring = ((radius >= source_config["ring_radius_inner"])
                & (radius <= source_config["ring_radius_outer"]))
        expected_background[ring] = complex(*source_config["background_ring"])
    if not np.array_equal(chi_b, expected_background):
        raise ValueError("A26_ORIGINAL_KNOWN_BACKGROUND_MISMATCH")


class FrozenEncoder:
    """One background and one fixed L/R physical update; no online label access.

    All numerical preparation and encoding is charged to ``book``.  To use a
    copied cache outside its A25 layout, pass ``a25_source_config_path`` in the
    A26 config.  This file must be the original A25 config: run protocol and
    source-commit fields never replace cache-construction provenance.
    """
    feature_names = FEATURE_NAMES

    def __init__(self, background_path, config, book, device="cpu"):
        if not isinstance(config, Mapping):
            raise ValueError("A26_CONFIG_OBJECT_REQUIRED")
        self.config = deepcopy(dict(config))
        self.background_path = _lawful_path(background_path)
        self.device = require_device(device)
        self.background_id = self.background_path.stem
        if self.background_id not in {"weak", "ring"}:
            raise ValueError("A26_EXPLICIT_KNOWN_BACKGROUND_ID_REQUIRED")
        if self.config.get("closure", "TASK_R32") != "TASK_R32":
            raise ValueError("A26_TASK_R32_REQUIRED_NO_FALLBACK")
        if int(self.config.get("observable_rank", 16)) != 16:
            raise ValueError("A26_FROZEN_OBSERVABLE_RANK_16_REQUIRED")
        if float(self.config.get("confidence_relative_cut", .05)) != .05:
            raise ValueError("A26_FROZEN_CONFIDENCE_RELATIVE_CUT_REQUIRED")
        with book.span("a26_original_background_cache_load", a26_background_cache_loads=1):
            self.source_config_path = _source_config_path(self.background_path, self.config)
            self.source_config = _json_object(self.source_config_path.read_text(encoding="utf-8"),
                                               "source_config")
            if (not str(self.source_config.get("protocol_id", "")).startswith("A25-")
                    or self.source_config.get("selected_nn_closure") != "TASK_R32"):
                raise ValueError("A26_ORIGINAL_TASK_R32_CONFIG_REQUIRED")
            for key in _FROZEN_RUNTIME_KEYS:
                if key in self.config and self.config[key] != self.source_config.get(key):
                    raise ValueError("A26_ORIGINAL_RUNTIME_TOLERANCE_CHANGED:" + key)
            # Inspect archive member names before any numerical payload load.
            with np.load(self.background_path, allow_pickle=False) as archive:
                names = set(archive.files)
                if names - _CACHE_KEYS:
                    raise ForbiddenAccess("A26_UNREGISTERED_BACKGROUND_CACHE_KEYS:" +
                                          ",".join(sorted(names - _CACHE_KEYS)))
                if _CACHE_REQUIRED - names or not ({"step", "fixed_step"} & names):
                    raise ValueError("A26_INCOMPLETE_KNOWN_BACKGROUND_CACHE")
                descriptor = _json_object(archive["descriptor"], "descriptor")
                geometry = _json_object(archive["geometry_descriptor"], "geometry_descriptor")
                provenance = _json_object(archive["closure_provenance"], "closure_provenance")
                if provenance.get("background_id") != self.background_id:
                    raise ValueError("A26_CACHE_BACKGROUND_ID_MISMATCH")
                arrays = {name: archive[name].copy() for name in archive.files}
            arrays["closure_provenance"] = provenance
            _validate_descriptors(arrays, descriptor, geometry, self.source_config, self.background_id)
            grid, volume = int(arrays["grid"]), float(arrays["volume"])
            fixed_step = float(arrays.get("fixed_step", arrays.get("step")))
            if "step" in arrays and "fixed_step" in arrays and float(arrays["step"]) != fixed_step:
                raise ValueError("A26_CONFLICTING_FROZEN_BACKGROUND_STEPS")
            jacobian = arrays["J"]
            if (jacobian.dtype != np.float64 or jacobian.ndim != 2
                    or not np.isfinite(jacobian).all()):
                raise ValueError("A26_CACHED_REAL_MASS_BACKGROUND_J_REQUIRED")
            for name in _REFERENCE_ARRAYS - {"closure_provenance", "D_indices",
                                              "source_points", "receiver_points"}:
                if name in arrays and arrays[name].dtype != np.complex128:
                    raise ValueError("A26_COMPLEX128_BACKGROUND_CACHE_REQUIRED:" + name)
        # Deliberately pass ORIGINAL config, not {**original, **A26_run_config}.
        self.reference = TorchReference({key: arrays[key] for key in _REFERENCE_ARRAYS if key in arrays},
            volume=volume, grid=grid, descriptor=descriptor, book=book,
            config=self.source_config, device=self.device, fixed_step=fixed_step,
            require_provenance=True)
        if (self.reference.closure_provenance["builder"] != "TASK_R32"
                or self.reference.rank != 32 or self.reference.D_indices.numel()):
            raise ValueError("A26_ORIGINAL_NONLOCAL_TASK_R32_REQUIRED")
        self.models = {kind: TorchRelativeModel(self.reference, kind) for kind in ("L", "R")}
        self.grid, self.volume = grid, volume
        self.geometry_descriptor = deepcopy(geometry)
        self.observable_ready = False
        self._prepare_operators(jacobian, arrays)

    @property
    def book(self):
        return self.reference.book

    def bind_book(self, book):
        self.reference.bind_book(book)
        return self

    @torch.no_grad()
    def _prepare_operators(self, saved_jacobian, arrays):
        ref = self.reference
        with self.book.span("a26_known_background_equation_validation", a26_background_equation_audits=1):
            # cb is the fixed known-background current, not an online truth
            # current or recovery from the estimated material state.
            field_residual = ref.eb - (ref.incident + ref.green(ref.cb))
            data_residual = ref.yb - ref.observe(ref.cb)
            db_residual = ref.Db @ ref.Lb - ref.S
            records = (("field", field_residual, ref.eb), ("data", data_residual, ref.yb),
                       ("Db", db_residual, ref.S))
            self.background_equation_errors = {}
            for name, residual, expected in records:
                denominator = torch.linalg.vector_norm(expected)
                if float(denominator) <= 0 or not torch.isfinite(denominator):
                    raise NumericalFailure("A26_INVALID_BACKGROUND_EQUATION_SCALE:" + name)
                error = float(torch.linalg.vector_norm(residual) / denominator)
                self.background_equation_errors[name] = error
                if not math.isfinite(error) or error > float(ref.config.get("identity_tolerance", 1e-10)):
                    raise ValueError("A26_KNOWN_BACKGROUND_EQUATION_MISMATCH:" + name)
        with self.book.span("a26_validate_background_real_mass_jacobian",
                            a26_background_material_directions=2 * ref.n):
            b = torch.einsum("qn,ns->sqn", ref.Db, ref.eb) / ref.sqrt_volume
            re = torch.cat((b.real, b.imag), dim=1).reshape(2 * ref.q * ref.s, ref.n)
            im = torch.cat((-b.imag, b.real), dim=1).reshape(2 * ref.q * ref.s, ref.n)
            derived = torch.cat((re, im), dim=1)
            saved = torch.as_tensor(saved_jacobian, dtype=torch.float64, device=self.device)
            if saved.shape != derived.shape:
                raise ValueError("A26_CACHED_BACKGROUND_J_SHAPE_MISMATCH")
            denominator = torch.linalg.vector_norm(derived)
            if float(denominator) <= 0 or not torch.isfinite(denominator):
                raise NumericalFailure("A26_ZERO_BACKGROUND_JACOBIAN")
            self.background_j_relative_error = float(torch.linalg.vector_norm(saved - derived) / denominator)
            if (not math.isfinite(self.background_j_relative_error)
                    or self.background_j_relative_error > float(ref.config.get("identity_tolerance", 1e-10))):
                raise ValueError("A26_CACHE_J_IS_NOT_KNOWN_BACKGROUND_REAL_MASS_J")
            norm2 = float(ref.yb_norm2)
            if "data_norm2" in arrays and not math.isclose(float(arrays["data_norm2"]), norm2,
                                                         rel_tol=1e-12, abs_tol=0.):
                raise ValueError("A26_FROZEN_BACKGROUND_DATA_NORM_MISMATCH")
            self.background_rms = math.sqrt(norm2 / (ref.q * ref.s))
            if "beta" in arrays and not math.isclose(float(arrays["beta"]), self.background_rms,
                                                    rel_tol=1e-12, abs_tol=0.):
                raise ValueError("A26_FROZEN_BACKGROUND_DATA_RMS_MISMATCH")
        with self.book.span("a26_real_mass_background_confidence_SVD", a26_confidence_SVDs=1):
            _, singular, vh = torch.linalg.svd(saved, full_matrices=False)
            if float(singular[0]) <= 0 or not torch.isfinite(singular).all():
                raise NumericalFailure("A26_ZERO_BACKGROUND_CONFIDENCE_SCALE")
            relative = singular / singular[0]
            weights = relative.square() / (relative.square() + .05 ** 2)
            self.background_spectrum = singular.detach()
            self.background_right = vh.detach()
            self.confidence_weights = weights.detach()
            self.confidence = ((vh.T * weights) @ vh).detach()
            self.weak_operator = (torch.eye(2 * ref.n, dtype=torch.float64, device=self.device)
                                  - self.confidence).detach()
            self.confidence_diagonal = self.confidence.diagonal().detach()
            expected_step = .8 * float(ref.yb_norm2) / float(singular[0].square())
            if not math.isclose(float(ref.step), expected_step, rel_tol=1e-10, abs_tol=0.):
                raise ValueError("A26_SAVED_BACKGROUND_STEP_RULE_MISMATCH")
        self.operator_metrics = {
            "background_id": self.background_id,
            "confidence_interpretation": "NUMERICAL_SOFT_OPERATOR_NOT_STATISTICAL_RECOVERABILITY",
            "jacobian_coordinates": "REAL_MASS:[Re_n,Im_n]",
            "background_J_relative_error": self.background_j_relative_error,
            "background_equation_relative_errors": self.background_equation_errors,
            "observable_rank": 16, "observable_prepared": False, "relative_confidence_cut": .05,
            "noisy_real_datum_variance": "sigma^2/2",
            "positive_sigma_whitening": "sqrt(2)/sigma; relative spectral weights unchanged",
            "noise_free_operator_scale": "background_data_RMS",
            "background_data_RMS": self.background_rms,
            "background_J_shape": list(saved.shape),
            "confidence_trace": float(self.confidence_diagonal.sum()),
            "confidence_diagonal_min": float(self.confidence_diagonal.min()),
            "confidence_diagonal_max": float(self.confidence_diagonal.max()),
            "background_spectral_max": float(self.background_spectrum[0]),
            "frozen_background_step": float(ref.step),
            "source_config_path": str(self.source_config_path),
            "source_protocol_id": self.source_config["protocol_id"],
            "source_commit": self.source_config["source_commit"],
            "run_protocol_id": self.config.get("protocol_id"),
            "run_source_commit": self.config.get("source_commit"),
            "closure_provenance": deepcopy(ref.closure_provenance),
            "reference_descriptor": deepcopy(ref.descriptor),
            "canonical_current_recovered": False,
        }

    @torch.no_grad()
    def prepare_observable(self):
        """Lazily prepare Db rank16 only for an O-feature method (N1 or N2)."""
        if self.observable_ready:
            return
        ref = self.reference
        with self.book.span("a26_Db_rank16_SVD", a26_observation_SVDs=1):
            u, singular, vh = torch.linalg.svd(ref.Db, full_matrices=False)
            if (len(singular) < 16 or float(singular[0]) <= 0
                    or float(singular[15] / singular[0]) < float(ref.config.get("orthogonal_rank_rtol", 1e-10))):
                raise NumericalFailure("UNSAFE_A26_OBSERVABLE_RANK_16")
            self.observable_left = u[:, :16].detach()
            self.observable_right = vh[:16].mH.detach()
            self.observable_singular = singular[:16].detach()
            self.db_spectrum = singular.detach()
        self.observable_ready = True
        self.operator_metrics.update(observable_prepared=True,
            Db_spectral_max=float(singular[0]), Db_retained_relative_min=float(singular[15] / singular[0]))

    def save_diagnostics(self, destination):
        """Save spectra, full C and I-C, and provenance; contains no labels."""
        destination = Path(destination)
        path = (destination if destination.suffix == ".npz" else
                destination / (self.background_id + "_encoder_diagnostics.npz"))
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.book.span("a26_fixed_encoder_diagnostics_export"):
            arrays = {"J_background_spectrum": self.background_spectrum,
                      "J_background_spectrum_normalized": self.background_spectrum / self.background_rms,
                      "J_background_right": self.background_right,
                      "confidence_weights": self.confidence_weights, "confidence": self.confidence,
                      "weak_operator": self.weak_operator, "confidence_diagonal": self.confidence_diagonal}
            if self.observable_ready:
                arrays["Db_spectrum"] = self.db_spectrum
            payload = {name: value.detach().cpu().numpy() for name, value in arrays.items()}
            payload.update(background_id=np.asarray(self.background_id),
                           feature_names=np.asarray(FEATURE_NAMES),
                           metrics_json=np.asarray(json.dumps(self.operator_metrics, sort_keys=True)))
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **payload)
            temporary.replace(path)
            metrics_path = path.with_suffix(".metrics.json")
            write_json(metrics_path, self.operator_metrics)
        return {"arrays": str(path), "metrics": str(metrics_path)}

    def _online_inputs(self, observation, sigma):
        if isinstance(observation, Mapping) or isinstance(sigma, Mapping):
            raise ForbiddenAccess("A26_ENCODER_RAW_OBSERVATION_ONLY_NO_TARGET_DICTIONARIES")
        if not isinstance(observation, (np.ndarray, torch.Tensor)):
            raise ValueError("A26_RAW_COMPLEX128_OBSERVATION_ARRAY_REQUIRED")
        if ((isinstance(observation, torch.Tensor) and observation.dtype != torch.complex128)
                or (isinstance(observation, np.ndarray) and observation.dtype != np.complex128)):
            raise ValueError("A26_COMPLEX128_OBSERVATION_REQUIRED")
        if ((isinstance(sigma, torch.Tensor) and sigma.is_complex())
                or (not isinstance(sigma, torch.Tensor) and np.iscomplexobj(sigma))):
            raise ValueError("A26_REAL_NOISE_SIGMA_REQUIRED")
        with self.book.span("a26_online_observation_transfer"):
            observation = torch.as_tensor(observation, dtype=torch.complex128, device=self.device)
            if observation.ndim == 2:
                observation = observation.unsqueeze(0)
            if (observation.ndim != 3 or observation.shape[1:] != (self.reference.q, self.reference.s)
                    or len(observation) == 0 or not torch.isfinite(observation).all()):
                raise ValueError("A26_FINITE_BATCHED_RECEIVER_SOURCE_OBSERVATION_REQUIRED")
            sigma = torch.as_tensor(sigma, dtype=torch.float64, device=self.device)
            if sigma.ndim == 0:
                sigma = sigma.expand(len(observation))
            if (sigma.shape != (len(observation),) or not torch.isfinite(sigma).all()
                    or bool((sigma < 0).any())):
                raise ValueError("A26_MATCHED_NONNEGATIVE_FINITE_NOISE_SIGMA_REQUIRED")
        return observation, sigma

    def _forward_residual(self, state):
        """||A w - h eb||/||h eb||, with explicit zero-RHS handling."""
        ref = self.reference
        with self.book.span("a26_relative_forward_residual_guard", a26_forward_residual_audits=len(state.mass)):
            rhs = state.h[..., None] * ref.eb
            # field-eb is the already computed matching feedback action.
            residual = state.w - state.h[..., None] * (state.field - ref.eb) - rhs
            norm = torch.linalg.vector_norm(residual, dim=(-2, -1))
            rhs_norm = torch.linalg.vector_norm(rhs, dim=(-2, -1))
            ratio = torch.zeros_like(norm)
            positive = rhs_norm > 0
            ratio[positive] = norm[positive] / rhs_norm[positive]
            if bool(((~positive) & (norm != 0)).any()):
                raise NumericalFailure("A26_NONZERO_RESIDUAL_AT_ZERO_RELATIVE_RHS")
            if (not torch.isfinite(ratio).all()
                    or bool((ratio > float(ref.config.get("backward_residual_tolerance", 1e-10))).any())):
                raise NumericalFailure("A26_RELATIVE_FORWARD_RESIDUAL_EXCEEDS_FROZEN_TOLERANCE")
        return ratio.detach()

    @torch.no_grad()
    def encode(self, observation, sigma, method):
        """Return features/base/initializer/beta/diagnostics for raw measurement.

        Single measurements retain a batch dimension.  N1 and N3 share the
        exact L physical base.  N1 and N2 share the exact rank-16 O feature.
        N2 computes beta from its matched L/R data.  N1/N3 never run R for a
        diagnostic: their beta is NaN with NO_FEEDBACK_NOT_COMPUTED status.
        Sigma==0 always has an explicit NOISE_FREE group.
        """
        method = str(method).upper()
        if method not in {"N1", "N2", "N3"}:
            raise ValueError("A26_ENCODER_METHOD_MUST_BE_N1_N2_N3")
        observation, sigma = self._online_inputs(observation, sigma)
        ref, batch = self.reference, len(observation)
        old_counts = dict(self.book.counts)
        with self.book.span("a26_frozen_encoder_" + method, a26_encoded_images=batch):
            with self.book.span("a26_common_boxed_background_adjoint", a26_common_initial_images=batch):
                load = (ref.yb - observation) / ref.yb_norm2
                adjoint = ref.apply_db(load, adjoint=True)
                background_gradient = mass_pack((ref.eb.conj() * adjoint).sum(-1), 1.) / ref.sqrt_volume
                initial = ref.project(mass_pack(ref.chi_b, ref.volume)[None] - ref.step * background_gradient)
            kind = "R" if method == "N2" else "L"
            required = ("L", "R") if method == "N2" else ("L",)
            states = {key: self.models[key](initial, recover_current=False) for key in required}
            residuals = {kind: self._forward_residual(state) for kind, state in states.items()}
            state = states[kind]
            gradient = self.models[kind].data_gradient(initial, observation, state)
            base = ref.project(initial - ref.step * gradient)
            if method == "N3":
                # Ordinary BP = common initializer increment, before box.
                image = mass_unpack(-ref.step * background_gradient, ref.volume)
            else:
                self.prepare_observable()
                with self.book.span("a26_rank16_observation_feature", a26_observable_source_rhs=batch * ref.s):
                    w_observable = self.observable_right @ ((self.observable_left.mH @ (observation - ref.yb))
                                                           / self.observable_singular[:, None])
                    image = mass_unpack(ref.material_ratio(w_observable, ref.eb[None].expand(batch, -1, -1)),
                                        ref.volume)
            delta = mass_unpack(base, ref.volume) - ref.chi_b
            background = ref.chi_b[None].expand(batch, -1)
            confidence = self.confidence_diagonal[None].expand(batch, -1)
            n = ref.n
            features = torch.stack((delta.real, delta.imag, image.real, image.imag,
                                    gradient[:, :n], gradient[:, n:],
                                    state.field.abs().square().mean(-1),
                                    state.w.abs().square().mean(-1),
                                    background.real, background.imag,
                                    confidence[:, :n], confidence[:, n:]), dim=1)
            features = features.reshape(batch, len(FEATURE_NAMES), ref.grid, ref.grid).to(torch.float32)
            if not torch.isfinite(features).all() or not torch.isfinite(base).all():
                raise NumericalFailure("A26_NONFINITE_FROZEN_ENCODER_OUTPUT")
            beta = torch.full((batch,), float("nan"), dtype=torch.float64, device=self.device)
            noisy = sigma > 0
            if method == "N2":
                gap = torch.linalg.vector_norm(states["R"].data - states["L"].data, dim=(-2, -1))
                beta[noisy] = gap[noisy] / (sigma[noisy] * math.sqrt(ref.q * ref.s))
            diagnostics = {"method": method, "physical_model": kind,
                           "feature_names": list(FEATURE_NAMES),
                           "beta_group": [("NOISY" if method == "N2" else "NO_FEEDBACK")
                                          if value else "NOISE_FREE" for value in noisy.cpu().tolist()],
                           "beta_status": "COMPUTED_FOR_N2" if method == "N2" else "NO_FEEDBACK_NOT_COMPUTED",
                           "beta_noise_free": (~noisy).cpu().tolist(),
                           "relative_forward_residual": _diagnostic_values(residuals),
                           "core_margins": _diagnostic_values(states["R"].margins) if method == "N2" else {},
                           "canonical_current_recovered": False,
                           "background_id": self.background_id,
                           "confidence_interpretation": self.operator_metrics["confidence_interpretation"]}
            diagnostics["physical_counts"] = {
                key: int(value - old_counts.get(key, 0)) for key, value in self.book.counts.items()
                if value != old_counts.get(key, 0)}
        return {"features": features.detach(), "base_mass": base.detach(),
                "initial_mass": initial.detach(), "beta": beta.detach(), "diagnostics": diagnostics}


def load_encoders(data_root, config, book, device="cpu"):
    """Load only the existing weak/ring background archives, never split data."""
    root = _lawful_path(data_root)
    if root.name == "background":
        background = root
    elif (root / "background").is_dir():
        background = root / "background"
    elif (root / "data/background").is_dir():
        background = root / "data/background"
    else:
        raise ValueError("A26_EXISTING_BACKGROUND_DIRECTORY_REQUIRED")
    return {name: FrozenEncoder(background / (name + ".npz"), config, book, device)
            for name in ("weak", "ring")}
