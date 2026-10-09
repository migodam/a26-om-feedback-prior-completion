"""A26 cache-only training; the caller owns physics, budgets and science gates.

Callable driver API (all numerical calls belong inside ``charged_job``):

* ``write_feature_cache(path, arrays, method=..., role=..., config=...)`` writes
  an immutable, target-free NPZ; ``load_feature_cache(path, role=..., method=...)``
  checks metadata before loading its feature payload.
* ``CachedDataset.from_npz(path, data_root, role=..., background_materials=...)``
  obtains labels only through ``data.load_split``. ``background_materials`` is
  the exact known ``reference.chi_b`` mapping, never a truth-state descriptor.
* ``fit_normalizer(train_datasets, config=..., book=...)`` pools the three TRAIN
  feature tensors and fits one scalar RMS of TRAIN delta material.
* ``benchmark_training(train_datasets, validation_datasets, normalizer=...,
  config=..., book=..., width=32, device='cpu')`` performs 1 warm + 10 actual
  TRAIN updates per arm and times one complete 128-row validation per arm.
* ``select_update_budget(benchmark, config=..., cpu_available=...,
  gpu_available=..., fallback_from=None)`` selects U using cost alone. A width
  16 benchmark requires the failed width 32 selection in ``fallback_from``.
* ``train_one_model(method, train_data, validation_data, selection=...,
  normalizer=..., config=..., out=..., book=..., device='cpu', seed=None)``
  resets model, optimizer and sampler and writes best/final/middle checkpoints.
* ``load_trained_model(path, device='cpu', config=None)`` uses weights-only
  deserialization and returns ``LoadedCheckpoint(model, normalizer, metadata,
  step)``. Its ``predict_mass`` / ``predict_chi`` methods accept raw features.

No physical model, current, measurement loss, full forward or locked cohort is
imported or called here. Cache timings cannot support an end-to-end speed claim.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import csv
import json
import math
import os
from pathlib import Path
import time
from typing import Mapping
import uuid

import numpy as np
import torch
from torch.nn import functional as F

from a24.contracts import ForbiddenAccess, NumericalFailure
from .budget import write_json
from .decoder import ResidualDecoder


METHODS = ("N1", "N2", "N3")
FAMILIES = ("shell", "asymmetric_piecewise", "smooth", "mixed_scale")
UPDATE_CHOICES = (2000, 1000, 500, 250, 100)
GRID, CELLS, CHANNELS, VOLUME = 24, 576, 12, 1. / 576.
CACHE_SCHEMA = "A26-TARGET-FREE-CACHE-v1"
METADATA_KEYS = ("case_id", "parent_id", "family", "background_id", "sigma", "noise_db")
PAYLOAD_KEYS = ("features", "base_mass", "initial_mass", "beta")
CACHE_KEYS = frozenset((*METADATA_KEYS, *PAYLOAD_KEYS, "method", "split", "cache_schema",
                        "protocol_id", "amplitude", "variant_id"))


def _role(role):
    if role not in {"train", "validation"}:
        raise ForbiddenAccess("A26_CACHE_TRAINING_ACCEPTS_ONLY_TRAIN_AND_VALIDATION")
    return role


def _method(method):
    if method not in METHODS:
        raise ValueError("A26_N1_N2_N3_METHOD_REQUIRED")
    return method


def _scalar_string(value, name):
    value = np.asarray(value)
    if value.size != 1 or value.dtype.kind not in "US":
        raise ValueError("A26_SCALAR_STRING_REQUIRED:" + name)
    return str(value.reshape(-1)[0])


def _same(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        return False
    if left.dtype.kind in "fc" and right.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _validate_metadata(arrays, role):
    """Small row metadata is checked before a cache feature/label payload read."""
    _role(role)
    if set(METADATA_KEYS) - set(arrays):
        raise ValueError("A26_COMPLETE_CACHE_METADATA_REQUIRED")
    count = 512 if role == "train" else 128
    expected = set(range(250001, 250065)) if role == "train" else set(range(250065, 250081))
    for key in METADATA_KEYS:
        if np.asarray(arrays[key]).shape != (count,):
            raise ValueError("A26_ALIGNED_FROZEN_METADATA_REQUIRED:" + key)
    parents = np.asarray(arrays["parent_id"])
    if parents.dtype.kind not in "iu" or set(map(int, parents)) != expected:
        raise ForbiddenAccess("A26_CACHE_PARENT_SPLIT_FIREWALL")
    cases = np.asarray(arrays["case_id"])
    if cases.dtype.kind not in "US" or len(set(map(str, cases))) != count:
        raise ValueError("A26_UNIQUE_STRING_CASE_IDS_REQUIRED")
    families = np.asarray(arrays["family"])
    backgrounds = np.asarray(arrays["background_id"])
    if set(map(str, families)) != set(FAMILIES) or set(map(str, backgrounds)) != {"weak", "ring"}:
        raise ValueError("A26_FROZEN_FAMILIES_AND_BACKGROUNDS_REQUIRED")
    for parent in expected:
        selected = parents == parent
        if int(selected.sum()) != 8 or len(set(map(str, families[selected]))) != 1:
            raise ValueError("A26_EIGHT_CONDITIONS_PER_SINGLE_FAMILY_PARENT_REQUIRED")
        if any(int((backgrounds[selected] == name).sum()) != 4 for name in ("weak", "ring")):
            raise ValueError("A26_ALL_BACKGROUND_CONDITIONS_REQUIRED")
    family_parent_counts = {name: len(set(map(int, parents[families == name]))) for name in FAMILIES}
    if set(family_parent_counts.values()) != {16 if role == "train" else 4}:
        raise ValueError("A26_BALANCED_FROZEN_FAMILY_PARENT_COUNTS_REQUIRED")
    sigma, noise_db = np.asarray(arrays["sigma"]), np.asarray(arrays["noise_db"])
    if sigma.dtype != np.float64 or noise_db.dtype != np.float64:
        raise ValueError("A26_FLOAT64_NOISE_METADATA_REQUIRED")
    if not np.isfinite(sigma).all() or (sigma < 0).any():
        raise ValueError("A26_FINITE_NONNEGATIVE_SIGMA_REQUIRED")
    noiseless = sigma == 0
    if not np.isnan(noise_db[noiseless]).all() or not np.isfinite(noise_db[~noiseless]).all():
        raise ValueError("A26_EXPLICIT_NOISE_FREE_GROUP_REQUIRED")
    if "split" in arrays:
        split = np.asarray(arrays["split"])
        if split.size not in {1, count} or set(map(str, split.reshape(-1))) != {role}:
            raise ForbiddenAccess("A26_CACHE_ROLE_SPLIT_MISMATCH")
    for key in ("amplitude", "variant_id"):
        if key in arrays and np.asarray(arrays[key]).shape != (count,):
            raise ValueError("A26_ALIGNED_CACHE_METADATA_REQUIRED:" + key)
    return count


def _validate_payload(arrays, count):
    expected = {"features": ((count, CHANNELS, GRID, GRID), np.float32),
                "base_mass": ((count, 2 * CELLS), np.float64),
                "initial_mass": ((count, 2 * CELLS), np.float64),
                "beta": ((count,), np.float64)}
    for key, (shape, dtype) in expected.items():
        value = np.asarray(arrays[key])
        if value.shape != shape or value.dtype != dtype:
            raise ValueError("A26_FROZEN_CACHE_SHAPE_OR_DTYPE_REQUIRED:" + key)
        if key != "beta" and not np.isfinite(value).all():
            raise NumericalFailure("A26_NONFINITE_FEATURE_CACHE:" + key)
    beta = np.asarray(arrays["beta"])
    noiseless = np.asarray(arrays["sigma"]) == 0
    method = _method(_scalar_string(arrays["method"], "method"))
    if method in {"N1", "N3"}:
        if not np.isnan(beta).all():
            raise ValueError("A26_N1_N3_BETA_NOT_COMPUTED_MUST_BE_NAN")
    elif (not np.isnan(beta[noiseless]).all() or not np.isfinite(beta[~noiseless]).all()
          or (beta[~noiseless] < 0).any()):
        raise ValueError("A26_N2_BETA_NAN_ONLY_FOR_NOISE_FREE_REQUIRED")


def write_feature_cache(path, arrays, *, method, role, config):
    """Immutable target-free cache; reject every unregistered payload key."""
    _role(role)
    _method(method)
    extra = set(arrays) - CACHE_KEYS
    if extra:
        raise ForbiddenAccess("A26_TARGET_FREE_CACHE_KEYS_ONLY:" + ",".join(sorted(extra)))
    values = {key: np.asarray(value) for key, value in arrays.items()}
    for key, expected in (("method", method), ("cache_schema", CACHE_SCHEMA),
                          ("protocol_id", str(config["protocol_id"]))):
        if key in values and _scalar_string(values[key], key) != expected:
            raise ForbiddenAccess("A26_FEATURE_CACHE_IDENTITY_MISMATCH:" + key)
        values[key] = np.asarray(expected)
    if "split" not in values:
        values["split"] = np.asarray(role)
    if set(PAYLOAD_KEYS) - set(values):
        raise ValueError("A26_COMPLETE_CACHE_PAYLOAD_REQUIRED")
    count = _validate_metadata(values, role)
    _validate_payload(values, count)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ForbiddenAccess("A26_IMMUTABLE_FEATURE_CACHE_ALREADY_EXISTS")
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        # Exclusive destination creation prevents concurrent retry overwrite.
        with path.open("xb") as target, temporary.open("rb") as source:
            while block := source.read(1024 * 1024):
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(path), "method": method, "split": role, "rows": count,
            "cache_schema": CACHE_SCHEMA, "contains_targets": False}


def load_feature_cache(path, *, role, method=None, config=None):
    """Never materialize feature payloads until the split firewall succeeds."""
    _role(role)  # Reject locked roles before even opening a file.
    if method is not None:
        _method(method)
    with np.load(Path(path), allow_pickle=False) as archive:
        keys = set(archive.files)
        if keys - CACHE_KEYS:
            raise ForbiddenAccess("A26_TARGET_FREE_CACHE_KEYS_ONLY:" + ",".join(sorted(keys - CACHE_KEYS)))
        required = {*METADATA_KEYS, *PAYLOAD_KEYS, "method", "split", "cache_schema", "protocol_id"}
        if required - keys:
            raise ValueError("A26_COMPLETE_FEATURE_CACHE_REQUIRED")
        values = {key: archive[key].copy() for key in keys - set(PAYLOAD_KEYS)}
        actual_method = _scalar_string(values["method"], "method")
        _method(actual_method)
        if method is not None and actual_method != method:
            raise ForbiddenAccess("A26_CACHE_METHOD_MISMATCH")
        if _scalar_string(values["cache_schema"], "cache_schema") != CACHE_SCHEMA:
            raise ForbiddenAccess("A26_FEATURE_CACHE_SCHEMA_MISMATCH")
        if config is not None and _scalar_string(values["protocol_id"], "protocol_id") != config["protocol_id"]:
            raise ForbiddenAccess("A26_FEATURE_CACHE_PROTOCOL_MISMATCH")
        count = _validate_metadata(values, role)
        values.update({key: archive[key].copy() for key in PAYLOAD_KEYS})
    _validate_payload(values, count)
    return values


class CachedDataset:
    """Trainer-only labels paired to a target-free cache by exact row metadata."""
    def __init__(self, cache, labels, *, role, background_materials):
        self.role = _role(role)
        if set(cache) - CACHE_KEYS:
            raise ForbiddenAccess("A26_TARGET_FREE_CACHE_KEYS_ONLY")
        self.method = _method(_scalar_string(cache["method"], "method"))
        self.count = _validate_metadata(cache, role)
        _validate_payload(cache, self.count)
        # Labels have already passed data.load_split's metadata-first firewall.
        _validate_metadata(labels, role)
        for key in METADATA_KEYS:
            if not _same(cache[key], labels[key]):
                raise ForbiddenAccess("A26_CACHE_LABEL_ALIGNMENT_MISMATCH:" + key)
        for key in ("amplitude", "variant_id"):
            if key in cache and (key not in labels or not _same(cache[key], labels[key])):
                raise ForbiddenAccess("A26_CACHE_LABEL_ALIGNMENT_MISMATCH:" + key)
        target = np.asarray(labels["chi"])
        if target.shape != (self.count, CELLS) or target.dtype != np.complex128:
            raise ValueError("A26_COMPLEX128_TARGET_MATERIAL_REQUIRED")
        if not np.isfinite(target).all():
            raise NumericalFailure("A26_NONFINITE_TARGET_MATERIAL")
        if set(background_materials) != {"weak", "ring"}:
            raise ValueError("A26_EXACT_KNOWN_BACKGROUND_MAPPING_REQUIRED")
        known = {}
        for name, value in background_materials.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            value = np.asarray(value)
            if value.shape != (CELLS,) or value.dtype != np.complex128 or not np.isfinite(value).all():
                raise ValueError("A26_COMPLEX128_KNOWN_BACKGROUND_REQUIRED:" + name)
            known[name] = value.copy()
        self.background = np.stack([known[str(name)] for name in cache["background_id"]])
        expected_bg = np.stack((self.background.real, self.background.imag), axis=1).reshape(
            self.count, 2, GRID, GRID).astype(np.float32)
        if not np.array_equal(np.asarray(cache["features"])[:, 8:10], expected_bg):
            raise ForbiddenAccess("A26_FEATURE_KNOWN_BACKGROUND_MISMATCH")
        self.cache = cache
        # No measurement or current array is retained by the trainer.
        self.chi = target.copy()
        self.delta_chi = self.chi - self.background
        self.parents = np.array(sorted(set(map(int, cache["parent_id"]))), dtype=np.int64)
        self.indices = np.arange(self.count, dtype=np.int64)
        self.by_parent = {int(parent): np.flatnonzero(cache["parent_id"] == parent) for parent in self.parents}

    @classmethod
    def from_npz(cls, path, data_root, *, role, background_materials, method=None, config=None):
        from .data import load_split
        _role(role)
        cache = load_feature_cache(path, role=role, method=method, config=config)
        labels = load_split(data_root, role)
        return cls(cache, labels, role=role, background_materials=background_materials)

    def batch(self, indices, device="cpu"):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices) or (indices < 0).any() or (indices >= self.count).any():
            raise ForbiddenAccess("A26_NONEMPTY_ALLOWED_CACHE_ROWS_REQUIRED")
        return {"features": torch.as_tensor(self.cache["features"][indices], dtype=torch.float32, device=device),
                "base_mass": torch.as_tensor(self.cache["base_mass"][indices], dtype=torch.float64, device=device),
                "chi": torch.as_tensor(self.chi[indices], dtype=torch.complex128, device=device),
                "background_chi": torch.as_tensor(self.background[indices], dtype=torch.complex128, device=device),
                "delta_chi": torch.as_tensor(self.delta_chi[indices], dtype=torch.complex128, device=device),
                "row_indices": indices.tolist(),
                "parent_id": np.asarray(self.cache["parent_id"])[indices].astype(int).tolist(),
                "case_id": list(map(str, np.asarray(self.cache["case_id"])[indices]))}


class ParentSampler:
    """One parent per family per batch; every 128 batches visits all 512 rows."""
    def __init__(self, dataset, seed, batch=4):
        if dataset.role != "train" or int(batch) != 4:
            raise ValueError("A26_FAMILY_BALANCED_FOUR_TRAIN_PARENTS_REQUIRED")
        self.dataset, self.batch = dataset, 4
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 26, 64, 8]))
        family = np.asarray(dataset.cache["family"])
        self.parents = {name: np.array([int(p) for p in dataset.parents
                                      if str(family[dataset.by_parent[int(p)][0]]) == name])
                        for name in FAMILIES}
        self.parent_order = {name: self.rng.permutation(parents) for name, parents in self.parents.items()}
        self.parent_position = {name: 0 for name in FAMILIES}
        self.condition_order = {int(p): self.rng.permutation(dataset.by_parent[int(p)]) for p in dataset.parents}
        self.condition_position = {int(p): 0 for p in dataset.parents}

    def next(self):
        rows = []
        for family in FAMILIES:
            if self.parent_position[family] == len(self.parent_order[family]):
                self.parent_order[family] = self.rng.permutation(self.parents[family])
                self.parent_position[family] = 0
            parent = int(self.parent_order[family][self.parent_position[family]])
            self.parent_position[family] += 1
            if self.condition_position[parent] == len(self.condition_order[parent]):
                self.condition_order[parent] = self.rng.permutation(self.dataset.by_parent[parent])
                self.condition_position[parent] = 0
            rows.append(int(self.condition_order[parent][self.condition_position[parent]]))
            self.condition_position[parent] += 1
        return np.asarray(rows, dtype=np.int64)[self.rng.permutation(4)]


class SharedNormalizer(torch.nn.Module):
    def __init__(self, channel_mean, channel_std, material_scale, provenance):
        super().__init__()
        mean = torch.as_tensor(channel_mean, dtype=torch.float32).reshape(1, CHANNELS, 1, 1)
        std = torch.as_tensor(channel_std, dtype=torch.float32).reshape_as(mean)
        scale = torch.as_tensor(material_scale, dtype=torch.float64).reshape(())
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 1e-6).any():
            raise ValueError("A26_FINITE_CHANNEL_STATISTICS_WITH_STD_FLOOR_REQUIRED")
        if not torch.isfinite(scale) or float(scale) <= 0:
            raise ValueError("A26_POSITIVE_FINITE_TRAIN_DELTA_RMS_REQUIRED")
        self.register_buffer("channel_mean", mean)
        self.register_buffer("channel_std", std)
        self.register_buffer("material_scale", scale)
        self.provenance = dict(provenance)

    def forward(self, features):
        if features.dtype != torch.float32 or tuple(features.shape[1:]) != (CHANNELS, GRID, GRID):
            raise ValueError("A26_FLOAT32_TWELVE_CHANNEL_FEATURES_REQUIRED")
        return (features - self.channel_mean.to(features.device)) / self.channel_std.to(features.device)


def _span(book, name, **counts):
    return book.span(name, **counts) if book is not None else nullcontext()


def _check(book):
    if book is not None:
        book.check()


def _synchronize(device, book=None):
    if book is not None:
        book.synchronize()
    elif torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _assert_datasets(datasets, role):
    if set(datasets) != set(METHODS):
        raise ValueError("A26_ALL_THREE_METHOD_DATASETS_REQUIRED")
    for method, dataset in datasets.items():
        if dataset.method != method or dataset.role != role:
            raise ForbiddenAccess("A26_MATCHED_DATASET_METHOD_ROLE_REQUIRED")
        for key in METADATA_KEYS:
            if not _same(dataset.cache[key], datasets["N1"].cache[key]):
                raise ForbiddenAccess("A26_MATCHED_METHOD_ROWS_REQUIRED:" + key)
        if not np.array_equal(dataset.delta_chi, datasets["N1"].delta_chi):
            raise ForbiddenAccess("A26_MATCHED_METHOD_TARGETS_REQUIRED")
    if not np.array_equal(datasets["N1"].cache["base_mass"], datasets["N3"].cache["base_mass"]):
        raise ForbiddenAccess("A26_N1_N3_IDENTICAL_PHYSICAL_BASE_REQUIRED")


def fit_normalizer(train_datasets, *, config, book, batch_size=32):
    _assert_datasets(train_datasets, "train")
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("A26_POSITIVE_STATISTICS_BATCH_REQUIRED")
    total, mean, m2 = 0, np.zeros(CHANNELS), np.zeros(CHANNELS)
    with _span(book, "a26_train_only_pooled_normalizer", normalization_images=1536):
        for method in METHODS:
            dataset = train_datasets[method]
            for start in range(0, dataset.count, int(batch_size)):
                _check(book)
                value = np.asarray(dataset.cache["features"][start:start + batch_size], dtype=np.float64)
                local_count = value.shape[0] * CELLS
                local_mean = value.mean(axis=(0, 2, 3))
                local_m2 = ((value - local_mean[None, :, None, None]) ** 2).sum(axis=(0, 2, 3))
                difference = local_mean - mean
                combined = total + local_count
                m2 += local_m2 + difference ** 2 * total * local_count / combined
                mean += difference * local_count / combined
                total = combined
        delta = train_datasets["N1"].delta_chi
        material_scale = float(np.sqrt(np.mean(np.abs(delta) ** 2)))
    std = np.maximum(np.sqrt(m2 / total), 1e-6)
    provenance = {"role": "train", "methods": list(METHODS), "parent_ids": list(range(250001, 250065)),
                  "rows_per_method": 512, "pooled_feature_images": 1536, "channel_value_count": total,
                  "material_scale_definition": "RMS(abs(TRAIN chi - exact known chi_b))",
                  "material_label_rows": 512, "protocol_id": str(config["protocol_id"]), "std_floor": 1e-6}
    return SharedNormalizer(mean, std, material_scale, provenance)


def configure_torch(seed, device="cpu"):
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"} or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("A26_CPU_OR_AVAILABLE_CUDA_ONLY")
    if int(seed) not in {261009, 261010}:
        raise ValueError("A26_FROZEN_PRIMARY_OR_REPEAT_SEED_REQUIRED")
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False


def _build_model(width, config, device, seed):
    if int(width) not in {32, 16}:
        raise ValueError("A26_WIDTH_32_OR_COST_FALLBACK_16_REQUIRED")
    configure_torch(seed, device)
    model = ResidualDecoder(width=int(width), **{key: float(config.get(key, default))
                            for key, default in (("real_lower", -.5), ("real_upper", 4.),
                                                 ("imag_lower", 0.), ("imag_upper", .5))}).to(device)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != (31810 if int(width) == 32 else 8994):
        raise ValueError("A26_FROZEN_DECODER_PARAMETER_COUNT_MISMATCH")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("A26_FLOAT32_CNN_PARAMETERS_REQUIRED")
    return model


def build_matched_models(*, config, width=32, seed=None, device="cpu"):
    seed = int(config["primary_seed"] if seed is None else seed)
    models = {method: _build_model(width, config, device, seed) for method in METHODS}
    for method in METHODS[1:]:
        if any(not torch.equal(value, models[method].state_dict()[key])
               for key, value in models["N1"].state_dict().items()):
            raise NumericalFailure("A26_IDENTICAL_INITIALIZATION_REQUIRED")
    return models


def _normalizer_admission(normalizer, config):
    provenance = normalizer.provenance
    if (provenance.get("role") != "train" or provenance.get("methods") != list(METHODS)
            or provenance.get("parent_ids") != list(range(250001, 250065))
            or provenance.get("protocol_id") != config["protocol_id"]
            or provenance.get("pooled_feature_images") != 1536):
        raise ForbiddenAccess("A26_SHARED_POOLED_TRAIN_NORMALIZER_REQUIRED")


def _optimizer(model, config):
    if float(config.get("lr", 1e-3)) != 1e-3 or float(config.get("weight_decay", 1e-5)) != 1e-5:
        raise ValueError("A26_FROZEN_ADAMW_HYPERPARAMETERS_REQUIRED")
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)


def mass_to_chi(mass):
    if mass.dtype != torch.float64 or mass.ndim != 2 or mass.shape[1] != 2 * CELLS:
        raise ValueError("A26_FLOAT64_DECODER_MASS_REQUIRED")
    return torch.complex(mass[:, :CELLS], mass[:, CELLS:]) / math.sqrt(VOLUME)


def ssim_delta(predicted, target):
    """Local complex delta SSIM, fixed material domain ranges 4.5 and .5."""
    if (predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != CELLS
            or predicted.dtype != torch.complex128 or target.dtype != torch.complex128):
        raise ValueError("A26_MATCHED_COMPLEX128_DELTA_REQUIRED")
    p = torch.stack((predicted.real, predicted.imag), 1).reshape(-1, 2, GRID, GRID)
    t = torch.stack((target.real, target.imag), 1).reshape_as(p)
    mp, mt = F.avg_pool2d(p, 7, stride=1), F.avg_pool2d(t, 7, stride=1)
    vp = (F.avg_pool2d(p.square(), 7, stride=1) - mp.square()).clamp_min(0)
    vt = (F.avg_pool2d(t.square(), 7, stride=1) - mt.square()).clamp_min(0)
    covariance = F.avg_pool2d(p * t, 7, stride=1) - mp * mt
    ranges = p.new_tensor([4.5, .5]).reshape(1, 2, 1, 1)
    c1, c2 = (.01 * ranges).square(), (.03 * ranges).square()
    score = ((2 * mp * mt + c1) * (2 * covariance + c2)
             / ((mp.square() + mt.square() + c1) * (vp + vt + c2)))
    return score.mean(dim=(1, 2, 3))


def delta_loss(predicted_chi, batch, material_scale):
    predicted_delta = predicted_chi - batch["background_chi"]
    expected_delta = batch["delta_chi"]
    scale = torch.as_tensor(material_scale, dtype=torch.float64, device=predicted_chi.device)
    mse = (predicted_delta - expected_delta).abs().square().mean(-1)
    normalized_mse = mse / scale.square()
    ssim_loss = 1. - ssim_delta(predicted_delta, expected_delta)
    total = (normalized_mse + .05 * ssim_loss).mean()
    if not torch.isfinite(total):
        raise NumericalFailure("A26_NONFINITE_DELTA_TRAINING_LOSS")
    return total, {"delta_mse": mse.mean(), "normalized_delta_mse": normalized_mse.mean(),
                   "ssim_delta_loss": ssim_loss.mean(), "loss_total": total}


def one_update(model, optimizer, batch, *, normalizer, config, book, phase="scientific"):
    if phase not in {"scientific", "benchmark_warmup", "benchmark"}:
        raise ValueError("A26_UPDATE_PHASE_REQUIRED")
    if float(config.get("ssim_weight", .05)) != .05 or float(config.get("grad_clip", 1.)) != 1.:
        raise ValueError("A26_FROZEN_DELTA_LOSS_AND_CLIP_REQUIRED")
    _normalizer_admission(normalizer, config)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    counts = {"optimizer_updates": 1, "optimizer_parent_images": len(batch["parent_id"]),
              "scientific_training_updates" if phase == "scientific" else "benchmark_training_updates": 1}
    with _span(book, "a26_cache_only_optimizer_update", **counts):
        mass = model(normalizer(batch["features"]), batch["base_mass"], normalizer.material_scale)
        loss, components = delta_loss(mass_to_chi(mass), batch, normalizer.material_scale)
        with _span(book, "a26_cache_only_backward", autograd_backward_calls=1):
            loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if not bool(torch.stack([torch.isfinite(parameter).all() for parameter in model.parameters()]).all()):
            raise NumericalFailure("A26_NONFINITE_CNN_PARAMETERS")
    return {key: float(value.detach().cpu()) for key, value in components.items()}, float(gradient_norm.detach().cpu())


def validate(model, dataset, *, normalizer, config, book, device="cpu", batch_size=4):
    if dataset.role != "validation" or dataset.count != 128 or len(dataset.parents) != 16:
        raise ForbiddenAccess("A26_COMPLETE_EXPOSED_VALIDATION_ONLY")
    _normalizer_admission(normalizer, config)
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("A26_POSITIVE_VALIDATION_BATCH_REQUIRED")
    model.eval()
    rows = []
    with _span(book, "a26_cache_only_validation", validation_images=128), torch.no_grad():
        for start in range(0, dataset.count, int(batch_size)):
            _check(book)
            indices = dataset.indices[start:start + batch_size]
            values = dataset.batch(indices, device)
            prediction = mass_to_chi(model(normalizer(values["features"]), values["base_mass"], normalizer.material_scale))
            delta = prediction - values["background_chi"]
            target = values["delta_chi"]
            denominator = torch.linalg.vector_norm(target, dim=-1)
            if bool((denominator <= 0).any()):
                raise NumericalFailure("A26_ZERO_DELTA_TARGET_NRMSE_DENOMINATOR")
            error = torch.linalg.vector_norm(delta - target, dim=-1) / denominator
            mse = (delta - target).abs().square().mean(-1)
            ssim = ssim_delta(delta, target)
            if not torch.isfinite(error).all() or not torch.isfinite(ssim).all():
                raise NumericalFailure("A26_NONFINITE_VALIDATION_METRIC")
            for local, index in enumerate(indices):
                sigma = float(dataset.cache["sigma"][index])
                rows.append({"case_id": str(dataset.cache["case_id"][index]),
                             "parent_id": int(dataset.cache["parent_id"][index]),
                             "family": str(dataset.cache["family"][index]),
                             "background_id": str(dataset.cache["background_id"][index]),
                             "sigma": sigma, "noise_group": "NOISE_FREE" if sigma == 0 else "NOISY",
                             "noise_db": None if sigma == 0 else float(dataset.cache["noise_db"][index]),
                             "delta_nrmse": float(error[local].cpu()), "delta_mse": float(mse[local].cpu()),
                             "ssim_delta": float(ssim[local].cpu())})
    parent_rows = [{"parent_id": int(parent), "conditions": 8,
                    "delta_nrmse": float(np.mean([row["delta_nrmse"] for row in rows if row["parent_id"] == parent]))}
                   for parent in dataset.parents]
    score = float(np.mean([row["delta_nrmse"] for row in parent_rows]))
    return {"role": "validation", "selection_metric": "equal_parent_delta_nrmse",
            "parent_mean_delta_nrmse": score, "parent_count": 16, "case_count": 128,
            "complete": True, "parent_rows": parent_rows, "rows": rows,
            "physical_forward_audit_performed": False, "locked_truth_opened": False}


def predict_cached(model, normalizer, dataset, device="cpu", batch=4, book=None):
    """Return aligned chi/mass arrays; this is explicitly a cache-only path."""
    _role(dataset.role)
    batch = int(batch)
    if batch < 1:
        raise ValueError("A26_POSITIVE_PREDICTION_BATCH_REQUIRED")
    normalizer = normalizer.to(device)
    model.eval()
    masses = []
    with _span(book, "a26_cached_prediction", cached_prediction_images=dataset.count), torch.no_grad():
        for start in range(0, dataset.count, batch):
            _check(book)
            indices = dataset.indices[start:start + batch]
            # Prediction accesses only online cache features and physical base.
            features = torch.as_tensor(dataset.cache["features"][indices], dtype=torch.float32, device=device)
            base = torch.as_tensor(dataset.cache["base_mass"][indices], dtype=torch.float64, device=device)
            mass = model(normalizer(features), base, normalizer.material_scale)
            if mass.dtype != torch.float64 or not torch.isfinite(mass).all():
                raise NumericalFailure("A26_INVALID_CACHED_PREDICTION_MASS")
            masses.append(mass.detach().cpu().numpy())
    mass = np.concatenate(masses)
    chi = (mass[:, :CELLS] + 1j * mass[:, CELLS:]) / math.sqrt(VOLUME)
    metadata = {key: np.asarray(dataset.cache[key]).copy() for key in METADATA_KEYS}
    for key in ("amplitude", "variant_id"):
        if key in dataset.cache:
            metadata[key] = np.asarray(dataset.cache[key]).copy()
    return {"mass": mass, "chi": chi.astype(np.complex128, copy=False), "metadata": metadata,
            "cache_only": True, "full_inference_latency_measured": False}


def benchmark_training(train_datasets, validation_datasets, *, normalizer, config, book,
                       width=32, device="cpu", seed=None):
    """Actual TRAIN micro updates are discarded; width/U selection uses costs."""
    _assert_datasets(train_datasets, "train")
    _assert_datasets(validation_datasets, "validation")
    _normalizer_admission(normalizer, config)
    seed = int(config["primary_seed"] if seed is None else seed)
    rows = []
    normalizer = normalizer.to(device)
    with _span(book, "a26_cache_training_benchmark", training_microbenchmarks=1):
        for method in METHODS:
            _check(book)
            _synchronize(device, book)
            build_wall, build_cpu = time.perf_counter(), time.process_time()
            model = _build_model(width, config, device, seed)
            optimizer = _optimizer(model, config)
            _synchronize(device, book)
            startup_wall, startup_cpu = time.perf_counter() - build_wall, time.process_time() - build_cpu
            indices = ParentSampler(train_datasets[method], seed).next()
            batch = train_datasets[method].batch(indices, device)
            one_update(model, optimizer, batch, normalizer=normalizer, config=config, book=book, phase="benchmark_warmup")
            wall, cpu, losses = [], [], []
            for _ in range(10):
                _check(book)
                _synchronize(device, book)
                started_wall, started_cpu = time.perf_counter(), time.process_time()
                batch = train_datasets[method].batch(indices, device)
                components, _ = one_update(model, optimizer, batch, normalizer=normalizer,
                                           config=config, book=book, phase="benchmark")
                _synchronize(device, book)
                wall.append(time.perf_counter() - started_wall)
                cpu.append(time.process_time() - started_cpu)
                losses.append(components)
            _synchronize(device, book)
            started_wall, started_cpu = time.perf_counter(), time.process_time()
            validation = validate(model, validation_datasets[method], normalizer=normalizer,
                                  config=config, book=book, device=device)
            _synchronize(device, book)
            validation_wall, validation_cpu = time.perf_counter() - started_wall, time.process_time() - started_cpu
            rows.append({"method": method, "width": int(width),
                         "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                         "startup_wall_seconds": startup_wall, "startup_cpu_seconds": startup_cpu,
                         "update_wall_seconds": max(wall), "update_cpu_seconds": max(cpu),
                         "samples_wall_seconds": wall, "samples_cpu_seconds": cpu,
                         "validation_wall_seconds": validation_wall, "validation_cpu_seconds": validation_cpu,
                         "validation_case_count": validation["case_count"], "validation_parent_count": validation["parent_count"],
                         "train_row_indices": indices.tolist(), "train_case_ids": batch["case_id"],
                         "train_parent_ids": batch["parent_id"], "batch_size": 4,
                         "warmup_updates": 1, "measured_updates": 10, "losses": losses,
                         "scientific_training_updates": 0})
            del model, optimizer, batch
    return {"status": "CACHE_ONLY_TRAIN_MICROBENCHMARK", "device": str(device), "width": int(width),
            "seed": seed, "rows": rows, "benchmark_updates": 33, "scientific_training_updates": 0,
            "weights_and_optimizer_discarded": True, "selection_basis": "cost_only",
            "full_inference_latency_measured": False, "locked_truth_opened": False}


def _validation_steps(updates, every=100):
    return sorted(set(range(every, int(updates) + 1, every)) | {int(updates)})


def select_update_budget(benchmark, *, config, cpu_available, gpu_available, fallback_from=None):
    """Largest frozen U for all three arms, with complete validation and reserve."""
    if tuple(config.get("updates_candidates", UPDATE_CHOICES)) != UPDATE_CHOICES:
        raise ValueError("A26_FROZEN_UPDATE_CHOICES_REQUIRED")
    if float(config.get("cost_safety_factor", 1.3)) != 1.3 or int(config.get("validation_every", 100)) != 100:
        raise ValueError("A26_FROZEN_COST_FACTOR_AND_VALIDATION_INTERVAL_REQUIRED")
    width = int(benchmark["width"])
    if width not in {32, 16}:
        raise ValueError("A26_FROZEN_BENCHMARK_WIDTH_REQUIRED")
    if width == 16 and (not isinstance(fallback_from, dict) or fallback_from.get("status") != "INSUFFICIENT_BUDGET"
                        or fallback_from.get("width") != 32 or fallback_from.get("updates") is not None):
        raise ForbiddenAccess("A26_WIDTH_16_REQUIRES_COST_ONLY_WIDTH_32_FAILURE")
    rows = {row["method"]: row for row in benchmark["rows"]}
    if set(rows) != set(METHODS) or len(benchmark["rows"]) != 3:
        raise ValueError("A26_ALL_THREE_ARM_COSTS_REQUIRED")
    for row in rows.values():
        if (row.get("width") != width or row.get("batch_size") != 4 or row.get("warmup_updates") != 1
                or row.get("measured_updates") != 10 or row.get("validation_case_count") != 128
                or row.get("validation_parent_count") != 16):
            raise ValueError("A26_ACTUAL_TRAIN_AND_COMPLETE_VALIDATION_BENCHMARK_REQUIRED")
        for key in ("update_wall_seconds", "update_cpu_seconds", "validation_wall_seconds", "validation_cpu_seconds"):
            if not math.isfinite(float(row[key])) or float(row[key]) <= 0:
                raise ValueError("A26_POSITIVE_MEASURED_COST_REQUIRED:" + key)
    device = torch.device(benchmark["device"])
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("A26_CPU_OR_CUDA_COST_DEVICE_REQUIRED")
    cpu_reserve = float(config.get("final_cpu_reserve_seconds", 1800))
    gpu_reserve = float(config.get("final_gpu_reserve_seconds", 900)) if device.type == "cuda" else 0.
    if cpu_reserve != 1800 or (device.type == "cuda" and gpu_reserve != 900):
        raise ValueError("A26_FROZEN_FINAL_RESERVES_REQUIRED")
    candidates = []
    for updates in UPDATE_CHOICES:
        validations = len(_validation_steps(updates))
        cpu = sum(updates * float(row["update_cpu_seconds"]) + validations * float(row["validation_cpu_seconds"])
                  + float(row.get("startup_cpu_seconds", 0)) for row in rows.values())
        wall = sum(updates * float(row["update_wall_seconds"]) + validations * float(row["validation_wall_seconds"])
                   + float(row.get("startup_wall_seconds", 0)) for row in rows.values())
        cpu_charge = 1.3 * (max(cpu, wall) if device.type == "cpu" else cpu)
        gpu_charge = 1.3 * wall if device.type == "cuda" else 0.
        candidates.append({"updates": updates, "validation_events_per_arm": validations,
                           "projected_cpu_seconds": cpu_charge, "projected_gpu_seconds": gpu_charge,
                           "required_cpu_with_reserve": cpu_charge + cpu_reserve,
                           "required_gpu_with_reserve": gpu_charge + gpu_reserve,
                           "fits": cpu_charge + cpu_reserve <= float(cpu_available)
                                   and gpu_charge + gpu_reserve <= float(gpu_available)})
    fitting = [row for row in candidates if row["fits"]]
    selected = fitting[0] if fitting else None
    return {"status": "COST_SELECTED" if selected else "INSUFFICIENT_BUDGET", "width": width,
            "updates": selected["updates"] if selected else None, "complete_jobs": 3,
            "device": str(device), "seed": int(benchmark["seed"]), "safety_factor": 1.3,
            "final_cpu_reserve_seconds": cpu_reserve, "final_gpu_reserve_seconds": gpu_reserve,
            "candidates": candidates, "selection_basis": "cost_only_no_quality_metric",
            "fallback_width_allowed": 16 if width == 32 and not selected else None,
            "width_32_failure": fallback_from if width == 16 else None,
            "scientific_training_updates": 0}


def _save_checkpoint(path, model, normalizer, *, metadata, step, kind, validation_score, book):
    payload = {"state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
               "normalizer_state_dict": {key: value.detach().cpu().clone() for key, value in normalizer.state_dict().items()},
               "normalizer_provenance": dict(normalizer.provenance), "metadata": dict(metadata),
               "step": int(step), "checkpoint_kind": kind, "validation_score": validation_score,
               "feature_mean": normalizer.channel_mean.detach().cpu().clone(),
               "feature_std": normalizer.channel_std.detach().cpu().clone(),
               "material_scale": normalizer.material_scale.detach().cpu().clone(),
               "scientific_training_updates": int(step), "targets_in_checkpoint": False,
               "exposed_validation_used_for_selection": kind == "best",
               "validation_cohort_status": "A25_HISTORICALLY_EXPOSED_DEVELOPMENT",
               "selection_metric": "equal_parent_delta_nrmse", "locked_truth_opened": False}
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with _span(book, "a26_checkpoint_write", checkpoint_writes=1):
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def train_one_model(method, train_data, validation_data, *, selection, normalizer, config,
                    out, book, device="cpu", seed=None):
    _method(method)
    if (selection.get("status") != "COST_SELECTED" or selection.get("complete_jobs") != 3
            or selection.get("selection_basis") != "cost_only_no_quality_metric"
            or selection.get("updates") not in UPDATE_CHOICES):
        raise ForbiddenAccess("A26_THREE_ARM_COST_ONLY_TRAINING_ADMISSION_REQUIRED")
    updates, width = int(selection["updates"]), int(selection["width"])
    if width == 16 and (not isinstance(selection.get("width_32_failure"), Mapping)
                        or selection["width_32_failure"].get("status") != "INSUFFICIENT_BUDGET"):
        raise ForbiddenAccess("A26_COST_ONLY_WIDTH_16_FALLBACK_REQUIRED")
    if train_data.role != "train" or validation_data.role != "validation" or train_data.method != method or validation_data.method != method:
        raise ForbiddenAccess("A26_MATCHED_TRAIN_VALIDATION_ARM_REQUIRED")
    if set(train_data.parents) & set(validation_data.parents):
        raise ForbiddenAccess("A26_TRAIN_VALIDATION_PARENT_OVERLAP")
    _normalizer_admission(normalizer, config)
    seed = int(config["primary_seed"] if seed is None else seed)
    if (int(selection.get("seed", seed)) not in {261009, 261010}
            or torch.device(selection["device"]) != torch.device(device)):
        raise ForbiddenAccess("A26_COST_SELECTION_SEED_DEVICE_MISMATCH")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("training_manifest.json", "best.pt", "middle.pt", "final.pt", "raw_loss.csv")):
        raise ForbiddenAccess("A26_TRAINING_ATTEMPT_ALREADY_EXISTS")
    normalizer = normalizer.to(device)
    # Rebuilding here resets every benchmark update, the optimizer and RNG.
    model = _build_model(width, config, device, seed)
    optimizer = _optimizer(model, config)
    sampler = ParentSampler(train_data, seed)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metadata = {"method": method, "width": width, "seed": seed, "parameter_count": parameter_count,
                "protocol_id": str(config["protocol_id"]), "source_commit": str(config["source_commit"]),
                "updates": updates, "cost_measurement_seed": int(selection.get("seed", seed)),
                "volume": VOLUME, "grid": GRID,
                "real_lower": float(config.get("real_lower", -.5)), "real_upper": float(config.get("real_upper", 4.)),
                "imag_lower": float(config.get("imag_lower", 0.)), "imag_upper": float(config.get("imag_upper", .5))}
    manifest = {"status": "TRAINING", **metadata, "updates": updates, "completed_updates": 0,
                "scientific_training_updates": 0, "benchmark_updates_in_model": 0,
                "train_parents": train_data.parents.tolist(), "validation_parents": validation_data.parents.tolist(),
                "normalizer_provenance": normalizer.provenance,
                "batch_rule": "one parent per family; parent and condition permutations cycle through all 512 rows",
                "sampler_seed": [seed, 26, 64, 8], "validation_steps": _validation_steps(updates),
                "middle_step": updates // 2, "selection_metric": "equal_parent_delta_nrmse",
                "supervision": "delta_material_MSE/shared_scale_squared + .05*(1-SSIM_delta)",
                "locked_truth_opened": False, "physical_backward_calls": 0}
    write_json(out / "training_manifest.json", manifest)
    history, validation_history = [], []
    best, best_step = float("inf"), None
    completed = 0
    fields = ["step", "parent_ids", "case_ids", "row_indices", "gradient_norm_before_clip",
              "delta_mse", "normalized_delta_mse", "ssim_delta_loss", "loss_total"]
    try:
        with (out / "raw_loss.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            with _span(book, "a26_cache_only_scientific_train", training_attempts=1):
                for step in range(1, updates + 1):
                    _check(book)
                    indices = sampler.next()
                    values = train_data.batch(indices, device)
                    components, gradient = one_update(model, optimizer, values, normalizer=normalizer,
                                                     config=config, book=book, phase="scientific")
                    completed = step
                    row = {"step": step, "parent_ids": values["parent_id"], "case_ids": values["case_id"],
                           "row_indices": indices.tolist(), "gradient_norm_before_clip": gradient, **components}
                    history.append(row)
                    csv_row = dict(row)
                    for key in ("parent_ids", "case_ids", "row_indices"):
                        csv_row[key] = json.dumps(row[key], separators=(",", ":"))
                    writer.writerow(csv_row)
                    handle.flush()
                    if step == updates // 2:
                        _save_checkpoint(out / "middle.pt", model, normalizer, metadata=metadata, step=step,
                                         kind="middle", validation_score=None, book=book)
                    if step in manifest["validation_steps"]:
                        score = validate(model, validation_data, normalizer=normalizer, config=config, book=book, device=device)
                        validation_history.append({"step": step, **score})
                        # Strict inequality retains the earliest checkpoint on exact ties.
                        if score["parent_mean_delta_nrmse"] < best:
                            best, best_step = score["parent_mean_delta_nrmse"], step
                            _save_checkpoint(out / "best.pt", model, normalizer, metadata=metadata, step=step,
                                             kind="best", validation_score=best, book=book)
                        write_json(out / "progress.json", {"status": "TRAINING", "step": step,
                                   "best_step": best_step, "best_validation_delta_nrmse": best})
            if best_step is None:
                raise NumericalFailure("A26_NO_COMPLETE_VALIDATION_CHECKPOINT")
            _save_checkpoint(out / "final.pt", model, normalizer, metadata=metadata, step=updates,
                             kind="final", validation_score=validation_history[-1]["parent_mean_delta_nrmse"], book=book)
        manifest.update(status="TRAINED_EXPOSED_VALIDATION_SELECTED", best_step=best_step,
                        best_validation_delta_nrmse=best,
                        checkpoints={kind: str(out / (kind + ".pt")) for kind in ("best", "middle", "final")})
    except Exception as error:
        manifest.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        manifest.update(completed_updates=completed, scientific_training_updates=completed)
        write_json(out / "raw_loss.json", history)
        write_json(out / "validation_history.json", validation_history)
        write_json(out / "training_manifest.json", manifest)
    return manifest


@dataclass
class LoadedCheckpoint:
    model: torch.nn.Module
    normalizer: SharedNormalizer
    metadata: dict
    step: int

    def predict_mass(self, features, base_mass):
        return self.model(self.normalizer(features), base_mass, self.normalizer.material_scale)

    def predict_chi(self, features, base_mass):
        return mass_to_chi(self.predict_mass(features, base_mass))


def load_trained_model(path, *, device="cpu", config=None):
    """Checkpoint payloads contain only tensors and weights-only-safe primitives."""
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    metadata = payload["metadata"]
    _method(metadata["method"])
    if metadata.get("grid") != GRID or metadata.get("volume") != VOLUME or metadata.get("seed") not in {261009, 261010}:
        raise ForbiddenAccess("A26_CHECKPOINT_FROZEN_TOPOLOGY_MISMATCH")
    local = {"primary_seed": metadata["seed"], "protocol_id": metadata["protocol_id"],
             **{key: metadata[key] for key in ("real_lower", "real_upper", "imag_lower", "imag_upper")}}
    if config is not None:
        if config["protocol_id"] != metadata["protocol_id"]:
            raise ForbiddenAccess("A26_CHECKPOINT_PROTOCOL_MISMATCH")
        if "source_commit" in config and config["source_commit"] != metadata.get("source_commit"):
            raise ForbiddenAccess("A26_CHECKPOINT_SOURCE_COMMIT_MISMATCH")
        for key in ("real_lower", "real_upper", "imag_lower", "imag_upper"):
            if key in config and float(config[key]) != metadata[key]:
                raise ForbiddenAccess("A26_CHECKPOINT_BOX_MISMATCH:" + key)
    normalizer_state = payload["normalizer_state_dict"]
    normalizer = SharedNormalizer(normalizer_state["channel_mean"].reshape(-1),
                                  normalizer_state["channel_std"].reshape(-1),
                                  normalizer_state["material_scale"], payload["normalizer_provenance"]).to(device)
    _normalizer_admission(normalizer, local)
    model = _build_model(metadata["width"], local, device, metadata["seed"])
    if sum(parameter.numel() for parameter in model.parameters()) != metadata["parameter_count"]:
        raise ForbiddenAccess("A26_CHECKPOINT_PARAMETER_COUNT_MISMATCH")
    model.load_state_dict(payload["state_dict"], strict=True)
    if not bool(torch.stack([torch.isfinite(parameter).all() for parameter in model.parameters()]).all()):
        raise NumericalFailure("A26_NONFINITE_CHECKPOINT_PARAMETERS")
    model.eval()
    return LoadedCheckpoint(model, normalizer, metadata, int(payload["step"]))
