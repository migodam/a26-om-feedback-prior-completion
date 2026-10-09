"""Read-only A25 data reuse with A26 metadata-first split and encoder firewalls.

The primary loader accepts only train/validation. It opens the public manifest,
then split metadata, and verifies every parent and condition before accessing
observations, legacy TV baselines, or chi labels. Full target currents and clean
measurements are never materialized. Locked access has a separate explicit API.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import numpy as np

from a24.contracts import ForbiddenAccess
from .budget import BudgetLedger, owned_file_lock, write_json

PROTOCOL_ID = "A26-FROZEN-ENCODER-20261009-v1"
SOURCE_MANIFEST_ID = "A25-NN-20261008-TASK-R32-v1:parents-v1"
FAMILIES = ("shell", "asymmetric_piecewise", "smooth", "mixed_scale")
PRIMARY_ROLES = frozenset({"train", "validation"})
LOCKED_ROLES = frozenset({"locked_id", "locked_ood"})
SPLIT_PARENT_IDS = {"train": tuple(range(250001, 250065)),
                    "validation": tuple(range(250065, 250081)),
                    "locked_id": tuple(range(250081, 250097)),
                    "locked_ood": tuple(range(250097, 250113))}
DEFAULT_DATA_ROOT = (Path(__file__).resolve().parents[4] /
                     "A25/a25-adaptive-om-feedback/data")
ROW_METADATA = frozenset({"parent_id", "background_id", "split", "family", "ood_type",
                          "variant_id", "case_id", "amplitude", "noise_db", "status"})
SCALAR_METADATA = frozenset({"manifest_id", "geometry_descriptor"})
BASELINE_KEYS = frozenset({"initial_mass", "lambda", "initial_wall", "initializer_id"})
ONLINE_KEYS = ROW_METADATA | SCALAR_METADATA | {"data", "sigma", "noise_group"}
ARCHIVE_KEYS = (ROW_METADATA | SCALAR_METADATA | BASELINE_KEYS |
                {"data", "sigma", "noise_seed", "chi", "current", "clean"})


def _path_is_locked(path):
    return any(part.lower() == "locked" or part.lower().startswith(("locked_id", "locked_ood"))
               for part in Path(path).parts)


def _safe_member(data_root, relative, *, allow_locked=False):
    """Resolve paths before opening; prevent disguised locked or external files."""
    data_root = Path(data_root)
    base = data_root.resolve()
    path = data_root / relative
    resolved = path.resolve()
    if not resolved.is_relative_to(base):
        raise ForbiddenAccess("DATA_MEMBER_ESCAPES_DECLARED_ROOT")
    if not allow_locked and (_path_is_locked(data_root) or _path_is_locked(path) or
                             _path_is_locked(resolved)):
        raise ForbiddenAccess("A26_LOCKED_INPUT_CLOSED")
    return path


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ForbiddenAccess("UNREADABLE_DATA_PROVENANCE:" + Path(path).name) from exc
    if not isinstance(value, dict):
        raise ForbiddenAccess("OBJECT_DATA_PROVENANCE_REQUIRED")
    return value


def _parent_definition(role, parent_id):
    ids = SPLIT_PARENT_IDS[role]
    per_family = len(ids) // len(FAMILIES)
    offset = int(parent_id) - ids[0]
    family = FAMILIES[offset // per_family]
    contrast = role == "locked_ood" and offset % per_family >= 2
    ood = ("contrast" if contrast else "structural_combination") if role == "locked_ood" else "ID"
    return family, ood, (1.5, 1.8) if contrast else (.3, 1.2)


def _validate_manifest(manifest, geometry, role):
    if (manifest.get("manifest_id") != SOURCE_MANIFEST_ID or manifest.get("master_seed") != 251008 or
            manifest.get("families") != list(FAMILIES) or manifest.get("backgrounds") != ["weak", "ring"] or
            manifest.get("amplitudes") != [.3, 1.2] or manifest.get("noise_snr_db") != [None, 20]):
        raise ForbiddenAccess("A25_PUBLIC_MANIFEST_MISMATCH")
    declared = manifest.get("splits", {}).get(role)
    expected = list(SPLIT_PARENT_IDS[role])
    if declared != expected:
        raise ForbiddenAccess("FROZEN_PARENT_WHITELIST_MISMATCH")
    rows = [row for row in manifest.get("parents", []) if row.get("split") == role]
    if len(rows) != len(expected) or Counter(row.get("parent_id") for row in rows) != Counter(expected):
        raise ForbiddenAccess("PUBLIC_PARENT_ROWS_INCOMPLETE_OR_DUPLICATED")
    for row in rows:
        family, ood, amplitudes = _parent_definition(role, row["parent_id"])
        if row.get("family") != family or row.get("ood_type") != ood or row.get("amplitudes") != list(amplitudes):
            raise ForbiddenAccess("PUBLIC_PARENT_DEFINITION_MISMATCH")
    if (geometry.get("grid") != 24 or geometry.get("sources") != 8 or geometry.get("receivers") != 32 or
            geometry.get("source_axis") != -1 or geometry.get("precision") != "complex128/float64" or
            geometry.get("current_normalization") != "sqrt(cell_area)*physical_current" or
            geometry.get("volume") != 1. / 576):
        raise ForbiddenAccess("FROZEN_GEOMETRY_LAYOUT_MISMATCH")
    for key, shape in (("points", (576, 2)), ("source_points", (8, 2)), ("receiver_points", (32, 2))):
        value = np.asarray(geometry.get(key), dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ForbiddenAccess("FROZEN_GEOMETRY_POINTS_MISMATCH:" + key)
    if not np.isfinite(geometry.get("wavenumber", np.nan)) or geometry["wavenumber"] <= 0:
        raise ForbiddenAccess("FROZEN_WAVENUMBER_REQUIRED")


def _string_scalar(value, key):
    value = np.asarray(value)
    if value.shape != () or value.dtype.kind not in "US":
        raise ForbiddenAccess("SCALAR_STRING_METADATA_REQUIRED:" + key)
    return str(value.item())


def _validate_identity(metadata, role, *, complete):
    parents = np.asarray(metadata["parent_id"])
    splits = np.asarray(metadata["split"])
    if parents.ndim != 1 or parents.dtype != np.dtype("int64") or splits.shape != parents.shape:
        raise ForbiddenAccess("ALIGNED_INT64_PARENTS_AND_SPLITS_REQUIRED")
    if splits.dtype.kind not in "US" or not np.all(splits == role):
        raise ForbiddenAccess("DECLARED_SPLIT_METADATA_MISMATCH")
    if not len(parents) or not set(parents.tolist()).issubset(SPLIT_PARENT_IDS[role]):
        raise ForbiddenAccess("PARENT_OUTSIDE_FROZEN_WHITELIST")
    if complete and Counter(parents.tolist()) != Counter({pid: 8 for pid in SPLIT_PARENT_IDS[role]}):
        raise ForbiddenAccess("EXACT_EIGHT_OBSERVATIONS_PER_PARENT_REQUIRED")
    return len(parents)


def _validate_metadata(metadata, role, *, complete, geometry=None):
    m = _validate_identity(metadata, role, complete=complete)
    for key in ROW_METADATA:
        value = np.asarray(metadata[key])
        if value.shape != (m,):
            raise ForbiddenAccess("ALIGNED_ROW_METADATA_REQUIRED:" + key)
        if key not in {"parent_id", "amplitude", "noise_db"} and value.dtype.kind not in "US":
            raise ForbiddenAccess("STRING_ROW_METADATA_REQUIRED:" + key)
    if _string_scalar(metadata["manifest_id"], "manifest_id") != SOURCE_MANIFEST_ID:
        raise ForbiddenAccess("ARCHIVE_MANIFEST_ID_MISMATCH")
    try:
        descriptor = json.loads(_string_scalar(metadata["geometry_descriptor"], "geometry_descriptor"))
    except ValueError as exc:
        raise ForbiddenAccess("INVALID_ARCHIVE_GEOMETRY_DESCRIPTOR") from exc
    if not isinstance(descriptor, dict) or (geometry is not None and descriptor != geometry):
        raise ForbiddenAccess("ARCHIVE_GEOMETRY_DESCRIPTOR_MISMATCH")
    sigma = np.asarray(metadata["sigma"])
    if sigma.shape != (m,) or sigma.dtype != np.dtype("float64") or not np.isfinite(sigma).all() or np.any(sigma < 0):
        raise ValueError("NONNEGATIVE_FLOAT64_SIGMA_PER_OBSERVATION_REQUIRED")
    for key in ("amplitude", "noise_db"):
        if np.asarray(metadata[key]).dtype != np.dtype("float64"):
            raise ForbiddenAccess("FLOAT64_CONDITION_METADATA_REQUIRED:" + key)
    if np.any(metadata["status"] != "OK"):
        raise RuntimeError("FAILED_INITIALIZATIONS_RETAINED; do not filter parents")
    cases = list(map(str, metadata["case_id"]))
    if len(set(cases)) != m:
        raise ForbiddenAccess("DUPLICATED_OBSERVATION_CASE_ID")
    counts = Counter()
    for i, pid in enumerate(metadata["parent_id"]):
        family, ood, amplitudes = _parent_definition(role, pid)
        background = str(metadata["background_id"][i])
        if background not in {"weak", "ring"} or metadata["family"][i] != family or metadata["ood_type"][i] != ood:
            raise ForbiddenAccess("PARENT_FAMILY_BACKGROUND_OR_OOD_MISMATCH")
        amplitude = float(metadata["amplitude"][i])
        if amplitude not in amplitudes:
            raise ForbiddenAccess("FROZEN_AMPLITUDE_REQUIRED")
        ai = amplitudes.index(amplitude)
        db = float(metadata["noise_db"][i])
        ni = 0 if np.isnan(db) else 1
        if (ni == 0 and sigma[i] != 0) or (ni == 1 and (db != 20. or sigma[i] <= 0)):
            raise ForbiddenAccess("NOISE_GROUP_SIGMA_MISMATCH")
        case = f"p{pid}_{background}_a{ai}_" + ("clean" if ni == 0 else "snr20")
        if cases[i] != case or metadata["variant_id"][i] != f"{background}:a{ai}:n{ni}":
            raise ForbiddenAccess("OBSERVATION_CONDITION_ID_MISMATCH")
        counts[(int(pid), background, ai, ni)] += 1
    if complete and any(count != 1 for count in counts.values()):
        raise ForbiddenAccess("DUPLICATED_OR_MISSING_PARENT_CONDITION")
    return m


def _load_verified(data_root, role, *, allow_locked=False):
    manifest_path = _safe_member(data_root, "manifest.json", allow_locked=allow_locked)
    geometry_path = _safe_member(data_root, "geometry.json", allow_locked=allow_locked)
    archive_path = _safe_member(data_root, ("locked/" if role in LOCKED_ROLES else "") + role + ".npz",
                                allow_locked=allow_locked)
    manifest, geometry = _read_json(manifest_path), _read_json(geometry_path)
    _validate_manifest(manifest, geometry, role)
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) - ARCHIVE_KEYS or len(set(archive.files)) != len(archive.files):
            raise ForbiddenAccess("UNKNOWN_OR_DUPLICATED_DATASET_KEY")
        required = ROW_METADATA | SCALAR_METADATA | {"sigma", "data", "chi"}
        if required - set(archive.files):
            raise ForbiddenAccess("INCOMPLETE_TRAINER_ARCHIVE")
        # NPZ is lazy: these are the first array payloads read, never chi/current.
        arrays = {key: archive[key].copy() for key in ("split", "parent_id")}
        _validate_identity(arrays, role, complete=True)
        for key in sorted((ROW_METADATA | SCALAR_METADATA | {"sigma"}) - set(arrays)):
            arrays[key] = archive[key].copy()
        m = _validate_metadata(arrays, role, complete=True, geometry=geometry)
        arrays["data"] = archive["data"].copy()
        if arrays["data"].shape != (m, 32, 8) or arrays["data"].dtype != np.dtype("complex128") or not np.isfinite(arrays["data"]).all():
            raise ValueError("FINITE_COMPLEX128_M_32_8_OBSERVATIONS_REQUIRED")
        # Legacy TV initialization remains an explicit additional no-NN baseline.
        for key in sorted(BASELINE_KEYS & set(archive.files)):
            arrays[key] = archive[key].copy()
        if "initial_mass" in arrays and (arrays["initial_mass"].shape != (m, 1152) or
                arrays["initial_mass"].dtype != np.dtype("float64") or not np.isfinite(arrays["initial_mass"]).all()):
            raise ValueError("FINITE_FLOAT64_M_1152_LEGACY_INITIAL_MASS_REQUIRED")
        for key in ("lambda", "initial_wall"):
            if key in arrays and (arrays[key].shape != (m,) or arrays[key].dtype != np.dtype("float64") or
                                  not np.isfinite(arrays[key]).all() or np.any(arrays[key] < 0)):
                raise ValueError("NONNEGATIVE_ALIGNED_BASELINE_METADATA_REQUIRED:" + key)
        if "initializer_id" in arrays:
            _string_scalar(arrays["initializer_id"], "initializer_id")
        arrays["chi"] = archive["chi"].copy()
        if arrays["chi"].shape != (m, 576) or arrays["chi"].dtype != np.dtype("complex128") or not np.isfinite(arrays["chi"]).all():
            raise ValueError("FINITE_COMPLEX128_M_576_MATERIAL_LABELS_REQUIRED")
        # current, clean, and noise_seed are intentionally never indexed.
        arrays["noise_group"] = np.where(arrays["sigma"] == 0, "NOISE_FREE", "NOISY")
    return arrays


def load_split(data_root, role):
    """Return the complete allowed trainer bundle from existing A25 data only.

    train has 512 observations / 64 exact parents; validation has 128 / 16.
    chi is trainer-only complex128 [M,576]. Old initial_mass/lambda are optional
    baseline fields. Call online_from_split before handing a bundle to an encoder.
    """
    if role not in PRIMARY_ROLES:
        raise ForbiddenAccess("A26_PRIMARY_TRAIN_OR_VALIDATION_ONLY")
    return _load_verified(data_root, role)


def online_from_split(arrays):
    """Copy only observation/provenance arrays, excluding all truth and TV fields.

    Accepts a complete split or a selected batch. It never indexes excluded keys,
    including chi/current/clean/noise and legacy initial_mass/lambda/initial_wall.
    """
    missing = (ROW_METADATA | SCALAR_METADATA | {"data", "sigma"}) - set(arrays)
    if missing:
        raise ForbiddenAccess("INCOMPLETE_ONLINE_METADATA:" + ",".join(sorted(missing)))
    online = {key: np.asarray(arrays[key]).copy() for key in ONLINE_KEYS & set(arrays)}
    roles = set(map(str, np.atleast_1d(online["split"])))
    if len(roles) != 1 or not roles.issubset(PRIMARY_ROLES):
        raise ForbiddenAccess("A26_PRIMARY_ONLINE_SPLIT_REQUIRED")
    m = _validate_metadata(online, next(iter(roles)), complete=False)
    data = online["data"]
    if data.shape != (m, 32, 8) or data.dtype != np.dtype("complex128") or not np.isfinite(data).all():
        raise ValueError("FINITE_COMPLEX128_ONLINE_M_32_8_REQUIRED")
    group = np.where(online["sigma"] == 0, "NOISE_FREE", "NOISY")
    if "noise_group" in online and not np.array_equal(online["noise_group"], group):
        raise ForbiddenAccess("ONLINE_NOISE_GROUP_MISMATCH")
    online["noise_group"] = group
    return online


def validate_locked_authorization(freeze_record, gate_record):
    """Metadata-only parent verdict guard; this function opens no file.

    gate_record: protocol_id, primary_gate='PASS', repeat_gate='PASS',
    repeat_independently_confirmed=True, data_complete=True.
    freeze_record: protocol_id, status='FROZEN', model_selection_frozen=True,
    width in {16,32}, updates in the frozen candidates, and six checkpoint rows
    with method, seed, checkpoint, status='TRAINED_VALIDATION_SELECTED', width,
    completed_updates. Gate interpretation remains exclusively with parent Codex.
    """
    if not isinstance(freeze_record, dict) or not isinstance(gate_record, dict):
        raise ForbiddenAccess("EXPLICIT_A26_GATE_AND_FREEZE_RECORDS_REQUIRED")
    if (gate_record.get("protocol_id") != PROTOCOL_ID or gate_record.get("primary_gate") != "PASS" or
            gate_record.get("repeat_gate") != "PASS" or gate_record.get("repeat_independently_confirmed") is not True or
            gate_record.get("data_complete") is not True):
        raise ForbiddenAccess("A26_PRIMARY_AND_INDEPENDENT_REPEAT_GATES_REQUIRED")
    if (freeze_record.get("protocol_id") != PROTOCOL_ID or freeze_record.get("status") != "FROZEN" or
            freeze_record.get("model_selection_frozen") is not True):
        raise ForbiddenAccess("EXPLICIT_A26_SIX_CHECKPOINT_FREEZE_REQUIRED")
    width, updates = freeze_record.get("width"), freeze_record.get("updates")
    if width not in {16, 32} or updates not in {100, 250, 500, 1000, 2000}:
        raise ForbiddenAccess("FROZEN_WIDTH_AND_UPDATE_COUNT_REQUIRED")
    rows = freeze_record.get("checkpoints")
    if not isinstance(rows, list) or len(rows) != 6 or not all(isinstance(row, dict) for row in rows):
        raise ForbiddenAccess("SIX_CHECKPOINT_RECORDS_REQUIRED")
    required = {(method, seed) for method in ("N1", "N2", "N3") for seed in (261009, 261010)}
    if {(row.get("method"), row.get("seed")) for row in rows} != required:
        raise ForbiddenAccess("EXACT_PRIMARY_AND_REPEAT_METHOD_SET_REQUIRED")
    paths = []
    for row in rows:
        path = row.get("checkpoint")
        if (row.get("status") != "TRAINED_VALIDATION_SELECTED" or row.get("width") != width or
                row.get("completed_updates") != updates or not isinstance(path, str) or not path or
                "\0" in path or ".." in Path(path).parts):
            raise ForbiddenAccess("CHECKPOINT_FREEZE_RECORD_MISMATCH")
        paths.append(path)
    if len(set(paths)) != 6:
        raise ForbiddenAccess("SIX_DISTINCT_CHECKPOINTS_REQUIRED")
    return {"protocol_id": PROTOCOL_ID, "width": width, "updates": updates,
            "checkpoints": sorted(rows, key=lambda row: (row["seed"], row["method"])),
            "gate_record": dict(gate_record)}


def load_locked_split(data_root, role, *, a26_root, book, freeze_record, gate_record):
    """Future explicit once-per-cohort access after both parent scientific gates.

    This function is never called by primary loading, cache creation, or training.
    All claims live under A26. Reservations are consumed before any locked input
    read; a failed attempt remains consumed. ID/OOD each open once in one cohort.
    """
    authorization = validate_locked_authorization(freeze_record, gate_record)
    if role not in LOCKED_ROLES:
        raise ForbiddenAccess("EXPLICIT_LOCKED_ID_OR_OOD_ROLE_REQUIRED")
    a26_root = Path(a26_root).resolve()
    namespace = a26_root / "results/a26/jobs"
    if (not isinstance(book, BudgetLedger) or book.path is None or not book.path.resolve().is_relative_to(namespace) or
            not book.owner_id or not book._owned_locks):
        raise ForbiddenAccess("OWNED_A26_CHARGED_JOB_REQUIRED_FOR_LOCKED_OPEN")
    book.check()
    directory = a26_root / "results/a26/locked"
    with owned_file_lock(a26_root / "runs/a26-locked-opening.lock", owner_id=book.owner_id, job=book.metadata["job"]):
        cohort = directory / "LOCKED_COHORT_OPEN.json"
        if cohort.exists():
            if _read_json(cohort).get("authorization") != authorization:
                raise ForbiddenAccess("A26_LOCKED_COHORT_ALREADY_CONSUMED_BY_OTHER_FREEZE")
        else:
            book.reserve("a26_locked_cohort_open", 1)
            write_json(cohort, {"authorization": authorization, "status": "CONSUMED_UNDER_A26"})
        claim = directory / (role + "_OPEN.json")
        if claim.exists():
            raise ForbiddenAccess("A26_LOCKED_SPLIT_ALREADY_OPENED:" + role)
        book.reserve("a26_" + role + "_open", 1)
        write_json(claim, {"protocol_id": PROTOCOL_ID, "role": role, "job": book.metadata["job"],
                           "status": "OPENING_RESERVED_UNDER_A26"})
    return _load_verified(data_root, role, allow_locked=True)
