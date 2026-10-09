"""Offline A26 image diagnostics and equal-parent exposed-development statistics.

Truth-derived support and edge thresholds are used solely in these offline
metrics. They are never encoder inputs, features, proposals, or prior scales.
No physics, label generation, training, checkpoint selection, or gate decision
is performed here. Missing or undefined paired evidence stays incomplete.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import numpy as np

FAMILIES = ("shell", "asymmetric_piecewise", "smooth", "mixed_scale")
COMPLEX_FAMILIES = ("asymmetric_piecewise", "shell", "mixed_scale")
VALIDATION_PARENTS = tuple(range(250065, 250081))
METRICS = ("delta_nrmse", "full_nrmse", "real_nrmse", "imag_nrmse", "ssim_delta",
           "support_iou", "edge_f1", "passive", "feasible")


def _relative(error, reference, name):
    denominator = float(np.linalg.norm(reference))
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("UNDEFINED_ZERO_OR_NONFINITE_NRMSE_DENOMINATOR:" + name)
    value = float(np.linalg.norm(error) / denominator)
    if not math.isfinite(value):
        raise ValueError("NONFINITE_OFFLINE_METRIC:" + name)
    return value


def ssim_delta(predicted, target, *, grid=24):
    """Same 7x7 valid-window SSIM definition as training.ssim_delta.

    Population variances clamp to zero; the two fixed material ranges are 4.5
    (real) and .5 (imaginary), with C1=(.01*range)^2, C2=(.03*range)^2.
    NumPy evaluates the identical float64 formula without importing Torch.
    """
    grid = int(grid)
    p, t = np.asarray(predicted, np.complex128), np.asarray(target, np.complex128)
    if grid < 7 or p.size != grid*grid or p.shape != t.shape:
        raise ValueError("MATCHED_DELTA_GRID_AT_LEAST_7_REQUIRED")
    p = np.stack((p.real, p.imag)).reshape(2, grid, grid)
    t = np.stack((t.real, t.imag)).reshape(2, grid, grid)
    if not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("FINITE_COMPLEX_DELTA_SSIM_REQUIRED")
    def pool(value):
        return np.lib.stride_tricks.sliding_window_view(value, (7, 7), axis=(-2, -1)).mean(axis=(-2, -1))
    mp, mt = pool(p), pool(t)
    vp, vt = np.maximum(pool(p*p)-mp*mp, 0), np.maximum(pool(t*t)-mt*mt, 0)
    covariance = pool(p*t)-mp*mt
    ranges = np.asarray([4.5, .5], np.float64)[:, None, None]
    c1, c2 = (.01*ranges)**2, (.03*ranges)**2
    score = ((2*mp*mt+c1)*(2*covariance+c2) /
             ((mp*mp+mt*mt+c1)*(vp+vt+c2)))
    value = float(score.mean())
    if not math.isfinite(value):
        raise ValueError("NONFINITE_OFFLINE_SSIM")
    return value


def _boundary(mask, grid):
    """Exact 4-connected support boundary; exterior grid pixels are absent."""
    image = np.asarray(mask, bool).reshape(grid, grid)
    pad = np.pad(image, 1, constant_values=False)
    interior = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    return image & ~interior


def image_metrics(predchi, truthchi, bgchi, grid=24):
    """Return finite full-grid scalar diagnostics for one offline image.

    real_nrmse/imag_nrmse use delta material component denominators. A zero
    NRMSE reference is undefined and raises; no hidden denominator floor is
    inserted. Support threshold is .25*max(abs(truthchi-bgchi)); both predicted
    and true masks use that same offline-only threshold. edge_f1 compares their
    exact boundary pixels with no matching tolerance. Bounds are the frozen box.
    """
    grid = int(grid)
    values = [np.asarray(value, np.complex128) for value in (predchi, truthchi, bgchi)]
    if any(value.shape not in {(grid*grid,), (grid, grid)} for value in values):
        raise ValueError("MATCHED_FULL_GRID_OFFLINE_MATERIAL_REQUIRED")
    predicted, truth, background = [value.reshape(-1) for value in values]
    if not all(np.isfinite(value).all() for value in (predicted, truth, background)):
        raise ValueError("FINITE_OFFLINE_MATERIAL_REQUIRED")
    expected, estimate = truth-background, predicted-background
    error = estimate-expected
    threshold = .25*float(np.max(np.abs(expected)))
    actual_support, predicted_support = np.abs(expected)>threshold, np.abs(estimate)>threshold
    union = int(np.count_nonzero(actual_support | predicted_support))
    actual_edge, predicted_edge = _boundary(actual_support, grid), _boundary(predicted_support, grid)
    edge_count = int(actual_edge.sum()+predicted_edge.sum())
    passive = bool(np.all(predicted.imag >= -1e-10))
    feasible = bool(np.all((predicted.real >= -.5-1e-10) & (predicted.real <= 4.+1e-10) &
                           (predicted.imag >= -1e-10) & (predicted.imag <= .5+1e-10)))
    result = {"delta_nrmse": _relative(error, expected, "delta"),
              "full_nrmse": _relative(predicted-truth, truth, "full"),
              "real_nrmse": _relative(error.real, expected.real, "real_delta"),
              "imag_nrmse": _relative(error.imag, expected.imag, "imag_delta"),
              "ssim_delta": ssim_delta(estimate, expected, grid=grid),
              "support_iou": float(np.count_nonzero(actual_support & predicted_support)/union) if union else 1.,
              "edge_f1": float(2*np.count_nonzero(actual_edge & predicted_edge)/edge_count) if edge_count else 1.,
              "passive": passive, "feasible": feasible}
    if not all(math.isfinite(float(value)) for value in result.values()):
        raise ValueError("NONFINITE_OFFLINE_IMAGE_METRICS")
    return result


def _number(value):
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _ok(row):
    return (row.get("valid", True) in (True, 1, "True", "true", "1") and
            row.get("complete", True) in (True, 1, "True", "true", "1") and
            row.get("status", "OK") in {"OK", "COMPLETE"})


def _keys(value):
    result = [value] if isinstance(value, str) else list(value)
    if len(set(result)) != len(result) or not all(isinstance(key, str) and key for key in result):
        raise ValueError("DISTINCT_NONEMPTY_COLUMN_NAMES_REQUIRED")
    return result


def _parent_family(parent):
    return FAMILIES[(int(parent)-250065)//4] if int(parent) in VALIDATION_PARENTS else None


def _expected_parents(group, by):
    present = {int(row["parent_id"]) for row in group}
    if not present:
        return []
    if "parent_id" in by:
        return sorted(present)
    if present.issubset(VALIDATION_PARENTS):
        families = {str(row["family"]) for row in group} if "family" in by else set(FAMILIES)
        return [parent for parent in VALIDATION_PARENTS if _parent_family(parent) in families]
    # Generic non-campaign rows can use this helper; the A26 scientific paired
    # statistics below always require the complete explicit validation whitelist.
    return sorted(present)


def aggregate_rows(rows, metrics, by):
    """Average conditions inside each independent parent, then weight parents equally.

    Raw A26 parent rows require eight distinct case IDs, or the frozen subset
    implied by grouping background/amplitude/noise. Calling this again on output
    aggregates parent rows directly. For recognized exposed validation IDs, the
    complete 16-parent or four-parent-per-family whitelist is required.
    Incomplete groups have metric=None plus descriptive-only available means.
    """
    metrics, by = _keys(metrics), _keys(by)
    grouped = defaultdict(list)
    for row in rows:
        if "parent_id" not in row or any(key not in row for key in by):
            raise ValueError("PARENT_AND_GROUP_COLUMNS_REQUIRED")
        grouped[tuple(row[key] for key in by)].append(row)
    output = []
    for key, group in grouped.items():
        result = dict(zip(by, key))
        parents = defaultdict(list)
        for row in group:
            parents[int(row["parent_id"])].append(row)
        expected = _expected_parents(group, by)
        expected_conditions = 8
        for names in ({"background_id"}, {"amplitude"}, {"noise_db", "noise_group"}):
            if names & set(by):
                expected_conditions //= 2
        parent_values, failures = {}, {}
        for parent in expected:
            local = parents.get(parent, [])
            aggregated = bool(local) and all(row.get("aggregation") == "parent_equal" for row in local)
            identifiers = [row.get("case_id") for row in local]
            duplicate = not aggregated and (any(case is None for case in identifiers) or len(set(identifiers)) != len(local))
            bad_count = len(local) != (1 if aggregated else expected_conditions)
            bad_family = (parent in VALIDATION_PARENTS and any(row.get("family") != _parent_family(parent) for row in local))
            numeric = {metric: [_number(row.get(metric)) for row in local] for metric in metrics}
            if not local or bad_count or duplicate or bad_family or any(not _ok(row) for row in local) or any(
                    value is None for values in numeric.values() for value in values):
                failures[parent] = ("MISSING_PARENT" if not local else "MISSING_OR_DUPLICATED_CONDITIONS" if bad_count or duplicate
                                    else "FAMILY_MISMATCH" if bad_family else "FAILED_OR_UNDEFINED_METRIC")
                parent_values[parent] = None
            else:
                parent_values[parent] = {metric: float(np.mean(values)) for metric, values in numeric.items()}
        complete = bool(expected) and not failures
        available = [value for value in parent_values.values() if value is not None]
        descriptive = {metric: float(np.mean([value[metric] for value in available])) if available else None for metric in metrics}
        observations = sum(int(row.get("observations", 1)) if row.get("aggregation") == "parent_equal" else 1 for row in group)
        result.update(aggregation="parent_equal", valid=complete, complete=complete, status="OK" if complete else "INCOMPLETE",
                      parent_count=len(available), expected_parent_count=len(expected), observations=observations,
                      conditions=observations if "parent_id" in by else expected_conditions,
                      failed_parents=failures, parent_values=parent_values,
                      available_parent_descriptive_mean=descriptive)
        result.update({metric: descriptive[metric] if complete else None for metric in metrics})
        output.append(result)
    return output


def _paired(left, right, parents, metric, seed):
    gains, undefined = [], []
    for parent in parents:
        candidate, reference = left.get(parent, []), right.get(parent, [])
        if len(candidate) != 1 or len(reference) != 1:
            undefined.append(parent); continue
        a, b = candidate[0], reference[0]
        av, bv = _number(a.get(metric)), _number(b.get(metric))
        if (not _ok(a) or not _ok(b) or av is None or bv is None or bv <= 0 or
                a.get("family") != _parent_family(parent) or b.get("family") != _parent_family(parent)):
            undefined.append(parent); continue
        gain = (bv-av)/bv
        if not math.isfinite(gain):
            undefined.append(parent); continue
        gains.append((parent, gain))
    result = {"metric": metric, "higher_is_better": False, "relative": True,
              "samples": 2000, "seed": int(seed), "expected_parents": len(parents), "paired_parents": len(gains),
              "undefined_or_failed_parents": undefined, "coverage_complete": bool(parents) and not undefined,
              "parent_gains": [{"parent_id": parent, "gain": gain} for parent, gain in gains],
              "point_estimate": None, "ci95": None}
    if result["coverage_complete"]:
        values = np.asarray([gain for _, gain in gains], np.float64)
        rng = np.random.default_rng(int(seed))
        sampled = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
        result.update(point_estimate=float(values.mean()), ci95=[float(value) for value in np.quantile(sampled, [.025, .975])])
    return result


def paired_statistics(parentrows, config):
    """Fixed 2000-resample paired-parent development statistics, no gate verdict.

    One input row per method/parent is required. Positive relative gain means
    smaller candidate error than its paired reference. Main comparisons are
    N2/N1, N2/N3, N1/N3 and each NN against its own explicit physical BASE_N*.
    Complex statistics use exactly asymmetric_piecewise, shell, mixed_scale.
    """
    if int(config.get("bootstrap_samples", 2000)) != 2000:
        raise ValueError("A26_FROZEN_2000_PARENT_BOOTSTRAP_REQUIRED")
    families = tuple(config.get("complex_families", COMPLEX_FAMILIES))
    if len(families) != 3 or set(families) != set(COMPLEX_FAMILIES):
        raise ValueError("A26_FROZEN_COMPLEX_FAMILY_SET_REQUIRED")
    methods = defaultdict(lambda: defaultdict(list))
    unexpected = set()
    for row in parentrows:
        parent = int(row["parent_id"])
        if parent not in VALIDATION_PARENTS:
            unexpected.add(parent)
        methods[str(row["method"])][parent].append(row)
    pairs = (("N2", "N1"), ("N2", "N3"), ("N1", "N3"),
             ("N1", "BASE_N1"), ("N2", "BASE_N2"), ("N3", "BASE_N3"))
    numeric_metrics = ("delta_nrmse", "full_nrmse", "real_nrmse", "imag_nrmse", "full_data_relative_residual")
    seed = int(config.get("bootstrap_seed", config.get("primary_seed", 261009)))
    output = {"samples": 2000, "seed": seed, "unit": "independent_parent", "exposure": "EXPOSED_DEVELOPMENT_VALIDATION",
              "holdout": False, "complex_families": list(families), "expected_parent_ids": list(VALIDATION_PARENTS),
              "unexpected_parent_ids": sorted(unexpected), "scientific_gate_decision_performed": False}
    for index, (candidate, reference) in enumerate(pairs):
        groups = {"all_families": list(VALIDATION_PARENTS),
                  "complex_families": [parent for parent in VALIDATION_PARENTS if _parent_family(parent) in families]}
        pair = {"candidate": candidate, "reference": reference, "all_families": {}, "complex_families": {}, "per_family": {}}
        for group_index, (name, parents) in enumerate(groups.items()):
            for metric in numeric_metrics:
                statistic = _paired(methods[candidate], methods[reference], parents, metric, seed+100*index+group_index)
                if unexpected:
                    statistic.update(coverage_complete=False, point_estimate=None, ci95=None)
                pair[name][metric] = statistic
        for family_index, family in enumerate(FAMILIES):
            parents = [parent for parent in VALIDATION_PARENTS if _parent_family(parent) == family]
            pair["per_family"][family] = {metric: _paired(methods[candidate], methods[reference], parents, metric,
                                                       seed+100*index+10+family_index) for metric in numeric_metrics}
            if unexpected:
                for statistic in pair["per_family"][family].values():
                    statistic.update(coverage_complete=False, point_estimate=None, ci95=None)
        complex_delta = pair["complex_families"]["delta_nrmse"]
        family_gains = {family: pair["per_family"][family]["delta_nrmse"]["point_estimate"] for family in FAMILIES}
        pair.update(complex_mean_gain=complex_delta["point_estimate"], complex_ci95=complex_delta["ci95"],
                    complex_coverage_complete=complex_delta["coverage_complete"], family_mean_gains=family_gains,
                    positive_complex_families=sum(family_gains[family] is not None and family_gains[family]>0 for family in families))
        output[candidate+"_vs_"+reference] = pair
    return output
