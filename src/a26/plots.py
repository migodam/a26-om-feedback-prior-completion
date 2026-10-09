"""Metered figures drawn only from actual saved A26 evaluation/training artifacts.

There is no fixture, synthetic-image, invented-loss, or invented-runtime path.
Missing evidence yields a NOT_RUN entry. Figure generation is technical output;
visual review and scientific acceptance remain separate parent responsibilities.
"""
from __future__ import annotations

from pathlib import Path
import csv
import json
import math
import numpy as np

from .budget import BudgetExceeded, write_json
from .metrics import FAMILIES, VALIDATION_PARENTS

METHODS = ("N1", "N2", "N3")
COLORS = {"N1": "#2474B5", "N2": "#D86A28", "N3": "#2D986A"}
PAIRS = (("N2_vs_N1", "N2 / N1"), ("N2_vs_N3", "N2 / N3"), ("N1_vs_N3", "N1 / N3"))


def _number(value):
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _yes(value):
    return value in (True, 1, "True", "true", "1")


def _json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _csv(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader(); writer.writerows(rows)


def _prediction_bundle(path):
    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        required = {"N1", "N2", "N3", "BASE_N1", "BASE_N2", "truth", "case_id", "parent_id", "family", "sigma"}
        if required - set(archive.files):
            raise ValueError("INCOMPLETE_ACTUAL_PREDICTIONS_ARCHIVE")
        background_key = "background_id" if "background_id" in archive.files else "bg"
        if background_key not in archive.files:
            raise ValueError("ACTUAL_PREDICTION_BACKGROUND_METADATA_REQUIRED")
        keys = required | {background_key}
        keys |= {"BASE_N3", "COMMON_BP_INITIAL", "A25_CERTIFIED_TV_INITIAL_OFFLINE_CACHE"} & set(archive.files)
        result = {key: archive[key].copy() for key in keys}
    result["background_id"] = result[background_key]
    m = len(result["parent_id"])
    if m != 128 or set(map(int, result["parent_id"])) != set(VALIDATION_PARENTS):
        raise ValueError("COMPLETE_ACTUAL_EXPOSED_VALIDATION_PREDICTIONS_REQUIRED")
    for key in ("case_id", "family", "background_id", "sigma"):
        if result[key].shape != (m,):
            raise ValueError("ALIGNED_PREDICTION_METADATA_REQUIRED:" + key)
    for key in METHODS+("BASE_N1", "BASE_N2", "truth"):
        if result[key].shape != (m, 576) or result[key].dtype != np.dtype("complex128") or not np.isfinite(result[key]).all():
            raise ValueError("FINITE_ACTUAL_COMPLEX128_PREDICTIONS_REQUIRED:" + key)
    return result


def _representatives(bundle):
    """Frozen metadata-only first parent, both backgrounds, high amplitude/20 dB."""
    selected = []
    for family in FAMILIES:
        family_rows = np.flatnonzero(bundle["family"] == family)
        if not len(family_rows):
            raise ValueError("MISSING_RECONSTRUCTION_FAMILY:" + family)
        parent = min(int(bundle["parent_id"][index]) for index in family_rows)
        for background in ("weak", "ring"):
            choices = [int(index) for index in family_rows if int(bundle["parent_id"][index]) == parent and
                       str(bundle["background_id"][index]) == background and float(bundle["sigma"][index]) > 0 and
                       str(bundle["case_id"][index]).endswith("_a1_snr20")]
            if len(choices) != 1:
                raise ValueError("EXACT_FROZEN_RECONSTRUCTION_CONDITION_REQUIRED")
            selected.append(choices[0])
    return selected


def _reconstruction(plt, bundle, mode):
    if bundle is None:
        return None
    selected = _representatives(bundle)
    columns = [("truth", "Truth"), ("COMMON_BP_INITIAL", "Common BP initial"),
               ("BASE_N1", "L physical base"), ("BASE_N2", "R physical base"),
               ("N1", "N1: OM + L + CNN"), ("N2", "N2: OM + R + CNN"), ("N3", "N3: BP + L + CNN")]
    columns = [(key, label) for key, label in columns if key in bundle]
    if mode == "error":
        columns = [(key, label) for key, label in columns if key != "truth"]
    fig, axes = plt.subplots(len(selected), len(columns), figsize=(2.05*len(columns)+.8, 2.05*len(selected)),
                             squeeze=False, constrained_layout=True)
    selection = []
    for row, index in enumerate(selected):
        truth = bundle["truth"][index]
        arrays = [np.abs(bundle[key][index]-truth) if mode == "error" else
                  getattr(bundle[key][index], mode) for key, _ in columns]
        lo = 0. if mode == "error" else min(float(value.min()) for value in arrays)
        hi = max(float(value.max()) for value in arrays)
        if hi <= lo:
            hi = lo+1.
        for column, ((key, label), image) in enumerate(zip(columns, arrays)):
            axis = axes[row, column]
            shown = axis.imshow(image.reshape(24, 24), origin="lower", cmap="inferno" if mode == "error" else
                                "viridis" if mode == "real" else "magma", vmin=lo, vmax=hi, interpolation="nearest")
            axis.set_xticks([]); axis.set_yticks([])
            if row == 0:
                axis.set_title(label, fontsize=9)
            if column == 0:
                axis.set_ylabel(str(bundle["family"][index]).replace("_", " ")+"\n"+str(bundle["background_id"][index])+
                                f" / p{bundle['parent_id'][index]}", fontsize=8)
        fig.colorbar(shown, ax=list(axes[row]), fraction=.012, pad=.008)
        selection.append({"index": index, "case_id": str(bundle["case_id"][index]),
                          "parent_id": int(bundle["parent_id"][index]), "family": str(bundle["family"][index]),
                          "background_id": str(bundle["background_id"][index]), "sigma": float(bundle["sigma"][index]),
                          "color_min": lo, "color_max": hi})
    title = "Absolute complex material error" if mode == "error" else "Reconstructed full material: "+mode+" component"
    fig.suptitle(title+"\nExposed validation; first parent per family, amplitude 1.2, 20 dB; shared scale within each row", fontsize=12)
    return fig, selection, title+". Cases selected by frozen metadata, never reconstruction quality. Full material is shown; errors are relative to truth."


def _delta_reconstruction(plt, bundle, backgrounds, mode):
    if bundle is None or backgrounds is None:
        return None
    delta = dict(bundle)
    bg = np.stack([backgrounds[str(b)] for b in bundle['background_id']])
    for key in ('truth', 'COMMON_BP_INITIAL', 'BASE_N1', 'BASE_N2', 'BASE_N3', 'N1', 'N2', 'N3'):
        if key in bundle:
            delta[key] = bundle[key] - bg
    fig, rows, _ = _reconstruction(plt, delta, mode)
    fig.suptitle('Reconstructed material change: '+mode+' component\nExposed validation; frozen cases; known background subtracted; shared scale within each row', fontsize=12)
    return fig, rows, 'Actual material increment relative to the known background, with identical case selection and row-wise scales.'


def _loss(plt, learning):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    rows, methods_present = [], []
    positive = True
    for method in METHODS:
        folder = learning/method
        train, validation, manifest = _json(folder/"raw_loss.json"), _json(folder/"validation_history.json"), _json(folder/"training_manifest.json")
        manifest = manifest or {}
        tr = [row for row in train or [] if _number(row.get("step")) is not None and _number(row.get("loss_total")) is not None]
        va = [row for row in validation or [] if _number(row.get("step")) is not None and _number(row.get("parent_mean_delta_nrmse")) is not None]
        if not tr and not va:
            continue
        methods_present.append(method)
        label = method+(" (partial)" if manifest.get("status") != "TRAINED_EXPOSED_VALIDATION_SELECTED" else "")
        if tr:
            values = [float(row["loss_total"]) for row in tr]
            positive &= all(value>0 for value in values)
            axes[0].plot([int(row["step"]) for row in tr], values, color=COLORS[method], label=label, linewidth=1.1)
            rows.extend({"method": method, "kind": "training_total_loss", "step": int(row["step"]),
                         "value": float(row["loss_total"])} for row in tr)
        if va:
            axes[1].plot([int(row["step"]) for row in va], [float(row["parent_mean_delta_nrmse"]) for row in va],
                         marker="o", color=COLORS[method], label=label, linewidth=1.4)
            rows.extend({"method": method, "kind": "equal_parent_validation_delta_nrmse", "step": int(row["step"]),
                         "value": float(row["parent_mean_delta_nrmse"])} for row in va)
            best = [row for row in va if row["step"] == manifest.get("best_step")]
            if best:
                axes[1].scatter([best[0]["step"]], [best[0]["parent_mean_delta_nrmse"]], marker="*", s=110,
                                color=COLORS[method], zorder=5)
    if not methods_present:
        plt.close(fig); return None
    if positive:
        axes[0].set_yscale("log")
    axes[0].set_title("Actual training loss"); axes[0].set_ylabel("Normalized delta MSE + .05 (1 − SSIM)")
    axes[1].set_title("Actual exposed validation; star = saved best"); axes[1].set_ylabel("Equal-parent delta NRMSE")
    for axis in axes:
        axis.set_xlabel("Completed optimizer updates"); axis.grid(alpha=.2)
        if axis.lines:
            axis.legend(frameon=False)
    return fig, rows, "Saved scientific loss/validation histories only. Partial histories remain marked. No inferred or synthetic training points."


def _paired_family(plt, statistics):
    if not isinstance(statistics, dict):
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, constrained_layout=True)
    rows = []
    for axis, (key, title) in zip(axes, PAIRS):
        pair = statistics.get(key, {})
        for y, family in enumerate(FAMILIES):
            result = pair.get("per_family", {}).get(family, {}).get("delta_nrmse", {})
            value, ci = _number(result.get("point_estimate")), result.get("ci95")
            if value is None or not result.get("coverage_complete") or not isinstance(ci, list) or len(ci) != 2:
                continue
            lower, upper = _number(ci[0]), _number(ci[1])
            if lower is None or upper is None:
                continue
            color = COLORS[key.split("_vs_")[0]]
            axis.hlines(y, 100*lower, 100*upper, color=color, linewidth=2)
            axis.scatter([100*value], [y], color=color, s=36, zorder=3)
            rows.append({"comparison": key, "family": family, "relative_gain": value,
                         "ci95_lower": lower, "ci95_upper": upper, "paired_parents": result["paired_parents"]})
        axis.axvline(0, color="#777777", linestyle="--", linewidth=.8)
        axis.set_title(title); axis.set_xlabel("Paired mean relative delta-error gain (%)"); axis.grid(axis="x", alpha=.15)
    axes[0].set_yticks(range(4), [family.replace("_", " ") for family in FAMILIES]); axes[0].invert_yaxis()
    if not rows:
        plt.close(fig); return None
    fig.suptitle("Exposed-validation paired families: 2000 independent-parent resamples", fontsize=12)
    return fig, rows, "Four parents per family, conditions averaged first. Positive gain means smaller candidate delta NRMSE. Intervals are development uncertainty."


def _paired_complex(plt, statistics):
    if not isinstance(statistics, dict):
        return None
    comparisons = list(PAIRS)+[(method+"_vs_BASE_"+method, method+" / own physical base") for method in METHODS]
    fig, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    rows = []
    for y, (key, label) in enumerate(comparisons):
        result = statistics.get(key, {}).get("complex_families", {}).get("delta_nrmse", {})
        value, ci = _number(result.get("point_estimate")), result.get("ci95")
        if value is None or not result.get("coverage_complete") or not isinstance(ci, list) or len(ci) != 2:
            continue
        lower, upper = _number(ci[0]), _number(ci[1])
        if lower is None or upper is None:
            continue
        color = COLORS[key.split("_vs_")[0]]
        axis.hlines(y, 100*lower, 100*upper, color=color, linewidth=2)
        axis.scatter([100*value], [y], color=color, s=40)
        rows.append({"comparison": key, "relative_gain": value, "ci95_lower": lower,
                     "ci95_upper": upper, "paired_parents": result["paired_parents"]})
    if not rows:
        plt.close(fig); return None
    axis.axvline(0, color="#777777", linestyle="--", linewidth=.8)
    axis.set_yticks(range(len(comparisons)), [label for _, label in comparisons]); axis.invert_yaxis()
    axis.set_xlabel("Paired mean relative delta-error gain (%)"); axis.grid(axis="x", alpha=.15)
    axis.set_title("Complex families: shell, asymmetric piecewise, mixed scale\n12 independent exposed-validation parents; 2000 resamples")
    return fig, rows, "NN pair and NN/own-base gains. This graphic is descriptive development evidence and makes no gate decision."


def _quality(parent_rows):
    result = {}
    for method in METHODS:
        local = [row for row in parent_rows if row.get("method") == method]
        if len(local) != 16 or {int(row["parent_id"]) for row in local} != set(VALIDATION_PARENTS):
            continue
        values = [_number(row.get("delta_nrmse")) for row in local]
        if any(value is None for value in values) or any(not _yes(row.get("valid", True)) for row in local):
            continue
        result[method] = float(np.mean(values))
    return result


def _quality_runtime(plt, parent_rows, warm, fresh):
    quality = _quality(parent_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=True, constrained_layout=True)
    rows = []
    for axis, timing, scope in ((axes[0], warm, "Resident warm"),
                                (axes[1], fresh, "Fresh encoder + checkpoint\nSame process; OS cache uncontrolled")):
        for method in METHODS:
            local = [row for row in timing if row.get("method") == method and _number(row.get("full_inference_seconds")) is not None]
            if not local or method not in quality:
                continue
            elapsed = float(np.median([float(row["full_inference_seconds"]) for row in local]))
            if elapsed <= 0:
                continue
            axis.scatter([elapsed], [quality[method]], color=COLORS[method], s=75, label=method)
            axis.annotate(method, (elapsed, quality[method]), xytext=(5, 5), textcoords="offset points")
            rows.append({"method": method, "timing_scope": scope.replace("\n", "; "), "median_full_inference_seconds": elapsed,
                         "equal_parent_delta_nrmse": quality[method], "quality_parents": 16,
                         "timing_records": len(local), "decoder_only_timing_used": False})
        axis.set_title(scope); axis.set_xlabel("Measured full inference median (s)"); axis.grid(alpha=.2)
        if axis.collections:
            axis.margins(.2)
    if not rows:
        plt.close(fig); return None
    axes[0].set_ylabel("Equal-parent delta NRMSE (lower is better)")
    return fig, rows, "Quality uses all 16 validation parents. Runtime uses only the registered measured cases. Raw measurement→encoder→transfer→CNN→box; offline full audit excluded. Fresh timing is not a fresh-process or disk-cold claim."


def _runtime_components(plt, warm):
    components = (("encoder_seconds", "Physical encoder"), ("cnn_box_seconds", "CNN + box"),
                  ("image_transfer_seconds", "Image transfer"))
    rows, means = [], []
    for method in METHODS:
        local = [row for row in warm if row.get("method") == method]
        if not local or any(_number(row.get(key)) is None for row in local for key, _ in components):
            continue
        values = [float(np.mean([float(row[key]) for row in local])) for key, _ in components]
        means.append((method, values))
        rows.extend({"method": method, "component": key, "mean_seconds": value,
                     "records": len(local)} for (key, _), value in zip(components, values))
    if not means:
        return None
    fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    bottom = np.zeros(len(means))
    for index, (_, label) in enumerate(components):
        value = np.asarray([values[index] for _, values in means])
        axis.bar([method for method, _ in means], value, bottom=bottom, label=label)
        bottom += value
    axis.set_ylabel("Mean measured warm inference seconds"); axis.set_title("Actual complete inference components")
    axis.legend(frameon=False); axis.grid(axis="y", alpha=.15)
    return fig, rows, "Inclusive raw-to-image inference is the sum of its measured encoder, CNN/box and image-transfer components. Offline forward audits are separate."


def render_figures(root, config=None, book=None, *, seed=None):
    """Render actual root evaluation/learning artifacts and return a figure manifest.

    Outputs: results/a26/evaluation/<seed>/figures/*.png, *.svg, plot tables and
    FIGURE_MANIFEST.json. Call inside an A26 charged job with its book. Missing
    sources are skipped explicitly, never filled with synthesized result data.
    """
    if book is None:
        raise ValueError("A26_METERED_FIGURE_BOOK_REQUIRED")
    config = dict(config or {})
    seed = int(config.get("primary_seed", 261009) if seed is None else seed)
    if seed not in {261009, 261010}:
        raise ValueError("A26_PRIMARY_OR_REPEAT_FIGURE_SEED_REQUIRED")
    root = Path(root).resolve()
    evaluation = root/"results/a26/evaluation"/str(seed)
    learning = root/"results/a26/learning"/str(seed)
    destination = evaluation/"figures"
    destination.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    manifest = {"seed": seed, "status": "NOT_RUN", "source_scope": "ACTUAL_EXPOSED_VALIDATION_AND_TRAINING_ARTIFACTS",
                "scientific_acceptance": "PARENT_REVIEW_REQUIRED", "visual_review": "NOT_RUN",
                "figures": [], "skipped": [], "failures": []}
    try:
        with book.span("a26_plot_source_read", a26_actual_plot_source_reads=1):
            bundle = _prediction_bundle(evaluation/"PREDICTIONS.npz")
            backgrounds = {}
            for name in ('weak', 'ring'):
                path = root/'results/a26/repro_inputs/data/background'/(name+'.npz')
                if not path.is_file():
                    backgrounds = None
                    break
                with np.load(path, allow_pickle=False) as known:
                    backgrounds[name] = known['chi_b'].copy()
            statistics = _json(evaluation/"PAIRED_STATISTICS.json")
            parents = _csv(evaluation/"PER_PARENT_METRICS.csv")
            warm, fresh = _csv(evaluation/"WARM_RUNTIME.csv"), _csv(evaluation/"FRESH_RUNTIME.csv")
        definitions = [("reconstruction_real", lambda: _reconstruction(plt, bundle, "real")),
                       ("reconstruction_imag", lambda: _reconstruction(plt, bundle, "imag")),
                       ("reconstruction_delta_real", lambda: _delta_reconstruction(plt, bundle, backgrounds, "real")),
                       ("reconstruction_delta_imag", lambda: _delta_reconstruction(plt, bundle, backgrounds, "imag")),
                       ("reconstruction_error", lambda: _reconstruction(plt, bundle, "error")),
                       ("loss_curves", lambda: _loss(plt, learning)),
                       ("paired_family_gains", lambda: _paired_family(plt, statistics)),
                       ("paired_complex_and_own_base_gains", lambda: _paired_complex(plt, statistics)),
                       ("quality_vs_full_runtime", lambda: _quality_runtime(plt, parents, warm, fresh)),
                       ("warm_inference_components", lambda: _runtime_components(plt, warm))]
        for name, draw in definitions:
            book.check(); figure = None
            try:
                with book.span("a26_actual_figure_render", a26_plot_attempts=1):
                    result = draw()
                    if result is None:
                        manifest["skipped"].append({"figure": name, "status": "NOT_RUN", "reason": "REQUIRED_ACTUAL_SOURCE_MISSING_OR_INCOMPLETE"})
                        continue
                    figure, rows, caption = result
                    paths = []
                    for suffix in ("png", "svg"):
                        book.check()
                        path = destination/(name+"."+suffix)
                        figure.savefig(path, dpi=160, facecolor="white")
                        paths.append(str(path))
                    table = destination/(name+"_plot_data.csv")
                    _write_csv(table, rows)
                    manifest["figures"].append({"name": name, "files": paths, "plot_data": str(table), "caption": caption})
            except BudgetExceeded:
                raise
            except Exception as exc:
                manifest["failures"].append({"figure": name, "status": "FAILED", "error": type(exc).__name__+": "+str(exc)})
            finally:
                if figure is not None:
                    plt.close(figure)
        manifest["status"] = "RENDERED_AVAILABLE_ACTUAL_EVIDENCE" if manifest["figures"] else "NOT_RUN"
    except BaseException as exc:
        manifest["status"] = "BUDGET_STOP" if isinstance(exc, BudgetExceeded) else "FAILED"
        manifest["error"] = type(exc).__name__+": "+str(exc)
        raise
    finally:
        write_json(destination/"FIGURE_MANIFEST.json", manifest)
    return manifest


render = render_figures
generate_figures = render_figures
