"""Factual A26 technical reports from existing, paid experiment artifacts.

``technical_reports(root, config, book, seed=261009)`` writes four root Markdown
files plus an aggregate ``COST_LEDGER.jsonl``. It opens only explicit primary or
repeat learning/evaluation paths, cache diagnostic JSON, A26 job receipts and
action ledgers. It never opens label, prediction, checkpoint or locked arrays.

``report_stage(root, config, book, *, seed=None, scientific_callback=None)``
also calls ``plots.render_figures`` under the same paid book. An optional parent
callback is called as ``callback(root, config, book)``. No scientific gate is
computed, and missing artifacts remain NOT_RUN.
"""
from __future__ import annotations

from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import statistics
import time
import uuid

from .budget import budget_history, plain


METHODS = ("N1", "N2", "N3")
FAMILIES = ("shell", "asymmetric_piecewise", "smooth", "mixed_scale")
FILES = ("A26_NETWORK_COMPARISON.md", "A26_PHYSICS_PRIOR_SPLIT.md",
         "A26_COST_AND_RUNTIME.md", "A26_EXPLORATION_LEDGER.md")


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _fmt(value):
    if value is None or value == "":
        return "NOT_RUN"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float):
        return f"{value:.8g}" if math.isfinite(value) else "UNDEFINED_RECORDED"
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(plain(value), ensure_ascii=False, sort_keys=True)
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _table(rows, columns):
    if not rows:
        return "NOT_RUN — required actual records are absent.\n"
    lines = ["| " + " | ".join(title for _, title in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(_fmt(row.get(key)) for key, _ in columns) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Sources:
    def __init__(self, root, book):
        self.root, self.book = Path(root), book
        self.records, self.issues = [], []

    def json(self, relative, expected=None):
        path = self.root / relative
        self.book.check()
        if not path.exists():
            self.records.append({"path": str(relative), "status": "NOT_RUN"})
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if expected is not None and not isinstance(value, expected):
                raise ValueError("ARTIFACT_TYPE_MISMATCH")
        except (OSError, ValueError) as error:
            record = {"path": str(relative), "status": "READ_FAILED", "error": type(error).__name__ + ": " + str(error)}
            self.records.append(record)
            self.issues.append(record)
            return None
        self.records.append({"path": str(relative), "status": "RECORDED",
                             "records": len(value) if isinstance(value, list) else None})
        return value

    def csv(self, relative):
        path = self.root / relative
        self.book.check()
        if not path.exists():
            self.records.append({"path": str(relative), "status": "NOT_RUN"})
            return []
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if any(None in row for row in rows):
                raise ValueError("MALFORMED_CSV_ROW")
        except (OSError, ValueError, csv.Error) as error:
            record = {"path": str(relative), "status": "READ_FAILED", "error": type(error).__name__ + ": " + str(error)}
            self.records.append(record)
            self.issues.append(record)
            return []
        self.records.append({"path": str(relative), "status": "RECORDED", "records": len(rows)})
        return rows


def _training_rows(sources, seed):
    rows, manifests = [], {}
    for method in METHODS:
        folder = f"results/a26/learning/{seed}/{method}"
        manifest = sources.json(folder + "/training_manifest.json", dict)
        losses = sources.json(folder + "/raw_loss.json", list)
        validation = sources.json(folder + "/validation_history.json", list)
        if losses is None:
            losses = sources.csv(folder + "/raw_loss.csv")
        manifest = manifest or {}
        manifests[method] = manifest
        recorded_losses = [row for row in losses or [] if isinstance(row, dict)]
        row = {"method": method, "status": manifest.get("status", "NOT_RUN"),
               "seed": manifest.get("seed"), "width": manifest.get("width"),
               "parameters": manifest.get("parameter_count"), "planned_updates": manifest.get("updates"),
               "scientific_updates": manifest.get("scientific_training_updates"),
               "completed_updates": manifest.get("completed_updates"),
               "loss_rows": len(recorded_losses) if recorded_losses else None,
               "first_loss": _number(recorded_losses[0].get("loss_total")) if recorded_losses else None,
               "last_loss": _number(recorded_losses[-1].get("loss_total")) if recorded_losses else None,
               "best_step": manifest.get("best_step"), "best_delta_nrmse": manifest.get("best_validation_delta_nrmse"),
               "validation_records": len(validation) if validation is not None else None,
               "best_checkpoint": (sources.root / folder / "best.pt").is_file(),
               "middle_checkpoint": (sources.root / folder / "middle.pt").is_file(),
               "final_checkpoint": (sources.root / folder / "final.pt").is_file(),
               "source": folder + "/training_manifest.json"}
        rows.append(row)
    return rows, manifests


def _diagnostic_rows(sources):
    rows, backgrounds, raw = [], [], {}
    for background in ("weak", "ring"):
        relative = f"results/a26/cache/{background}_BACKGROUND_DIAGNOSTICS.metrics.json"
        value = sources.json(relative, dict)
        backgrounds.append({"background": background, "status": "RECORDED" if value is not None else "NOT_RUN",
                            **{key: (value or {}).get(key) for key in (
                                "background_J_relative_error", "confidence_trace", "confidence_diagonal_min",
                                "confidence_diagonal_max", "confidence_interpretation", "observable_rank",
                                "Db_retained_relative_min", "source_commit", "run_source_commit")}, "source": relative})
    for role in ("train", "validation"):
        for method in METHODS:
            relative = f"results/a26/cache/{role}_{method}_DIAGNOSTICS.json"
            value = sources.json(relative, list)
            raw[(role, method)] = value
            counts, models, beta_status, recovery, feature_names = Counter(), set(), set(), set(), []
            for record in value or []:
                result = record.get("result", {}) if isinstance(record, dict) else {}
                if not isinstance(result, dict):
                    continue
                models.add(str(result.get("physical_model", "NOT_RECORDED")))
                beta_status.add(str(result.get("beta_status", "NOT_RECORDED")))
                if "canonical_current_recovered" in result:
                    recovery.add(result["canonical_current_recovered"])
                feature_names = result.get("feature_names", feature_names)
                local = result.get("physical_counts", {})
                if isinstance(local, dict):
                    counts.update({key: value for key, value in local.items() if isinstance(value, int) and not isinstance(value, bool)})
            rows.append({"split": role, "method": method, "status": "RECORDED" if value is not None else "NOT_RUN",
                         "batches": len(value) if value is not None else None, "images": counts.get("a26_encoded_images") if value is not None else None,
                         "physical_model": sorted(models) or None, "beta_status": sorted(beta_status) or None,
                         "current_recovered": sorted(recovery) if recovery else None,
                         "feature_names": feature_names or None, "counts": dict(counts) if value is not None else None,
                         "source": relative})
    return rows, backgrounds, raw


def _paired_rows(statistics_record):
    rows = []
    for key, pair in (statistics_record or {}).items():
        if not isinstance(pair, dict) or "candidate" not in pair:
            continue
        for scope in ("all_families", "complex_families"):
            record = pair.get(scope, {}).get("delta_nrmse", {})
            rows.append({"comparison": key, "scope": scope, "gain": record.get("point_estimate"),
                         "ci95": record.get("ci95"), "paired_parents": record.get("paired_parents"),
                         "complete": record.get("coverage_complete")})
        for family in FAMILIES:
            record = pair.get("per_family", {}).get(family, {}).get("delta_nrmse", {})
            rows.append({"comparison": key, "scope": family, "gain": record.get("point_estimate"),
                         "ci95": record.get("ci95"), "paired_parents": record.get("paired_parents"),
                         "complete": record.get("coverage_complete")})
    return rows


def _runtime_rows(warm, fresh, instrumented, process_cold):
    rows = []
    for scope, records, key in (("RESIDENT_WARM_COUNTED_NATIVE", warm, "full_inference_seconds"),
                                ("INSTRUMENTED_RESIDENT_WARM", instrumented, "full_inference_seconds"),
                                ("FRESH_ENCODER_CHECKPOINT_SAME_PROCESS", fresh, "full_inference_seconds")):
        for method in METHODS:
            selected = [row for row in records if row.get("method") == method]
            values = [_number(row.get(key)) for row in selected]
            values = [value for value in values if value is not None]
            rows.append({"scope": scope, "method": method, "samples": len(values) if values else None,
                         "median_seconds": statistics.median(values) if values else None,
                         "minimum_seconds": min(values) if values else None,
                         "maximum_seconds": max(values) if values else None,
                         "source_status": "RECORDED" if selected else "NOT_RUN"})
    for method in METHODS:
        cold = process_cold.get(method) or {}
        for record in cold.get("rows", []):
            rows.append({"scope": "ACTUAL_FRESH_INTERPRETER" if record.get("fresh_process") else "NEW_ENCODER_IN_ALREADY_STARTED_INTERPRETER",
                         "method": method, "samples": 1, "median_seconds": record.get("process_start_to_image_seconds")
                         if record.get("fresh_process") else record.get("new_encoder_seconds"),
                         "source_status": cold.get("status"), "background": record.get("background_id")})
        if not cold:
            rows.append({"scope": "ACTUAL_FRESH_INTERPRETER", "method": method, "source_status": "NOT_RUN"})
    return rows


def _cost_snapshot(root, config, book):
    history = budget_history(root)
    active = dict(plain(book.receipt()))
    active.update(status="ACTIVE_CUTOFF", terminal_receipt=False,
                  cutoff_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  cost_scope="CURRENT_JOB_PARTIAL_AT_REPORT_CUTOFF")
    receipts = history["receipts"]
    accepted_jobs = {row["job"] for row in receipts}
    included = active.get("job") not in accepted_jobs
    unresolved_other = [job for job in history["unresolved_jobs"] if job != active.get("job")]
    active_cpu = _number(active.get("charged_cpu_seconds")) if included else 0.
    active_gpu = _number(active.get("gpu_occupation_seconds")) if included else 0.
    known_cpu = history["cpu_seconds"] + active_cpu if active_cpu is not None else None
    known_gpu = history["gpu_seconds"] + active_gpu if active_gpu is not None else None
    return {"history": history, "active": active, "active_included_in_cutoff": included,
            "known_cpu_seconds_at_cutoff": known_cpu, "known_gpu_seconds_at_cutoff": known_gpu,
            "unresolved_other_jobs": unresolved_other,
            "remaining_cpu_at_cutoff": max(0., float(config.get("cpu_limit_seconds", 7200)) - known_cpu)
            if known_cpu is not None and not unresolved_other else None,
            "remaining_gpu_at_cutoff": max(0., float(config.get("gpu_limit_seconds", 14400)) - known_gpu)
            if known_gpu is not None and not unresolved_other else None}


def _aggregate_ledger(root, sources, cost, evaluation_failures):
    """Copy actual actions with provenance; nested action durations are not added."""
    accepted = {row["job"]: row for row in cost["history"]["receipts"]}
    records, action_count = [], 0
    for path in sorted((root / "results/a26/jobs").glob("*/COST_LEDGER.jsonl")):
        sources.book.check()
        job = path.parent.name
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line_number % 256 == 0:
                    sources.book.check()
                try:
                    action = json.loads(line)
                    if not isinstance(action, dict):
                        raise ValueError("ACTION_MUST_BE_JSON_OBJECT")
                except ValueError as error:
                    records.append({"record_type": "ACTION_LEDGER_READ_FAILURE", "source_file": str(path.relative_to(root)),
                                    "source_line": line_number, "job": job, "error": str(error)})
                    sources.issues.append({"path": str(path.relative_to(root)), "status": "READ_FAILED",
                                           "error": f"line {line_number}: {error}"})
                    continue
                records.append({"record_type": "ACTION", "source_file": str(path.relative_to(root)), "source_line": line_number,
                                "job": job, "accepted_terminal_receipt": job in accepted,
                                "receipt_status": accepted[job]["status"] if job in accepted else "ACTIVE_OR_UNRESOLVED",
                                "action": action})
                action_count += 1
    for path in sorted((root / "research/delegated").glob("a26*/*.jsonl")):
        sources.book.check()
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    action = json.loads(line)
                except ValueError as error:
                    records.append({"record_type": "WORKER_LEDGER_READ_FAILURE", "source_file": str(path.relative_to(root)),
                                    "source_line": line_number, "error": str(error)})
                    continue
                records.append({"record_type": "WORKER_COMMAND_EVIDENCE", "source_file": str(path.relative_to(root)),
                                "source_line": line_number, "budget_charge": "NOT_ADDED_SEPARATELY_TO_ACCEPTED_HISTORY",
                                "action": action})
    for receipt in cost["history"]["receipts"]:
        records.append({"record_type": "TERMINAL_JOB_RECEIPT", "job": receipt["job"],
                        "receipt_status": receipt["status"], "receipt": receipt})
    for index, failure in enumerate(evaluation_failures or []):
        records.append({"record_type": "EVALUATION_FAILURE", "source_file": sources.evaluation + "/FAILURES.json",
                        "record_index": index, "failure": failure})
    records.append({"record_type": "ACTIVE_CUTOFF", "terminal_receipt": False,
                    "included_as_partial_in_known_cost": cost["active_included_in_cutoff"], "receipt": cost["active"]})
    _atomic_text(root / "COST_LEDGER.jsonl", "\n".join(json.dumps(plain(row), ensure_ascii=False, allow_nan=False) for row in records))
    return {"path": "COST_LEDGER.jsonl", "records": len(records), "job_actions": action_count,
            "accepted_terminal_receipts": len(cost["history"]["receipts"]), "terminal": False}


def _source_footer(sources):
    return "\n## Artifact sources\n\n" + _table(sources.records, (("path", "Path"), ("status", "Source status"), ("records", "Records"))) + (
        "\n## Read failures\n\n" + _table(sources.issues, (("path", "Path"), ("status", "Status"), ("error", "Recorded error"))) if sources.issues else "")


def _network_report(seed, training, case_rows, parent_rows, family_rows, pairs, summary, gate_path):
    return f"""# A26 network comparison — seed {seed}

This report reproduces recorded training and exposed-validation metrics. It makes no scientific acceptance decision. The parent's decision artifact is `{gate_path}`. Missing outputs are NOT_RUN. Validation is historically exposed development evidence, never a blind holdout.

## Training and checkpoint records

""" + _table(training, (("method", "Arm"), ("status", "Recorded status"), ("width", "Width"), ("parameters", "Parameters"),
        ("planned_updates", "Planned U"), ("completed_updates", "Completed U"), ("scientific_updates", "Scientific updates"),
        ("loss_rows", "Loss rows"), ("first_loss", "First loss"), ("last_loss", "Last loss"), ("best_step", "Best step"),
        ("best_delta_nrmse", "Selected validation δ NRMSE"), ("best_checkpoint", "Best exists"),
        ("middle_checkpoint", "Middle exists"), ("final_checkpoint", "Final exists"))) + f"""
Checkpoint existence is a filesystem observation, not a checkpoint-reload or numerical-validity verdict. Loss values are copied from actual history; no interpolated curve is generated. Evaluation status: {_fmt((summary or {}).get('status'))}. Recorded case-metric rows: {len(case_rows) if case_rows else 'NOT_RUN'}.

## All four families

The frozen family list is shell, asymmetric_piecewise, smooth and mixed_scale. The tables below preserve incomplete groups and every available method/baseline row. The registered metric is equal-parent delta-material NRMSE; full-material NRMSE is separately reported.

""" + _table(family_rows, (("method", "Method"), ("family", "Family"), ("status", "Status"), ("complete", "Complete"),
        ("parent_count", "Parents"), ("expected_parent_count", "Expected parents"), ("delta_nrmse", "δ NRMSE"),
        ("full_nrmse", "Full χ NRMSE"), ("ssim_delta", "SSIM δ"), ("support_iou", "Support IoU"),
        ("edge_f1", "Edge F1"), ("full_data_relative_residual", "Full data residual"))) + "\n## Individual exposed parents\n\n" + _table(
        parent_rows, (("method", "Method"), ("parent_id", "Parent"), ("family", "Family"), ("status", "Status"),
        ("conditions", "Conditions"), ("complete", "Complete"), ("delta_nrmse", "δ NRMSE"),
        ("full_nrmse", "Full χ NRMSE"), ("real_nrmse", "Re NRMSE"), ("imag_nrmse", "Im NRMSE"),
        ("full_data_relative_residual", "Full data residual"))) + """
## Recorded paired-parent statistics

Positive relative gain in the source statistic means the candidate's error is smaller than its reference's error. Intervals and coverage flags are copied from the recorded 2,000-parent-resample artifact, including own-physical-base comparisons. They are development statistics and do not authorize expansion or a locked opening.

""" + _table(pairs, (("comparison", "Comparison"), ("scope", "Cohort/family"), ("gain", "Relative δ gain"),
                     ("ci95", "Recorded CI95"), ("paired_parents", "Paired parents"), ("complete", "Coverage complete")))


def _physics_report(diag_rows, backgrounds, diagnostics, receipts, manifests):
    training_receipts = [row for row in receipts if isinstance(row.get("counts"), dict)
                         and (_number(row["counts"].get("scientific_training_updates")) or 0) > 0]
    receipt_rows = []
    for row in training_receipts:
        counts = row["counts"]
        # Never infer an explicit zero for an absent physical counter.
        receipt_rows.append({"job": row["job"], "status": row["status"],
                             "scientific_updates": counts.get("scientific_training_updates"),
                             "autograd_backward_calls": counts.get("autograd_backward_calls"),
                             "physical_backward_calls": counts.get("physical_backward_calls", row.get("physical_backward_calls")),
                             "full_forward_calls": counts.get("full_forward_calls"),
                             "relative_forward_calls": counts.get("relative_forward_calls"),
                             "data_adjoint_directions": counts.get("data_adjoint_directions")})
    audit_rows = [{key: record.get(key) for key in ("phase", "method", "family", "parent_id", "background_id", "status",
        "forward_relative_error", "differential_forward_relative_error", "jvp_relative_error", "adjoint_relative_error",
        "gradient_relative_error", "gradient_cosine", "L_gradient_cosine", "boxed_update_relative_error",
        "reduced_adjoint_dot_defect", "full_backward_residual", "strict_A25_task_gate_pass", "core_margins", "error")}
        for record in diagnostics or [] if isinstance(record, dict)]
    for row in audit_rows:
        if row["status"] is None:
            row["status"] = "RECORDED"
    declared = [{"method": method, "manifest_physical_backward_calls": value.get("physical_backward_calls"),
                 "status": value.get("status", "NOT_RUN")} for method, value in manifests.items()]
    return """# A26 explicit physics and learned-prior split

The frozen encoder forms an explicit physical base from the observation and known background. The CNN adds a learned material increment to that base, followed by the common material box. This gives two separate evidence sources: the recorded encoder operations describe explicit physics, while training receipts and loss histories describe prior learning. Full nonlinear checks at predicted images are offline audits and are not labels or training-backward operations.

## Recorded encoder operations

""" + _table(diag_rows, (("split", "Split"), ("method", "Arm"), ("status", "Status"), ("batches", "Batches"),
        ("images", "Encoded images"), ("physical_model", "Physical model"), ("beta_status", "β status"),
        ("current_recovered", "Canonical current recovered"), ("counts", "Actual operation counters"))) + """
N1/N3 β is intentionally not computed; the controls do not run R solely to log β. N2's NOISE_FREE group has undefined β. These are feature-schema rules, not a measured advantage.

## Background confidence and observation operators

""" + _table(backgrounds, (("background", "Background"), ("status", "Status"), ("background_J_relative_error", "Cached J discrepancy"),
        ("observable_rank", "O rank"), ("Db_retained_relative_min", "Retained relative singular minimum"),
        ("confidence_trace", "Confidence trace"), ("confidence_diagonal_min", "Diagonal minimum"),
        ("confidence_diagonal_max", "Diagonal maximum"), ("confidence_interpretation", "Recorded interpretation"),
        ("source_commit", "Background source commit"), ("run_source_commit", "Run source commit"))) + """
The recorded confidence operator is numerical spectral confidence. Its values do not establish calibrated statistical recoverability. Background provenance and run provenance are shown separately without replacing either identifier.

## Training-backward evidence

""" + _table(receipt_rows, (("job", "Accepted training job"), ("status", "Receipt status"),
        ("scientific_updates", "Scientific updates"), ("autograd_backward_calls", "Autograd calls"),
        ("physical_backward_calls", "Explicit physical backward count"), ("full_forward_calls", "Explicit full forward count"),
        ("relative_forward_calls", "Explicit relative forward count"), ("data_adjoint_directions", "Explicit data adjoint count"))) + """
An absent physical counter is NOT_RUN/NOT_RECORDED in this table; it is not silently converted to zero. A zero training-physics statement requires an explicit accepted-receipt field. The following manifest declarations are preserved separately and are not substituted for that receipt evidence.

""" + _table(declared, (("method", "Arm"), ("status", "Manifest status"),
                        ("manifest_physical_backward_calls", "Manifest-declared physical backward count"))) + "\n## Recorded audits at predicted materials\n\n" + _table(
        audit_rows, (("phase", "Phase"), ("method", "Arm"), ("family", "Family"), ("parent_id", "Parent"),
        ("background_id", "Background"), ("status", "Recorded status"), ("forward_relative_error", "R/F forward error"),
        ("differential_forward_relative_error", "Scattered forward error"), ("jvp_relative_error", "JVP error"),
        ("adjoint_relative_error", "Adjoint error"), ("gradient_cosine", "R/F gradient cosine"),
        ("L_gradient_cosine", "L/F gradient cosine"), ("boxed_update_relative_error", "Boxed update error"),
        ("full_backward_residual", "Full solve residual"), ("strict_A25_task_gate_pass", "Source A25 check flag"),
        ("core_margins", "Core margins"), ("error", "Failure"))) + "\nThe source A25 check flag is a recorded diagnostic only. The parent owns its scientific interpretation and all A26 gate decisions.\n"


def _cost_report(config, cost, selected, benchmarks, runtime, runtime_rows, ledger):
    history, active = cost["history"], cost["active"]
    job_rows = [{"job": row["job"], "status": row["status"], "device": row["device"],
                 "wall": row.get("wall_seconds"), "process": row.get("process_cpu_seconds"),
                 "cpu": row.get("charged_cpu_seconds"), "gpu": row.get("gpu_occupation_seconds"),
                 "error": row.get("error")} for row in history["receipts"]]
    benchmark_rows = []
    for benchmark in benchmarks:
        for row in (benchmark or {}).get("rows", []):
            benchmark_rows.append({**row, "device": benchmark.get("device"), "benchmark_seed": benchmark.get("seed")})
    selected = selected or {}
    return f"""# A26 cost and runtime evidence

Resource accounting uses accepted terminal A26 job receipts. Nested span durations remain action evidence and are never summed into the budget. CPU jobs charge max(inclusive wall, process CPU); CUDA jobs charge process CPU and the entire occupied GPU wall. Failed, interrupted and budget-stopped attempts remain charged. Static-worker command evidence is retained in the aggregate ledger but is not added a second time to accepted totals.

## Resource snapshot

| Quantity | Recorded seconds/value |
| --- | --- |
| CPU limit | {_fmt(config.get('cpu_limit_seconds', 7200))} |
| GPU limit | {_fmt(config.get('gpu_limit_seconds', 14400))} |
| Accepted terminal CPU charge | {_fmt(history['cpu_seconds'])} |
| Accepted terminal GPU occupation | {_fmt(history['gpu_seconds'])} |
| Active job | {_fmt(active.get('job'))} |
| Active status | ACTIVE_CUTOFF — partial, nonterminal |
| Active cutoff UTC | {_fmt(active.get('cutoff_utc'))} |
| Active partial CPU charge | {_fmt(active.get('charged_cpu_seconds'))} |
| Active partial GPU occupation | {_fmt(active.get('gpu_occupation_seconds'))} |
| Known CPU through cutoff | {_fmt(cost['known_cpu_seconds_at_cutoff'])} |
| Known GPU through cutoff | {_fmt(cost['known_gpu_seconds_at_cutoff'])} |
| CPU allowance remaining at cutoff | {_fmt(cost['remaining_cpu_at_cutoff'])} |
| GPU allowance remaining at cutoff | {_fmt(cost['remaining_gpu_at_cutoff'])} |
| Other unresolved job tails | {_fmt(cost['unresolved_other_jobs'])} |
| Durable new-label reservations | {_fmt(history['reservations'].get('new_full_forward_labels', 0))} |

This cutoff is not the reporting job's terminal receipt. Report writing, figure rendering or later work after the sampled boundary remains outside this partial row until an accepted terminal receipt is available. An unknown interrupted tail keeps the total incomplete.

## Accepted attempts, including failures

""" + _table(job_rows, (("job", "Job"), ("status", "Status"), ("device", "Device"), ("wall", "Inclusive wall s"),
        ("process", "Process CPU s"), ("cpu", "CPU charge s"), ("gpu", "GPU occupation s"), ("error", "Recorded error"))) + f"""
## Cost-only architecture/update selection

Recorded status: {_fmt(selected.get('status'))}; width: {_fmt(selected.get('width'))}; U: {_fmt(selected.get('updates'))}; jobs: {_fmt(selected.get('complete_jobs'))}; basis: {_fmt(selected.get('selection_basis'))}. Safety factor: {_fmt(selected.get('safety_factor'))}. Final reserves from the selection: CPU {_fmt(selected.get('final_cpu_reserve_seconds'))} s, GPU {_fmt(selected.get('final_gpu_reserve_seconds'))} s. These are cost-admission records, not accuracy results.

""" + _table(benchmark_rows, (("method", "Arm"), ("width", "Width"), ("device", "Device"), ("benchmark_seed", "Seed"),
        ("warmup_updates", "Warm updates"), ("measured_updates", "Measured updates"), ("update_wall_seconds", "Max update wall s"),
        ("update_cpu_seconds", "Max update CPU s"), ("validation_case_count", "Full validation cases"),
        ("validation_wall_seconds", "Full validation wall s"), ("validation_cpu_seconds", "Full validation CPU s"))) + "\n" + _table(
        selected.get("candidates", []), (("updates", "Candidate U"), ("validation_events_per_arm", "Validations/arm"),
        ("projected_cpu_seconds", "Projected CPU s"), ("projected_gpu_seconds", "Projected GPU s"),
        ("required_cpu_with_reserve", "CPU with reserve s"), ("required_gpu_with_reserve", "GPU with reserve s"), ("fits", "Fits recorded allowance"))) + f"""
## Actual inference runtimes

Runtime status: {_fmt((runtime or {}).get('status'))}. The scope is raw measurement → encoder → transfer → CNN → material box. Offline full-physics audits are excluded from full-inference timing and remain paid separately. Cache-only CNN microbenchmarks above cannot establish full-inference speed. Fresh encoder/checkpoint timings in an existing process and actual fresh-interpreter records remain separate; OS disk cache is uncontrolled.

""" + _table(runtime_rows, (("scope", "Measured scope"), ("method", "Arm"), ("source_status", "Status"),
        ("samples", "Recorded samples"), ("background", "Background if specified"), ("median_seconds", "Median/individual s"),
        ("minimum_seconds", "Minimum s"), ("maximum_seconds", "Maximum s"))) + f"""
Aggregate action/failure ledger: `{ledger['path']}` ({ledger['records']} provenance-wrapped records; {ledger['job_actions']} job actions). Scientific quality/speed noninferiority, expansion and locked-opening decisions are owned by the parent.
"""


def _exploration_report(seed, training, selected, summary, runtime, cost, failures, gate_path):
    rows = [{"activity": "Frozen feature extraction", "hypothesis": "Separate explicit physics and observation features under the frozen protocol",
             "evidence": "results/a26/cache/{train,validation}_N*_DIAGNOSTICS.json", "change": "Frozen N1/N2/N3 feature definitions",
             "development": "TRAIN plus historically exposed validation", "result": "See recorded diagnostic coverage; missing entries are NOT_RUN",
             "decision": "Parent-owned GATE_DECISION.json", "cost": "Accepted feature-job receipts and action counters",
             "eligibility": "Development only; no locked arrays read by this report"},
            {"activity": "Actual TRAIN microbenchmark", "hypothesis": "Three mandatory arms fit the frozen resource allowance",
             "evidence": "results/a26/learning/BENCHMARK_WIDTH*.json; COST_SELECTED_UPDATES.json",
             "change": "Predeclared width32; width16 only after cost failure", "development": "One actual TRAIN batch, warm1 + measured10 updates/arm",
             "result": (selected or {}).get("status", "NOT_RUN"), "decision": (selected or {}).get("selection_basis", "NOT_RUN"),
             "cost": "Measured update and full128-row validation costs; safety1.3 plus reserves", "eligibility": "Cost admission only; benchmark weights discarded"}]
    for record in training:
        rows.append({"activity": f"{record['method']} training seed {seed}", "hypothesis": "Frozen learned increment may improve its own physical base",
                     "evidence": record["source"], "change": "Identical architecture, initialization, loss and parent/condition schedule",
                     "development": "64 TRAIN parents; checkpoint selection on all16 exposed validation parents",
                     "result": record["status"], "decision": "Checkpoint record only; parent reviews science",
                     "cost": f"Recorded scientific updates: {_fmt(record.get('scientific_updates'))}", "eligibility": "Exposed development; not blind confirmation"})
    rows.extend([{ "activity": "Independent predicted-image evaluation", "hypothesis": "Image metrics and exact full-physics residuals may expose model/prior failures",
                   "evidence": f"results/a26/evaluation/{seed}/EVALUATION_SUMMARY.json; PER_*_METRICS.csv; PHYSICS_DIAGNOSTICS.json",
                   "change": "Offline full audits at predictions; no new truth labels", "development": "All128 exposed observations; recorded typical-state diagnostics",
                   "result": (summary or {}).get("status", "NOT_RUN"), "decision": "Parent-owned GATE_DECISION.json", "cost": "Accepted evaluation receipts",
                   "eligibility": "Historically exposed validation; failures retained"},
                  {"activity": "Full inference runtime", "hypothesis": "Complete raw-measurement inference cost may differ across arms",
                   "evidence": f"results/a26/evaluation/{seed}/WARM_RUNTIME.csv; FRESH_RUNTIME.csv; PROCESS_COLD_N*.json",
                   "change": "Recorded warm, instrumented, same-process fresh and fresh-interpreter scopes",
                   "development": "Frozen metadata-selected cases; no quality-selected timing cases", "result": (runtime or {}).get("status", "NOT_RUN"),
                   "decision": "Parent owns quality/speed gate", "cost": "Actual timing plus separately paid offline audits",
                   "eligibility": "Runtime evidence only; OS disk cache uncontrolled"}])
    for receipt in cost["history"]["receipts"]:
        if receipt["status"] != "COMPLETE":
            rows.append({"activity": receipt["job"], "hypothesis": "Operational attempt as recorded by its job metadata",
                         "evidence": f"results/a26/jobs/{receipt['job']}/job_receipt.json", "change": receipt.get("stage", receipt.get("action")),
                         "development": "Accepted attempt receipt", "result": receipt["status"] + ": " + str(receipt.get("error")),
                         "decision": "Attempt terminated; no scientific inference",
                         "cost": f"CPU {_fmt(receipt.get('charged_cpu_seconds'))} s; GPU {_fmt(receipt.get('gpu_occupation_seconds'))} s",
                         "eligibility": "Failure is retained; retries do not erase its cost"})
    return f"""# A26 exploration ledger — seed {seed}

These are factual records of frozen requirements and available attempts. Hypotheses label the registered purpose of an activity; they are not findings. Every result and decision remains tied to its source. Parent gate artifact: `{gate_path}`. Missing primary/repeat stages are NOT_RUN, and this report never opens locked cohorts.

""" + _table(rows, (("activity", "Activity"), ("hypothesis", "Hypothesis/purpose"), ("evidence", "Evidence"),
        ("change", "Change"), ("development", "Development scope"), ("result", "Recorded result"),
        ("decision", "Decision owner/basis"), ("cost", "Cost evidence"), ("eligibility", "Validation eligibility"))) + "\n## Actual evaluation failures\n\n" + _table(
        [record for record in failures or [] if isinstance(record, dict)], (("method", "Method"), ("phase", "Phase"),
        ("case_id", "Case"), ("parent_id", "Parent"), ("family", "Family"), ("error", "Recorded error")))


def technical_reports(root, config, book, seed=261009):
    if book is None:
        raise ValueError("A26_PAID_REPORT_BOOK_REQUIRED")
    seed = int(seed)
    if seed not in {261009, 261010}:
        raise ValueError("A26_PRIMARY_OR_REPEAT_REPORT_SEED_REQUIRED")
    root = Path(root).resolve()
    sources = Sources(root, book)
    sources.evaluation = f"results/a26/evaluation/{seed}"
    evaluation = sources.evaluation
    with book.span("a26_technical_report_sources", a26_technical_report_source_reads=1):
        training, manifests = _training_rows(sources, seed)
        selected = sources.json("results/a26/learning/COST_SELECTED_UPDATES.json", dict)
        benchmarks = [sources.json(f"results/a26/learning/BENCHMARK_WIDTH{width}.json", dict) for width in (32, 16)]
        summary = sources.json(evaluation + "/EVALUATION_SUMMARY.json", dict)
        cases = sources.csv(evaluation + "/PER_CASE_METRICS.csv")
        parents = sources.csv(evaluation + "/PER_PARENT_METRICS.csv")
        families = sources.csv(evaluation + "/PER_FAMILY_METRICS.csv")
        paired = sources.json(evaluation + "/PAIRED_STATISTICS.json", dict)
        diagnostics = sources.json(evaluation + "/PHYSICS_DIAGNOSTICS.json", list)
        failures = sources.json(evaluation + "/FAILURES.json", list)
        runtime = sources.json(evaluation + "/RUNTIME_SUMMARY.json", dict)
        warm, fresh = sources.csv(evaluation + "/WARM_RUNTIME.csv"), sources.csv(evaluation + "/FRESH_RUNTIME.csv")
        instrumented = sources.csv(evaluation + "/INSTRUMENTED_RUNTIME.csv")
        process_cold = {method: sources.json(evaluation + f"/PROCESS_COLD_{method}.json", dict) for method in METHODS}
        diag_rows, backgrounds, _ = _diagnostic_rows(sources)
        cost = _cost_snapshot(root, config, book)
        ledger = _aggregate_ledger(root, sources, cost, failures)
    gate_path = evaluation + "/GATE_DECISION.json"
    documents = {
        FILES[0]: _network_report(seed, training, cases, parents, families, _paired_rows(paired), summary, gate_path),
        FILES[1]: _physics_report(diag_rows, backgrounds, diagnostics, cost["history"]["receipts"], manifests),
        FILES[2]: _cost_report(config, cost, selected, benchmarks, runtime, _runtime_rows(warm, fresh, instrumented, process_cold), ledger),
        FILES[3]: _exploration_report(seed, training, selected, summary, runtime, cost, failures, gate_path)}
    footer = _source_footer(sources)
    for filename, content in documents.items():
        book.check()
        with book.span("a26_technical_markdown_write", a26_technical_reports_written=1):
            _atomic_text(root / filename, content + footer)
    return {"status": "TECHNICAL_REPORTS_WRITTEN", "seed": seed, "reports": [str(root / name) for name in FILES],
            "cost_ledger": str(root / "COST_LEDGER.jsonl"), "cost_cutoff": cost["active"],
            "source_read_failures": sources.issues, "scientific_gate_decision_performed": False,
            "locked_arrays_opened": False}


def report_stage(root, config, book, *, seed=None, scientific_callback=None):
    """Paid technical artifacts plus actual plots, optionally a parent callback."""
    if book is None:
        raise ValueError("A26_PAID_REPORT_BOOK_REQUIRED")
    seed = int(config.get("primary_seed", 261009) if seed is None else seed)
    technical = technical_reports(root, config, book, seed=seed)
    from .plots import render_figures
    figures = render_figures(root, config, book, seed=seed)
    scientific = None
    if scientific_callback is not None:
        if not callable(scientific_callback):
            raise TypeError("A26_PARENT_SCIENTIFIC_CALLBACK_MUST_BE_CALLABLE")
        with book.span("a26_parent_scientific_report_callback"):
            scientific = scientific_callback(root, config, book)
    # Refresh the explicitly partial cutoff after the paid plot/callback work.
    technical = technical_reports(root, config, book, seed=seed)
    return {"status": "REPORT_STAGE_RECORDED", "seed": seed, "technical": technical,
            "figures": figures, "parent_scientific_callback": scientific,
            "scientific_gate_decision_performed_by_technical_reporter": False}
