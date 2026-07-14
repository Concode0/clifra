"""Benchmark run archives, comparisons, graphs, and documentation publishing."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from .models import BenchmarkConfig


def create_run_directory(config: BenchmarkConfig) -> Path:
    """Create one immutable timestamped run directory."""

    digest = hashlib.sha256(json.dumps(config.raw, sort_keys=True).encode()).hexdigest()[:10]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = config.output.root / f"{timestamp}-{digest}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "profiles").mkdir()
    (run_dir / "graphs").mkdir()
    return run_dir


def environment_metadata(config: BenchmarkConfig, run_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": config.schema_version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_dir.name,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
    }


def write_run_artifacts(
    run_dir: Path,
    *,
    config: BenchmarkConfig,
    measurements: list[dict[str, Any]],
    cumulative: list[dict[str, Any]],
    events: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write complete machine-readable artifacts and compact summaries."""

    metadata = environment_metadata(config, run_dir)
    states = Counter(row.get("status", "error") for row in measurements)
    event_states = Counter(row.get("status", "error") for row in events)
    summary = {
        "measurement_count": len(measurements),
        "cumulative_count": len(cumulative),
        "profile_count": len(profiles),
        "ok": states["ok"],
        "skipped": event_states["skipped"],
        "errors": event_states["error"],
    }
    metadata["summary"] = summary
    _write_json(run_dir / "config.json", config.raw)
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "profiles.json", profiles)
    for name, rows in (("measurements", measurements), ("cumulative", cumulative), ("events", events)):
        _write_jsonl(run_dir / f"{name}.jsonl", rows)
        _write_csv(run_dir / f"{name}.csv", rows)

    comparison = compare_with_baseline(measurements, config.output.baseline)
    _write_json(run_dir / "comparison.json", comparison)
    graph_paths = render_graphs(run_dir / "graphs", measurements, cumulative)
    graph_names = {path.name for path in graph_paths}
    (run_dir / "report.md").write_text(
        render_report(metadata, measurements, cumulative, events, comparison, graph_names=graph_names)
    )
    if config.output.publish:
        publish_documentation(run_dir, config.output.docs_root)
    return metadata


def compare_with_baseline(rows: list[dict[str, Any]], baseline: Path | None) -> dict[str, Any]:
    """Pair compatible rows with another modular benchmark archive."""

    if baseline is None:
        return {"compatible": False, "reason": "no baseline configured", "pairs": []}
    metadata_path = baseline / "metadata.json"
    rows_path = baseline / "measurements.jsonl"
    if not metadata_path.is_file() or not rows_path.is_file():
        return {"compatible": False, "reason": "baseline does not use the modular artifact schema", "pairs": []}
    metadata = json.loads(metadata_path.read_text())
    if int(metadata.get("schema_version", -1)) != 1:
        return {"compatible": False, "reason": "baseline schema version differs", "pairs": []}
    old_rows = _read_jsonl(rows_path)
    keys = ("sweep_id", "case_id", "signature", "device", "dtype", "batch", "compile_mode")
    old_index = {tuple(row.get(key) for key in keys): row for row in old_rows if row.get("status") == "ok"}
    pairs: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = tuple(row.get(column) for column in keys)
        old = old_index.get(key)
        if old is None:
            continue
        new_ms = _number(row.get("forward_median_ms"))
        old_ms = _number(old.get("forward_median_ms"))
        if new_ms is None or old_ms is None:
            continue
        pairs.append(
            {
                **{column: row.get(column) for column in keys},
                "old_forward_median_ms": old_ms,
                "new_forward_median_ms": new_ms,
                "change_percent": None if old_ms == 0.0 else (new_ms - old_ms) / old_ms * 100.0,
            }
        )
    return {"compatible": True, "reason": "", "pairs": pairs}


def render_graphs(graph_dir: Path, rows: list[dict[str, Any]], cumulative: list[dict[str, Any]]) -> list[Path]:
    """Generate operation-oriented benchmark graphs."""

    import matplotlib.pyplot as plt

    graph_dir.mkdir(parents=True, exist_ok=True)
    ok = [row for row in rows if row.get("status") == "ok"]
    paths: list[Path] = []
    graph_groups = (
        ("compact-products.png", "Compact products", "compact", {"product"}),
        (
            "compact-linear-metric.png",
            "Compact linear and metric operations",
            "compact",
            {"unary", "signature_norm", "pseudoscalar_product"},
        ),
        (
            "compact-exp-actions.png",
            "Compact exponentials and actions",
            "compact",
            {"bivector_exp", "sandwich_action", "versor_action", "multi_versor_action", "paired_bivector_action"},
        ),
        ("full-products.png", "Full-layout products", "full", {"product"}),
        (
            "full-linear-metric.png",
            "Full-layout linear and metric operations",
            "full",
            {"unary", "signature_norm", "pseudoscalar_product"},
        ),
        (
            "full-actions.png",
            "Full-layout actions",
            "full",
            {"bivector_exp", "sandwich_action", "versor_action", "multi_versor_action", "paired_bivector_action"},
        ),
    )
    for filename, title, layout, kinds in graph_groups:
        selected = [row for row in ok if row.get("layout_preset") == layout and row.get("kind") in kinds]
        paths.extend(_operation_latency_graph(plt, graph_dir / filename, selected, title=title))
    paths.extend(_exponential_routes_graph(plt, graph_dir / "compact-exponential-routes.png", ok))
    paths.extend(_startup_cost_graph(plt, graph_dir / "startup-costs.png", ok))
    paths.extend(_compiler_speedup_graph(plt, graph_dir / "compiler-speedup.png", ok))
    cumulative_ok = [row for row in cumulative if row.get("status") == "ok"]
    paths.extend(_cumulative_error_graph(plt, graph_dir / "cumulative-relative-error.png", cumulative_ok))
    return paths


def _operation_latency_graph(plt, path: Path, rows: list[dict[str, Any]], *, title: str) -> list[Path]:
    rows = _representative_rows(rows)
    case_ids = sorted({str(row.get("case_id", "")) for row in rows})
    if not case_ids:
        return []
    columns = min(3, len(case_ids))
    grid_rows = math.ceil(len(case_ids) / columns)
    figure, axes = plt.subplots(grid_rows, columns, figsize=(4.25 * columns, 3.05 * grid_rows), squeeze=False)
    modes = _display_compile_modes(rows)
    colors = {mode: plt.get_cmap("tab10")(index) for index, mode in enumerate(modes)}
    legend: dict[str, Any] = {}
    for axis, case_id in zip(axes.flat, case_ids, strict=False):
        case_rows = [row for row in rows if row.get("case_id") == case_id]
        for mode in modes:
            mode_rows = [row for row in case_rows if row.get("compile_mode") == mode]
            for metric, linestyle, marker, label_suffix in (
                ("forward_median_ms", "-", "o", "forward"),
                ("forward_backward_median_ms", "--", "s", "forward + backward"),
            ):
                points = _median_points(mode_rows, "n", metric)
                if not points:
                    continue
                label = f"{_compiler_label(mode)} · {label_suffix}"
                (line,) = axis.plot(
                    [x for x, _ in points],
                    [y for _, y in points],
                    color=colors[mode],
                    linestyle=linestyle,
                    marker=marker,
                    markersize=3.2,
                    linewidth=1.25,
                    label=label,
                )
                legend[label] = line
        axis.set_title(_case_label(case_id), fontsize=10)
        axis.set_yscale("log")
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("Dimension n")
        axis.set_ylabel("Median latency (ms)")
    for axis in axes.flat[len(case_ids) :]:
        axis.set_visible(False)
    figure.suptitle(f"{title}\nEuclidean · float32 · batch 1", fontsize=14)
    if legend:
        figure.legend(
            legend.values(),
            legend.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(4, len(legend)),
            frameon=False,
        )
    figure.tight_layout(rect=(0, 0.1, 1, 0.95))
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return [path]


def _exponential_routes_graph(plt, path: Path, rows: list[dict[str, Any]]) -> list[Path]:
    rows = [
        row
        for row in _representative_rows(rows, signature_family=None, compile_mode="eager")
        if row.get("layout_preset") == "compact" and row.get("exp_executor_family")
    ]
    if not rows:
        return []
    families = ("euclidean", "mixed", "degenerate")
    figure, axes = plt.subplots(1, len(families), figsize=(13.2, 3.8), squeeze=False)
    labels: dict[str, Any] = {}
    combinations = sorted({(str(row["case_id"]), str(row["exp_executor_family"])) for row in rows})
    colors = {
        case: plt.get_cmap("tab10")(index) for index, case in enumerate(sorted({case for case, _ in combinations}))
    }
    markers = ("o", "s", "^", "D", "v", "P")
    route_markers = {
        route: markers[index % len(markers)] for index, route in enumerate(sorted({r for _, r in combinations}))
    }
    for axis, family in zip(axes.flat, families, strict=True):
        family_rows = [row for row in rows if _signature_family(row) == family]
        for case_id, route in combinations:
            selected = [
                row for row in family_rows if row.get("case_id") == case_id and row.get("exp_executor_family") == route
            ]
            points = _median_points(selected, "n", "forward_median_ms")
            if not points:
                continue
            label = f"{_case_label(case_id)} · {route.replace('_', ' ')}"
            (line,) = axis.plot(
                [x for x, _ in points],
                [y for _, y in points],
                color=colors[case_id],
                marker=route_markers[route],
                linewidth=1.2,
                markersize=3.2,
                label=label,
            )
            labels[label] = line
        axis.set_title(family.capitalize())
        axis.set_xlabel("Dimension n")
        axis.set_ylabel("Eager forward median (ms)")
        axis.set_yscale("log")
        axis.grid(True, alpha=0.25)
    figure.suptitle("Compact exponential execution routes\nfloat32 · batch 1", fontsize=14)
    if labels:
        figure.legend(labels.values(), labels.keys(), loc="lower center", ncol=2, fontsize=8, frameon=False)
    figure.tight_layout(rect=(0, 0.18, 1, 0.92))
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return [path]


def _startup_cost_graph(plt, path: Path, rows: list[dict[str, Any]]) -> list[Path]:
    rows = _representative_rows(rows)
    groups = sorted({_operation_group(row) for row in rows})
    if not groups:
        return []
    columns = min(3, len(groups))
    grid_rows = math.ceil(len(groups) / columns)
    figure, axes = plt.subplots(grid_rows, columns, figsize=(4.25 * columns, 3.15 * grid_rows), squeeze=False)
    modes = _display_compile_modes(rows)
    legend: dict[str, Any] = {}
    for axis, group in zip(axes.flat, groups, strict=False):
        group_rows = [row for row in rows if _operation_group(row) == group]
        eager_rows = [row for row in group_rows if row.get("compile_mode") == "eager"]
        for selected, metric, label, color, marker in (
            (eager_rows, "plan_ms", "setup", "#4c78a8", "o"),
            (eager_rows, "cold_forward_ms", "eager cold call", "#59a14f", "s"),
        ):
            points = _median_points(selected, "n", metric)
            if points:
                (line,) = axis.plot(
                    [x for x, _ in points],
                    [y for _, y in points],
                    label=label,
                    color=color,
                    marker=marker,
                    markersize=3,
                )
                legend[label] = line
        for index, mode in enumerate(mode for mode in modes if mode != "eager"):
            selected = [row for row in group_rows if row.get("compile_mode") == mode]
            points = _median_points(selected, "n", "cold_forward_ms")
            if points:
                label = f"{_compiler_label(mode)} cold call"
                (line,) = axis.plot(
                    [x for x, _ in points],
                    [y for _, y in points],
                    label=label,
                    color=plt.get_cmap("tab10")(index + 1),
                    marker="^",
                    markersize=3,
                )
                legend[label] = line
        axis.set_title(group)
        axis.set_xlabel("Dimension n")
        axis.set_ylabel("Median across operations (ms)")
        axis.set_yscale("log")
        axis.grid(True, alpha=0.25)
    for axis in axes.flat[len(groups) :]:
        axis.set_visible(False)
    figure.suptitle("Setup and cold-call costs\nEuclidean · float32 · batch 1", fontsize=14)
    if legend:
        figure.legend(legend.values(), legend.keys(), loc="lower center", ncol=min(4, len(legend)), frameon=False)
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return [path]


def _compiler_speedup_graph(plt, path: Path, rows: list[dict[str, Any]]) -> list[Path]:
    if "inductor" not in _compile_modes(rows):
        return []
    compiled = "inductor"
    keys = ("sweep_id", "case_id", "signature", "device", "dtype", "batch")
    indexed: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        indexed[tuple(row.get(key) for key in keys)][str(row.get("compile_mode"))] = row
    grouped: dict[str, list[float]] = defaultdict(list)
    for pair in indexed.values():
        eager = pair.get("eager")
        target = pair.get(compiled)
        if eager is None or target is None:
            continue
        eager_ms = _number(eager.get("forward_median_ms"))
        target_ms = _number(target.get("forward_median_ms"))
        if eager_ms is not None and target_ms is not None and target_ms > 0.0:
            grouped[_operation_group(eager)].append(eager_ms / target_ms)
    if not grouped:
        return []
    labels = sorted(grouped)
    figure, axis = plt.subplots(figsize=(10.5, 4.6))
    axis.boxplot(
        [grouped[label] for label in labels],
        tick_labels=labels,
        showfliers=False,
        medianprops={"color": "#d62728", "linewidth": 1.5},
    )
    axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
    axis.set_yscale("log")
    axis.set_ylabel(f"Eager / {_compiler_label(compiled)} forward latency")
    axis.set_title(f"{_compiler_label(compiled)} steady-state speedup\nall measured signatures, dtypes, and batches")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return [path]


def _cumulative_error_graph(plt, path: Path, rows: list[dict[str, Any]]) -> list[Path]:
    rows = [row for row in rows if row.get("dtype") == "float32"]
    if not rows:
        return []
    families = ("euclidean", "mixed", "degenerate")
    figure, axes = plt.subplots(1, len(families), figsize=(12.8, 3.8), squeeze=False)
    for axis, family in zip(axes.flat, families, strict=True):
        family_rows = [row for row in rows if _signature_family(row) == family]
        by_step: dict[float, list[float]] = defaultdict(list)
        for row in family_rows:
            step = _number(row.get("step"))
            error = _number(row.get("max_rel_error"))
            if step is not None and error is not None and error > 0.0 and math.isfinite(error):
                by_step[step].append(error)
        steps = sorted(by_step)
        if steps:
            medians = [statistics.median(by_step[step]) for step in steps]
            q1 = [_percentile(by_step[step], 0.25) for step in steps]
            q3 = [_percentile(by_step[step], 0.75) for step in steps]
            axis.plot(steps, medians, color="#4c78a8", marker="o", markersize=3, label="median max relative error")
            axis.fill_between(steps, q1, q3, color="#4c78a8", alpha=0.2, label="interquartile range")
        axis.set_title(family.capitalize())
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Repeated steps")
        axis.set_ylabel("Relative error")
        axis.grid(True, alpha=0.25)
    figure.suptitle("Float32 repeated-action error against a float64 trajectory", fontsize=14)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 0.92))
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return [path]


def _representative_rows(
    rows: list[dict[str, Any]], *, signature_family: str | None = "euclidean", compile_mode: str | None = None
) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("dtype") == "float32" and int(row.get("batch", 0)) == 1]
    if signature_family is not None:
        selected = [row for row in selected if _signature_family(row) == signature_family]
    if compile_mode is not None:
        selected = [row for row in selected if row.get("compile_mode") == compile_mode]
    return selected


def _median_points(rows: list[dict[str, Any]], x_key: str, y_key: str) -> list[tuple[float, float]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        x = _number(row.get(x_key))
        y = _number(row.get(y_key))
        if x is not None and y is not None and y > 0.0 and math.isfinite(y):
            grouped[x].append(y)
    return [(x, statistics.median(grouped[x])) for x in sorted(grouped)]


def _operation_group(row: dict[str, Any]) -> str:
    layout = "Full" if row.get("layout_preset") == "full" else "Compact"
    kind = str(row.get("kind", ""))
    if kind == "product":
        family = "products"
    elif kind in {"unary", "signature_norm", "pseudoscalar_product"}:
        family = "linear / metric"
    else:
        family = "exp / actions"
    return f"{layout} · {family}"


def _signature_family(row: dict[str, Any]) -> str:
    p = int(row.get("p", 0))
    q = int(row.get("q", 0))
    r = int(row.get("r", 0))
    if p > 0 and q > 0:
        return "mixed"
    return "degenerate" if r > 0 else "euclidean"


def _compile_modes(rows: list[dict[str, Any]]) -> list[str]:
    modes = sorted({str(row.get("compile_mode", "")) for row in rows})
    return sorted(modes, key=lambda mode: (mode != "eager", mode))


def _display_compile_modes(rows: list[dict[str, Any]]) -> list[str]:
    available = set(_compile_modes(rows))
    return [mode for mode in ("eager", "inductor") if mode in available]


def _compiler_label(mode: str) -> str:
    return {"eager": "Eager", "inductor": "Inductor full graph"}.get(mode, mode.replace("_", " ").title())


def _case_label(case_id: str) -> str:
    return case_id.replace("_", " ").replace("anti commutator", "anticommutator").title()


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def render_report(
    metadata: dict[str, Any],
    measurements: list[dict[str, Any]],
    cumulative: list[dict[str, Any]],
    events: list[dict[str, Any]],
    comparison: dict[str, Any],
    *,
    graph_names: set[str],
) -> str:
    """Render a compact benchmark report."""

    summary = metadata["summary"]
    successful = [row for row in measurements if row.get("status") == "ok"]
    lines = [
        "# Benchmarks",
        "",
        "The suite measures eager forward, backward, startup, and cumulative behavior. `benchmarks/config.json` is the source of truth for the operation matrix and its dimension bounds.",
        "",
        "## Latest Published Run",
        "",
        f"Run `{metadata['run_id']}` on `{metadata['platform']}` with PyTorch `{metadata['torch_version']}` and `{metadata['torch_threads']}` Torch threads.",
        "",
        f"- successful rows: `{summary['ok']}`",
        f"- skipped cases: `{summary['skipped']}`",
        f"- errors: `{summary['errors']}`",
        f"- cumulative samples: `{len(cumulative)}`",
        "",
        "## Coverage",
        "",
        "| sweep | layout | dimensions | signatures | cases | dtypes | batches | compiler modes |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for sweep_id in sorted({str(row.get("sweep_id", "")) for row in measurements}):
        sweep_rows = [row for row in measurements if row.get("sweep_id") == sweep_id]
        dimensions = sorted({int(row["n"]) for row in sweep_rows if row.get("n") is not None})
        dimension_text = "" if not dimensions else f"{dimensions[0]}–{dimensions[-1]}"
        lines.append(
            "| "
            + " | ".join(
                (
                    sweep_id,
                    ", ".join(sorted({str(row.get("layout_preset", "")) for row in sweep_rows})),
                    dimension_text,
                    str(len({row.get("signature") for row in sweep_rows})),
                    str(len({row.get("case_id") for row in sweep_rows})),
                    ", ".join(sorted({str(row.get("dtype", "")) for row in sweep_rows})),
                    ", ".join(str(value) for value in sorted({int(row.get("batch", 0)) for row in sweep_rows})),
                    ", ".join(sorted({str(row.get("compile_mode", "")) for row in sweep_rows})),
                )
            )
            + " |"
        )
    lines.extend(["", "## Result Audit", ""])
    lines.extend(_result_audit(successful, cumulative))
    lines.extend(
        [
            "## Graphs",
            "",
            "Operation panels use Euclidean float32 with batch 1. Solid lines are forward latency; dashed lines are forward plus backward.",
            "",
            "### Compact Layout",
            "",
        ]
    )
    for name, caption in (
        ("compact-products.png", "Compact products"),
        ("compact-linear-metric.png", "Compact linear and metric operations"),
        ("compact-exp-actions.png", "Compact exponentials and actions"),
    ):
        if name in graph_names:
            lines.extend([f"![{caption}](graphs/{name})", ""])
    lines.extend(["### Full Layout", ""])
    for name, caption in (
        ("full-products.png", "Full-layout products"),
        ("full-linear-metric.png", "Full-layout linear and metric operations"),
        ("full-actions.png", "Full-layout actions"),
    ):
        if name in graph_names:
            lines.extend([f"![{caption}](graphs/{name})", ""])
    lines.extend(["### Execution Routes and Startup", ""])
    for name, caption in (
        ("compact-exponential-routes.png", "Compact exponential execution routes"),
        ("startup-costs.png", "Setup and cold-call costs"),
        ("compiler-speedup.png", "Compiled steady-state speedup"),
    ):
        if name in graph_names:
            lines.extend([f"![{caption}](graphs/{name})", ""])
    if "cumulative-relative-error.png" in graph_names:
        lines.extend(
            [
                "### Repeated-operation Numerics",
                "",
                "Float32 trajectories are compared with float64 trajectories.",
                "",
                "![Cumulative relative error](graphs/cumulative-relative-error.png)",
                "",
            ]
        )
    if events:
        lines.extend(["## Skips and Errors", "", "| state | case | stage | message |", "| --- | --- | --- | --- |"])
        for event in events[:20]:
            message = str(event.get("message", "")).replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {event.get('status', '')} | {event.get('case_id', '')} | {event.get('stage', '')} | {message} |"
            )
        if len(events) > 20:
            lines.append(f"|  |  |  | {len(events) - 20} more events in raw artifacts |")
        lines.append("")
    artifact_lines = [
        "## Reproduce",
        "",
        "```bash",
        "uv run --group benchmark benchmarks/run.py",
        "```",
        "",
        "See the [published configuration](artifacts/config.json) for the complete matrix.",
        "",
        "## Artifacts",
        "",
        "- [Configuration](artifacts/config.json)",
        "- [Environment metadata](artifacts/metadata.json)",
        "- [Summary](artifacts/summary.json)",
        "- [Measurements (JSONL)](artifacts/measurements.jsonl)",
        "- [Measurements (CSV)](artifacts/measurements.csv)",
        "- [Cumulative profiles (JSONL)](artifacts/cumulative.jsonl)",
        "- [Cumulative profiles (CSV)](artifacts/cumulative.csv)",
        "- [Events (JSONL)](artifacts/events.jsonl)",
        "- [Events (CSV)](artifacts/events.csv)",
    ]
    if summary["profile_count"]:
        artifact_lines.append("- [Profiler index](artifacts/profiles.json)")
    if comparison.get("pairs"):
        artifact_lines.append("- [Baseline comparison](artifacts/comparison.json)")
    artifact_lines.extend(
        [
            "",
            "JSONL contains raw samples and complete measurement fields. CSV contains the same rows in tabular form.",
            "",
        ]
    )
    lines.extend(artifact_lines)
    return "\n".join(lines)


def _result_audit(measurements: list[dict[str, Any]], cumulative: list[dict[str, Any]]) -> list[str]:
    output_nonfinite = sum((_number(row.get("output_finite_fraction")) or 0.0) < 1.0 for row in measurements)
    gradient_rows = [row for row in measurements if _number(row.get("gradient_finite_fraction")) is not None]
    gradient_nonfinite = sum((_number(row.get("gradient_finite_fraction")) or 0.0) < 1.0 for row in gradient_rows)
    noisy_forward = 0
    for row in measurements:
        median = _number(row.get("forward_median_ms"))
        spread = _number(row.get("forward_iqr_ms"))
        if median is not None and spread is not None and median > 0.0 and spread / median > 0.5:
            noisy_forward += 1
    lines = [
        f"- Output finiteness: `{len(measurements) - output_nonfinite}/{len(measurements)}` rows contain only finite values.",
        f"- Gradient finiteness: `{len(gradient_rows) - gradient_nonfinite}/{len(gradient_rows)}` measured gradients contain only finite values.",
        f"- Timing stability: `{noisy_forward}` rows have a forward IQR greater than 50% of their median. These are retained rather than silently filtered.",
    ]
    lines.extend(_compiler_audit(measurements))
    final_step = max((int(row.get("step", 0)) for row in cumulative), default=0)
    for family in ("euclidean", "mixed", "degenerate"):
        values = [
            _number(row.get("max_rel_error"))
            for row in cumulative
            if row.get("status") == "ok"
            and row.get("dtype") == "float32"
            and int(row.get("step", 0)) == final_step
            and _signature_family(row) == family
        ]
        finite = [value for value in values if value is not None and math.isfinite(value)]
        if finite:
            lines.append(
                f"- Repeated action, {family}: median float32 maximum relative error after `{final_step}` steps is `{statistics.median(finite):.3e}`."
            )
    lines.append("")
    return lines


def _compiler_audit(rows: list[dict[str, Any]]) -> list[str]:
    if "inductor" not in _compile_modes(rows):
        return []
    keys = ("sweep_id", "case_id", "signature", "device", "dtype", "batch")
    indexed: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        indexed[tuple(row.get(key) for key in keys)][str(row.get("compile_mode"))] = row
    lines: list[str] = []
    for mode in ("inductor",):
        speedups: list[float] = []
        for pair in indexed.values():
            eager = pair.get("eager")
            compiled = pair.get(mode)
            if eager is None or compiled is None:
                continue
            eager_ms = _number(eager.get("forward_median_ms"))
            compiled_ms = _number(compiled.get("forward_median_ms"))
            if eager_ms is not None and compiled_ms is not None and compiled_ms > 0.0:
                speedups.append(eager_ms / compiled_ms)
        if speedups:
            faster = sum(speedup > 1.0 for speedup in speedups) / len(speedups)
            lines.append(
                f"- {_compiler_label(mode)}: median steady-state forward speedup is `{statistics.median(speedups):.2f}×`; `{faster:.1%}` of paired rows are faster than eager."
            )
    return lines


def publish_documentation(run_dir: Path, docs_root: Path) -> None:
    """Atomically publish manifest-managed documentation outputs."""

    docs_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="clifra-benchmark-publish-", dir=docs_root.parent) as temporary:
        stage = Path(temporary)
        shutil.copy2(run_dir / "report.md", stage / "index.md")
        shutil.copytree(run_dir / "graphs", stage / "graphs")
        artifacts = stage / "artifacts"
        artifacts.mkdir()
        for name in (
            "config.json",
            "metadata.json",
            "summary.json",
            "measurements.jsonl",
            "measurements.csv",
            "cumulative.jsonl",
            "cumulative.csv",
            "events.jsonl",
            "events.csv",
            "profiles.json",
            "comparison.json",
        ):
            shutil.copy2(run_dir / name, artifacts / name)
        if any((run_dir / "profiles").iterdir()):
            shutil.copytree(run_dir / "profiles", artifacts / "profiles")
        generated = sorted(str(path.relative_to(stage)) for path in stage.rglob("*") if path.is_file())
        (stage / ".generated-manifest.json").write_text(json.dumps(generated, indent=2))
        generated.append(".generated-manifest.json")

        previous_manifest = docs_root / ".generated-manifest.json"
        previous = json.loads(previous_manifest.read_text()) if previous_manifest.is_file() else []
        for relative in generated:
            source = stage / relative
            destination = docs_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
        for relative in previous:
            if relative not in generated:
                stale = docs_root / relative
                if stale.is_file():
                    stale.unlink()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, default=_json_default) + "\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if columns:
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in columns})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, default=_json_default)
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None
