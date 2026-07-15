"""Execute the configured benchmark matrix."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import torch

from clifra.core.foundation.device import FLOAT_DTYPES, dtype_name, resolve_device

from .cases import PreflightSkip, build_case, make_benchmark_algebra, preflight_case
from .config import case_applies, expand_signatures
from .metrics import (
    case_numeric_profile,
    compile_callable,
    differentiable_args,
    error_stats,
    logarithmic_steps,
    measure_backward,
    measure_forward,
    scalar_loss,
    timed_call,
)
from .models import BenchmarkConfig, PreparedCase, SignatureSpec, SweepConfig


def run_suite(
    config: BenchmarkConfig,
    run_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run performance, cumulative, and optional profiler measurements."""

    measurements: list[dict[str, Any]] = []
    cumulative_rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for sweep in config.sweeps:
        signatures = expand_signatures(sweep)
        for requested_device in sweep.devices:
            device = str(resolve_device(requested_device))
            for spec in signatures:
                for dtype_declaration in sweep.dtypes:
                    dtype = FLOAT_DTYPES[dtype_declaration]
                    for batch in sweep.batch_sizes:
                        for case_index, case in enumerate(sweep.cases):
                            if not case_applies(case, spec):
                                continue
                            base = base_identity(case, sweep, spec, device=device, dtype=dtype, batch=batch)
                            seed = config.seed + spec.n * 1009 + batch * 37 + case_index * 17
                            try:
                                feasible = preflight_case(
                                    case,
                                    spec,
                                    batch=batch,
                                    channels=sweep.channels,
                                    actions=sweep.actions,
                                    pairs=sweep.pairs,
                                    dtype=dtype,
                                    device=device,
                                    resources=config.resources,
                                )
                            except PreflightSkip as exc:
                                events.append(
                                    event_row(
                                        base, status="skipped", stage="preflight", exc=exc, include_traceback=False
                                    )
                                )
                                print_event(base, status="SKIP", stage="preflight", message=str(exc))
                                continue

                            try:
                                algebra, algebra_ms = timed_call(
                                    lambda: make_benchmark_algebra(spec, device=device, dtype=dtype),
                                    device=device,
                                )
                                prepared, plan_ms = timed_call(
                                    lambda: build_case(
                                        algebra,
                                        case,
                                        feasible,
                                        batch=batch,
                                        channels=sweep.channels,
                                        actions=sweep.actions,
                                        pairs=sweep.pairs,
                                        device=device,
                                        dtype=dtype,
                                        seed=seed,
                                    ),
                                    device=device,
                                )
                            except Exception as exc:  # benchmark continues after recording a generic error.
                                for mode in sweep.compile_modes:
                                    row = {**base, "compile_mode": mode, "status": "error"}
                                    measurements.append(row)
                                    events.append(event_row(row, status="error", stage="planning", exc=exc))
                                print_event(base, status="ERROR", stage="planning", message=str(exc))
                                continue

                            for mode in sweep.compile_modes:
                                row = {
                                    **base,
                                    "compile_mode": mode,
                                    "status": "ok",
                                    "algebra_ms": algebra_ms,
                                    "plan_ms": plan_ms,
                                    **prepared.metadata,
                                }
                                try:
                                    _reset_compile_cache(mode)
                                    callable_target, compile_wrap_ms = timed_call(
                                        lambda: compile_callable(prepared.module, mode),
                                        device=device,
                                    )
                                    output, forward = measure_forward(
                                        callable_target,
                                        prepared.args,
                                        device=device,
                                        warmup_calls=config.timing.warmup_calls,
                                        samples=config.timing.samples,
                                    )
                                    row.update(compile_wrap_ms=compile_wrap_ms, **forward)
                                    row.update(case_numeric_profile(prepared, output))
                                    if prepared.backward:
                                        row.update(
                                            measure_backward(
                                                callable_target,
                                                prepared.args,
                                                device=device,
                                                warmup_calls=config.timing.backward_warmup_calls,
                                                samples=config.timing.backward_samples,
                                            )
                                        )
                                    add_throughput(row, output, batch=batch)
                                    measurements.append(row)
                                    print_measurement(row)
                                    if config.profiler.enabled and prepared.case_id in config.profiler.case_ids:
                                        try:
                                            profiles.append(
                                                profile_case(
                                                    callable_target,
                                                    prepared,
                                                    row,
                                                    run_dir=run_dir,
                                                    device=device,
                                                    record_shapes=config.profiler.record_shapes,
                                                    profile_memory=config.profiler.profile_memory,
                                                )
                                            )
                                        except Exception as exc:  # profiling is separate from the timing row.
                                            events.append(event_row(row, status="error", stage="profiler", exc=exc))
                                except Exception as exc:  # benchmark continues after recording a generic error.
                                    row["status"] = "error"
                                    measurements.append(row)
                                    events.append(event_row(row, status="error", stage="execution", exc=exc))
                                    print_event(row, status="ERROR", stage="execution", message=str(exc))

    cumulative_rows.extend(run_cumulative(config, events))
    return measurements, cumulative_rows, events, profiles


def run_cumulative(
    config: BenchmarkConfig,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Measure repeated-operation drift against a CPU float64 trajectory."""

    rows: list[dict[str, Any]] = []
    sweeps = {sweep.sweep_id: sweep for sweep in config.sweeps}
    for declaration in config.cumulative:
        sweep = sweeps[str(declaration["sweep_id"])]
        signatures = expand_signatures(sweep)
        cases = {str(case["id"]): case for case in sweep.cases}
        case = cases[str(declaration["case_id"])]
        for requested_device in sweep.devices:
            device = str(resolve_device(requested_device))
            cumulative_dtypes = tuple(str(value) for value in declaration.get("dtypes", sweep.dtypes))
            for spec in signatures:
                if not int(declaration.get("min_dimension", 1)) <= spec.n <= int(declaration.get("max_dimension", 63)):
                    continue
                for dtype_declaration in cumulative_dtypes:
                    dtype = FLOAT_DTYPES[dtype_declaration]
                    for batch in sweep.batch_sizes:
                        if not case_applies(case, spec):
                            continue
                        base = base_identity(case, sweep, spec, device=device, dtype=dtype, batch=batch)
                        base.update(suite="cumulative", compile_mode="eager")
                        try:
                            feasible = preflight_case(
                                case,
                                spec,
                                batch=batch,
                                channels=sweep.channels,
                                actions=sweep.actions,
                                pairs=sweep.pairs,
                                dtype=dtype,
                                device=device,
                                resources=config.resources,
                            )
                            reference_feasible = preflight_case(
                                case,
                                spec,
                                batch=batch,
                                channels=sweep.channels,
                                actions=sweep.actions,
                                pairs=sweep.pairs,
                                dtype=torch.float64,
                                device="cpu",
                                resources=config.resources,
                            )
                            target_algebra = make_benchmark_algebra(spec, device=device, dtype=dtype)
                            reference_algebra = make_benchmark_algebra(spec, device="cpu", dtype=torch.float64)
                            case_index = list(sweep.cases).index(case)
                            seed = config.seed + spec.n * 1009 + batch * 37 + case_index * 17
                            target = build_case(
                                target_algebra,
                                case,
                                feasible,
                                batch=batch,
                                channels=sweep.channels,
                                actions=sweep.actions,
                                pairs=sweep.pairs,
                                device=device,
                                dtype=dtype,
                                seed=seed,
                            )
                            reference = build_case(
                                reference_algebra,
                                case,
                                reference_feasible,
                                batch=batch,
                                channels=sweep.channels,
                                actions=sweep.actions,
                                pairs=sweep.pairs,
                                device="cpu",
                                dtype=torch.float64,
                                seed=seed,
                            )
                            state_index = recurrence_state_index(target)
                            if target.input_layout is None or target.output_layout is None:
                                raise PreflightSkip("cumulative cases require explicit input and output layouts")
                            if target.input_layout.grades != target.output_layout.grades:
                                raise PreflightSkip("cumulative cases require matching input and output layouts")
                            current = target.args[state_index]
                            reference_current = reference.args[state_index]
                            metric = target_algebra.plan_signature_norm_squared(input_layout=target.input_layout)
                            initial_metric = metric(current).detach()
                            sample_steps = set(
                                logarithmic_steps(int(declaration["steps"]), int(declaration["samples"]))
                            )
                            for step in range(1, int(declaration["steps"]) + 1):
                                current = advance_state(target, current, state_index)
                                reference_current = advance_state(reference, reference_current, state_index)
                                if step not in sample_steps:
                                    continue
                                current_metric = metric(current).detach()
                                invariant_delta = (current_metric - initial_metric).abs()
                                invariant_drift = float(invariant_delta.max())
                                invariant_scale = torch.maximum(current_metric.abs(), initial_metric.abs()).clamp_min(
                                    torch.finfo(current_metric.dtype).eps
                                )
                                relative_invariant_drift = float((invariant_delta / invariant_scale).max())
                                rows.append(
                                    {
                                        **base,
                                        "status": "ok",
                                        "step": step,
                                        "invariant_drift": invariant_drift,
                                        "relative_invariant_drift": relative_invariant_drift,
                                        "state_rms": float(current.square().mean().sqrt()),
                                        "reference_rms": float(reference_current.square().mean().sqrt()),
                                        **error_stats(current, reference_current),
                                    }
                                )
                        except PreflightSkip as exc:
                            events.append(
                                event_row(
                                    base,
                                    status="skipped",
                                    stage="cumulative_preflight",
                                    exc=exc,
                                    include_traceback=False,
                                )
                            )
                        except Exception as exc:  # cumulative failures use the same generic error state.
                            events.append(event_row(base, status="error", stage="cumulative", exc=exc))
                            rows.append({**base, "status": "error"})
    return rows


def recurrence_state_index(prepared: PreparedCase) -> int:
    if prepared.kind in {"unary", "versor_action", "multi_versor_action", "paired_bivector_action"}:
        return 0
    if prepared.kind == "sandwich_action":
        return 1
    raise PreflightSkip(f"case kind {prepared.kind!r} has no cumulative recurrence")


def advance_state(prepared: PreparedCase, state: torch.Tensor, state_index: int) -> torch.Tensor:
    args = list(prepared.args)
    args[state_index] = state
    return prepared.module(*args)


def add_throughput(row: dict[str, Any], output: torch.Tensor, *, batch: int) -> None:
    median_ms = float(row["forward_median_ms"])
    seconds = median_ms / 1000.0
    if seconds <= 0.0:
        return
    row["items_per_second"] = batch / seconds
    row["coefficients_per_second"] = int(output.numel()) / seconds
    output_lanes = int(row.setdefault("output_lanes", int(output.shape[-1]) if output.ndim else 1))
    work_items = int(output.numel()) // max(output_lanes, 1)
    row["work_items_per_second"] = work_items / seconds
    row["interactions_per_second"] = work_items * int(row.get("pair_count", 0)) / seconds


def profile_case(
    fn,
    prepared: PreparedCase,
    identity: dict[str, Any],
    *,
    run_dir: Path,
    device: str,
    record_shapes: bool,
    profile_memory: bool,
) -> dict[str, Any]:
    """Collect one untimed Torch operator profile."""

    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)
    trace_name = "-".join(
        str(identity[key]).replace("/", "_").replace(" ", "_")
        for key in ("sweep_id", "case_id", "signature", "dtype", "compile_mode")
    )
    trace_path = run_dir / "profiles" / f"{trace_name}.json"
    args = differentiable_args(prepared.args) if prepared.backward else prepared.args
    with profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        acc_events=True,
    ) as recorded:
        output = fn(*args)
        if prepared.backward:
            scalar_loss(output).backward()
    recorded.export_chrome_trace(str(trace_path))
    operators = []
    for item in recorded.key_averages().table(sort_by="self_cpu_time_total", row_limit=20).splitlines():
        if item.strip():
            operators.append(item)
    return {
        **{
            key: identity.get(key)
            for key in ("sweep_id", "case_id", "signature", "device", "dtype", "batch", "compile_mode")
        },
        "status": "ok",
        "trace": str(trace_path.relative_to(run_dir)),
        "operator_table": operators,
    }


def base_identity(
    case: dict[str, Any],
    sweep: SweepConfig,
    spec: SignatureSpec,
    *,
    device: str,
    dtype: torch.dtype,
    batch: int,
) -> dict[str, Any]:
    return {
        "sweep_id": sweep.sweep_id,
        "layout_preset": sweep.layout_preset,
        "case_id": str(case["id"]),
        "kind": str(case["kind"]),
        "operation": str(case.get("operation", case["kind"])),
        "case_min_dimension": int(case.get("min_dimension", 1)),
        "case_max_dimension": int(case.get("max_dimension", 63)),
        "signature": spec.label,
        "p": spec.p,
        "q": spec.q,
        "r": spec.r,
        "n": spec.n,
        "device": device,
        "dtype": dtype_name(dtype),
        "batch": batch,
    }


def event_row(
    identity: dict[str, Any],
    *,
    status: str,
    stage: str,
    exc: BaseException,
    include_traceback: bool = True,
) -> dict[str, Any]:
    return {
        **{
            key: identity.get(key)
            for key in (
                "sweep_id",
                "layout_preset",
                "case_id",
                "kind",
                "operation",
                "signature",
                "p",
                "q",
                "r",
                "n",
                "device",
                "dtype",
                "batch",
                "compile_mode",
            )
        },
        "status": status,
        "stage": stage,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc() if include_traceback else "",
    }


def print_measurement(row: dict[str, Any]) -> None:
    timings = [
        _format_timing("setup", row.get("plan_ms")),
        _format_timing("cold", row.get("cold_forward_ms")),
        _format_timing("forward", row.get("forward_median_ms")),
        _format_timing("backward", row.get("backward_median_ms")),
        _format_timing("forward+backward", row.get("forward_backward_median_ms")),
    ]
    exp_family = row.get("exp_executor_family")
    exp_text = "" if exp_family is None else f" | exp={exp_family}"
    print(
        f"OK layout={row['layout_preset']} {row['signature']} {row['case_id']} dtype={row['dtype']} "
        f"batch={row['batch']} compiler={row['compile_mode']} | {' '.join(timings)}{exp_text}",
        flush=True,
    )


def print_event(identity: dict[str, Any], *, status: str, stage: str, message: str) -> None:
    compiler = identity.get("compile_mode", "none")
    print(
        f"{status} layout={identity['layout_preset']} {identity['signature']} {identity['case_id']} "
        f"dtype={identity['dtype']} batch={identity['batch']} compiler={compiler} "
        f"stage={stage} | {message}",
        flush=True,
    )


def _format_timing(label: str, value: Any) -> str:
    return f"{label}=-" if value is None else f"{label}={float(value):.4f}ms"


def _reset_compile_cache(mode: str) -> None:
    if mode == "eager":
        return
    reset = getattr(getattr(torch, "compiler", None), "reset", None)
    if callable(reset):
        reset()
