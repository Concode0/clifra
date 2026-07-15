"""Timing and numerical measurements for benchmark cases."""

from __future__ import annotations

import math
import statistics
import time
from typing import Any, Callable

import torch
import torch.nn as nn

from .models import PreparedCase


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize(device)


def timed_call(fn: Callable[[], Any], *, device: str) -> tuple[Any, float]:
    synchronize(device)
    start = time.perf_counter_ns()
    result = fn()
    synchronize(device)
    return result, (time.perf_counter_ns() - start) / 1_000_000.0


def compile_callable(module: Callable[..., torch.Tensor] | nn.Module, mode: str):
    """Prepare an eager or compiled benchmark callable."""

    if mode == "eager":
        return module
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable")
    if mode == "inductor":
        return torch.compile(module, backend="inductor", fullgraph=True)
    if mode == "reduce_overhead":
        return torch.compile(module, backend="inductor", mode="reduce-overhead", fullgraph=True)
    raise ValueError(f"unsupported compile mode {mode!r}")


def measure_forward(
    fn: Callable[..., torch.Tensor] | nn.Module,
    args: tuple[torch.Tensor, ...],
    *,
    device: str,
    warmup_calls: int,
    samples: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Measure cold and steady forward calls."""

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    output, cold_ms = timed_call(lambda: fn(*args), device=device)
    cold_memory = device_memory_fields(device)
    for _ in range(warmup_calls):
        fn(*args)
    synchronize(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    sample_values = [timed_call(lambda: fn(*args), device=device)[1] for _ in range(samples)]
    return output, {
        "cold_forward_ms": cold_ms,
        "cold_device_peak_bytes": cold_memory["device_peak_bytes"],
        **distribution("forward", sample_values),
        **device_memory_fields(device),
    }


def measure_backward(
    fn: Callable[..., torch.Tensor] | nn.Module,
    args: tuple[torch.Tensor, ...],
    *,
    device: str,
    warmup_calls: int,
    samples: int,
) -> dict[str, Any]:
    """Measure isolated backward and complete forward/backward calls."""

    cold_args = differentiable_args(args)
    cold_output = fn(*cold_args)
    weights = scalar_loss_weights(cold_output)
    cold_loss = scalar_loss(cold_output, weights=weights)
    _, cold_backward_ms = timed_call(cold_loss.backward, device=device)

    cold_complete_args = differentiable_args(args)

    def cold_complete_step() -> torch.Tensor:
        output = fn(*cold_complete_args)
        loss = scalar_loss(output, weights=weights)
        loss.backward()
        return loss

    _, cold_complete_ms = timed_call(cold_complete_step, device=device)
    for _ in range(warmup_calls):
        warmup_args = differentiable_args(args)
        warmup_output = fn(*warmup_args)
        scalar_loss(warmup_output, weights=weights).backward()
    synchronize(device)

    backward_values: list[float] = []
    complete_values: list[float] = []
    profile_grads: tuple[torch.Tensor, ...] = ()
    for _ in range(samples):
        backward_args = differentiable_args(args)
        output = fn(*backward_args)
        loss = scalar_loss(output, weights=weights)
        _, elapsed = timed_call(loss.backward, device=device)
        backward_values.append(elapsed)
        profile_grads = tuple(arg.grad for arg in backward_args if arg.requires_grad and arg.grad is not None)

        complete_args = differentiable_args(args)

        def complete_step() -> torch.Tensor:
            complete_output = fn(*complete_args)
            complete_loss = scalar_loss(complete_output, weights=weights)
            complete_loss.backward()
            return complete_loss

        _, elapsed = timed_call(complete_step, device=device)
        complete_values.append(elapsed)
    return {
        "cold_backward_ms": cold_backward_ms,
        "cold_forward_backward_ms": cold_complete_ms,
        **distribution("backward", backward_values),
        **distribution("forward_backward", complete_values),
        **aggregate_tensor_stats("gradient", profile_grads),
    }


def differentiable_args(args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    cloned: list[torch.Tensor] = []
    for arg in args:
        value = arg.detach().clone()
        if value.is_floating_point() or value.is_complex():
            value.requires_grad_(True)
        cloned.append(value)
    return tuple(cloned)


def scalar_loss(output: torch.Tensor, *, weights: torch.Tensor | None = None) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"benchmark callable must return a Tensor, got {type(output)!r}")
    if weights is None:
        weights = scalar_loss_weights(output)
    return (output.real * weights).mean()


def scalar_loss_weights(output: torch.Tensor) -> torch.Tensor:
    """Build a stable, nonuniform output seed once for repeated backward timing."""

    return torch.linspace(0.5, 1.5, output.numel(), device=output.device, dtype=output.real.dtype).reshape(output.shape)


def distribution(prefix: str, values: list[float]) -> dict[str, Any]:
    """Return detailed distribution statistics while retaining raw samples."""

    if not values:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_min_ms": None,
            f"{prefix}_max_ms": None,
            f"{prefix}_mean_ms": None,
            f"{prefix}_std_ms": None,
            f"{prefix}_median_ms": None,
            f"{prefix}_p10_ms": None,
            f"{prefix}_q1_ms": None,
            f"{prefix}_q3_ms": None,
            f"{prefix}_iqr_ms": None,
            f"{prefix}_p90_ms": None,
            f"{prefix}_p95_ms": None,
            f"{prefix}_p99_ms": None,
            f"{prefix}_samples_ms": [],
        }
    ordered = sorted(float(value) for value in values)
    q1 = percentile(ordered, 0.25)
    q3 = percentile(ordered, 0.75)
    return {
        f"{prefix}_count": len(ordered),
        f"{prefix}_min_ms": ordered[0],
        f"{prefix}_max_ms": ordered[-1],
        f"{prefix}_mean_ms": float(statistics.fmean(ordered)),
        f"{prefix}_std_ms": float(statistics.pstdev(ordered)),
        f"{prefix}_median_ms": float(statistics.median(ordered)),
        f"{prefix}_p10_ms": percentile(ordered, 0.10),
        f"{prefix}_q1_ms": q1,
        f"{prefix}_q3_ms": q3,
        f"{prefix}_iqr_ms": q3 - q1,
        f"{prefix}_p90_ms": percentile(ordered, 0.90),
        f"{prefix}_p95_ms": percentile(ordered, 0.95),
        f"{prefix}_p99_ms": percentile(ordered, 0.99),
        f"{prefix}_samples_ms": ordered,
    }


def percentile(ordered: list[float], quantile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def tensor_stats(prefix: str, tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().to(device="cpu", dtype=torch.float64)
    finite = torch.isfinite(values)
    finite_values = values[finite]
    count = int(values.numel())
    result: dict[str, Any] = {
        f"{prefix}_numel": count,
        f"{prefix}_finite_fraction": float(finite.float().mean()) if count else 1.0,
    }
    if finite_values.numel() == 0:
        result.update(
            {
                f"{prefix}_mean": None,
                f"{prefix}_std": None,
                f"{prefix}_rms": None,
                f"{prefix}_max_abs": None,
                f"{prefix}_norm": None,
            }
        )
        return result
    result.update(
        {
            f"{prefix}_mean": float(finite_values.mean()),
            f"{prefix}_std": float(finite_values.std(unbiased=False)),
            f"{prefix}_rms": float(finite_values.square().mean().sqrt()),
            f"{prefix}_max_abs": float(finite_values.abs().max()),
            f"{prefix}_norm": float(torch.linalg.vector_norm(finite_values)),
        }
    )
    return result


def aggregate_tensor_stats(prefix: str, tensors: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    if not tensors:
        return tensor_stats(prefix, torch.empty(0))
    flattened = torch.cat([tensor.detach().reshape(-1).to(device="cpu", dtype=torch.float64) for tensor in tensors])
    return tensor_stats(prefix, flattened)


def error_stats(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual64 = actual.detach().to(device="cpu", dtype=torch.float64)
    reference64 = reference.detach().to(device="cpu", dtype=torch.float64)
    if actual64.shape != reference64.shape:
        raise ValueError(f"cumulative trajectory shape mismatch: {actual64.shape} != {reference64.shape}")
    difference = actual64 - reference64
    absolute = difference.abs()
    reference_scale = reference64.abs().max().clamp_min(1.0e-30)
    return {
        "max_abs_error": float(absolute.max()) if absolute.numel() else 0.0,
        "max_rel_error": float(absolute.max() / reference_scale) if absolute.numel() else 0.0,
        "rms_error": float(difference.square().mean().sqrt()) if difference.numel() else 0.0,
    }


def logarithmic_steps(max_steps: int, sample_count: int) -> tuple[int, ...]:
    values = {1, int(max_steps)}
    for index in range(sample_count):
        fraction = index / max(sample_count - 1, 1)
        values.add(max(1, int(round(math.exp(fraction * math.log(max_steps))))))
    return tuple(sorted(values))


def case_numeric_profile(prepared: PreparedCase, output: torch.Tensor) -> dict[str, Any]:
    return {
        "input_bytes": sum(tensor.numel() * tensor.element_size() for tensor in prepared.args),
        "output_bytes": output.numel() * output.element_size(),
        **aggregate_tensor_stats("input", tuple(arg for arg in prepared.args if arg.is_floating_point())),
        **tensor_stats("output", output),
    }


def device_memory_fields(device: str) -> dict[str, int | None]:
    """Return allocator measurements exposed by the selected backend."""

    if device.startswith("cuda"):
        return {
            "device_peak_bytes": int(torch.cuda.max_memory_allocated(device)),
            "device_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        }
    if device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return {
            "device_peak_bytes": None,
            "device_allocated_bytes": int(torch.mps.current_allocated_memory()),
        }
    return {"device_peak_bytes": None, "device_allocated_bytes": None}
