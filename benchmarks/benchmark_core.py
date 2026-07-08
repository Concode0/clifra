#!/usr/bin/env python3
"""Planner/executor benchmark suite for the CLIFRA core.

Artifacts are written to ``benchmarks/results/benchmark_core_<timestamp>/``.

Examples:
    uv run python benchmarks/benchmark_core.py --quick
    uv run python benchmarks/benchmark_core.py --device cpu --signatures 3,4,6,8,16
    uv run python benchmarks/benchmark_core.py --device mps --dtypes float32 --compile-modes eager,aot_eager
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from clifra.core.foundation.basis import expand_output_grades
from clifra.core.foundation.device import FLOAT_DTYPES, dtype_name, resolve_device, resolve_dtype
from clifra.core.foundation.layout import GradeLayout
from clifra.core.planning.policy import (
    DEFAULT_PLANNING_LIMITS,
    DEFAULT_PRODUCT_EXECUTION_POLICY,
    PlanningLimits,
    ProductExecutionPolicy,
    estimate_product_executor_cost,
)
from clifra.core.runtime.algebra import AlgebraContext

DTYPES: dict[str, torch.dtype] = FLOAT_DTYPES
DEFAULT_OPS = (
    "full_gp,full_wedge,full_symmetric_product,full_commutator_product,"
    "vector_gp,bivector_vector_commutator,bivector_bivector_commutator,"
    "signature_norm_vector,pseudoscalar_product_vector,bivector_exp,"
    "full_sandwich,versor_vector,multi_versor_vector,paired_bivector_vector"
)
PRODUCT_OPS = {
    "gp",
    "geometric_product",
    "wedge",
    "symmetric_product",
    "commutator_product",
    "anti_commutator_product",
    "left_contraction",
    "right_contraction",
}
UNLOCKED_LIMIT = 1 << 62
DIAGNOSTIC_SUITES = {"backward", "cumulative", "convergence"}


@dataclass(frozen=True)
class TierPreset:
    dimension_range: str
    signature_families: str
    batch_sizes: str
    ops: str
    compile_modes: str
    warmup: int
    iterations: int
    channels: int
    actions: int
    pairs: int
    max_full_lanes: int


TIER_PRESETS: dict[str, TierPreset] = {
    "smoke": TierPreset(
        dimension_range="3:5",
        signature_families="euclidean",
        batch_sizes="4",
        ops="full_gp,vector_gp,bivector_exp,versor_vector",
        compile_modes="eager,aot_eager",
        warmup=1,
        iterations=2,
        channels=2,
        actions=2,
        pairs=2,
        max_full_lanes=64,
    ),
    "standard": TierPreset(
        dimension_range="3:8",
        signature_families="euclidean",
        batch_sizes="32",
        ops=DEFAULT_OPS,
        compile_modes="eager,aot_eager",
        warmup=3,
        iterations=10,
        channels=8,
        actions=8,
        pairs=4,
        max_full_lanes=256,
    ),
    "stress": TierPreset(
        dimension_range="3:12",
        signature_families="euclidean,minkowski,degenerate",
        batch_sizes="8,32",
        ops=DEFAULT_OPS,
        compile_modes="eager,aot_eager,inductor",
        warmup=5,
        iterations=20,
        channels=8,
        actions=8,
        pairs=4,
        max_full_lanes=4096,
    ),
    "exhaustive": TierPreset(
        dimension_range="2:16",
        signature_families="euclidean,minkowski,degenerate",
        batch_sizes="1,8,32",
        ops=DEFAULT_OPS,
        compile_modes="eager,aot_eager,inductor,reduce-overhead",
        warmup=5,
        iterations=30,
        channels=8,
        actions=8,
        pairs=4,
        max_full_lanes=0,
    ),
}


@dataclass(frozen=True)
class SignatureSpec:
    p: int
    q: int = 0
    r: int = 0

    @property
    def n(self) -> int:
        return self.p + self.q + self.r

    @property
    def label(self) -> str:
        return f"Cl({self.p},{self.q},{self.r})"


@dataclass
class BenchTarget:
    name: str
    family: str
    op: str
    layout_case: str
    module: Callable[..., torch.Tensor] | nn.Module
    args: tuple[torch.Tensor, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkPolicy:
    planning_limits: PlanningLimits
    product_execution_policy: ProductExecutionPolicy


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_int_csv(value: str) -> list[int]:
    return [int(part) for part in _parse_csv(value)]


def _parse_dimension_range(value: str) -> tuple[int, int]:
    text = str(value).strip().replace(",", ":")
    if ":" in text:
        lo, hi = (int(part) for part in text.split(":", 1))
    else:
        lo = hi = int(text)
    if lo < 1 or hi < lo:
        raise ValueError(f"invalid dimension range {value!r}; use n or min:max with min >= 1")
    return lo, hi


def _parse_signature_csv(value: str) -> list[SignatureSpec]:
    specs: list[SignatureSpec] = []
    for raw in _parse_csv(value):
        cleaned = raw.lower().removeprefix("cl").replace("(", "").replace(")", "").replace("/", ":")
        parts = [part for part in cleaned.split(":") if part]
        if len(parts) == 1:
            p, q, r = int(parts[0]), 0, 0
        elif len(parts) == 2:
            p, q = (int(part) for part in parts)
            r = 0
        elif len(parts) == 3:
            p, q, r = (int(part) for part in parts)
        else:
            raise ValueError(f"invalid signature {raw!r}; use n, p:q, or p:q:r")
        if p < 0 or q < 0 or r < 0 or p + q + r < 1 or p + q + r > 63:
            raise ValueError(f"invalid signature {raw!r}; dimensions must sum to 1..63")
        specs.append(SignatureSpec(p, q, r))
    return specs


def _signatures_from_range(value: str, families: str) -> list[SignatureSpec]:
    n_min, n_max = _parse_dimension_range(value)
    result: list[SignatureSpec] = []
    requested = _parse_csv(families)
    valid = {"euclidean", "minkowski", "degenerate"}
    unknown = sorted(set(requested) - valid)
    if unknown:
        raise ValueError(f"unknown signature family {unknown}; valid: {sorted(valid)}")
    for n in range(n_min, n_max + 1):
        if "euclidean" in requested:
            result.append(SignatureSpec(n, 0, 0))
        if "minkowski" in requested and n >= 2:
            result.append(SignatureSpec(n - 1, 1, 0))
        if "degenerate" in requested and n >= 2:
            result.append(SignatureSpec(n - 1, 0, 1))
    return result


def _resolve_signature_specs(args: argparse.Namespace) -> list[SignatureSpec]:
    if args.signatures:
        specs = _parse_signature_csv(args.signatures)
    else:
        specs = _signatures_from_range(args.dimension_range, args.signature_families)
    if args.n_min is not None or args.n_max is not None:
        n_min = 1 if args.n_min is None else int(args.n_min)
        n_max = 63 if args.n_max is None else int(args.n_max)
        specs = [spec for spec in specs if n_min <= spec.n <= n_max]
    if not specs:
        raise ValueError("no signatures selected after tier/range filters")
    return specs


def _resolve_device(value: str) -> str:
    if value == "auto":
        return str(resolve_device("auto"))
    if value == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise SystemExit("MPS was requested but is not available.")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"{value} was requested but CUDA is not available.")
    return value


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def _release_memory(device: str) -> None:
    gc.collect()
    gc.collect()
    _sync(device)
    if device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _reset_compile_cache() -> bool:
    """Clear torch.compile/Dynamo state between independent benchmark rows."""
    compiler = getattr(torch, "compiler", None)
    compiler_reset = getattr(compiler, "reset", None)
    if callable(compiler_reset):
        compiler_reset()
        return True

    dynamo = getattr(torch, "_dynamo", None)
    dynamo_reset = getattr(dynamo, "reset", None)
    if callable(dynamo_reset):
        dynamo_reset()
        return True
    return False


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _time_block(fn: Callable[[], Any], *, device: str) -> tuple[Any, float]:
    _sync(device)
    start = _now_ms()
    value = fn()
    _sync(device)
    return value, _now_ms() - start


def _time_callable(
    fn: Callable[..., torch.Tensor] | nn.Module,
    args: tuple[torch.Tensor, ...],
    *,
    device: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    first_output, first_call_ms = _time_block(lambda: fn(*args), device=device)
    for _ in range(warmup):
        fn(*args)
    _sync(device)

    samples: list[float] = []
    for _ in range(iterations):
        _, elapsed = _time_block(lambda: fn(*args), device=device)
        samples.append(elapsed)
    sorted_samples = sorted(samples)
    return {
        "output": first_output,
        "first_call_ms": first_call_ms,
        "median_ms": _median(samples),
        "mean_ms": sum(samples) / len(samples) if samples else float("nan"),
        "std_ms": _std(samples),
        "min_ms": min(samples) if samples else float("nan"),
        "max_ms": max(samples) if samples else float("nan"),
        "p10_ms": _percentile(sorted_samples, 0.10),
        "p90_ms": _percentile(sorted_samples, 0.90),
        "runs": len(samples),
        "samples_ms": samples,
    }


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _max_abs_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.shape != expected.shape:
        return float("inf")
    diff = (actual.detach().float() - expected.detach().float()).abs()
    return float(diff.max().item()) if diff.numel() else 0.0


def _max_rel_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.shape != expected.shape:
        return float("inf")
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    diff = (actual_f - expected_f).abs()
    scale = expected_f.abs().max().clamp_min(torch.finfo(torch.float32).tiny)
    return float((diff.max() / scale).item()) if diff.numel() else 0.0


def _error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    if actual.shape != expected.shape:
        return {"max_abs_error": float("inf"), "rms_error": float("inf"), "max_rel_error": float("inf")}
    actual64 = actual.detach().cpu().to(torch.float64)
    expected64 = expected.detach().cpu().to(torch.float64)
    diff = actual64 - expected64
    abs_diff = diff.abs()
    scale = expected64.abs().max().clamp_min(1e-30)
    return {
        "max_abs_error": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "rms_error": float(torch.sqrt((diff * diff).mean()).item()) if diff.numel() else 0.0,
        "max_rel_error": float((abs_diff.max() / scale).item()) if abs_diff.numel() else 0.0,
    }


def _loss_from_output(output: torch.Tensor) -> torch.Tensor:
    return output.float().square().mean()


def _clone_inputs_for_backward(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    cloned: list[torch.Tensor] = []
    for tensor in inputs:
        leaf = tensor.detach().clone()
        if leaf.is_floating_point():
            leaf.requires_grad_(True)
        cloned.append(leaf)
    return tuple(cloned)


def _zero_input_grads(inputs: tuple[torch.Tensor, ...]) -> None:
    for tensor in inputs:
        if tensor.grad is not None:
            tensor.grad = None


def _grad_norm_and_finite(inputs: tuple[torch.Tensor, ...]) -> tuple[float, bool]:
    grad_norm_squared = 0.0
    finite = True
    saw_grad = False
    for tensor in inputs:
        if tensor.grad is None:
            continue
        saw_grad = True
        grad = tensor.grad.detach().float()
        finite = finite and bool(torch.isfinite(grad).all().item())
        grad_norm_squared += float((grad * grad).sum().item())
    return math.sqrt(grad_norm_squared), finite and saw_grad


def _time_forward_backward(
    fn: Callable[..., torch.Tensor] | nn.Module,
    args: tuple[torch.Tensor, ...],
    *,
    device: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    backward_args = _clone_inputs_for_backward(args)

    def step() -> torch.Tensor:
        _zero_input_grads(backward_args)
        output = fn(*backward_args)
        _loss_from_output(output).backward()
        return output

    first_output, first_call_ms = _time_block(step, device=device)
    for _ in range(warmup):
        step()
    _sync(device)

    samples: list[float] = []
    for _ in range(iterations):
        output, elapsed = _time_block(step, device=device)
        samples.append(elapsed)
    sorted_samples = sorted(samples)

    _zero_input_grads(backward_args)
    output = fn(*backward_args)
    _loss_from_output(output).backward()
    grad_norm, grad_finite = _grad_norm_and_finite(backward_args)
    _zero_input_grads(backward_args)
    return {
        "output": output.detach(),
        "first_output": first_output.detach(),
        "first_call_ms": first_call_ms,
        "median_ms": _median(samples),
        "mean_ms": sum(samples) / len(samples) if samples else float("nan"),
        "std_ms": _std(samples),
        "min_ms": min(samples) if samples else float("nan"),
        "max_ms": max(samples) if samples else float("nan"),
        "p10_ms": _percentile(sorted_samples, 0.10),
        "p90_ms": _percentile(sorted_samples, 0.90),
        "runs": len(samples),
        "samples_ms": samples,
        "grad_norm": grad_norm,
        "grad_finite": grad_finite,
    }


def _sample_steps(max_steps: int, samples: int) -> list[int]:
    if max_steps <= 1:
        return [1]
    values = {1, int(max_steps)}
    for index in range(max(int(samples), 1)):
        t = index / max(int(samples) - 1, 1)
        values.add(max(1, int(round(math.exp(t * math.log(max_steps))))))
    return sorted(values)


def _basis_bivector_index(i: int, j: int) -> int:
    return (1 << int(i)) | (1 << int(j))


def _commuting_pairs(n: int) -> list[tuple[int, int]]:
    pairs = [(0, 1), (2, 3)]
    if n >= 6:
        pairs.append((4, 5))
    if n >= 8:
        pairs.append((6, 7))
    return pairs


def _metric_vector_square(spec: SignatureSpec, bit: int) -> float:
    if bit < spec.p:
        return 1.0
    if bit < spec.p + spec.q:
        return -1.0
    return 0.0


def _bivector_square_sign(spec: SignatureSpec, i: int, j: int) -> float:
    return -_metric_vector_square(spec, i) * _metric_vector_square(spec, j)


def _make_commuting_bivector(algebra: AlgebraContext, batch: int, scale: float) -> torch.Tensor:
    if algebra.n < 4:
        raise ValueError("controlled commuting bivectors require n >= 4")
    full = torch.zeros(batch, algebra.dim, device=algebra.device, dtype=algebra.dtype)
    multipliers = torch.linspace(0.85, 1.15, int(batch), device=algebra.device, dtype=algebra.dtype)
    for coeff, (i, j) in zip((0.37, -0.23, 0.13, -0.07), _commuting_pairs(algebra.n)):
        full[:, _basis_bivector_index(i, j)] = float(scale) * coeff * multipliers
    return algebra.layout((2,)).compact(full)


def _simple_plane_exp(
    algebra: AlgebraContext,
    b: torch.Tensor,
    *,
    pair: tuple[int, int],
    output_layout: GradeLayout,
) -> torch.Tensor:
    bivector_layout = algebra.layout((2,))
    index = _basis_bivector_index(*pair)
    try:
        position = bivector_layout.basis_indices.index(index)
    except ValueError:
        coeff = b.new_zeros(*b.shape[:-1])
    else:
        coeff = b[..., position]

    sign = _bivector_square_sign(SignatureSpec(algebra.p, algebra.q, algebra.r), *pair)
    if sign < 0:
        scalar = torch.cos(coeff)
        blade = torch.sin(coeff)
    elif sign > 0:
        scalar = torch.cosh(coeff)
        blade = torch.sinh(coeff)
    else:
        scalar = torch.ones_like(coeff)
        blade = coeff

    output = b.new_zeros(*b.shape[:-1], output_layout.dim)
    if 0 in output_layout.basis_indices:
        scalar_pos = output_layout.basis_indices.index(0)
        output[..., scalar_pos] = scalar
    if index in output_layout.basis_indices:
        blade_pos = output_layout.basis_indices.index(index)
        output[..., blade_pos] = blade
    return output


def _exact_commuting_exp(algebra: AlgebraContext, b: torch.Tensor, output_layout: GradeLayout) -> torch.Tensor:
    """Exact exp for the benchmark's disjoint-coordinate-plane bivectors."""
    result = b.new_zeros(*b.shape[:-1], output_layout.dim)
    if 0 not in output_layout.basis_indices:
        raise ValueError("exact commuting exp output layout must include the scalar lane")
    result[..., output_layout.basis_indices.index(0)] = 1.0
    product = algebra.plan_product(op="gp", left_layout=output_layout, right_layout=output_layout, output_layout=output_layout)
    for pair in _commuting_pairs(algebra.n):
        factor = _simple_plane_exp(algebra, b, pair=pair, output_layout=output_layout)
        result = product(result, factor)
    return result


def _identity_for_layout(layout: GradeLayout, *, like: torch.Tensor) -> torch.Tensor:
    identity = like.new_zeros(*like.shape[:-1], layout.dim)
    if 0 in layout.basis_indices:
        identity[..., layout.basis_indices.index(0)] = 1.0
    return identity


def _grade_leak(values: torch.Tensor, layout: GradeLayout, grade: int) -> float:
    if layout.grades == (int(grade),):
        return 0.0
    positions = layout.positions_for_grades((int(grade),), device=values.device)
    target = values.new_zeros(*values.shape[:-1], layout.dim)
    if positions.numel() > 0:
        target = target.index_copy(-1, positions, torch.index_select(values, -1, positions))
    return float((values - target).detach().float().norm().item())


def _dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _args_bytes(args: tuple[torch.Tensor, ...]) -> int:
    return sum(_tensor_bytes(arg) for arg in args if isinstance(arg, torch.Tensor))


def _output_bytes(output: torch.Tensor) -> int:
    if not isinstance(output, torch.Tensor):
        return 0
    return _tensor_bytes(output)


def _pair_buffer_estimate_bytes(row: dict[str, Any], dtype: torch.dtype) -> int:
    pair_count = int(row.get("pair_count") or 0)
    family = str(row.get("executor_family", ""))
    if pair_count <= 0:
        return 0
    scalar_bytes = _dtype_size(dtype)
    if family == "full_table":
        return pair_count * (8 + scalar_bytes)
    if family == "sparse":
        return pair_count * (24 + scalar_bytes)
    if family == "action_matrix":
        return pair_count * (16 + scalar_bytes)
    return pair_count * scalar_bytes


def _memory_estimate_fields(target: BenchTarget, row: dict[str, Any], dtype: torch.dtype) -> dict[str, Any]:
    arg_bytes = _args_bytes(target.args)
    pair_buffer_bytes = _pair_buffer_estimate_bytes(row, dtype)
    return {
        "arg_bytes": arg_bytes,
        "pair_buffer_estimate_bytes": pair_buffer_bytes,
        "static_estimated_bytes": arg_bytes + pair_buffer_bytes,
    }


def _cache_snapshot(algebra: AlgebraContext) -> dict[str, int]:
    planner = algebra.planner
    return {
        "product": len(planner._product_executors),
        "unary": len(planner._unary_executors),
        "signature_norm_squared": len(planner._signature_norm_squared_executors),
        "pseudoscalar_product": len(planner._pseudoscalar_product_executors),
        "bivector_exp": len(planner._bivector_exp_executors),
        "full_sandwich": len(planner._full_sandwich_action_executors),
    }


def _cache_delta(before: dict[str, int], after: dict[str, int]) -> int:
    return sum(max(0, after[key] - before.get(key, 0)) for key in after)


def _full_layout_allowed(layout: GradeLayout, max_lanes: int) -> bool:
    return max_lanes <= 0 or layout.dim <= max_lanes


def _benchmark_policy(args: argparse.Namespace) -> BenchmarkPolicy:
    planning_limits = DEFAULT_PLANNING_LIMITS
    product_policy = DEFAULT_PRODUCT_EXECUTION_POLICY

    if args.unlock_policy_limits:
        planning_limits = PlanningLimits(
            warn_lanes=UNLOCKED_LIMIT,
            max_lanes=UNLOCKED_LIMIT,
            warn_pairs=UNLOCKED_LIMIT,
            max_pairs=UNLOCKED_LIMIT,
        )
        product_policy = replace(product_policy, full_table_max_lanes=UNLOCKED_LIMIT)

    if args.max_plan_lanes is not None:
        planning_limits = replace(
            planning_limits,
            warn_lanes=max(int(args.max_plan_lanes) // 2, 1),
            max_lanes=int(args.max_plan_lanes),
        )
    if args.max_plan_pairs is not None:
        planning_limits = replace(
            planning_limits,
            warn_pairs=max(int(args.max_plan_pairs) // 2, 1),
            max_pairs=int(args.max_plan_pairs),
        )
    if args.full_table_max_lanes is not None:
        product_policy = replace(product_policy, full_table_max_lanes=int(args.full_table_max_lanes))
    return BenchmarkPolicy(planning_limits=planning_limits, product_execution_policy=product_policy)


def _make_algebra(
    args: argparse.Namespace,
    spec: SignatureSpec,
    dtype: torch.dtype,
    *,
    device: str | None = None,
) -> AlgebraContext:
    policy = _benchmark_policy(args)
    return AlgebraContext(
        spec.p,
        spec.q,
        spec.r,
        device=args.device if device is None else device,
        dtype=dtype,
        planning_limits=policy.planning_limits,
        product_execution_policy=policy.product_execution_policy,
    )


def _policy_metadata(args: argparse.Namespace) -> dict[str, Any]:
    policy = _benchmark_policy(args)
    return {
        "tier": args.tier,
        "dimension_range": args.dimension_range,
        "signature_families": args.signature_families,
        "compile_cache_scope": args.compile_cache_scope,
        "policy_unlocked": bool(args.unlock_policy_limits),
        "planning_max_lanes": policy.planning_limits.max_lanes,
        "planning_max_pairs": policy.planning_limits.max_pairs,
        "full_table_max_lanes": policy.product_execution_policy.full_table_max_lanes,
        "benchmark_max_full_lanes": int(args.max_full_lanes),
        "exp_output_grades_requested": args.exp_output_grades,
        "exp_spectral_max_planes_requested": args.exp_spectral_max_planes,
        "exp_spectral_tol_abs_requested": args.exp_spectral_tol_abs,
        "exp_spectral_tol_rel_requested": args.exp_spectral_tol_rel,
        "exp_spectral_dominant_rel_requested": args.exp_spectral_dominant_rel,
        "exp_spectral_allow_degenerate": not bool(args.exp_spectral_disable_degenerate),
        "exp_spectral_allow_truncated_degenerate": not bool(args.exp_spectral_disable_truncated_degenerate),
    }


def _exp_plan_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "spectral_max_planes": args.exp_spectral_max_planes,
        "spectral_tol_abs": args.exp_spectral_tol_abs,
        "spectral_tol_rel": args.exp_spectral_tol_rel,
        "spectral_dominant_rel": args.exp_spectral_dominant_rel,
        "spectral_allow_degenerate": False if args.exp_spectral_disable_degenerate else None,
        "spectral_allow_truncated_degenerate": False if args.exp_spectral_disable_truncated_degenerate else None,
    }


def _exp_pair_count(executor: Any) -> int:
    left_product = getattr(executor, "left_product", None)
    return int(getattr(left_product, "pair_count", 0) or 0)


def _exp_executor_metadata(executor: Any) -> dict[str, Any]:
    return {
        "exp_spectral_max_planes": getattr(executor, "spectral_max_planes", None),
        "exp_spectral_tol_abs": getattr(executor, "spectral_tol_abs", None),
        "exp_spectral_tol_rel": getattr(executor, "spectral_tol_rel", None),
        "exp_spectral_dominant_rel": getattr(executor, "spectral_dominant_rel", None),
        "exp_spectral_allow_degenerate_resolved": getattr(executor, "spectral_allow_degenerate", None),
        "exp_spectral_allow_truncated_degenerate_resolved": getattr(
            executor,
            "spectral_allow_truncated_degenerate",
            None,
        ),
        "exp_spectral_local_axis_count": getattr(executor, "spectral_local_axis_count", None),
        "exp_nondegenerate_dim": getattr(executor, "nondegenerate_dim", None),
        "exp_ideal_dim": getattr(executor, "ideal_dim", None),
    }


def _exp_output_layout(algebra: AlgebraContext, selector: str) -> tuple[GradeLayout, str]:
    normalized = str(selector).strip().lower()
    if normalized in {"", "even"}:
        return algebra.layout(range(0, algebra.n + 1, 2)), "even"
    if normalized == "full":
        return algebra.spec.full_layout(), "full"
    grades = tuple(int(part) for part in _parse_csv(normalized.replace(":", ",")))
    if not grades:
        raise ValueError("exp output grades selector must be 'even', 'full', or a comma-separated grade list")
    return algebra.layout(grades), ":".join(str(grade) for grade in grades)


def _randn(shape: tuple[int, ...], *, device: str, dtype: torch.dtype, seed: int, scale: float = 1.0) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn(*shape, dtype=dtype, generator=generator)
    return values.to(device=device) * scale


def _layout_metadata(prefix: str, layout: GradeLayout) -> dict[str, Any]:
    return {
        f"{prefix}_grades": ":".join(str(grade) for grade in layout.grades),
        f"{prefix}_lanes": layout.dim,
    }


def _product_policy_metadata(
    algebra: AlgebraContext,
    *,
    op: str,
    left_layout: GradeLayout,
    right_layout: GradeLayout,
    output_layout: GradeLayout,
    dtype: torch.dtype,
    device: str,
) -> dict[str, Any]:
    cost = estimate_product_executor_cost(
        algebra,
        op=op,
        left_layout=left_layout,
        right_layout=right_layout,
        output_layout=output_layout,
        dtype=dtype,
        device=device,
    )
    return {
        "policy_executor_family": cost.executor_family,
        "policy_selected_score": cost.selected_score,
        "policy_full_table_score": cost.full_table_score,
        "policy_sparse_score": cost.sparse_score,
        "policy_full_table_pair_count": cost.full_table_pair_count,
        "policy_sparse_estimated_pairs": cost.sparse_estimated_pairs,
        "policy_path_count": cost.path_count,
        "policy_full_table_estimated_bytes": cost.full_table_estimated_bytes,
        "policy_sparse_estimated_bytes": cost.sparse_estimated_bytes,
        "policy_backend": cost.backend,
    }


def _product_target(
    algebra: AlgebraContext,
    *,
    name: str,
    op: str,
    left_layout: GradeLayout,
    right_layout: GradeLayout,
    output_layout: GradeLayout | None,
    batch: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
) -> BenchTarget:
    handle = algebra.plan_product(op=op, left_layout=left_layout, right_layout=right_layout, output_layout=output_layout)
    left = _randn((batch, left_layout.dim), device=device, dtype=dtype, seed=seed)
    right = _randn((batch, right_layout.dim), device=device, dtype=dtype, seed=seed + 1)
    metadata = {
        "category": "product_full" if left_layout.dim == left_layout.spec.dim else "product_active",
        "call_method": "plan_product_handle",
        **_layout_metadata("left", left_layout),
        **_layout_metadata("right", right_layout),
        **_layout_metadata("output", handle.output_layout),
        "pair_count": handle.pair_count,
        **_product_policy_metadata(
            algebra,
            op=op,
            left_layout=left_layout,
            right_layout=right_layout,
            output_layout=handle.output_layout,
            dtype=dtype,
            device=device,
        ),
    }
    return BenchTarget(
        name=name,
        family=handle.executor_family,
        op=op,
        layout_case=name,
        module=handle,
        args=(left, right),
        metadata=metadata,
    )


def _build_target(
    algebra: AlgebraContext,
    *,
    op_name: str,
    batch: int,
    channels: int,
    actions: int,
    pairs: int,
    max_full_lanes: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
    exp_plan_kwargs: dict[str, Any] | None = None,
    exp_output_grades: str = "even",
) -> BenchTarget | None:
    full = algebra.spec.full_layout()
    vector = algebra.layout((1,))

    if op_name == "full_sandwich":
        if not _full_layout_allowed(full, max_full_lanes):
            return None
        handle = algebra.plan_sandwich_action(layout=full, dtype=dtype, device=device)
        left = _randn((channels, full.dim), device=device, dtype=dtype, seed=seed, scale=0.1)
        right = _randn((channels, full.dim), device=device, dtype=dtype, seed=seed + 1, scale=0.1)
        values = _randn((batch, channels, full.dim), device=device, dtype=dtype, seed=seed + 2)
        return BenchTarget(
            name=op_name,
            family=handle.executor_family,
            op="sandwich_action",
            layout_case=op_name,
            module=handle,
            args=(left, values, right),
            metadata={
                "category": "action_full",
                "call_method": "plan_sandwich_action_handle",
                **_layout_metadata("input", full),
                **_layout_metadata("output", full),
                "pair_count": full.dim * full.dim,
            },
        )

    if op_name.startswith("full_"):
        if not _full_layout_allowed(full, max_full_lanes):
            return None
        product_op = op_name.removeprefix("full_")
        if product_op not in PRODUCT_OPS:
            raise ValueError(f"unknown full product op {op_name!r}")
        return _product_target(
            algebra,
            name=op_name,
            op=product_op,
            left_layout=full,
            right_layout=full,
            output_layout=full,
            batch=batch,
            device=device,
            dtype=dtype,
            seed=seed,
        )

    if op_name in {
        "bivector_vector_commutator",
        "bivector_bivector_commutator",
        "bivector_exp",
        "versor_vector",
        "multi_versor_vector",
        "paired_bivector_vector",
    } and algebra.n < 2:
        return None

    bivector = algebra.layout((2,))

    if op_name == "vector_gp":
        return _product_target(
            algebra,
            name=op_name,
            op="gp",
            left_layout=vector,
            right_layout=vector,
            output_layout=algebra.layout(expand_output_grades(vector.grades, vector.grades, algebra.n, op="gp")),
            batch=batch,
            device=device,
            dtype=dtype,
            seed=seed,
        )

    if op_name == "bivector_vector_commutator":
        return _product_target(
            algebra,
            name=op_name,
            op="commutator_product",
            left_layout=bivector,
            right_layout=vector,
            output_layout=algebra.layout(
                expand_output_grades(bivector.grades, vector.grades, algebra.n, op="commutator_product")
            ),
            batch=batch,
            device=device,
            dtype=dtype,
            seed=seed,
        )

    if op_name == "bivector_bivector_commutator":
        return _product_target(
            algebra,
            name=op_name,
            op="commutator_product",
            left_layout=bivector,
            right_layout=bivector,
            output_layout=algebra.layout(
                expand_output_grades(bivector.grades, bivector.grades, algebra.n, op="commutator_product")
            ),
            batch=batch,
            device=device,
            dtype=dtype,
            seed=seed,
        )

    if op_name == "signature_norm_vector":
        executor = algebra.plan_signature_norm_squared(input_layout=vector)
        values = _randn((batch, vector.dim), device=device, dtype=dtype, seed=seed)
        return BenchTarget(
            name=op_name,
            family=executor.executor_family,
            op="signature_norm_squared",
            layout_case=op_name,
            module=executor,
            args=(values,),
            metadata={
                "category": "metric",
                "call_method": "plan_signature_norm_squared_executor",
                **_layout_metadata("input", vector),
                "output_lanes": 1,
                "pair_count": vector.dim,
            },
        )

    if op_name == "pseudoscalar_product_vector":
        output = algebra.layout((algebra.n - 1,))
        executor = algebra.plan_pseudoscalar_product(input_layout=vector, output_layout=output)
        values = _randn((batch, vector.dim), device=device, dtype=dtype, seed=seed)
        return BenchTarget(
            name=op_name,
            family=executor.executor_family,
            op="pseudoscalar_product",
            layout_case=op_name,
            module=executor,
            args=(values,),
            metadata={
                "category": "permutation",
                "call_method": "plan_pseudoscalar_product_executor",
                **_layout_metadata("input", vector),
                **_layout_metadata("output", executor.output_layout),
                "pair_count": vector.dim,
            },
        )

    if op_name == "bivector_exp":
        output_layout, output_selector = _exp_output_layout(algebra, exp_output_grades)
        executor = algebra.plan_bivector_exp(input_layout=bivector, output_layout=output_layout, **(exp_plan_kwargs or {}))
        values = _randn((batch, bivector.dim), device=device, dtype=dtype, seed=seed, scale=0.1)
        return BenchTarget(
            name=op_name,
            family=executor.executor_family,
            op="bivector_exp",
            layout_case=op_name,
            module=executor,
            args=(values,),
            metadata={
                "category": "exp",
                "call_method": "plan_bivector_exp_executor",
                **_layout_metadata("input", bivector),
                **_layout_metadata("output", executor.output_layout),
                "exp_output_grades_selector": output_selector,
                "pair_count": _exp_pair_count(executor),
                **_exp_executor_metadata(executor),
            },
        )

    if op_name == "versor_vector":
        handle = algebra.plan_versor_action(grade=2, input_layout=vector, output_layout=vector, parameter_layout=bivector)
        values = _randn((batch, channels, vector.dim), device=device, dtype=dtype, seed=seed)
        weights = _randn((channels, bivector.dim), device=device, dtype=dtype, seed=seed + 1, scale=0.1)
        return BenchTarget(
            name=op_name,
            family=handle.executor_family,
            op="versor_action",
            layout_case=op_name,
            module=handle,
            args=(values, weights),
            metadata={
                "category": "action_active",
                "call_method": "plan_versor_action_handle",
                **_layout_metadata("input", vector),
                **_layout_metadata("parameter", bivector),
                **_layout_metadata("output", vector),
            },
        )

    if op_name == "multi_versor_vector":
        handle = algebra.plan_multi_versor_action(grade=2, input_layout=vector, output_layout=vector, parameter_layout=bivector)
        values = _randn((batch, channels, vector.dim), device=device, dtype=dtype, seed=seed)
        weights = _randn((actions, bivector.dim), device=device, dtype=dtype, seed=seed + 1, scale=0.1)
        mix = _randn((channels, actions), device=device, dtype=dtype, seed=seed + 2)
        return BenchTarget(
            name=op_name,
            family=handle.executor_family,
            op="multi_versor_action",
            layout_case=op_name,
            module=handle,
            args=(values, weights, mix),
            metadata={
                "category": "action_active",
                "call_method": "plan_multi_versor_action_handle",
                **_layout_metadata("input", vector),
                **_layout_metadata("parameter", bivector),
                **_layout_metadata("output", vector),
                "actions": actions,
            },
        )

    if op_name == "paired_bivector_vector":
        handle = algebra.plan_paired_bivector_action(input_layout=vector, output_layout=vector, parameter_layout=bivector)
        values = _randn((batch, channels, vector.dim), device=device, dtype=dtype, seed=seed)
        left_weights = _randn((pairs, bivector.dim), device=device, dtype=dtype, seed=seed + 1, scale=0.1)
        right_weights = _randn((pairs, bivector.dim), device=device, dtype=dtype, seed=seed + 2, scale=0.1)
        channel_to_pair = torch.arange(channels, device=device, dtype=torch.long) % pairs
        return BenchTarget(
            name=op_name,
            family=handle.executor_family,
            op="paired_bivector_action",
            layout_case=op_name,
            module=handle,
            args=(values, left_weights, right_weights, channel_to_pair),
            metadata={
                "category": "action_active",
                "call_method": "plan_paired_bivector_action_handle",
                **_layout_metadata("input", vector),
                **_layout_metadata("parameter", bivector),
                **_layout_metadata("output", vector),
                "pairs": pairs,
            },
        )

    raise ValueError(f"unknown benchmark op {op_name!r}")


def _compile_callable(module: Callable[..., torch.Tensor] | nn.Module, mode: str) -> Callable[..., torch.Tensor] | nn.Module:
    if mode == "eager":
        return module
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available")
    if mode == "aot_eager":
        return torch.compile(module, backend="aot_eager", fullgraph=True)
    if mode == "inductor":
        return torch.compile(module, fullgraph=True)
    if mode == "reduce-overhead":
        return torch.compile(module, mode="reduce-overhead", fullgraph=True)
    raise ValueError(f"unknown compile mode {mode!r}")


def _benchmark_target(
    algebra: AlgebraContext,
    target: BenchTarget,
    *,
    mode: str,
    device: str,
    warmup: int,
    iterations: int,
    compile_cache_scope: str,
) -> dict[str, Any]:
    cache_before = _cache_snapshot(algebra)
    eager_timing = _time_callable(target.module, target.args, device=device, warmup=1, iterations=1)
    expected = eager_timing["output"]
    if mode == "eager":
        timing = _time_callable(target.module, target.args, device=device, warmup=warmup, iterations=iterations)
        cache_after = _cache_snapshot(algebra)
        return {
            "compile_ok": True,
            "cache_mutation": _cache_delta(cache_before, cache_after),
            "max_abs_diff": 0.0,
            "max_rel_diff": 0.0,
            "output_finite": bool(torch.isfinite(timing["output"]).all().item()),
            "compile_cache_reset_before": False,
            "compile_cache_reset_after": False,
            **_timing_fields(timing),
        }

    reset_before = _reset_compile_cache() if compile_cache_scope == "row" else False
    try:
        compiled = _compile_callable(target.module, mode)
        timing = _time_callable(compiled, target.args, device=device, warmup=warmup, iterations=iterations)
    finally:
        reset_after = _reset_compile_cache() if compile_cache_scope == "row" else False
    cache_after = _cache_snapshot(algebra)
    return {
        "compile_ok": True,
        "cache_mutation": _cache_delta(cache_before, cache_after),
        "max_abs_diff": _max_abs_diff(timing["output"], expected),
        "max_rel_diff": _max_rel_diff(timing["output"], expected),
        "output_finite": bool(torch.isfinite(timing["output"]).all().item()),
        "compile_cache_reset_before": reset_before,
        "compile_cache_reset_after": reset_after,
        **_timing_fields(timing),
    }


def _timing_fields(timing: dict[str, Any]) -> dict[str, Any]:
    output = timing["output"]
    return {
        "first_call_ms": timing["first_call_ms"],
        "median_ms": timing["median_ms"],
        "mean_ms": timing["mean_ms"],
        "std_ms": timing["std_ms"],
        "min_ms": timing["min_ms"],
        "max_ms": timing["max_ms"],
        "p10_ms": timing["p10_ms"],
        "p90_ms": timing["p90_ms"],
        "runs": timing["runs"],
        "samples_ms": timing["samples_ms"],
        "output_numel": int(output.numel()) if isinstance(output, torch.Tensor) else 0,
        "output_bytes": _output_bytes(output),
    }


def _base_row(
    *,
    args: argparse.Namespace,
    spec: SignatureSpec,
    dtype: torch.dtype,
    batch: int,
    target: BenchTarget,
    algebra_ms: float,
    plan_ms: float,
) -> dict[str, Any]:
    row = {
        "signature": spec.label,
        "n": spec.n,
        "device": args.device,
        "dtype": dtype_name(dtype),
        "batch": batch,
        "channels": args.channels,
        "actions": args.actions,
        "pairs": args.pairs,
        "target": target.name,
        "op": target.op,
        "layout_case": target.layout_case,
        "executor_family": target.family,
        "algebra_init_ms": algebra_ms,
        "cold_plan_ms": plan_ms,
    }
    row.update(_policy_metadata(args))
    row.update(target.metadata)
    row.update(_memory_estimate_fields(target, row, dtype))
    return row


def _skip_reason(algebra: AlgebraContext, *, op_name: str, max_full_lanes: int) -> str:
    full = algebra.spec.full_layout()
    if op_name.startswith("full_") or op_name == "full_sandwich":
        if not _full_layout_allowed(full, max_full_lanes):
            return "full_lane_cap"
    if op_name in {
        "bivector_vector_commutator",
        "bivector_bivector_commutator",
        "bivector_exp",
        "versor_vector",
        "multi_versor_vector",
        "paired_bivector_vector",
    } and algebra.n < 2:
        return "grade_absent"
    return "not_applicable"


def run_benchmarks(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    signatures = _resolve_signature_specs(args)
    dtypes = [resolve_dtype(DTYPES[name] if name in DTYPES else name) for name in _parse_csv(args.dtypes)]
    batches = _parse_int_csv(args.batch_sizes)
    ops = _parse_csv(args.ops)
    modes = _parse_csv(args.compile_modes)

    for spec in signatures:
        for dtype in dtypes:
            for batch in batches:
                for op_index, op_name in enumerate(ops):
                    seed = args.seed + spec.n * 1009 + batch * 37 + op_index * 17
                    try:
                        algebra, algebra_ms = _time_block(
                            lambda: _make_algebra(args, spec, dtype),
                            device=args.device,
                        )
                        target, plan_ms = _time_block(
                            lambda: _build_target(
                                algebra,
                                op_name=op_name,
                                batch=batch,
                                channels=args.channels,
                                actions=args.actions,
                                pairs=args.pairs,
                                max_full_lanes=args.max_full_lanes,
                                device=args.device,
                                dtype=dtype,
                                seed=seed,
                                exp_plan_kwargs=_exp_plan_kwargs(args),
                                exp_output_grades=args.exp_output_grades,
                            ),
                            device=args.device,
                        )
                        if target is None:
                            skip = {
                                "signature": spec.label,
                                "n": spec.n,
                                "device": args.device,
                                "dtype": dtype_name(dtype),
                                "batch": batch,
                                "target": op_name,
                                "reason": _skip_reason(algebra, op_name=op_name, max_full_lanes=args.max_full_lanes),
                            }
                            skip.update(_policy_metadata(args))
                            skips.append(skip)
                            print(
                                f"SKIP {spec.label} {dtype_name(dtype)} {op_name}: "
                                f"{skip['reason']}"
                            )
                            continue
                        base = _base_row(args=args, spec=spec, dtype=dtype, batch=batch, target=target, algebra_ms=algebra_ms, plan_ms=plan_ms)
                        for mode in modes:
                            row = dict(base)
                            row["compile_mode"] = mode
                            try:
                                row.update(
                                    _benchmark_target(
                                        algebra,
                                        target,
                                        mode=mode,
                                        device=args.device,
                                        warmup=args.warmup,
                                        iterations=args.iterations,
                                        compile_cache_scope=args.compile_cache_scope,
                                    )
                                )
                                rows.append(row)
                                _print_row(row)
                            except Exception as exc:  # noqa: BLE001 - benchmark rows record failures instead of aborting.
                                failure = dict(base)
                                failure.update({"compile_mode": mode, "error": repr(exc), "compile_ok": False})
                                failures.append(failure)
                                print(f"FAIL {spec.label} {dtype_name(dtype)} {op_name} {mode}: {exc}")
                    except Exception as exc:  # noqa: BLE001 - keep the benchmark matrix moving.
                        failure = {
                            "signature": spec.label,
                            "n": spec.n,
                            "device": args.device,
                            "dtype": dtype_name(dtype),
                            "batch": batch,
                            "target": op_name,
                            "error": repr(exc),
                        }
                        failures.append(failure)
                        print(f"FAIL {spec.label} {dtype_name(dtype)} {op_name}: {exc}")
                    finally:
                        _release_memory(args.device)
    return rows, failures, skips


def _print_row(row: dict[str, Any]) -> None:
    print(
        f"{row['signature']:<12s} {row['dtype']:<8s} {row['target']:<30s} "
        f"{row['compile_mode']:<12s} {row['executor_family']:<18s} "
        f"plan={float(row['cold_plan_ms']):7.3f}ms med={float(row['median_ms']):8.3f}ms "
        f"cache+={row['cache_mutation']} diff={float(row['max_abs_diff']):.2e}"
    )


def _ordered_modes(value: str) -> list[str]:
    modes = _parse_csv(value)
    if "eager" not in modes:
        return modes
    return ["eager"] + [mode for mode in modes if mode != "eager"]


def _diagnostic_suites(value: str) -> list[str]:
    suites = [suite for suite in _parse_csv(value) if suite.lower() not in {"none", "off"}]
    unknown = sorted(set(suites) - DIAGNOSTIC_SUITES)
    if unknown:
        raise ValueError(f"unknown diagnostic suite {unknown}; valid: {sorted(DIAGNOSTIC_SUITES)}")
    return suites


def _diagnostic_signature_specs(args: argparse.Namespace, dimension_range: str | None) -> list[SignatureSpec]:
    selected_range = args.dimension_range if dimension_range is None else dimension_range
    return _signatures_from_range(selected_range, args.signature_families)


def _diagnostic_dtypes(args: argparse.Namespace) -> list[torch.dtype]:
    return [resolve_dtype(DTYPES[name] if name in DTYPES else name) for name in _parse_csv(args.dtypes)]


def _diagnostic_gate(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if row.get("status") != "ok":
        return False
    if not bool(row.get("output_finite", True)):
        return False
    if not bool(row.get("grad_finite", True)):
        return False
    if float(row.get("max_abs_diff") or 0.0) > float(args.correctness_atol):
        return False
    if float(row.get("max_rel_diff") or 0.0) > float(args.correctness_rtol):
        return False
    if float(row.get("max_abs_error") or 0.0) > float(args.diagnostic_error_atol):
        return False
    if float(row.get("max_rel_error") or 0.0) > float(args.diagnostic_error_rtol):
        return False
    if float(row.get("norm_drift") or 0.0) > float(args.diagnostic_norm_drift_atol):
        return False
    if float(row.get("grade_leak") or 0.0) > float(args.diagnostic_grade_leak_atol):
        return False
    if float(row.get("unitarity_error") or 0.0) > float(args.diagnostic_error_atol):
        return False
    return True


def _diagnostic_failure_row(
    *,
    suite: str,
    args: argparse.Namespace,
    spec: SignatureSpec,
    dtype: torch.dtype,
    batch: int,
    target: str,
    exc: Exception,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "suite": suite,
        "status": "failed",
        "error": repr(exc),
        "signature": spec.label,
        "n": spec.n,
        "device": args.device,
        "dtype": dtype_name(dtype),
        "batch": batch,
        "target": target,
    }
    row.update(_policy_metadata(args))
    if extra:
        row.update(extra)
    row["diagnostic_gate_ok"] = False
    return row


def run_backward_diagnostics(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    signatures = _diagnostic_signature_specs(args, args.backward_dimension_range)
    dtypes = _diagnostic_dtypes(args)
    batches = _parse_int_csv(args.backward_batch_sizes or args.batch_sizes)
    ops = _parse_csv(args.backward_ops)
    modes = _ordered_modes(args.backward_compile_modes or args.compile_modes)

    for spec in signatures:
        for dtype in dtypes:
            for batch in batches:
                for op_index, op_name in enumerate(ops):
                    seed = args.seed + spec.n * 2801 + batch * 79 + op_index * 23
                    try:
                        algebra, algebra_ms = _time_block(
                            lambda: _make_algebra(args, spec, dtype),
                            device=args.device,
                        )
                        target, plan_ms = _time_block(
                            lambda: _build_target(
                                algebra,
                                op_name=op_name,
                                batch=batch,
                                channels=args.channels,
                                actions=args.actions,
                                pairs=args.pairs,
                                max_full_lanes=args.max_full_lanes,
                                device=args.device,
                                dtype=dtype,
                                seed=seed,
                                exp_plan_kwargs=_exp_plan_kwargs(args),
                                exp_output_grades=args.exp_output_grades,
                            ),
                            device=args.device,
                        )
                        if target is None:
                            row = {
                                "suite": "backward",
                                "status": "skipped",
                                "signature": spec.label,
                                "n": spec.n,
                                "device": args.device,
                                "dtype": dtype_name(dtype),
                                "batch": batch,
                                "target": op_name,
                                "reason": _skip_reason(algebra, op_name=op_name, max_full_lanes=args.max_full_lanes),
                            }
                            row.update(_policy_metadata(args))
                            row["diagnostic_gate_ok"] = not args.fail_on_skip
                            rows.append(row)
                            print(f"DIAG SKIP backward {spec.label} {dtype_name(dtype)} {op_name}: {row['reason']}")
                            continue

                        expected = target.module(*target.args).detach()
                        base = _base_row(
                            args=args,
                            spec=spec,
                            dtype=dtype,
                            batch=batch,
                            target=target,
                            algebra_ms=algebra_ms,
                            plan_ms=plan_ms,
                        )
                        for mode in modes:
                            row = dict(base)
                            row.update({"suite": "backward", "status": "ok", "compile_mode": mode})
                            cache_before = _cache_snapshot(algebra)
                            reset_before = False
                            reset_after = False
                            try:
                                if mode == "eager":
                                    callable_target = target.module
                                else:
                                    reset_before = _reset_compile_cache() if args.compile_cache_scope == "row" else False
                                    callable_target = _compile_callable(target.module, mode)
                                timing = _time_forward_backward(
                                    callable_target,
                                    target.args,
                                    device=args.device,
                                    warmup=args.backward_warmup,
                                    iterations=args.backward_iterations,
                                )
                            finally:
                                if mode != "eager" and args.compile_cache_scope == "row":
                                    reset_after = _reset_compile_cache()
                            cache_after = _cache_snapshot(algebra)
                            row.update(
                                {
                                    "cache_mutation": _cache_delta(cache_before, cache_after),
                                    "max_abs_diff": _max_abs_diff(timing["output"], expected),
                                    "max_rel_diff": _max_rel_diff(timing["output"], expected),
                                    "output_finite": bool(torch.isfinite(timing["output"]).all().item()),
                                    "grad_norm": timing["grad_norm"],
                                    "grad_finite": timing["grad_finite"],
                                    "compile_cache_reset_before": reset_before,
                                    "compile_cache_reset_after": reset_after,
                                    **_timing_fields(timing),
                                }
                            )
                            row["diagnostic_gate_ok"] = _diagnostic_gate(row, args)
                            rows.append(row)
                            print(
                                f"DIAG backward {spec.label:<12s} {dtype_name(dtype):<8s} {op_name:<30s} "
                                f"{mode:<12s} med={float(row['median_ms']):8.3f}ms "
                                f"grad={float(row['grad_norm']):.2e} gate={row['diagnostic_gate_ok']}"
                            )
                    except Exception as exc:  # noqa: BLE001 - diagnostic rows preserve the matrix.
                        row = _diagnostic_failure_row(
                            suite="backward",
                            args=args,
                            spec=spec,
                            dtype=dtype,
                            batch=batch,
                            target=op_name,
                            exc=exc,
                        )
                        rows.append(row)
                        print(f"DIAG FAIL backward {spec.label} {dtype_name(dtype)} {op_name}: {exc}")
                    finally:
                        _release_memory(args.device)
    return rows


def run_cumulative_diagnostics(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    signatures = [spec for spec in _diagnostic_signature_specs(args, args.cumulative_dimension_range) if spec.n >= 4]
    dtypes = _diagnostic_dtypes(args)
    batches = _parse_int_csv(args.cumulative_batch_sizes)
    sample_steps = set(_sample_steps(args.chain_steps, args.chain_samples))

    for spec in signatures:
        full_lanes = 1 << spec.n
        if args.diagnostic_max_full_lanes > 0 and full_lanes > args.diagnostic_max_full_lanes:
            for dtype in dtypes:
                row = {
                    "suite": "cumulative",
                    "status": "skipped",
                    "signature": spec.label,
                    "n": spec.n,
                    "device": args.device,
                    "dtype": dtype_name(dtype),
                    "batch": 0,
                    "target": "cumulative_action",
                    "reason": "diagnostic_full_lane_cap",
                    "diagnostic_gate_ok": not args.fail_on_skip,
                }
                row.update(_policy_metadata(args))
                rows.append(row)
            continue

        for batch in batches:
            ref_alg = _make_algebra(args, spec, torch.float64, device="cpu")
            ref_full = ref_alg.spec.full_layout()
            ref_even = ref_alg.layout(range(0, ref_alg.n + 1, 2))
            ref_b = _make_commuting_bivector(ref_alg, batch, args.chain_bivector_scale)
            ref_reverse = ref_alg.plan_unary(op="reverse", input_layout=ref_even, output_layout=ref_even)
            ref_action = ref_alg.plan_sandwich_action(layout=ref_full, dtype=torch.float64, device="cpu")
            ref_x0 = torch.zeros(batch, ref_full.dim, dtype=torch.float64)
            ref_x0[:, 1] = 1.0

            for dtype in dtypes:
                try:
                    algebra = _make_algebra(args, spec, dtype)
                    full = algebra.spec.full_layout()
                    even = algebra.layout(range(0, algebra.n + 1, 2))
                    vector = algebra.layout((1,))
                    b = _make_commuting_bivector(algebra, batch, args.chain_bivector_scale)
                    rotor = _exact_commuting_exp(algebra, -0.5 * b, even)
                    reverse = algebra.plan_unary(op="reverse", input_layout=even, output_layout=even)
                    action = algebra.plan_sandwich_action(layout=full, dtype=dtype, device=args.device)
                    left = even.full(rotor)
                    right = even.full(reverse(rotor))
                    x = torch.zeros(batch, full.dim, device=args.device, dtype=dtype)
                    x[:, 1] = 1.0
                    _sync(args.device)
                    start = _now_ms()
                    last_row: dict[str, Any] | None = None
                    for step in range(1, args.chain_steps + 1):
                        x = action.batched(left, x, right)
                        if step not in sample_steps:
                            continue
                        ref_rotor = _exact_commuting_exp(ref_alg, -0.5 * float(step) * ref_b, ref_even)
                        ref_left = ref_even.full(ref_rotor)
                        ref_right = ref_even.full(ref_reverse(ref_rotor))
                        ref_x = ref_action.batched(ref_left, ref_x0, ref_right)
                        errors = _error_stats(x, ref_x)
                        norm = x.detach().float().norm(dim=-1)
                        norm_drift = float((norm - 1.0).abs().max().item()) if spec.q == 0 and spec.r == 0 else 0.0
                        grade_leak = _grade_leak(x, full, 1)
                        row = {
                            "suite": "cumulative",
                            "status": "ok",
                            "signature": spec.label,
                            "n": spec.n,
                            "device": args.device,
                            "dtype": dtype_name(dtype),
                            "batch": batch,
                            "target": "cumulative_action",
                            "step": step,
                            "chain_steps": args.chain_steps,
                            "elapsed_ms": _now_ms() - start,
                            "norm": float(norm.mean().item()),
                            "norm_drift": norm_drift,
                            "grade_leak": grade_leak,
                            "output_finite": bool(torch.isfinite(x).all().item()),
                            **errors,
                        }
                        row.update(_policy_metadata(args))
                        row["diagnostic_gate_ok"] = _diagnostic_gate(row, args)
                        rows.append(row)
                        last_row = row
                    if last_row is not None:
                        print(
                            f"DIAG cumulative {spec.label:<12s} {dtype_name(dtype):<8s} b={batch:<4d} "
                            f"steps={args.chain_steps:<5d} err={float(last_row['max_abs_error']):.2e} "
                            f"leak={float(last_row['grade_leak']):.2e} gate={last_row['diagnostic_gate_ok']}"
                        )
                except Exception as exc:  # noqa: BLE001
                    rows.append(
                        _diagnostic_failure_row(
                            suite="cumulative",
                            args=args,
                            spec=spec,
                            dtype=dtype,
                            batch=batch,
                            target="cumulative_action",
                            exc=exc,
                        )
                    )
                    print(f"DIAG FAIL cumulative {spec.label} {dtype_name(dtype)} b={batch}: {exc}")
                finally:
                    _release_memory(args.device)
    return rows


def run_convergence_diagnostics(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    signatures = [spec for spec in _diagnostic_signature_specs(args, args.convergence_dimension_range) if spec.n >= 4]
    dtypes = _diagnostic_dtypes(args)
    iterations = _parse_int_csv(args.convergence_iters)

    for spec in signatures:
        for dtype in dtypes:
            ref_alg = _make_algebra(args, spec, torch.float64, device="cpu")
            ref_even = ref_alg.layout(range(0, ref_alg.n + 1, 2))
            ref_b = _make_commuting_bivector(ref_alg, args.diagnostic_batch_size, args.diagnostic_bivector_scale)
            ref_rotor = _exact_commuting_exp(ref_alg, ref_b, ref_even)
            for spectral_max_planes_probe in iterations:
                try:
                    algebra = _make_algebra(args, spec, dtype)
                    bivector = algebra.layout((2,))
                    even = algebra.layout(range(0, algebra.n + 1, 2))
                    exp_kwargs = _exp_plan_kwargs(args)
                    if args.exp_spectral_max_planes is None:
                        exp_kwargs["spectral_max_planes"] = spectral_max_planes_probe
                    executor = algebra.plan_bivector_exp(input_layout=bivector, output_layout=even, **exp_kwargs)
                    b = _make_commuting_bivector(algebra, args.diagnostic_batch_size, args.diagnostic_bivector_scale)
                    timing = _time_callable(
                        executor,
                        (b,),
                        device=args.device,
                        warmup=args.diagnostic_warmup,
                        iterations=args.diagnostic_iterations,
                    )
                    output = timing["output"]
                    reverse = algebra.plan_unary(op="reverse", input_layout=even, output_layout=even)
                    product = algebra.plan_product(op="gp", left_layout=even, right_layout=even, output_layout=even)
                    unit = product(output, reverse(output))
                    identity = _identity_for_layout(even, like=unit)
                    errors = _error_stats(output, ref_rotor)
                    row = {
                        "suite": "convergence",
                        "status": "ok",
                        "signature": spec.label,
                        "n": spec.n,
                        "device": args.device,
                        "dtype": dtype_name(dtype),
                        "batch": args.diagnostic_batch_size,
                        "target": "bivector_exp_convergence",
                        "fixed_iterations": spectral_max_planes_probe,
                        "spectral_max_planes_probe": spectral_max_planes_probe,
                        "executor_family": executor.executor_family,
                        "convergence_active": executor.executor_family == "spectral_local",
                        "spectral_local_active": executor.executor_family == "spectral_local",
                        "unitarity_error": _error_stats(unit, identity)["max_abs_error"],
                        "output_finite": bool(torch.isfinite(output).all().item()),
                        **_exp_executor_metadata(executor),
                        **errors,
                        **_timing_fields(timing),
                    }
                    row.update(_policy_metadata(args))
                    row["diagnostic_gate_ok"] = _diagnostic_gate(row, args)
                    rows.append(row)
                    print(
                        f"DIAG convergence {spec.label:<12s} {dtype_name(dtype):<8s} "
                        f"k={spectral_max_planes_probe:<4d} {executor.executor_family:<16s} "
                        f"err={float(row['max_abs_error']):.2e} gate={row['diagnostic_gate_ok']}"
                    )
                except Exception as exc:  # noqa: BLE001
                    rows.append(
                        _diagnostic_failure_row(
                            suite="convergence",
                            args=args,
                            spec=spec,
                            dtype=dtype,
                            batch=args.diagnostic_batch_size,
                            target="bivector_exp_convergence",
                            exc=exc,
                            extra={
                                "fixed_iterations": spectral_max_planes_probe,
                                "spectral_max_planes_probe": spectral_max_planes_probe,
                            },
                        )
                    )
                    print(
                        f"DIAG FAIL convergence {spec.label} {dtype_name(dtype)} "
                        f"k={spectral_max_planes_probe}: {exc}"
                    )
                finally:
                    _release_memory(args.device)
    return rows


def run_diagnostics(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    suites = _diagnostic_suites(args.extra_suites)
    if "backward" in suites:
        rows.extend(run_backward_diagnostics(args))
    if "cumulative" in suites:
        rows.extend(run_cumulative_diagnostics(args))
    if "convergence" in suites:
        rows.extend(run_convergence_diagnostics(args))
    return rows


def _calibration_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("device", "")),
        str(row.get("dtype", "")),
        str(row.get("signature", "")),
        str(row.get("target", "")),
        str(row.get("op", "")),
        str(row.get("layout_case", "")),
    )


def _estimated_working_set_bytes(row: dict[str, Any]) -> int:
    return int(row.get("static_estimated_bytes") or 0) + int(row.get("output_bytes") or 0)


def _row_valid_for_validation(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if not bool(row.get("compile_ok", False)):
        return False
    if not bool(row.get("output_finite", False)):
        return False
    if int(row.get("cache_mutation") or 0) > int(args.max_cache_mutation):
        return False
    if float(row.get("max_abs_diff") or 0.0) > float(args.correctness_atol):
        return False
    if float(row.get("max_rel_diff") or 0.0) > float(args.correctness_rtol):
        return False
    max_bytes = int(args.max_estimated_bytes)
    return max_bytes <= 0 or _estimated_working_set_bytes(row) <= max_bytes


def _calibration_summary_rows(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    skips: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_calibration_key(row), []).append(row)

    failure_counts: dict[tuple[str, str, str, str, str, str], int] = {}
    for failure in failures:
        key = _calibration_key(failure)
        failure_counts[key] = failure_counts.get(key, 0) + 1

    skip_counts: dict[tuple[str, str, str, str, str, str], int] = {}
    for skip in skips:
        key = _calibration_key(skip)
        skip_counts[key] = skip_counts.get(key, 0) + 1

    summaries: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        valid_rows = [row for row in group_rows if _row_valid_for_validation(row, args)]
        ranked_rows = valid_rows if valid_rows else group_rows
        best = min(ranked_rows, key=lambda row: float(row.get("median_ms", float("inf"))))
        eager_rows = [row for row in group_rows if row.get("compile_mode") == "eager" and bool(row.get("compile_ok", False))]
        compiled_rows = [row for row in group_rows if row.get("compile_mode") != "eager"]
        valid_compiled_rows = [row for row in compiled_rows if _row_valid_for_validation(row, args)]

        eager_median = min((float(row["median_ms"]) for row in eager_rows), default=float("nan"))
        best_compiled_median = min((float(row["median_ms"]) for row in valid_compiled_rows), default=float("nan"))
        best_median = float(best.get("median_ms", float("nan")))
        speedup_vs_eager = eager_median / best_median if math.isfinite(eager_median) and best_median > 0 else float("nan")

        correctness_gate_ok = all(float(row.get("max_abs_diff") or 0.0) <= float(args.correctness_atol) for row in group_rows)
        relative_gate_ok = all(float(row.get("max_rel_diff") or 0.0) <= float(args.correctness_rtol) for row in group_rows)
        finite_gate_ok = all(bool(row.get("output_finite", False)) for row in group_rows)
        cache_gate_ok = all(int(row.get("cache_mutation") or 0) <= int(args.max_cache_mutation) for row in group_rows)
        max_bytes = int(args.max_estimated_bytes)
        memory_gate_ok = max_bytes <= 0 or all(_estimated_working_set_bytes(row) <= max_bytes for row in group_rows)
        failure_count = failure_counts.get(key, 0)
        fullgraph_gate_ok = failure_count == 0
        if compiled_rows:
            fullgraph_gate_ok = (
                failure_count == 0
                and bool(valid_compiled_rows)
                and all(bool(row.get("compile_ok", False)) for row in compiled_rows)
            )

        summaries.append(
            {
                "device": key[0],
                "dtype": key[1],
                "signature": key[2],
                "target": key[3],
                "op": key[4],
                "layout_case": key[5],
                "observed_best_executor_family": best.get("executor_family"),
                "observed_best_compile_mode": best.get("compile_mode"),
                "observed_best_median_ms": best_median,
                "eager_median_ms": eager_median,
                "best_compiled_median_ms": best_compiled_median,
                "speedup_vs_eager": speedup_vs_eager,
                "cold_plan_ms": float(best.get("cold_plan_ms", float("nan"))),
                "cache_mutation": int(best.get("cache_mutation") or 0),
                "max_abs_diff": float(best.get("max_abs_diff") or 0.0),
                "arg_bytes": int(best.get("arg_bytes") or 0),
                "output_bytes": int(best.get("output_bytes") or 0),
                "pair_buffer_estimate_bytes": int(best.get("pair_buffer_estimate_bytes") or 0),
                "estimated_working_set_bytes": _estimated_working_set_bytes(best),
                "pair_count": int(best.get("pair_count") or 0),
                "row_count": len(group_rows),
                "valid_row_count": len(valid_rows),
                "failure_count": failure_count,
                "skip_count": skip_counts.get(key, 0),
                "fullgraph_gate_ok": fullgraph_gate_ok,
                "correctness_gate_ok": correctness_gate_ok,
                "relative_gate_ok": relative_gate_ok,
                "finite_gate_ok": finite_gate_ok,
                "cache_gate_ok": cache_gate_ok,
                "memory_gate_ok": memory_gate_ok,
                "validation_gate_ok": bool(valid_rows)
                and fullgraph_gate_ok
                and correctness_gate_ok
                and relative_gate_ok
                and finite_gate_ok
                and cache_gate_ok
                and memory_gate_ok,
            }
        )
    return summaries


def _print_calibration_summary(calibration_rows: list[dict[str, Any]]) -> None:
    if not calibration_rows:
        return
    print("\nCalibration summary:")
    for row in calibration_rows:
        print(
            f"{row['signature']:<12s} {row['dtype']:<8s} {row['target']:<30s} "
            f"{str(row['observed_best_compile_mode']):<12s} {str(row['observed_best_executor_family']):<18s} "
            f"med={float(row['observed_best_median_ms']):8.3f}ms "
            f"speedup={float(row['speedup_vs_eager']):6.2f}x gate={row['validation_gate_ok']}"
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return dtype_name(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_artifacts(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    skips: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_rows = _calibration_summary_rows(rows, failures, skips, args=args)
    csv_path = output_dir / "rows.csv"
    all_columns = sorted({key for row in rows + failures + skips for key in row})
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    calibration_csv_path = output_dir / "calibration_summary.csv"
    calibration_columns = sorted({key for row in calibration_rows for key in row})
    with calibration_csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=calibration_columns)
        writer.writeheader()
        for row in calibration_rows:
            writer.writerow(row)

    (output_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=_json_default))
    (output_dir / "calibration_summary.json").write_text(json.dumps(calibration_rows, indent=2, default=_json_default))
    (output_dir / "failures.json").write_text(json.dumps(failures, indent=2, default=_json_default))
    (output_dir / "skips.json").write_text(json.dumps(skips, indent=2, default=_json_default))
    if diagnostics:
        diagnostics_csv_path = output_dir / "diagnostics.csv"
        diagnostics_columns = sorted({key for row in diagnostics for key in row})
        with diagnostics_csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=diagnostics_columns)
            writer.writeheader()
            for row in diagnostics:
                writer.writerow(row)
    (output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, default=_json_default))
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "torch_version": torch.__version__,
        "device": args.device,
        "config": vars(args),
        "policy": _policy_metadata(args),
        "row_count": len(rows),
        "failure_count": len(failures),
        "skip_count": len(skips),
        "diagnostic_row_count": len(diagnostics),
        "diagnostic_failure_count": sum(1 for row in diagnostics if row.get("status") == "failed"),
        "diagnostic_gate_failure_count": sum(1 for row in diagnostics if not bool(row.get("diagnostic_gate_ok", True))),
        "calibration_row_count": len(calibration_rows),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_default))
    (output_dir / "summary.md").write_text(_summary_markdown(rows, failures, skips, diagnostics, calibration_rows, metadata))
    print(f"Wrote benchmark artifacts to {output_dir}")


def _summary_markdown(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    skips: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    lines = [
        "# benchmark_core Summary",
        "",
        f"- created: {metadata['created_at']}",
        f"- torch: {metadata['torch_version']}",
        f"- device: {metadata['device']}",
        f"- rows: {len(rows)}",
        f"- failures: {len(failures)}",
        f"- skips: {len(skips)}",
        f"- diagnostic rows: {len(diagnostics)}",
        f"- calibration rows: {len(calibration_rows)}",
        f"- tier: {metadata['policy']['tier']}",
        f"- dimension range: {metadata['policy']['dimension_range']}",
        f"- policy unlocked: {metadata['policy']['policy_unlocked']}",
        "",
        "## Calibration Summary",
        "",
        "| signature | dtype | target | observed best mode | family | median ms | speedup vs eager | workset bytes | gate |",
        "|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in calibration_rows:
        speedup = float(row["speedup_vs_eager"])
        speedup_text = f"{speedup:.3f}" if math.isfinite(speedup) else ""
        lines.append(
            f"| {row['signature']} | {row['dtype']} | {row['target']} | {row['observed_best_compile_mode']} | "
            f"{row['observed_best_executor_family']} | {float(row['observed_best_median_ms']):.4f} | {speedup_text} | "
            f"{row['estimated_working_set_bytes']} | {row['validation_gate_ok']} |"
        )
    if skips:
        lines += ["", "## Skips", "", "| signature | dtype | batch | target | reason |", "|---|---|---:|---|---|"]
        for skip in skips[:80]:
            lines.append(
                f"| {skip.get('signature', '')} | {skip.get('dtype', '')} | {skip.get('batch', '')} | "
                f"{skip.get('target', '')} | {skip.get('reason', '')} |"
            )
        if len(skips) > 80:
            lines.append(f"| ... | ... | ... | ... | {len(skips) - 80} more in `skips.json` |")
    if diagnostics:
        lines += [
            "",
            "## Diagnostics",
            "",
            "| suite | signature | dtype | target | mode/step | median ms | error | gate |",
            "|---|---|---|---|---|---:|---:|---|",
        ]
        for row in diagnostics[:120]:
            mode_or_step = row.get("compile_mode", row.get("step", row.get("fixed_iterations", "")))
            median = row.get("median_ms", "")
            median_text = f"{float(median):.4f}" if isinstance(median, (int, float)) and math.isfinite(float(median)) else ""
            error = row.get("max_abs_error", row.get("max_abs_diff", ""))
            error_text = f"{float(error):.3e}" if isinstance(error, (int, float)) and math.isfinite(float(error)) else ""
            lines.append(
                f"| {row.get('suite', '')} | {row.get('signature', '')} | {row.get('dtype', '')} | "
                f"{row.get('target', '')} | {mode_or_step} | {median_text} | {error_text} | "
                f"{row.get('diagnostic_gate_ok', '')} |"
            )
        if len(diagnostics) > 120:
            lines.append(f"| ... | ... | ... | ... | ... | ... | ... | {len(diagnostics) - 120} more in `diagnostics.json` |")
    lines += [
        "",
        "## Fastest Rows By Target",
        "",
        "| signature | dtype | target | mode | family | median ms | cold plan ms | cache mutation |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["signature"]), str(row["dtype"]), str(row["target"]))
        if key not in best or float(row["median_ms"]) < float(best[key]["median_ms"]):
            best[key] = row
    for row in sorted(best.values(), key=lambda r: (r["signature"], r["dtype"], r["target"])):
        lines.append(
            f"| {row['signature']} | {row['dtype']} | {row['target']} | {row['compile_mode']} | "
            f"{row['executor_family']} | {float(row['median_ms']):.4f} | {float(row['cold_plan_ms']):.4f} | "
            f"{row['cache_mutation']} |"
        )
    if failures:
        lines += ["", "## Failures", "", "| signature | dtype | target | mode | error |", "|---|---|---|---|---|"]
        for failure in failures[:50]:
            lines.append(
                f"| {failure.get('signature', '')} | {failure.get('dtype', '')} | {failure.get('target', '')} | "
                f"{failure.get('compile_mode', '')} | {failure.get('error', '')} |"
            )
    lines.append("")
    return "\n".join(lines)


def _apply_tier_defaults(args: argparse.Namespace) -> None:
    if args.quick:
        args.tier = "smoke"
    if args.tier not in TIER_PRESETS:
        raise ValueError(f"unknown tier {args.tier!r}; valid: {sorted(TIER_PRESETS)}")
    preset = TIER_PRESETS[args.tier]
    for field in (
        "dimension_range",
        "signature_families",
        "batch_sizes",
        "ops",
        "compile_modes",
        "warmup",
        "iterations",
        "channels",
        "actions",
        "pairs",
        "max_full_lanes",
    ):
        if getattr(args, field) is None:
            setattr(args, field, getattr(preset, field))
    if args.unlock_policy_limits and args.max_full_lanes == preset.max_full_lanes:
        args.max_full_lanes = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda, cuda:0, or auto")
    parser.add_argument("--dtypes", default="float32", help="comma-separated dtype names")
    parser.add_argument("--tier", default="standard", choices=sorted(TIER_PRESETS), help="benchmark matrix preset")
    parser.add_argument("--dimension-range", default=None, help="inclusive n range such as 3:8; filled by --tier when omitted")
    parser.add_argument("--n-min", type=int, default=None, help="optional lower n filter applied after tier/signature resolution")
    parser.add_argument("--n-max", type=int, default=None, help="optional upper n filter applied after tier/signature resolution")
    parser.add_argument(
        "--signature-families",
        default=None,
        help="comma list for generated signatures: euclidean,minkowski,degenerate",
    )
    parser.add_argument("--signatures", default=None, help="explicit comma-separated n, p:q, or p:q:r signatures")
    parser.add_argument("--batch-sizes", default=None, help="comma-separated batch sizes; filled by --tier when omitted")
    parser.add_argument("--ops", default=None, help="comma-separated benchmark targets; filled by --tier when omitted")
    parser.add_argument("--compile-modes", default=None, help="eager,aot_eager,inductor,reduce-overhead")
    parser.add_argument("--channels", type=int, default=None, help="channel count for action benchmarks")
    parser.add_argument("--actions", type=int, default=None, help="action count for multi-versor benchmarks")
    parser.add_argument("--pairs", type=int, default=None, help="rotor-pair count for paired-bivector benchmarks")
    parser.add_argument("--max-full-lanes", type=int, default=None, help="skip full-layout targets above this lane count; 0 disables this benchmark cap")
    parser.add_argument("--warmup", type=int, default=None, help="warmup calls before timed iterations")
    parser.add_argument("--iterations", type=int, default=None, help="timed calls per benchmark row")
    parser.add_argument(
        "--compile-cache-scope",
        default="row",
        choices=("row", "global"),
        help="row resets torch.compile caches around each compiled row; global preserves cache pressure across rows",
    )
    parser.add_argument("--correctness-atol", type=float, default=1e-4, help="max_abs_diff allowed by calibration summary")
    parser.add_argument("--correctness-rtol", type=float, default=1e-4, help="max_rel_diff allowed by calibration summary")
    parser.add_argument("--max-cache-mutation", type=int, default=0, help="cache entries allowed during measured forward")
    parser.add_argument("--exp-spectral-max-planes", type=int, default=None, help="override spectral-local bivector exp plane cap")
    parser.add_argument("--exp-spectral-tol-abs", type=float, default=None, help="override spectral-local absolute tail tolerance")
    parser.add_argument("--exp-spectral-tol-rel", type=float, default=None, help="override spectral-local relative tail tolerance")
    parser.add_argument(
        "--exp-output-grades",
        default="even",
        help="bivector exp output layout for benchmark rows: even, full, or comma-separated grades such as 0,2,4",
    )
    parser.add_argument(
        "--exp-spectral-dominant-rel",
        type=float,
        default=None,
        help="override dominant-plane relative cutoff for spectral-local exp",
    )
    parser.add_argument(
        "--exp-spectral-disable-degenerate",
        action="store_true",
        help="route degenerate signatures away from spectral-local exp",
    )
    parser.add_argument(
        "--exp-spectral-disable-truncated-degenerate",
        action="store_true",
        help="require full local coverage for degenerate spectral-local exp",
    )
    parser.add_argument(
        "--extra-suites",
        default="",
        help="optional diagnostics: backward,cumulative,convergence",
    )
    parser.add_argument("--backward-ops", default="vector_gp,bivector_exp,versor_vector", help="diagnostic backward targets")
    parser.add_argument("--backward-dimension-range", default=None, help="dimension range for backward diagnostics")
    parser.add_argument("--backward-batch-sizes", default=None, help="batch sizes for backward diagnostics")
    parser.add_argument("--backward-compile-modes", default=None, help="compile modes for backward diagnostics")
    parser.add_argument("--backward-warmup", type=int, default=1, help="backward diagnostic warmup calls")
    parser.add_argument("--backward-iterations", type=int, default=3, help="backward diagnostic timed calls")
    parser.add_argument("--cumulative-dimension-range", default="4:6", help="dimension range for cumulative diagnostics")
    parser.add_argument("--cumulative-batch-sizes", default="4", help="batch sizes for cumulative diagnostics")
    parser.add_argument("--chain-steps", type=int, default=128, help="steps in cumulative action diagnostics")
    parser.add_argument("--chain-samples", type=int, default=8, help="log-spaced cumulative samples")
    parser.add_argument("--chain-bivector-scale", type=float, default=0.025, help="controlled bivector scale for cumulative diagnostics")
    parser.add_argument("--convergence-dimension-range", default="4:6", help="dimension range for exp convergence diagnostics")
    parser.add_argument(
        "--convergence-iters",
        default="1,2,4,8,16,32,64",
        help="spectral plane-cap probes for exp convergence diagnostics",
    )
    parser.add_argument("--diagnostic-batch-size", type=int, default=4, help="batch size for convergence diagnostics")
    parser.add_argument("--diagnostic-bivector-scale", type=float, default=0.15, help="controlled bivector scale for convergence diagnostics")
    parser.add_argument("--diagnostic-warmup", type=int, default=1, help="convergence diagnostic warmup calls")
    parser.add_argument("--diagnostic-iterations", type=int, default=3, help="convergence diagnostic timed calls")
    parser.add_argument(
        "--diagnostic-max-full-lanes",
        type=int,
        default=None,
        help="full-lane cap for diagnostics; defaults to --max-full-lanes",
    )
    parser.add_argument("--diagnostic-error-atol", type=float, default=1e-3, help="diagnostic max absolute error gate")
    parser.add_argument("--diagnostic-error-rtol", type=float, default=1e-3, help="diagnostic max relative error gate")
    parser.add_argument("--diagnostic-norm-drift-atol", type=float, default=1e-3, help="Euclidean cumulative norm drift gate")
    parser.add_argument("--diagnostic-grade-leak-atol", type=float, default=1e-3, help="cumulative off-grade leak gate")
    parser.add_argument(
        "--max-estimated-bytes",
        type=int,
        default=0,
        help="optional estimated working-set byte cap for calibration summary; 0 disables the memory gate",
    )
    parser.add_argument(
        "--unlock-policy-limits",
        action="store_true",
        help="raise planner lane/pair limits and full-table policy limits so requested cases can execute",
    )
    parser.add_argument("--max-plan-lanes", type=int, default=None, help="override planner max_lanes for this benchmark")
    parser.add_argument("--max-plan-pairs", type=int, default=None, help="override planner max_pairs for this benchmark")
    parser.add_argument("--full-table-max-lanes", type=int, default=None, help="override product policy full_table_max_lanes")
    parser.add_argument("--fail-on-error", action="store_true", help="exit nonzero if any benchmark row fails")
    parser.add_argument("--fail-on-gate", action="store_true", help="exit nonzero if any calibration validation gate fails")
    parser.add_argument("--fail-on-skip", action="store_true", help="exit nonzero if any case is skipped")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="benchmarks/results", help="artifact root")
    parser.add_argument("--no-save", action="store_true", help="print rows without writing artifacts")
    parser.add_argument("--quick", action="store_true", help="small smoke benchmark matrix")
    args = parser.parse_args()
    args.device = _resolve_device(args.device)
    _apply_tier_defaults(args)
    if args.diagnostic_max_full_lanes is None:
        args.diagnostic_max_full_lanes = args.max_full_lanes
    return args


def main() -> None:
    args = parse_args()
    rows, failures, skips = run_benchmarks(args)
    diagnostics = run_diagnostics(args)
    if not rows:
        if not skips and not diagnostics:
            raise SystemExit("No benchmark rows were produced.")
        if not args.no_save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(args.out) / f"benchmark_core_{timestamp}"
            _write_artifacts(rows, failures, skips, diagnostics, args=args, output_dir=output_dir)
        if args.fail_on_skip:
            raise SystemExit(f"{len(skips)} benchmark skips")
        diagnostic_failures = [row for row in diagnostics if row.get("status") == "failed"]
        if args.fail_on_error and diagnostic_failures:
            raise SystemExit(f"{len(diagnostic_failures)} diagnostic failures")
        if args.fail_on_gate and not all(bool(row.get("diagnostic_gate_ok", True)) for row in diagnostics):
            raise SystemExit("one or more diagnostic gates failed")
        return
    calibration_rows = _calibration_summary_rows(rows, failures, skips, args=args)
    _print_calibration_summary(calibration_rows)
    if not args.no_save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.out) / f"benchmark_core_{timestamp}"
        _write_artifacts(rows, failures, skips, diagnostics, args=args, output_dir=output_dir)
    diagnostic_failures = [row for row in diagnostics if row.get("status") == "failed"]
    if args.fail_on_error and failures:
        raise SystemExit(f"{len(failures)} benchmark failures")
    if args.fail_on_error and diagnostic_failures:
        raise SystemExit(f"{len(diagnostic_failures)} diagnostic failures")
    if args.fail_on_skip and skips:
        raise SystemExit(f"{len(skips)} benchmark skips")
    if args.fail_on_gate and not all(bool(row.get("validation_gate_ok")) for row in calibration_rows):
        raise SystemExit("one or more calibration gates failed")
    if args.fail_on_gate and not all(bool(row.get("diagnostic_gate_ok", True)) for row in diagnostics):
        raise SystemExit("one or more diagnostic gates failed")


if __name__ == "__main__":
    main()
