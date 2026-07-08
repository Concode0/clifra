#!/usr/bin/env python3
"""Generate CLIFRA benchmark artifacts and MkDocs pages.

This benchmark is intentionally independent from ``benchmark_core.py``. It is
designed for documentation: every run produces machine-readable artifacts and
Markdown pages under ``docs/benchmarks`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from clifra import make_algebra
from clifra.core.foundation.basis import expand_output_grades
from clifra.core.foundation.device import dtype_name, resolve_device, resolve_dtype
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout
from clifra.core.planning.exp import (
    BivectorExpExecutionPolicy,
    select_bivector_exp_executor_family,
    spectral_exp_preselection,
)
from clifra.core.planning.policy import PlanningLimits, ProductExecutionPolicy

DEFAULT_OPERATIONS = (
    "vector_gp",
    "bivector_vector_commutator",
    "bivector_bivector_commutator",
    "bivector_exp",
    "signature_norm_default",
    "reverse_default",
    "grade_involution_default",
    "pseudoscalar_product_vector",
    "versor_vector",
)
MAX_DENSITY_OPERATIONS = DEFAULT_OPERATIONS
EXP_RELATED_OPERATIONS = {"bivector_exp", "multi_versor_vector", "paired_bivector_vector"}
MATRIX_EXP_FAMILIES = {"left_matrix_exp", "cpu_matrix_exp"}
BENCHMARK_SPECTRAL_MAX_PLANES = 4
BENCHMARK_SPECTRAL_TRANSITION_N = 8
MAX_DENSITY_MIN_N = 2
MAX_DENSITY_MAX_N = 63
UNLOCKED_LIMIT = 1 << 62
BENCHMARK_PLANNING_LIMITS = PlanningLimits(
    warn_lanes=UNLOCKED_LIMIT,
    max_lanes=UNLOCKED_LIMIT,
    warn_pairs=UNLOCKED_LIMIT,
    max_pairs=UNLOCKED_LIMIT,
)
BENCHMARK_PRODUCT_POLICY = ProductExecutionPolicy(full_table_max_lanes=UNLOCKED_LIMIT)
BENCHMARK_BIVECTOR_EXP_POLICY = BivectorExpExecutionPolicy(
    spectral_max_planes=BENCHMARK_SPECTRAL_MAX_PLANES,
    spectral_transition_n=BENCHMARK_SPECTRAL_TRANSITION_N,
)


@dataclass(frozen=True)
class SignatureCase:
    """One Clifford signature in the continuous verification matrix."""

    family: str
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
    """Prepared callable plus enough metadata to report its layout contract."""

    name: str
    op: str
    module: Callable[..., torch.Tensor] | nn.Module
    args: tuple[torch.Tensor, ...]
    metadata: dict[str, Any]
    output_layout: GradeLayout | None = None


def parse_csv(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def parse_int_csv(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        return [int(part) for part in parse_csv(value)]
    return [int(part) for part in value]


def parse_grades(value: str | Sequence[int]) -> tuple[int, ...]:
    grades = tuple(parse_int_csv(value))
    if not grades:
        raise ValueError("at least one grade is required")
    return grades


def dimension_range_csv(min_n: int, max_n: int) -> str:
    if min_n < 1:
        raise ValueError(f"minimum verification dimension must be positive, got {min_n}")
    if max_n < min_n:
        raise ValueError(f"maximum verification dimension {max_n} is below minimum {min_n}")
    return ",".join(str(n) for n in range(int(min_n), int(max_n) + 1))


def max_density_command(*, min_n: int = MAX_DENSITY_MIN_N, max_n: int = MAX_DENSITY_MAX_N) -> str:
    return "\n".join(
        [
            "uv run python benchmarks/benchmark_mkdocs.py \\",
            "  --max-density \\",
            f"  --max-density-min-n {int(min_n)} \\",
            f"  --max-density-max-n {int(max_n)} \\",
            "  --device auto \\",
            "  --fail-on-gate",
        ]
    )


def expand_verification_signatures(n_values: Sequence[int], families: Sequence[str]) -> list[SignatureCase]:
    """Expand n values into Euclidean, q=1, and r=1 verification signatures."""
    requested = [family.lower().replace("-", "_") for family in families]
    valid = {"euclidean", "q1", "r1"}
    unknown = sorted(set(requested) - valid)
    if unknown:
        raise ValueError(f"unknown signature families {unknown}; valid families are {sorted(valid)}")

    signatures: list[SignatureCase] = []
    for n in n_values:
        if n < 1:
            raise ValueError(f"signature dimension must be positive, got {n}")
        if "euclidean" in requested:
            signatures.append(SignatureCase("euclidean", n, 0, 0))
        if "q1" in requested:
            if n < 2:
                raise ValueError("q1 signatures require n >= 2")
            signatures.append(SignatureCase("q1", n - 1, 1, 0))
        if "r1" in requested:
            if n < 2:
                raise ValueError("r1 signatures require n >= 2")
            signatures.append(SignatureCase("r1", n - 1, 0, 1))
    return signatures


def sync_device(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def reset_compile_cache() -> bool:
    compiler_reset = getattr(getattr(torch, "compiler", None), "reset", None)
    if callable(compiler_reset):
        compiler_reset()
        return True
    dynamo_reset = getattr(getattr(torch, "_dynamo", None), "reset", None)
    if callable(dynamo_reset):
        dynamo_reset()
        return True
    return False


def randn(shape: tuple[int, ...], *, device: str, dtype: torch.dtype, seed: int, scale: float = 1.0) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    values = torch.randn(*shape, dtype=torch.float64, generator=generator) * float(scale)
    return values.to(device=device, dtype=dtype)


def layout_grades(layout: GradeLayout | None) -> str:
    if layout is None:
        return ""
    return ",".join(str(grade) for grade in layout.grades)


def layout_metadata(prefix: str, layout: GradeLayout) -> dict[str, Any]:
    return {
        f"{prefix}_grades": layout_grades(layout),
        f"{prefix}_lanes": int(layout.dim),
    }


def grade_histogram(layout: GradeLayout) -> dict[str, int]:
    counts: dict[str, int] = {str(grade): 0 for grade in range(layout.spec.n + 1)}
    for index in layout.basis_indices:
        key = str(int(index).bit_count())
        counts[key] = counts.get(key, 0) + 1
    return counts


def timer_ms(fn: Callable[[], Any], *, device: str) -> tuple[Any, float]:
    sync_device(device)
    start = time.perf_counter()
    value = fn()
    sync_device(device)
    return value, (time.perf_counter() - start) * 1000.0


def timing_stats(samples: list[float]) -> dict[str, float]:
    if not samples:
        nan = float("nan")
        return {"median_ms": nan, "mean_ms": nan, "std_ms": nan, "min_ms": nan, "max_ms": nan, "p10_ms": nan, "p90_ms": nan}
    ordered = sorted(samples)
    return {
        "median_ms": float(statistics.median(samples)),
        "mean_ms": float(statistics.fmean(samples)),
        "std_ms": float(statistics.stdev(samples)) if len(samples) > 1 else 0.0,
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
        "p10_ms": percentile(ordered, 0.10),
        "p90_ms": percentile(ordered, 0.90),
    }


def percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return float(ordered[lo])
    frac = position - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def time_callable(
    fn: Callable[..., torch.Tensor] | nn.Module,
    args: tuple[torch.Tensor, ...],
    *,
    device: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    output, first_call_ms = timer_ms(lambda: fn(*args), device=device)
    for _ in range(max(int(warmup), 0)):
        fn(*args)
    sync_device(device)

    samples: list[float] = []
    for _ in range(max(int(iterations), 1)):
        _, elapsed = timer_ms(lambda: fn(*args), device=device)
        samples.append(elapsed)
    return {
        "output": output,
        "first_call_ms": float(first_call_ms),
        "runs": len(samples),
        "samples_ms": samples,
        **timing_stats(samples),
    }


def error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
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
        "max_rel_error": float((abs_diff.max() / scale).item()) if diff.numel() else 0.0,
    }


def sample_steps(max_steps: int, samples: int) -> list[int]:
    if max_steps <= 1:
        return [1]
    values = {1, int(max_steps)}
    for index in range(max(int(samples), 1)):
        t = index / max(int(samples) - 1, 1)
        values.add(max(1, int(round(math.exp(t * math.log(max_steps))))))
    return sorted(values)


def tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def args_bytes(args: Iterable[Any]) -> int:
    return sum(tensor_bytes(arg) for arg in args if isinstance(arg, torch.Tensor))


def output_numel(output: Any) -> int:
    return int(output.numel()) if isinstance(output, torch.Tensor) else 0


def output_bytes(output: Any) -> int:
    return tensor_bytes(output) if isinstance(output, torch.Tensor) else 0


def finite_tensor(output: Any) -> bool:
    return bool(torch.isfinite(output).all().item()) if isinstance(output, torch.Tensor) else True


def grade_leak(values: torch.Tensor, layout: GradeLayout | None, target_grades: Iterable[int] | None) -> float:
    if layout is None or target_grades is None or values.shape[-1] != layout.dim:
        return 0.0
    target = set(int(grade) for grade in target_grades)
    leaked_positions = [pos for pos, index in enumerate(layout.basis_indices) if int(index).bit_count() not in target]
    if not leaked_positions:
        return 0.0
    positions = torch.tensor(leaked_positions, dtype=torch.long, device=values.device)
    leaked = torch.index_select(values.detach().float(), -1, positions)
    return float(leaked.norm().item())


def make_context(signature: SignatureCase, *, device: str, dtype: torch.dtype, default_grades: tuple[int, ...]):
    return make_algebra(
        signature.p,
        signature.q,
        signature.r,
        device=device,
        dtype=dtype,
        default_grades=default_grades,
        planning_limits=BENCHMARK_PLANNING_LIMITS,
        product_execution_policy=BENCHMARK_PRODUCT_POLICY,
        bivector_exp_execution_policy=BENCHMARK_BIVECTOR_EXP_POLICY,
    )


def build_target(
    algebra,
    *,
    operation: str,
    batch: int,
    channels: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
) -> BenchTarget:
    default_layout = algebra.default_layout()
    vector_layout = algebra.layout((1,))
    bivector_layout = algebra.layout((2,))

    if operation == "vector_gp":
        output_layout = algebra.layout(expand_output_grades((1,), (1,), algebra.n, op="gp"))
        module = algebra.plan_product(op="gp", left_layout=vector_layout, right_layout=vector_layout, output_layout=output_layout)
        left = randn((batch, vector_layout.dim), device=device, dtype=dtype, seed=seed)
        right = randn((batch, vector_layout.dim), device=device, dtype=dtype, seed=seed + 1)
        return BenchTarget(
            operation,
            "gp",
            module,
            (left, right),
            {
                "category": "product",
                "executor_family": module.executor_family,
                "pair_count": int(module.pair_count),
                **layout_metadata("left", vector_layout),
                **layout_metadata("right", vector_layout),
                **layout_metadata("output", output_layout),
            },
            output_layout,
        )

    if operation == "bivector_vector_commutator":
        output_layout = algebra.layout(expand_output_grades((2,), (1,), algebra.n, op="commutator_product"))
        module = algebra.plan_product(
            op="commutator_product",
            left_layout=bivector_layout,
            right_layout=vector_layout,
            output_layout=output_layout,
        )
        left = randn((batch, bivector_layout.dim), device=device, dtype=dtype, seed=seed, scale=0.2)
        right = randn((batch, vector_layout.dim), device=device, dtype=dtype, seed=seed + 1)
        return BenchTarget(
            operation,
            "commutator_product",
            module,
            (left, right),
            {
                "category": "product",
                "executor_family": module.executor_family,
                "pair_count": int(module.pair_count),
                **layout_metadata("left", bivector_layout),
                **layout_metadata("right", vector_layout),
                **layout_metadata("output", output_layout),
            },
            output_layout,
        )

    if operation == "bivector_bivector_commutator":
        output_layout = algebra.layout(expand_output_grades((2,), (2,), algebra.n, op="commutator_product"))
        module = algebra.plan_product(
            op="commutator_product",
            left_layout=bivector_layout,
            right_layout=bivector_layout,
            output_layout=output_layout,
        )
        left = randn((batch, bivector_layout.dim), device=device, dtype=dtype, seed=seed, scale=0.2)
        right = randn((batch, bivector_layout.dim), device=device, dtype=dtype, seed=seed + 1, scale=0.2)
        return BenchTarget(
            operation,
            "commutator_product",
            module,
            (left, right),
            {
                "category": "product",
                "executor_family": module.executor_family,
                "pair_count": int(module.pair_count),
                **layout_metadata("left", bivector_layout),
                **layout_metadata("right", bivector_layout),
                **layout_metadata("output", output_layout),
            },
            output_layout,
        )

    if operation == "signature_norm_default":
        module = algebra.plan_signature_norm_squared(input_layout=default_layout, dtype=dtype, device=device)
        values = randn((batch, default_layout.dim), device=device, dtype=dtype, seed=seed)
        return BenchTarget(
            operation,
            "signature_norm_squared",
            module,
            (values,),
            {
                "category": "metric",
                "executor_family": getattr(module, "executor_family", "metric_diagonal"),
                "pair_count": int(default_layout.dim),
                **layout_metadata("input", default_layout),
                "output_grades": "scalar",
                "output_lanes": 1,
            },
            None,
        )

    if operation == "reverse_default":
        module = algebra.plan_unary(
            op="reverse",
            input_layout=default_layout,
            output_layout=default_layout,
            dtype=dtype,
            device=device,
        )
        values = randn((batch, default_layout.dim), device=device, dtype=dtype, seed=seed)
        return BenchTarget(
            operation,
            "reverse",
            module,
            (values,),
            {
                "category": "unary",
                "executor_family": getattr(module, "executor_family", "unary_sign"),
                "pair_count": int(default_layout.dim),
                **layout_metadata("input", default_layout),
                **layout_metadata("output", default_layout),
            },
            default_layout,
        )

    if operation == "grade_involution_default":
        module = algebra.plan_unary(
            op="grade_involution",
            input_layout=default_layout,
            output_layout=default_layout,
            dtype=dtype,
            device=device,
        )
        values = randn((batch, default_layout.dim), device=device, dtype=dtype, seed=seed)
        return BenchTarget(
            operation,
            "grade_involution",
            module,
            (values,),
            {
                "category": "unary",
                "executor_family": getattr(module, "executor_family", "unary_sign"),
                "pair_count": int(default_layout.dim),
                **layout_metadata("input", default_layout),
                **layout_metadata("output", default_layout),
            },
            default_layout,
        )

    if operation == "pseudoscalar_product_vector":
        output_layout = algebra.layout((algebra.n - 1,))
        module = algebra.plan_pseudoscalar_product(input_layout=vector_layout, output_layout=output_layout, dtype=dtype, device=device)
        values = randn((batch, vector_layout.dim), device=device, dtype=dtype, seed=seed)
        return BenchTarget(
            operation,
            "pseudoscalar_product",
            module,
            (values,),
            {
                "category": "permutation",
                "executor_family": getattr(module, "executor_family", "signed_permutation"),
                "pair_count": int(vector_layout.dim),
                **layout_metadata("input", vector_layout),
                **layout_metadata("output", output_layout),
            },
            output_layout,
        )

    if operation == "bivector_exp":
        output_layout = algebra.layout((0, 2))
        module = algebra.plan_bivector_exp(
            input_layout=bivector_layout,
            output_layout=output_layout,
            dtype=dtype,
            device=device,
            spectral_max_planes=BENCHMARK_SPECTRAL_MAX_PLANES,
            spectral_transition_n=BENCHMARK_SPECTRAL_TRANSITION_N,
        )
        values = randn((batch, bivector_layout.dim), device=device, dtype=dtype, seed=seed, scale=0.05)
        return BenchTarget(
            operation,
            "bivector_exp",
            module,
            (values,),
            {
                "category": "exp",
                "executor_family": getattr(module, "executor_family", "unknown"),
                "pair_count": int(getattr(getattr(module, "left_product", None), "pair_count", 0) or 0),
                "spectral_local_axis_count": getattr(module, "spectral_local_axis_count", None),
                "nondegenerate_dim": getattr(module, "nondegenerate_dim", None),
                "ideal_dim": getattr(module, "ideal_dim", None),
                **layout_metadata("input", bivector_layout),
                **layout_metadata("output", output_layout),
            },
            output_layout,
        )

    if operation == "versor_vector":
        module = algebra.plan_versor_action(
            grade=2,
            input_layout=vector_layout,
            output_layout=vector_layout,
            parameter_layout=bivector_layout,
        )
        executor = getattr(module, "executor", module)
        if getattr(executor, "vector_matrix", None) is not None:
            executor_family = "vector_matrix_exp"
        elif bool(getattr(executor, "use_rotor_product_action", False)):
            exp_family = getattr(getattr(executor, "bivector_exp", None), "executor_family", "unknown")
            executor_family = f"rotor_product:{exp_family}"
        else:
            executor_family = getattr(executor, "executor_family", "versor_action")
        values = randn((batch, channels, vector_layout.dim), device=device, dtype=dtype, seed=seed)
        weights = randn((channels, bivector_layout.dim), device=device, dtype=dtype, seed=seed + 1, scale=0.05)
        return BenchTarget(
            operation,
            "versor_action",
            module,
            (values, weights),
            {
                "category": "action",
                "executor_family": executor_family,
                "pair_count": 0,
                "channels": channels,
                "use_full_action": bool(getattr(executor, "use_full_action", False)),
                "use_rotor_product_action": bool(getattr(executor, "use_rotor_product_action", False)),
                **layout_metadata("input", vector_layout),
                **layout_metadata("parameter", bivector_layout),
                **layout_metadata("output", vector_layout),
            },
            vector_layout,
        )

    raise ValueError(f"unknown benchmark operation {operation!r}")


def compile_callable(module: Callable[..., torch.Tensor] | nn.Module, mode: str):
    if mode == "eager":
        return module
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available")
    if mode == "aot_eager":
        return torch.compile(module, backend="aot_eager", fullgraph=True)
    if mode == "inductor":
        return torch.compile(module, fullgraph=True)
    raise ValueError(f"unknown compile mode {mode!r}")


def benchmark_target(
    target: BenchTarget,
    *,
    mode: str,
    device: str,
    warmup: int,
    iterations: int,
    correctness_atol: float,
    correctness_rtol: float,
) -> dict[str, Any]:
    eager = time_callable(target.module, target.args, device=device, warmup=0, iterations=1)
    expected = eager["output"]

    try:
        reset_before = False
        reset_after = False
        if mode == "eager":
            callable_target = target.module
        else:
            reset_before = reset_compile_cache()
            callable_target = compile_callable(target.module, mode)
        timing = time_callable(callable_target, target.args, device=device, warmup=warmup, iterations=iterations)
        if mode != "eager":
            reset_after = reset_compile_cache()
        diff = error_stats(timing["output"], expected)
        compile_ok = True
        fullgraph_ok = True
        error = ""
    except Exception as exc:
        timing = eager
        diff = {"max_abs_error": float("inf"), "rms_error": float("inf"), "max_rel_error": float("inf")}
        compile_ok = False
        fullgraph_ok = False
        reset_before = False
        reset_after = False
        error = repr(exc)

    output = timing["output"]
    finite = finite_tensor(output)
    compile_abs = float(diff["max_abs_error"])
    compile_rel = float(diff["max_rel_error"])
    gate_ok = bool(compile_ok and finite and compile_abs <= correctness_atol and compile_rel <= correctness_rtol)
    return {
        "compile_mode": mode,
        "compile_ok": compile_ok,
        "fullgraph_ok": fullgraph_ok if mode != "eager" else True,
        "compile_error": error,
        "compile_cache_reset_before": reset_before,
        "compile_cache_reset_after": reset_after,
        "output_finite": finite,
        "compile_max_abs_diff": compile_abs,
        "compile_rms_diff": float(diff["rms_error"]),
        "compile_max_rel_diff": compile_rel,
        "grade_leak": grade_leak(output, target.output_layout, None if target.output_layout is None else target.output_layout.grades),
        "gate_ok": gate_ok,
        "arg_bytes": args_bytes(target.args),
        "output_numel": output_numel(output),
        "output_bytes": output_bytes(output),
        "first_call_ms": float(timing["first_call_ms"]),
        "runs": int(timing["runs"]),
        "samples_ms": timing["samples_ms"],
        **{key: value for key, value in timing.items() if key.endswith("_ms") and key != "samples_ms"},
    }


def collect_layout_invariants(
    signatures: Sequence[SignatureCase],
    *,
    default_grades: tuple[int, ...],
    device: str,
    dtype: torch.dtype,
    max_full_lanes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for signature in signatures:
        algebra = make_context(signature, device=device, dtype=dtype, default_grades=default_grades)
        layout = algebra.default_layout()
        full_lanes = int(algebra.dim)
        full_allowed = max_full_lanes <= 0 or full_lanes <= max_full_lanes
        roundtrip_ok = True
        roundtrip_error = 0.0
        if full_allowed:
            values = randn((2, layout.dim), device=device, dtype=dtype, seed=9100 + signature.n)
            compact = layout.compact(layout.full(values))
            diff = error_stats(compact, values)
            roundtrip_error = float(diff["max_abs_error"])
            roundtrip_ok = roundtrip_error == 0.0
        rows.append(
            {
                "signature": signature.label,
                "family": signature.family,
                "n": signature.n,
                "p": signature.p,
                "q": signature.q,
                "r": signature.r,
                "default_grades": layout_grades(layout),
                "full_lanes": full_lanes,
                "active_lanes": int(layout.dim),
                "grade_histogram": json.dumps(grade_histogram(layout), sort_keys=True),
                "basis_first": ",".join(str(index) for index in layout.basis_indices[:8]),
                "basis_last": ",".join(str(index) for index in layout.basis_indices[-8:]),
                "full_materialization_allowed": full_allowed,
                "full_materialization_avoided": not full_allowed,
                "roundtrip_checked": full_allowed,
                "roundtrip_ok": roundtrip_ok,
                "roundtrip_max_abs_error": roundtrip_error,
                "gate_ok": bool(roundtrip_ok),
            }
        )
    return rows


def run_benchmarks(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    signatures = expand_verification_signatures(parse_int_csv(args.verification_n), parse_csv(args.signature_families))
    default_grades = parse_grades(args.default_grades)
    dtypes = [resolve_dtype(name) for name in parse_csv(args.dtypes)]
    operations = parse_csv(args.operations)
    compile_modes = parse_csv(args.compile_modes)

    layout_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []

    for dtype in dtypes:
        layout_rows.extend(
            collect_layout_invariants(
                signatures,
                default_grades=default_grades,
                device=args.device,
                dtype=dtype,
                max_full_lanes=args.max_full_lanes,
            )
        )

    for cycle in range(1, int(args.soak_cycles) + 1):
        for signature_index, signature in enumerate(signatures):
            for dtype in dtypes:
                algebra = make_context(signature, device=args.device, dtype=dtype, default_grades=default_grades)
                for operation_index, operation in enumerate(operations):
                    seed = int(args.seed) + cycle * 100_003 + signature_index * 10_007 + operation_index * 97
                    base = {
                        "cycle": cycle,
                        "signature": signature.label,
                        "family": signature.family,
                        "n": signature.n,
                        "p": signature.p,
                        "q": signature.q,
                        "r": signature.r,
                        "device": args.device,
                        "dtype": dtype_name(dtype),
                        "default_grades": ",".join(str(grade) for grade in default_grades),
                        "batch": int(args.batch_size),
                        "channels": int(args.channels),
                        "operation": operation,
                    }
                    skip_details = operation_skip_details(signature, operation, dtype, args)
                    if skip_details:
                        row = dict(base)
                        row.update(
                            {
                                "status": "skipped",
                                "compile_mode": "not_run",
                                "compile_ok": True,
                                "fullgraph_ok": True,
                                "output_finite": True,
                                "gate_ok": True,
                            }
                        )
                        row.update(skip_details)
                        performance_rows.append(row)
                        print(f"SKIP {signature.label} {dtype_name(dtype)} {operation}: {skip_details['skip_reason']}")
                        continue
                    try:
                        target, plan_ms = timer_ms(
                            lambda: build_target(
                                algebra,
                                operation=operation,
                                batch=int(args.batch_size),
                                channels=int(args.channels),
                                device=args.device,
                                dtype=dtype,
                                seed=seed,
                            ),
                            device=args.device,
                        )
                        for mode in compile_modes:
                            row = dict(base)
                            row.update(target.metadata)
                            row.update(
                                {
                                    "plan_ms": float(plan_ms),
                                    "output_layout_grades": layout_grades(target.output_layout),
                                    "output_layout_lanes": int(target.output_layout.dim) if target.output_layout else 1,
                                }
                            )
                            row.update(
                                benchmark_target(
                                    target,
                                    mode=mode,
                                    device=args.device,
                                    warmup=int(args.warmup),
                                    iterations=int(args.iterations),
                                    correctness_atol=float(args.correctness_atol),
                                    correctness_rtol=float(args.correctness_rtol),
                                )
                            )
                            row["status"] = "ok"
                            performance_rows.append(row)
                            print_row(row)
                    except Exception as exc:
                        row = dict(base)
                        row.update(
                            {
                                "compile_mode": "setup",
                                "compile_ok": False,
                                "fullgraph_ok": False,
                                "output_finite": False,
                                "gate_ok": False,
                                "compile_error": repr(exc),
                            }
                        )
                        performance_rows.append(row)
                        print(f"FAIL {signature.label} {dtype_name(dtype)} {operation}: {exc}")
    if not args.skip_drift_suite:
        for dtype in dtypes:
            drift_rows.extend(
                collect_rotor_chain_drift(
                    signatures,
                    default_grades=default_grades,
                    device=args.device,
                    dtype=dtype,
                    batch=int(args.drift_batch_size),
                    steps=int(args.drift_steps),
                    samples=int(args.drift_samples),
                    angle=float(args.drift_angle),
                    atol=float(args.drift_atol),
                    rtol=float(args.drift_rtol),
                )
            )
    return layout_rows, performance_rows, drift_rows


def operation_skip_details(
    signature: SignatureCase,
    operation: str,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if operation == "bivector_bivector_commutator" and signature.n < 3:
        return {
            "status": "skipped",
            "category": "product",
            "skip_reason": "structural_empty_output:bivector_bivector_commutator_requires_n>=3",
        }
    if operation in EXP_RELATED_OPERATIONS and signature.n > int(args.exp_matrix_max_n):
        family, reason = exp_executor_selection(signature, dtype=dtype, device=args.device)
        if family in MATRIX_EXP_FAMILIES:
            return {
                "status": "skipped",
                "category": "exp",
                "executor_family": family,
                "exp_preselection_reason": reason,
                "skip_reason": (
                    f"{family}:{reason}:n={signature.n}_exceeds_exp_matrix_max_n={int(args.exp_matrix_max_n)}"
                ),
            }
    return {}


def exp_executor_selection(signature: SignatureCase, *, dtype: torch.dtype, device: str) -> tuple[str, str]:
    spec = AlgebraSpec(signature.p, signature.q, signature.r)
    kwargs = {
        "dtype": dtype,
        "spectral_max_planes": BENCHMARK_SPECTRAL_MAX_PLANES,
        "spectral_transition_n": BENCHMARK_SPECTRAL_TRANSITION_N,
    }
    family = select_bivector_exp_executor_family(spec, device, **kwargs)
    preselection = spectral_exp_preselection(
        spec,
        device,
        dtype=dtype,
        max_planes=BENCHMARK_SPECTRAL_MAX_PLANES,
        transition_n=BENCHMARK_SPECTRAL_TRANSITION_N,
    )
    return family, preselection.reason


def collect_rotor_chain_drift(
    signatures: Sequence[SignatureCase],
    *,
    default_grades: tuple[int, ...],
    device: str,
    dtype: torch.dtype,
    batch: int,
    steps: int,
    samples: int,
    angle: float,
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for signature in signatures:
        if signature.n < 2:
            continue
        if signature.p < 2:
            rows.append(
                {
                    "suite": "accumulated_drift",
                    "target": "positive_euclidean_plane_rotor_chain",
                    "status": "skipped",
                    "signature": signature.label,
                    "family": signature.family,
                    "n": signature.n,
                    "p": signature.p,
                    "q": signature.q,
                    "r": signature.r,
                    "device": device,
                    "dtype": dtype_name(dtype),
                    "default_grades": ",".join(str(grade) for grade in default_grades),
                    "batch": int(batch),
                    "chain_steps": int(steps),
                    "angle_per_step": float(angle),
                    "step": "",
                    "elapsed_ms": 0.0,
                    "drift_max_abs_error": 0.0,
                    "drift_rms_error": 0.0,
                    "drift_max_rel_error": 0.0,
                    "norm_drift_max_abs_error": 0.0,
                    "norm_drift_max_rel_error": 0.0,
                    "output_finite": True,
                    "gate_ok": True,
                    "error": "skipped:requires_two_positive_axes_for_euclidean_plane_rotor",
                }
            )
            continue
        try:
            rows.extend(
                rotor_chain_rows_for_signature(
                    signature,
                    default_grades=default_grades,
                    device=device,
                    dtype=dtype,
                    batch=batch,
                    steps=steps,
                    samples=samples,
                    angle=angle,
                    atol=atol,
                    rtol=rtol,
                )
            )
        except Exception as exc:  # noqa: BLE001 - drift failures are report data.
            rows.append(
                {
                    "suite": "accumulated_drift",
                    "target": "rotor_chain",
                    "status": "failed",
                    "signature": signature.label,
                    "family": signature.family,
                    "n": signature.n,
                    "p": signature.p,
                    "q": signature.q,
                    "r": signature.r,
                    "device": device,
                    "dtype": dtype_name(dtype),
                    "gate_ok": False,
                    "error": repr(exc),
                }
            )
    return rows


def rotor_chain_rows_for_signature(
    signature: SignatureCase,
    *,
    default_grades: tuple[int, ...],
    device: str,
    dtype: torch.dtype,
    batch: int,
    steps: int,
    samples: int,
    angle: float,
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    algebra = make_context(signature, device=device, dtype=dtype, default_grades=default_grades)
    ref_algebra = make_context(signature, device="cpu", dtype=torch.float64, default_grades=default_grades)

    vector_layout = algebra.layout((1,))
    rotor_layout = algebra.layout((0, 2))
    middle_layout = algebra.layout(expand_output_grades(rotor_layout.grades, vector_layout.grades, algebra.n, op="gp"))
    left_product = algebra.plan_product(op="gp", left_layout=rotor_layout, right_layout=vector_layout, output_layout=middle_layout)
    right_product = algebra.plan_product(op="gp", left_layout=middle_layout, right_layout=rotor_layout, output_layout=vector_layout)
    reverse = algebra.plan_unary(op="reverse", input_layout=rotor_layout, output_layout=rotor_layout)

    ref_vector_layout = ref_algebra.layout((1,))
    ref_rotor_layout = ref_algebra.layout((0, 2))
    ref_middle_layout = ref_algebra.layout(
        expand_output_grades(ref_rotor_layout.grades, ref_vector_layout.grades, ref_algebra.n, op="gp")
    )
    ref_left_product = ref_algebra.plan_product(
        op="gp",
        left_layout=ref_rotor_layout,
        right_layout=ref_vector_layout,
        output_layout=ref_middle_layout,
    )
    ref_right_product = ref_algebra.plan_product(
        op="gp",
        left_layout=ref_middle_layout,
        right_layout=ref_rotor_layout,
        output_layout=ref_vector_layout,
    )
    ref_reverse = ref_algebra.plan_unary(op="reverse", input_layout=ref_rotor_layout, output_layout=ref_rotor_layout)

    x = initial_chain_vector(vector_layout, batch=batch, device=device, dtype=dtype)
    ref_x0 = initial_chain_vector(ref_vector_layout, batch=batch, device="cpu", dtype=torch.float64)
    step_rotor = plane_rotor(rotor_layout, angle=angle, batch=batch, device=device, dtype=dtype)
    step_reverse = reverse(step_rotor)
    sample_set = set(sample_steps(int(steps), int(samples)))

    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for step in range(1, int(steps) + 1):
        x = right_product(left_product(step_rotor, x), step_reverse)
        if step not in sample_set:
            continue
        ref_rotor = plane_rotor(ref_rotor_layout, angle=float(angle) * step, batch=batch, device="cpu", dtype=torch.float64)
        ref_expected = ref_right_product(ref_left_product(ref_rotor, ref_x0), ref_reverse(ref_rotor))
        drift = error_stats(x, ref_expected)
        norm_current = algebra.signature_norm_squared(x, input_layout=vector_layout).detach().cpu().to(torch.float64)
        norm_expected = ref_algebra.signature_norm_squared(ref_expected, input_layout=ref_vector_layout).detach().cpu().to(torch.float64)
        norm_drift = error_stats(norm_current, norm_expected)
        gate_ok = float(drift["max_abs_error"]) <= float(atol) and float(drift["max_rel_error"]) <= float(rtol)
        rows.append(
            {
                "suite": "accumulated_drift",
                "target": "rotor_chain",
                "status": "ok",
                "signature": signature.label,
                "family": signature.family,
                "n": signature.n,
                "p": signature.p,
                "q": signature.q,
                "r": signature.r,
                "device": device,
                "dtype": dtype_name(dtype),
                "default_grades": ",".join(str(grade) for grade in default_grades),
                "step": step,
                "chain_steps": int(steps),
                "angle_per_step": float(angle),
                "batch": int(batch),
                "elapsed_ms": (time.perf_counter() - start) * 1000.0,
                "drift_max_abs_error": float(drift["max_abs_error"]),
                "drift_rms_error": float(drift["rms_error"]),
                "drift_max_rel_error": float(drift["max_rel_error"]),
                "norm_drift_max_abs_error": float(norm_drift["max_abs_error"]),
                "norm_drift_max_rel_error": float(norm_drift["max_rel_error"]),
                "output_finite": bool(torch.isfinite(x).all().item()),
                "gate_ok": bool(gate_ok and torch.isfinite(x).all().item()),
            }
        )
    if rows:
        last = rows[-1]
        print(
            f"DRIFT {signature.label:<12s} {dtype_name(dtype):<8s} "
            f"steps={int(steps):<5d} drift={float(last['drift_max_abs_error']):.2e} gate={last['gate_ok']}"
        )
    return rows


def initial_chain_vector(layout: GradeLayout, *, batch: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    values = torch.zeros(int(batch), layout.dim, device=device, dtype=dtype)
    position = layout.basis_indices.index(1)
    values[:, position] = 1.0
    return values


def plane_rotor(layout: GradeLayout, *, angle: float, batch: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    values = torch.zeros(int(batch), layout.dim, device=device, dtype=dtype)
    scalar_position = layout.basis_indices.index(0)
    bivector_index = (1 << 0) | (1 << 1)
    bivector_position = layout.basis_indices.index(bivector_index)
    theta = torch.tensor(float(angle) * 0.5, dtype=dtype, device=device)
    values[:, scalar_position] = torch.cos(theta)
    values[:, bivector_position] = -torch.sin(theta)
    return values


def print_row(row: dict[str, Any]) -> None:
    print(
        f"{row['signature']:<12s} {row['dtype']:<8s} {row['operation']:<30s} "
        f"{row['compile_mode']:<9s} {str(row.get('executor_family', '')):<16s} "
        f"med={float(row.get('median_ms', float('nan'))):8.3f}ms "
        f"compile_diff={float(row.get('compile_max_abs_diff', float('nan'))):.2e} gate={row.get('gate_ok')}"
    )


def summarize_gates(
    layout_rows: Sequence[dict[str, Any]],
    performance_rows: Sequence[dict[str, Any]],
    drift_rows: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    all_rows = [*layout_rows, *performance_rows, *drift_rows]
    total = len(all_rows)
    failed = [row for row in all_rows if not bool(row.get("gate_ok", False))]
    measured_rows = [row for row in performance_rows if row.get("status") != "skipped"]
    skipped_rows = [row for row in performance_rows if row.get("status") == "skipped"]
    compile_failed = [row for row in measured_rows if not bool(row.get("compile_ok", False))]
    fullgraph_failed = [row for row in measured_rows if not bool(row.get("fullgraph_ok", True))]
    drift_failed = [row for row in drift_rows if not bool(row.get("gate_ok", True))]
    return {
        "total_gate_rows": total,
        "gate_pass_count": total - len(failed),
        "gate_failure_count": len(failed),
        "skip_count": len(skipped_rows),
        "compile_failure_count": len(compile_failed),
        "fullgraph_failure_count": len(fullgraph_failed),
        "drift_failure_count": len(drift_failed),
        "all_gates_ok": len(failed) == 0,
    }


def device_metadata(args: argparse.Namespace) -> dict[str, Any]:
    resolved = torch.device(args.device)
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    mps_is_built = getattr(mps_backend, "is_built", None)
    metadata: dict[str, Any] = {
        "requested": getattr(args, "requested_device", args.device),
        "resolved": str(resolved),
        "type": resolved.type,
        "index": resolved.index,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_platform": platform.platform(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": mps_available,
        "mps_built": bool(mps_is_built()) if callable(mps_is_built) else False,
    }
    if torch.cuda.is_available():
        cuda_devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "major": int(props.major),
                    "minor": int(props.minor),
                    "multi_processor_count": int(props.multi_processor_count),
                }
            )
        metadata["cuda_devices"] = cuda_devices
        metadata["selected_cuda_device"] = None
        if resolved.type == "cuda":
            selected_index = resolved.index if resolved.index is not None else torch.cuda.current_device()
            metadata["selected_cuda_device"] = cuda_devices[selected_index] if selected_index < len(cuda_devices) else None
        cudnn_version = torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None
        metadata["cudnn_version"] = cudnn_version
    if resolved.type == "mps" and hasattr(torch, "mps"):
        current_allocated = getattr(torch.mps, "current_allocated_memory", None)
        driver_allocated = getattr(torch.mps, "driver_allocated_memory", None)
        metadata["mps_current_allocated_bytes"] = int(current_allocated()) if callable(current_allocated) else None
        metadata["mps_driver_allocated_bytes"] = int(driver_allocated()) if callable(driver_allocated) else None
    return metadata


def format_device_summary(info: dict[str, Any]) -> str:
    resolved = str(info.get("resolved", "unknown"))
    if info.get("type") == "cuda" and info.get("selected_cuda_device"):
        selected = info["selected_cuda_device"]
        memory_gib = float(selected.get("total_memory_bytes", 0)) / float(1 << 30)
        return f"{resolved} / {selected.get('name')} ({memory_gib:.1f} GiB)"
    if info.get("type") == "mps":
        return f"{resolved} / Apple MPS (available={info.get('mps_available')}, built={info.get('mps_built')})"
    threads = f"{info.get('torch_num_threads')}/{info.get('torch_num_interop_threads')}"
    machine = str(info.get("machine") or "unknown")
    return f"{resolved} / {machine} (torch threads={threads})"


def json_default(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return dtype_name(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_cell(row.get(key, "")) for key in columns})


def serialize_cell(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, default=json_default, sort_keys=True)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_artifacts(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    layout_rows: Sequence[dict[str, Any]],
    performance_rows: Sequence[dict[str, Any]],
    drift_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_gates(layout_rows, performance_rows, drift_rows)
    benchmark_device = device_metadata(args)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "argv": sys.argv,
        "config": vars(args),
        "benchmark_device": benchmark_device,
        "device_summary": format_device_summary(benchmark_device),
        "max_density_command": max_density_command(
            min_n=int(args.max_density_min_n),
            max_n=int(args.max_density_max_n),
        ),
        "max_density_verification_n": dimension_range_csv(
            int(args.max_density_min_n),
            int(args.max_density_max_n),
        ),
        "planning_limits_unlocked": True,
        "planning_max_lanes": UNLOCKED_LIMIT,
        "planning_max_pairs": UNLOCKED_LIMIT,
        "full_table_max_lanes": UNLOCKED_LIMIT,
        **summary,
        "layout_row_count": len(layout_rows),
        "performance_row_count": len(performance_rows),
        "drift_row_count": len(drift_rows),
    }

    write_csv(artifacts_dir / "layout_invariants.csv", layout_rows)
    write_csv(artifacts_dir / "performance.csv", performance_rows)
    write_csv(artifacts_dir / "accumulated_drift.csv", drift_rows)
    (artifacts_dir / "layout_invariants.json").write_text(json.dumps(layout_rows, indent=2, default=json_default))
    (artifacts_dir / "performance.json").write_text(json.dumps(performance_rows, indent=2, default=json_default))
    (artifacts_dir / "accumulated_drift.json").write_text(json.dumps(drift_rows, indent=2, default=json_default))
    (artifacts_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=json_default))

    (output_dir / "index.md").write_text(render_index(metadata, layout_rows, performance_rows))
    (output_dir / "layouts.md").write_text(render_layouts(layout_rows))
    (output_dir / "performance.md").write_text(render_performance(performance_rows))
    (output_dir / "correctness.md").write_text(render_correctness(performance_rows, drift_rows))
    return metadata


def markdown_escape(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if abs(value) >= 1000 or (0 < abs(value) < 0.001):
            text = f"{value:.3e}"
        else:
            text = f"{value:.6g}"
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[tuple[str, str]], *, limit: int | None = None) -> str:
    selected = list(rows[:limit] if limit is not None else rows)
    lines = [
        "| " + " | ".join(header for header, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected:
        lines.append("| " + " | ".join(markdown_escape(row.get(key, "")) for _, key in columns) + " |")
    if limit is not None and len(rows) > limit:
        lines.append("| " + " | ".join([f"{len(rows) - limit} more rows in artifacts"] + ["" for _ in columns[1:]]) + " |")
    return "\n".join(lines)


def render_index(
    metadata: dict[str, Any],
    layout_rows: Sequence[dict[str, Any]],
    performance_rows: Sequence[dict[str, Any]],
) -> str:
    fastest = fastest_rows(performance_rows)
    lines = [
        "# Benchmarks",
        "",
        "This section is generated by `uv run python benchmarks/benchmark_mkdocs.py`.",
        "",
        "## Latest Run",
        "",
        f"- Created: `{metadata['created_at']}`",
        f"- Torch: `{metadata['torch_version']}`",
        f"- Python: `{metadata['python']}`",
        f"- Device: `{metadata.get('device_summary', '')}`",
        f"- Planner limits unlocked: `{metadata['planning_limits_unlocked']}`",
        f"- Planner max lanes/pairs: `{metadata['planning_max_lanes']}` / `{metadata['planning_max_pairs']}`",
        f"- Gate rows: `{metadata['gate_pass_count']}/{metadata['total_gate_rows']}`",
        f"- Gate failures: `{metadata['gate_failure_count']}`",
        f"- Operation skips: `{metadata['skip_count']}`",
        f"- Fullgraph failures: `{metadata['fullgraph_failure_count']}`",
        f"- Drift failures: `{metadata['drift_failure_count']}`",
        "",
        "## Maximum-Density Verification",
        "",
        "```bash",
        metadata["max_density_command"],
        "```",
        "",
        f"- Dimension sweep: `{metadata['max_density_verification_n']}`",
        f"- Signature families: `euclidean,q1,r1`",
        f"- Dtypes: `float32,float64`",
        f"- Operations: `{','.join(MAX_DENSITY_OPERATIONS)}`",
        "",
        "## Fastest Observed Rows",
        "",
        markdown_table(
            fastest,
            [
                ("signature", "signature"),
                ("dtype", "dtype"),
                ("operation", "operation"),
                ("mode", "compile_mode"),
                ("family", "executor_family"),
                ("median ms", "median_ms"),
                ("compile diff max abs", "compile_max_abs_diff"),
                ("gate", "gate_ok"),
            ],
            limit=40,
        ),
        "",
        "## Artifacts",
        "",
        "- [Metadata](artifacts/metadata.json)",
        "- [Layout invariants CSV](artifacts/layout_invariants.csv)",
        "- [Performance CSV](artifacts/performance.csv)",
        "- [Accumulated drift CSV](artifacts/accumulated_drift.csv)",
        "",
    ]
    return "\n".join(lines)


def render_layouts(layout_rows: Sequence[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# Layout Invariants",
            "",
            "## Continuous Verification Layouts",
            "",
            markdown_table(
                layout_rows,
                [
                    ("signature", "signature"),
                    ("family", "family"),
                    ("n", "n"),
                    ("default grades", "default_grades"),
                    ("full lanes", "full_lanes"),
                    ("active lanes", "active_lanes"),
                    ("roundtrip", "roundtrip_ok"),
                    ("full avoided", "full_materialization_avoided"),
                    ("gate", "gate_ok"),
                ],
            ),
            "",
        ]
    )


def render_performance(performance_rows: Sequence[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# Performance",
            "",
            markdown_table(
                performance_rows,
                [
                    ("cycle", "cycle"),
                    ("signature", "signature"),
                    ("dtype", "dtype"),
                    ("operation", "operation"),
                    ("status", "status"),
                    ("mode", "compile_mode"),
                    ("family", "executor_family"),
                    ("plan ms", "plan_ms"),
                    ("first ms", "first_call_ms"),
                    ("median ms", "median_ms"),
                    ("p90 ms", "p90_ms"),
                    ("arg bytes", "arg_bytes"),
                    ("output bytes", "output_bytes"),
                    ("gate", "gate_ok"),
                    ("skip", "skip_reason"),
                ],
                limit=160,
            ),
            "",
        ]
    )


def render_correctness(
    performance_rows: Sequence[dict[str, Any]],
    drift_rows: Sequence[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "# Correctness",
            "",
            "## Compile/Eager Fullgraph Status",
            "",
            markdown_table(
                performance_rows,
                [
                    ("signature", "signature"),
                    ("operation", "operation"),
                    ("status", "status"),
                    ("mode", "compile_mode"),
                    ("compile", "compile_ok"),
                    ("fullgraph", "fullgraph_ok"),
                    ("finite", "output_finite"),
                    ("compile abs", "compile_max_abs_diff"),
                    ("compile rel", "compile_max_rel_diff"),
                    ("gate", "gate_ok"),
                    ("skip", "skip_reason"),
                    ("error", "compile_error"),
                ],
                limit=160,
            ),
            "",
            "## Accumulated Rotor-Chain Drift",
            "",
            markdown_table(
                drift_rows,
                [
                    ("signature", "signature"),
                    ("dtype", "dtype"),
                    ("step", "step"),
                    ("steps", "chain_steps"),
                    ("angle", "angle_per_step"),
                    ("elapsed ms", "elapsed_ms"),
                    ("drift abs", "drift_max_abs_error"),
                    ("drift rel", "drift_max_rel_error"),
                    ("norm drift abs", "norm_drift_max_abs_error"),
                    ("finite", "output_finite"),
                    ("gate", "gate_ok"),
                    ("error", "error"),
                ],
                limit=160,
            ),
            "",
        ]
    )


def fastest_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") == "skipped":
            continue
        if not bool(row.get("compile_ok", False)):
            continue
        key = (str(row.get("signature")), str(row.get("dtype")), str(row.get("operation")))
        current = best.get(key)
        if current is None or float(row.get("median_ms", float("inf"))) < float(current.get("median_ms", float("inf"))):
            best[key] = row
    return sorted(best.values(), key=lambda row: (str(row.get("signature")), str(row.get("dtype")), str(row.get("operation"))))


def apply_quick_defaults(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.verification_n = "4,6"
    args.batch_size = min(int(args.batch_size), 4)
    args.channels = min(int(args.channels), 2)
    args.warmup = 0
    args.iterations = 1
    args.operations = "vector_gp,bivector_vector_commutator,bivector_exp,signature_norm_default"
    args.drift_steps = min(int(args.drift_steps), 8)
    args.drift_samples = min(int(args.drift_samples), 4)


def apply_max_density_defaults(args: argparse.Namespace) -> None:
    if not args.max_density:
        return
    min_n = int(args.max_density_min_n)
    max_n = int(args.max_density_max_n)
    if min_n < 2:
        raise ValueError("max-density verification requires --max-density-min-n >= 2 for q=1 and r=1 families")
    if max_n > 63:
        raise ValueError("max-density verification is capped at n=63 because basis indices must fit in int64")
    args.default_grades = "1,2"
    args.verification_n = dimension_range_csv(min_n, max_n)
    args.signature_families = "euclidean,q1,r1"
    args.dtypes = "float32"
    args.operations = ",".join(MAX_DENSITY_OPERATIONS)
    args.batch_size = 32
    args.channels = 8
    args.warmup = 5
    args.iterations = 30
    args.soak_cycles = 1
    args.drift_steps = 512
    args.drift_samples = 16
    args.drift_batch_size = 32


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-grades", default="1,2")
    parser.add_argument("--verification-n", default="8,16")
    parser.add_argument("--signature-families", default="euclidean,q1,r1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtypes", default="float32")
    parser.add_argument("--operations", default=",".join(DEFAULT_OPERATIONS))
    parser.add_argument("--compile-modes", default="eager,aot_eager")
    parser.add_argument("--include-inductor", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--soak-cycles", type=int, default=1)
    parser.add_argument("--max-full-lanes", type=int, default=4096)
    parser.add_argument(
        "--exp-matrix-max-n",
        type=int,
        default=8,
        help="execute pseudo-Euclidean matrix-exp rows only up to this n; spectral-local exp rows are unaffected",
    )
    parser.add_argument("--correctness-atol", type=float, default=1e-4)
    parser.add_argument("--correctness-rtol", type=float, default=1e-4)
    parser.add_argument("--skip-drift-suite", action="store_true")
    parser.add_argument("--drift-steps", type=int, default=64)
    parser.add_argument("--drift-samples", type=int, default=8)
    parser.add_argument("--drift-batch-size", type=int, default=8)
    parser.add_argument("--drift-angle", type=float, default=0.015)
    parser.add_argument("--drift-atol", type=float, default=1e-3)
    parser.add_argument("--drift-rtol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="docs/benchmarks")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--max-density",
        action="store_true",
        help="use the high-density verification profile spanning every n in the configured low-to-high range",
    )
    parser.add_argument("--max-density-min-n", type=int, default=MAX_DENSITY_MIN_N)
    parser.add_argument("--max-density-max-n", type=int, default=MAX_DENSITY_MAX_N)
    args = parser.parse_args(argv)
    if args.quick and args.max_density:
        parser.error("--quick and --max-density cannot be combined")
    args.requested_device = args.device
    args.device = resolve_device(args.device)
    try:
        apply_max_density_defaults(args)
    except ValueError as exc:
        parser.error(str(exc))
    apply_quick_defaults(args)
    if args.include_inductor:
        modes = parse_csv(args.compile_modes)
        if "inductor" not in modes:
            modes.append("inductor")
        args.compile_modes = ",".join(modes)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    layout_rows, performance_rows, drift_rows = run_benchmarks(args)
    metadata = summarize_gates(layout_rows, performance_rows, drift_rows)
    if not args.no_save:
        metadata = write_artifacts(
            output_dir=Path(args.out),
            args=args,
            layout_rows=layout_rows,
            performance_rows=performance_rows,
            drift_rows=drift_rows,
        )
        print(f"Wrote MkDocs benchmark pages to {args.out}")
    if args.fail_on_gate and not bool(metadata.get("all_gates_ok", False)):
        raise SystemExit(f"benchmark gates failed: {metadata['gate_failure_count']} failures")


if __name__ == "__main__":
    main()
