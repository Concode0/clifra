#!/usr/bin/env python3
"""Benchmark framework-level dense, compact, pairwise, and layer pipeline paths.

The core benchmark suite measures many algebra kernels in detail. This script is
smaller and pipeline-oriented: it times the execution paths users hit when they
compose product layers, functional products, compact layouts, and planned
contexts.

Examples:
    uv run python benchmarks/benchmark_framework_pipeline.py --quick
    uv run python benchmarks/benchmark_framework_pipeline.py --device cpu --n 8 --batch-size 1024
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from clifra.core.foundation.device import FLOAT_DTYPES, resolve_device  # noqa: E402
from clifra.core.runtime.algebra import CliffordAlgebra  # noqa: E402
from clifra.core.runtime.context import AlgebraContext  # noqa: E402
from clifra.layers import ProductLayer, WedgeLayer  # noqa: E402


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    host: str
    n: int
    op: str
    storage: str
    batch_size: int
    left_lanes: int
    right_lanes: int
    output_lanes: int
    pairwise: bool
    fn: Callable[[], torch.Tensor]
    reference_fn: Optional[Callable[[], torch.Tensor]] = None


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def _time_case(case: BenchmarkCase, *, warmups: int, repeats: int, device: str) -> dict:
    with torch.no_grad():
        start = time.perf_counter()
        first_output = case.fn()
        _sync(device)
        first_ms = (time.perf_counter() - start) * 1000.0

        for _ in range(warmups):
            case.fn()
        _sync(device)

        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            output = case.fn()
            _sync(device)
            samples.append((time.perf_counter() - start) * 1000.0)

    output_lanes = int(first_output.shape[-1])
    if output_lanes != case.output_lanes:
        raise RuntimeError(f"{case.name} expected {case.output_lanes} output lanes, got {output_lanes}")
    parity = _parity_metrics(first_output, case.reference_fn, device=device)

    median_ms = statistics.median(samples)
    mean_ms = statistics.fmean(samples)
    return {
        "name": case.name,
        "host": case.host,
        "n": case.n,
        "op": case.op,
        "storage": case.storage,
        "batch_size": case.batch_size,
        "left_lanes": case.left_lanes,
        "right_lanes": case.right_lanes,
        "output_lanes": case.output_lanes,
        "pairwise": case.pairwise,
        "first_call_ms": first_ms,
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "repeats": repeats,
        "items_per_sec": case.batch_size * 1000.0 / median_ms if median_ms > 0 else float("inf"),
        **parity,
    }


def _parity_metrics(
    output: torch.Tensor,
    reference_fn: Optional[Callable[[], torch.Tensor]],
    *,
    device: str,
) -> dict:
    output_finite = bool(torch.isfinite(output).all().item())
    if reference_fn is None:
        return {
            "output_finite": output_finite,
            "parity_checked": False,
            "reference_finite": False,
            "max_abs_error": float("nan"),
            "max_rel_error": float("nan"),
            "rms_error": float("nan"),
        }

    with torch.no_grad():
        reference = reference_fn()
        _sync(device)
    if reference.shape != output.shape:
        raise RuntimeError(f"reference shape {tuple(reference.shape)} does not match output shape {tuple(output.shape)}")

    output64 = output.detach().cpu().to(torch.float64)
    reference64 = reference.detach().cpu().to(torch.float64)
    diff = output64 - reference64
    abs_diff = diff.abs()
    max_abs = float(abs_diff.max().item()) if abs_diff.numel() else 0.0
    reference_scale = float(reference64.abs().max().item()) if reference64.numel() else 0.0
    max_rel = max_abs / max(reference_scale, torch.finfo(torch.float64).tiny)
    rms = float(torch.sqrt(torch.mean(diff.square())).item()) if diff.numel() else 0.0
    return {
        "output_finite": output_finite,
        "parity_checked": True,
        "reference_finite": bool(torch.isfinite(reference).all().item()),
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "rms_error": rms,
    }


def _dense_reference_algebra(args, n: int, dtype: torch.dtype, device: str) -> Optional[CliffordAlgebra]:
    if args.disable_parity or n > args.parity_max_n:
        return None
    return CliffordAlgebra(n, 0, device=device, dtype=dtype, allow_large_dense=True)


def _dense_product_case(args, dtype: torch.dtype, device: str) -> BenchmarkCase:
    algebra = CliffordAlgebra(args.n, 0, device=device, dtype=dtype)
    left = algebra.embed_vector(torch.randn(args.batch_size, algebra.n, device=device, dtype=dtype))
    right = algebra.embed_vector(torch.randn(args.batch_size, algebra.n, device=device, dtype=dtype))

    return BenchmarkCase(
        name="dense_full_gp_1x1",
        host="CliffordAlgebra",
        n=algebra.n,
        op="gp",
        storage="dense",
        batch_size=args.batch_size,
        left_lanes=algebra.dim,
        right_lanes=algebra.dim,
        output_lanes=algebra.dim,
        pairwise=False,
        fn=lambda: algebra.geometric_product(left, right),
    )


def _compact_product_case(args, dtype: torch.dtype, device: str) -> BenchmarkCase:
    algebra = CliffordAlgebra(args.n, 0, device=device, dtype=dtype)
    layout_1 = algebra.layout((1,))
    layout_02 = algebra.layout((0, 2))
    left = layout_1.compact(algebra.embed_vector(torch.randn(args.batch_size, algebra.n, device=device, dtype=dtype)))
    right = layout_1.compact(algebra.embed_vector(torch.randn(args.batch_size, algebra.n, device=device, dtype=dtype)))

    layer = ProductLayer(
        algebra,
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
    )

    return BenchmarkCase(
        name="compact_layer_gp_1x1_to_02",
        host="CliffordAlgebra",
        n=algebra.n,
        op="gp",
        storage="compact",
        batch_size=args.batch_size,
        left_lanes=layout_1.dim,
        right_lanes=layout_1.dim,
        output_lanes=layout_02.dim,
        pairwise=False,
        fn=lambda: layer(left, right),
        reference_fn=lambda: layout_02.compact(
            algebra.geometric_product(layout_1.dense(left), layout_1.dense(right))
        ),
    )


def _context_compact_case(args, dtype: torch.dtype, device: str) -> BenchmarkCase:
    context = AlgebraContext(args.context_n, 0, device=device, dtype=dtype)
    layout_1 = context.layout((1,))
    layout_02 = context.layout((0, 2))
    left = torch.randn(args.batch_size, layout_1.dim, device=device, dtype=dtype)
    right = torch.randn(args.batch_size, layout_1.dim, device=device, dtype=dtype)
    reference_algebra = _dense_reference_algebra(args, context.n, dtype, device)
    reference_fn = None
    if reference_algebra is not None:
        reference_layout_1 = reference_algebra.layout((1,))
        reference_layout_02 = reference_algebra.layout((0, 2))

        def reference_product() -> torch.Tensor:
            return reference_layout_02.compact(
                reference_algebra.geometric_product(reference_layout_1.dense(left), reference_layout_1.dense(right))
            )

        reference_fn = reference_product

    return BenchmarkCase(
        name="context_compact_gp_1x1_to_02",
        host="AlgebraContext",
        n=context.n,
        op="gp",
        storage="compact",
        batch_size=args.batch_size,
        left_lanes=layout_1.dim,
        right_lanes=layout_1.dim,
        output_lanes=layout_02.dim,
        pairwise=False,
        fn=lambda: context.geometric_product(
            left,
            right,
            left_grades=(1,),
            right_grades=(1,),
            output_grades=(0, 2),
            active_output=True,
        ),
        reference_fn=reference_fn,
    )


def _pairwise_context_case(args, dtype: torch.dtype, device: str) -> BenchmarkCase:
    context = AlgebraContext(args.context_n, 0, device=device, dtype=dtype)
    layout_2 = context.layout((2,))
    layout_1 = context.layout((1,))
    layout_3 = context.layout((3,))
    left = torch.randn(args.batch_size, args.left_items, layout_2.dim, device=device, dtype=dtype)
    right = torch.randn(args.batch_size, args.right_items, layout_1.dim, device=device, dtype=dtype)
    reference_algebra = _dense_reference_algebra(args, context.n, dtype, device)
    reference_fn = None
    if reference_algebra is not None:
        reference_layout_2 = reference_algebra.layout((2,))
        reference_layout_1 = reference_algebra.layout((1,))
        reference_layout_3 = reference_algebra.layout((3,))

        def reference_pairwise_wedge() -> torch.Tensor:
            return reference_layout_3.compact(
                reference_algebra.wedge(
                    reference_layout_2.dense(left).unsqueeze(-2),
                    reference_layout_1.dense(right).unsqueeze(-3),
                )
            )

        reference_fn = reference_pairwise_wedge

    return BenchmarkCase(
        name="context_pairwise_wedge_2x1_to_3",
        host="AlgebraContext",
        n=context.n,
        op="wedge",
        storage="compact_pairwise",
        batch_size=args.batch_size * args.left_items * args.right_items,
        left_lanes=layout_2.dim,
        right_lanes=layout_1.dim,
        output_lanes=layout_3.dim,
        pairwise=True,
        fn=lambda: context.wedge(
            left,
            right,
            left_grades=(2,),
            right_grades=(1,),
            output_grades=(3,),
            active_output=True,
            pairwise=True,
        ),
        reference_fn=reference_fn,
    )


def _layer_pipeline_case(args, dtype: torch.dtype, device: str) -> BenchmarkCase:
    context = AlgebraContext(args.context_n, 0, device=device, dtype=dtype)
    layout_1 = context.layout((1,))
    layout_3 = context.layout((3,))

    class Pipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.wedge_vectors = WedgeLayer(
                context,
                left_grades=(1,),
                right_grades=(1,),
                output_grades=(2,),
            )
            self.wedge_trivector = WedgeLayer(
                context,
                left_grades=(2,),
                right_grades=(1,),
                output_grades=(3,),
            )

        def forward(self, left, right, third):
            bivector = self.wedge_vectors(left, right)
            return self.wedge_trivector(bivector, third)

    model = Pipeline().to(device=device)
    left = context.embed_vector(torch.randn(args.batch_size, context.n, device=device, dtype=dtype))
    right = context.embed_vector(torch.randn(args.batch_size, context.n, device=device, dtype=dtype))
    third = layout_1.compact(context.embed_vector(torch.randn(args.batch_size, context.n, device=device, dtype=dtype)))
    reference_algebra = _dense_reference_algebra(args, context.n, dtype, device)
    reference_fn = None
    if reference_algebra is not None:
        reference_layout_1 = reference_algebra.layout((1,))
        reference_layout_3 = reference_algebra.layout((3,))

        def reference_pipeline() -> torch.Tensor:
            return reference_layout_3.compact(
                reference_algebra.wedge(reference_algebra.wedge(left, right), reference_layout_1.dense(third))
            )

        reference_fn = reference_pipeline

    return BenchmarkCase(
        name="layer_pipeline_wedge_wedge",
        host="AlgebraContext",
        n=context.n,
        op="wedge_pipeline",
        storage="dense_to_compact",
        batch_size=args.batch_size,
        left_lanes=context.dim,
        right_lanes=layout_1.dim,
        output_lanes=layout_3.dim,
        pairwise=False,
        fn=lambda: model(left, right, third),
        reference_fn=reference_fn,
    )


def _compact_wedge_chain_case(args, dtype: torch.dtype, device: str) -> BenchmarkCase:
    context = AlgebraContext(args.context_n, 0, device=device, dtype=dtype)
    steps = max(1, min(args.chain_steps, context.n))
    layouts = {grade: context.layout((grade,)) for grade in range(1, steps + 1)}
    vector_layout = layouts[1]
    vectors = [torch.randn(args.batch_size, context.n, device=device, dtype=dtype) for _ in range(steps)]
    compact_vectors = [vector_layout.compact(context.embed_vector(vector)) for vector in vectors]
    output_layout = layouts[steps]

    def run_chain() -> torch.Tensor:
        value = compact_vectors[0]
        left_layout = vector_layout
        for output_grade, right in enumerate(compact_vectors[1:], start=2):
            next_layout = layouts[output_grade]
            value = context.wedge(
                value,
                right,
                left_layout=left_layout,
                right_layout=vector_layout,
                output_layout=next_layout,
                active_output=True,
            )
            left_layout = next_layout
        return value

    reference_algebra = _dense_reference_algebra(args, context.n, dtype, device)
    reference_fn = None
    if reference_algebra is not None:
        reference_output_layout = reference_algebra.layout((steps,))
        dense_vectors = [reference_algebra.embed_vector(vector) for vector in vectors]

        def reference_chain() -> torch.Tensor:
            value = dense_vectors[0]
            for right in dense_vectors[1:]:
                value = reference_algebra.wedge(value, right)
            return reference_output_layout.compact(value)

        reference_fn = reference_chain

    return BenchmarkCase(
        name=f"context_compact_wedge_chain_1_to_{steps}",
        host="AlgebraContext",
        n=context.n,
        op="wedge_chain",
        storage="compact",
        batch_size=args.batch_size,
        left_lanes=vector_layout.dim,
        right_lanes=vector_layout.dim,
        output_lanes=output_layout.dim,
        pairwise=False,
        fn=run_chain,
        reference_fn=reference_fn,
    )


def _write_results(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "framework_pipeline.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "summary.md"
    with summary_path.open("w") as handle:
        handle.write("# Framework Pipeline Benchmark\n\n")
        handle.write("| case | host | storage | n | median ms | items/sec | max abs error | max rel error |\n")
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write(
                f"| {row['name']} | {row['host']} | {row['storage']} | {row['n']} | "
                f"{row['median_ms']:.4f} | {row['items_per_sec']:.2f} | "
                f"{_format_metric(row['max_abs_error'])} | {_format_metric(row['max_rel_error'])} |\n"
            )


def _format_metric(value: float) -> str:
    if value != value:
        return "skipped"
    return f"{value:.3e}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, mps, or auto")
    parser.add_argument("--dtype", default="float32", choices=sorted(FLOAT_DTYPES))
    parser.add_argument("--n", type=int, default=6, help="Dense algebra dimension for dense-vs-compact comparison")
    parser.add_argument("--context-n", type=int, default=12, help="Planned context dimension")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--left-items", type=int, default=16)
    parser.add_argument("--right-items", type=int, default=16)
    parser.add_argument("--chain-steps", type=int, default=4, help="Number of vector factors in the compact wedge chain")
    parser.add_argument(
        "--parity-max-n",
        type=int,
        default=8,
        help="Largest signature dimension that will materialize a dense reference for parity checks",
    )
    parser.add_argument("--disable-parity", action="store_true", help="Skip dense-reference parity checks")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--quick", action="store_true", help="Use smaller dimensions and repeat counts")
    parser.add_argument("--no-save", action="store_true", help="Print rows without writing benchmark artifacts")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n = min(args.n, 5)
        args.context_n = min(args.context_n, 8)
        args.batch_size = min(args.batch_size, 64)
        args.left_items = min(args.left_items, 8)
        args.right_items = min(args.right_items, 8)
        args.chain_steps = min(args.chain_steps, 4)
        args.warmups = min(args.warmups, 2)
        args.repeats = min(args.repeats, 5)

    device = resolve_device(args.device) if args.device == "auto" else args.device
    dtype = FLOAT_DTYPES[args.dtype]
    torch.manual_seed(2026)

    cases = [
        _dense_product_case(args, dtype, device),
        _compact_product_case(args, dtype, device),
        _context_compact_case(args, dtype, device),
        _pairwise_context_case(args, dtype, device),
        _layer_pipeline_case(args, dtype, device),
        _compact_wedge_chain_case(args, dtype, device),
    ]
    rows = [_time_case(case, warmups=args.warmups, repeats=args.repeats, device=device) for case in cases]

    for row in rows:
        parity = (
            f", max_abs={row['max_abs_error']:.3e}, max_rel={row['max_rel_error']:.3e}"
            if row["parity_checked"]
            else ", parity=skipped"
        )
        print(
            f"{row['name']}: median={row['median_ms']:.4f} ms, "
            f"first={row['first_call_ms']:.4f} ms, items/sec={row['items_per_sec']:.2f}{parity}"
        )

    if not args.no_save:
        output_dir = args.output_dir
        if output_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = _REPO_ROOT / "benchmarks" / "results" / f"framework_pipeline_{stamp}"
        _write_results(rows, output_dir)
        print(f"Wrote benchmark artifacts to {output_dir}")


if __name__ == "__main__":
    main()
