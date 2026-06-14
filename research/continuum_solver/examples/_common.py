# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for continuum solver research examples."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

DTYPE_CHOICES = ("float64", "float32")
DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")
DEFAULT_DTYPE = torch.float64


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved example runtime settings."""

    device: torch.device
    dtype: torch.dtype
    compile_step: bool
    compile_backend: str | None
    compile_mode: str | None
    compile_fullgraph: bool


def add_runtime_arguments(parser) -> None:
    """Add shared runtime options to an example parser."""
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="cpu")
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default="float64")
    parser.add_argument("--compile", action="store_true", dest="compile_step", help="compile the loss step with torch.compile")
    parser.add_argument("--compile-backend", default=None)
    parser.add_argument("--compile-mode", default=None)
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--max-threads", type=int, default=4)


def resolve_runtime(args) -> RuntimeConfig:
    """Resolve device and dtype options without introducing lower-precision modes."""
    dtype = resolve_dtype(args.dtype)
    device = resolve_device(args.device, dtype=dtype)
    return RuntimeConfig(
        device=device,
        dtype=dtype,
        compile_step=bool(args.compile_step),
        compile_backend=args.compile_backend,
        compile_mode=args.compile_mode,
        compile_fullgraph=bool(args.compile_fullgraph),
    )


def resolve_dtype(name: str) -> torch.dtype:
    """Resolve the only supported example dtypes."""
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"dtype must be one of {DTYPE_CHOICES}, got {name!r}")


def resolve_device(name: str, *, dtype: torch.dtype) -> torch.device:
    """Resolve a requested device and validate dtype support."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if dtype == torch.float32 and _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if name == "mps":
        if not _mps_available():
            raise ValueError("MPS was requested but is not available")
        if dtype != torch.float32:
            raise ValueError("MPS does not support float64 here; use --dtype float32")
    return torch.device(name)


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def configure_runtime(*, seed: int = 0, max_threads: int = 4) -> None:
    """Keep examples deterministic and reduce small-tensor CPU thread overhead."""
    torch.manual_seed(int(seed))
    if max_threads > 0:
        torch.set_num_threads(max(1, min(int(max_threads), torch.get_num_threads())))


def bootstrap_repo_root(file: str) -> None:
    """Make the repository root importable when an example is run as a script."""
    root = Path(file).resolve().parents[3]
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def coordinate_grid_2d(
    height: int,
    width: int,
    *,
    device=None,
    dtype=DEFAULT_DTYPE,
    extent: float = 1.0,
) -> torch.Tensor:
    """Return a direct coordinate tensor with shape ``[height, width, 2]``."""
    y = torch.linspace(-float(extent), float(extent), int(height), device=device, dtype=dtype)
    x = torch.linspace(-float(extent), float(extent), int(width), device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def coordinate_grid_3d(
    depth: int,
    height: int,
    width: int,
    *,
    device=None,
    dtype=DEFAULT_DTYPE,
    extent: float = 1.0,
) -> torch.Tensor:
    """Return a direct coordinate tensor with shape ``[depth, height, width, 3]``."""
    z = torch.linspace(-float(extent), float(extent), int(depth), device=device, dtype=dtype)
    y = torch.linspace(-float(extent), float(extent), int(height), device=device, dtype=dtype)
    x = torch.linspace(-float(extent), float(extent), int(width), device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1)


def print_latest(history, *, title: str, prefixes: Iterable[str] | None = None, limit: int = 24) -> None:
    """Print a compact metric summary from the latest logged record."""
    latest = history.latest()
    if latest is None:
        print(f"{title}: no metrics logged")
        return
    print(f"\n== {title} ==")
    print(f"step: {latest.step}")
    selected = latest.metrics.items()
    if prefixes is not None:
        prefix_tuple = tuple(prefixes)
        selected = [(key, value) for key, value in selected if key.startswith(prefix_tuple)]
    for count, (key, value) in enumerate(selected):
        if count >= limit:
            print("...")
            break
        print(f"{key}: {_format_metric(value)}")


def _format_metric(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (float, int)):
        return f"{float(value):.8g}"
    if isinstance(value, torch.Tensor):
        detached = value.detach()
        if detached.numel() == 0:
            return "nan"
        if detached.numel() > 1:
            detached = detached.float().mean()
        if detached.dtype == torch.bool:
            return str(bool(detached.cpu().reshape(())))
        return f"{float(detached.cpu().reshape(())):.8g}"
    return str(value)
