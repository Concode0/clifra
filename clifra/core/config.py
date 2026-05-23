# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Algebra construction config and backend selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Optional

import torch

from clifra.core.foundation.device import resolve_device, resolve_dtype
from clifra.core.foundation.module import AlgebraLike
from clifra.core.planning.policy import DENSE_AUTO_MAX_N, PlanningLimits
from clifra.core.runtime.algebra import AlgebraContext, CliffordAlgebra

AlgebraKernel = Literal["auto", "dense", "context"]


@dataclass(frozen=True)
class AlgebraConfig:
    """Dense/context algebra declaration."""

    p: int
    q: int = 0
    r: int = 0
    kernel: AlgebraKernel = "auto"
    dense_threshold: int = DENSE_AUTO_MAX_N
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    exp_policy: str = "balanced"
    fixed_iterations: Optional[int] = None
    default_grades: Optional[tuple[int, ...]] = None
    planning_limits: Optional[PlanningLimits] = None

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any], **overrides) -> "AlgebraConfig":
        """Build an algebra declaration from Hydra/OmegaConf config."""
        values = {
            "p": int(_mapping_get(config, "p", 0)),
            "q": int(_mapping_get(config, "q", 0)),
            "r": int(_mapping_get(config, "r", 0)),
            "kernel": _mapping_get(config, "kernel", "auto"),
            "dense_threshold": int(_mapping_get(config, "dense_threshold", DENSE_AUTO_MAX_N)),
            "device": _mapping_get(config, "device", "cuda"),
            "dtype": resolve_dtype(_mapping_get(config, "dtype", torch.float32)),
            "exp_policy": _mapping_get(config, "exp_policy", "balanced"),
            "fixed_iterations": _optional_int(_mapping_get(config, "fixed_iterations", None)),
            "default_grades": _optional_grades(_mapping_get(config, "default_grades", None)),
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        values["dtype"] = resolve_dtype(values["dtype"])
        return cls(**values)


def make_algebra(
    p: int,
    q: int = 0,
    r: int = 0,
    *,
    kernel: AlgebraKernel = "auto",
    dense_threshold: int = DENSE_AUTO_MAX_N,
    device="cuda",
    dtype: torch.dtype = torch.float32,
    exp_policy: str = "balanced",
    fixed_iterations: Optional[int] = None,
    default_grades: Optional[Iterable[int]] = None,
    planning_limits: Optional[PlanningLimits] = None,
) -> AlgebraLike:
    """Construct a dense low-dimensional algebra or high-dimensional planning context."""
    kernel = _normalize_kernel(kernel)
    n = p + q + r
    selected_kernel = "context" if kernel == "auto" and n > dense_threshold else kernel
    if selected_kernel == "auto":
        selected_kernel = "dense"

    resolved_device = resolve_device(device) if str(device) == "auto" else device
    resolved_dtype = resolve_dtype(dtype)

    if selected_kernel == "dense":
        return CliffordAlgebra(
            p,
            q,
            r,
            device=resolved_device,
            dtype=resolved_dtype,
            exp_policy=exp_policy,
            fixed_iterations=fixed_iterations,
            allow_large_dense=kernel == "dense",
            planning_limits=planning_limits,
        )

    return AlgebraContext(
        p,
        q,
        r,
        device=resolved_device,
        dtype=resolved_dtype,
        default_grades=default_grades,
        planning_limits=planning_limits,
    )


def make_algebra_from_config(config: Mapping[str, Any], **overrides) -> AlgebraLike:
    """Construct an algebra from a Hydra/OmegaConf-compatible config mapping."""
    algebra_config = AlgebraConfig.from_mapping(config, **overrides)
    return make_algebra(
        algebra_config.p,
        algebra_config.q,
        algebra_config.r,
        kernel=algebra_config.kernel,
        dense_threshold=algebra_config.dense_threshold,
        device=algebra_config.device,
        dtype=algebra_config.dtype,
        exp_policy=algebra_config.exp_policy,
        fixed_iterations=algebra_config.fixed_iterations,
        default_grades=algebra_config.default_grades,
        planning_limits=algebra_config.planning_limits,
    )


def _mapping_get(config: Mapping[str, Any], key: str, default):
    """Return a value from plain mappings or OmegaConf DictConfig objects."""
    if config is None:
        return default
    return config.get(key, default)


def _normalize_kernel(kernel: str) -> AlgebraKernel:
    """Validate and normalize algebra kernel names."""
    normalized = str(kernel).lower()
    if normalized not in {"auto", "dense", "context"}:
        raise ValueError(f"Unknown algebra kernel {kernel!r}; expected 'auto', 'dense', or 'context'")
    return normalized  # type: ignore[return-value]


def _optional_int(value) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _optional_grades(value) -> Optional[tuple[int, ...]]:
    if value is None:
        return None
    return tuple(int(grade) for grade in value)

