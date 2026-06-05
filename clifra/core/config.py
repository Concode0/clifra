# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Algebra construction config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import torch

from clifra.core.foundation.device import resolve_device, resolve_dtype
from clifra.core.foundation.module import AlgebraLike
from clifra.core.planning.policy import PlanningLimits, ProductExecutionPolicy
from clifra.core.runtime.algebra import AlgebraContext


@dataclass(frozen=True)
class AlgebraConfig:
    """Planner-first algebra declaration."""

    p: int
    q: int = 0
    r: int = 0
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    exp_policy: str = "balanced"
    fixed_iterations: Optional[int] = None
    default_grades: Optional[tuple[int, ...]] = None
    planning_limits: Optional[PlanningLimits] = None
    product_execution_policy: Optional[ProductExecutionPolicy] = None

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any], **overrides) -> "AlgebraConfig":
        """Build an algebra declaration from a mapping-like config."""
        values = {
            "p": int(_mapping_get(config, "p", 0)),
            "q": int(_mapping_get(config, "q", 0)),
            "r": int(_mapping_get(config, "r", 0)),
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
    device="cuda",
    dtype: torch.dtype = torch.float32,
    exp_policy: str = "balanced",
    fixed_iterations: Optional[int] = None,
    default_grades: Optional[Iterable[int]] = None,
    planning_limits: Optional[PlanningLimits] = None,
    product_execution_policy: Optional[ProductExecutionPolicy] = None,
) -> AlgebraLike:
    """Construct the planner-owned algebra host."""
    resolved_device = resolve_device(device) if str(device) == "auto" else device
    resolved_dtype = resolve_dtype(dtype)

    return AlgebraContext(
        p,
        q,
        r,
        device=resolved_device,
        dtype=resolved_dtype,
        default_grades=default_grades,
        exp_policy=exp_policy,
        fixed_iterations=fixed_iterations,
        planning_limits=planning_limits,
        product_execution_policy=product_execution_policy,
    )


def make_algebra_from_config(config: Mapping[str, Any], **overrides) -> AlgebraLike:
    """Construct an algebra from a mapping-like config."""
    algebra_config = AlgebraConfig.from_mapping(config, **overrides)
    return make_algebra(
        algebra_config.p,
        algebra_config.q,
        algebra_config.r,
        device=algebra_config.device,
        dtype=algebra_config.dtype,
        exp_policy=algebra_config.exp_policy,
        fixed_iterations=algebra_config.fixed_iterations,
        default_grades=algebra_config.default_grades,
        planning_limits=algebra_config.planning_limits,
        product_execution_policy=algebra_config.product_execution_policy,
    )


def _mapping_get(config: Mapping[str, Any], key: str, default):
    """Return a value from a mapping-like object."""
    if config is None:
        return default
    return config.get(key, default)


def _optional_int(value) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _optional_grades(value) -> Optional[tuple[int, ...]]:
    if value is None:
        return None
    return tuple(int(grade) for grade in value)
