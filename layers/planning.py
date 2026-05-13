# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Static optimization descriptors for composed layer graphs.

The translator side is compile-time only: it inspects modules and returns
immutable route metadata before ``torch.compile`` captures tensor computation.
It must not execute algebra or branch on runtime tensor values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn

from core.foundation.layout import GradeLayout
from core.foundation.module import AlgebraLike
from core.foundation.validation import VALIDATE, check_multivector

__all__ = [
    "LayerOptimizationPlan",
    "resolve_layer_layout",
    "lane_count",
    "check_multivector_lanes",
    "layer_optimization_plan",
    "collect_layer_optimization_plans",
]


@dataclass(frozen=True)
class LayerOptimizationPlan:
    """Static metadata for one algebra-aware module in a composed model."""

    path: str
    module_type: str
    operators: tuple[str, ...]
    input_grades: Optional[tuple[int, ...]]
    output_grades: Optional[tuple[int, ...]]
    parameter_grades: Optional[tuple[int, ...]]
    score_grades: Optional[tuple[int, ...]]
    basis_dim: int
    dense_dim: int
    compact: bool
    dense_only_reason: Optional[str] = None

    @property
    def compression_ratio(self) -> float:
        """Return active basis lanes divided by dense basis lanes."""
        if self.dense_dim == 0:
            return 1.0
        return self.basis_dim / self.dense_dim

    def uses_grade(self, grade: int) -> bool:
        """Return whether the plan mentions ``grade`` in any static grade slot."""
        grade = int(grade)
        grade_sets = (self.input_grades, self.output_grades, self.parameter_grades, self.score_grades)
        return any(grades is not None and grade in grades for grades in grade_sets)


def resolve_layer_layout(algebra: AlgebraLike, grades: Optional[Iterable[int]]) -> Optional[GradeLayout]:
    """Return a compact layout for declared grades, or ``None`` for dense lanes."""
    if grades is None:
        return None
    return algebra.planner.layout(grades)


def lane_count(algebra: AlgebraLike, layout: Optional[GradeLayout]) -> int:
    """Return the active basis-lane count for a declared layout."""
    return algebra.dim if layout is None else layout.dim


def check_multivector_lanes(
    values: torch.Tensor,
    algebra: AlgebraLike,
    layout: Optional[GradeLayout],
    name: str,
) -> None:
    """Validate dense or declared compact multivector lanes."""
    if layout is None:
        check_multivector(values, algebra, name)
        return
    if not VALIDATE:
        return
    assert values.ndim >= 1, f"{name}: expected ndim >= 1, got shape {tuple(values.shape)}"
    assert values.shape[-1] == layout.dim, (
        f"{name}: last dim should be {layout.dim} for grades {layout.grades}, "
        f"got {values.shape[-1]} (shape {tuple(values.shape)})"
    )


def layer_optimization_plan(module: nn.Module, *, path: str = "") -> Optional[LayerOptimizationPlan]:
    """Return static optimization metadata for ``module`` when it exposes algebra routes."""
    custom_plan = _custom_plan(module, path)
    if custom_plan is not None:
        return custom_plan

    algebra = _module_algebra(module)
    if algebra is None:
        return None

    layout = _declared_layout(module)
    score_grades = _score_grades(module)
    parameter_grades = _parameter_grades(module)
    operators = _operators(module)
    if layout is None and score_grades is None and parameter_grades is None and not operators:
        return None

    input_grades, output_grades = _io_grades(module, layout)
    compact = layout is not None
    dense_dim = int(algebra.dim)
    basis_dim = lane_count(algebra, layout)
    dense_only_reason = None if compact else _dense_only_reason(module)
    return LayerOptimizationPlan(
        path=path or "<root>",
        module_type=module.__class__.__name__,
        operators=operators,
        input_grades=input_grades,
        output_grades=output_grades,
        parameter_grades=parameter_grades,
        score_grades=score_grades,
        basis_dim=basis_dim,
        dense_dim=dense_dim,
        compact=compact,
        dense_only_reason=dense_only_reason,
    )


def collect_layer_optimization_plans(
    module: nn.Module,
    *,
    compact_only: bool = False,
) -> tuple[LayerOptimizationPlan, ...]:
    """Collect static optimization metadata from a composed module tree."""
    plans = []
    for path, child in module.named_modules():
        plan = layer_optimization_plan(child, path=path)
        if plan is None:
            continue
        if compact_only and not plan.compact:
            continue
        plans.append(plan)
    return tuple(plans)


def _custom_plan(module: nn.Module, path: str) -> Optional[LayerOptimizationPlan]:
    plan_fn = getattr(module, "optimization_plan", None)
    if plan_fn is None or not callable(plan_fn):
        return None
    try:
        plan = plan_fn(path=path or "<root>")
    except TypeError:
        plan = plan_fn()
    if plan is None:
        return None
    if not isinstance(plan, LayerOptimizationPlan):
        raise TypeError(f"{module.__class__.__name__}.optimization_plan() must return LayerOptimizationPlan or None")
    return plan


def _module_algebra(module: nn.Module) -> Optional[AlgebraLike]:
    algebra = getattr(module, "algebra", None)
    if algebra is None:
        algebra = getattr(module, "_algebra", None)
    if algebra is None or not hasattr(algebra, "planner") or not hasattr(algebra, "dim"):
        return None
    return algebra


def _declared_layout(module: nn.Module) -> Optional[GradeLayout]:
    layout = getattr(module, "layout", None)
    if layout is None:
        layout = getattr(module, "feature_layout", None)
    return layout


def _grades_from_layout(layout: Optional[GradeLayout]) -> Optional[tuple[int, ...]]:
    if layout is None:
        return None
    return tuple(int(grade) for grade in layout.grades)


def _score_grades(module: nn.Module) -> Optional[tuple[int, ...]]:
    grades = getattr(module, "score_grades", None)
    if grades is None:
        score_layout = getattr(module, "_score_layout", None)
        return _grades_from_layout(score_layout)
    return tuple(int(grade) for grade in grades)


def _parameter_grades(module: nn.Module) -> Optional[tuple[int, ...]]:
    if hasattr(module, "grade"):
        return (int(getattr(module, "grade")),)
    return None


def _io_grades(
    module: nn.Module,
    layout: Optional[GradeLayout],
) -> tuple[Optional[tuple[int, ...]], Optional[tuple[int, ...]]]:
    grades = _grades_from_layout(layout)
    if module.__class__.__name__ in {"MultivectorEmbedding", "MotherEmbedding"}:
        return None, grades
    return grades, grades


def _operators(module: nn.Module) -> tuple[str, ...]:
    module_type = module.__class__.__name__
    if module_type == "CliffordLinear":
        return (f"linear:{getattr(module, 'backend', 'traditional')}",)
    if module_type == "CliffordLayerNorm":
        return ("normalize",)
    if module_type == "BladeSelector":
        return ("blade_gate",)
    if module_type == "GeometricProductAttention":
        return ("linear", "gp_score", "softmax", "linear")
    if module_type == "EntropyGatedAttention":
        return ("grade_energy", "gate", "attention")
    if module_type in {"MultivectorEmbedding", "MotherEmbedding"}:
        return ("embed",)
    if module_type == "PhaseShiftHead":
        return ("grade_readout",)
    if module_type in {"RotorLayer", "MultiRotorLayer", "RotaryBivectorPE", "ReflectionLayer"}:
        return ("dense_sandwich",)
    if module_type == "RotorGadget":
        return ("dense_rotor_toolbox",)
    if module_type == "GeometricNeutralizer":
        return ("grade_projection", "linear_solve")
    return ()


def _dense_only_reason(module: nn.Module) -> Optional[str]:
    module_type = module.__class__.__name__
    if module_type in {"RotorLayer", "MultiRotorLayer", "RotaryBivectorPE", "ReflectionLayer", "RotorGadget"}:
        return "sandwich path still materializes dense multivectors"
    if module_type == "GeometricNeutralizer":
        return "neutralizer reads fixed dense grade positions"
    if module_type == "CliffordLinear" and getattr(module, "backend", "traditional") == "rotor":
        return "rotor backend requires dense sandwich execution"
    return "no compact grade layout declared"
