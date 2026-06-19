# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Continuum solver engine for coordinate-tensor morphing research."""

from .criteria import TargetFieldCriterion
from .curriculum import ConstantCurriculum, CurriculumKnot, LossWeightSchedule, PhaseCurriculum
from .engine import ContinuumSolverEngine, OptimizationStepContext, SolverRun
from .field import CoordinateChart, InvertibleBivectorField
from .logging import MetricLogger, MetricRecord
from .policies import BivectorNormPolicy, InvertiblePathConsistencyPolicy
from .types import (
    ContinuumState,
    CriterionResult,
    GeometricPolicy,
    PolicyResult,
    SolverEvaluation,
    TargetCriterion,
)

__all__ = [
    "BivectorNormPolicy",
    "ConstantCurriculum",
    "ContinuumSolverEngine",
    "ContinuumState",
    "CoordinateChart",
    "CriterionResult",
    "CurriculumKnot",
    "GeometricPolicy",
    "InvertibleBivectorField",
    "InvertiblePathConsistencyPolicy",
    "LossWeightSchedule",
    "MetricLogger",
    "MetricRecord",
    "OptimizationStepContext",
    "PhaseCurriculum",
    "PolicyResult",
    "SolverEvaluation",
    "SolverRun",
    "TargetCriterion",
    "TargetFieldCriterion",
]
