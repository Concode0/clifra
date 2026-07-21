# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Differentiable Clifford transformation fields with injectable objectives."""

from .criteria import TargetFieldCriterion
from .curriculum import ConstantCurriculum, CurriculumKnot, LossWeightSchedule, PhaseCurriculum
from .engine import ContinuumSolverEngine, OptimizationStepContext, SolverRun
from .field import CoordinateChart, InvertibleBivectorField
from .inputs import CoordinateFieldInput, CoordinateLike
from .logging import MetricLogger, MetricRecord
from .policies import BivectorNormPolicy, InvertiblePathConsistencyPolicy
from .sampling import (
    BroadcastGeneratorSampler,
    GeneratorFieldSample,
    GeneratorFieldSampler,
    RBFGeneratorSampler,
    RegularGridGeneratorSampler,
)
from .types import (
    ContinuumState,
    CoordinateTransformationField,
    CriterionResult,
    GeometricPolicy,
    PolicyResult,
    SolverEvaluation,
    TargetCriterion,
)

# Generic terminology for callers that do not use continuum-mechanics policies.
TransformationFieldEngine = ContinuumSolverEngine
TransformationState = ContinuumState

__all__ = [
    "BivectorNormPolicy",
    "BroadcastGeneratorSampler",
    "ConstantCurriculum",
    "ContinuumSolverEngine",
    "ContinuumState",
    "CoordinateChart",
    "CoordinateFieldInput",
    "CoordinateLike",
    "CoordinateTransformationField",
    "CriterionResult",
    "CurriculumKnot",
    "GeometricPolicy",
    "GeneratorFieldSample",
    "GeneratorFieldSampler",
    "InvertibleBivectorField",
    "InvertiblePathConsistencyPolicy",
    "LossWeightSchedule",
    "MetricLogger",
    "MetricRecord",
    "OptimizationStepContext",
    "PhaseCurriculum",
    "PolicyResult",
    "RBFGeneratorSampler",
    "RegularGridGeneratorSampler",
    "SolverEvaluation",
    "SolverRun",
    "TargetCriterion",
    "TargetFieldCriterion",
    "TransformationFieldEngine",
    "TransformationState",
]
