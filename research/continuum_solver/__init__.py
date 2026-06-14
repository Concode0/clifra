# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Continuum solver engine for coordinate-tensor morphing research."""

from .criteria import TargetFieldCriterion
from .engine import ContinuumSolverEngine, SolverRun
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
    "ContinuumSolverEngine",
    "ContinuumState",
    "CoordinateChart",
    "CriterionResult",
    "GeometricPolicy",
    "InvertibleBivectorField",
    "InvertiblePathConsistencyPolicy",
    "MetricLogger",
    "MetricRecord",
    "PolicyResult",
    "SolverEvaluation",
    "SolverRun",
    "TargetCriterion",
    "TargetFieldCriterion",
]
