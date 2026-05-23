# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Geometric data analysis toolkit.

Provides analyzers for effective dimension, metric signature, spectral
structure, symmetry / null detection, and commutator (exchange) analysis,
orchestrated by :class:`GeometricAnalyzer`.

Also re-exports :class:`MetricSearch`,
:class:`GeodesicFlow`, :class:`DimensionLifter`, and the standalone
:func:`compute_uncertainty_and_alignment` that were previously in
``clifra.core.search``.
"""

from ._types import (
    CONSTANTS,
    AnalysisConfig,
    AnalysisConstants,
    AnalysisReport,
    CommutatorResult,
    DimensionResult,
    SamplingConfig,
    SignatureResult,
    SpectralResult,
    SymmetryResult,
)
from .commutator import CommutatorAnalyzer, compute_uncertainty_and_alignment
from .dimension import DimensionLifter, EffectiveDimensionAnalyzer
from .geodesic import GeodesicFlow
from .pipeline import GeometricAnalyzer
from .sampler import StatisticalSampler
from .signature import MetricSearch, SignatureSearchAnalyzer
from .spectral import SpectralAnalyzer
from .symmetry import SymmetryDetector

__all__ = [
    # Constants
    "AnalysisConstants",
    "CONSTANTS",
    # Config / result types
    "SamplingConfig",
    "AnalysisConfig",
    "DimensionResult",
    "SignatureResult",
    "SpectralResult",
    "SymmetryResult",
    "CommutatorResult",
    "AnalysisReport",
    # Analyzers
    "StatisticalSampler",
    "EffectiveDimensionAnalyzer",
    "SignatureSearchAnalyzer",
    "SpectralAnalyzer",
    "SymmetryDetector",
    "CommutatorAnalyzer",
    "GeometricAnalyzer",
    # Legacy (from clifra.core.search)
    "MetricSearch",
    "GeodesicFlow",
    "DimensionLifter",
    "compute_uncertainty_and_alignment",
]
