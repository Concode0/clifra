# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Geometric data analysis toolkit.

Provides analyzers for effective dimension, metric signature, spectral
structure, symmetry / null detection, and commutator (exchange) analysis,
orchestrated by :class:`GeometricAnalyzer`.
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
from .commutator import CommutatorAnalyzer
from .dimension import EffectiveDimensionAnalyzer
from .pipeline import GeometricAnalyzer
from .policy import (
    AnalysisCostPolicy,
    AnalysisFeasibility,
    MatrixAnalysisCost,
    ProductAnalysisCost,
)
from .sampler import StatisticalSampler
from .signature import SignatureSearchAnalyzer
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
    "AnalysisCostPolicy",
    "AnalysisFeasibility",
    "MatrixAnalysisCost",
    "ProductAnalysisCost",
    # Analyzers
    "StatisticalSampler",
    "EffectiveDimensionAnalyzer",
    "SignatureSearchAnalyzer",
    "SpectralAnalyzer",
    "SymmetryDetector",
    "CommutatorAnalyzer",
    "GeometricAnalyzer"
]
