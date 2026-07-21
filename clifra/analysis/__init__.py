# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Descriptive and geometric-representation diagnostics.

The package measures coefficient-space and operator structure. Its
rotor-probe signature result ranks candidate algebras as a learned heuristic.
"""

from ._types import (
    CONSTANTS,
    AnalysisConfig,
    AnalysisConstants,
    AnalysisReport,
    CommutatorResult,
    DimensionResult,
    SamplingConfig,
    SignatureEstimate,
    SpectralResult,
    TransformationDiagnosticsResult,
)
from .commutator import CommutatorAnalyzer
from .dimension import CovarianceDimensionAnalyzer
from .pipeline import GeometricAnalyzer
from .policy import (
    AnalysisCostPolicy,
    AnalysisFeasibility,
    MatrixAnalysisCost,
    ProductAnalysisCost,
)
from .sampler import StatisticalSampler
from .signature import SignatureProbeAnalyzer
from .spectral import SpectralAnalyzer
from .symmetry import TransformationDiagnosticsAnalyzer

__all__ = [
    # Constants
    "AnalysisConstants",
    "CONSTANTS",
    # Config / result types
    "SamplingConfig",
    "AnalysisConfig",
    "DimensionResult",
    "SignatureEstimate",
    "SpectralResult",
    "TransformationDiagnosticsResult",
    "CommutatorResult",
    "AnalysisReport",
    "AnalysisCostPolicy",
    "AnalysisFeasibility",
    "MatrixAnalysisCost",
    "ProductAnalysisCost",
    # Analyzers
    "StatisticalSampler",
    "CovarianceDimensionAnalyzer",
    "SignatureProbeAnalyzer",
    "SpectralAnalyzer",
    "TransformationDiagnosticsAnalyzer",
    "CommutatorAnalyzer",
    "GeometricAnalyzer",
]
