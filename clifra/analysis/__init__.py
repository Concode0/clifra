# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Experimental descriptive and geometric-representation diagnostics.

Results report the implemented coefficient-space or operator calculation.
Metric, manifold, causal, and symmetry conclusions require evidence beyond
these heuristic outputs.
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
