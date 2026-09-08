# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import importlib
import subprocess
import sys

import pytest

import clifra.analysis as analysis
from clifra.analysis import experimental

pytestmark = pytest.mark.unit


def test_exact_public_surfaces():
    assert analysis.__all__ == [
        "SpectralAnalyzer",
        "SpectralResult",
        "CommutatorAnalyzer",
        "CommutatorResult",
        "TransformationDiagnosticsAnalyzer",
        "TransformationDiagnosticsResult",
    ]
    assert experimental.__all__ == [
        "SignatureProbeAnalyzer",
        "SignatureProbeResult",
        "NeighborhoodBivectorAnalyzer",
        "compare_coordinate_lifts",
    ]
    for module, names in (
        ("spectral", ["SpectralAnalyzer", "SpectralResult"]),
        ("commutator", ["CommutatorAnalyzer", "CommutatorResult"]),
        ("transformation", ["TransformationDiagnosticsAnalyzer", "TransformationDiagnosticsResult"]),
        ("experimental.signature", ["SignatureProbeAnalyzer", "SignatureProbeResult"]),
        ("experimental.neighborhood", ["NeighborhoodBivectorAnalyzer", "compare_coordinate_lifts"]),
    ):
        owner = importlib.import_module("clifra.analysis." + module)
        assert owner.__all__ == names
        for name in names:
            assert getattr(owner, name).__module__ == owner.__name__


def test_stable_import_does_not_load_experimental_code():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import clifra.analysis
assert not any(name.startswith('clifra.analysis.experimental') for name in sys.modules)
""",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "name",
    ["dimension", "sampler", "pipeline", "signature", "geodesic", "symmetry", "_types", "_statistics", "_sampling"],
)
def test_obsolete_modules_are_removed(name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("clifra.analysis." + name)


def test_old_symbols_have_no_compatibility_bridges():
    for name in (
        "GeometricAnalyzer",
        "AnalysisConfig",
        "AnalysisReport",
        "AnalysisConstants",
        "CONSTANTS",
        "DimensionResult",
        "CovarianceDimensionAnalyzer",
        "SamplingConfig",
        "StatisticalSampler",
        "SignatureProbeAnalyzer",
    ):
        assert not hasattr(analysis, name)
    for name in (
        "RotorProbeSignatureEstimator",
        "CoordinateLiftAnalyzer",
        "approximate_bivector_interpolation",
        "SignatureEstimate",
        "NeighborhoodBivectorFlow",
    ):
        assert not hasattr(experimental, name)
