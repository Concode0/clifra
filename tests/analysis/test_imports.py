# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import importlib
import subprocess
import sys

import pytest

import clifra.analysis as analysis
from clifra.analysis import experimental

pytestmark = pytest.mark.unit


def test_public_surfaces_expose_stable_analyzers_from_owning_modules():
    assert {
        "SpectralAnalyzer",
        "SpectralResult",
        "CommutatorAnalyzer",
        "CommutatorResult",
        "TransformationDiagnosticsAnalyzer",
        "TransformationDiagnosticsResult",
    } <= set(analysis.__all__)
    assert {
        "SignatureProbeAnalyzer",
        "SignatureProbeResult",
        "NeighborhoodBivectorAnalyzer",
        "compare_coordinate_lifts",
    } <= set(experimental.__all__)
    for module, names in (
        ("spectral", ["SpectralAnalyzer", "SpectralResult"]),
        ("commutator", ["CommutatorAnalyzer", "CommutatorResult"]),
        ("transformation", ["TransformationDiagnosticsAnalyzer", "TransformationDiagnosticsResult"]),
        ("experimental.signature", ["SignatureProbeAnalyzer", "SignatureProbeResult"]),
        ("experimental.neighborhood", ["NeighborhoodBivectorAnalyzer", "compare_coordinate_lifts"]),
    ):
        owner = importlib.import_module("clifra.analysis." + module)
        for name in names:
            assert name in owner.__all__
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
