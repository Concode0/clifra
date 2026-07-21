# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest

import clifra.analysis as analysis
import clifra.core as core
import clifra.core.foundation as foundation
import clifra.core.runtime as runtime
from clifra.analysis.geodesic import NeighborhoodBivectorFlow
from clifra.analysis.signature import RotorProbeSignatureEstimator
from clifra.core.runtime.algebra import AlgebraContext

pytestmark = pytest.mark.unit


def test_runtime_package_has_no_lazy_getattr_import_bridge():
    assert "__getattr__" not in runtime.__dict__
    assert "AlgebraContext" not in runtime.__dict__
    assert AlgebraContext.__name__ == "AlgebraContext"


def test_core_packages_do_not_reexport_training_device_config():
    assert "DeviceConfig" not in core.__dict__
    assert "DeviceConfig" not in core.__all__
    assert "DeviceConfig" not in foundation.__dict__
    assert "DeviceConfig" not in foundation.__all__
