# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest

import clifra.core as core
import clifra.core.analysis as analysis
import clifra.core.runtime as runtime
from clifra.core.analysis.geodesic import GeodesicFlow
from clifra.core.analysis.signature import MetricSearch
from clifra.core.runtime.algebra import AlgebraContext

pytestmark = pytest.mark.unit


def test_runtime_package_has_no_lazy_getattr_import_bridge():
    assert "__getattr__" not in runtime.__dict__
    assert "AlgebraContext" not in runtime.__dict__
    assert AlgebraContext.__name__ == "AlgebraContext"


def test_core_package_has_no_lazy_analysis_getattr_bridge():
    assert "__getattr__" not in core.__dict__
    assert "MetricSearch" not in core.__dict__
    assert "GeodesicFlow" not in core.__dict__


def test_analysis_package_does_not_reexport_legacy_search_names():
    assert "MetricSearch" not in analysis.__all__
    assert "GeodesicFlow" not in analysis.__all__
    assert not hasattr(analysis, "MetricSearch")
    assert not hasattr(analysis, "GeodesicFlow")
    assert MetricSearch.__name__ == "MetricSearch"
    assert GeodesicFlow.__name__ == "GeodesicFlow"
