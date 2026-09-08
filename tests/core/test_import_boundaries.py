# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest

import clifra.core as core
import clifra.core._kernel.device as device
from clifra.core.algebra import AlgebraContext

pytestmark = pytest.mark.unit


def test_public_api_has_no_lazy_import_bridge_or_kernel_exports():
    assert "__getattr__" not in core.__dict__
    assert not {"GradePlanner", "PlanFacts", "PlanningPolicy", "GradeProductExecutor", "AlgebraLike"} & set(
        core.__all__
    )
    assert AlgebraContext.__name__ == "AlgebraContext"
    assert AlgebraContext.__bases__ == (object,)


def test_training_device_config_is_not_part_of_core():
    assert "DeviceConfig" not in device.__dict__
    assert "DeviceConfig" not in core.__dict__
    assert "DeviceConfig" not in core.__all__
