# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.core._kernel.numerics import signed_clamp_min

pytestmark = pytest.mark.unit


def test_signed_clamp_preserves_negative_denominator_sign():
    values = torch.tensor([[-1.0e-30, 0.0, 1.0e-30]], dtype=torch.float64)

    clamped = signed_clamp_min(values, 1.0e-12)

    assert clamped[0, 0] < 0
    assert clamped[0, 1] > 0
    assert clamped[0, 2] > 0
    assert clamped.abs().min() >= 1.0e-12
