# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.core.runtime.algebra import AlgebraContext

pytestmark = pytest.mark.unit


def test_blade_inverse_preserves_negative_signature_denominator():
    algebra = AlgebraContext(0, 1, 0, device="cpu", dtype=torch.float64)
    blade = torch.zeros(1, algebra.dim, dtype=torch.float64)
    blade[0, 1] = 1.0

    inverse = algebra.blade_inverse(blade)
    product = algebra.geometric_product(blade, inverse)

    assert torch.allclose(inverse[0, 1], torch.tensor(-1.0, dtype=torch.float64))
    assert torch.allclose(product[0, 0], torch.tensor(1.0, dtype=torch.float64))


def test_blade_inverse_of_null_blade_stays_finite():
    algebra = AlgebraContext(0, 0, 1, device="cpu", dtype=torch.float32)
    blade = torch.zeros(1, algebra.dim)
    blade[0, 1] = 1.0

    inverse = algebra.blade_inverse(blade)

    assert torch.isfinite(inverse).all()
