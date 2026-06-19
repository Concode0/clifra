# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.core.runtime.algebra import AlgebraContext
from clifra.core.runtime.energy import lane_energy, lane_norm
from clifra.core.runtime.forms import (
    conjugate_scalar_form,
    conjugate_scalar_form_signs,
    signature_norm_squared,
)

pytestmark = pytest.mark.unit


def test_null_blade_has_positive_lane_energy_but_zero_signed_forms():
    algebra = AlgebraContext(1, 0, 1, device="cpu", dtype=torch.float64)
    values = torch.zeros(algebra.dim, dtype=torch.float64)
    null_index = 1 << algebra.p
    values[null_index] = 1.0

    assert torch.allclose(lane_energy(algebra, values), torch.tensor([1.0], dtype=torch.float64))
    assert torch.allclose(lane_norm(algebra, values), torch.tensor([1.0], dtype=torch.float64))
    assert torch.allclose(conjugate_scalar_form(algebra, values, values), torch.zeros(1, dtype=torch.float64))
    assert torch.allclose(signature_norm_squared(algebra, values), torch.zeros(1, dtype=torch.float64))


def test_conjugate_scalar_form_signs_preserve_null_degeneracy_in_layouts():
    algebra = AlgebraContext(2, 0, 1, device="cpu", dtype=torch.float64)
    vector_layout = algebra.layout((1,))
    signs = conjugate_scalar_form_signs(algebra, layout=vector_layout)
    null_basis_index = 1 << algebra.p
    null_position = vector_layout.basis_indices.index(null_basis_index)

    assert signs.shape == (vector_layout.dim,)
    assert signs[null_position] == 0.0
    assert (signs != 0).any()
