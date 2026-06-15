# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.core import Multivector, basis_blade_label, format_multivector
from clifra.core.runtime.algebra import AlgebraContext

pytestmark = pytest.mark.unit


def test_basis_blade_label_uses_canonical_bitmask_order():
    assert basis_blade_label(0, n=3) == "1"
    assert basis_blade_label(1, n=3) == "e1"
    assert basis_blade_label(3, n=3) == "e12"
    assert basis_blade_label((1 << 0) | (1 << 9), n=10) == "e[1,10]"


def test_format_full_multivector_terms_without_layout_noise():
    algebra = AlgebraContext(3, 0, 0, device="cpu", dtype=torch.float64)
    values = torch.zeros(algebra.dim, dtype=torch.float64)
    values[0] = 1.0
    values[1] = 2.0
    values[3] = -0.5
    values[7] = 1e-14

    assert format_multivector(algebra, values) == "1 + 2e1 - 0.5e12"
    assert algebra.format_multivector(values, name="x") == "x = 1 + 2e1 - 0.5e12"


def test_format_compact_multivector_requires_declared_grades():
    algebra = AlgebraContext(3, 0, 0, device="cpu", dtype=torch.float32)
    layout = algebra.layout((2,))
    values = torch.tensor([3.0, 0.0, -1.0])

    assert format_multivector(algebra, values, layout=layout) == "3e12 - e23"
    assert format_multivector(algebra, values, grades=(2,)) == "3e12 - e23"

    with pytest.raises(ValueError, match="requires layout or grades"):
        format_multivector(algebra, values)


def test_multivector_proxy_formats_batched_sample_only():
    algebra = AlgebraContext(2, 0, 0, device="cpu")
    values = torch.zeros(2, 3, algebra.dim)
    values[1, 2, 0] = 1.0
    values[1, 2, 1] = -1.0

    mv = Multivector(algebra, values, name="debug")

    assert mv.format(sample=(1, 2)) == "debug = shape=(2, 3, 4), sample[1,2] = 1 - e1"
    assert repr(mv).startswith("Multivector(")
    assert "sample[0,0]" in str(mv)
