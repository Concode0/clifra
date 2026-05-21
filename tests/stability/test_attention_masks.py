import pytest
import torch

from clifra.core.runtime.algebra import CliffordAlgebra
from clifra.core.runtime.context import AlgebraContext
from clifra.layers.blocks.attention import GeometricProductAttention

pytestmark = pytest.mark.unit


def test_attention_fully_masked_rows_are_finite_zero_outputs():
    algebra = CliffordAlgebra(3, 0, 0, device="cpu", dtype=torch.float32)
    attention = GeometricProductAttention(algebra, channels=4, num_heads=2, causal=False)
    x = torch.randn(2, 4, 4, algebra.dim)
    key_padding_mask = torch.ones(2, 4, dtype=torch.bool)

    output = attention(x, key_padding_mask=key_padding_mask)

    assert torch.isfinite(output).all()
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-6, rtol=0.0)


def test_compact_attention_fully_masked_rows_are_finite_zero_outputs():
    algebra = AlgebraContext(5, 0, device="cpu", dtype=torch.float32, default_grades=(1,))
    layout = algebra.layout((1,))
    attention = GeometricProductAttention(algebra, channels=4, num_heads=2, causal=False)
    x = torch.randn(2, 4, 4, layout.dim)
    key_padding_mask = torch.ones(2, 4, dtype=torch.bool)

    output = attention(x, key_padding_mask=key_padding_mask)

    assert torch.isfinite(output).all()
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-6, rtol=0.0)
