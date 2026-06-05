import pytest
import torch

from clifra.core.runtime.algebra import AlgebraContext
from clifra.layers.blocks.attention import EntropyGatedAttention
from clifra.layers.primitives.activation import GeometricGELU, GradeSwish

pytestmark = pytest.mark.unit


def test_geometric_gelu_zero_input_stays_finite():
    algebra = AlgebraContext(3, 0, device="cpu", dtype=torch.float32)
    activation = GeometricGELU(algebra, channels=2)
    x = torch.zeros(4, 2, algebra.dim)

    y = activation(x)

    assert torch.isfinite(y).all()
    assert torch.allclose(y, torch.zeros_like(y))


def test_grade_swish_zero_input_stays_finite():
    algebra = AlgebraContext(3, 0, device="cpu", dtype=torch.float32)
    activation = GradeSwish(algebra, channels=2)
    x = torch.zeros(4, 2, algebra.dim)

    y = activation(x)

    assert torch.isfinite(y).all()
    assert torch.allclose(y, torch.zeros_like(y))


def test_entropy_gated_attention_all_masked_entropy_is_finite():
    algebra = AlgebraContext(3, 0, device="cpu", dtype=torch.float32)
    attention = EntropyGatedAttention(algebra, channels=4, num_heads=2)
    x = torch.randn(2, 4, 4, algebra.dim)
    key_padding_mask = torch.ones(2, 4, dtype=torch.bool)

    output, entropy, gate = attention(x, key_padding_mask=key_padding_mask, return_gating=True)

    assert torch.isfinite(output).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(gate).all()
    assert torch.allclose(entropy, torch.zeros_like(entropy))


def test_entropy_gated_attention_accepts_compact_grade1_context():
    algebra = AlgebraContext(5, 0, device="cpu", dtype=torch.float32, default_grades=(1,))
    layout = algebra.layout((1,))
    attention = EntropyGatedAttention(algebra, channels=4, num_heads=2)
    x = torch.randn(2, 4, 4, layout.dim)

    output, entropy, gate = attention(x, return_gating=True)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(gate).all()
    assert torch.allclose(entropy, torch.zeros_like(entropy))
