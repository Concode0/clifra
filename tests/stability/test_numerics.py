import pytest
import torch

from clifra.core.foundation.numerics import covariance_regularizer, eps_for, signed_clamp_min
from clifra.core.runtime.algebra import AlgebraContext
from clifra.layers.primitives.projection import GeometricNeutralizer

pytestmark = pytest.mark.unit


def test_signed_clamp_preserves_negative_denominator_sign():
    values = torch.tensor([[-1.0e-30, 0.0, 1.0e-30]], dtype=torch.float64)

    clamped = signed_clamp_min(values, 1.0e-12)

    assert clamped[0, 0] < 0
    assert clamped[0, 1] > 0
    assert clamped[0, 2] > 0
    assert clamped.abs().min() >= 1.0e-12


def test_covariance_regularizer_is_dtype_aware_and_scale_aware():
    cov32 = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    cov64 = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    large_cov = 10.0 * cov32

    reg32 = covariance_regularizer(cov32)
    reg64 = covariance_regularizer(cov64)
    large_reg = covariance_regularizer(large_cov)

    assert torch.allclose(reg32, torch.full_like(reg32, eps_for(torch.float32, multiplier=32.0)))
    assert torch.allclose(reg64, torch.full_like(reg64, eps_for(torch.float64, multiplier=32.0)))
    assert torch.allclose(large_reg, 10.0 * reg32)


def test_geometric_neutralizer_zero_covariance_stays_finite():
    algebra = AlgebraContext(3, 0, 0, device="cpu", dtype=torch.float32)
    neutralizer = GeometricNeutralizer(algebra, channels=2)
    x = torch.zeros(4, 2, algebra.dim)

    output = neutralizer(x)

    assert torch.isfinite(output).all()
