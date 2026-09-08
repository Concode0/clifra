import torch
from hypothesis import given
from hypothesis import strategies as st

from clifra.optimizers import clip_coefficients_, normalize_coefficients_
from tests.helpers.hypothesis_cases import PROPERTY_SETTINGS, tensor_with_shape


@PROPERTY_SETTINGS
@given(data=st.data())
def test_coefficient_norm_contracts(data):
    shape = (data.draw(st.integers(1, 4)), data.draw(st.integers(1, 8)))
    bound = data.draw(st.floats(0.01, 3.0, allow_nan=False, allow_infinity=False))
    values = data.draw(tensor_with_shape(shape))
    original = values.clone()
    clipped = clip_coefficients_(values, bound)
    assert torch.all(clipped.norm(dim=-1) <= bound * (1.0 + 1e-12))
    inside = original.norm(dim=-1) <= bound
    assert torch.equal(clipped[inside], original[inside])
    nonzero = clipped.abs().amax(dim=-1) > 0
    normalize_coefficients_(clipped)
    torch.testing.assert_close(clipped.norm(dim=-1), nonzero.to(clipped.dtype))
