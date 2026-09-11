"""Reduced numerical issue exposed by v2 qualification."""

import torch

from clifra.core import AlgebraContext


def test_vector_action_float32_small_generator_accuracy():
    a = AlgebraContext(3, dtype=torch.float32)
    vector, bivector = a.layout((1,)), a.layout((2,))
    operation = a.plan_versor_action(grade=2, input=vector, parameter=bivector, output=vector)
    x = torch.tensor([2.8830723762512207, -0.5116939544677734, -1.1327883005142212])
    b = torch.tensor([-0.07566938549280167, 0.38816285133361816, 0.036167751997709274])
    reference_algebra = AlgebraContext(3, dtype=torch.float64)
    expected = reference_algebra.versor_action(
        x.double(), b.double(), grade=2, input=reference_algebra.layout((1,)), parameter=reference_algebra.layout((2,))
    )
    torch.testing.assert_close(operation(x, b).double(), expected, rtol=1e-4, atol=1e-5)
