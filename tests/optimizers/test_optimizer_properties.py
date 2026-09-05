# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from hypothesis import given
from hypothesis import strategies as st

from clifra.core.algebra import AlgebraContext
from clifra.core.manifold import MANIFOLD_EUCLIDEAN, MANIFOLD_SPHERE, MANIFOLD_SPIN
from clifra.optimizers import ExponentialSGD, RiemannianAdam
from tests.helpers.hypothesis_cases import PROPERTY_SETTINGS, signature_strategy, tensor_with_shape

pytestmark = [pytest.mark.unit, pytest.mark.property]


@PROPERTY_SETTINGS
@given(data=st.data())
def test_exponential_sgd_matches_two_step_reference(data):
    shape = (data.draw(st.integers(1, 3)), data.draw(st.integers(1, 6)))
    initial = data.draw(tensor_with_shape(shape))
    gradients = [data.draw(tensor_with_shape(shape)) for _ in range(2)]
    lr = data.draw(st.floats(0.0, 0.5, allow_nan=False, allow_infinity=False))
    momentum = data.draw(st.floats(0.0, 0.99, allow_nan=False, allow_infinity=False))
    algebra = AlgebraContext(1, 0, 0, dtype=torch.float64)
    parameter = nn.Parameter(initial.clone())
    optimizer = ExponentialSGD(
        [{"params": [parameter], "manifold": MANIFOLD_EUCLIDEAN}],
        lr=lr,
        momentum=momentum,
        algebra=algebra,
    )
    expected = initial.clone()
    buffer = torch.zeros_like(initial)

    for gradient in gradients:
        parameter.grad = gradient.clone()
        optimizer.step()
        buffer = momentum * buffer + gradient
        expected = expected - lr * buffer
        assert torch.allclose(parameter, expected, atol=1e-12, rtol=1e-12)


@PROPERTY_SETTINGS
@given(data=st.data())
def test_riemannian_adam_matches_two_step_reference(data):
    shape = (data.draw(st.integers(1, 3)), data.draw(st.integers(1, 6)))
    initial = data.draw(tensor_with_shape(shape))
    gradients = [data.draw(tensor_with_shape(shape)) for _ in range(2)]
    lr = data.draw(st.floats(1e-5, 0.1, allow_nan=False, allow_infinity=False))
    beta1 = data.draw(st.floats(0.0, 0.95, allow_nan=False, allow_infinity=False))
    beta2 = data.draw(st.floats(0.0, 0.999, allow_nan=False, allow_infinity=False))
    eps = data.draw(st.floats(1e-12, 1e-4, allow_nan=False, allow_infinity=False))
    algebra = AlgebraContext(1, 0, 0, dtype=torch.float64)
    parameter = nn.Parameter(initial.clone())
    optimizer = RiemannianAdam(
        [{"params": [parameter], "manifold": MANIFOLD_EUCLIDEAN}],
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        algebra=algebra,
    )
    expected = initial.clone()
    first = torch.zeros_like(initial)
    second = torch.zeros_like(initial)

    for step, gradient in enumerate(gradients, 1):
        parameter.grad = gradient.clone()
        optimizer.step()
        first = beta1 * first + (1.0 - beta1) * gradient
        second = beta2 * second + (1.0 - beta2) * gradient.square()
        denominator = second.sqrt() / (1.0 - beta2**step) ** 0.5 + eps
        expected = expected - (lr / (1.0 - beta1**step)) * first / denominator
        assert torch.allclose(parameter, expected, atol=1e-11, rtol=1e-11)


@PROPERTY_SETTINGS
@given(optimizer_name=st.sampled_from(("sgd", "adam")), data=st.data())
def test_optimizer_state_machine_resume_matches_uninterrupted_updates(optimizer_name, data):
    gradients = data.draw(st.lists(tensor_with_shape((2, 3)), min_size=2, max_size=6))
    split = data.draw(st.integers(1, len(gradients) - 1))
    algebra = AlgebraContext(1, 0, 0, dtype=torch.float64)
    initial = data.draw(tensor_with_shape((2, 3)))

    def make(parameter):
        group = [{"params": [parameter], "manifold": MANIFOLD_EUCLIDEAN}]
        if optimizer_name == "sgd":
            return ExponentialSGD(group, lr=0.07, momentum=0.8, algebra=algebra)
        return RiemannianAdam(group, lr=0.03, betas=(0.7, 0.9), eps=1e-9, algebra=algebra)

    uninterrupted_parameter = nn.Parameter(initial.clone())
    resumed_parameter = nn.Parameter(initial.clone())
    uninterrupted = make(uninterrupted_parameter)
    resumed = make(resumed_parameter)
    for gradient in gradients[:split]:
        uninterrupted_parameter.grad = gradient.clone()
        resumed_parameter.grad = gradient.clone()
        uninterrupted.step()
        resumed.step()

    checkpoint = resumed.state_dict()
    loaded_parameter = nn.Parameter(resumed_parameter.detach().clone())
    loaded = make(loaded_parameter)
    loaded.load_state_dict(checkpoint)
    for gradient in gradients[split:]:
        uninterrupted_parameter.grad = gradient.clone()
        loaded_parameter.grad = gradient.clone()
        uninterrupted.step()
        loaded.step()

    assert torch.equal(loaded_parameter, uninterrupted_parameter)
    loaded_state = next(iter(loaded.state.values()))
    uninterrupted_state = next(iter(uninterrupted.state.values()))
    assert loaded_state.keys() == uninterrupted_state.keys()
    for key in loaded_state:
        if torch.is_tensor(loaded_state[key]):
            assert torch.equal(loaded_state[key], uninterrupted_state[key])
        else:
            assert loaded_state[key] == uninterrupted_state[key]


@pytest.mark.parametrize("optimizer_cls", (ExponentialSGD, RiemannianAdam))
@PROPERTY_SETTINGS
@given(signature=signature_strategy(min_n=1, max_n=4), data=st.data())
def test_manifold_retractions_satisfy_declared_norm_contracts(optimizer_cls, signature, data):
    algebra = AlgebraContext(*signature, dtype=torch.float64)
    initial = data.draw(tensor_with_shape((2, algebra.n)))
    initial[:, 0] += 3.0
    vector = nn.Parameter(initial)
    vector.grad = data.draw(tensor_with_shape((2, algebra.n)))
    optimizer = optimizer_cls(
        [{"params": [vector], "manifold": MANIFOLD_SPHERE}],
        lr=0.01,
        algebra=algebra,
    )
    optimizer.step()
    metric_norm = algebra.signature_norm_squared(vector, input=algebra.layout((1,))).abs()
    euclidean_norm = vector.norm(dim=-1, keepdim=True)

    assert torch.isfinite(vector).all()
    error = torch.minimum((metric_norm - 1.0).abs(), (euclidean_norm - 1.0).abs())
    assert torch.all(error < 1e-10)


@PROPERTY_SETTINGS
@given(data=st.data())
def test_spin_retraction_clips_arbitrary_parameter_shapes(data):
    shape = (data.draw(st.integers(1, 4)), data.draw(st.integers(1, 8)))
    bound = data.draw(st.floats(0.01, 3.0, allow_nan=False, allow_infinity=False))
    parameter = nn.Parameter(4.0 * data.draw(tensor_with_shape(shape)))
    parameter.grad = data.draw(tensor_with_shape(shape))
    optimizer = ExponentialSGD(
        [{"params": [parameter], "manifold": MANIFOLD_SPIN}],
        lr=0.1,
        algebra=AlgebraContext(1, 0, 0, dtype=torch.float64),
        max_bivector_norm=bound,
    )
    optimizer.step()
    assert torch.all(parameter.norm(dim=-1) <= bound * (1.0 + 1e-12))
