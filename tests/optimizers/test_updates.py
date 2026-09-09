from copy import deepcopy
from functools import partial

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from torch import nn

import clifra.optimizers as optimizers
from clifra import AlgebraContext, TensorContract
from clifra.optimizers import (
    PostUpdateSGD,
    clip_coefficients_,
    exponential_update,
    normalize_coefficients_,
    post_update,
    project_to_rotor_tangent_space,
)
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_coefficient_adjustments_preserve_identity_storage_and_gradients(dtype):
    p = nn.Parameter(torch.tensor([[3.0, 4.0], [0.0, 0.0], [0.3, 0.4]], dtype=dtype))
    p.grad = torch.ones_like(p)
    gradient, pointer = p.grad, p.data_ptr()
    assert clip_coefficients_(p, 2.0) is p
    torch.testing.assert_close(p, torch.tensor([[1.2, 1.6], [0.0, 0.0], [0.3, 0.4]], dtype=dtype))
    assert normalize_coefficients_(p) is p
    torch.testing.assert_close(p, torch.tensor([[0.6, 0.8], [0.0, 0.0], [0.6, 0.8]], dtype=dtype))
    assert p.is_leaf and p.grad_fn is None and p.data_ptr() == pointer and p.grad is gradient
    p.sum().backward()
    torch.testing.assert_close(p.grad, torch.full_like(p, 2.0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_large_and_tiny_finite_coefficients(dtype):
    info = torch.finfo(dtype)
    p = torch.tensor([[info.max / 2, info.max / 2], [info.tiny, info.tiny]], dtype=dtype)
    normalized = normalize_coefficients_(p.clone())
    torch.testing.assert_close(normalized, torch.full_like(p, 2**-0.5))
    clipped = clip_coefficients_(p.clone(), 1.0)
    torch.testing.assert_close(clipped[0], normalized[0])
    assert torch.equal(clipped[1], p[1])


def test_explicit_axis_on_noncontiguous_coefficients():
    p = torch.tensor([[3.0, 4.0], [0.0, 5.0]]).T
    assert not p.is_contiguous()
    normalize_coefficients_(p, dim=0)
    torch.testing.assert_close(p, torch.tensor([[0.6, 0.0], [0.8, 1.0]]))
    clip_coefficients_(p, 0.5, dim=0)
    torch.testing.assert_close(p.norm(dim=0), torch.tensor([0.5, 0.5]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("helper", [normalize_coefficients_, partial(clip_coefficients_, max_norm=1.0)])
def test_nonfinite_coefficients_rejected_before_any_mutation(bad, helper):
    p = nn.Parameter(torch.tensor([[3.0, 4.0], [bad, 1.0]]))
    before = p.detach().clone()
    with pytest.raises(ValueError, match="coefficients must be finite"):
        helper(p)
    torch.testing.assert_close(p, before, equal_nan=True)


@pytest.mark.parametrize("bound", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_clip_bound_does_not_mutate(bound):
    p = torch.tensor([3.0, 4.0])
    with pytest.raises(ValueError, match="finite and positive"):
        clip_coefficients_(p, bound)
    assert torch.equal(p, torch.tensor([3.0, 4.0]))


@pytest.mark.parametrize("helper", [normalize_coefficients_, partial(clip_coefficients_, max_norm=1.0)])
def test_adjustments_require_real_coefficients_and_an_axis(helper):
    for dtype in (torch.int64, torch.complex64):
        with pytest.raises(TypeError, match="real floating"):
            helper(torch.ones(2, dtype=dtype))
    with pytest.raises(ValueError, match="coefficient axis"):
        helper(torch.tensor(1.0))


def test_post_update_selects_parameters_explicitly_and_runs_in_order_without_grads():
    selected, untouched = nn.Parameter(torch.tensor([3.0, 4.0])), nn.Parameter(torch.tensor([6.0, 8.0]))
    calls = []

    def first():
        assert not torch.is_grad_enabled()
        calls.append("first")
        selected.mul_(2.0)  # No decorator needed on a user callback.

    post_update(first, partial(clip_coefficients_, selected, 1.0), lambda: calls.append("last"))
    assert calls == ["first", "last"]
    assert selected.grad is None  # Explicit adjustments also run without gradients.
    torch.testing.assert_close(selected, torch.tensor([0.6, 0.8]))
    torch.testing.assert_close(untouched, torch.tensor([6.0, 8.0]))
    assert post_update() is None
    with pytest.raises(TypeError, match="callable"):
        post_update(first, None)
    assert calls == ["first", "last"]


def test_adam_integration_keeps_moments_and_explicitly_adjusts_only_selected_tensor():
    p, other = nn.Parameter(torch.tensor([3.0, 4.0])), nn.Parameter(torch.tensor([6.0, 8.0]))
    optimizer = torch.optim.Adam([p, other], lr=0.05)
    for _ in range(3):
        (p.square().sum() + other.square().sum()).backward()
        optimizer.step()
        state = deepcopy(optimizer.state_dict())
        other_after_step = other.detach().clone()
        post_update(partial(clip_coefficients_, p, 1.0))
        assert optimizer.param_groups[0]["params"][0] is p
        assert p.norm() <= 1.0 + 1e-6
        assert torch.equal(other, other_after_step)
        for key, value in state["state"][0].items():
            torch.testing.assert_close(optimizer.state[p][key], value)
        optimizer.zero_grad()
        assert p.grad is None


def test_adam_can_fit_bivector_coordinates_without_layers_or_tags():
    algebra = AlgebraContext(2, dtype=torch.float64)
    grade2_layout = algebra.layout((2,))
    p = nn.Parameter(torch.zeros(1, dtype=torch.float64))
    target = algebra.bivector_exp(torch.tensor([0.35], dtype=torch.float64), input=grade2_layout)
    optimizer = torch.optim.Adam([p], lr=0.05)
    initial_loss = (algebra.bivector_exp(p, input=grade2_layout) - target).square().sum().item()
    for _ in range(80):
        loss = (algebra.bivector_exp(p, input=grade2_layout) - target).square().sum()
        loss.backward()
        optimizer.step()
        post_update(partial(clip_coefficients_, p, 1.0))
        optimizer.zero_grad()
    assert loss.item() < initial_loss / 100
    assert not hasattr(p, "_manifold")


def test_sgd_closure_runs_before_single_adjustment_and_returns_loss():
    p = nn.Parameter(torch.tensor([3.0, 4.0]))
    events = []

    def adjustment():
        assert not torch.is_grad_enabled()
        events.append("adjustment")
        torch.testing.assert_close(p, torch.tensor([2.4, 3.2]))
        normalize_coefficients_(p)

    optimizer = PostUpdateSGD([p], lr=0.1, momentum=0.9, post_update=adjustment)

    def closure():
        assert torch.is_grad_enabled()
        events.append("closure")
        optimizer.zero_grad()
        loss = p.square().sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert loss.item() == 25.0 and events == ["closure", "adjustment"]
    torch.testing.assert_close(p, torch.tensor([0.6, 0.8]))
    torch.testing.assert_close(optimizer.state[p]["momentum_buffer"], torch.tensor([6.0, 8.0]))


def test_sgd_without_callback_matches_pytorch_and_ignores_parameter_metadata():
    p = nn.Parameter(torch.tensor([3.0, 4.0]))
    reference = nn.Parameter(p.detach().clone())
    p._manifold = "sphere"  # Arbitrary user metadata has no optimizer meaning.
    ours = PostUpdateSGD([p], lr=0.1, momentum=0.8, weight_decay=0.02, nesterov=True)
    ordinary = torch.optim.SGD([reference], lr=0.1, momentum=0.8, weight_decay=0.02, nesterov=True)
    for _ in range(3):
        p.grad = torch.ones_like(p)
        reference.grad = p.grad.clone()
        ours.step()
        ordinary.step()
        assert torch.equal(p, reference)
    assert not hasattr(PostUpdateSGD, "from_model")
    assert set(optimizers.__all__) == {
        "clip_coefficients_",
        "normalize_coefficients_",
        "post_update",
        "PostUpdateSGD",
        "project_to_rotor_tangent_space",
        "exponential_update",
    }
    for name in (
        "tag_manifold",
        "group_parameters_by_manifold",
        "make_riemannian_optimizer",
        "ExponentialSGD",
        "RiemannianAdam",
        "MANIFOLD_SPIN",
        "MANIFOLD_SPHERE",
        "MANIFOLD_EUCLIDEAN",
    ):
        assert not hasattr(optimizers, name)


def test_sgd_state_restore_keeps_new_explicit_callback():
    p = nn.Parameter(torch.tensor([3.0, 4.0]))
    optimizer = PostUpdateSGD([p], momentum=0.8, post_update=partial(clip_coefficients_, p, 1.0))
    p.grad = torch.ones_like(p)
    optimizer.step()
    q = nn.Parameter(p.detach().clone())
    resumed = PostUpdateSGD([q], post_update=partial(clip_coefficients_, q, 1.0))
    resumed.load_state_dict(deepcopy(optimizer.state_dict()))
    for _ in range(2):
        p.grad, q.grad = torch.ones_like(p), torch.ones_like(q)
        optimizer.step()
        resumed.step()
        assert torch.equal(p, q)
    assert set(optimizer.state_dict()) == {"state", "param_groups"}


def test_sgd_callback_runs_without_gradients_but_not_after_failed_closure():
    p = nn.Parameter(torch.tensor([3.0, 4.0]))
    calls = []
    optimizer = PostUpdateSGD([p], post_update=lambda: calls.append(1))
    optimizer.step()
    assert calls == [1]

    def failing():
        raise RuntimeError("closure failed")

    with pytest.raises(RuntimeError, match="closure failed"):
        optimizer.step(failing)
    assert calls == [1]
    with pytest.raises(ValueError, match="differentiable"):
        PostUpdateSGD([p], differentiable=True)
    with pytest.raises(TypeError, match="callable"):
        PostUpdateSGD([p], post_update="normalize")


@pytest.mark.parametrize("signature", [(3, 0, 0), (1, 2, 0), (2, 0, 1)])
def test_rotor_helpers_are_idempotent_and_preserve_unit_rotors(signature):
    a = AlgebraContext(*signature, dtype=torch.float64)
    grade2_layout = a.layout((2,))
    full_contract = TensorContract.canonical(a.layout())
    point = a.bivector_exp(
        torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64), input=grade2_layout, output=full_contract
    )
    ambient = torch.arange(a.dim, dtype=torch.float64) * 0.01
    tangent = project_to_rotor_tangent_space(point, ambient, a)
    torch.testing.assert_close(project_to_rotor_tangent_space(point, tangent, a), tangent)
    b = a.geometric_product(a.reverse(point), tangent, output=grade2_layout)
    torch.testing.assert_close(
        a.geometric_product(point, b, left=full_contract, right=grade2_layout, output=full_contract), tangent
    )
    updated = exponential_update(point, ambient, a)
    torch.testing.assert_close(
        updated, a.geometric_product(point, a.bivector_exp(b, input=grade2_layout, output=full_contract))
    )
    identity = torch.zeros_like(point)
    identity[0] = 1.0
    torch.testing.assert_close(a.geometric_product(a.reverse(updated), updated), identity)
    torch.testing.assert_close(exponential_update(point, torch.zeros_like(point), a), point)
    # The derivative at zero is the tangent map, including ambient input.
    _, derivative = torch.autograd.functional.jvp(
        lambda t: exponential_update(point, t, a), torch.zeros_like(point), ambient
    )
    torch.testing.assert_close(derivative, tangent)


@pytest.mark.parametrize("helper", [project_to_rotor_tangent_space, exponential_update])
def test_rotor_helpers_gradients_and_storage_assumptions(helper):
    a = AlgebraContext(3, dtype=torch.float64)
    b = torch.tensor([0.1, 0.2, -0.1], dtype=torch.float64, requires_grad=True)
    v = torch.linspace(-0.1, 0.1, a.dim, dtype=torch.float64, requires_grad=True)
    full_contract = TensorContract.canonical(a.layout())
    assert torch.autograd.gradcheck(
        lambda x, y: helper(a.bivector_exp(x, input=a.layout((2,)), output=full_contract), y, a), (b, v)
    )
    with pytest.raises(ValueError, match="same shape"):
        helper(v, v.unsqueeze(0), a)
    with pytest.raises(ValueError, match="canonical lanes"):
        helper(b, b, a)
    with pytest.raises(ValueError, match="same dtype"):
        helper(v, v.float(), a)
    with pytest.raises(ValueError, match="match the algebra"):
        helper(v.float(), v.float(), a)
    with pytest.raises(ValueError, match="match the algebra"):
        helper(v, v, AlgebraContext(3, device="meta", dtype=torch.float64))
    # Membership is a precondition, not a normalization service.
    invalid_point = torch.zeros(a.dim, dtype=torch.float64)
    invalid_point[0] = 2.0
    if helper is exponential_update:
        torch.testing.assert_close(helper(invalid_point, torch.zeros_like(v), a), invalid_point)


def test_exponential_update_has_no_implicit_sign_or_half_angle():
    a = AlgebraContext(2, dtype=torch.float64)
    rotor = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    tangent = torch.tensor([0.0, 0.0, 0.0, 0.4], dtype=torch.float64)
    expected = torch.tensor(
        [
            torch.cos(torch.tensor(0.4, dtype=torch.float64)),
            0.0,
            0.0,
            torch.sin(torch.tensor(0.4, dtype=torch.float64)),
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(exponential_update(rotor, tangent, a), expected)


@pytest.mark.parametrize(
    "device,dtype", [("cpu", torch.float32), ("cpu", torch.float64), ("mps", torch.float32), ("cuda", torch.float32)]
)
def test_device_dtype_and_gradients(device, dtype):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    a = AlgebraContext(3, device=device, dtype=dtype)
    p = nn.Parameter(torch.tensor([0.1, 0.2, 0.3], device=device, dtype=dtype))
    opt = PostUpdateSGD([p], post_update=partial(normalize_coefficients_, p))
    point = a.bivector_exp(p, input=a.layout((2,)), output=TensorContract.canonical(a.layout()))
    update = exponential_update(point, point * 0.01, a)
    assert update.device == p.device and update.dtype == dtype
    update.sum().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    opt.step()
    clip_coefficients_(p, 0.5)
    torch.testing.assert_close(p.norm(), torch.tensor(0.5, device=device, dtype=dtype))
