# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Explicit adjustments for ordinary PyTorch optimization loops."""

import math
from collections.abc import Callable

import torch

from clifra.core.tensors import TensorContract


def _scaled_coefficients(values: torch.Tensor, dim: int):
    if not values.is_floating_point():
        raise TypeError("coefficients must be real floating-point tensors")
    if values.ndim == 0:
        raise ValueError("coefficients require a coefficient axis")
    if not torch.isfinite(values).all():
        raise ValueError("coefficients must be finite")
    # Scale before taking the norm so finite large coefficients do not overflow.
    work = values.float() if values.dtype in (torch.float16, torch.bfloat16) else values
    scale = work.abs().amax(dim=dim, keepdim=True)
    scaled = work / torch.where(scale == 0, 1, scale)
    length = torch.linalg.vector_norm(scaled, dim=dim, keepdim=True)
    return scaled, scale, torch.where(length == 0, 1, length)


@torch.no_grad()
def clip_coefficients_(values: torch.Tensor, max_norm: float, *, dim: int = -1) -> torch.Tensor:
    """Bound Euclidean coefficient length along dim, preserving tensor identity.

    Values must be finite real coefficients. max_norm must be finite and
    positive. No algebra, grade, or parameter meaning is inferred. Zero vectors
    and vectors within the bound are unchanged; optimizer state is untouched.
    """
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be finite and positive")
    scaled, scale, length = _scaled_coefficients(values, dim)
    values.copy_(torch.where(scale <= max_norm / length, values, (scaled / length) * max_norm))
    return values


@torch.no_grad()
def normalize_coefficients_(values: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Normalize finite real coefficient vectors in place along dim.

    Every nonzero vector receives unit Euclidean coefficient length; zero
    vectors remain zero. This does not enforce a Clifford signature norm or
    rotor membership. Tensor identity and optimizer state are preserved.
    """
    scaled, _, length = _scaled_coefficients(values, dim)
    values.copy_(scaled / length)
    return values


@torch.no_grad()
def post_update(*adjustments: Callable[[], object]) -> None:
    """Run explicit zero-argument adjustments in order, without recording grads.

    Call after optimizer.step(), outside its closure. Callbacks select their
    own tensors and should mutate them in place. No parameters are discovered
    or filtered by gradient presence, and optimizer moments are not transported.
    A callback exception propagates; earlier adjustments are not rolled back.
    """
    if any(not callable(adjustment) for adjustment in adjustments):
        raise TypeError("adjustments must be callable")
    for adjustment in adjustments:
        adjustment()


class PostUpdateSGD(torch.optim.SGD):
    """PyTorch SGD followed by one optional zero-argument adjustment callback.

    The callback runs once under no_grad after each successful step, outside
    the loss closure, including steps with no gradients. Select parameters
    explicitly in that callback. Other options are passed to torch.optim.SGD;
    differentiable=True is unsupported because adjustments are not recorded.

    Momentum is unchanged by adjustments. The callback is not optimizer state:
    supply it again when constructing an optimizer to load a state_dict.
    """

    def __init__(self, params, lr=0.01, *, post_update: Callable[[], object] | None = None, **kwargs):
        if post_update is not None and not callable(post_update):
            raise TypeError("post_update must be callable or None")
        if kwargs.get("differentiable", False):
            raise ValueError("PostUpdateSGD does not support differentiable=True")
        super().__init__(params, lr=lr, **kwargs)
        self._post_update = post_update

    def step(self, closure=None):
        loss = super().step(closure)
        if self._post_update is not None:
            post_update(self._post_update)
        return loss


def project_to_rotor_tangent_space(rotor: torch.Tensor, vector: torch.Tensor, algebra) -> torch.Tensor:
    """Return R <reverse(R) V>_2 for a unit rotor R and ambient coefficients V.

    Both inputs must have identical shape and canonical full-lane storage,
    with dtype and device matching the algebra.
    The caller must supply a rotor with reverse(R) R = 1; membership is not
    checked or repaired. Under that assumption this is an idempotent map onto
    tangents R B. It is not claimed to be a Euclidean orthogonal projection in
    mixed or degenerate signatures. The operation is differentiable.
    """
    bivector, full_contract, grade2_layout = _left_trivialized_bivector(rotor, vector, algebra)
    return algebra.geometric_product(rotor, bivector, left=full_contract, right=grade2_layout, output=full_contract)


def exponential_update(rotor: torch.Tensor, tangent_vector: torch.Tensor, algebra) -> torch.Tensor:
    """Return R exp(<reverse(R) T>_2), with the step size already included in T.

    Inputs have identical shape and canonical full-lane storage, with dtype
    and device matching the algebra. R must already be a unit rotor; this
    function does not normalize arbitrary coefficients.
    For T = R B the update is R exp(B); ambient T is first mapped to a tangent.
    There is no implicit minus sign or half-angle factor: callers supply the
    complete signed and scaled update, including any parameterization convention.
    This differentiable group update is not a general Riemannian exponential
    for a chosen metric, and does not transport optimizer moments. Numerical
    support and rounding follow the core Clifford product and bivector_exp.
    """
    bivector, full_contract, grade2_layout = _left_trivialized_bivector(rotor, tangent_vector, algebra)
    update = algebra.bivector_exp(bivector, input=grade2_layout, output=full_contract)
    return algebra.geometric_product(rotor, update, left=full_contract, right=full_contract, output=full_contract)


def _left_trivialized_bivector(point, other, algebra):
    if point.shape != other.shape:
        raise ValueError("point and update must have the same shape")
    if point.ndim < 1 or point.shape[-1] != algebra.dim:
        raise ValueError(f"point and update must use {algebra.dim} canonical lanes")
    if point.dtype != other.dtype or point.device != other.device:
        raise ValueError("point and update must have the same dtype and device")
    device = algebra.device
    if point.dtype != algebra.dtype or point.device.type != device.type:
        raise ValueError("rotor and update dtype and device must match the algebra")
    if device.type != "cpu" and device.index is not None and point.device.index != device.index:
        raise ValueError("rotor and update device must match the algebra")
    full_contract = TensorContract.canonical(algebra.layout())
    grade2_layout = algebra.layout((2,))
    reverse = algebra.reverse(point, input=full_contract, output=full_contract)
    bivector = algebra.geometric_product(reverse, other, left=full_contract, right=full_contract, output=grade2_layout)
    return bivector, full_contract, grade2_layout
