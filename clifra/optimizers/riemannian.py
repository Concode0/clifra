# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""SGD and Adam variants with post-update handling for tagged parameters.

The ``from_model()`` constructors recognize three simplified tags via
``p._manifold``:
    - ``'spin'``: bivector coordinates, optionally clipped by coefficient norm
    - ``'sphere'``: vectors normalized after each update
    - ``'euclidean'`` (or untagged): Standard unconstrained parameters

They convert those tags into optimizer parameter groups. Parameters without a
tag are treated as Euclidean. Parameters passed directly to an optimizer are
also Euclidean unless their parameter group declares ``manifold``. The labels
select optimizer behavior while layers define geometric meaning.

Background references:
    - Absil et al. "Optimization Algorithms on Matrix Manifolds" (2008)
    - Boumal "An Introduction to Optimization on Smooth Manifolds" (2023)
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

from clifra.core.foundation.manifold import (
    MANIFOLD_EUCLIDEAN,
    MANIFOLD_ORDER,
    MANIFOLD_SPHERE,
    MANIFOLD_SPIN,
    tag_manifold,
    validate_manifold,
)
from clifra.core.foundation.numerics import eps_like

__all__ = [
    "ExponentialSGD",
    "RiemannianAdam",
    "project_to_tangent_space",
    "exponential_retraction",
    "tag_manifold",
    "group_parameters_by_manifold",
    "make_riemannian_optimizer",
    "MANIFOLD_SPIN",
    "MANIFOLD_SPHERE",
    "MANIFOLD_EUCLIDEAN",
]


def group_parameters_by_manifold(
    model: nn.Module,
) -> Dict[str, List[nn.Parameter]]:
    """Group a model's parameters by their ``_manifold`` tag.

    Parameters without a ``_manifold`` attribute are placed in the
    ``'euclidean'`` group.

    Args:
        model: The model whose parameters to group.

    Returns:
        Dict mapping manifold name to list of parameters.
    """
    groups: Dict[str, List[nn.Parameter]] = {
        MANIFOLD_SPIN: [],
        MANIFOLD_SPHERE: [],
        MANIFOLD_EUCLIDEAN: [],
    }
    for p in model.parameters():
        manifold = validate_manifold(getattr(p, "_manifold", MANIFOLD_EUCLIDEAN))
        groups[manifold].append(p)
    return groups


def _parameter_groups_for_model(model: nn.Module) -> list[dict]:
    grouped = group_parameters_by_manifold(model)
    param_groups = []
    for manifold in MANIFOLD_ORDER:
        params = grouped[manifold]
        if params:
            param_groups.append({"params": params, "manifold": manifold})
    if not param_groups:
        raise ValueError("Model has no parameters")
    return param_groups


def make_riemannian_optimizer(
    model: nn.Module,
    algebra,
    *,
    optimizer: str = "adam",
    **kwargs,
) -> Optimizer:
    """Create a tag-aware optimizer from a model.

    Args:
        model: Model whose parameters may be tagged with ``_manifold``.
        algebra: Clifford algebra instance used by the optimizer.
        optimizer: ``"adam"``/``"riemannian_adam"`` or
            ``"sgd"``/``"exponential_sgd"``.
        **kwargs: Optimizer-specific keyword arguments.

    Returns:
        ``RiemannianAdam`` or ``ExponentialSGD`` with per-manifold groups.
    """
    key = optimizer.lower().replace("-", "_")
    if key in {"adam", "riemannian_adam"}:
        return RiemannianAdam.from_model(model, algebra=algebra, **kwargs)
    if key in {"sgd", "exponential_sgd"}:
        return ExponentialSGD.from_model(model, algebra=algebra, **kwargs)
    raise ValueError("optimizer must be one of 'adam', 'riemannian_adam', 'sgd', or 'exponential_sgd'")


def _layout_for_parameter(algebra, values: torch.Tensor, grade: int):
    try:
        layout = algebra.layout((int(grade),))
    except (AttributeError, ValueError):
        return None
    return layout if values.shape[-1] == layout.dim else None


def _metric_signature_norm_squared_or_none(algebra, values: torch.Tensor, grade: int) -> torch.Tensor | None:
    layout = _layout_for_parameter(algebra, values, grade)
    if layout is None:
        return None
    return algebra.signature_norm_squared(values, input_layout=layout)


def _euclidean_norm(values: torch.Tensor) -> torch.Tensor:
    floor = eps_like(values, min_value=torch.finfo(values.dtype).tiny)
    return values.norm(dim=-1, keepdim=True).clamp_min(floor)


def _sphere_retract_(values: torch.Tensor, algebra) -> None:
    metric_signature_norm_squared = _metric_signature_norm_squared_or_none(algebra, values, 1)
    if metric_signature_norm_squared is None:
        values.div_(_euclidean_norm(values))
        return

    floor = eps_like(
        metric_signature_norm_squared,
        multiplier=32.0,
        min_value=torch.finfo(metric_signature_norm_squared.dtype).tiny,
    )
    metric_scale = metric_signature_norm_squared.abs().clamp_min(floor).sqrt()
    euclidean_scale = _euclidean_norm(values)
    scale = torch.where(metric_signature_norm_squared.abs() > floor, metric_scale, euclidean_scale)
    values.div_(scale)


def _clip_bivector_coefficients_(values: torch.Tensor, max_norm: float | None) -> None:
    if max_norm is None:
        return
    norm = _euclidean_norm(values)
    values.div_(torch.clamp(norm / float(max_norm), min=1.0))


def _retract_parameter_(values: torch.Tensor, *, manifold: str, algebra, max_bivector_norm: float | None) -> None:
    if manifold == MANIFOLD_SPHERE:
        _sphere_retract_(values, algebra)
    elif manifold == MANIFOLD_EUCLIDEAN:
        return
    elif manifold == MANIFOLD_SPIN:
        _clip_bivector_coefficients_(values, max_bivector_norm)


class ExponentialSGD(Optimizer):
    """SGD followed by tag-specific post-update handling.

    The optimizer applies an ordinary SGD update to each stored parameter.
    According to each parameter group's ``manifold`` value, it then clips
    ``spin`` parameters by coefficient norm, normalizes
    ``sphere`` parameters, and leaves ``euclidean`` parameters unchanged.
    For layers parameterized by bivectors, the layer—not this optimizer—uses
    the updated coordinates to construct a rotor during its forward pass.

    Args:
        params (Iterable): Iterable of parameters to optimize
        lr: Learning rate
        momentum: Momentum factor (default: 0)
        algebra: Algebra context used for signature-aware sphere normalization.
        max_bivector_norm: Maximum Euclidean coefficient norm for ``spin``
            parameter groups. ``None`` disables clipping. Defaults to ``10.0``.

    Example:
        >>> algebra = AlgebraContext(p=3, q=0, device='cpu')
        >>> model = VersorLayer(algebra, channels=4)
        >>> optimizer = ExponentialSGD.from_model(
        ...     model, lr=0.01, algebra=algebra
        ... )
    """

    def __init__(
        self, params, lr: float = 0.01, momentum: float = 0, algebra=None, max_bivector_norm: Optional[float] = 10.0
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if algebra is None:
            raise ValueError("Must provide Layout-first algebra context")
        if max_bivector_norm is not None and max_bivector_norm <= 0.0:
            raise ValueError(f"Invalid max_bivector_norm: {max_bivector_norm}")

        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        self.algebra = algebra
        self.max_bivector_norm = max_bivector_norm

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0,
        algebra=None,
        max_bivector_norm: Optional[float] = 10.0,
    ):
        """Create optimizer with auto-detected manifold parameter groups.

        Inspects ``p._manifold`` tags on each parameter and creates separate
        groups for spin, sphere, and euclidean parameters so that each group
        receives the correct retraction in :meth:`step`.

        Args:
            model: The model to optimize.
            lr: Learning rate.
            momentum: Momentum factor.
            algebra: Layout-first algebra context (required).
            max_bivector_norm: Clip threshold for spin params.

        Returns:
            ExponentialSGD instance with per-manifold parameter groups.
        """
        param_groups = _parameter_groups_for_model(model)
        return cls(param_groups, lr=lr, momentum=momentum, algebra=algebra, max_bivector_norm=max_bivector_norm)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[torch.Tensor]:
        """Perform one SGD step and dispatch post-update handling by tag.

        Args:
            closure (Callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            Optional[torch.Tensor]: The loss if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            manifold = validate_manifold(group.get("manifold", MANIFOLD_EUCLIDEAN))

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply momentum (if enabled)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(grad)
                    else:
                        buf = param_state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    grad = buf

                # Update parameters
                p.add_(grad, alpha=-lr)

                _retract_parameter_(
                    p,
                    manifold=manifold,
                    algebra=self.algebra,
                    max_bivector_norm=self.max_bivector_norm,
                )

        return loss


class RiemannianAdam(Optimizer):
    """Adam followed by tag-specific post-update handling.

    Adam moments and updates are computed in the stored parameter coordinates.
    According to each parameter group's ``manifold`` value, the optimizer then
    clips ``spin`` parameters by coefficient norm,
    normalizes ``sphere`` parameters, and leaves ``euclidean`` parameters
    unchanged. A layer may subsequently exponentiate updated bivector
    coordinates in its forward pass; that exponential is not part of
    :meth:`step`.

    Args:
        params (Iterable): Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added for numerical stability (default: 1e-8)
        algebra: Algebra context used for signature-aware sphere normalization.
        max_bivector_norm: Maximum Euclidean coefficient norm for ``spin``
            parameter groups. ``None`` disables clipping. Defaults to ``10.0``.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        algebra=None,
        max_bivector_norm: Optional[float] = 10.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if algebra is None:
            raise ValueError("Must provide Layout-first algebra context")
        if max_bivector_norm is not None and max_bivector_norm <= 0.0:
            raise ValueError(f"Invalid max_bivector_norm: {max_bivector_norm}")

        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.algebra = algebra
        self.max_bivector_norm = max_bivector_norm

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        algebra=None,
        max_bivector_norm: Optional[float] = 10.0,
    ):
        """Create optimizer with auto-detected manifold parameter groups.

        Inspects ``p._manifold`` tags on each parameter and creates separate
        groups for spin, sphere, and euclidean parameters so that each group
        receives the correct retraction in :meth:`step`.

        Args:
            model: The model to optimize.
            lr: Learning rate.
            betas: Coefficients for running averages.
            eps: Numerical stability term.
            algebra: Layout-first algebra context (required).
            max_bivector_norm: Clip threshold for spin params.

        Returns:
            RiemannianAdam instance with per-manifold parameter groups.
        """
        param_groups = _parameter_groups_for_model(model)
        return cls(param_groups, lr=lr, betas=betas, eps=eps, algebra=algebra, max_bivector_norm=max_bivector_norm)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[torch.Tensor]:
        """Performs a single optimization step.

        Applies Adam momentum updates to all parameters, then dispatches
        per-manifold retraction:

        - **spin**: bivector norm clipping
        - **sphere**: L2 normalization to unit sphere
        - **euclidean**: no retraction

        Args:
            closure (Callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            Optional[torch.Tensor]: The loss if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            manifold = validate_manifold(group.get("manifold", MANIFOLD_EUCLIDEAN))

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Compute step size
                step_size = lr / bias_correction1
                bias_correction2_sqrt = bias_correction2**0.5

                # Adam update in parameter space
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

                _retract_parameter_(
                    p,
                    manifold=manifold,
                    algebra=self.algebra,
                    max_bivector_norm=self.max_bivector_norm,
                )

        return loss


def project_to_tangent_space(point, vector, algebra):
    """Project a vector to the tangent space at a point on Spin(n).

    For rotors R in Spin(n), the tangent space at R is:
        T_R Spin(n) = { R . B | B is a bivector }

    Args:
        point: Current point on manifold (rotor) [..., dim]
        vector: Vector to project [..., dim]
        algebra: Layout-first algebra context

    Returns:
        Projected vector in tangent space [..., dim]
    """
    # Compute ~R . vector
    R_rev = algebra.reverse(point)
    tangent = algebra.geometric_product(R_rev, vector)

    # Project to bivector part (Lie algebra)
    # This extracts only the grade-2 (bivector) components
    bivector = algebra.grade_projection(tangent, grade=2)

    # Map back to tangent space: R . bivector
    return algebra.geometric_product(point, bivector)


def exponential_retraction(point, tangent_vector, algebra):
    """Exponential map: move from point along tangent vector on manifold.

    For Spin(n), the exponential map is:
        Exp_R(R.B) = R . exp(B)

    where B is a bivector in the Lie algebra.

    Args:
        point: Current point on manifold (rotor) [..., dim]
        tangent_vector: Tangent vector (direction to move) [..., dim]
        algebra: Layout-first algebra context

    Returns:
        New point on manifold [..., dim]
    """
    # Extract bivector from tangent vector
    R_rev = algebra.reverse(point)
    bivector = algebra.geometric_product(R_rev, tangent_vector)
    bivector = algebra.grade_projection(bivector, grade=2)

    # Exponential map
    update = algebra.bivector_exp(bivector)

    # Apply update: R_new = R_old . exp(B)
    return algebra.geometric_product(point, update)
