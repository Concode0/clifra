"""Static compositions of existing tensor kernels for geometric operations."""

import torch
from torch import nn

from clifra.core.layout import GradeLayout

from .basis import expand_output_grades
from .numerics import eps_like, require_nonzero, signed_clamp_min


class Geometry(nn.Module):
    def __init__(self, algebra, kind, inputs, output):
        super().__init__()
        self.kind = kind
        self.eps_sq = algebra.eps_sq
        if kind == "sandwich":
            left, values, right = inputs
            middle = algebra.layout(expand_output_grades(left.grades, values.grades, algebra.n, op="geometric_product"))
            self.first = algebra.plan_product(left=left, right=values, output=middle)
            self.second = algebra.plan_product(left=middle, right=right, output=output)
            return
        blade = inputs[0] if kind == "blade_inverse" else inputs[1]
        self.reverse = algebra.plan_unary(op="reverse", input=blade, output=blade)
        self.norm = algebra.plan_signature_norm_squared(input=blade)
        if kind == "blade_inverse":
            self.project = algebra.plan_unary(op="grade_projection", input=blade, output=output)
            return
        values = inputs[0]
        if kind == "versor":
            self.involution = algebra.plan_unary(op="grade_involution", input=blade, output=blade)
            middle = algebra.layout(
                expand_output_grades(blade.grades, values.grades, algebra.n, op="geometric_product")
            )
            self.first = algebra.plan_product(left=blade, right=values, output=middle)
        else:
            if kind == "blade_reject" and output != values:
                raise ValueError("blade rejection output must match input layout")
            middle = algebra.layout(expand_output_grades(values.grades, blade.grades, algebra.n, op="left_contraction"))
            self.first = algebra.plan_product(op="left_contraction", left=values, right=blade, output=middle)
        self.second = algebra.plan_product(left=middle, right=blade, output=output)

    def forward(self, *values):
        if self.kind == "sandwich":
            left, x, right = values
            return self.second(self.first(left, x), right)
        blade = values[0] if self.kind == "blade_inverse" else values[1]
        inverse = self._inverse(blade)
        if self.kind == "blade_inverse":
            return self.project(inverse)
        x = values[0]
        middle = self.first(self.involution(blade), x) if self.kind == "versor" else self.first(x, blade)
        result = self.second(middle, inverse)
        return x - result if self.kind == "blade_reject" else result

    def _inverse(self, blade):
        denominator = signed_clamp_min(self.norm(blade), self.eps_sq)
        return self.reverse(blade) / denominator


class StrictGeometry(Geometry):
    """Geometry composition using the exact mathematical inverse denominator."""

    def _inverse(self, blade):
        denominator = self.norm(blade)
        require_nonzero(denominator, name=self.kind.replace("_", " "))
        return self.reverse(blade) / denominator


def full_versor_factors(
    algebra,
    weights: torch.Tensor,
    *,
    grade: int,
    parameter_layout: GradeLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full-lane left/right factors for a grade-1 or grade-2 versor action."""
    grade = int(grade)
    if grade == 2:
        rotor_layout = parameter_layout.spec.layout(range(0, parameter_layout.spec.n + 1, 2))
        rotor = _bivector_exp(
            algebra,
            -0.5 * weights,
            parameter_layout=parameter_layout,
            rotor_layout=rotor_layout,
        )
        right = algebra.reverse(rotor, input=rotor_layout, output=rotor_layout)
        return rotor_layout.full(rotor), rotor_layout.full(right)

    if grade == 1:
        signature_norm_squared = algebra.signature_norm_squared(weights, input=parameter_layout)
        scale = signature_norm_squared.abs().clamp_min(eps_like(signature_norm_squared)).sqrt()
        versor = weights / scale
    else:
        norm = weights.norm(dim=-1, keepdim=True).clamp_min(eps_like(weights))
        versor = weights / norm

    left = algebra.grade_involution(versor, input=parameter_layout, output=parameter_layout)
    right = algebra.blade_inverse(versor, input=parameter_layout)
    return parameter_layout.full(left), parameter_layout.full(right)


def _bivector_exp(
    algebra,
    values: torch.Tensor,
    *,
    parameter_layout: GradeLayout,
    rotor_layout: GradeLayout,
) -> torch.Tensor:
    return algebra.bivector_exp(values, input=parameter_layout, output=rotor_layout)
