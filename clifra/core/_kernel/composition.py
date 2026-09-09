"""Static compositions of existing tensor kernels for geometric operations."""

from torch import nn

from .basis import expand_output_grades
from .numerics import require_nonzero, signed_clamp_min


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
        if kind in {"reflect", "versor"}:
            if kind == "reflect" and blade.grades != (1,):
                raise ValueError("reflection normal must have grade-one layout")
            self.involution = algebra.plan_unary(op="grade_involution", input=blade, output=blade)
            if kind == "reflect":
                self.involution = algebra.plan_unary(op="grade_involution", input=values, output=values)
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
        if self.kind == "versor":
            middle = self.first(self.involution(blade), x)
        elif self.kind == "reflect":
            middle = self.first(blade, self.involution(x))
        else:
            middle = self.first(x, blade)
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
