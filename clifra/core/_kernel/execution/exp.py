# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Materialized Clifford exponentials using closed formulas and planned products."""

from __future__ import annotations

import torch
from torch import nn

from clifra.core._kernel.planning.exp import BivectorExpPlan, taylor_degree


class TaylorPolynomial(nn.Module):
    """Horner evaluation with statically pruned intermediate grade contracts."""

    def __init__(self, products, output_layouts, *, dtype, device):
        super().__init__()
        self.products = nn.ModuleList(products)
        self.output_layouts = tuple(output_layouts)
        self.degree = len(products)
        for step, layout in enumerate(output_layouts):
            mask = torch.tensor([float(i == 0) for i in layout.basis_indices], dtype=dtype, device=device)
            self.register_buffer(f"scalar_{step}", mask, persistent=False)

    def set_degree(self, degree, full_product, full_layout):
        """Change precision by inserting/removing repeated full-even stages.

        All supported dimensions reach the full even layout before the middle
        six stages that distinguish degrees 12 and 18. The pruned suffix is
        unchanged, so movement needs no new product planning or provider lookup.
        """
        if degree == self.degree:
            return
        pivot = full_layout.spec.n // 2
        products, layouts = list(self.products), list(self.output_layouts)
        if degree > self.degree:
            count = degree - self.degree
            products[pivot:pivot] = [full_product] * count
            layouts[pivot:pivot] = [full_layout] * count
        else:
            del products[pivot : pivot + self.degree - degree]
            del layouts[pivot : pivot + self.degree - degree]
        prototype = self.scalar_0
        for step in range(self.degree):
            delattr(self, f"scalar_{step}")
        self.products = nn.ModuleList(products)
        self.output_layouts = tuple(layouts)
        self.degree = degree
        for step, layout in enumerate(layouts):
            self.register_buffer(
                f"scalar_{step}", prototype.new_tensor([float(i == 0) for i in layout.basis_indices]), persistent=False
            )

    def forward(self, values):
        result = torch.ones_like(values[..., :1])
        for step, product in enumerate(self.products):
            result = product.forward_compact(values, result) / (self.degree - step)
            result = result + getattr(self, f"scalar_{step}")
        return result


class BivectorExpExecutor(nn.Module):
    """Return actual exp(B) coefficients in the declared output layout."""

    op = "bivector_exp"

    def __init__(
        self,
        plan: BivectorExpPlan,
        left_product=None,
        *,
        bivector_wedge=None,
        grade4_square=None,
        bivector_grade4_product=None,
        polynomial=None,
        full_polynomial=None,
        square=None,
    ):
        super().__init__()
        self.spec = plan.spec
        self.input_layout = plan.input_layout
        self.output_layout = plan.output_layout
        self.operator_layout = plan.operator_layout
        self.grade4_layout = plan.grade4_layout
        self.input_contract = plan.input_contract
        self.output_contract = plan.output_contract
        self.route = plan.route
        self._set_tolerances(plan.eps)
        self.stabilize_split_divided_differences = plan.spec.r > 0 or (plan.spec.p > 0 and plan.spec.q > 0)
        self.left_product = left_product
        self.bivector_wedge = bivector_wedge
        self.grade4_square = grade4_square
        self.bivector_grade4_product = bivector_grade4_product
        self.polynomial = polynomial
        self.full_polynomial = full_polynomial
        self.square = square
        for name in (
            "bivector_squared_signs",
            "output_scalar_mask",
            "bivector_to_output",
            "grade4_to_output",
            "operator_to_output",
            "operator_output_mask",
            "operator_eye",
        ):
            self.register_buffer(name, getattr(plan, name), persistent=False)

    def _set_tolerances(self, eps):
        self.eps = eps
        self.eps_sq = eps**2
        self.real_exp_series_limit = min(0.5, (eps * 40320.0) ** 0.25)
        self.sinhc_derivative_series_limit = min(0.5, (eps * 7_983_360.0) ** 0.2)
        self.cosh_divided_difference_limit = min(0.5, (eps * 3_628_800.0) ** 0.2)
        self.sinhc_divided_difference_limit = min(0.5, (eps * 39_916_800.0) ** 0.2)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        dtype = self.output_scalar_mask.dtype
        if self.route != "closed" and dtype not in (torch.float32, torch.float64):
            raise ValueError("general bivector_exp requires float32 or float64")
        self._set_tolerances(torch.finfo(dtype).eps)
        if self.route == "left_matrix_exp" or (
            self.route == "closed" and self.spec.n >= 4 and self.output_scalar_mask.device.type == "mps"
        ):
            # Preserve CPU execution for unsupported matrix exponentials and
            # MPS compiled biquadratic derivatives that disagree with eager.
            super()._apply(lambda tensor: tensor.cpu(), recurse=recurse)
        elif self.route == "taylor":
            full_product = self.full_polynomial.products[self.spec.n // 2]
            degree = taylor_degree(dtype)
            self.polynomial.set_degree(degree, full_product, self.operator_layout)
            self.full_polynomial.set_degree(degree, full_product, self.operator_layout)
        return self

    def forward(self, values):
        self.input_contract.validate(values, name="values")
        if self.route == "closed":
            prepared = values.to(device=self.output_scalar_mask.device)
            result = self._closed_simple(prepared) if self.spec.n <= 3 else self._closed_biquadratic(prepared)
            return result.to(device=values.device)
        if self.route == "left_matrix_exp":
            matrix_values = values.to(device=self.operator_eye.device)
            columns = self.left_product.forward_compact(matrix_values.unsqueeze(-2), self.operator_eye)
            result = torch.matrix_exp(columns.transpose(-1, -2))[..., :, 0]
            return self._project(result).to(device=values.device)
        return self._taylor(values)

    def _project(self, values):
        return values.index_select(-1, self.operator_to_output) * self.operator_output_mask

    def _taylor(self, values):
        norm = values.detach().abs().sum(-1, keepdim=True)
        # A finite numerical envelope also bounds the compiled control-flow graph.
        valid = (norm <= 65536.0).all()
        # MPS lacks the assertion kernel; transfer only this validation scalar.
        if values.device.type == "mps":
            valid = valid.cpu()
        torch._assert_async(valid, "bivector_exp Taylor requires coefficient L1 norm <= 65536")
        if torch.compiler.is_compiling():
            return torch.cond((norm > 1).any(), self._scaled_taylor, self.polynomial, (values,))
        if bool((norm > 1).any()):
            return self._scaled_taylor(values)
        return self.polynomial(values)

    def _scaled_taylor(self, values):
        scaling = torch.ceil(torch.log2(values.detach().abs().sum(-1, keepdim=True).clamp_min(1)))
        result = self.full_polynomial(values * torch.exp2(-scaling))
        for step in range(16):
            active = scaling > step
            if torch.compiler.is_compiling():
                result = torch.cond(
                    active.any(),
                    lambda x, mask: torch.where(mask, self.square.forward_compact(x, x), x),
                    lambda x, mask: x.clone(),
                    (result, active),
                )
            elif bool(active.any()):
                result = torch.where(active, self.square.forward_compact(result, result), result)
            else:
                break
        return self._project(result)

    def _closed_simple(self, values: torch.Tensor) -> torch.Tensor:
        alpha = (values * values * self.bivector_squared_signs).sum(dim=-1, keepdim=True)
        scalar_part, coeff_part = self._real_cosh_sinhc_sqrt(alpha)
        return scalar_part * self.output_scalar_mask + (values * coeff_part) @ self.bivector_to_output

    def _closed_biquadratic(self, values: torch.Tensor) -> torch.Tensor:
        if self.bivector_wedge is None or self.grade4_square is None or self.bivector_grade4_product is None:
            raise RuntimeError("closed_biquadratic executor is missing its grade-4 product plans")

        # For n <= 5, B^2 = s + K with K grade-4 and K^2 scalar, so exp(B)
        # closes over {1, B, K, B K}.
        scalar_square = (values * values * self.bivector_squared_signs).sum(dim=-1, keepdim=True)
        grade4_part = self.bivector_wedge.forward_compact(values, values)
        grade4_square = self.grade4_square.forward_compact(grade4_part, grade4_part)
        scalar_part, bivector_coeff, grade4_coeff, bivector_grade4_coeff = self._closed_biquadratic_coefficients(
            scalar_square,
            grade4_square,
        )

        output = (
            scalar_part * self.output_scalar_mask
            + (values * bivector_coeff) @ self.bivector_to_output
            + (grade4_part * grade4_coeff) @ self.grade4_to_output
        )
        bivector_grade4 = self.bivector_grade4_product.forward_compact(values, grade4_part)
        return output + bivector_grade4 * bivector_grade4_coeff

    def _closed_biquadratic_coefficients(
        self,
        scalar_square: torch.Tensor,
        grade4_square: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        split_mask = grade4_square > self.eps_sq
        complex_mask = grade4_square < -self.eps_sq
        base_mask = ~(split_mask | complex_mask)

        zeros = torch.zeros_like(scalar_square)
        ones = torch.ones_like(scalar_square)

        split_scalar = torch.where(split_mask, scalar_square, zeros)
        split_mu = torch.sqrt(torch.where(split_mask, grade4_square, ones))
        plus = split_scalar + split_mu
        minus = split_scalar - split_mu
        c_plus, s_plus = self._real_cosh_sinhc_sqrt(plus)
        c_minus, s_minus = self._real_cosh_sinhc_sqrt(minus)
        split_scalar_coeff = 0.5 * (c_plus + c_minus)
        split_grade4_coeff = (c_plus - c_minus) / (2.0 * split_mu)
        split_bivector_coeff = 0.5 * (s_plus + s_minus)
        split_bivector_grade4_coeff = (s_plus - s_minus) / (2.0 * split_mu)

        complex_nu = torch.sqrt(torch.where(complex_mask, -grade4_square, ones))
        if self.stabilize_split_divided_differences:
            cosh_coalescing = split_mask & (split_mu <= self.cosh_divided_difference_limit)
            sinhc_coalescing = split_mask & (split_mu <= self.sinhc_divided_difference_limit)
            complex_cosh_coalescing = complex_mask & (complex_nu <= self.cosh_divided_difference_limit)
            complex_sinhc_coalescing = complex_mask & (complex_nu <= self.sinhc_divided_difference_limit)
            center_mask = (
                base_mask | cosh_coalescing | sinhc_coalescing | complex_cosh_coalescing | complex_sinhc_coalescing
            )
        else:
            center_mask = base_mask
        center_scalar = torch.where(center_mask, scalar_square, zeros)
        center_scalar_coeff, center_bivector_coeff = self._real_cosh_sinhc_sqrt(center_scalar)
        center_bivector_grade4_coeff = self._real_sinhc_sqrt_derivative(
            center_scalar,
            center_scalar_coeff,
            center_bivector_coeff,
        )
        if self.stabilize_split_divided_differences:
            cosh_correction, sinhc_correction = self._real_coalescing_corrections(
                center_scalar,
                center_bivector_coeff,
                center_bivector_grade4_coeff,
            )
            split_grade4_coeff = torch.where(
                cosh_coalescing,
                torch.addcmul(0.5 * center_bivector_coeff, grade4_square, cosh_correction),
                split_grade4_coeff,
            )
            split_bivector_grade4_coeff = torch.where(
                sinhc_coalescing,
                torch.addcmul(center_bivector_grade4_coeff, grade4_square, sinhc_correction),
                split_bivector_grade4_coeff,
            )

        complex_scalar = torch.where(complex_mask, scalar_square, zeros)
        (
            complex_scalar_coeff,
            complex_bivector_coeff,
            complex_grade4_coeff,
            complex_bivector_grade4_coeff,
        ) = self._complex_biquadratic_coefficients(
            complex_scalar,
            complex_nu,
        )
        if self.stabilize_split_divided_differences:
            complex_grade4_coeff = torch.where(
                complex_cosh_coalescing,
                torch.addcmul(0.5 * center_bivector_coeff, grade4_square, cosh_correction),
                complex_grade4_coeff,
            )
            complex_bivector_grade4_coeff = torch.where(
                complex_sinhc_coalescing,
                torch.addcmul(center_bivector_grade4_coeff, grade4_square, sinhc_correction),
                complex_bivector_grade4_coeff,
            )

        base_scalar_coeff = center_scalar_coeff
        base_bivector_coeff = center_bivector_coeff
        base_grade4_coeff = 0.5 * center_bivector_coeff
        base_bivector_grade4_coeff = center_bivector_grade4_coeff

        scalar_coeff = torch.where(
            split_mask,
            split_scalar_coeff,
            torch.where(complex_mask, complex_scalar_coeff, base_scalar_coeff),
        )
        bivector_coeff = torch.where(
            split_mask,
            split_bivector_coeff,
            torch.where(complex_mask, complex_bivector_coeff, base_bivector_coeff),
        )
        grade4_coeff = torch.where(
            split_mask,
            split_grade4_coeff,
            torch.where(complex_mask, complex_grade4_coeff, base_grade4_coeff),
        )
        bivector_grade4_coeff = torch.where(
            split_mask,
            split_bivector_grade4_coeff,
            torch.where(complex_mask, complex_bivector_grade4_coeff, base_bivector_grade4_coeff),
        )
        return scalar_coeff, bivector_coeff, grade4_coeff, bivector_grade4_coeff

    def _real_cosh_sinhc_sqrt(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Use the tighter cosh(sqrt(x)) remainder, x**4 / 8!, for the cutoff.
        positive = values > self.real_exp_series_limit
        negative = values < -self.real_exp_series_limit
        active = positive | negative
        theta = torch.sqrt(torch.where(active, values.abs(), torch.ones_like(values)))
        # Unselected overflowing branches still poison autograd (0 * inf).
        hyperbolic_theta = torch.where(positive, theta, torch.ones_like(theta))
        series_values = torch.where(active, torch.zeros_like(values), values)
        values_sq = series_values * series_values
        values_cube = values_sq * series_values
        cosh_series = 1.0 + 0.5 * series_values + values_sq / 24.0 + values_cube / 720.0
        sinhc_series = 1.0 + series_values / 6.0 + values_sq / 120.0 + values_cube / 5040.0
        cosh_sqrt = torch.where(
            positive, torch.cosh(hyperbolic_theta), torch.where(negative, torch.cos(theta), cosh_series)
        )
        sinhc_sqrt = torch.where(
            positive,
            torch.sinh(hyperbolic_theta) / hyperbolic_theta,
            torch.where(negative, torch.sin(theta) / theta, sinhc_series),
        )
        return cosh_sqrt, sinhc_sqrt

    def _real_sinhc_sqrt_derivative(
        self,
        values: torch.Tensor,
        cosh_sqrt: torch.Tensor,
        sinhc_sqrt: torch.Tensor,
    ) -> torch.Tensor:
        # Balance the values**4 / 7_983_360 truncation term against roundoff
        # amplified by the direct expression's division by values.
        active = values.abs() > self.sinhc_derivative_series_limit
        safe_values = torch.where(active, values, torch.ones_like(values))
        raw = (cosh_sqrt - sinhc_sqrt) / (2.0 * safe_values)
        values_sq = values * values
        series = 1.0 / 6.0 + values / 60.0 + values_sq / 1680.0 + (values_sq * values) / 90720.0
        return torch.where(active, raw, series)

    def _real_coalescing_corrections(
        self,
        values: torch.Tensor,
        sinhc_sqrt: torch.Tensor,
        derivative: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active = values.abs() > self.sinhc_derivative_series_limit
        safe_values = torch.where(active, values, torch.ones_like(values))
        cosh_raw = (sinhc_sqrt - 6.0 * derivative) / (48.0 * safe_values)
        sinhc_raw = ((4.0 * safe_values + 60.0) * derivative - 10.0 * sinhc_sqrt) / (96.0 * safe_values * safe_values)
        values_sq = values * values
        cosh_series = 1.0 / 720.0 + values / 10080.0 + values_sq / 362880.0
        sinhc_series = 1.0 / 5040.0 + values / 90720.0 + values_sq / 3991680.0
        return torch.where(active, cosh_raw, cosh_series), torch.where(active, sinhc_raw, sinhc_series)

    def _complex_biquadratic_coefficients(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        u, v, radius = self._complex_sqrt_parts(real, imag)
        sinh_u = torch.sinh(u)
        cosh_u = torch.cosh(u)
        sin_v = torch.sin(v)
        cos_v = torch.cos(v)
        cosh_sqrt_real = cosh_u * cos_v
        cosh_sqrt_imag = sinh_u * sin_v
        real_numerator = sinh_u * cos_v
        imag_numerator = cosh_u * sin_v
        sinhc_sqrt_real = (real_numerator * u + imag_numerator * v) / radius
        sinhc_sqrt_imag = (imag_numerator * u - real_numerator * v) / radius
        return cosh_sqrt_real, sinhc_sqrt_real, cosh_sqrt_imag / imag, sinhc_sqrt_imag / imag

    def _complex_sqrt_parts(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        radius = torch.sqrt(real * real + imag * imag)
        nonnegative = real >= 0.0
        magnitude = torch.where(nonnegative, real, -real)
        large = torch.sqrt(0.5 * (radius + magnitude))
        small = imag / (2.0 * large.clamp_min(self.eps))
        u = torch.where(nonnegative, large, small)
        v = torch.where(nonnegative, small, large)
        return u, v, radius
