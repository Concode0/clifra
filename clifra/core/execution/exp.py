# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Bivector exponential executors for static exp plans."""

from __future__ import annotations

import torch
import torch.nn as nn

from clifra.core.execution.product import GradeProductExecutor
from clifra.core.planning.exp import BivectorExpPlan


class BivectorExpExecutor(nn.Module):
    """Compile-friendly ``exp(B)`` executor for grade-2 inputs.

    Dimensions up to five use closed formulas. Higher-dimensional CPU/CUDA
    plans use the left-product matrix exponential over the full even
    subalgebra. Higher-dimensional MPS plans use fixed-iteration bivector
    decomposition because native ``torch.matrix_exp`` is not available there.
    """

    op = "bivector_exp"

    def __init__(
        self,
        plan: BivectorExpPlan,
        left_product: GradeProductExecutor | None,
        *,
        bivector_wedge: GradeProductExecutor | None = None,
        grade4_square: GradeProductExecutor | None = None,
        bivector_grade4_product: GradeProductExecutor | None = None,
        vector_contraction: GradeProductExecutor | None = None,
        vector_wedge: GradeProductExecutor | None = None,
        rotor_product: GradeProductExecutor | None = None,
    ):
        super().__init__()
        self.spec = plan.spec
        self.input_layout = plan.input_layout
        self.vector_layout = plan.vector_layout
        self.grade4_layout = plan.grade4_layout
        self.operator_layout = plan.operator_layout
        self.output_layout = plan.output_layout
        self.executor_family = plan.executor_family
        self.eps = plan.eps
        self.eps_sq = plan.eps_sq
        self.component_count = int(plan.component_count)
        self.fixed_iterations = int(plan.fixed_iterations)
        self.decomposition_tolerance = float(plan.decomposition_tolerance)
        self.left_product = left_product
        self.bivector_wedge = bivector_wedge
        self.grade4_square = grade4_square
        self.bivector_grade4_product = bivector_grade4_product
        self.vector_contraction = vector_contraction
        self.vector_wedge = vector_wedge
        self.rotor_product = rotor_product
        self.register_buffer("bivector_squared_signs", plan.bivector_squared_signs, persistent=False)
        self.register_buffer("vector_seed", plan.vector_seed, persistent=False)
        self.register_buffer("output_scalar_mask", plan.output_scalar_mask, persistent=False)
        self.register_buffer("operator_scalar_mask", plan.operator_scalar_mask, persistent=False)
        self.register_buffer("bivector_to_output", plan.bivector_to_output, persistent=False)
        self.register_buffer("bivector_to_operator", plan.bivector_to_operator, persistent=False)
        self.register_buffer("grade4_to_output", plan.grade4_to_output, persistent=False)
        self.register_buffer("operator_to_output", plan.operator_to_output, persistent=False)
        self.register_buffer("operator_eye", plan.operator_eye, persistent=False)
        self.operator_scalar_position = int(plan.operator_scalar_position)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return ``exp(values)`` in ``output_layout`` lanes."""
        if values.shape[-1] != self.input_layout.dim:
            raise ValueError(f"bivector exp input dimension must be {self.input_layout.dim}, got {values.shape[-1]}")
        if self.executor_family == "closed_simple":
            return self._closed_simple(values)
        if self.executor_family == "closed_biquadratic":
            return self._closed_biquadratic(values)
        if self.executor_family == "decomposed":
            return self._decomposed(values)
        return self._left_matrix_exp(values)

    def _closed_simple(self, values: torch.Tensor) -> torch.Tensor:
        alpha = (values * values * self._signs_for(values)).sum(dim=-1, keepdim=True)
        scalar_part, coeff_part = self._real_cosh_sinhc_sqrt(alpha)
        return scalar_part * self.output_scalar_mask + (values * coeff_part) @ self.bivector_to_output

    def _closed_simple_operator(self, values: torch.Tensor) -> torch.Tensor:
        alpha = (values * values * self._signs_for(values)).sum(dim=-1, keepdim=True)
        scalar_part, coeff_part = self._real_cosh_sinhc_sqrt(alpha)
        return scalar_part * self.operator_scalar_mask + (values * coeff_part) @ self.bivector_to_operator

    def _closed_biquadratic(self, values: torch.Tensor) -> torch.Tensor:
        if self.bivector_wedge is None or self.grade4_square is None or self.bivector_grade4_product is None:
            raise RuntimeError("closed_biquadratic executor is missing its grade-4 product plans")

        # For n <= 5, B^2 = s + K with K grade-4 and K^2 scalar, so exp(B)
        # closes over {1, B, K, B K}.
        scalar_square = (values * values * self._signs_for(values)).sum(dim=-1, keepdim=True)
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

        complex_scalar = torch.where(complex_mask, scalar_square, zeros)
        complex_nu = torch.sqrt(torch.where(complex_mask, -grade4_square, ones))
        (
            complex_scalar_coeff,
            complex_bivector_coeff,
            complex_grade4_coeff,
            complex_bivector_grade4_coeff,
        ) = self._complex_biquadratic_coefficients(
            complex_scalar,
            complex_nu,
        )

        base_scalar = torch.where(base_mask, scalar_square, zeros)
        base_scalar_coeff, base_bivector_coeff = self._real_cosh_sinhc_sqrt(base_scalar)
        base_grade4_coeff = 0.5 * base_bivector_coeff
        base_bivector_grade4_coeff = self._real_sinhc_sqrt_derivative(
            base_scalar,
            base_scalar_coeff,
            base_bivector_coeff,
        )

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
        positive = values > self.eps_sq
        negative = values < -self.eps_sq
        active = positive | negative
        theta = torch.sqrt(torch.where(active, values.abs(), torch.ones_like(values)))
        values_sq = values * values
        cosh_series = 1.0 + 0.5 * values + values_sq / 24.0 + (values_sq * values) / 720.0
        sinhc_series = 1.0 + values / 6.0 + values_sq / 120.0 + (values_sq * values) / 5040.0
        cosh_sqrt = torch.where(positive, torch.cosh(theta), torch.where(negative, torch.cos(theta), cosh_series))
        sinhc_sqrt = torch.where(
            positive,
            torch.sinh(theta) / theta,
            torch.where(negative, torch.sin(theta) / theta, sinhc_series),
        )
        return cosh_sqrt, sinhc_sqrt

    def _real_sinhc_sqrt_derivative(
        self,
        values: torch.Tensor,
        cosh_sqrt: torch.Tensor,
        sinhc_sqrt: torch.Tensor,
    ) -> torch.Tensor:
        active = values.abs() > self.eps_sq
        safe_values = torch.where(active, values, torch.ones_like(values))
        raw = (cosh_sqrt - sinhc_sqrt) / (2.0 * safe_values)
        values_sq = values * values
        series = 1.0 / 6.0 + values / 60.0 + values_sq / 1680.0 + (values_sq * values) / 90720.0
        return torch.where(active, raw, series)

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
        u_sq = (radius + real).clamp_min(0.0) * 0.5
        v_sq = (radius - real).clamp_min(0.0) * 0.5
        u = torch.sqrt(u_sq.clamp_min(self.eps_sq))
        v = torch.sqrt(v_sq.clamp_min(self.eps_sq))
        return u, v, radius

    def _left_matrix_exp(self, values: torch.Tensor) -> torch.Tensor:
        if self.left_product is None:
            raise RuntimeError("left_matrix_exp executor is missing its left-product plan")
        basis = self._basis_for(values)
        columns = self.left_product.forward_compact(values.unsqueeze(-2), basis)
        operator = columns.transpose(-1, -2)
        exp_operator = torch.matrix_exp(operator)
        even_output = exp_operator[..., :, self.operator_scalar_position]
        return even_output @ self.operator_to_output

    def _decomposed(self, values: torch.Tensor) -> torch.Tensor:
        if self.vector_contraction is None or self.vector_wedge is None or self.rotor_product is None:
            raise RuntimeError("decomposed bivector exp executor is missing decomposition product plans")

        with torch.no_grad():
            components = self._decompose(values.detach())

        result = self._operator_identity(values)
        residual = values
        for component in components:
            plane_norm = component.norm(dim=-1, keepdim=True).clamp(min=self.eps_sq)
            plane_direction = component / plane_norm
            coefficient = (residual * plane_direction).sum(dim=-1, keepdim=True)
            live_component = coefficient * plane_direction
            residual = residual - live_component
            rotor = self._closed_simple_operator(live_component)
            result = self.rotor_product.forward_compact(result, rotor)
        return self._operator_to_output(result)

    def _decompose(self, values: torch.Tensor) -> list[torch.Tensor]:
        components: list[torch.Tensor] = []
        residual = values
        for _ in range(self.component_count):
            component = self._power_iteration_component(residual)
            active = residual.norm(dim=-1, keepdim=True) > self.eps
            component = component * active
            components.append(component)
            residual = residual - component
        return components

    def _power_iteration_component(self, values: torch.Tensor) -> torch.Tensor:
        vector = self._seed_vector(values)
        vector = vector / vector.norm(dim=-1, keepdim=True).clamp(min=self.eps)

        for _ in range(self.fixed_iterations):
            previous = vector
            updated = self.vector_contraction.forward_compact(values, vector)
            updated = updated / updated.norm(dim=-1, keepdim=True).clamp(min=self.eps)
            converged = (updated - previous).norm(dim=-1, keepdim=True) < self.decomposition_tolerance
            vector = torch.where(converged, previous, updated)

        u = self.vector_contraction.forward_compact(values, vector)
        u_norm = u.norm(dim=-1, keepdim=True)
        u = u / u_norm.clamp(min=self.eps)
        return u_norm * self.vector_wedge.forward_compact(u, vector)

    def _seed_vector(self, values: torch.Tensor) -> torch.Tensor:
        leading = values.shape[:-1]
        uniform = self._vector_seed_for(values).expand(*leading, self.vector_layout.dim)
        probe = self.vector_contraction.forward_compact(values, uniform)
        probe_norm = probe.norm(dim=-1, keepdim=True)
        return torch.where(probe_norm > self.eps, probe, uniform)

    def _operator_identity(self, values: torch.Tensor) -> torch.Tensor:
        ones = values.new_ones(*values.shape[:-1], 1)
        return ones * self.operator_scalar_mask

    def _operator_to_output(self, operator_values: torch.Tensor) -> torch.Tensor:
        return operator_values @ self.operator_to_output

    def _basis_for(self, values: torch.Tensor) -> torch.Tensor:
        return self.operator_eye

    def _vector_seed_for(self, values: torch.Tensor) -> torch.Tensor:
        return self.vector_seed

    def _signs_for(self, values: torch.Tensor) -> torch.Tensor:
        return self.bivector_squared_signs


__all__ = ["BivectorExpExecutor"]
