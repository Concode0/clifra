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

    Low-dimensional algebras use the closed simple-bivector formula. CPU/CUDA
    higher-dimensional plans use the left-product matrix exponential over the
    full even subalgebra. MPS plans use fixed-iteration bivector decomposition
    because native ``torch.matrix_exp`` is not available on that backend.
    """

    op = "bivector_exp"

    def __init__(
        self,
        plan: BivectorExpPlan,
        left_product: GradeProductExecutor | None,
        *,
        vector_contraction: GradeProductExecutor | None = None,
        vector_wedge: GradeProductExecutor | None = None,
        rotor_product: GradeProductExecutor | None = None,
    ):
        super().__init__()
        self.spec = plan.spec
        self.input_layout = plan.input_layout
        self.vector_layout = plan.vector_layout
        self.operator_layout = plan.operator_layout
        self.output_layout = plan.output_layout
        self.executor_family = plan.executor_family
        self.regime = plan.regime
        self.eps = plan.eps
        self.eps_sq = plan.eps_sq
        self.component_count = int(plan.component_count)
        self.fixed_iterations = int(plan.fixed_iterations)
        self.decomposition_tolerance = float(plan.decomposition_tolerance)
        self.left_product = left_product
        self.vector_contraction = vector_contraction
        self.vector_wedge = vector_wedge
        self.rotor_product = rotor_product
        self.register_buffer("bivector_squared_signs", plan.bivector_squared_signs, persistent=False)
        self.register_buffer("vector_seed", plan.vector_seed, persistent=False)
        self.register_buffer("bivector_input_positions", plan.bivector_input_positions, persistent=False)
        self.register_buffer("bivector_output_positions", plan.bivector_output_positions, persistent=False)
        self.register_buffer("bivector_operator_positions", plan.bivector_operator_positions, persistent=False)
        self.register_buffer("output_from_operator_positions", plan.output_from_operator_positions, persistent=False)
        self.register_buffer("operator_to_output_positions", plan.operator_to_output_positions, persistent=False)
        self.register_buffer("scalar_output_index", plan.scalar_output_index, persistent=False)
        self.register_buffer("operator_scalar_index", plan.operator_scalar_index, persistent=False)
        self.register_buffer("operator_eye", plan.operator_eye, persistent=False)
        self.scalar_output_position = int(plan.scalar_output_position)
        self.operator_scalar_position = int(plan.operator_scalar_position)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return ``exp(values)`` in ``output_layout`` lanes."""
        if values.shape[-1] != self.input_layout.dim:
            raise ValueError(f"bivector exp input dimension must be {self.input_layout.dim}, got {values.shape[-1]}")
        if self.executor_family == "closed_simple":
            return self._closed_simple(values)
        if self.executor_family == "decomposed":
            return self._decomposed(values)
        return self._left_matrix_exp(values)

    def _closed_simple(self, values: torch.Tensor) -> torch.Tensor:
        alpha = (values * values * self._signs_for(values)).sum(dim=-1, keepdim=True)
        abs_alpha = alpha.abs().clamp(min=self.eps_sq)
        theta = torch.sqrt(abs_alpha)

        if self.regime == "elliptic":
            scalar_part = torch.cos(theta)
            coeff_part = torch.where(theta > self.eps, torch.sin(theta) / theta, 1.0 - abs_alpha / 6.0)
        elif self.regime == "hyperbolic":
            scalar_part = torch.cosh(theta)
            coeff_part = torch.where(theta > self.eps, torch.sinh(theta) / theta, 1.0 + abs_alpha / 6.0)
        else:
            cos_theta = torch.cos(theta)
            sinc_theta = torch.where(theta > self.eps, torch.sin(theta) / theta, 1.0 - abs_alpha / 6.0)
            cosh_theta = torch.cosh(theta)
            sinhc_theta = torch.where(theta > self.eps, torch.sinh(theta) / theta, 1.0 + abs_alpha / 6.0)
            is_elliptic = alpha < -self.eps_sq
            is_hyperbolic = alpha > self.eps_sq
            scalar_part = torch.where(
                is_elliptic, cos_theta, torch.where(is_hyperbolic, cosh_theta, torch.ones_like(theta))
            )
            coeff_part = torch.where(
                is_elliptic, sinc_theta, torch.where(is_hyperbolic, sinhc_theta, torch.ones_like(theta))
            )

        output = values.new_zeros(*values.shape[:-1], self.output_layout.dim)
        if self.scalar_output_position >= 0:
            output = output.index_copy(-1, self._index_for(self.scalar_output_index, values), scalar_part)
        if self.bivector_input_positions.numel() == 0:
            return output
        input_positions = self._index_for(self.bivector_input_positions, values)
        output_positions = self._index_for(self.bivector_output_positions, values)
        bivector_values = torch.index_select(values * coeff_part, -1, input_positions)
        return output.index_copy(-1, output_positions, bivector_values)

    def _closed_simple_operator(self, values: torch.Tensor) -> torch.Tensor:
        alpha = (values * values * self._signs_for(values)).sum(dim=-1, keepdim=True)
        abs_alpha = alpha.abs().clamp(min=self.eps_sq)
        theta = torch.sqrt(abs_alpha)

        if self.regime == "elliptic":
            scalar_part = torch.cos(theta)
            coeff_part = torch.where(theta > self.eps, torch.sin(theta) / theta, 1.0 - abs_alpha / 6.0)
        elif self.regime == "hyperbolic":
            scalar_part = torch.cosh(theta)
            coeff_part = torch.where(theta > self.eps, torch.sinh(theta) / theta, 1.0 + abs_alpha / 6.0)
        else:
            cos_theta = torch.cos(theta)
            sinc_theta = torch.where(theta > self.eps, torch.sin(theta) / theta, 1.0 - abs_alpha / 6.0)
            cosh_theta = torch.cosh(theta)
            sinhc_theta = torch.where(theta > self.eps, torch.sinh(theta) / theta, 1.0 + abs_alpha / 6.0)
            is_elliptic = alpha < -self.eps_sq
            is_hyperbolic = alpha > self.eps_sq
            scalar_part = torch.where(
                is_elliptic, cos_theta, torch.where(is_hyperbolic, cosh_theta, torch.ones_like(theta))
            )
            coeff_part = torch.where(
                is_elliptic, sinc_theta, torch.where(is_hyperbolic, sinhc_theta, torch.ones_like(theta))
            )

        output = values.new_zeros(*values.shape[:-1], self.operator_layout.dim)
        output = output.index_copy(-1, self._index_for(self.operator_scalar_index, values), scalar_part)
        if self.bivector_operator_positions.numel() == 0:
            return output
        return output.index_copy(
            -1,
            self._index_for(self.bivector_operator_positions, values),
            values * coeff_part,
        )

    def _left_matrix_exp(self, values: torch.Tensor) -> torch.Tensor:
        if self.left_product is None:
            raise RuntimeError("left_matrix_exp executor is missing its left-product plan")
        basis = self._basis_for(values)
        columns = self.left_product.forward_compact(values.unsqueeze(-2), basis)
        operator = columns.transpose(-1, -2)
        exp_operator = torch.matrix_exp(operator)
        even_output = exp_operator[..., :, self.operator_scalar_position]
        output = values.new_zeros(*values.shape[:-1], self.output_layout.dim)
        if self.output_from_operator_positions.numel() == 0:
            return output
        gather = self._index_for(self.output_from_operator_positions, values)
        scatter = self._index_for(self.operator_to_output_positions, values)
        return output.index_copy(-1, scatter, torch.index_select(even_output, -1, gather))

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
        return self._operator_to_output(result, values)

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
        output = values.new_zeros(*values.shape[:-1], self.operator_layout.dim)
        ones = values.new_ones(*values.shape[:-1], 1)
        return output.index_copy(-1, self._index_for(self.operator_scalar_index, values), ones)

    def _operator_to_output(self, operator_values: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        output = values.new_zeros(*values.shape[:-1], self.output_layout.dim)
        if self.output_from_operator_positions.numel() == 0:
            return output
        gather = self._index_for(self.output_from_operator_positions, values)
        scatter = self._index_for(self.operator_to_output_positions, values)
        return output.index_copy(-1, scatter, torch.index_select(operator_values, -1, gather))

    def _basis_for(self, values: torch.Tensor) -> torch.Tensor:
        return self.operator_eye

    def _vector_seed_for(self, values: torch.Tensor) -> torch.Tensor:
        return self.vector_seed

    def _signs_for(self, values: torch.Tensor) -> torch.Tensor:
        return self.bivector_squared_signs

    @staticmethod
    def _index_for(index: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return index


__all__ = ["BivectorExpExecutor"]
