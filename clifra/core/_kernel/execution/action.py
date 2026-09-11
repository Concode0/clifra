# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Compile-friendly executors for planned linear and versor actions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from clifra.core._kernel.contracts import _check_contract_spec
from clifra.core._kernel.numerics import _matrix_exp_singleton_workaround, eps_like, signed_clamp_min
from clifra.core.layout import GradeLayout
from clifra.core.tensors import TensorContract


class GradedLinearActionExecutor(nn.Module):
    """Apply a vector-space map lifted to declared multivector grades."""

    def __init__(
        self,
        *,
        input_layout: GradeLayout,
        output_layout: GradeLayout,
        scalar_flat_positions: torch.Tensor,
        grade_buffers: tuple[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor], ...],
    ):
        super().__init__()
        self.input_contract = TensorContract.compact(input_layout)
        self.output_contract = _check_contract_spec(
            self.input_contract.spec, TensorContract.compact(output_layout), "output_layout"
        )
        self.input_layout = self.input_contract.layout
        self.output_layout = self.output_contract.layout
        self.input_dim = input_layout.dim
        self.output_dim = output_layout.dim
        self.n = input_layout.spec.n
        self.register_buffer("scalar_flat_positions", scalar_flat_positions, persistent=False)
        self._vector_only = input_layout.grades == output_layout.grades == (1,)
        self._grades = tuple(grade for grade in input_layout.grades if grade > 0 and grade in output_layout.grades)
        for grade, flat_positions, row_indices, col_indices in grade_buffers:
            self.register_buffer(f"flat_positions_{grade}", flat_positions, persistent=False)
            self.register_buffer(f"row_indices_{grade}", row_indices, persistent=False)
            self.register_buffer(f"col_indices_{grade}", col_indices, persistent=False)

    def forward(self, values: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """Return output lanes in ``output_layout``."""
        self._check_values(values)
        if matrix.shape[-2:] != (self.n, self.n):
            raise ValueError(f"matrix trailing shape must be {(self.n, self.n)}, got {tuple(matrix.shape[-2:])}")
        return self.execute(values, matrix)

    def execute(self, values: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """Validation-free grade-lift action for prepared tensors."""
        coefficients = self.coefficients_unchecked(matrix)
        return torch.matmul(coefficients, values.unsqueeze(-1)).squeeze(-1)

    def coefficients(self, matrices: torch.Tensor) -> torch.Tensor:
        """Return lifted action coefficients for vector-space matrices."""
        if matrices.shape[-2:] != (self.n, self.n):
            raise ValueError(f"matrix trailing shape must be {(self.n, self.n)}, got {tuple(matrices.shape[-2:])}")
        return self.coefficients_unchecked(matrices)

    def coefficients_unchecked(self, matrices: torch.Tensor) -> torch.Tensor:
        """Validation-free lifted action coefficients for prepared matrices."""
        if self._vector_only:
            return matrices
        flat = matrices.new_zeros(*matrices.shape[:-2], self.output_dim * self.input_dim)
        if not self._grades:
            flat = flat + matrices.sum(dim=(-2, -1)).unsqueeze(-1) * 0.0
        scalar_positions = self.scalar_flat_positions
        if scalar_positions.numel() > 0:
            scalar_values = matrices.new_ones(*matrices.shape[:-2], scalar_positions.numel())
            flat = flat.index_copy(-1, scalar_positions, scalar_values)

        if 1 in self._grades:
            row_indices = self.row_indices_1[:, 0]
            col_indices = self.col_indices_1[:, 0]
            coefficients = matrices[..., row_indices, col_indices]
            flat = flat.index_copy(-1, self.flat_positions_1, coefficients)

        for grade in self._grades:
            if grade == 1:
                continue
            positions = getattr(self, f"flat_positions_{grade}")
            row_indices = getattr(self, f"row_indices_{grade}")
            col_indices = getattr(self, f"col_indices_{grade}")
            # Explicit cofactors preserve derivatives at singular minors,
            # including the off-diagonal minors of the identity map.
            if grade in (2, 3):
                coefficients = _small_action_minors(matrices, row_indices, col_indices, grade)
            else:
                submatrix = matrices[..., row_indices.unsqueeze(-1), col_indices.unsqueeze(-2)]
                coefficients = _ActionDeterminant.apply(submatrix)
            flat = flat.index_copy(-1, positions, coefficients)

        return flat.reshape(*matrices.shape[:-2], self.output_dim, self.input_dim)

    def _check_values(self, values: torch.Tensor) -> None:
        self.input_contract.validate(values, name="values")


class _ActionDeterminant(torch.autograd.Function):
    """LU forward with polynomial cofactor derivatives, also at singular maps."""

    @staticmethod
    def forward(ctx, matrix):
        ctx.save_for_backward(matrix)
        return torch.linalg.det(matrix.reshape(-1, matrix.shape[-1], matrix.shape[-1])).reshape(matrix.shape[:-2])

    @staticmethod
    def backward(ctx, gradient):
        (matrix,) = ctx.saved_tensors
        n = matrix.shape[-1]
        if n == 1:
            return gradient[..., None, None]
        rows = []
        for i in range(n):
            without_row = torch.cat((matrix[..., :i, :], matrix[..., i + 1 :, :]), dim=-2)
            cofactors = []
            for j in range(n):
                minor = torch.cat((without_row[..., :j], without_row[..., j + 1 :]), dim=-1)
                cofactors.append((-1) ** (i + j) * _ActionDeterminant.apply(minor))
            rows.append(torch.stack(cofactors, dim=-1))
        return gradient[..., None, None] * torch.stack(rows, dim=-2)


def _small_action_minors(matrices: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, grade: int) -> torch.Tensor:
    """Evaluate requested 2x2 or 3x3 minors directly from a vector-space map."""
    a = matrices[..., rows[:, 0], cols[:, 0]]
    b = matrices[..., rows[:, 0], cols[:, 1]]
    d = matrices[..., rows[:, 1], cols[:, 0]]
    e = matrices[..., rows[:, 1], cols[:, 1]]
    if grade == 2:
        return a * e - b * d
    c = matrices[..., rows[:, 0], cols[:, 2]]
    f = matrices[..., rows[:, 1], cols[:, 2]]
    g = matrices[..., rows[:, 2], cols[:, 0]]
    h = matrices[..., rows[:, 2], cols[:, 1]]
    i = matrices[..., rows[:, 2], cols[:, 2]]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


class BivectorVectorGeneratorExecutor(nn.Module):
    """Build vector-space generators induced by grade-2 bivectors."""

    def __init__(
        self,
        *,
        bivector_layout: GradeLayout,
        lane_positions: torch.Tensor,
        flat_positions: torch.Tensor,
        coefficients: torch.Tensor,
    ):
        super().__init__()
        self.bivector_contract = TensorContract.compact(bivector_layout)
        bivector_layout = self.bivector_contract.layout
        if bivector_layout.grades != (2,):
            raise ValueError(f"bivector_layout must contain grade 2 only, got {bivector_layout.grades}")
        self.bivector_layout = bivector_layout
        self.n = bivector_layout.spec.n
        self.register_buffer("lane_positions", lane_positions, persistent=False)
        self.register_buffer("flat_positions", flat_positions, persistent=False)
        self.register_buffer("coefficients", coefficients, persistent=False)

    def forward(self, bivectors: torch.Tensor) -> torch.Tensor:
        """Return vector-space generator matrices for bivectors."""
        self.bivector_contract.validate(bivectors, name="bivectors")
        return self.execute(bivectors)

    def execute(self, bivectors: torch.Tensor) -> torch.Tensor:
        """Validation-free vector generator construction for prepared bivectors."""
        output = bivectors.new_zeros(*bivectors.shape[:-1], self.n * self.n)
        if self.flat_positions.numel() == 0:
            output = output + bivectors.sum(-1, keepdim=True) * 0.0
            return output.reshape(*bivectors.shape[:-1], self.n, self.n)
        terms = torch.index_select(bivectors, -1, self.lane_positions) * self.coefficients
        return output.index_add(-1, self.flat_positions, terms).reshape(*bivectors.shape[:-1], self.n, self.n)


class VersorVectorMatrixExecutor(nn.Module):
    """Return vector-space matrices for compact grade-1 or grade-2 actions."""

    def __init__(
        self,
        *,
        grade: int,
        parameter_layout: GradeLayout,
        eps: float,
        generator: BivectorVectorGeneratorExecutor | None,
        metric_signs: torch.Tensor,
        eye: torch.Tensor,
    ):
        super().__init__()
        self.grade = int(grade)
        self.parameter_layout = parameter_layout
        self.parameter_contract = TensorContract.compact(parameter_layout)
        self.parameter_layout = self.parameter_contract.layout
        self.n = parameter_layout.spec.n
        self.eps = float(eps)
        if self.grade == 2:
            if generator is None:
                raise ValueError("grade-2 vector action requires a prepared generator")
            self.generator = generator
            self.register_buffer("metric_signs", metric_signs, persistent=False)
            self.register_buffer("eye", eye, persistent=False)
        elif self.grade == 1:
            if parameter_layout.grades != (1,):
                raise ValueError(f"parameter_layout must contain grade 1, got {parameter_layout.grades}")
            if generator is not None:
                raise ValueError("grade-1 vector action cannot use a bivector generator")
            self.generator = None
            self.register_buffer("metric_signs", metric_signs, persistent=False)
            self.register_buffer("eye", eye, persistent=False)
        else:
            raise ValueError("planned versor execution currently supports grade=1 and grade=2")

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """Return one vector-space action matrix per input weight row."""
        self.parameter_contract.validate(weights, name="weights")
        return self.execute(weights)

    def execute(self, weights: torch.Tensor) -> torch.Tensor:
        """Validation-free vector-space action matrices for prepared weights."""
        if self.grade == 2:
            matrix = self.generator.execute(weights)
            # MPS has no native matrix exponential; use the same differentiable
            # CPU bridge as materialized Clifford matrix exponentials.
            if matrix.device.type == "mps":
                return _matrix_exp_singleton_workaround(matrix.cpu()).to(matrix.device)
            return _matrix_exp_singleton_workaround(matrix)
        signs = self.metric_signs
        signature_norm_squared = (weights * weights * signs).sum(dim=-1, keepdim=True)
        scale = signature_norm_squared.abs().clamp_min(eps_like(signature_norm_squared)).sqrt()
        normals = weights / scale
        denominator = (normals * normals * signs).sum(dim=-1, keepdim=True)
        denominator = signed_clamp_min(denominator, self.eps)
        weighted_normals = normals * signs
        outer = normals.unsqueeze(-1) * weighted_normals.unsqueeze(-2)
        return self.eye - 2.0 * outer / denominator.unsqueeze(-1)


class FullSandwichActionExecutor(nn.Module):
    """Apply full-layout sandwich action matrices from static Cayley buffers."""

    route = "full_action_matrix"
    op = "sandwich_action"

    def __init__(
        self,
        *,
        layout: GradeLayout,
        cayley_indices: torch.Tensor,
        left_sign_t: torch.Tensor,
        geometric_product_sign_t: torch.Tensor,
    ):
        super().__init__()
        self.contract = TensorContract.compact(layout)
        layout = self.contract.layout
        if layout.grades != tuple(range(layout.spec.n + 1)):
            raise ValueError(f"full sandwich action requires full layout, got {layout.grades}")
        self.layout = layout
        self.dim = layout.dim
        self.register_buffer("cayley_indices", cayley_indices, persistent=False)
        self.register_buffer("left_sign_t", left_sign_t, persistent=False)
        self.register_buffer("geometric_product_sign_t", geometric_product_sign_t, persistent=False)

    def action_matrices_unchecked(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return sandwich coefficients with broadcast factor batch dimensions."""
        indices = self.cayley_indices.reshape(-1)
        left_action = left.index_select(-1, indices).reshape(*left.shape[:-1], self.dim, self.dim)
        right_action = right.index_select(-1, indices).reshape(*right.shape[:-1], self.dim, self.dim)
        return torch.matmul(right_action * self.geometric_product_sign_t, left_action * self.left_sign_t)

    def forward(self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Apply L X R with ordinary broadcasting over all leading dimensions."""
        self._check_factors(left, right)
        self.contract.validate(values, name="sandwich values")
        return self.execute(left, values, right)

    def execute(self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        matrices = self.action_matrices_unchecked(left, right)
        return torch.matmul(matrices, values.unsqueeze(-1)).squeeze(-1)

    def _check_factors(self, left: torch.Tensor, right: torch.Tensor) -> None:
        self.contract.validate(left, name="left")
        self.contract.validate(right, name="right")


@dataclass(frozen=True)
class ActionComponents:
    rotor_layout: object
    middle_layout: object
    exponential: object
    reverse: object
    left_product: object
    right_product: object
    norm: object
    involution: object


class VersorActionExecutor(nn.Module):
    """Prepared execution components for versor actions."""

    action_name = "versor"

    def __init__(
        self,
        *,
        grade: int,
        input_layout: GradeLayout,
        output_layout: GradeLayout,
        parameter_layout: GradeLayout,
        route: str,
        components: ActionComponents,
        action: GradedLinearActionExecutor | None,
        vector_matrix: VersorVectorMatrixExecutor | None,
        full_action: FullSandwichActionExecutor | None,
        rotor_full_indices: torch.Tensor,
        parameter_full_indices: torch.Tensor,
        eps_sq: float,
    ):
        super().__init__()
        self.input_contract = TensorContract.compact(input_layout)
        self.output_contract = _check_contract_spec(
            self.input_contract.spec, TensorContract.compact(output_layout), "output_layout"
        )
        self.parameter_contract = _check_contract_spec(
            self.input_contract.spec, TensorContract.compact(parameter_layout), "parameter_layout"
        )
        self.input_layout = self.input_contract.layout
        self.output_layout = self.output_contract.layout
        self.parameter_layout = self.parameter_contract.layout
        self.grade = int(grade)
        if self.grade not in {1, 2}:
            raise ValueError(f"planned {self.action_name} execution currently supports grade=1 and grade=2")

        self.route = str(route)
        if self.route not in {"vector_matrix", "rotor_product", "full_action_matrix"}:
            raise ValueError(f"unsupported {self.action_name} action execution path {self.route!r}")
        self.use_full_action = self.route == "full_action_matrix"
        self.use_rotor_product_action = self.route == "rotor_product"
        self.action = action
        self.vector_matrix = vector_matrix
        self.left_product = None
        self.right_product = None
        self.middle_layout = None
        self.full_action = full_action
        self.full_dim = input_layout.spec.dim
        self.eps_sq = float(eps_sq)
        self.rotor_layout = components.rotor_layout
        self.middle_layout = components.middle_layout
        self.bivector_exp = components.exponential
        self.rotor_reverse = components.reverse if self.grade == 2 else None
        self.parameter_reverse = components.reverse if self.grade == 1 else None
        self.parameter_signature_norm_squared = components.norm
        self.input_involution = components.involution
        self.left_product = components.left_product
        self.right_product = components.right_product
        self.register_buffer("rotor_full_indices", rotor_full_indices, persistent=False)
        self.register_buffer("parameter_full_indices", parameter_full_indices, persistent=False)

    def _planned_full_versor_factors(self, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.grade == 2:
            rotor, right = self._planned_rotor_factors(weights)
            left_full = _materialize_full_from_indices(rotor, self.rotor_full_indices, self.full_dim)
            right_full = _materialize_full_from_indices(right, self.rotor_full_indices, self.full_dim)
            return left_full, right_full

        signature_norm_squared = self.parameter_signature_norm_squared(weights)
        scale = signature_norm_squared.abs().clamp_min(eps_like(signature_norm_squared)).sqrt()
        versor = weights / scale
        left = versor
        denominator = signed_clamp_min(self.parameter_signature_norm_squared(versor), self.eps_sq)
        right = self.parameter_reverse(versor) / denominator
        left_full = _materialize_full_from_indices(left, self.parameter_full_indices, self.full_dim)
        right_full = _materialize_full_from_indices(right, self.parameter_full_indices, self.full_dim)
        return left_full, right_full

    def _planned_rotor_factors(self, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rotor = self.bivector_exp(-0.5 * weights)
        return rotor, self.rotor_reverse(rotor)

    def forward(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Return transformed values in ``output_layout`` lanes."""
        self.input_contract.validate(values, name="versor values")
        self.parameter_contract.validate(weights, name="versor weights")
        return self.execute(values, weights)

    def execute(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Validation-free versor action for prepared tensors."""
        if self.use_full_action:
            left, right = self._planned_full_versor_factors(weights)
            if self.grade == 1:
                values = self.input_involution(values)
            return self.full_action.execute(left, values, right)
        if self.use_rotor_product_action:
            left, right = self._planned_rotor_factors(weights)
            middle = self.left_product(left, values)
            return self.right_product(middle, right)
        matrix = self.vector_matrix.execute(weights)
        return self.action.execute(values, matrix)


def _materialize_full_from_indices(values: torch.Tensor, indices: torch.Tensor, dim: int) -> torch.Tensor:
    output = values.new_zeros(*values.shape[:-1], dim)
    return output.index_copy(-1, indices, values)
