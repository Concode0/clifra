# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Compile-friendly executors for planned linear and versor actions."""

from __future__ import annotations

import torch
import torch.nn as nn

from clifra.core.foundation.basis import expand_output_grades, operation_coefficient
from clifra.core.foundation.layout import GradeLayout
from clifra.core.foundation.numerics import eps_like, signed_clamp_min
from clifra.core.foundation.validation import validate_channel_values
from clifra.core.storage import materialize_full, metric_self_signs


class GradedLinearActionExecutor(nn.Module):
    """Apply a vector-space map lifted to declared multivector grades."""

    def __init__(self, *, input_layout: GradeLayout, output_layout: GradeLayout):
        super().__init__()
        if input_layout.spec != output_layout.spec:
            raise ValueError(f"layout mismatch: {input_layout.spec} vs {output_layout.spec}")
        self.input_layout = input_layout
        self.output_layout = output_layout
        self.input_dim = input_layout.dim
        self.output_dim = output_layout.dim
        self.n = input_layout.spec.n
        self.register_buffer("scalar_flat_positions", _scalar_action_positions(input_layout, output_layout), persistent=False)
        self._grade_count = self.n + 1
        for grade in range(1, self._grade_count):
            flat_positions, row_indices, col_indices = _graded_action_plan_tensors(
                input_layout,
                output_layout,
                grade=grade,
            )
            self.register_buffer(f"flat_positions_{grade}", flat_positions, persistent=False)
            self.register_buffer(f"row_indices_{grade}", row_indices, persistent=False)
            self.register_buffer(f"col_indices_{grade}", col_indices, persistent=False)

    def forward(self, values: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """Return output lanes in ``output_layout``."""
        self._check_values(values)
        if matrix.shape[0] != values.shape[-2]:
            raise ValueError(f"matrix channels {matrix.shape[0]} do not match input channels {values.shape[-2]}")
        return self.execute(values, matrix)

    def execute(self, values: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """Validation-free grade-lift action for prepared tensors."""
        coefficients = self.coefficients_unchecked(matrix)
        return torch.einsum("coi,...ci->...co", coefficients, values)

    def multi(self, values: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
        """Apply multiple vector-space maps to declared grade lanes."""
        self._check_values(values)
        return self.multi_execute(values, matrices)

    def multi_execute(self, values: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
        """Validation-free multi-action for prepared tensors."""
        coefficients = self.coefficients_unchecked(matrices)
        return torch.einsum("koi,...ci->...cko", coefficients, values)

    def coefficients(self, matrices: torch.Tensor) -> torch.Tensor:
        """Return lifted action coefficients for vector-space matrices."""
        if matrices.shape[-2:] != (self.n, self.n):
            raise ValueError(f"matrix trailing shape must be {(self.n, self.n)}, got {tuple(matrices.shape[-2:])}")
        if matrices.ndim != 3:
            raise ValueError(f"matrix must have shape [items, {self.n}, {self.n}], got {tuple(matrices.shape)}")
        return self.coefficients_unchecked(matrices)

    def coefficients_unchecked(self, matrices: torch.Tensor) -> torch.Tensor:
        """Validation-free lifted action coefficients for prepared matrices."""
        flat = matrices.new_zeros(matrices.shape[0], self.output_dim * self.input_dim)
        scalar_positions = self.scalar_flat_positions
        if scalar_positions.numel() > 0:
            scalar_values = matrices.new_ones(matrices.shape[0], scalar_positions.numel())
            flat = flat.index_copy(-1, scalar_positions, scalar_values)

        for grade in range(1, self._grade_count):
            positions = getattr(self, f"flat_positions_{grade}")
            if positions.numel() == 0:
                continue
            row_indices = getattr(self, f"row_indices_{grade}")
            col_indices = getattr(self, f"col_indices_{grade}")
            submatrix = matrices[:, row_indices.unsqueeze(-1), col_indices.unsqueeze(-2)]
            flat = flat.index_copy(-1, positions, torch.linalg.det(submatrix))

        return flat.reshape(matrices.shape[0], self.output_dim, self.input_dim)

    def _check_values(self, values: torch.Tensor) -> None:
        if values.shape[-1] != self.input_dim:
            raise ValueError(f"input dimension must be {self.input_dim}, got {values.shape[-1]}")
        if values.ndim < 2:
            raise ValueError(f"values must include channel and lane axes, got shape {tuple(values.shape)}")


class BivectorVectorGeneratorExecutor(nn.Module):
    """Build vector-space generators induced by grade-2 bivectors."""

    def __init__(self, *, bivector_layout: GradeLayout, dtype: torch.dtype = torch.float32, device=None):
        super().__init__()
        if bivector_layout.grades != (2,):
            raise ValueError(f"bivector_layout must contain grade 2 only, got {bivector_layout.grades}")
        self.bivector_layout = bivector_layout
        self.n = bivector_layout.spec.n
        lane_positions: list[int] = []
        flat_positions: list[int] = []
        coefficients: list[float] = []
        vector_layout = bivector_layout.spec.layout((1,))
        vector_positions = {index: position for position, index in enumerate(vector_layout.basis_indices)}
        for bivector_position, bivector_index in enumerate(bivector_layout.basis_indices):
            for input_position, input_index in enumerate(vector_layout.basis_indices):
                output_index = bivector_index ^ input_index
                output_position = vector_positions.get(output_index)
                if output_position is None:
                    continue
                coefficient = -0.5 * operation_coefficient(
                    bivector_index,
                    input_index,
                    bivector_layout.spec.p,
                    bivector_layout.spec.q,
                    bivector_layout.spec.r,
                    "commutator",
                )
                if coefficient == 0.0:
                    continue
                lane_positions.append(bivector_position)
                flat_positions.append(output_position * self.n + input_position)
                coefficients.append(coefficient)
        self.register_buffer("lane_positions", torch.tensor(lane_positions, dtype=torch.long, device=device), persistent=False)
        self.register_buffer("flat_positions", torch.tensor(flat_positions, dtype=torch.long, device=device), persistent=False)
        self.register_buffer("coefficients", torch.tensor(coefficients, dtype=dtype, device=device), persistent=False)

    def forward(self, bivectors: torch.Tensor) -> torch.Tensor:
        """Return vector-space generator matrices for bivectors."""
        if bivectors.shape[-1] != self.bivector_layout.dim:
            raise ValueError(f"bivector dimension must be {self.bivector_layout.dim}, got {bivectors.shape[-1]}")
        return self.execute(bivectors)

    def execute(self, bivectors: torch.Tensor) -> torch.Tensor:
        """Validation-free vector generator construction for prepared bivectors."""
        output = bivectors.new_zeros(*bivectors.shape[:-1], self.n * self.n)
        if self.flat_positions.numel() == 0:
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
        dtype: torch.dtype = torch.float32,
        device=None,
    ):
        super().__init__()
        self.grade = int(grade)
        self.parameter_layout = parameter_layout
        self.n = parameter_layout.spec.n
        self.eps = float(eps)
        if self.grade == 2:
            self.generator = BivectorVectorGeneratorExecutor(
                bivector_layout=parameter_layout,
                dtype=dtype,
                device=device,
            )
            self.register_buffer("metric_signs", torch.empty(0, dtype=dtype, device=device), persistent=False)
            self.register_buffer("eye", torch.empty(0, dtype=dtype, device=device), persistent=False)
        elif self.grade == 1:
            if parameter_layout.grades != (1,):
                raise ValueError(f"parameter_layout must contain grade 1, got {parameter_layout.grades}")
            self.generator = None
            signs = [
                operation_coefficient(
                    index,
                    index,
                    parameter_layout.spec.p,
                    parameter_layout.spec.q,
                    parameter_layout.spec.r,
                    "gp",
                )
                for index in parameter_layout.basis_indices
            ]
            self.register_buffer("metric_signs", torch.tensor(signs, dtype=dtype, device=device), persistent=False)
            self.register_buffer("eye", torch.eye(self.n, dtype=dtype, device=device), persistent=False)
        else:
            raise ValueError("planned versor execution currently supports grade=1 and grade=2")

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """Return one vector-space action matrix per input weight row."""
        if weights.shape[-1] != self.parameter_layout.dim:
            raise ValueError(f"weights last dimension must be {self.parameter_layout.dim}, got {weights.shape[-1]}")
        return self.execute(weights)

    def execute(self, weights: torch.Tensor) -> torch.Tensor:
        """Validation-free vector-space action matrices for prepared weights."""
        if self.grade == 2:
            return torch.matrix_exp(self.generator.execute(weights))
        signs = self.metric_signs
        norm_sq = (weights * weights * signs).sum(dim=-1, keepdim=True)
        scale = norm_sq.abs().clamp_min(eps_like(norm_sq)).sqrt()
        normals = weights / scale
        denominator = (normals * normals * signs).sum(dim=-1, keepdim=True)
        denominator = signed_clamp_min(denominator, self.eps)
        weighted_normals = normals * signs
        outer = normals.unsqueeze(-1) * weighted_normals.unsqueeze(-2)
        return self.eye - 2.0 * outer / denominator.unsqueeze(-1)


class FullSandwichActionExecutor(nn.Module):
    """Apply full-layout sandwich action matrices from static Cayley buffers."""

    executor_family = "action_matrix"
    op = "sandwich_action"

    def __init__(self, *, layout: GradeLayout, cayley_indices: torch.Tensor, left_sign_t: torch.Tensor, gp_sign_t: torch.Tensor):
        super().__init__()
        if layout.grades != tuple(range(layout.spec.n + 1)):
            raise ValueError(f"full sandwich action requires full layout, got {layout.grades}")
        self.layout = layout
        self.dim = layout.dim
        self.register_buffer("cayley_indices", cayley_indices, persistent=False)
        self.register_buffer("left_sign_t", left_sign_t, persistent=False)
        self.register_buffer("gp_sign_t", gp_sign_t, persistent=False)

    @classmethod
    def from_layout(cls, layout: GradeLayout, *, device=None, dtype: torch.dtype = torch.float32):
        """Build full-layout sandwich action buffers from algebra metadata."""
        dim = layout.spec.dim
        indices = torch.arange(dim, dtype=torch.long, device=device)
        cayley_indices = indices.unsqueeze(0) ^ indices.unsqueeze(1)
        sign_rows: list[list[float]] = []
        for left_index in range(dim):
            row = []
            for output_index in range(dim):
                right_index = left_index ^ output_index
                row.append(operation_coefficient(left_index, right_index, layout.spec.p, layout.spec.q, layout.spec.r, "gp"))
            sign_rows.append(row)
        gp_signs = torch.tensor(sign_rows, dtype=dtype, device=device)
        output_indices = torch.arange(dim, dtype=torch.long, device=device).unsqueeze(0).expand(dim, dim)
        left_sign_t = gp_signs[cayley_indices, output_indices].T.contiguous()
        return cls(
            layout=layout,
            cayley_indices=cayley_indices,
            left_sign_t=left_sign_t,
            gp_sign_t=gp_signs.T.contiguous(),
        )

    def action_matrices(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return matrices ``M[..., k, j]`` such that ``output[..., k] = M @ x``."""
        if left.ndim != 2 or right.ndim != 2:
            raise ValueError(f"left and right factors must have shape [items, {self.dim}]")
        if left.shape != right.shape or left.shape[-1] != self.dim:
            raise ValueError(f"left and right factors must have matching shape [items, {self.dim}]")
        return self.action_matrices_unchecked(left, right)

    def action_matrices_unchecked(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Validation-free action matrices for prepared full-layout factors."""
        left_gathered = left[:, self.cayley_indices]
        left_action = left_gathered.permute(0, 2, 1) * self.left_sign_t.unsqueeze(0)

        right_gathered = right[:, self.cayley_indices]
        right_action = right_gathered.permute(0, 2, 1) * self.gp_sign_t.unsqueeze(0)
        return torch.bmm(right_action, left_action)

    def per_channel(self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Apply one sandwich action per channel in ``values``."""
        validate_channel_values(values, self.layout, left.shape[0], "full sandwich values")
        return self.per_channel_unchecked(left, values, right)

    def per_channel_unchecked(self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Validation-free per-channel action for prepared tensors."""
        matrices = self.action_matrices_unchecked(left, right)
        return torch.einsum("...cj,ckj->...ck", values, matrices)

    def batched(self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Apply one full-layout sandwich action per leading batch item."""
        if values.ndim != 3:
            raise ValueError(f"batched sandwich values must have shape [items, channels, {self.dim}]")
        if values.shape[0] != left.shape[0]:
            raise ValueError(f"values first dimension must be {left.shape[0]}, got {values.shape[0]}")
        if values.shape[-1] != self.dim:
            raise ValueError(f"values last dimension must be {self.dim}, got {values.shape[-1]}")
        return self.batched_unchecked(left, values, right)

    def batched_unchecked(self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Validation-free batched action for prepared tensors."""
        matrices = self.action_matrices_unchecked(left, right)
        return torch.matmul(values, matrices.transpose(-2, -1))

    def multi(self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Apply every full-layout action to every input channel."""
        if values.shape[-1] != self.dim:
            raise ValueError(f"values last dimension must be {self.dim}, got {values.shape[-1]}")
        return self.multi_unchecked(left, values, right)

    def multi_unchecked(self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Validation-free multi-action for prepared tensors."""
        matrices = self.action_matrices_unchecked(left, right)
        return torch.einsum("...cj,kej->...cke", values, matrices)

    def routed(self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor, channel_to_pair: torch.Tensor) -> torch.Tensor:
        """Apply pair actions selected by channel index."""
        return self.routed_unchecked(left, values, right, channel_to_pair)

    def routed_unchecked(
        self,
        left: torch.Tensor,
        values: torch.Tensor,
        right: torch.Tensor,
        channel_to_pair: torch.Tensor,
    ) -> torch.Tensor:
        """Validation-free routed action for prepared tensors."""
        matrices = self.action_matrices_unchecked(left, right)
        channel_matrices = torch.index_select(matrices, 0, channel_to_pair)
        return torch.einsum("...cj,ckj->...ck", values, channel_matrices)

    def _indices_for(self, values: torch.Tensor) -> torch.Tensor:
        return self.cayley_indices

    def _left_signs_for(self, values: torch.Tensor) -> torch.Tensor:
        return self.left_sign_t

    def _gp_signs_for(self, values: torch.Tensor) -> torch.Tensor:
        return self.gp_sign_t


class _VersorFactorPlanMixin:
    """Shared preplanned factor construction for full-layout versor actions."""

    def _configure_versor_factor_plans(self, algebra, *, grade: int, parameter_layout: GradeLayout) -> None:
        self.full_dim = int(algebra.dim)
        self.eps_sq = float(algebra.eps_sq)
        self.rotor_layout = (
            parameter_layout.spec.layout(range(0, parameter_layout.spec.n + 1, 2)) if int(grade) == 2 else None
        )
        device = getattr(algebra, "device", None)
        dtype = getattr(algebra, "dtype", torch.float32)

        self.bivector_exp = None
        self.rotor_reverse = None
        self.parameter_norm_sq = None
        self.parameter_involution = None
        self.parameter_reverse = None
        self.register_buffer("rotor_full_indices", torch.empty(0, dtype=torch.long, device=device), persistent=False)
        self.register_buffer(
            "parameter_full_indices", torch.empty(0, dtype=torch.long, device=device), persistent=False
        )

        if not self.use_full_action and not getattr(self, "use_rotor_product_action", False):
            return
        if int(grade) == 2:
            self.bivector_exp = algebra.plan_exp(
                input_layout=parameter_layout,
                output_layout=self.rotor_layout,
                dtype=dtype,
                device=device,
            )
            self.rotor_reverse = algebra.plan_unary(
                op="reverse",
                input_layout=self.rotor_layout,
                output_layout=self.rotor_layout,
                dtype=dtype,
                device=device,
            )
            if self.use_full_action:
                self.rotor_full_indices = _layout_indices(self.rotor_layout, device=device)
            return

        self.parameter_norm_sq = algebra.plan_norm_sq(input_layout=parameter_layout, dtype=dtype, device=device)
        self.parameter_involution = algebra.plan_unary(
            op="grade_involution",
            input_layout=parameter_layout,
            output_layout=parameter_layout,
            dtype=dtype,
            device=device,
        )
        self.parameter_reverse = algebra.plan_unary(
            op="reverse",
            input_layout=parameter_layout,
            output_layout=parameter_layout,
            dtype=dtype,
            device=device,
        )
        self.parameter_full_indices = _layout_indices(parameter_layout, device=device)

    def _planned_full_versor_factors(self, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.grade == 2:
            rotor, right = self._planned_rotor_factors(weights)
            left_full = _materialize_full_from_indices(rotor, self.rotor_full_indices, self.full_dim)
            right_full = _materialize_full_from_indices(right, self.rotor_full_indices, self.full_dim)
            return left_full, right_full

        norm_sq = self.parameter_norm_sq(weights)
        scale = norm_sq.abs().clamp_min(eps_like(norm_sq)).sqrt()
        versor = weights / scale
        left = self.parameter_involution(versor)
        denominator = signed_clamp_min(self.parameter_norm_sq(versor), self.eps_sq)
        right = self.parameter_reverse(versor) / denominator
        left_full = _materialize_full_from_indices(left, self.parameter_full_indices, self.full_dim)
        right_full = _materialize_full_from_indices(right, self.parameter_full_indices, self.full_dim)
        return left_full, right_full

    def _planned_rotor_factors(self, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rotor = self.bivector_exp(-0.5 * weights)
        return rotor, self.rotor_reverse(rotor)


class VersorActionExecutor(_VersorFactorPlanMixin, nn.Module):
    """Apply one planned grade-1 or grade-2 versor action."""

    def __init__(
        self,
        algebra,
        *,
        grade: int,
        input_layout: GradeLayout,
        output_layout: GradeLayout,
        parameter_layout: GradeLayout,
    ):
        super().__init__()
        object.__setattr__(self, "algebra", algebra)
        self.grade = int(grade)
        self.input_layout = input_layout
        self.output_layout = output_layout
        self.parameter_layout = parameter_layout
        self.use_full_action = input_layout.dim == algebra.dim and output_layout.dim == algebra.dim
        self.use_rotor_product_action = _prefer_rotor_product_action(algebra, grade=self.grade, use_full_action=self.use_full_action)
        self.action = None
        self.vector_matrix = None
        self.left_product = None
        self.right_product = None
        self.middle_layout = None
        self.full_action = (
            FullSandwichActionExecutor.from_layout(
                input_layout,
                device=getattr(algebra, "device", None),
                dtype=getattr(algebra, "dtype", torch.float32),
            )
            if self.use_full_action
            else None
        )
        if self.grade not in {1, 2}:
            raise ValueError("planned versor execution currently supports grade=1 and grade=2")
        if not self.use_full_action and not self.use_rotor_product_action:
            self.action = GradedLinearActionExecutor(input_layout=input_layout, output_layout=output_layout)
            self.vector_matrix = VersorVectorMatrixExecutor(
                grade=self.grade,
                parameter_layout=parameter_layout,
                eps=algebra.eps_sq,
                dtype=getattr(algebra, "dtype", torch.float32),
                device=getattr(algebra, "device", None),
            )
        self._configure_versor_factor_plans(algebra, grade=self.grade, parameter_layout=parameter_layout)
        if self.use_rotor_product_action:
            self._configure_rotor_product_action(algebra)

    def _configure_rotor_product_action(self, algebra) -> None:
        device = getattr(algebra, "device", None)
        dtype = getattr(algebra, "dtype", torch.float32)
        middle_grades = expand_output_grades(self.rotor_layout.grades, self.input_layout.grades, algebra.n, op="gp")
        self.middle_layout = algebra.layout(middle_grades)
        self.left_product = algebra.plan_product(
            op="gp",
            left_layout=self.rotor_layout,
            right_layout=self.input_layout,
            output_layout=self.middle_layout,
            dtype=dtype,
            device=device,
        )
        self.right_product = algebra.plan_product(
            op="gp",
            left_layout=self.middle_layout,
            right_layout=self.rotor_layout,
            output_layout=self.output_layout,
            dtype=dtype,
            device=device,
        )

    def forward(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Return transformed values in ``output_layout`` lanes."""
        validate_channel_values(values, self.input_layout, weights.shape[0], "versor values")
        return self.execute(values, weights)

    def execute(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Validation-free versor action for prepared tensors."""
        if self.use_full_action:
            left, right = self._planned_full_versor_factors(weights)
            return self.full_action.per_channel_unchecked(left, values, right)
        if self.use_rotor_product_action:
            left, right = self._planned_rotor_factors(weights)
            leading_rank = values.ndim - 2
            left = left.reshape(*((1,) * leading_rank), left.shape[0], left.shape[-1])
            right = right.reshape(*((1,) * leading_rank), right.shape[0], right.shape[-1])
            middle = self.left_product(left, values)
            return self.right_product(middle, right)
        matrix = self.vector_matrix.execute(weights)
        return self.action.execute(values, matrix)


class MultiVersorActionExecutor(_VersorFactorPlanMixin, nn.Module):
    """Apply a weighted superposition of planned grade-1 or grade-2 actions."""

    def __init__(
        self,
        algebra,
        *,
        grade: int,
        input_layout: GradeLayout,
        output_layout: GradeLayout,
        parameter_layout: GradeLayout,
    ):
        super().__init__()
        object.__setattr__(self, "algebra", algebra)
        self.grade = int(grade)
        self.input_layout = input_layout
        self.output_layout = output_layout
        self.parameter_layout = parameter_layout
        self.use_full_action = input_layout.dim == algebra.dim and output_layout.dim == algebra.dim
        self.use_rotor_product_action = _prefer_rotor_product_action(algebra, grade=self.grade, use_full_action=self.use_full_action)
        self.action = None
        self.vector_matrix = None
        self.left_product = None
        self.right_product = None
        self.middle_layout = None
        self.full_action = (
            FullSandwichActionExecutor.from_layout(
                input_layout,
                device=getattr(algebra, "device", None),
                dtype=getattr(algebra, "dtype", torch.float32),
            )
            if self.use_full_action
            else None
        )
        if self.grade not in {1, 2}:
            raise ValueError("planned multi-versor execution currently supports grade=1 and grade=2")
        if not self.use_full_action and not self.use_rotor_product_action:
            self.action = GradedLinearActionExecutor(input_layout=input_layout, output_layout=output_layout)
            self.vector_matrix = VersorVectorMatrixExecutor(
                grade=self.grade,
                parameter_layout=parameter_layout,
                eps=algebra.eps_sq,
                dtype=getattr(algebra, "dtype", torch.float32),
                device=getattr(algebra, "device", None),
            )
        self._configure_versor_factor_plans(algebra, grade=self.grade, parameter_layout=parameter_layout)
        if self.use_rotor_product_action:
            self._configure_rotor_product_action(algebra)

    def _configure_rotor_product_action(self, algebra) -> None:
        device = getattr(algebra, "device", None)
        dtype = getattr(algebra, "dtype", torch.float32)
        middle_grades = expand_output_grades(self.rotor_layout.grades, self.input_layout.grades, algebra.n, op="gp")
        self.middle_layout = algebra.layout(middle_grades)
        self.left_product = algebra.plan_product(
            op="gp",
            left_layout=self.rotor_layout,
            right_layout=self.input_layout,
            output_layout=self.middle_layout,
            dtype=dtype,
            device=device,
        )
        self.right_product = algebra.plan_product(
            op="gp",
            left_layout=self.middle_layout,
            right_layout=self.rotor_layout,
            output_layout=self.output_layout,
            dtype=dtype,
            device=device,
        )

    def forward(self, values: torch.Tensor, weights: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
        """Return transformed values in ``output_layout`` lanes."""
        validate_channel_values(values, self.input_layout, mix.shape[0], "multi-versor values")
        if not self.use_full_action and mix.shape != (values.shape[-2], weights.shape[0]):
            raise ValueError(f"mix shape must be {(values.shape[-2], weights.shape[0])}, got {tuple(mix.shape)}")
        return self.execute(values, weights, mix)

    def execute(self, values: torch.Tensor, weights: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
        """Validation-free multi-versor action for prepared tensors."""
        if self.use_full_action:
            left, right = self._planned_full_versor_factors(weights)
            transformed = self.full_action.multi_unchecked(left, values, right)
            return torch.einsum("ck,...cke->...ce", mix, transformed)
        if self.use_rotor_product_action:
            left, right = self._planned_rotor_factors(weights)
            leading_rank = values.ndim - 2
            left = left.reshape(*((1,) * leading_rank), left.shape[0], 1, left.shape[-1])
            right = right.reshape(*((1,) * leading_rank), right.shape[0], 1, right.shape[-1])
            expanded_values = values.unsqueeze(-3)
            middle = self.left_product(left, expanded_values)
            transformed = self.right_product(middle, right).transpose(-3, -2)
            return torch.einsum("ck,...cko->...co", mix, transformed)
        matrices = self.vector_matrix.execute(weights)
        transformed = self.action.multi_execute(values, matrices)
        return torch.einsum("ck,...cko->...co", mix, transformed)


class PairedBivectorActionExecutor(nn.Module):
    """Apply independent left/right bivector rotor pairs to channel values."""

    def __init__(
        self,
        algebra,
        *,
        input_layout: GradeLayout,
        output_layout: GradeLayout,
        parameter_layout: GradeLayout,
        rotor_layout: GradeLayout,
        middle_layout: GradeLayout,
    ):
        super().__init__()
        if (
            input_layout.spec != output_layout.spec
            or input_layout.spec != parameter_layout.spec
            or input_layout.spec != rotor_layout.spec
            or input_layout.spec != middle_layout.spec
        ):
            raise ValueError("paired bivector action layouts must share one algebra spec")
        if parameter_layout.grades != (2,):
            raise ValueError(f"parameter_layout must contain grade 2, got {parameter_layout.grades}")
        object.__setattr__(self, "algebra", algebra)
        self.input_layout = input_layout
        self.output_layout = output_layout
        self.parameter_layout = parameter_layout
        self.rotor_layout = rotor_layout
        self.middle_layout = middle_layout
        self.use_full_action = input_layout.dim == algebra.dim and output_layout.dim == algebra.dim
        self.full_dim = int(algebra.dim)
        self.full_action = (
            FullSandwichActionExecutor.from_layout(
                input_layout,
                device=getattr(algebra, "device", None),
                dtype=getattr(algebra, "dtype", torch.float32),
            )
            if self.use_full_action
            else None
        )
        device = getattr(algebra, "device", None)
        dtype = getattr(algebra, "dtype", torch.float32)
        self.bivector_exp = algebra.plan_exp(
            input_layout=parameter_layout,
            output_layout=rotor_layout,
            dtype=dtype,
            device=device,
        )
        self.rotor_reverse = algebra.plan_unary(
            op="reverse",
            input_layout=rotor_layout,
            output_layout=rotor_layout,
            dtype=dtype,
            device=device,
        )
        self.register_buffer("rotor_full_indices", _layout_indices(rotor_layout, device=device), persistent=False)
        self.left_product = None
        self.right_product = None
        if not self.use_full_action:
            self.left_product = algebra.plan_product(
                op="gp",
                left_layout=rotor_layout,
                right_layout=input_layout,
                output_layout=middle_layout,
                dtype=dtype,
                device=device,
            )
            self.right_product = algebra.plan_product(
                op="gp",
                left_layout=middle_layout,
                right_layout=rotor_layout,
                output_layout=output_layout,
                dtype=dtype,
                device=device,
            )

    def forward(
        self,
        values: torch.Tensor,
        left_weights: torch.Tensor,
        right_weights: torch.Tensor,
        channel_to_pair: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``R_left x R_right_reverse`` for each routed input channel."""
        validate_channel_values(values, self.input_layout, channel_to_pair.shape[0], "paired bivector values")
        if left_weights.shape != right_weights.shape:
            raise ValueError(
                f"left and right weights must have matching shapes, got {tuple(left_weights.shape)} "
                f"and {tuple(right_weights.shape)}"
            )
        if left_weights.ndim != 2 or left_weights.shape[-1] != self.parameter_layout.dim:
            raise ValueError(
                f"bivector weights must have shape [pairs, {self.parameter_layout.dim}], "
                f"got {tuple(left_weights.shape)}"
            )

        return self.execute(values, left_weights, right_weights, channel_to_pair)

    def execute(
        self,
        values: torch.Tensor,
        left_weights: torch.Tensor,
        right_weights: torch.Tensor,
        channel_to_pair: torch.Tensor,
    ) -> torch.Tensor:
        """Validation-free paired-bivector action for prepared tensors."""
        left, right = self._planned_paired_factors(left_weights, right_weights)

        if self.use_full_action:
            left = _materialize_full_from_indices(left, self.rotor_full_indices, self.full_dim)
            right = _materialize_full_from_indices(right, self.rotor_full_indices, self.full_dim)
            return self.full_action.routed_unchecked(left, values, right, channel_to_pair)

        left_by_channel = torch.index_select(left, 0, channel_to_pair)
        right_by_channel = torch.index_select(right, 0, channel_to_pair)

        middle = self.left_product(left_by_channel, values)
        return self.right_product(middle, right_by_channel)

    def _planned_paired_factors(
        self,
        left_weights: torch.Tensor,
        right_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_rotor = self.bivector_exp(-0.5 * left_weights)
        right_rotor = self.bivector_exp(-0.5 * right_weights)
        return left_rotor, self.rotor_reverse(right_rotor)


def apply_graded_linear_action(
    values: torch.Tensor,
    matrix: torch.Tensor,
    *,
    input_layout: GradeLayout,
    output_layout: GradeLayout,
) -> torch.Tensor:
    """Apply the outermorphism induced by a vector-space linear action."""
    return GradedLinearActionExecutor(input_layout=input_layout, output_layout=output_layout)(values, matrix)


def apply_multi_graded_linear_action(
    values: torch.Tensor,
    matrices: torch.Tensor,
    *,
    input_layout: GradeLayout,
    output_layout: GradeLayout,
) -> torch.Tensor:
    """Apply multiple outermorphisms to declared grade lanes."""
    return GradedLinearActionExecutor(input_layout=input_layout, output_layout=output_layout).multi(values, matrices)


def versor_vector_matrix(algebra, weights: torch.Tensor, *, grade: int, parameter_layout: GradeLayout) -> torch.Tensor:
    """Return the vector-space matrix represented by grade-1 or grade-2 weights."""
    return VersorVectorMatrixExecutor(
        grade=grade,
        parameter_layout=parameter_layout,
        eps=algebra.eps_sq,
        dtype=weights.dtype,
        device=weights.device,
    )(weights)


def bivector_vector_generator(bivectors: torch.Tensor, *, bivector_layout: GradeLayout) -> torch.Tensor:
    """Return the vector-space generator induced by bivectors."""
    return BivectorVectorGeneratorExecutor(
        bivector_layout=bivector_layout,
        dtype=bivectors.dtype,
        device=bivectors.device,
    )(bivectors)


def reflection_vector_matrix(normals: torch.Tensor, *, vector_layout: GradeLayout, eps: float) -> torch.Tensor:
    """Return the vector-space reflection matrix for normal vectors."""
    if vector_layout.grades != (1,):
        raise ValueError(f"vector_layout must contain grade 1 only, got {vector_layout.grades}")
    if normals.shape[-1] != vector_layout.dim:
        raise ValueError(f"normal dimension must be {vector_layout.dim}, got {normals.shape[-1]}")

    signs = metric_self_signs(vector_layout, device=normals.device, dtype=normals.dtype)
    denominator = (normals * normals * signs).sum(dim=-1, keepdim=True)
    denominator = signed_clamp_min(denominator, eps)
    weighted_normals = normals * signs
    outer = normals.unsqueeze(-1) * weighted_normals.unsqueeze(-2)
    eye = torch.eye(vector_layout.spec.n, device=normals.device, dtype=normals.dtype)
    return eye - 2.0 * outer / denominator.unsqueeze(-1)


def _graded_action_coefficients(
    matrix: torch.Tensor,
    *,
    input_layout: GradeLayout,
    output_layout: GradeLayout,
) -> torch.Tensor:
    coefficients = matrix.new_zeros(matrix.shape[0], output_layout.dim, input_layout.dim)
    input_by_grade = _positions_by_grade(input_layout)

    for output_position, output_index in enumerate(output_layout.basis_indices):
        grade = output_index.bit_count()
        input_positions = input_by_grade.get(grade, ())
        if not input_positions:
            continue

        if grade == 0:
            for input_position in input_positions:
                if input_layout.basis_indices[input_position] == 0:
                    coefficients[:, output_position, input_position] = 1.0
            continue

        output_bits = _basis_bits(output_index, input_layout.spec.n, device=matrix.device)
        for input_position in input_positions:
            input_index = input_layout.basis_indices[input_position]
            input_bits = _basis_bits(input_index, input_layout.spec.n, device=matrix.device)
            submatrix = torch.index_select(matrix, -2, output_bits)
            submatrix = torch.index_select(submatrix, -1, input_bits)
            coefficients[:, output_position, input_position] = torch.linalg.det(submatrix)

    return coefficients


def _positions_by_grade(layout: GradeLayout) -> dict[int, tuple[int, ...]]:
    positions: dict[int, list[int]] = {}
    for position, index in enumerate(layout.basis_indices):
        positions.setdefault(index.bit_count(), []).append(position)
    return {grade: tuple(values) for grade, values in positions.items()}


def _basis_bits(index: int, n: int, *, device=None) -> torch.Tensor:
    bits = [bit for bit in range(n) if index & (1 << bit)]
    return torch.tensor(bits, dtype=torch.long, device=device)


def _basis_bits_tuple(index: int, n: int) -> tuple[int, ...]:
    return tuple(bit for bit in range(n) if index & (1 << bit))


def _scalar_action_positions(input_layout: GradeLayout, output_layout: GradeLayout) -> torch.Tensor:
    positions: list[int] = []
    for output_position, output_index in enumerate(output_layout.basis_indices):
        if output_index != 0:
            continue
        for input_position, input_index in enumerate(input_layout.basis_indices):
            if input_index == 0:
                positions.append(output_position * input_layout.dim + input_position)
    return torch.tensor(positions, dtype=torch.long)


def _graded_action_plan_tensors(
    input_layout: GradeLayout,
    output_layout: GradeLayout,
    *,
    grade: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_positions: list[int] = []
    row_indices: list[tuple[int, ...]] = []
    col_indices: list[tuple[int, ...]] = []
    input_items = [
        (input_position, input_index)
        for input_position, input_index in enumerate(input_layout.basis_indices)
        if input_index.bit_count() == grade
    ]
    for output_position, output_index in enumerate(output_layout.basis_indices):
        if output_index.bit_count() != grade:
            continue
        output_bits = _basis_bits_tuple(output_index, input_layout.spec.n)
        for input_position, input_index in input_items:
            flat_positions.append(output_position * input_layout.dim + input_position)
            row_indices.append(output_bits)
            col_indices.append(_basis_bits_tuple(input_index, input_layout.spec.n))

    if not flat_positions:
        empty = torch.empty(0, dtype=torch.long)
        return empty, torch.empty(0, grade, dtype=torch.long), torch.empty(0, grade, dtype=torch.long)
    return (
        torch.tensor(flat_positions, dtype=torch.long),
        torch.tensor(row_indices, dtype=torch.long),
        torch.tensor(col_indices, dtype=torch.long),
    )


def _prefer_rotor_product_action(algebra, *, grade: int, use_full_action: bool) -> bool:
    return int(grade) == 2 and not use_full_action


def _layout_indices(layout: GradeLayout, *, device=None) -> torch.Tensor:
    return torch.tensor(layout.basis_indices, dtype=torch.long, device=device)


def _materialize_full_from_indices(values: torch.Tensor, indices: torch.Tensor, dim: int) -> torch.Tensor:
    output = values.new_zeros(*values.shape[:-1], dim)
    return output.index_copy(-1, indices, values)


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
        right = algebra.reverse(rotor, input_layout=rotor_layout, output_layout=rotor_layout)
        return materialize_full(algebra, rotor, layout=rotor_layout), materialize_full(
            algebra,
            right,
            layout=rotor_layout,
        )

    if grade == 1:
        norm_sq = algebra.norm_sq(weights, input_layout=parameter_layout)
        scale = norm_sq.abs().clamp_min(eps_like(norm_sq)).sqrt()
        versor = weights / scale
    else:
        norm = weights.norm(dim=-1, keepdim=True).clamp_min(eps_like(weights))
        versor = weights / norm

    left = algebra.grade_involution(versor, input_layout=parameter_layout, output_layout=parameter_layout)
    right = algebra.blade_inverse(versor, input_layout=parameter_layout)
    return materialize_full(algebra, left, layout=parameter_layout), materialize_full(
        algebra,
        right,
        layout=parameter_layout,
    )


def full_paired_bivector_factors(
    algebra,
    left_weights: torch.Tensor,
    right_weights: torch.Tensor,
    *,
    parameter_layout: GradeLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full-lane ``(R_left, reverse(R_right))`` for independent bivectors."""
    rotor_layout = parameter_layout.spec.layout(range(0, parameter_layout.spec.n + 1, 2))
    left_rotor = _bivector_exp(
        algebra,
        -0.5 * left_weights,
        parameter_layout=parameter_layout,
        rotor_layout=rotor_layout,
    )
    right_rotor = _bivector_exp(
        algebra,
        -0.5 * right_weights,
        parameter_layout=parameter_layout,
        rotor_layout=rotor_layout,
    )
    right_reverse = algebra.reverse(right_rotor, input_layout=rotor_layout, output_layout=rotor_layout)
    return materialize_full(algebra, left_rotor, layout=rotor_layout), materialize_full(
        algebra,
        right_reverse,
        layout=rotor_layout,
    )


def paired_bivector_factors(
    algebra,
    left_weights: torch.Tensor,
    right_weights: torch.Tensor,
    *,
    parameter_layout: GradeLayout,
    rotor_layout: GradeLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return compact ``(R_left, reverse(R_right))`` for paired bivector actions."""
    left_rotor = _bivector_exp(
        algebra,
        -0.5 * left_weights,
        parameter_layout=parameter_layout,
        rotor_layout=rotor_layout,
    )
    right_rotor = _bivector_exp(
        algebra,
        -0.5 * right_weights,
        parameter_layout=parameter_layout,
        rotor_layout=rotor_layout,
    )
    right_reverse = algebra.reverse(right_rotor, input_layout=rotor_layout, output_layout=rotor_layout)
    return left_rotor, right_reverse


def _bivector_exp(
    algebra,
    values: torch.Tensor,
    *,
    parameter_layout: GradeLayout,
    rotor_layout: GradeLayout,
) -> torch.Tensor:
    return algebra.exp(values, input_layout=parameter_layout, output_layout=rotor_layout)
