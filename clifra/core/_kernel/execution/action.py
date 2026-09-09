# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Compile-friendly executors for planned linear and versor actions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from clifra.core._kernel.basis import operation_coefficient
from clifra.core._kernel.contracts import _check_contract_spec, resolve_contract
from clifra.core._kernel.numerics import eps_like, signed_clamp_min
from clifra.core._kernel.planning.action import (
    _graded_action_plan_tensors,
    _scalar_action_positions,
    build_bivector_vector_generator_buffers,
    build_full_sandwich_action_buffers,
)
from clifra.core.layout import GradeLayout
from clifra.core.tensors import TensorContract


class GradedLinearActionExecutor(nn.Module):
    """Apply a vector-space map lifted to declared multivector grades."""

    def __init__(self, *, input_layout: GradeLayout, output_layout: GradeLayout):
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
        self.register_buffer(
            "scalar_flat_positions", _scalar_action_positions(input_layout, output_layout), persistent=False
        )
        self._vector_only = input_layout.grades == output_layout.grades == (1,)
        self._grades = tuple(grade for grade in input_layout.grades if grade > 0 and grade in output_layout.grades)
        for grade in () if self._vector_only else self._grades:
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
        if self._vector_only:
            return matrices
        flat = matrices.new_zeros(matrices.shape[0], self.output_dim * self.input_dim)
        scalar_positions = self.scalar_flat_positions
        if scalar_positions.numel() > 0:
            scalar_values = matrices.new_ones(matrices.shape[0], scalar_positions.numel())
            flat = flat.index_copy(-1, scalar_positions, scalar_values)

        if 1 in self._grades:
            row_indices = self.row_indices_1[:, 0]
            col_indices = self.col_indices_1[:, 0]
            coefficients = matrices[:, row_indices, col_indices]
            flat = flat.index_copy(-1, self.flat_positions_1, coefficients)

        for grade in self._grades:
            if grade == 1:
                continue
            positions = getattr(self, f"flat_positions_{grade}")
            row_indices = getattr(self, f"row_indices_{grade}")
            col_indices = getattr(self, f"col_indices_{grade}")
            # Keep tiny CPU determinant batches on LU; many small minors and MPS
            # benefit from direct arithmetic without materializing submatrices.
            if grade in (2, 3) and (positions.numel() >= 128 or matrices.device.type == "mps"):
                coefficients = _small_action_minors(matrices, row_indices, col_indices, grade)
            else:
                submatrix = matrices[:, row_indices.unsqueeze(-1), col_indices.unsqueeze(-2)]
                coefficients = torch.linalg.det(submatrix)
            flat = flat.index_copy(-1, positions, coefficients)

        return flat.reshape(matrices.shape[0], self.output_dim, self.input_dim)

    def _check_values(self, values: torch.Tensor) -> None:
        if values.ndim < 2:
            raise ValueError(f"values must include channel and lane axes, got shape {tuple(values.shape)}")
        self.input_contract.validate(values, name="values")


def _small_action_minors(matrices: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, grade: int) -> torch.Tensor:
    """Evaluate requested 2x2 or 3x3 minors directly from a vector-space map."""
    a = matrices[:, rows[:, 0], cols[:, 0]]
    b = matrices[:, rows[:, 0], cols[:, 1]]
    d = matrices[:, rows[:, 1], cols[:, 0]]
    e = matrices[:, rows[:, 1], cols[:, 1]]
    if grade == 2:
        return a * e - b * d
    c = matrices[:, rows[:, 0], cols[:, 2]]
    f = matrices[:, rows[:, 1], cols[:, 2]]
    g = matrices[:, rows[:, 2], cols[:, 0]]
    h = matrices[:, rows[:, 2], cols[:, 1]]
    i = matrices[:, rows[:, 2], cols[:, 2]]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


class BivectorVectorGeneratorExecutor(nn.Module):
    """Build vector-space generators induced by grade-2 bivectors."""

    def __init__(self, *, bivector_layout: GradeLayout, dtype: torch.dtype = torch.float32, device=None):
        super().__init__()
        self.bivector_contract = TensorContract.compact(bivector_layout)
        bivector_layout = self.bivector_contract.layout
        if bivector_layout.grades != (2,):
            raise ValueError(f"bivector_layout must contain grade 2 only, got {bivector_layout.grades}")
        self.bivector_layout = bivector_layout
        self.n = bivector_layout.spec.n
        lane_positions, flat_positions, coefficients = build_bivector_vector_generator_buffers(
            bivector_layout, dtype=dtype, device=device
        )
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
        self.parameter_contract = TensorContract.compact(parameter_layout)
        self.parameter_layout = self.parameter_contract.layout
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
                    "geometric_product",
                )
                for index in parameter_layout.basis_indices
            ]
            self.register_buffer("metric_signs", torch.tensor(signs, dtype=dtype, device=device), persistent=False)
            self.register_buffer("eye", torch.eye(self.n, dtype=dtype, device=device), persistent=False)
        else:
            raise ValueError("planned versor execution currently supports grade=1 and grade=2")

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """Return one vector-space action matrix per input weight row."""
        self.parameter_contract.validate(weights, name="weights")
        return self.execute(weights)

    def execute(self, weights: torch.Tensor) -> torch.Tensor:
        """Validation-free vector-space action matrices for prepared weights."""
        if self.grade == 2:
            return torch.matrix_exp(self.generator.execute(weights))
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

    @classmethod
    def from_layout(cls, layout: GradeLayout, *, device=None, dtype: torch.dtype = torch.float32):
        """Build full-layout sandwich action buffers from algebra metadata."""
        cayley_indices, left_sign_t, geometric_product_sign_t = build_full_sandwich_action_buffers(
            layout, device=device, dtype=dtype
        )
        return cls(
            layout=layout,
            cayley_indices=cayley_indices,
            left_sign_t=left_sign_t,
            geometric_product_sign_t=geometric_product_sign_t,
        )

    def action_matrices(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return matrices ``M[..., k, j]`` such that ``output[..., k] = M @ x``."""
        self._check_factors(left, right)
        return self.action_matrices_unchecked(left, right)

    def action_matrices_unchecked(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Validation-free action matrices for prepared full-layout factors."""
        if self.dim >= 256 and left.device.type == "cpu" and left.dtype == right.dtype == torch.float32:
            # XOR addressing is symmetric. Gather directly in the transposed
            # action's order without another table or a strided intermediate.
            indices = self.cayley_indices.reshape(-1)
            left_action = left.index_select(-1, indices).reshape(left.shape[0], self.dim, self.dim) * self.left_sign_t
            right_action = (
                right.index_select(-1, indices).reshape(right.shape[0], self.dim, self.dim)
                * self.geometric_product_sign_t
            )
            return torch.bmm(right_action, left_action)
        left_gathered = left[:, self.cayley_indices]
        left_action = left_gathered.permute(0, 2, 1) * self.left_sign_t.unsqueeze(0)

        right_gathered = right[:, self.cayley_indices]
        right_action = right_gathered.permute(0, 2, 1) * self.geometric_product_sign_t.unsqueeze(0)
        return torch.bmm(right_action, left_action)

    def per_channel(self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Apply one sandwich action per channel in ``values``."""
        self._check_factors(left, right)
        _validate_channels(self.contract, values, channels=left.shape[0], name="full sandwich values")
        return self.per_channel_unchecked(left, values, right)

    def per_channel_unchecked(self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Validation-free per-channel action for prepared tensors."""
        matrices = self.action_matrices_unchecked(left, right)
        return torch.einsum("...cj,ckj->...ck", values, matrices)

    def batched(self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Apply one full-layout sandwich action per leading batch item."""
        self._check_factors(left, right)
        if values.ndim != 3:
            raise ValueError(f"batched sandwich values must have shape [items, channels, {self.dim}]")
        if values.shape[0] != left.shape[0]:
            raise ValueError(f"values first dimension must be {left.shape[0]}, got {values.shape[0]}")
        self.contract.validate(values, name="batched sandwich values")
        return self.batched_unchecked(left, values, right)

    def batched_unchecked(self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Validation-free batched action for prepared tensors."""
        matrices = self.action_matrices_unchecked(left, right)
        return torch.matmul(values, matrices.transpose(-2, -1))

    def multi(self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Apply every full-layout action to every input channel."""
        self._check_factors(left, right)
        self.contract.validate(values, name="multi sandwich values")
        return self.multi_unchecked(left, values, right)

    def multi_unchecked(self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Validation-free multi-action for prepared tensors."""
        matrices = self.action_matrices_unchecked(left, right)
        return torch.einsum("...cj,kej->...cke", values, matrices)

    def routed(
        self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor, channel_to_pair: torch.Tensor
    ) -> torch.Tensor:
        """Apply pair actions selected by channel index."""
        self._check_factors(left, right)
        _validate_channels(self.contract, values, channels=channel_to_pair.numel(), name="routed sandwich values")
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

    def _check_factors(self, left: torch.Tensor, right: torch.Tensor) -> None:
        if left.ndim != 2 or right.ndim != 2:
            raise ValueError(f"left and right factors must have shape [items, {self.dim}]")
        if left.shape != right.shape:
            raise ValueError(f"left and right factors must have matching shape [items, {self.dim}]")
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


class _VersorActionExecutor(nn.Module):
    """Shared execution setup for single and mixed versor actions."""

    action_name = "versor"

    def __init__(
        self,
        algebra,
        *,
        grade: int,
        input_layout: GradeLayout,
        output_layout: GradeLayout,
        parameter_layout: GradeLayout,
        route: str,
        components: ActionComponents,
    ):
        super().__init__()
        self.input_contract = resolve_contract(algebra, layout=input_layout, name="input_layout")
        self.output_contract = resolve_contract(algebra, layout=output_layout, name="output_layout")
        self.parameter_contract = resolve_contract(algebra, layout=parameter_layout, name="parameter_layout")
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
        self.action = None
        self.vector_matrix = None
        self.left_product = None
        self.right_product = None
        self.middle_layout = None
        self.full_action = (
            FullSandwichActionExecutor.from_layout(
                self.input_layout,
                device=getattr(algebra, "device", None),
                dtype=getattr(algebra, "dtype", torch.float32),
            )
            if self.use_full_action
            else None
        )
        if not self.use_full_action and not self.use_rotor_product_action:
            self.action = GradedLinearActionExecutor(
                input_layout=self.input_layout,
                output_layout=self.output_layout,
            )
            self.vector_matrix = VersorVectorMatrixExecutor(
                grade=self.grade,
                parameter_layout=self.parameter_layout,
                eps=algebra.eps_sq,
                dtype=getattr(algebra, "dtype", torch.float32),
                device=getattr(algebra, "device", None),
            )
        self._configure_components(algebra, components)

    def _configure_components(self, algebra, components):
        self.full_dim = int(algebra.dim)
        self.eps_sq = float(algebra.eps_sq)
        self.rotor_layout = components.rotor_layout
        self.middle_layout = components.middle_layout
        self.bivector_exp = components.exponential
        self.rotor_reverse = components.reverse if self.grade == 2 else None
        self.parameter_reverse = components.reverse if self.grade == 1 else None
        self.parameter_signature_norm_squared = components.norm
        self.parameter_involution = components.involution
        self.left_product = components.left_product
        self.right_product = components.right_product
        self.register_buffer(
            "rotor_full_indices",
            _layout_indices(self.rotor_layout, device=algebra.device)
            if self.rotor_layout is not None
            else torch.empty(0, dtype=torch.long, device=algebra.device),
            persistent=False,
        )
        self.register_buffer(
            "parameter_full_indices",
            _layout_indices(self.parameter_layout, device=algebra.device),
            persistent=False,
        )

    def _planned_full_versor_factors(self, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.grade == 2:
            rotor, right = self._planned_rotor_factors(weights)
            left_full = _materialize_full_from_indices(rotor, self.rotor_full_indices, self.full_dim)
            right_full = _materialize_full_from_indices(right, self.rotor_full_indices, self.full_dim)
            return left_full, right_full

        signature_norm_squared = self.parameter_signature_norm_squared(weights)
        scale = signature_norm_squared.abs().clamp_min(eps_like(signature_norm_squared)).sqrt()
        versor = weights / scale
        left = self.parameter_involution(versor)
        denominator = signed_clamp_min(self.parameter_signature_norm_squared(versor), self.eps_sq)
        right = self.parameter_reverse(versor) / denominator
        left_full = _materialize_full_from_indices(left, self.parameter_full_indices, self.full_dim)
        right_full = _materialize_full_from_indices(right, self.parameter_full_indices, self.full_dim)
        return left_full, right_full

    def _planned_rotor_factors(self, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rotor = self.bivector_exp(-0.5 * weights)
        return rotor, self.rotor_reverse(rotor)


class VersorActionExecutor(_VersorActionExecutor):
    """Apply one planned grade-1 or grade-2 versor action."""

    def forward(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Return transformed values in ``output_layout`` lanes."""
        _validate_channels(self.input_contract, values, channels=weights.shape[0], name="versor values")
        self.parameter_contract.validate(weights, name="versor weights")
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


class MultiVersorActionExecutor(_VersorActionExecutor):
    """Apply a weighted superposition of planned grade-1 or grade-2 actions."""

    action_name = "multi-versor"

    def forward(self, values: torch.Tensor, weights: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
        """Return transformed values in ``output_layout`` lanes."""
        _validate_channels(self.input_contract, values, channels=mix.shape[0], name="multi-versor values")
        self.parameter_contract.validate(weights, name="multi-versor weights")
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
        route: str,
        components: ActionComponents,
    ):
        super().__init__()
        self.input_contract = resolve_contract(algebra, layout=input_layout, name="input_layout")
        self.output_contract = resolve_contract(algebra, layout=output_layout, name="output_layout")
        self.parameter_contract = resolve_contract(algebra, layout=parameter_layout, name="parameter_layout")
        self.rotor_contract = resolve_contract(algebra, layout=rotor_layout, name="rotor_layout")
        self.middle_contract = resolve_contract(algebra, layout=middle_layout, name="middle_layout")
        input_layout = self.input_contract.layout
        output_layout = self.output_contract.layout
        parameter_layout = self.parameter_contract.layout
        rotor_layout = self.rotor_contract.layout
        middle_layout = self.middle_contract.layout
        if parameter_layout.grades != (2,):
            raise ValueError(f"parameter_layout must contain grade 2, got {parameter_layout.grades}")
        self.input_layout = input_layout
        self.output_layout = output_layout
        self.parameter_layout = parameter_layout
        self.rotor_layout = rotor_layout
        self.middle_layout = middle_layout
        self.route = str(route)
        if self.route not in {"full_action_matrix", "paired_rotor_product"}:
            raise ValueError(f"unsupported paired action execution path {self.route!r}")
        self.use_full_action = self.route == "full_action_matrix"
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
        getattr(algebra, "dtype", torch.float32)
        self.bivector_exp = components.exponential
        self.rotor_reverse = components.reverse
        self.register_buffer("rotor_full_indices", _layout_indices(rotor_layout, device=device), persistent=False)
        self.left_product = components.left_product
        self.right_product = components.right_product

    def forward(
        self,
        values: torch.Tensor,
        left_weights: torch.Tensor,
        right_weights: torch.Tensor,
        channel_to_pair: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``R_left x R_right_reverse`` for each routed input channel."""
        _validate_channels(
            self.input_contract,
            values,
            channels=channel_to_pair.shape[0],
            name="paired bivector values",
        )
        if left_weights.shape != right_weights.shape:
            raise ValueError(
                f"left and right weights must have matching shapes, got {tuple(left_weights.shape)} "
                f"and {tuple(right_weights.shape)}"
            )
        if left_weights.ndim != 2:
            raise ValueError(
                f"bivector weights must have shape [pairs, {self.parameter_layout.dim}], "
                f"got {tuple(left_weights.shape)}"
            )
        self.parameter_contract.validate(left_weights, name="left bivector weights")
        self.parameter_contract.validate(right_weights, name="right bivector weights")

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


def _layout_indices(layout: GradeLayout, *, device=None) -> torch.Tensor:
    return torch.tensor(layout.basis_indices, dtype=torch.long, device=device)


def _materialize_full_from_indices(values: torch.Tensor, indices: torch.Tensor, dim: int) -> torch.Tensor:
    output = values.new_zeros(*values.shape[:-1], dim)
    return output.index_copy(-1, indices, values)


def _validate_channels(contract, values, *, channels, name):
    if values.ndim < 3:
        raise ValueError(f"{name}: expected ndim >= 3, got shape {tuple(values.shape)}")
    if values.shape[-2] != channels:
        raise ValueError(f"{name}: expected {channels} channels, got {values.shape[-2]} (shape {tuple(values.shape)})")
    contract.validate(values, name=name)
