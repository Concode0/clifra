"""Compile-friendly executors for planned linear and versor actions."""

from __future__ import annotations

import torch
import torch.nn as nn

from clifra.core.foundation.basis import operation_coefficient
from clifra.core.foundation.layout import GradeLayout
from clifra.core.foundation.numerics import eps_like, signed_clamp_min
from clifra.core.storage import materialize_full, metric_self_signs


class GradedLinearActionExecutor(nn.Module):
    """Apply a vector-space map lifted to declared multivector grades."""

    def __init__(self, *, input_layout: GradeLayout, output_layout: GradeLayout):
        super().__init__()
        if input_layout.spec != output_layout.spec:
            raise ValueError(f"layout mismatch: {input_layout.spec} vs {output_layout.spec}")
        self.input_layout = input_layout
        self.output_layout = output_layout

    def forward(self, values: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """Return output lanes in ``output_layout``."""
        return apply_graded_linear_action(
            values,
            matrix,
            input_layout=self.input_layout,
            output_layout=self.output_layout,
        )


class VersorActionExecutor(nn.Module):
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
        self.algebra = algebra
        self.grade = int(grade)
        self.input_layout = input_layout
        self.output_layout = output_layout
        self.parameter_layout = parameter_layout
        self.action = GradedLinearActionExecutor(input_layout=input_layout, output_layout=output_layout)
        if self.grade not in {1, 2}:
            raise ValueError("planned versor execution currently supports grade=1 and grade=2")

    def forward(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Return transformed values in ``output_layout`` lanes."""
        _check_channel_values(values, self.input_layout, weights.shape[0], "versor values")
        if (
            self.input_layout.dim == self.algebra.dim
            and self.output_layout.dim == self.algebra.dim
            and hasattr(self.algebra, "per_channel_sandwich")
        ):
            left, right = dense_versor_factors(
                self.algebra,
                weights.to(device=values.device, dtype=values.dtype),
                grade=self.grade,
                parameter_layout=self.parameter_layout,
            )
            return self.algebra.per_channel_sandwich(left, values, right)
        matrix = versor_vector_matrix(
            self.algebra,
            weights.to(device=values.device, dtype=values.dtype),
            grade=self.grade,
            parameter_layout=self.parameter_layout,
        )
        return self.action(values, matrix)


class MultiVersorActionExecutor(nn.Module):
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
        self.algebra = algebra
        self.grade = int(grade)
        self.input_layout = input_layout
        self.output_layout = output_layout
        self.parameter_layout = parameter_layout
        if self.grade not in {1, 2}:
            raise ValueError("planned multi-versor execution currently supports grade=1 and grade=2")

    def forward(self, values: torch.Tensor, weights: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
        """Return transformed values in ``output_layout`` lanes."""
        _check_channel_values(values, self.input_layout, mix.shape[0], "multi-versor values")
        if (
            self.input_layout.dim == self.algebra.dim
            and self.output_layout.dim == self.algebra.dim
            and hasattr(self.algebra, "multi_rotor_sandwich")
        ):
            left, right = dense_versor_factors(
                self.algebra,
                weights.to(device=values.device, dtype=values.dtype),
                grade=self.grade,
                parameter_layout=self.parameter_layout,
            )
            transformed = self.algebra.multi_rotor_sandwich(left, values, right)
            mix = mix.to(device=values.device, dtype=values.dtype)
            return torch.einsum("ck,...cke->...ce", mix, transformed)
        matrices = versor_vector_matrix(
            self.algebra,
            weights.to(device=values.device, dtype=values.dtype),
            grade=self.grade,
            parameter_layout=self.parameter_layout,
        )
        mix = mix.to(device=values.device, dtype=values.dtype)
        if mix.shape != (values.shape[-2], matrices.shape[0]):
            raise ValueError(f"mix shape must be {(values.shape[-2], matrices.shape[0])}, got {tuple(mix.shape)}")
        transformed = apply_multi_graded_linear_action(
            values,
            matrices,
            input_layout=self.input_layout,
            output_layout=self.output_layout,
        )
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

    def forward(
        self,
        values: torch.Tensor,
        left_weights: torch.Tensor,
        right_weights: torch.Tensor,
        channel_to_pair: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``R_left x R_right_reverse`` for each routed input channel."""
        _check_channel_values(values, self.input_layout, channel_to_pair.shape[0], "paired bivector values")
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

        if (
            self.input_layout.dim == self.algebra.dim
            and self.output_layout.dim == self.algebra.dim
            and hasattr(self.algebra, "sandwich_action_matrices")
            and hasattr(self.algebra, "exp")
        ):
            left, right = dense_paired_bivector_factors(
                self.algebra,
                left_weights.to(device=values.device, dtype=values.dtype),
                right_weights.to(device=values.device, dtype=values.dtype),
                parameter_layout=self.parameter_layout,
            )
            pair_index = channel_to_pair.to(device=values.device)
            matrices = self.algebra.sandwich_action_matrices(left, right)
            channel_matrices = torch.index_select(matrices, 0, pair_index)
            return torch.einsum("...cj,ckj->...ck", values, channel_matrices)

        left, right = paired_bivector_factors(
            self.algebra,
            left_weights.to(device=values.device, dtype=values.dtype),
            right_weights.to(device=values.device, dtype=values.dtype),
            parameter_layout=self.parameter_layout,
            rotor_layout=self.rotor_layout,
        )
        pair_index = channel_to_pair.to(device=values.device)
        left_by_channel = torch.index_select(left, 0, pair_index)
        right_by_channel = torch.index_select(right, 0, pair_index)

        middle = self.algebra.geometric_product(
            left_by_channel,
            values,
            left_layout=self.rotor_layout,
            right_layout=self.input_layout,
            output_layout=self.middle_layout,
        )
        return self.algebra.geometric_product(
            middle,
            right_by_channel,
            left_layout=self.middle_layout,
            right_layout=self.rotor_layout,
            output_layout=self.output_layout,
        )


def apply_graded_linear_action(
    values: torch.Tensor,
    matrix: torch.Tensor,
    *,
    input_layout: GradeLayout,
    output_layout: GradeLayout,
) -> torch.Tensor:
    """Apply the outermorphism induced by a vector-space linear action."""
    if input_layout.spec != output_layout.spec:
        raise ValueError(f"layout mismatch: {input_layout.spec} vs {output_layout.spec}")
    if values.shape[-1] != input_layout.dim:
        raise ValueError(f"input dimension must be {input_layout.dim}, got {values.shape[-1]}")
    if values.ndim < 2:
        raise ValueError(f"values must include channel and lane axes, got shape {tuple(values.shape)}")

    spec = input_layout.spec
    if matrix.shape[-2:] != (spec.n, spec.n):
        raise ValueError(f"matrix trailing shape must be {(spec.n, spec.n)}, got {tuple(matrix.shape[-2:])}")
    if matrix.ndim != 3:
        raise ValueError(f"matrix must have shape [channels, n, n], got {tuple(matrix.shape)}")
    if matrix.shape[0] != values.shape[-2]:
        raise ValueError(f"matrix channels {matrix.shape[0]} do not match input channels {values.shape[-2]}")

    coefficients = _graded_action_coefficients(matrix, input_layout=input_layout, output_layout=output_layout)
    return torch.einsum("coi,...ci->...co", coefficients, values)


def apply_multi_graded_linear_action(
    values: torch.Tensor,
    matrices: torch.Tensor,
    *,
    input_layout: GradeLayout,
    output_layout: GradeLayout,
) -> torch.Tensor:
    """Apply multiple outermorphisms to declared grade lanes."""
    if input_layout.spec != output_layout.spec:
        raise ValueError(f"layout mismatch: {input_layout.spec} vs {output_layout.spec}")
    if values.shape[-1] != input_layout.dim:
        raise ValueError(f"input dimension must be {input_layout.dim}, got {values.shape[-1]}")
    if values.ndim < 2:
        raise ValueError(f"values must include channel and lane axes, got shape {tuple(values.shape)}")

    spec = input_layout.spec
    if matrices.shape[-2:] != (spec.n, spec.n):
        raise ValueError(f"matrices trailing shape must be {(spec.n, spec.n)}, got {tuple(matrices.shape[-2:])}")
    if matrices.ndim != 3:
        raise ValueError(f"matrices must have shape [actions, n, n], got {tuple(matrices.shape)}")

    coefficients = _graded_action_coefficients(matrices, input_layout=input_layout, output_layout=output_layout)
    return torch.einsum("koi,...ci->...cko", coefficients, values)


def versor_vector_matrix(algebra, weights: torch.Tensor, *, grade: int, parameter_layout: GradeLayout) -> torch.Tensor:
    """Return the vector-space matrix represented by grade-1 or grade-2 weights."""
    grade = int(grade)
    if grade == 2:
        return torch.matrix_exp(bivector_vector_generator(weights, bivector_layout=parameter_layout))
    if grade == 1:
        signs = metric_self_signs(parameter_layout, device=weights.device, dtype=weights.dtype)
        norm_sq = (weights * weights * signs).sum(dim=-1, keepdim=True)
        scale = norm_sq.abs().clamp_min(eps_like(norm_sq)).sqrt()
        normals = weights / scale
        return reflection_vector_matrix(normals, vector_layout=parameter_layout, eps=algebra.eps_sq)
    raise ValueError("planned versor execution currently supports grade=1 and grade=2")


def bivector_vector_generator(bivectors: torch.Tensor, *, bivector_layout: GradeLayout) -> torch.Tensor:
    """Return the vector-space generator induced by bivectors."""
    if bivector_layout.grades != (2,):
        raise ValueError(f"bivector_layout must contain grade 2 only, got {bivector_layout.grades}")
    spec = bivector_layout.spec
    vector_layout = spec.layout((1,))
    if bivectors.shape[-1] != bivector_layout.dim:
        raise ValueError(f"bivector dimension must be {bivector_layout.dim}, got {bivectors.shape[-1]}")

    generator = bivectors.new_zeros(*bivectors.shape[:-1], spec.n, spec.n)
    vector_positions = {index: position for position, index in enumerate(vector_layout.basis_indices)}

    for bivector_position, bivector_index in enumerate(bivector_layout.basis_indices):
        coeffs = bivectors[..., bivector_position]
        if coeffs.ndim == 0:
            coeffs = coeffs.unsqueeze(0)
        for input_position, input_index in enumerate(vector_layout.basis_indices):
            output_index = bivector_index ^ input_index
            output_position = vector_positions.get(output_index)
            if output_position is None:
                continue
            coefficient = -0.5 * operation_coefficient(
                bivector_index,
                input_index,
                spec.p,
                spec.q,
                spec.r,
                "commutator",
            )
            if coefficient == 0.0:
                continue
            generator[..., output_position, input_position] = (
                generator[..., output_position, input_position] + coeffs * coefficient
            )
    return generator


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


def _check_channel_values(values: torch.Tensor, layout: GradeLayout, channels: int, name: str) -> None:
    if values.ndim < 3:
        raise ValueError(f"{name}: expected ndim >= 3, got shape {tuple(values.shape)}")
    if values.shape[-2] != channels:
        raise ValueError(f"{name}: expected {channels} channels, got {values.shape[-2]}")
    if values.shape[-1] != layout.dim:
        raise ValueError(f"{name}: last dim must be {layout.dim} for grades {layout.grades}, got {values.shape[-1]}")


def dense_versor_factors(algebra, weights: torch.Tensor, *, grade: int, parameter_layout: GradeLayout):
    """Explicit dense versor factors for layers that call dense sandwich kernels."""
    versor = materialize_full(algebra, weights, layout=parameter_layout)
    grade = int(grade)
    if grade == 2:
        rotor = algebra.exp(-0.5 * versor)
        return rotor, algebra.reverse(rotor)

    if grade == 1:
        norm_sq = algebra.norm_sq(versor)
        scale = norm_sq.abs().clamp_min(eps_like(norm_sq)).sqrt()
        versor = versor / scale
    else:
        norm = versor.norm(dim=-1, keepdim=True).clamp_min(eps_like(versor))
        versor = versor / norm
    return algebra.grade_involution(versor), algebra.blade_inverse(versor)


def dense_paired_bivector_factors(
    algebra,
    left_weights: torch.Tensor,
    right_weights: torch.Tensor,
    *,
    parameter_layout: GradeLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dense ``(R_left, reverse(R_right))`` for independent bivectors."""
    left = materialize_full(algebra, left_weights, layout=parameter_layout)
    right = materialize_full(algebra, right_weights, layout=parameter_layout)
    left_rotor = algebra.exp(-0.5 * left)
    right_rotor = algebra.exp(-0.5 * right)
    return left_rotor, algebra.reverse(right_rotor)


def paired_bivector_factors(
    algebra,
    left_weights: torch.Tensor,
    right_weights: torch.Tensor,
    *,
    parameter_layout: GradeLayout,
    rotor_layout: GradeLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return compact ``(R_left, reverse(R_right))`` for paired bivector actions."""
    if hasattr(algebra, "sandwich_action_matrices"):
        left, right = dense_paired_bivector_factors(
            algebra,
            left_weights,
            right_weights,
            parameter_layout=parameter_layout,
        )
        return rotor_layout.compact(left), rotor_layout.compact(right)

    left_rotor = algebra.exp(
        -0.5 * left_weights,
        input_layout=parameter_layout,
        output_layout=rotor_layout,
    )
    right_rotor = algebra.exp(
        -0.5 * right_weights,
        input_layout=parameter_layout,
        output_layout=rotor_layout,
    )
    right_reverse = algebra.reverse(right_rotor, input_layout=rotor_layout, output_layout=rotor_layout)
    return left_rotor, right_reverse
