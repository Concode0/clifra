"""Planned linear actions on compact grade layouts."""

from __future__ import annotations

import torch

from clifra.core.foundation.basis import operation_coefficient
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout


def apply_graded_linear_action(
    values: torch.Tensor,
    matrix: torch.Tensor,
    *,
    input_layout: GradeLayout,
    output_layout: GradeLayout,
) -> torch.Tensor:
    """Apply the outermorphism induced by a vector-space linear action.

    ``matrix`` stores per-channel vector coefficients with shape
    ``[channels, n, n]`` using ``output_vector = matrix @ input_vector``.
    ``values`` stores compact grade lanes with shape ``[..., channels,
    input_layout.dim]``. The result is compact in ``output_layout``.
    """
    if input_layout.spec != output_layout.spec:
        raise ValueError(f"layout mismatch: {input_layout.spec} vs {output_layout.spec}")
    if values.shape[-1] != input_layout.dim:
        raise ValueError(f"input compact dimension must be {input_layout.dim}, got {values.shape[-1]}")
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
    """Apply multiple outermorphisms to compact grade lanes.

    ``matrices`` stores ``[actions, n, n]`` vector-space maps. ``values``
    stores ``[..., channels, input_layout.dim]`` compact lanes. The result is
    ``[..., channels, actions, output_layout.dim]``.
    """
    if input_layout.spec != output_layout.spec:
        raise ValueError(f"layout mismatch: {input_layout.spec} vs {output_layout.spec}")
    if values.shape[-1] != input_layout.dim:
        raise ValueError(f"input compact dimension must be {input_layout.dim}, got {values.shape[-1]}")
    if values.ndim < 2:
        raise ValueError(f"values must include channel and lane axes, got shape {tuple(values.shape)}")

    spec = input_layout.spec
    if matrices.shape[-2:] != (spec.n, spec.n):
        raise ValueError(f"matrices trailing shape must be {(spec.n, spec.n)}, got {tuple(matrices.shape[-2:])}")
    if matrices.ndim != 3:
        raise ValueError(f"matrices must have shape [actions, n, n], got {tuple(matrices.shape)}")

    coefficients = _graded_action_coefficients(matrices, input_layout=input_layout, output_layout=output_layout)
    return torch.einsum("koi,...ci->...cko", coefficients, values)


def bivector_vector_generator(bivectors: torch.Tensor, *, bivector_layout: GradeLayout) -> torch.Tensor:
    """Return the vector-space generator induced by compact bivectors."""
    if bivector_layout.grades != (2,):
        raise ValueError(f"bivector_layout must contain grade 2 only, got {bivector_layout.grades}")
    spec = bivector_layout.spec
    vector_layout = spec.layout((1,))
    if bivectors.shape[-1] != bivector_layout.dim:
        raise ValueError(f"bivector compact dimension must be {bivector_layout.dim}, got {bivectors.shape[-1]}")

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
    """Return the vector-space reflection matrix for compact normal vectors."""
    if vector_layout.grades != (1,):
        raise ValueError(f"vector_layout must contain grade 1 only, got {vector_layout.grades}")
    if normals.shape[-1] != vector_layout.dim:
        raise ValueError(f"normal compact dimension must be {vector_layout.dim}, got {normals.shape[-1]}")

    signs = metric_self_signs(vector_layout, device=normals.device, dtype=normals.dtype)
    denominator = (normals * normals * signs).sum(dim=-1, keepdim=True)
    denominator = signed_clamp_min(denominator, eps)
    weighted_normals = normals * signs
    outer = normals.unsqueeze(-1) * weighted_normals.unsqueeze(-2)
    eye = torch.eye(vector_layout.spec.n, device=normals.device, dtype=normals.dtype)
    return eye - 2.0 * outer / denominator.unsqueeze(-1)


def metric_self_signs(layout: GradeLayout, *, device=None, dtype=None) -> torch.Tensor:
    """Return basis self-product signs for a layout."""
    signs = [
        operation_coefficient(index, index, layout.spec.p, layout.spec.q, layout.spec.r, "gp")
        for index in layout.basis_indices
    ]
    return torch.tensor(signs, device=device, dtype=torch.float32 if dtype is None else dtype)


def signed_clamp_min(values: torch.Tensor, eps: float) -> torch.Tensor:
    """Clamp magnitude while preserving sign for inverse denominators."""
    signs = torch.where(values < 0, -torch.ones_like(values), torch.ones_like(values))
    return signs * values.abs().clamp_min(eps)


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
