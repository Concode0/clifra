# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Static bivector exponential plans and executor-family selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from clifra.core.foundation.basis import basis_index_tuple_for_grades, basis_product, operation_coefficient
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout

SPECTRAL_LOCAL_MAX_PLANES = 4
SPECTRAL_LOCAL_MAX_IDEAL_DIM = 4


@dataclass(frozen=True)
class BivectorExpExecutionPolicy:
    """Public policy knobs for planned bivector exponentials."""

    spectral_max_planes: int | None = None
    spectral_tol_abs: float | None = None
    spectral_tol_rel: float = 0.0
    spectral_dominant_rel: float | None = None
    spectral_allow_degenerate: bool = True
    spectral_allow_truncated_degenerate: bool = True


DEFAULT_BIVECTOR_EXP_EXECUTION_POLICY = BivectorExpExecutionPolicy()


@dataclass(frozen=True)
class BivectorExpPlan:
    """Static layout contract for a bivector exponential executor."""

    spec: AlgebraSpec
    input_layout: GradeLayout
    grade4_layout: GradeLayout | None
    operator_layout: GradeLayout
    output_layout: GradeLayout
    executor_family: str
    metric_signs: torch.Tensor
    bivector_squared_signs: torch.Tensor
    nondegenerate_bivector_positions: torch.Tensor
    mixed_degenerate_bivector_positions: torch.Tensor
    nilpotent_bivector_positions: torch.Tensor
    bivector_to_nondegenerate_generator: torch.Tensor
    nondegenerate_generator_to_bivector: torch.Tensor
    bivector_to_mixed_generator: torch.Tensor
    output_scalar_mask: torch.Tensor
    operator_scalar_mask: torch.Tensor
    bivector_to_output: torch.Tensor
    bivector_to_operator: torch.Tensor
    grade4_to_output: torch.Tensor
    operator_to_output: torch.Tensor
    operator_eye: torch.Tensor
    operator_scalar_position: int
    spectral_max_planes: int
    spectral_local_scalar_mask: torch.Tensor
    spectral_local_plane_masks: torch.Tensor
    spectral_local_product_table: torch.Tensor
    spectral_local_sparse_left_positions: torch.Tensor
    spectral_local_sparse_right_positions: torch.Tensor
    spectral_local_sparse_output_positions: torch.Tensor
    spectral_local_sparse_coefficients: torch.Tensor
    spectral_plane_bivector_map: torch.Tensor
    spectral_plane_eye: torch.Tensor
    spectral_plane_left_positions: torch.Tensor
    spectral_plane_right_positions: torch.Tensor
    spectral_plane_output_positions: torch.Tensor
    spectral_plane_coefficients: torch.Tensor
    spectral_plane_to_local: torch.Tensor
    spectral_nilpotent_to_local: torch.Tensor
    spectral_ideal_basis: torch.Tensor
    spectral_lift_grades: torch.Tensor
    spectral_lift_local_positions: torch.Tensor
    spectral_lift_local_axes: torch.Tensor
    spectral_lift_local_mask: torch.Tensor
    spectral_lift_target_axes: torch.Tensor
    spectral_lift_target_map: torch.Tensor
    spectral_tolerances: torch.Tensor
    spectral_tol_abs: float
    spectral_tol_rel: float
    spectral_dominant_rel: float
    spectral_allow_degenerate: bool
    spectral_allow_truncated_degenerate: bool
    nondegenerate_dim: int
    ideal_dim: int
    spectral_local_axis_count: int
    eps: float
    eps_sq: float


@dataclass(frozen=True)
class SpectralExpPreselection:
    """Static eligibility record for the spectral-local exp path."""

    eligible: bool
    reason: str
    max_planes: int
    tol_abs: float
    tol_rel: float
    dominant_rel: float
    nondegenerate_dim: int
    ideal_dim: int
    solver_family: str


def build_bivector_exp_plan(
    spec: AlgebraSpec,
    *,
    input_layout: GradeLayout,
    output_layout: GradeLayout,
    dtype: torch.dtype,
    device,
    spectral_max_planes: int | None = None,
    spectral_tol_abs: float | None = None,
    spectral_tol_rel: float = 0.0,
    spectral_dominant_rel: float | None = None,
    spectral_allow_degenerate: bool = True,
    spectral_allow_truncated_degenerate: bool = True,
) -> BivectorExpPlan:
    """Build a static plan for ``exp(B)`` where ``B`` is grade-2."""
    if input_layout.spec != spec:
        raise ValueError(f"input_layout signature {input_layout.spec} does not match algebra signature {spec}")
    if output_layout.spec != spec:
        raise ValueError(f"output_layout signature {output_layout.spec} does not match algebra signature {spec}")
    if input_layout.grades != (2,):
        raise ValueError(f"bivector exp requires grade-2 input layout, got {input_layout.grades}")

    resolved_device = torch.device(device)
    grade4_layout = spec.layout((4,)) if spec.n >= 4 else None
    operator_layout = spec.layout(range(0, spec.n + 1, 2))
    finfo = torch.finfo(dtype)
    operator_position_by_index = {index: position for position, index in enumerate(operator_layout.basis_indices)}
    metric_signs = _metric_signs(spec, dtype=dtype, device=resolved_device)
    partition = _bivector_axis_partition(input_layout, spec, device=resolved_device)
    preselection = spectral_exp_preselection(
        spec,
        resolved_device,
        dtype=dtype,
        max_planes=spectral_max_planes,
        tol_abs=spectral_tol_abs,
        tol_rel=spectral_tol_rel,
        dominant_rel=spectral_dominant_rel,
        allow_degenerate=spectral_allow_degenerate,
        allow_truncated_degenerate=spectral_allow_truncated_degenerate,
    )
    local_buffers = _spectral_local_buffers(
        spec,
        input_layout,
        output_layout,
        preselection.max_planes,
        ideal_dim=preselection.ideal_dim if preselection.eligible else 0,
        dtype=dtype,
        device=resolved_device,
    )

    signs = []
    for index in input_layout.basis_indices:
        bits = [bit for bit in range(spec.n) if index & (1 << bit)]
        if len(bits) != 2:
            signs.append(0.0)
            continue
        a, b = bits
        s_a = 1.0 if a < spec.p else (-1.0 if a < spec.p + spec.q else 0.0)
        s_b = 1.0 if b < spec.p else (-1.0 if b < spec.p + spec.q else 0.0)
        signs.append(-s_a * s_b)

    return BivectorExpPlan(
        spec=spec,
        input_layout=input_layout,
        grade4_layout=grade4_layout,
        operator_layout=operator_layout,
        output_layout=output_layout,
        executor_family=select_bivector_exp_executor_family(
            spec,
            resolved_device,
            dtype=dtype,
            spectral_max_planes=spectral_max_planes,
            spectral_tol_abs=spectral_tol_abs,
            spectral_tol_rel=spectral_tol_rel,
            spectral_dominant_rel=spectral_dominant_rel,
            spectral_allow_degenerate=spectral_allow_degenerate,
            spectral_allow_truncated_degenerate=spectral_allow_truncated_degenerate,
        ),
        metric_signs=metric_signs,
        bivector_squared_signs=torch.tensor(signs, dtype=dtype, device=resolved_device),
        nondegenerate_bivector_positions=partition[0],
        mixed_degenerate_bivector_positions=partition[1],
        nilpotent_bivector_positions=partition[2],
        bivector_to_nondegenerate_generator=_bivector_to_nondegenerate_generator(
            input_layout,
            spec,
            dtype=dtype,
            device=resolved_device,
        ),
        nondegenerate_generator_to_bivector=_nondegenerate_generator_to_bivector(
            input_layout,
            spec,
            dtype=dtype,
            device=resolved_device,
        ),
        bivector_to_mixed_generator=_bivector_to_mixed_generator(
            input_layout,
            spec,
            dtype=dtype,
            device=resolved_device,
        ),
        output_scalar_mask=_scalar_mask(output_layout, dtype=dtype, device=resolved_device),
        operator_scalar_mask=_scalar_mask(operator_layout, dtype=dtype, device=resolved_device),
        bivector_to_output=_layout_map(input_layout, output_layout, dtype=dtype, device=resolved_device),
        bivector_to_operator=_layout_map(input_layout, operator_layout, dtype=dtype, device=resolved_device),
        grade4_to_output=_layout_map(grade4_layout, output_layout, dtype=dtype, device=resolved_device),
        operator_to_output=_layout_map(operator_layout, output_layout, dtype=dtype, device=resolved_device),
        operator_eye=torch.eye(operator_layout.dim, dtype=dtype, device=resolved_device),
        operator_scalar_position=operator_position_by_index[0],
        spectral_max_planes=preselection.max_planes,
        spectral_local_scalar_mask=local_buffers["scalar_mask"],
        spectral_local_plane_masks=local_buffers["plane_masks"],
        spectral_local_product_table=local_buffers["product_table"],
        spectral_local_sparse_left_positions=local_buffers["sparse_left_positions"],
        spectral_local_sparse_right_positions=local_buffers["sparse_right_positions"],
        spectral_local_sparse_output_positions=local_buffers["sparse_output_positions"],
        spectral_local_sparse_coefficients=local_buffers["sparse_coefficients"],
        spectral_plane_bivector_map=local_buffers["plane_bivector_map"],
        spectral_plane_eye=local_buffers["plane_eye"],
        spectral_plane_left_positions=local_buffers["plane_left_positions"],
        spectral_plane_right_positions=local_buffers["plane_right_positions"],
        spectral_plane_output_positions=local_buffers["plane_output_positions"],
        spectral_plane_coefficients=local_buffers["plane_coefficients"],
        spectral_plane_to_local=local_buffers["plane_to_local"],
        spectral_nilpotent_to_local=local_buffers["nilpotent_to_local"],
        spectral_ideal_basis=local_buffers["ideal_basis"],
        spectral_lift_grades=local_buffers["lift_grades"],
        spectral_lift_local_positions=local_buffers["lift_local_positions"],
        spectral_lift_local_axes=local_buffers["lift_local_axes"],
        spectral_lift_local_mask=local_buffers["lift_local_mask"],
        spectral_lift_target_axes=local_buffers["lift_target_axes"],
        spectral_lift_target_map=local_buffers["lift_target_map"],
        spectral_tolerances=torch.tensor(
            [preselection.tol_abs, preselection.tol_rel, preselection.dominant_rel],
            dtype=dtype,
            device=resolved_device,
        ),
        spectral_tol_abs=preselection.tol_abs,
        spectral_tol_rel=preselection.tol_rel,
        spectral_dominant_rel=preselection.dominant_rel,
        spectral_allow_degenerate=bool(spectral_allow_degenerate),
        spectral_allow_truncated_degenerate=bool(spectral_allow_truncated_degenerate),
        nondegenerate_dim=preselection.nondegenerate_dim,
        ideal_dim=preselection.ideal_dim,
        spectral_local_axis_count=int(local_buffers["axis_count"]),
        eps=float(finfo.eps),
        eps_sq=float(finfo.eps**2),
    )


def _spectral_local_buffers(
    spec: AlgebraSpec,
    input_layout: GradeLayout,
    output_layout: GradeLayout,
    max_planes: int,
    *,
    ideal_dim: int,
    dtype: torch.dtype,
    device,
) -> dict[str, torch.Tensor]:
    nondegenerate_dim = spec.p + spec.q
    use_ideal_block = int(ideal_dim) > 0
    local_nondegenerate_dim = max(2 * int(max_planes), 2)
    axis_count = local_nondegenerate_dim + int(ideal_dim)
    local_indices = basis_index_tuple_for_grades(axis_count, range(0, axis_count + 1, 2))
    local_positions = {index: position for position, index in enumerate(local_indices)}
    local_dim = len(local_indices)

    scalar_mask = torch.zeros(local_dim, dtype=dtype, device=device)
    scalar_mask[local_positions[0]] = 1.0

    plane_masks = torch.zeros((int(max_planes), local_dim), dtype=dtype, device=device)
    for plane in range(int(max_planes)):
        plane_index = (1 << (2 * plane)) | (1 << (2 * plane + 1))
        plane_masks[plane, local_positions[plane_index]] = 1.0

    product_table = torch.zeros((local_dim, local_dim, local_dim), dtype=dtype, device=device)
    sparse_product = _empty_sparse_product_buffers(dtype=dtype, device=device)
    if use_ideal_block:
        product_table = torch.zeros((1, 1, 1), dtype=dtype, device=device)
        local_p = local_nondegenerate_dim if spec.q == 0 else 0
        local_q = local_nondegenerate_dim if spec.p == 0 else 0
        sparse_product = _sparse_product_buffers(
            local_indices,
            axis_count,
            local_p,
            local_q,
            int(ideal_dim),
            dtype=dtype,
            device=device,
        )
    else:
        for left_position, left_index in enumerate(local_indices):
            for right_position, right_index in enumerate(local_indices):
                output_index, sign = basis_product(left_index, right_index, axis_count, 0, 0)
                output_position = local_positions.get(output_index)
                if output_position is not None and sign != 0.0:
                    product_table[left_position, right_position, output_position] = float(sign)

    output_positions_by_grade: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
    for output_position, output_index in enumerate(output_layout.basis_indices):
        grade = int(output_index).bit_count()
        if grade % 2 == 0 and grade <= axis_count:
            output_positions_by_grade.setdefault(grade, []).append((output_position, tuple(_basis_bits(output_index, spec.n))))

    grades = [grade for grade in range(0, axis_count + 1, 2) if grade in output_positions_by_grade]
    if not grades:
        grades = [0]
    local_blades_by_grade = {
        grade: [
            (position, tuple(_basis_bits(index, axis_count)))
            for position, index in enumerate(local_indices)
            if int(index).bit_count() == grade
        ]
        for grade in grades
    }
    target_blades_by_grade = {grade: output_positions_by_grade.get(grade, []) for grade in grades}
    max_local = max(len(local_blades_by_grade[grade]) for grade in grades)
    max_target = max(max(len(target_blades_by_grade[grade]), 1) for grade in grades)
    max_grade = axis_count
    grade_count = len(grades)

    lift_grades = torch.tensor(grades, dtype=torch.long, device=device)
    lift_local_positions = torch.zeros((grade_count, max_local), dtype=torch.long, device=device)
    lift_local_axes = torch.zeros((grade_count, max_local, max_grade), dtype=torch.long, device=device)
    lift_local_mask = torch.zeros((grade_count, max_local), dtype=dtype, device=device)
    lift_target_axes = torch.zeros((grade_count, max_target, max_grade), dtype=torch.long, device=device)
    lift_target_map = torch.zeros((grade_count, max_target, output_layout.dim), dtype=dtype, device=device)

    for grade_index, grade in enumerate(grades):
        lift_sign = 1.0 if use_ideal_block else (-1.0 if spec.p == 0 and spec.q > 0 else 1.0) ** (grade // 2)
        for local_slot, (local_position, local_axes) in enumerate(local_blades_by_grade[grade]):
            lift_local_positions[grade_index, local_slot] = local_position
            lift_local_mask[grade_index, local_slot] = 1.0
            if grade:
                lift_local_axes[grade_index, local_slot, :grade] = torch.tensor(
                    local_axes,
                    dtype=torch.long,
                    device=device,
                )
        for target_slot, (target_position, target_axes) in enumerate(target_blades_by_grade[grade]):
            lift_target_map[grade_index, target_slot, target_position] = lift_sign
            if grade:
                lift_target_axes[grade_index, target_slot, :grade] = torch.tensor(
                    target_axes,
                    dtype=torch.long,
                    device=device,
                )

    plane_buffers = _plane_exp_buffers(
        max_planes,
        local_indices,
        local_positions,
        local_nondegenerate_dim,
        int(ideal_dim),
        spec,
        dtype=dtype,
        device=device,
    )
    nilpotent_to_local = _nilpotent_to_local_map(
        input_layout,
        local_positions,
        local_nondegenerate_dim,
        int(ideal_dim),
        spec,
        dtype=dtype,
        device=device,
    )
    ideal_basis = torch.zeros((int(ideal_dim), spec.n), dtype=dtype, device=device)
    for ideal_axis in range(int(ideal_dim)):
        ideal_basis[ideal_axis, nondegenerate_dim + ideal_axis] = 1.0

    return {
        "axis_count": axis_count,
        "scalar_mask": scalar_mask,
        "plane_masks": plane_masks,
        "product_table": product_table,
        "sparse_left_positions": sparse_product["left_positions"],
        "sparse_right_positions": sparse_product["right_positions"],
        "sparse_output_positions": sparse_product["output_positions"],
        "sparse_coefficients": sparse_product["coefficients"],
        "plane_bivector_map": plane_buffers["bivector_map"],
        "plane_eye": plane_buffers["eye"],
        "plane_left_positions": plane_buffers["left_positions"],
        "plane_right_positions": plane_buffers["right_positions"],
        "plane_output_positions": plane_buffers["output_positions"],
        "plane_coefficients": plane_buffers["coefficients"],
        "plane_to_local": plane_buffers["to_local"],
        "nilpotent_to_local": nilpotent_to_local,
        "ideal_basis": ideal_basis,
        "lift_grades": lift_grades,
        "lift_local_positions": lift_local_positions,
        "lift_local_axes": lift_local_axes,
        "lift_local_mask": lift_local_mask,
        "lift_target_axes": lift_target_axes,
        "lift_target_map": lift_target_map,
    }


def _empty_sparse_product_buffers(*, dtype: torch.dtype, device) -> dict[str, torch.Tensor]:
    return {
        "left_positions": torch.zeros(0, dtype=torch.long, device=device),
        "right_positions": torch.zeros(0, dtype=torch.long, device=device),
        "output_positions": torch.zeros(0, dtype=torch.long, device=device),
        "coefficients": torch.zeros(0, dtype=dtype, device=device),
    }


def _sparse_product_buffers(
    basis_indices: tuple[int, ...],
    axis_count: int,
    p: int,
    q: int,
    r: int,
    *,
    dtype: torch.dtype,
    device,
) -> dict[str, torch.Tensor]:
    positions = {index: position for position, index in enumerate(basis_indices)}
    left_positions: list[int] = []
    right_positions: list[int] = []
    output_positions: list[int] = []
    coefficients: list[float] = []
    for left_position, left_index in enumerate(basis_indices):
        for right_position, right_index in enumerate(basis_indices):
            output_index, sign = basis_product(left_index, right_index, p, q, r)
            output_position = positions.get(output_index)
            if output_position is None or sign == 0.0:
                continue
            left_positions.append(left_position)
            right_positions.append(right_position)
            output_positions.append(output_position)
            coefficients.append(float(sign))
    return {
        "left_positions": torch.tensor(left_positions, dtype=torch.long, device=device),
        "right_positions": torch.tensor(right_positions, dtype=torch.long, device=device),
        "output_positions": torch.tensor(output_positions, dtype=torch.long, device=device),
        "coefficients": torch.tensor(coefficients, dtype=dtype, device=device),
    }


def _plane_exp_buffers(
    max_planes: int,
    local_indices: tuple[int, ...],
    local_positions: dict[int, int],
    local_nondegenerate_dim: int,
    ideal_dim: int,
    spec: AlgebraSpec,
    *,
    dtype: torch.dtype,
    device,
) -> dict[str, torch.Tensor]:
    if ideal_dim == 0:
        return {
            "bivector_map": torch.zeros((1, 1), dtype=dtype, device=device),
            "eye": torch.eye(1, dtype=dtype, device=device),
            "left_positions": torch.zeros(0, dtype=torch.long, device=device),
            "right_positions": torch.zeros(0, dtype=torch.long, device=device),
            "output_positions": torch.zeros(0, dtype=torch.long, device=device),
            "coefficients": torch.zeros(0, dtype=dtype, device=device),
            "to_local": torch.zeros((int(max_planes), 1, len(local_indices)), dtype=dtype, device=device),
        }

    plane_axis_count = 2 + ideal_dim
    plane_indices = basis_index_tuple_for_grades(plane_axis_count, range(0, plane_axis_count + 1, 2))
    plane_positions = {index: position for position, index in enumerate(plane_indices)}
    plane_dim = len(plane_indices)
    feature_count = 1 + 2 * ideal_dim
    feature_sign = -1.0 if spec.p == 0 and spec.q > 0 else 1.0
    bivector_map = torch.zeros((feature_count, plane_dim), dtype=dtype, device=device)
    bivector_map[0, plane_positions[0b11]] = feature_sign
    for ideal_axis in range(ideal_dim):
        ideal_bit = 1 << (2 + ideal_axis)
        bivector_map[1 + ideal_axis, plane_positions[(1 << 0) | ideal_bit]] = feature_sign
        bivector_map[1 + ideal_dim + ideal_axis, plane_positions[(1 << 1) | ideal_bit]] = feature_sign

    plane_p = 2 if spec.q == 0 else 0
    plane_q = 2 if spec.p == 0 else 0
    sparse_product = _sparse_product_buffers(
        plane_indices,
        plane_axis_count,
        plane_p,
        plane_q,
        ideal_dim,
        dtype=dtype,
        device=device,
    )
    to_local = torch.zeros((int(max_planes), plane_dim, len(local_indices)), dtype=dtype, device=device)
    for plane in range(int(max_planes)):
        axis_map = {0: 2 * plane, 1: 2 * plane + 1}
        axis_map.update({2 + ideal_axis: local_nondegenerate_dim + ideal_axis for ideal_axis in range(ideal_dim)})
        for plane_position, plane_index in enumerate(plane_indices):
            local_index = 0
            for axis in _basis_bits(plane_index, plane_axis_count):
                local_index |= 1 << axis_map[axis]
            to_local[plane, plane_position, local_positions[local_index]] = 1.0

    return {
        "bivector_map": bivector_map,
        "eye": torch.eye(plane_dim, dtype=dtype, device=device),
        "left_positions": sparse_product["left_positions"],
        "right_positions": sparse_product["right_positions"],
        "output_positions": sparse_product["output_positions"],
        "coefficients": sparse_product["coefficients"],
        "to_local": to_local,
    }


def _nilpotent_to_local_map(
    input_layout: GradeLayout,
    local_positions: dict[int, int],
    local_nondegenerate_dim: int,
    ideal_dim: int,
    spec: AlgebraSpec,
    *,
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    output = torch.zeros((input_layout.dim, len(local_positions)), dtype=dtype, device=device)
    if ideal_dim < 2:
        return output
    nondegenerate_dim = spec.p + spec.q
    for input_position, input_index in enumerate(input_layout.basis_indices):
        bits = _basis_bits(input_index, spec.n)
        if len(bits) != 2 or bits[0] < nondegenerate_dim or bits[1] < nondegenerate_dim:
            continue
        local_index = 0
        for bit in bits:
            local_index |= 1 << (local_nondegenerate_dim + bit - nondegenerate_dim)
        local_position = local_positions.get(local_index)
        if local_position is not None:
            output[input_position, local_position] = 1.0
    return output


def _metric_signs(spec: AlgebraSpec, *, dtype: torch.dtype, device) -> torch.Tensor:
    signs = [1.0] * spec.p + [-1.0] * spec.q + [0.0] * spec.r
    return torch.tensor(signs, dtype=dtype, device=device)


def _bivector_axis_partition(
    input_layout: GradeLayout,
    spec: AlgebraSpec,
    *,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nondegenerate: list[int] = []
    mixed: list[int] = []
    nilpotent: list[int] = []
    split = spec.p + spec.q

    for position, index in enumerate(input_layout.basis_indices):
        bits = _basis_bits(index, spec.n)
        if len(bits) != 2:
            continue
        left_is_null = bits[0] >= split
        right_is_null = bits[1] >= split
        if left_is_null and right_is_null:
            nilpotent.append(position)
        elif left_is_null or right_is_null:
            mixed.append(position)
        else:
            nondegenerate.append(position)

    return (
        torch.tensor(nondegenerate, dtype=torch.long, device=device),
        torch.tensor(mixed, dtype=torch.long, device=device),
        torch.tensor(nilpotent, dtype=torch.long, device=device),
    )


def _bivector_to_nondegenerate_generator(
    input_layout: GradeLayout,
    spec: AlgebraSpec,
    *,
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    nondegenerate_dim = spec.p + spec.q
    output = torch.zeros((input_layout.dim, nondegenerate_dim, nondegenerate_dim), dtype=dtype, device=device)
    vector_indices = [1 << axis for axis in range(nondegenerate_dim)]
    vector_positions = {index: position for position, index in enumerate(vector_indices)}

    for bivector_position, bivector_index in enumerate(input_layout.basis_indices):
        bits = _basis_bits(bivector_index, spec.n)
        if len(bits) != 2 or bits[0] >= nondegenerate_dim or bits[1] >= nondegenerate_dim:
            continue
        for input_position, input_index in enumerate(vector_indices):
            output_index = bivector_index ^ input_index
            output_position = vector_positions.get(output_index)
            if output_position is None:
                continue
            output[bivector_position, output_position, input_position] = _vector_generator_coefficient(
                bivector_index,
                input_index,
                spec,
            )
    return output


def _bivector_to_mixed_generator(
    input_layout: GradeLayout,
    spec: AlgebraSpec,
    *,
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    nondegenerate_dim = spec.p + spec.q
    output = torch.zeros((input_layout.dim, spec.r, nondegenerate_dim), dtype=dtype, device=device)
    if spec.r == 0 or nondegenerate_dim == 0:
        return output

    nondegenerate_indices = [1 << axis for axis in range(nondegenerate_dim)]
    ideal_positions = {1 << (nondegenerate_dim + axis): axis for axis in range(spec.r)}

    for bivector_position, bivector_index in enumerate(input_layout.basis_indices):
        bits = _basis_bits(bivector_index, spec.n)
        if len(bits) != 2 or not (bits[0] < nondegenerate_dim <= bits[1]):
            continue
        for input_position, input_index in enumerate(nondegenerate_indices):
            output_index = bivector_index ^ input_index
            output_position = ideal_positions.get(output_index)
            if output_position is None:
                continue
            output[bivector_position, output_position, input_position] = _vector_generator_coefficient(
                bivector_index,
                input_index,
                spec,
            )
    return output


def _nondegenerate_generator_to_bivector(
    input_layout: GradeLayout,
    spec: AlgebraSpec,
    *,
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    nondegenerate_dim = spec.p + spec.q
    output = torch.zeros((nondegenerate_dim, nondegenerate_dim, input_layout.dim), dtype=dtype, device=device)
    input_positions = {index: position for position, index in enumerate(input_layout.basis_indices)}

    for i in range(nondegenerate_dim):
        for j in range(i + 1, nondegenerate_dim):
            bivector_position = input_positions.get((1 << i) | (1 << j))
            if bivector_position is None:
                continue
            metric_sign = 1.0 if i < spec.p else -1.0
            output[j, i, bivector_position] = metric_sign
    return output


def _vector_generator_coefficient(bivector_index: int, vector_index: int, spec: AlgebraSpec) -> float:
    return -0.5 * operation_coefficient(bivector_index, vector_index, spec.p, spec.q, spec.r, "commutator")


def _basis_bits(index: int, n: int) -> list[int]:
    return [bit for bit in range(n) if index & (1 << bit)]


def _scalar_mask(layout: GradeLayout, *, dtype: torch.dtype, device) -> torch.Tensor:
    mask = torch.zeros(layout.dim, dtype=dtype, device=device)
    scalar_position = {index: position for position, index in enumerate(layout.basis_indices)}.get(0)
    if scalar_position is not None:
        mask[scalar_position] = 1.0
    return mask


def _layout_map(source: GradeLayout | None, target: GradeLayout, *, dtype: torch.dtype, device) -> torch.Tensor:
    if source is None:
        return torch.zeros((0, target.dim), dtype=dtype, device=device)

    matrix = torch.zeros((source.dim, target.dim), dtype=dtype, device=device)
    target_positions = {index: position for position, index in enumerate(target.basis_indices)}
    for source_position, index in enumerate(source.basis_indices):
        target_position = target_positions.get(index)
        if target_position is not None:
            matrix[source_position, target_position] = 1.0
    return matrix


def spectral_exp_preselection(
    spec: AlgebraSpec,
    device,
    *,
    dtype: torch.dtype,
    max_planes: int | None = None,
    tol_abs: float | None = None,
    tol_rel: float = 0.0,
    dominant_rel: float | None = None,
    allow_degenerate: bool = True,
    allow_truncated_degenerate: bool = True,
) -> SpectralExpPreselection:
    """Return the static eligibility record for the spectral-local exp path."""
    nondegenerate_dim = spec.p + spec.q
    full_rank_planes = nondegenerate_dim // 2
    if max_planes is not None and int(max_planes) <= 0:
        raise ValueError(f"max_planes must be positive, got {max_planes}")
    resolved_max_planes = (
        min(full_rank_planes, SPECTRAL_LOCAL_MAX_PLANES)
        if max_planes is None
        else min(int(max_planes), full_rank_planes, SPECTRAL_LOCAL_MAX_PLANES)
    )
    if tol_rel < 0.0:
        raise ValueError(f"tol_rel must be non-negative, got {tol_rel}")
    resolved_tol_abs = torch.finfo(dtype).eps * 32.0 if tol_abs is None else float(tol_abs)
    if resolved_tol_abs < 0.0:
        raise ValueError(f"tol_abs must be non-negative, got {tol_abs}")
    resolved_dominant_rel = max(torch.finfo(dtype).eps**0.5, torch.finfo(dtype).eps * 32.0) if dominant_rel is None else float(dominant_rel)
    if resolved_dominant_rel < 0.0:
        raise ValueError(f"dominant_rel must be non-negative, got {dominant_rel}")

    solver_family = "symmetric" if spec.p == 0 or spec.q == 0 else "general_complex"
    if spec.n <= 5:
        return SpectralExpPreselection(
            False,
            "closed_formula_preferred",
            resolved_max_planes,
            resolved_tol_abs,
            float(tol_rel),
            resolved_dominant_rel,
            nondegenerate_dim,
            spec.r,
            solver_family,
        )
    if spec.p > 0 and spec.q > 0:
        return SpectralExpPreselection(
            False,
            "pseudo_euclidean_deferred",
            resolved_max_planes,
            resolved_tol_abs,
            float(tol_rel),
            resolved_dominant_rel,
            nondegenerate_dim,
            spec.r,
            solver_family,
        )
    if spec.r != 0 and not allow_degenerate:
        return SpectralExpPreselection(
            False,
            "degenerate_disabled_by_policy",
            resolved_max_planes,
            resolved_tol_abs,
            float(tol_rel),
            resolved_dominant_rel,
            nondegenerate_dim,
            spec.r,
            solver_family,
        )
    if spec.r > SPECTRAL_LOCAL_MAX_IDEAL_DIM:
        return SpectralExpPreselection(
            False,
            "ideal_dim_exceeds_block_cap",
            resolved_max_planes,
            resolved_tol_abs,
            float(tol_rel),
            resolved_dominant_rel,
            nondegenerate_dim,
            spec.r,
            solver_family,
        )
    if spec.r != 0 and not allow_truncated_degenerate and nondegenerate_dim % 2 != 0:
        return SpectralExpPreselection(
            False,
            "odd_nondegenerate_kernel_deferred",
            resolved_max_planes,
            resolved_tol_abs,
            float(tol_rel),
            resolved_dominant_rel,
            nondegenerate_dim,
            spec.r,
            solver_family,
        )
    if spec.r != 0 and not allow_truncated_degenerate and resolved_max_planes < full_rank_planes:
        return SpectralExpPreselection(
            False,
            "degenerate_block_requires_full_plane_cap",
            resolved_max_planes,
            resolved_tol_abs,
            float(tol_rel),
            resolved_dominant_rel,
            nondegenerate_dim,
            spec.r,
            solver_family,
        )
    if dtype in {torch.float16, torch.bfloat16}:
        return SpectralExpPreselection(
            False,
            "dtype_error_floor_too_high",
            resolved_max_planes,
            resolved_tol_abs,
            float(tol_rel),
            resolved_dominant_rel,
            nondegenerate_dim,
            spec.r,
            solver_family,
        )
    if torch.device(device).type == "mps":
        return SpectralExpPreselection(
            False,
            "mps_solver_deferred",
            resolved_max_planes,
            resolved_tol_abs,
            float(tol_rel),
            resolved_dominant_rel,
            nondegenerate_dim,
            spec.r,
            solver_family,
        )
    if nondegenerate_dim < 2:
        return SpectralExpPreselection(
            False,
            "empty_nondegenerate_block",
            resolved_max_planes,
            resolved_tol_abs,
            float(tol_rel),
            resolved_dominant_rel,
            nondegenerate_dim,
            spec.r,
            solver_family,
        )
    return SpectralExpPreselection(
        True,
        "eligible",
        resolved_max_planes,
        resolved_tol_abs,
        float(tol_rel),
        resolved_dominant_rel,
        nondegenerate_dim,
        spec.r,
        solver_family,
    )


def select_bivector_exp_executor_family(
    spec: AlgebraSpec,
    device,
    *,
    dtype: torch.dtype = torch.float32,
    spectral_max_planes: int | None = None,
    spectral_tol_abs: float | None = None,
    spectral_tol_rel: float = 0.0,
    spectral_dominant_rel: float | None = None,
    spectral_allow_degenerate: bool = True,
    spectral_allow_truncated_degenerate: bool = True,
) -> str:
    """Return the planner-selected bivector-exp executor family."""
    if spec.n <= 3:
        return "closed_simple"
    if spec.n <= 5:
        return "closed_biquadratic"
    spectral = spectral_exp_preselection(
        spec,
        device,
        dtype=dtype,
        max_planes=spectral_max_planes,
        tol_abs=spectral_tol_abs,
        tol_rel=spectral_tol_rel,
        dominant_rel=spectral_dominant_rel,
        allow_degenerate=spectral_allow_degenerate,
        allow_truncated_degenerate=spectral_allow_truncated_degenerate,
    )
    if spectral.eligible:
        return "spectral_local"
    return "left_matrix_exp"
