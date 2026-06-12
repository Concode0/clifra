# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Static bivector exponential plans and executor-family selection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from clifra.core.foundation.basis import basis_index_tuple_for_grades, basis_product, operation_coefficient
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout

SPECTRAL_LOCAL_MAX_PLANES = 4
SPECTRAL_LOCAL_MAX_IDEAL_DIM = 4
SPECTRAL_LOCAL_TRUNCATION_NOTICE = (
    "spectral_local extracts k dominant orthogonal planes. Clipping errors can occur when rotation energy is "
    "evenly distributed; because orthogonal plane rotors commute, the strict tail guard is bounded by the sum of "
    "the clipped tail angles."
)


@dataclass(frozen=True)
class BivectorExpExecutionPolicy:
    """Public policy knobs for planned bivector exponentials.

    ``spectral_local`` keeps the dominant plane spectrum up to ``spectral_max_planes``.
    Use the diagnostics in this module when evenly distributed rotation energy would make
    the clipped tail numerically relevant.

    .. caution::
        Spectral truncation diagnostics (`spectral_exp_angle_diagnostics`) are intended
        for static design-time evaluation or validation/inference logging. 
        Avoid extracting scalar values inside compiled 
        training loops to prevent hardware stream synchronization bottlenecks.
    """

    spectral_max_planes: int | None = None
    spectral_tol_abs: float | None = None
    spectral_tol_rel: float = 0.0
    spectral_dominant_rel: float | None = None
    spectral_transition_n: int = 10
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
    spectral_nondegenerate_generator_input_positions: torch.Tensor
    spectral_nondegenerate_generator_row_positions: torch.Tensor
    spectral_nondegenerate_generator_col_positions: torch.Tensor
    spectral_nondegenerate_generator_coefficients: torch.Tensor
    spectral_mixed_generator_input_positions: torch.Tensor
    spectral_mixed_generator_row_positions: torch.Tensor
    spectral_mixed_generator_col_positions: torch.Tensor
    spectral_mixed_generator_coefficients: torch.Tensor
    spectral_plane_bivector_map: torch.Tensor
    spectral_plane_eye: torch.Tensor
    spectral_plane_left_positions: torch.Tensor
    spectral_plane_right_positions: torch.Tensor
    spectral_plane_output_positions: torch.Tensor
    spectral_plane_coefficients: torch.Tensor
    spectral_plane_to_local: torch.Tensor
    spectral_nilpotent_input_positions: torch.Tensor
    spectral_nilpotent_local_positions: torch.Tensor
    spectral_ideal_basis: torch.Tensor
    spectral_lift_grades: torch.Tensor
    spectral_lift_grade_values: tuple[int, ...]
    spectral_lift_local_positions: torch.Tensor
    spectral_lift_local_axes: torch.Tensor
    spectral_lift_local_mask: torch.Tensor
    spectral_lift_target_axes: torch.Tensor
    spectral_lift_target_positions: torch.Tensor
    spectral_lift_target_mask: torch.Tensor
    spectral_lift_output_dim: int
    spectral_tolerances: torch.Tensor
    spectral_tol_abs: float
    spectral_tol_rel: float
    spectral_dominant_rel: float
    spectral_transition_n: int
    spectral_allow_degenerate: bool
    spectral_allow_truncated_degenerate: bool
    nondegenerate_dim: int
    ideal_dim: int
    spectral_local_axis_count: int
    eps: float
    eps_sq: float


@dataclass(frozen=True)
class SpectralExpPreselection:
    """Static eligibility record for the spectral-local exp path.

    See :data:`SPECTRAL_LOCAL_TRUNCATION_NOTICE` for the precision caveat when
    ``max_planes`` is smaller than the full orthogonal plane count.
    """

    eligible: bool
    reason: str
    max_planes: int
    tol_abs: float
    tol_rel: float
    dominant_rel: float
    nondegenerate_dim: int
    ideal_dim: int
    solver_family: str


@dataclass(frozen=True)
class SpectralExpAngleDiagnostics:
    """Runtime truncation diagnostics for a solver angle spectrum."""

    selected_planes: int
    total_planes: int
    sorted_abs_angles: torch.Tensor
    tail_angle_sum_bound: torch.Tensor
    geometric_variance_captured: torch.Tensor
    truncates: bool
    notice: str = SPECTRAL_LOCAL_TRUNCATION_NOTICE


@dataclass(frozen=True)
class SpectralExpUniformTailStress:
    """Static uniform-energy stress row for spectral-local truncation."""

    spec: AlgebraSpec
    max_planes: int
    total_planes: int
    selected_planes: int
    clipped_planes: int
    bivector_norm: float
    uniform_angle: float
    tail_angle_sum_bound: float
    geometric_variance_captured: float
    truncates: bool
    notice: str = SPECTRAL_LOCAL_TRUNCATION_NOTICE


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
    spectral_transition_n: int = 10,
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
    finfo = torch.finfo(dtype)
    preselection = spectral_exp_preselection(
        spec,
        resolved_device,
        dtype=dtype,
        max_planes=spectral_max_planes,
        tol_abs=spectral_tol_abs,
        tol_rel=spectral_tol_rel,
        dominant_rel=spectral_dominant_rel,
        transition_n=spectral_transition_n,
        allow_degenerate=spectral_allow_degenerate,
        allow_truncated_degenerate=spectral_allow_truncated_degenerate,
    )
    executor_family = _executor_family_from_preselection(spec, preselection, resolved_device)
    buffer_device = torch.device("cpu") if executor_family == "cpu_matrix_exp" else resolved_device
    grade4_layout = spec.layout((4,)) if executor_family == "closed_biquadratic" else None
    operator_layout = spec.layout((0,)) if executor_family == "spectral_local" else spec.layout(range(0, spec.n + 1, 2))
    operator_position_by_index = {index: position for position, index in enumerate(operator_layout.basis_indices)}
    metric_signs = _metric_signs(spec, dtype=dtype, device=buffer_device)
    partition = _bivector_axis_partition(input_layout, spec, device=buffer_device)
    if executor_family == "spectral_local":
        local_buffers = _spectral_local_buffers(
            spec,
            input_layout,
            output_layout,
            preselection.max_planes,
            ideal_dim=preselection.ideal_dim,
            dtype=dtype,
            device=buffer_device,
        )
        spectral_nondegenerate_entries = _bivector_to_nondegenerate_generator_entries(
            input_layout,
            spec,
            dtype=dtype,
            device=buffer_device,
        )
        spectral_mixed_entries = _bivector_to_mixed_generator_entries(
            input_layout,
            spec,
            dtype=dtype,
            device=buffer_device,
        )
        bivector_to_nondegenerate_generator = torch.zeros((0, 0, 0), dtype=dtype, device=buffer_device)
        bivector_to_mixed_generator = torch.zeros((0, 0, 0), dtype=dtype, device=buffer_device)
        bivector_to_output = torch.zeros((0, 0), dtype=dtype, device=buffer_device)
        bivector_to_operator = torch.zeros((0, 0), dtype=dtype, device=buffer_device)
        operator_to_output = torch.zeros((0, 0), dtype=dtype, device=buffer_device)
    else:
        local_buffers = _empty_spectral_local_buffers(spec, input_layout, output_layout, dtype=dtype, device=buffer_device)
        spectral_nondegenerate_entries = _empty_generator_entries(dtype=dtype, device=buffer_device)
        spectral_mixed_entries = _empty_generator_entries(dtype=dtype, device=buffer_device)
        bivector_to_nondegenerate_generator = _bivector_to_nondegenerate_generator(
            input_layout,
            spec,
            dtype=dtype,
            device=buffer_device,
        )
        bivector_to_mixed_generator = _bivector_to_mixed_generator(
            input_layout,
            spec,
            dtype=dtype,
            device=buffer_device,
        )
        bivector_to_output = _layout_map(input_layout, output_layout, dtype=dtype, device=buffer_device)
        bivector_to_operator = _layout_map(input_layout, operator_layout, dtype=dtype, device=buffer_device)
        operator_to_output = _layout_map(operator_layout, output_layout, dtype=dtype, device=buffer_device)

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
        executor_family=executor_family,
        metric_signs=metric_signs,
        bivector_squared_signs=torch.tensor(signs, dtype=dtype, device=buffer_device),
        nondegenerate_bivector_positions=partition[0],
        mixed_degenerate_bivector_positions=partition[1],
        nilpotent_bivector_positions=partition[2],
        bivector_to_nondegenerate_generator=bivector_to_nondegenerate_generator,
        bivector_to_mixed_generator=bivector_to_mixed_generator,
        output_scalar_mask=_scalar_mask(output_layout, dtype=dtype, device=buffer_device),
        operator_scalar_mask=_scalar_mask(operator_layout, dtype=dtype, device=buffer_device),
        bivector_to_output=bivector_to_output,
        bivector_to_operator=bivector_to_operator,
        grade4_to_output=_layout_map(grade4_layout, output_layout, dtype=dtype, device=buffer_device),
        operator_to_output=operator_to_output,
        operator_eye=torch.eye(operator_layout.dim, dtype=dtype, device=buffer_device),
        operator_scalar_position=operator_position_by_index[0],
        spectral_max_planes=preselection.max_planes,
        spectral_local_scalar_mask=local_buffers["scalar_mask"],
        spectral_local_plane_masks=local_buffers["plane_masks"],
        spectral_local_product_table=local_buffers["product_table"],
        spectral_local_sparse_left_positions=local_buffers["sparse_left_positions"],
        spectral_local_sparse_right_positions=local_buffers["sparse_right_positions"],
        spectral_local_sparse_output_positions=local_buffers["sparse_output_positions"],
        spectral_local_sparse_coefficients=local_buffers["sparse_coefficients"],
        spectral_nondegenerate_generator_input_positions=spectral_nondegenerate_entries["input_positions"],
        spectral_nondegenerate_generator_row_positions=spectral_nondegenerate_entries["row_positions"],
        spectral_nondegenerate_generator_col_positions=spectral_nondegenerate_entries["col_positions"],
        spectral_nondegenerate_generator_coefficients=spectral_nondegenerate_entries["coefficients"],
        spectral_mixed_generator_input_positions=spectral_mixed_entries["input_positions"],
        spectral_mixed_generator_row_positions=spectral_mixed_entries["row_positions"],
        spectral_mixed_generator_col_positions=spectral_mixed_entries["col_positions"],
        spectral_mixed_generator_coefficients=spectral_mixed_entries["coefficients"],
        spectral_plane_bivector_map=local_buffers["plane_bivector_map"],
        spectral_plane_eye=local_buffers["plane_eye"],
        spectral_plane_left_positions=local_buffers["plane_left_positions"],
        spectral_plane_right_positions=local_buffers["plane_right_positions"],
        spectral_plane_output_positions=local_buffers["plane_output_positions"],
        spectral_plane_coefficients=local_buffers["plane_coefficients"],
        spectral_plane_to_local=local_buffers["plane_to_local"],
        spectral_nilpotent_input_positions=local_buffers["nilpotent_input_positions"],
        spectral_nilpotent_local_positions=local_buffers["nilpotent_local_positions"],
        spectral_ideal_basis=local_buffers["ideal_basis"],
        spectral_lift_grades=local_buffers["lift_grades"],
        spectral_lift_grade_values=local_buffers["lift_grade_values"],
        spectral_lift_local_positions=local_buffers["lift_local_positions"],
        spectral_lift_local_axes=local_buffers["lift_local_axes"],
        spectral_lift_local_mask=local_buffers["lift_local_mask"],
        spectral_lift_target_axes=local_buffers["lift_target_axes"],
        spectral_lift_target_positions=local_buffers["lift_target_positions"],
        spectral_lift_target_mask=local_buffers["lift_target_mask"],
        spectral_lift_output_dim=output_layout.dim,
        spectral_tolerances=torch.tensor(
            [preselection.tol_abs, preselection.tol_rel, preselection.dominant_rel],
            dtype=dtype,
            device=resolved_device,
        ),
        spectral_tol_abs=preselection.tol_abs,
        spectral_tol_rel=preselection.tol_rel,
        spectral_dominant_rel=preselection.dominant_rel,
        spectral_transition_n=spectral_transition_n,
        spectral_allow_degenerate=spectral_allow_degenerate,
        spectral_allow_truncated_degenerate=spectral_allow_truncated_degenerate,
        nondegenerate_dim=preselection.nondegenerate_dim,
        ideal_dim=preselection.ideal_dim,
        spectral_local_axis_count=int(local_buffers["axis_count"]),
        eps=float(finfo.eps),
        eps_sq=float(finfo.eps**2),
    )


def _executor_family_from_preselection(spec: AlgebraSpec, preselection: SpectralExpPreselection, device) -> str:
    if spec.n <= 3:
        return "closed_simple"
    if spec.n <= 5:
        return "closed_biquadratic"
    if preselection.eligible:
        return "spectral_local"
    if torch.device(device).type == "mps" and preselection.reason.endswith("cpu_matrix_exp"):
        return "cpu_matrix_exp"
    return "left_matrix_exp"


def _empty_spectral_local_buffers(
    spec: AlgebraSpec,
    input_layout: GradeLayout,
    output_layout: GradeLayout,
    *,
    dtype: torch.dtype,
    device,
) -> dict[str, torch.Tensor | tuple[int, ...] | int]:
    empty_sparse = _empty_sparse_product_buffers(dtype=dtype, device=device)
    return {
        "axis_count": 0,
        "scalar_mask": torch.ones(1, dtype=dtype, device=device),
        "plane_masks": torch.zeros((0, 1), dtype=dtype, device=device),
        "product_table": torch.zeros((1, 1, 1), dtype=dtype, device=device),
        "sparse_left_positions": empty_sparse["left_positions"],
        "sparse_right_positions": empty_sparse["right_positions"],
        "sparse_output_positions": empty_sparse["output_positions"],
        "sparse_coefficients": empty_sparse["coefficients"],
        "plane_bivector_map": torch.zeros((1, 1), dtype=dtype, device=device),
        "plane_eye": torch.eye(1, dtype=dtype, device=device),
        "plane_left_positions": empty_sparse["left_positions"],
        "plane_right_positions": empty_sparse["right_positions"],
        "plane_output_positions": empty_sparse["output_positions"],
        "plane_coefficients": empty_sparse["coefficients"],
        "plane_to_local": torch.zeros((0, 1, 1), dtype=dtype, device=device),
        "nilpotent_input_positions": torch.zeros(0, dtype=torch.long, device=device),
        "nilpotent_local_positions": torch.zeros(0, dtype=torch.long, device=device),
        "ideal_basis": torch.zeros((0, spec.n), dtype=dtype, device=device),
        "lift_grades": torch.zeros(1, dtype=torch.long, device=device),
        "lift_grade_values": (0,),
        "lift_local_positions": torch.zeros((1, 1), dtype=torch.long, device=device),
        "lift_local_axes": torch.zeros((1, 1, 0), dtype=torch.long, device=device),
        "lift_local_mask": torch.zeros((1, 1), dtype=dtype, device=device),
        "lift_target_axes": torch.zeros((1, 1, 0), dtype=torch.long, device=device),
        "lift_target_positions": torch.zeros((1, 1), dtype=torch.long, device=device),
        "lift_target_mask": torch.zeros((1, 1), dtype=dtype, device=device),
    }


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
        product_table = torch.zeros((local_dim, local_dim, local_dim), dtype=dtype, device=device)
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
    lift_target_positions = torch.zeros((grade_count, max_target), dtype=torch.long, device=device)
    lift_target_mask = torch.zeros((grade_count, max_target), dtype=dtype, device=device)

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
            lift_target_positions[grade_index, target_slot] = target_position
            lift_target_mask[grade_index, target_slot] = lift_sign
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
    nilpotent_input_positions, nilpotent_local_positions = _nilpotent_to_local_positions(
        input_layout,
        local_positions,
        local_nondegenerate_dim,
        int(ideal_dim),
        spec,
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
        "nilpotent_input_positions": nilpotent_input_positions,
        "nilpotent_local_positions": nilpotent_local_positions,
        "ideal_basis": ideal_basis,
        "lift_grades": lift_grades,
        "lift_grade_values": tuple(grades),
        "lift_local_positions": lift_local_positions,
        "lift_local_axes": lift_local_axes,
        "lift_local_mask": lift_local_mask,
        "lift_target_axes": lift_target_axes,
        "lift_target_positions": lift_target_positions,
        "lift_target_mask": lift_target_mask,
    }


def _empty_sparse_product_buffers(*, dtype: torch.dtype, device) -> dict[str, torch.Tensor]:
    return {
        "left_positions": torch.zeros(0, dtype=torch.long, device=device),
        "right_positions": torch.zeros(0, dtype=torch.long, device=device),
        "output_positions": torch.zeros(0, dtype=torch.long, device=device),
        "coefficients": torch.zeros(0, dtype=dtype, device=device),
    }


def _empty_generator_entries(*, dtype: torch.dtype, device) -> dict[str, torch.Tensor]:
    return {
        "input_positions": torch.zeros(0, dtype=torch.long, device=device),
        "row_positions": torch.zeros(0, dtype=torch.long, device=device),
        "col_positions": torch.zeros(0, dtype=torch.long, device=device),
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
    basis = torch.tensor(basis_indices, dtype=torch.long)
    dim = basis.numel()
    lookup = torch.full((1 << int(axis_count),), -1, dtype=torch.long)
    lookup[basis] = torch.arange(dim, dtype=torch.long)
    parity_lookup = torch.tensor([index.bit_count() & 1 for index in range(1 << int(axis_count))], dtype=torch.bool)
    right_indices = basis.view(1, dim)
    right_positions_template = torch.arange(dim, dtype=torch.long).view(1, dim)
    negative_mask = sum(1 << bit for bit in range(int(p), int(p) + int(q)))
    null_mask = sum(1 << bit for bit in range(int(p) + int(q), int(axis_count)))

    left_position_chunks: list[torch.Tensor] = []
    right_position_chunks: list[torch.Tensor] = []
    output_position_chunks: list[torch.Tensor] = []
    coefficient_chunks: list[torch.Tensor] = []
    rows_per_chunk = max(1, min(dim, 262_144 // max(dim, 1)))
    for start in range(0, dim, rows_per_chunk):
        stop = min(start + rows_per_chunk, dim)
        left_indices = basis[start:stop].view(-1, 1)
        outputs = torch.bitwise_xor(left_indices, right_indices)
        valid = lookup[outputs] >= 0
        if null_mask:
            valid = valid & (torch.bitwise_and(torch.bitwise_and(left_indices, right_indices), null_mask) == 0)
        swap_parity = torch.zeros_like(outputs, dtype=torch.bool)
        for bit in range(int(axis_count)):
            left_has_bit = torch.bitwise_and(left_indices, 1 << bit) != 0
            lower_right_parity = parity_lookup[torch.bitwise_and(right_indices, (1 << bit) - 1)]
            swap_parity = swap_parity ^ (left_has_bit & lower_right_parity)
        if negative_mask:
            metric_parity = parity_lookup[torch.bitwise_and(torch.bitwise_and(left_indices, right_indices), negative_mask)]
            swap_parity = swap_parity ^ metric_parity

        left_positions = torch.arange(start, stop, dtype=torch.long).view(-1, 1).expand(-1, dim)
        right_positions = right_positions_template.expand(stop - start, -1)
        left_position_chunks.append(left_positions[valid])
        right_position_chunks.append(right_positions[valid])
        output_position_chunks.append(lookup[outputs][valid])
        coefficient_chunks.append(torch.where(swap_parity[valid], -torch.ones((), dtype=dtype), torch.ones((), dtype=dtype)))

    left_positions = torch.cat(left_position_chunks) if left_position_chunks else torch.zeros(0, dtype=torch.long)
    right_positions = torch.cat(right_position_chunks) if right_position_chunks else torch.zeros(0, dtype=torch.long)
    output_positions = torch.cat(output_position_chunks) if output_position_chunks else torch.zeros(0, dtype=torch.long)
    coefficients = torch.cat(coefficient_chunks) if coefficient_chunks else torch.zeros(0, dtype=dtype)
    return {
        "left_positions": left_positions.to(device=device),
        "right_positions": right_positions.to(device=device),
        "output_positions": output_positions.to(device=device),
        "coefficients": coefficients.to(device=device),
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


def _nilpotent_to_local_positions(
    input_layout: GradeLayout,
    local_positions: dict[int, int],
    local_nondegenerate_dim: int,
    ideal_dim: int,
    spec: AlgebraSpec,
    *,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if ideal_dim < 2:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty
    nondegenerate_dim = spec.p + spec.q
    input_positions: list[int] = []
    output_positions: list[int] = []
    for input_position, input_index in enumerate(input_layout.basis_indices):
        bits = _basis_bits(input_index, spec.n)
        if len(bits) != 2 or bits[0] < nondegenerate_dim or bits[1] < nondegenerate_dim:
            continue
        local_index = 0
        for bit in bits:
            local_index |= 1 << (local_nondegenerate_dim + bit - nondegenerate_dim)
        local_position = local_positions.get(local_index)
        if local_position is not None:
            input_positions.append(input_position)
            output_positions.append(local_position)
    return (
        torch.tensor(input_positions, dtype=torch.long, device=device),
        torch.tensor(output_positions, dtype=torch.long, device=device),
    )


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


def _bivector_to_nondegenerate_generator_entries(
    input_layout: GradeLayout,
    spec: AlgebraSpec,
    *,
    dtype: torch.dtype,
    device,
) -> dict[str, torch.Tensor]:
    nondegenerate_dim = spec.p + spec.q
    vector_indices = [1 << axis for axis in range(nondegenerate_dim)]
    vector_positions = {index: position for position, index in enumerate(vector_indices)}
    input_positions: list[int] = []
    row_positions: list[int] = []
    col_positions: list[int] = []
    coefficients: list[float] = []

    for bivector_position, bivector_index in enumerate(input_layout.basis_indices):
        bits = _basis_bits(bivector_index, spec.n)
        if len(bits) != 2 or bits[0] >= nondegenerate_dim or bits[1] >= nondegenerate_dim:
            continue
        for input_position, input_index in enumerate(vector_indices):
            output_index = bivector_index ^ input_index
            output_position = vector_positions.get(output_index)
            if output_position is None:
                continue
            input_positions.append(bivector_position)
            row_positions.append(output_position)
            col_positions.append(input_position)
            coefficients.append(_vector_generator_coefficient(bivector_index, input_index, spec))

    return {
        "input_positions": torch.tensor(input_positions, dtype=torch.long, device=device),
        "row_positions": torch.tensor(row_positions, dtype=torch.long, device=device),
        "col_positions": torch.tensor(col_positions, dtype=torch.long, device=device),
        "coefficients": torch.tensor(coefficients, dtype=dtype, device=device),
    }


def _bivector_to_mixed_generator_entries(
    input_layout: GradeLayout,
    spec: AlgebraSpec,
    *,
    dtype: torch.dtype,
    device,
) -> dict[str, torch.Tensor]:
    nondegenerate_dim = spec.p + spec.q
    if spec.r == 0 or nondegenerate_dim == 0:
        return _empty_generator_entries(dtype=dtype, device=device)

    nondegenerate_indices = [1 << axis for axis in range(nondegenerate_dim)]
    ideal_positions = {1 << (nondegenerate_dim + axis): axis for axis in range(spec.r)}
    input_positions: list[int] = []
    row_positions: list[int] = []
    col_positions: list[int] = []
    coefficients: list[float] = []

    for bivector_position, bivector_index in enumerate(input_layout.basis_indices):
        bits = _basis_bits(bivector_index, spec.n)
        if len(bits) != 2 or not (bits[0] < nondegenerate_dim <= bits[1]):
            continue
        for input_position, input_index in enumerate(nondegenerate_indices):
            output_index = bivector_index ^ input_index
            output_position = ideal_positions.get(output_index)
            if output_position is None:
                continue
            input_positions.append(bivector_position)
            row_positions.append(output_position)
            col_positions.append(input_position)
            coefficients.append(_vector_generator_coefficient(bivector_index, input_index, spec))

    return {
        "input_positions": torch.tensor(input_positions, dtype=torch.long, device=device),
        "row_positions": torch.tensor(row_positions, dtype=torch.long, device=device),
        "col_positions": torch.tensor(col_positions, dtype=torch.long, device=device),
        "coefficients": torch.tensor(coefficients, dtype=dtype, device=device),
    }


def spectral_exp_angle_diagnostics(
    angle_spectrum: torch.Tensor,
    *,
    max_planes: int = SPECTRAL_LOCAL_MAX_PLANES,
    sort: bool = True,
) -> SpectralExpAngleDiagnostics:
    """Return tail-bound and GVC diagnostics for an actual angle spectrum.

    The strict tail guard follows ``||exp(B_tail) - I||_inf <= sum(abs(theta_tail))``.
    ``geometric_variance_captured`` is the retained squared angle energy divided by
    the total squared angle energy. By default, angles are sorted by absolute value
    before selecting the dominant planes.

    .. note::
        This function internally detaches the input tensor to ensure diagnostics are
         not tracked in autograd.
    """
    if angle_spectrum.ndim == 0:
        raise ValueError("angle_spectrum must have at least one dimension")
    if int(max_planes) <= 0:
        raise ValueError(f"max_planes must be positive, got {max_planes}")

    angles = angle_spectrum.detach().abs()
    if sort:
        angles = torch.sort(angles, dim=-1, descending=True).values

    total_planes = int(angles.shape[-1])
    selected_planes = min(int(max_planes), total_planes, SPECTRAL_LOCAL_MAX_PLANES)
    kept = angles[..., :selected_planes]
    tail = angles[..., selected_planes:]
    kept_energy = (kept * kept).sum(dim=-1)
    total_energy = (angles * angles).sum(dim=-1)
    gvc = torch.where(total_energy > 0.0, kept_energy / total_energy, torch.ones_like(total_energy))
    return SpectralExpAngleDiagnostics(
        selected_planes=selected_planes,
        total_planes=total_planes,
        sorted_abs_angles=angles,
        tail_angle_sum_bound=tail.sum(dim=-1),
        geometric_variance_captured=gvc,
        truncates=selected_planes < total_planes,
    )


def spectral_exp_uniform_tail_stress(
    signatures: Iterable[AlgebraSpec | tuple[int, int] | tuple[int, int, int]],
    *,
    max_planes: int = SPECTRAL_LOCAL_MAX_PLANES,
    bivector_norm: float = 1.0,
) -> tuple[SpectralExpUniformTailStress, ...]:
    """Estimate worst-case spectral-local clipping for static signatures.

    This static stress model assumes angle energy is uniformly distributed across
    ``M = n // 2`` orthogonal planes, so each angle has magnitude
    ``abs(bivector_norm) / sqrt(M)``. It is intentionally conservative for
    degenerate signatures; use :func:`spectral_exp_angle_diagnostics` on the
    actual solver spectrum for runtime supervision.
    """
    if int(max_planes) <= 0:
        raise ValueError(f"max_planes must be positive, got {max_planes}")
    if float(bivector_norm) < 0.0:
        raise ValueError(f"bivector_norm must be non-negative, got {bivector_norm}")

    rows: list[SpectralExpUniformTailStress] = []
    for signature in signatures:
        spec = _coerce_algebra_spec(signature)
        total_planes = spec.n // 2
        selected_planes = min(int(max_planes), total_planes, SPECTRAL_LOCAL_MAX_PLANES)
        clipped_planes = max(total_planes - selected_planes, 0)
        if total_planes == 0:
            uniform_angle = 0.0
            tail_bound = 0.0
            gvc = 1.0
        else:
            uniform_angle = abs(float(bivector_norm)) / float(total_planes) ** 0.5
            tail_bound = float(clipped_planes) * uniform_angle
            gvc = 1.0 if float(bivector_norm) == 0.0 else float(selected_planes) / float(total_planes)
        rows.append(
            SpectralExpUniformTailStress(
                spec=spec,
                max_planes=int(max_planes),
                total_planes=total_planes,
                selected_planes=selected_planes,
                clipped_planes=clipped_planes,
                bivector_norm=float(bivector_norm),
                uniform_angle=uniform_angle,
                tail_angle_sum_bound=tail_bound,
                geometric_variance_captured=gvc,
                truncates=clipped_planes > 0,
            )
        )
    return tuple(rows)


def format_spectral_exp_uniform_tail_stress(rows: Iterable[SpectralExpUniformTailStress]) -> str:
    """Format static spectral-local truncation stress rows as a compact table."""
    header = "signature | planes | kept | clipped | theta_uniform | tail_bound | GVC"
    lines = [header, "-" * len(header)]
    for row in rows:
        signature = f"Cl({row.spec.p},{row.spec.q},{row.spec.r})"
        lines.append(
            f"{signature} | {row.total_planes} | {row.selected_planes} | {row.clipped_planes} | "
            f"{row.uniform_angle:.6g} | {row.tail_angle_sum_bound:.6g} | "
            f"{row.geometric_variance_captured:.6g}"
        )
    return "\n".join(lines)


def _coerce_algebra_spec(signature: AlgebraSpec | tuple[int, int] | tuple[int, int, int]) -> AlgebraSpec:
    if isinstance(signature, AlgebraSpec):
        return signature
    if len(signature) == 2:
        return AlgebraSpec(int(signature[0]), int(signature[1]), 0)
    if len(signature) == 3:
        return AlgebraSpec(int(signature[0]), int(signature[1]), int(signature[2]))
    raise ValueError(f"signature must be AlgebraSpec, (p, q), or (p, q, r), got {signature!r}")


def spectral_exp_preselection(
    spec: AlgebraSpec,
    device,
    *,
    dtype: torch.dtype,
    max_planes: int | None = None,
    tol_abs: float | None = None,
    tol_rel: float = 0.0,
    dominant_rel: float | None = None,
    transition_n: int = 10,
    allow_degenerate: bool = True,
    allow_truncated_degenerate: bool = True,
) -> SpectralExpPreselection:
    """Return the static eligibility record for the spectral-local exp path."""
    nondegenerate_dim = spec.p + spec.q
    full_rank_planes = nondegenerate_dim // 2
    if max_planes is not None and int(max_planes) <= 0:
        raise ValueError(f"max_planes must be positive, got {max_planes}")
    resolved_transition_n = int(transition_n)
    if resolved_transition_n <= 0:
        raise ValueError(f"transition_n must be positive, got {transition_n}")
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

    device_type = torch.device(device).type
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
        reason = "pseudo_euclidean_mps_cpu_matrix_exp" if device_type == "mps" else "pseudo_euclidean_matrix_exp"
        return SpectralExpPreselection(
            False,
            reason,
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
    if device_type != "mps" and spec.n < resolved_transition_n:
        return SpectralExpPreselection(
            False,
            "matrix_exp_below_spectral_transition",
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
    spectral_transition_n: int = 10,
    spectral_allow_degenerate: bool = True,
    spectral_allow_truncated_degenerate: bool = True,
) -> str:
    """Return the planner-selected bivector-exp executor family."""
    spectral = spectral_exp_preselection(
        spec,
        device,
        dtype=dtype,
        max_planes=spectral_max_planes,
        tol_abs=spectral_tol_abs,
        tol_rel=spectral_tol_rel,
        dominant_rel=spectral_dominant_rel,
        transition_n=spectral_transition_n,
        allow_degenerate=spectral_allow_degenerate,
        allow_truncated_degenerate=spectral_allow_truncated_degenerate,
    )
    return _executor_family_from_preselection(spec, spectral, device)
