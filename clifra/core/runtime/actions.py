"""Runtime actions shared by full-kernel and active-lane algebra hosts."""

from __future__ import annotations

import torch

from clifra.core.foundation.layout import GradeLayout
from clifra.core.foundation.module import is_dense_kernel_host, require_dense_kernel_host
from clifra.core.foundation.numerics import eps_like
from clifra.core.foundation.validation import check_multivector
from clifra.core.planning.action import (
    apply_multi_graded_linear_action,
    bivector_vector_generator,
    metric_self_signs,
    reflection_vector_matrix,
)
from clifra.core.runtime.accessors import materialize_full


def apply_versor_action(
    algebra,
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    grade: int,
    input_grades=None,
    output_grades=None,
    input_layout: GradeLayout | None = None,
    output_layout: GradeLayout | None = None,
    parameter_layout: GradeLayout | None = None,
    active_output: bool = False,
    channels: int | None = None,
    name: str = "versor_action",
    dense_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    cache_dense: bool = False,
    return_cache: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Apply one versor action while the algebra host chooses the lane execution path."""
    input_layout, output_layout, parameter_layout = _action_layouts(
        algebra,
        grade=grade,
        input_grades=input_grades,
        output_grades=output_grades,
        input_layout=input_layout,
        output_layout=output_layout,
        parameter_layout=parameter_layout,
    )

    input_active_lanes = _validate_action_values(
        algebra,
        values,
        layout=input_layout,
        channels=channels,
        name=name,
    )
    if input_active_lanes:
        output = compact_versor_action(
            algebra,
            values,
            weights,
            grade=grade,
            input_layout=input_layout,
            output_layout=output_layout,
            parameter_layout=parameter_layout,
            active_output=active_output,
        )
        return (output, dense_cache) if return_cache else output

    _require_dense_action(algebra, name)
    left, right, next_cache = _dense_versor_factors(
        algebra,
        values,
        weights,
        grade=grade,
        parameter_layout=parameter_layout,
        dense_cache=dense_cache,
        cache_dense=cache_dense,
    )
    output = algebra.per_channel_sandwich(left, values, right)
    output = _project_dense_action_output(algebra, output, output_layout=output_layout, active_output=active_output)
    return (output, next_cache) if return_cache else output


def apply_multi_versor_action(
    algebra,
    values: torch.Tensor,
    weights: torch.Tensor,
    mix: torch.Tensor,
    *,
    grade: int,
    input_grades=None,
    output_grades=None,
    input_layout: GradeLayout | None = None,
    output_layout: GradeLayout | None = None,
    parameter_layout: GradeLayout | None = None,
    active_output: bool = False,
    channels: int | None = None,
    name: str = "multi_versor_action",
    dense_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    cache_dense: bool = False,
    return_cache: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Apply a weighted versor superposition with host-owned lane dispatch."""
    input_layout, output_layout, parameter_layout = _action_layouts(
        algebra,
        grade=grade,
        input_grades=input_grades,
        output_grades=output_grades,
        input_layout=input_layout,
        output_layout=output_layout,
        parameter_layout=parameter_layout,
    )

    input_active_lanes = _validate_action_values(
        algebra,
        values,
        layout=input_layout,
        channels=channels,
        name=name,
    )
    if input_active_lanes:
        output = compact_multi_versor_action(
            algebra,
            values,
            weights,
            mix,
            grade=grade,
            input_layout=input_layout,
            output_layout=output_layout,
            parameter_layout=parameter_layout,
            active_output=active_output,
        )
        return (output, dense_cache) if return_cache else output

    _require_dense_action(algebra, name)
    left, right, next_cache = _dense_versor_factors(
        algebra,
        values,
        weights,
        grade=grade,
        parameter_layout=parameter_layout,
        dense_cache=dense_cache,
        cache_dense=cache_dense,
    )
    versored = algebra.multi_rotor_sandwich(left, values, right)
    output = torch.einsum("ck,...cke->...ce", mix.to(device=values.device, dtype=values.dtype), versored)
    output = _project_dense_action_output(algebra, output, output_layout=output_layout, active_output=active_output)
    return (output, next_cache) if return_cache else output


def grade_norms(
    algebra,
    values: torch.Tensor,
    *,
    input_grades=None,
    layout: GradeLayout | None = None,
) -> torch.Tensor:
    """Return per-grade coefficient norms for dense or compact values."""
    layout = _declared_layout(algebra, input_grades, layout)
    input_active_lanes = _values_use_active_lanes(algebra, values, layout)
    if input_active_lanes:
        return compact_grade_norms(algebra, values, layout)
    if is_dense_kernel_host(algebra):
        return algebra.get_grade_norms(values)

    check_multivector(values, algebra, "grade_norms(values)")
    full_layout = algebra.layout(range(algebra.num_grades))
    return compact_grade_norms(algebra, values, full_layout)


def compact_versor_action(
    algebra,
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    grade: int,
    input_layout: GradeLayout,
    output_layout: GradeLayout,
    parameter_layout: GradeLayout,
    active_output: bool,
) -> torch.Tensor:
    """Apply one compact versor action to layer values."""
    matrix = versor_vector_matrix(
        algebra,
        weights.to(device=values.device, dtype=values.dtype),
        grade=grade,
        parameter_layout=parameter_layout,
    )
    return algebra.planned_linear_action(
        values,
        matrix,
        input_layout=input_layout,
        output_layout=output_layout,
        input_active_lanes=True,
        active_output=active_output,
    )


def compact_multi_versor_action(
    algebra,
    values: torch.Tensor,
    weights: torch.Tensor,
    mix: torch.Tensor,
    *,
    grade: int,
    input_layout: GradeLayout,
    output_layout: GradeLayout,
    parameter_layout: GradeLayout,
    active_output: bool,
) -> torch.Tensor:
    """Apply a weighted compact superposition of versor actions."""
    matrices = versor_vector_matrix(
        algebra,
        weights.to(device=values.device, dtype=values.dtype),
        grade=grade,
        parameter_layout=parameter_layout,
    )
    mix = mix.to(device=values.device, dtype=values.dtype)
    if mix.shape != (values.shape[-2], matrices.shape[0]):
        raise ValueError(f"mix shape must be {(values.shape[-2], matrices.shape[0])}, got {tuple(mix.shape)}")

    transformed = apply_multi_graded_linear_action(
        values,
        matrices,
        input_layout=input_layout,
        output_layout=output_layout,
    )
    result = torch.einsum("ck,...cko->...co", mix, transformed)
    if active_output:
        return result
    return materialize_full(algebra, result, layout=output_layout)


def versor_vector_matrix(algebra, weights: torch.Tensor, *, grade: int, parameter_layout: GradeLayout) -> torch.Tensor:
    """Return the vector-space matrix represented by compact versor weights."""
    grade = int(grade)
    if grade == 2:
        return torch.matrix_exp(bivector_vector_generator(weights, bivector_layout=parameter_layout))
    if grade == 1:
        signs = parameter_layout_signs(parameter_layout, device=weights.device, dtype=weights.dtype)
        norm_sq = (weights * weights * signs).sum(dim=-1, keepdim=True)
        scale = norm_sq.abs().clamp_min(eps_like(norm_sq)).sqrt()
        normals = weights / scale
        return reflection_vector_matrix(normals, vector_layout=parameter_layout, eps=algebra.eps_sq)
    raise ValueError("compact versor execution currently supports grade=1 and grade=2")


def parameter_layout_signs(layout: GradeLayout, *, device=None, dtype=None) -> torch.Tensor:
    """Return basis self-product signs for compact parameter weights."""
    return metric_self_signs(layout, device=device, dtype=dtype)


def compact_grade_norms(algebra, values: torch.Tensor, layout: GradeLayout) -> torch.Tensor:
    """Return per-grade coefficient norms for compact values."""
    flat = values.pow(2).reshape(-1, layout.dim)
    grade_ids = layout.grade_indices_tensor(device=values.device).unsqueeze(0).expand_as(flat)
    result = values.new_zeros(flat.shape[0], algebra.num_grades)
    result.scatter_add_(1, grade_ids, flat)
    return result.reshape(*values.shape[:-1], algebra.num_grades).clamp(min=algebra.eps).sqrt()


def dense_versor_factors(
    algebra,
    weights: torch.Tensor,
    *,
    grade: int,
    parameter_layout: GradeLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dense left/right factors for a parameterized versor."""
    versor = parameter_layout.dense(weights)
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


def _dense_versor_factors(
    algebra,
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    grade: int,
    parameter_layout: GradeLayout,
    dense_cache: tuple[torch.Tensor, torch.Tensor] | None,
    cache_dense: bool,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    if _cache_matches(dense_cache, values):
        left, right = dense_cache
    else:
        dense_weights = weights.to(device=values.device, dtype=values.dtype)
        left, right = dense_versor_factors(
            algebra,
            dense_weights,
            grade=grade,
            parameter_layout=parameter_layout,
        )
    return left, right, (left, right) if cache_dense else None


def _declared_layout(algebra, grades, layout: GradeLayout | None) -> GradeLayout | None:
    if hasattr(algebra, "_declared_layout"):
        return algebra._declared_layout(grades, layout)
    if layout is not None:
        return layout
    if grades is not None:
        return algebra.layout(grades)
    default_grades = getattr(algebra, "_default_grades", None)
    if default_grades is None:
        return None
    return algebra.layout(default_grades)


def _action_layouts(
    algebra,
    *,
    grade: int,
    input_grades,
    output_grades,
    input_layout: GradeLayout | None,
    output_layout: GradeLayout | None,
    parameter_layout: GradeLayout | None,
) -> tuple[GradeLayout | None, GradeLayout | None, GradeLayout]:
    input_layout = _declared_layout(algebra, input_grades, input_layout)
    output_layout = _declared_layout(algebra, output_grades, output_layout) or input_layout
    parameter_layout = parameter_layout or algebra.layout((int(grade),))
    return input_layout, output_layout, parameter_layout


def _validate_action_values(
    algebra,
    values: torch.Tensor,
    *,
    layout: GradeLayout | None,
    channels: int | None,
    name: str,
) -> bool:
    if values.ndim < 3:
        raise ValueError(f"{name}: expected ndim >= 3, got shape {tuple(values.shape)}")
    if channels is not None and values.shape[-2] != channels:
        raise ValueError(f"{name}: expected {channels} channels, got {values.shape[-2]} (shape {tuple(values.shape)})")

    if _values_use_active_lanes(algebra, values, layout):
        return True
    if values.shape[-1] == algebra.dim:
        return False

    expected = [str(algebra.dim)]
    if layout is not None:
        expected.insert(0, f"{layout.dim} for grades {layout.grades}")
    raise ValueError(f"{name}: last dim must be {' or '.join(expected)}, got {values.shape[-1]}")


def _values_use_active_lanes(algebra, values: torch.Tensor, layout: GradeLayout | None) -> bool:
    if layout is None or values.shape[-1] != layout.dim:
        return False
    return layout.dim != algebra.dim or not is_dense_kernel_host(algebra)


def _require_dense_action(algebra, name: str) -> None:
    require_dense_kernel_host(algebra, f"{name} dense execution")


def _project_dense_action_output(
    algebra,
    output: torch.Tensor,
    *,
    output_layout: GradeLayout | None,
    active_output: bool,
) -> torch.Tensor:
    if output_layout is None:
        return output
    compact = output_layout.compact(output)
    if active_output:
        return compact
    return materialize_full(algebra, compact, layout=output_layout)


def _cache_matches(cache: tuple[torch.Tensor, ...] | None, reference: torch.Tensor) -> bool:
    if cache is None:
        return False
    return all(tensor.device == reference.device and tensor.dtype == reference.dtype for tensor in cache)
