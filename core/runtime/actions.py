"""Runtime actions used by compact-capable primitive layers."""

from __future__ import annotations

import torch

from core.foundation.layout import GradeLayout
from core.planning.action import bivector_vector_generator, metric_self_signs, reflection_vector_matrix
from core.runtime.accessors import materialize_dense


def compact_versor_action(
    algebra,
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    grade: int,
    input_layout: GradeLayout,
    output_layout: GradeLayout,
    parameter_layout: GradeLayout,
    compact_output: bool,
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
        input_compact=True,
        compact_output=compact_output,
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
    compact_output: bool,
) -> torch.Tensor:
    """Apply a weighted compact superposition of versor actions."""
    matrices = versor_vector_matrix(
        algebra,
        weights.to(device=values.device, dtype=values.dtype),
        grade=grade,
        parameter_layout=parameter_layout,
    )
    outputs = []
    for index in range(matrices.shape[0]):
        matrix = matrices[index].unsqueeze(0).expand(values.shape[-2], -1, -1)
        outputs.append(
            algebra.planned_linear_action(
                values,
                matrix,
                input_layout=input_layout,
                output_layout=output_layout,
                input_compact=True,
                compact_output=True,
            )
        )

    stacked = torch.stack(outputs, dim=-2)
    result = torch.einsum("ck,...ckd->...cd", mix.to(device=values.device, dtype=values.dtype), stacked)
    if compact_output:
        return result
    return materialize_dense(algebra, result, layout=output_layout)


def versor_vector_matrix(algebra, weights: torch.Tensor, *, grade: int, parameter_layout: GradeLayout) -> torch.Tensor:
    """Return the vector-space matrix represented by compact versor weights."""
    grade = int(grade)
    if grade == 2:
        return torch.matrix_exp(bivector_vector_generator(weights, bivector_layout=parameter_layout))
    if grade == 1:
        signs = parameter_layout_signs(parameter_layout, device=weights.device, dtype=weights.dtype)
        norm_sq = (weights * weights * signs).sum(dim=-1, keepdim=True)
        scale = norm_sq.abs().clamp_min(1e-12).sqrt()
        normals = weights / scale
        return reflection_vector_matrix(normals, vector_layout=parameter_layout, eps=algebra.eps_sq)
    raise ValueError("compact versor execution currently supports grade=1 and grade=2")


def parameter_layout_signs(layout: GradeLayout, *, device=None, dtype=None) -> torch.Tensor:
    """Return basis self-product signs for compact parameter weights."""
    return metric_self_signs(layout, device=device, dtype=dtype)
