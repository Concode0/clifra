# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Pure activation formulas for multivector tensors."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from clifra.core.foundation.numerics import eps_like


def _channel_parameter(parameter: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Broadcast a per-channel parameter over leading and lane dimensions."""
    if parameter.ndim != 1 or values.ndim < 2:
        return parameter
    return parameter.view((1,) * (values.ndim - 2) + (-1, 1))


def geometric_gelu(values: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Scale multivectors by ``GELU(norm + bias) / norm`` while preserving direction."""
    norm = values.norm(dim=-1, keepdim=True)
    shifted_norm = norm if bias is None else norm + _channel_parameter(bias, values)
    eps = eps_like(norm, min_value=torch.finfo(norm.dtype).tiny)
    return values * (F.gelu(shifted_norm) / norm.clamp_min(eps))


def geometric_square(algebra, values: torch.Tensor, gate: torch.Tensor | None = None, *, layout=None) -> torch.Tensor:
    """Return ``values + gate * (values * values)`` using the algebra's geometric product."""
    if layout is None:
        product = algebra.geometric_product(values, values)
    else:
        product = algebra.geometric_product(
            values,
            values,
            left_layout=layout,
            right_layout=layout,
            output_layout=layout,
        )
    if gate is None:
        return values + product
    return values + _channel_parameter(gate, values) * product


def grade_swish(
    values: torch.Tensor,
    *,
    grade_index: torch.Tensor,
    grade_weights: torch.Tensor,
    grade_biases: torch.Tensor,
    n_grades: int | None = None,
) -> torch.Tensor:
    """Apply per-grade sigmoid gates computed from per-grade coefficient norms."""
    if n_grades is None:
        n_grades = int(grade_weights.shape[0])

    lane_dim = values.shape[-1]
    batch_shape = values.shape[:-1]
    grade_idx = grade_index.to(device=values.device).expand(*batch_shape, lane_dim)

    norm_sq = values.new_zeros(*batch_shape, n_grades)
    norm_sq.scatter_add_(-1, grade_idx, values * values)
    eps = eps_like(norm_sq, min_value=torch.finfo(norm_sq.dtype).tiny)
    norms = torch.sqrt(norm_sq.clamp_min(eps))

    gates = torch.sigmoid(grade_weights * norms + grade_biases)
    per_component_gate = gates.gather(-1, grade_idx)
    return values * per_component_gate
