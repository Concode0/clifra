# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Bivector exponential executors for static exp plans."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from clifra.core.execution.product import GradeProductExecutor
from clifra.core.planning.exp import BivectorExpPlan


@torch.library.custom_op("clifra::filtered_symmetric_eigh", mutates_args=())
def _filtered_symmetric_eigh_op(matrix: Tensor, tolerances: Tensor) -> tuple[Tensor, Tensor]:
    return torch.linalg.eigh(0.5 * (matrix + matrix.transpose(-1, -2)))


@_filtered_symmetric_eigh_op.register_fake
def _filtered_symmetric_eigh_fake(matrix: Tensor, tolerances: Tensor) -> tuple[Tensor, Tensor]:
    return matrix.new_empty(*matrix.shape[:-1]), matrix.new_empty(*matrix.shape)


def _filtered_symmetric_eigh_setup_context(ctx, inputs, output) -> None:
    ctx.save_for_backward(output[0], output[1], inputs[1])


def _filtered_symmetric_eigh_backward(ctx, grad_eigenvalues: Tensor, grad_eigenvectors: Tensor):
    eigenvalues, eigenvectors, tolerances = ctx.saved_tensors
    if grad_eigenvalues is None:
        grad_eigenvalues = torch.zeros_like(eigenvalues)
    if grad_eigenvectors is None:
        grad_eigenvectors = torch.zeros_like(eigenvectors)

    inner = eigenvectors.transpose(-1, -2) @ grad_eigenvectors
    cauchy = _filtered_eigenvalue_cauchy_inverse(eigenvalues, tolerances)
    tangent = cauchy * inner
    diagonal = torch.diag_embed(grad_eigenvalues)
    spectral_grad = diagonal + 0.5 * (tangent + tangent.transpose(-1, -2))
    grad_matrix = eigenvectors @ spectral_grad @ eigenvectors.transpose(-1, -2)
    return 0.5 * (grad_matrix + grad_matrix.transpose(-1, -2)), None


_filtered_symmetric_eigh_op.register_autograd(
    _filtered_symmetric_eigh_backward,
    setup_context=_filtered_symmetric_eigh_setup_context,
)


def _filtered_eigenvalue_cauchy_inverse(eigenvalues: Tensor, tolerances: Tensor) -> Tensor:
    delta = eigenvalues.unsqueeze(-2) - eigenvalues.unsqueeze(-1)
    scale = eigenvalues.abs().amax(dim=-1, keepdim=True).unsqueeze(-1)
    finfo = torch.finfo(eigenvalues.dtype)
    relative = torch.maximum(
        tolerances[2].abs().to(dtype=eigenvalues.dtype, device=eigenvalues.device),
        eigenvalues.new_tensor(finfo.eps**0.5),
    )
    absolute = torch.maximum(
        tolerances[0].abs().to(dtype=eigenvalues.dtype, device=eigenvalues.device) ** 2,
        eigenvalues.new_tensor(finfo.eps * 32.0),
    )
    threshold = absolute + scale * relative
    inverse = torch.where(delta.abs() > threshold, delta.reciprocal(), torch.zeros_like(delta))
    eye = torch.eye(eigenvalues.shape[-1], dtype=eigenvalues.dtype, device=eigenvalues.device)
    return inverse * (1.0 - eye)


@torch.library.custom_op("clifra::spectral_local_bivector_exp", mutates_args=())
def _spectral_local_bivector_exp_op(
    values: Tensor,
    bivector_to_generator: Tensor,
    local_scalar_mask: Tensor,
    local_plane_masks: Tensor,
    local_product_table: Tensor,
    local_sparse_left_positions: Tensor,
    local_sparse_right_positions: Tensor,
    local_sparse_output_positions: Tensor,
    local_sparse_coefficients: Tensor,
    plane_bivector_map: Tensor,
    plane_eye: Tensor,
    plane_left_positions: Tensor,
    plane_right_positions: Tensor,
    plane_output_positions: Tensor,
    plane_coefficients: Tensor,
    plane_to_local: Tensor,
    nilpotent_to_local: Tensor,
    ideal_basis: Tensor,
    lift_grades: Tensor,
    lift_local_positions: Tensor,
    lift_local_axes: Tensor,
    lift_local_mask: Tensor,
    lift_target_axes: Tensor,
    lift_target_map: Tensor,
    tolerances: Tensor,
    left_active_positions: Tensor,
    right_active_positions: Tensor,
    product_output_positions: Tensor,
    product_coefficients: Tensor,
    operator_eye: Tensor,
    operator_to_output: Tensor,
    bivector_to_mixed_generator: Tensor,
) -> Tensor:
    return _spectral_local_forward_impl(
        values,
        bivector_to_generator,
        local_scalar_mask,
        local_plane_masks,
        local_product_table,
        local_sparse_left_positions,
        local_sparse_right_positions,
        local_sparse_output_positions,
        local_sparse_coefficients,
        plane_bivector_map,
        plane_eye,
        plane_left_positions,
        plane_right_positions,
        plane_output_positions,
        plane_coefficients,
        plane_to_local,
        nilpotent_to_local,
        ideal_basis,
        lift_grades,
        lift_local_positions,
        lift_local_axes,
        lift_local_mask,
        lift_target_axes,
        lift_target_map,
        tolerances,
        bivector_to_mixed_generator,
    )


@_spectral_local_bivector_exp_op.register_fake
def _spectral_local_bivector_exp_fake(
    values: Tensor,
    bivector_to_generator: Tensor,
    local_scalar_mask: Tensor,
    local_plane_masks: Tensor,
    local_product_table: Tensor,
    local_sparse_left_positions: Tensor,
    local_sparse_right_positions: Tensor,
    local_sparse_output_positions: Tensor,
    local_sparse_coefficients: Tensor,
    plane_bivector_map: Tensor,
    plane_eye: Tensor,
    plane_left_positions: Tensor,
    plane_right_positions: Tensor,
    plane_output_positions: Tensor,
    plane_coefficients: Tensor,
    plane_to_local: Tensor,
    nilpotent_to_local: Tensor,
    ideal_basis: Tensor,
    lift_grades: Tensor,
    lift_local_positions: Tensor,
    lift_local_axes: Tensor,
    lift_local_mask: Tensor,
    lift_target_axes: Tensor,
    lift_target_map: Tensor,
    tolerances: Tensor,
    left_active_positions: Tensor,
    right_active_positions: Tensor,
    product_output_positions: Tensor,
    product_coefficients: Tensor,
    operator_eye: Tensor,
    operator_to_output: Tensor,
    bivector_to_mixed_generator: Tensor,
) -> Tensor:
    return values.new_empty(*values.shape[:-1], operator_to_output.shape[-1])


def _spectral_local_setup_context(ctx, inputs, output) -> None:
    ctx.save_for_backward(*inputs)


def _spectral_local_backward(ctx, grad_output: Tensor):
    saved = ctx.saved_tensors
    values = saved[0]
    repeated_root_mask = _spectral_repeated_root_mask(values, saved[1], saved[24])
    use_exact_fallback = bool(repeated_root_mask.any().item())
    if use_exact_fallback:
        exact_grad_values = _matrix_exp_frechet_vjp_values_impl(
            values,
            grad_output,
            saved[25],
            saved[26],
            saved[27],
            saved[28],
            saved[29],
            saved[30],
        )
        if bool(repeated_root_mask.all().item()):
            return (exact_grad_values, *(None for _ in range(len(saved) - 1)))

    with torch.enable_grad():
        reference_values = values.detach().requires_grad_(True)
        reference = _spectral_local_forward_impl(
            reference_values,
            saved[1],
            saved[2],
            saved[3],
            saved[4],
            saved[5],
            saved[6],
            saved[7],
            saved[8],
            saved[9],
            saved[10],
            saved[11],
            saved[12],
            saved[13],
            saved[14],
            saved[15],
            saved[16],
            saved[17],
            saved[18],
            saved[19],
            saved[20],
            saved[21],
            saved[22],
            saved[23],
            saved[24],
            saved[31],
        )
        (spectral_grad_values,) = torch.autograd.grad(reference, reference_values, grad_output)
    if use_exact_fallback:
        spectral_grad_values = torch.where(repeated_root_mask, exact_grad_values, spectral_grad_values)
    grad_values = spectral_grad_values
    return (grad_values, *(None for _ in range(len(ctx.saved_tensors) - 1)))


_spectral_local_bivector_exp_op.register_autograd(
    _spectral_local_backward,
    setup_context=_spectral_local_setup_context,
)


def _sparse_product_compact_impl(
    left: Tensor,
    right: Tensor,
    left_active_positions: Tensor,
    right_active_positions: Tensor,
    output_positions: Tensor,
    coefficients: Tensor,
    output_dim: int,
) -> Tensor:
    left_terms = torch.index_select(left, -1, left_active_positions)
    right_terms = torch.index_select(right, -1, right_active_positions)
    left_terms, right_terms = torch.broadcast_tensors(left_terms, right_terms)
    terms = left_terms * right_terms * coefficients
    output = terms.new_zeros(*terms.shape[:-1], output_dim)
    return output.index_add(-1, output_positions, terms)


def _sparse_product_left_vjp_impl(
    grad_output: Tensor,
    right: Tensor,
    left_active_positions: Tensor,
    right_active_positions: Tensor,
    output_positions: Tensor,
    coefficients: Tensor,
    input_dim: int,
) -> Tensor:
    grad_terms = torch.index_select(grad_output, -1, output_positions)
    right_terms = torch.index_select(right, -1, right_active_positions)
    contributions = grad_terms * right_terms * coefficients
    contributions = contributions.sum(dim=-2)
    grad_left = contributions.new_zeros(*contributions.shape[:-1], input_dim)
    return grad_left.index_add(-1, left_active_positions, contributions)


def _matrix_exp_adjoint_impl(operator: Tensor, grad_exp_operator: Tensor) -> Tensor:
    operator_dim = operator.shape[-1]
    zeros = torch.zeros_like(operator)
    block = torch.cat(
        [
            torch.cat([operator.transpose(-1, -2), grad_exp_operator], dim=-1),
            torch.cat([zeros, operator.transpose(-1, -2)], dim=-1),
        ],
        dim=-2,
    )
    return torch.matrix_exp(block)[..., :operator_dim, operator_dim:]


def _matrix_exp_frechet_vjp_values_impl(
    values: Tensor,
    grad_output: Tensor,
    left_active_positions: Tensor,
    right_active_positions: Tensor,
    product_output_positions: Tensor,
    product_coefficients: Tensor,
    operator_eye: Tensor,
    operator_to_output: Tensor,
) -> Tensor:
    operator_dim = operator_to_output.shape[0]
    columns = _sparse_product_compact_impl(
        values.unsqueeze(-2),
        operator_eye,
        left_active_positions,
        right_active_positions,
        product_output_positions,
        product_coefficients,
        operator_dim,
    )
    operator = columns.transpose(-1, -2)
    grad_even = grad_output @ operator_to_output.transpose(-1, -2)
    grad_exp_operator = values.new_zeros(*values.shape[:-1], operator_dim, operator_dim)
    grad_exp_operator[..., :, 0] = grad_even
    grad_operator = _matrix_exp_adjoint_impl(operator, grad_exp_operator)
    grad_columns = grad_operator.transpose(-1, -2)
    return _sparse_product_left_vjp_impl(
        grad_columns,
        operator_eye,
        left_active_positions,
        right_active_positions,
        product_output_positions,
        product_coefficients,
        values.shape[-1],
    )


def _spectral_repeated_root_mask(values: Tensor, bivector_to_generator: Tensor, tolerances: Tensor) -> Tensor:
    input_dim = values.shape[-1]
    nondegenerate_dim = bivector_to_generator.shape[-1]
    generator = torch.matmul(values, bivector_to_generator.reshape(input_dim, -1)).reshape(
        *values.shape[:-1],
        nondegenerate_dim,
        nondegenerate_dim,
    )
    squared = -(generator @ generator)
    squared = 0.5 * (squared + squared.transpose(-1, -2))
    eigenvalues = torch.linalg.eigvalsh(squared).clamp_min(0.0)
    delta = eigenvalues.unsqueeze(-2) - eigenvalues.unsqueeze(-1)
    scale = eigenvalues.abs().amax(dim=-1, keepdim=True).unsqueeze(-1)
    finfo = torch.finfo(values.dtype)
    relative = torch.maximum(
        tolerances[2].abs().to(dtype=values.dtype, device=values.device),
        values.new_tensor(finfo.eps**0.5),
    )
    absolute = torch.maximum(
        tolerances[0].abs().to(dtype=values.dtype, device=values.device) ** 2,
        values.new_tensor(finfo.eps * 32.0),
    )
    threshold = absolute + scale * relative
    near = delta.abs() <= threshold
    multiplicity = near.sum(dim=-1)
    return (multiplicity > 2).any(dim=-1, keepdim=True)


def _spectral_local_forward_impl(
    values: Tensor,
    bivector_to_generator: Tensor,
    local_scalar_mask: Tensor,
    local_plane_masks: Tensor,
    local_product_table: Tensor,
    local_sparse_left_positions: Tensor,
    local_sparse_right_positions: Tensor,
    local_sparse_output_positions: Tensor,
    local_sparse_coefficients: Tensor,
    plane_bivector_map: Tensor,
    plane_eye: Tensor,
    plane_left_positions: Tensor,
    plane_right_positions: Tensor,
    plane_output_positions: Tensor,
    plane_coefficients: Tensor,
    plane_to_local: Tensor,
    nilpotent_to_local: Tensor,
    ideal_basis: Tensor,
    lift_grades: Tensor,
    lift_local_positions: Tensor,
    lift_local_axes: Tensor,
    lift_local_mask: Tensor,
    lift_target_axes: Tensor,
    lift_target_map: Tensor,
    tolerances: Tensor,
    bivector_to_mixed_generator: Tensor,
) -> Tensor:
    if ideal_basis.shape[0] > 0:
        return _spectral_local_degenerate_forward_impl(
            values,
            bivector_to_generator,
            bivector_to_mixed_generator,
            local_scalar_mask,
            local_sparse_left_positions,
            local_sparse_right_positions,
            local_sparse_output_positions,
            local_sparse_coefficients,
            plane_bivector_map,
            plane_eye,
            plane_left_positions,
            plane_right_positions,
            plane_output_positions,
            plane_coefficients,
            plane_to_local,
            nilpotent_to_local,
            ideal_basis,
            lift_grades,
            lift_local_positions,
            lift_local_axes,
            lift_local_mask,
            lift_target_axes,
            lift_target_map,
            tolerances,
        )

    input_dim = values.shape[-1]
    nondegenerate_dim = bivector_to_generator.shape[-1]
    generator = torch.matmul(values, bivector_to_generator.reshape(input_dim, -1)).reshape(
        *values.shape[:-1],
        nondegenerate_dim,
        nondegenerate_dim,
    )
    squared = -(generator @ generator)
    eigenvalues, eigenvectors = _filtered_symmetric_eigh_op(squared, tolerances)
    eigenvalues = torch.flip(eigenvalues, dims=(-1,)).clamp_min(0.0)
    eigenvectors = torch.flip(eigenvectors, dims=(-1,))
    theta_all = torch.sqrt(eigenvalues)
    plane_theta = theta_all[..., 0::2]
    total = plane_theta.sum(dim=-1, keepdim=True)
    tolerance = torch.maximum(
        tolerances[0].expand_as(total),
        total * tolerances[1],
    )
    dominant_tolerance = torch.maximum(
        tolerances[0].expand_as(total),
        plane_theta[..., :1] * tolerances[2],
    )
    tail = torch.flip(torch.cumsum(torch.flip(plane_theta, dims=(-1,)), dim=-1), dims=(-1,))

    local_plane_count = local_plane_masks.shape[0]
    eps = torch.finfo(values.dtype).eps
    projector = torch.eye(nondegenerate_dim, dtype=values.dtype, device=values.device)
    projector = projector * values.new_ones(*values.shape[:-1], 1, 1)
    basis_vectors: list[Tensor] = []
    selected_theta: list[Tensor] = []
    active_masks: list[Tensor] = []
    for plane in range(local_plane_count):
        candidates = projector @ eigenvectors
        residual_norm_sq = (candidates * candidates).sum(dim=-2)
        selected = (residual_norm_sq * theta_all).argmax(dim=-1)
        selected_matrix = selected.unsqueeze(-1).unsqueeze(-1).expand(*selected.shape, nondegenerate_dim, 1)
        basis_x_raw = torch.take_along_dim(candidates, selected_matrix, dim=-1).squeeze(-1)
        theta = torch.take_along_dim(theta_all, selected.unsqueeze(-1), dim=-1)

        x_norm = basis_x_raw.norm(dim=-1, keepdim=True)
        basis_x = basis_x_raw / torch.where(x_norm > eps, x_norm, torch.ones_like(x_norm))
        safe_theta = torch.where(theta > eps, theta, torch.ones_like(theta))
        basis_y_raw = (generator @ basis_x.unsqueeze(-1)).squeeze(-1) / safe_theta
        basis_y_raw = basis_y_raw - (basis_y_raw * basis_x).sum(dim=-1, keepdim=True) * basis_x
        y_norm = basis_y_raw.norm(dim=-1, keepdim=True)
        basis_y = basis_y_raw / torch.where(y_norm > eps, y_norm, torch.ones_like(y_norm))
        active = (
            (tail[..., plane : plane + 1] > tolerance)
            & (theta > dominant_tolerance)
            & (x_norm > eps)
            & (y_norm > eps)
        )
        active = active.to(values.dtype)

        basis_vectors.extend([basis_x, basis_y])
        selected_theta.append(theta)
        active_masks.append(active)
        plane_projector = basis_x.unsqueeze(-1) * basis_x.unsqueeze(-2) + basis_y.unsqueeze(-1) * basis_y.unsqueeze(-2)
        projector = projector - active.unsqueeze(-1) * plane_projector
        projector = 0.5 * (projector + projector.transpose(-1, -2))

    theta = torch.cat(selected_theta, dim=-1)
    active = torch.cat(active_masks, dim=-1)
    scalar = torch.where(active > 0.0, torch.cos(theta), torch.ones_like(theta))
    plane = torch.where(active > 0.0, torch.sin(theta), torch.zeros_like(theta))
    local_rotors = scalar.unsqueeze(-1) * local_scalar_mask + plane.unsqueeze(-1) * local_plane_masks
    local_rotor = _spectral_local_reduce_impl(local_rotors, local_product_table)
    v_local = torch.stack(basis_vectors, dim=-2)
    return _spectral_local_lift_impl(
        local_rotor,
        v_local,
        lift_grades,
        lift_local_positions,
        lift_local_axes,
        lift_local_mask,
        lift_target_axes,
        lift_target_map,
    )


def _spectral_local_degenerate_forward_impl(
    values: Tensor,
    bivector_to_generator: Tensor,
    bivector_to_mixed_generator: Tensor,
    local_scalar_mask: Tensor,
    local_sparse_left_positions: Tensor,
    local_sparse_right_positions: Tensor,
    local_sparse_output_positions: Tensor,
    local_sparse_coefficients: Tensor,
    plane_bivector_map: Tensor,
    plane_eye: Tensor,
    plane_left_positions: Tensor,
    plane_right_positions: Tensor,
    plane_output_positions: Tensor,
    plane_coefficients: Tensor,
    plane_to_local: Tensor,
    nilpotent_to_local: Tensor,
    ideal_basis: Tensor,
    lift_grades: Tensor,
    lift_local_positions: Tensor,
    lift_local_axes: Tensor,
    lift_local_mask: Tensor,
    lift_target_axes: Tensor,
    lift_target_map: Tensor,
    tolerances: Tensor,
) -> Tensor:
    input_dim = values.shape[-1]
    nondegenerate_dim = bivector_to_generator.shape[-1]
    ideal_dim = ideal_basis.shape[0]
    generator = torch.matmul(values, bivector_to_generator.reshape(input_dim, -1)).reshape(
        *values.shape[:-1],
        nondegenerate_dim,
        nondegenerate_dim,
    )
    mixed = torch.matmul(values, bivector_to_mixed_generator.reshape(input_dim, -1)).reshape(
        *values.shape[:-1],
        ideal_dim,
        nondegenerate_dim,
    )

    squared = -(generator @ generator)
    eigenvalues, eigenvectors = _filtered_symmetric_eigh_op(squared, tolerances)
    eigenvalues = torch.flip(eigenvalues, dims=(-1,)).clamp_min(0.0)
    eigenvectors = torch.flip(eigenvectors, dims=(-1,))
    theta_all = torch.sqrt(eigenvalues)
    plane_theta = theta_all[..., 0::2]
    total = plane_theta.sum(dim=-1, keepdim=True)
    tolerance = torch.maximum(tolerances[0].expand_as(total), total * tolerances[1])
    dominant_tolerance = torch.maximum(tolerances[0].expand_as(total), plane_theta[..., :1] * tolerances[2])
    tail = torch.flip(torch.cumsum(torch.flip(plane_theta, dims=(-1,)), dim=-1), dims=(-1,))

    batch_shape = values.shape[:-1]
    eps = torch.finfo(values.dtype).eps
    projector = torch.eye(nondegenerate_dim, dtype=values.dtype, device=values.device)
    projector = projector * values.new_ones(*batch_shape, 1, 1)
    local_rotor = local_scalar_mask * values.new_ones(*batch_shape, 1)
    basis_vectors: list[Tensor] = []

    for plane in range(plane_to_local.shape[0]):
        candidates = projector @ eigenvectors
        residual_norm_sq = (candidates * candidates).sum(dim=-2)
        rotation_scores = residual_norm_sq * theta_all
        fallback_scores = residual_norm_sq
        use_fallback = rotation_scores.amax(dim=-1, keepdim=True) <= eps
        selected = torch.where(use_fallback, fallback_scores, rotation_scores).argmax(dim=-1)
        selected_matrix = selected.unsqueeze(-1).unsqueeze(-1).expand(*selected.shape, nondegenerate_dim, 1)
        basis_x_raw = torch.take_along_dim(candidates, selected_matrix, dim=-1).squeeze(-1)
        theta = torch.take_along_dim(theta_all, selected.unsqueeze(-1), dim=-1)

        x_norm = basis_x_raw.norm(dim=-1, keepdim=True)
        basis_x = basis_x_raw / torch.where(x_norm > eps, x_norm, torch.ones_like(x_norm))
        safe_theta = torch.where(theta > eps, theta, torch.ones_like(theta))
        basis_y_from_generator_raw = (generator @ basis_x.unsqueeze(-1)).squeeze(-1) / safe_theta
        basis_y_from_generator_raw = (
            basis_y_from_generator_raw
            - (basis_y_from_generator_raw * basis_x).sum(dim=-1, keepdim=True) * basis_x
        )
        y_generator_norm = basis_y_from_generator_raw.norm(dim=-1, keepdim=True)
        basis_y_from_generator = basis_y_from_generator_raw / torch.where(
            y_generator_norm > eps,
            y_generator_norm,
            torch.ones_like(y_generator_norm),
        )

        projector_without_x = projector - basis_x.unsqueeze(-1) * basis_x.unsqueeze(-2)
        fallback_candidates = projector_without_x @ eigenvectors
        fallback_residual_norm_sq = (fallback_candidates * fallback_candidates).sum(dim=-2)
        selected_y = fallback_residual_norm_sq.argmax(dim=-1)
        selected_y_matrix = selected_y.unsqueeze(-1).unsqueeze(-1).expand(*selected_y.shape, nondegenerate_dim, 1)
        basis_y_fallback_raw = torch.take_along_dim(fallback_candidates, selected_y_matrix, dim=-1).squeeze(-1)
        y_fallback_norm = basis_y_fallback_raw.norm(dim=-1, keepdim=True)
        basis_y_fallback = basis_y_fallback_raw / torch.where(
            y_fallback_norm > eps,
            y_fallback_norm,
            torch.ones_like(y_fallback_norm),
        )
        use_generator_y = (theta > eps) & (y_generator_norm > eps)
        basis_y = torch.where(use_generator_y, basis_y_from_generator, basis_y_fallback)
        y_norm = torch.where(use_generator_y, y_generator_norm, y_fallback_norm)

        rotation_active = (
            (tail[..., plane : plane + 1] > tolerance)
            & (theta > dominant_tolerance)
            & (x_norm > eps)
            & (y_norm > eps)
        )
        theta = torch.where(rotation_active, theta, torch.zeros_like(theta))
        cx = (mixed @ basis_x.unsqueeze(-1)).squeeze(-1)
        cy = (mixed @ basis_y.unsqueeze(-1)).squeeze(-1)
        plane_features = torch.cat([theta, cx, cy], dim=-1)
        plane_bivector = plane_features @ plane_bivector_map
        plane_factor = _spectral_local_matrix_exp_factor_impl(
            plane_bivector,
            plane_eye,
            plane_left_positions,
            plane_right_positions,
            plane_output_positions,
            plane_coefficients,
        )
        local_factor = plane_factor @ plane_to_local[plane]
        local_rotor = _spectral_local_sparse_product_impl(
            local_rotor,
            local_factor,
            local_sparse_left_positions,
            local_sparse_right_positions,
            local_sparse_output_positions,
            local_sparse_coefficients,
        )

        basis_vectors.extend([basis_x, basis_y])
        pair_valid = (x_norm > eps) & (y_norm > eps)
        plane_projector = basis_x.unsqueeze(-1) * basis_x.unsqueeze(-2) + basis_y.unsqueeze(-1) * basis_y.unsqueeze(-2)
        projector = projector - pair_valid.unsqueeze(-1) * plane_projector
        projector = 0.5 * (projector + projector.transpose(-1, -2))

    nilpotent = values @ nilpotent_to_local
    nilpotent_factor = _spectral_local_nilpotent_exp_impl(
        nilpotent,
        local_scalar_mask,
        local_sparse_left_positions,
        local_sparse_right_positions,
        local_sparse_output_positions,
        local_sparse_coefficients,
        ideal_dim,
    )
    local_rotor = _spectral_local_sparse_product_impl(
        local_rotor,
        nilpotent_factor,
        local_sparse_left_positions,
        local_sparse_right_positions,
        local_sparse_output_positions,
        local_sparse_coefficients,
    )

    nondegenerate_basis = torch.stack(basis_vectors, dim=-2)
    nondegenerate_basis = torch.cat(
        [
            nondegenerate_basis,
            values.new_zeros(*batch_shape, nondegenerate_basis.shape[-2], ideal_dim),
        ],
        dim=-1,
    )
    ideal_rows = ideal_basis * values.new_ones(*batch_shape, 1, 1)
    v_local = torch.cat([nondegenerate_basis, ideal_rows], dim=-2)
    return _spectral_local_lift_impl(
        local_rotor,
        v_local,
        lift_grades,
        lift_local_positions,
        lift_local_axes,
        lift_local_mask,
        lift_target_axes,
        lift_target_map,
    )


def _spectral_local_matrix_exp_factor_impl(
    values: Tensor,
    operator_eye: Tensor,
    left_active_positions: Tensor,
    right_active_positions: Tensor,
    product_output_positions: Tensor,
    product_coefficients: Tensor,
) -> Tensor:
    columns = _sparse_product_compact_impl(
        values.unsqueeze(-2),
        operator_eye,
        left_active_positions,
        right_active_positions,
        product_output_positions,
        product_coefficients,
        values.shape[-1],
    )
    operator = columns.transpose(-1, -2)
    exp_operator = torch.matrix_exp(operator)
    return exp_operator[..., :, 0]


def _spectral_local_sparse_product_impl(
    left: Tensor,
    right: Tensor,
    left_active_positions: Tensor,
    right_active_positions: Tensor,
    output_positions: Tensor,
    coefficients: Tensor,
) -> Tensor:
    return _sparse_product_compact_impl(
        left,
        right,
        left_active_positions,
        right_active_positions,
        output_positions,
        coefficients,
        left.shape[-1],
    )


def _spectral_local_nilpotent_exp_impl(
    nilpotent: Tensor,
    scalar_mask: Tensor,
    left_active_positions: Tensor,
    right_active_positions: Tensor,
    output_positions: Tensor,
    coefficients: Tensor,
    ideal_dim: int,
) -> Tensor:
    result = scalar_mask * nilpotent.new_ones(*nilpotent.shape[:-1], 1)
    if ideal_dim < 2:
        return result
    term = nilpotent
    result = result + term
    factorial = 1.0
    for order in range(2, ideal_dim // 2 + 1):
        term = _spectral_local_sparse_product_impl(
            term,
            nilpotent,
            left_active_positions,
            right_active_positions,
            output_positions,
            coefficients,
        )
        factorial *= float(order)
        result = result + term / factorial
    return result


def _spectral_local_product_impl(left: Tensor, right: Tensor, product_table: Tensor) -> Tensor:
    return torch.einsum("...i,...j,ijk->...k", left, right, product_table)


def _spectral_local_reduce_impl(local_rotors: Tensor, product_table: Tensor) -> Tensor:
    plane_count = local_rotors.shape[-2]
    if plane_count == 1:
        return local_rotors[..., 0, :]
    if plane_count == 2:
        return _spectral_local_product_impl(local_rotors[..., 0, :], local_rotors[..., 1, :], product_table)
    if plane_count == 3:
        first = _spectral_local_product_impl(local_rotors[..., 0, :], local_rotors[..., 1, :], product_table)
        return _spectral_local_product_impl(first, local_rotors[..., 2, :], product_table)
    first = _spectral_local_product_impl(local_rotors[..., 0, :], local_rotors[..., 1, :], product_table)
    second = _spectral_local_product_impl(local_rotors[..., 2, :], local_rotors[..., 3, :], product_table)
    return _spectral_local_product_impl(first, second, product_table)


def _spectral_local_lift_impl(
    local_rotor: Tensor,
    v_local: Tensor,
    lift_grades: Tensor,
    lift_local_positions: Tensor,
    lift_local_axes: Tensor,
    lift_local_mask: Tensor,
    lift_target_axes: Tensor,
    lift_target_map: Tensor,
) -> Tensor:
    output = local_rotor.new_zeros(*local_rotor.shape[:-1], lift_target_map.shape[-1])
    batch_shape = local_rotor.shape[:-1]
    for grade_slot in range(lift_grades.shape[0]):
        grade = int(lift_grades[grade_slot].item())
        local_positions = lift_local_positions[grade_slot]
        local_coefficients = torch.index_select(local_rotor, -1, local_positions)
        local_coefficients = local_coefficients * lift_local_mask[grade_slot]
        target_map = lift_target_map[grade_slot]
        if grade == 0:
            output = output + local_coefficients[..., :1] @ target_map[:1]
            continue

        local_axes = lift_local_axes[grade_slot, :, :grade]
        target_axes = lift_target_axes[grade_slot, :, :grade]
        local_count = local_axes.shape[0]
        target_count = target_axes.shape[0]
        rows = torch.index_select(v_local, -2, local_axes.reshape(-1)).reshape(
            *batch_shape,
            local_count,
            grade,
            v_local.shape[-1],
        )
        expanded = rows.unsqueeze(-3).expand(*batch_shape, local_count, target_count, grade, v_local.shape[-1])
        column_index = target_axes.reshape(
            *((1,) * len(batch_shape)),
            1,
            target_count,
            1,
            grade,
        ).expand(*batch_shape, local_count, target_count, grade, grade)
        submatrices = torch.gather(expanded, -1, column_index)
        determinants = _expanded_small_det_impl(submatrices)
        target_coefficients = torch.einsum("...l,...lt->...t", local_coefficients, determinants)
        output = output + target_coefficients @ target_map
    return output


def _expanded_small_det_impl(matrix: Tensor) -> Tensor:
    size = matrix.shape[-1]
    if size == 0:
        return matrix.new_ones(*matrix.shape[:-2])
    if size == 1:
        return matrix[..., 0, 0]
    if size == 2:
        return _expanded_det2_impl(matrix)
    if size == 3:
        return _expanded_det3_impl(matrix)
    if size == 4:
        return _expanded_det4_impl(matrix)

    result = matrix.new_zeros(*matrix.shape[:-2])
    for column in range(size):
        if column == 0:
            minor = matrix[..., 1:, 1:]
        elif column == size - 1:
            minor = matrix[..., 1:, :-1]
        else:
            minor = torch.cat([matrix[..., 1:, :column], matrix[..., 1:, column + 1 :]], dim=-1)
        sign = -1.0 if column % 2 else 1.0
        result = result + sign * matrix[..., 0, column] * _expanded_small_det_impl(minor)
    return result


def _expanded_det2_impl(matrix: Tensor) -> Tensor:
    return matrix[..., 0, 0] * matrix[..., 1, 1] - matrix[..., 0, 1] * matrix[..., 1, 0]


def _expanded_det3_impl(matrix: Tensor) -> Tensor:
    minor0 = matrix[..., 1, 1] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 1]
    minor1 = matrix[..., 1, 0] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 0]
    minor2 = matrix[..., 1, 0] * matrix[..., 2, 1] - matrix[..., 1, 1] * matrix[..., 2, 0]
    return matrix[..., 0, 0] * minor0 - matrix[..., 0, 1] * minor1 + matrix[..., 0, 2] * minor2


def _expanded_det4_impl(matrix: Tensor) -> Tensor:
    minor0 = _expanded_det3_impl(matrix[..., 1:, 1:])
    minor1 = _expanded_det3_impl(torch.stack([matrix[..., 1:, 0], matrix[..., 1:, 2], matrix[..., 1:, 3]], dim=-1))
    minor2 = _expanded_det3_impl(torch.stack([matrix[..., 1:, 0], matrix[..., 1:, 1], matrix[..., 1:, 3]], dim=-1))
    minor3 = _expanded_det3_impl(matrix[..., 1:, :3])
    return (
        matrix[..., 0, 0] * minor0
        - matrix[..., 0, 1] * minor1
        + matrix[..., 0, 2] * minor2
        - matrix[..., 0, 3] * minor3
    )


class BivectorExpExecutor(nn.Module):
    """Compile-friendly ``exp(B)`` executor for grade-2 inputs.

    Dimensions up to five use closed formulas. Eligible high-dimensional
    Euclidean plans use the spectral-local path; other high-dimensional plans
    use the left-product matrix exponential over the full even subalgebra.
    """

    op = "bivector_exp"

    def __init__(
        self,
        plan: BivectorExpPlan,
        left_product: GradeProductExecutor | None,
        *,
        bivector_wedge: GradeProductExecutor | None = None,
        grade4_square: GradeProductExecutor | None = None,
        bivector_grade4_product: GradeProductExecutor | None = None,
    ):
        super().__init__()
        self.spec = plan.spec
        self.input_layout = plan.input_layout
        self.grade4_layout = plan.grade4_layout
        self.operator_layout = plan.operator_layout
        self.output_layout = plan.output_layout
        self.executor_family = plan.executor_family
        self.eps = plan.eps
        self.eps_sq = plan.eps_sq
        self.spectral_max_planes = int(plan.spectral_max_planes)
        self.spectral_tol_abs = float(plan.spectral_tol_abs)
        self.spectral_tol_rel = float(plan.spectral_tol_rel)
        self.spectral_dominant_rel = float(plan.spectral_dominant_rel)
        self.spectral_allow_degenerate = bool(plan.spectral_allow_degenerate)
        self.spectral_allow_truncated_degenerate = bool(plan.spectral_allow_truncated_degenerate)
        self.nondegenerate_dim = int(plan.nondegenerate_dim)
        self.ideal_dim = int(plan.ideal_dim)
        self.spectral_local_axis_count = int(plan.spectral_local_axis_count)
        self.left_product = left_product
        self.bivector_wedge = bivector_wedge
        self.grade4_square = grade4_square
        self.bivector_grade4_product = bivector_grade4_product
        self.register_buffer("metric_signs", plan.metric_signs, persistent=False)
        self.register_buffer("bivector_squared_signs", plan.bivector_squared_signs, persistent=False)
        self.register_buffer("nondegenerate_bivector_positions", plan.nondegenerate_bivector_positions, persistent=False)
        self.register_buffer("mixed_degenerate_bivector_positions", plan.mixed_degenerate_bivector_positions, persistent=False)
        self.register_buffer("nilpotent_bivector_positions", plan.nilpotent_bivector_positions, persistent=False)
        self.register_buffer(
            "bivector_to_nondegenerate_generator",
            plan.bivector_to_nondegenerate_generator,
            persistent=False,
        )
        self.register_buffer(
            "nondegenerate_generator_to_bivector",
            plan.nondegenerate_generator_to_bivector,
            persistent=False,
        )
        self.register_buffer("bivector_to_mixed_generator", plan.bivector_to_mixed_generator, persistent=False)
        self.register_buffer("output_scalar_mask", plan.output_scalar_mask, persistent=False)
        self.register_buffer("operator_scalar_mask", plan.operator_scalar_mask, persistent=False)
        self.register_buffer("bivector_to_output", plan.bivector_to_output, persistent=False)
        self.register_buffer("bivector_to_operator", plan.bivector_to_operator, persistent=False)
        self.register_buffer("grade4_to_output", plan.grade4_to_output, persistent=False)
        self.register_buffer("operator_to_output", plan.operator_to_output, persistent=False)
        self.register_buffer("operator_eye", plan.operator_eye, persistent=False)
        self.register_buffer("spectral_local_scalar_mask", plan.spectral_local_scalar_mask, persistent=False)
        self.register_buffer("spectral_local_plane_masks", plan.spectral_local_plane_masks, persistent=False)
        self.register_buffer("spectral_local_product_table", plan.spectral_local_product_table, persistent=False)
        self.register_buffer(
            "spectral_local_sparse_left_positions",
            plan.spectral_local_sparse_left_positions,
            persistent=False,
        )
        self.register_buffer(
            "spectral_local_sparse_right_positions",
            plan.spectral_local_sparse_right_positions,
            persistent=False,
        )
        self.register_buffer(
            "spectral_local_sparse_output_positions",
            plan.spectral_local_sparse_output_positions,
            persistent=False,
        )
        self.register_buffer("spectral_local_sparse_coefficients", plan.spectral_local_sparse_coefficients, persistent=False)
        self.register_buffer("spectral_plane_bivector_map", plan.spectral_plane_bivector_map, persistent=False)
        self.register_buffer("spectral_plane_eye", plan.spectral_plane_eye, persistent=False)
        self.register_buffer("spectral_plane_left_positions", plan.spectral_plane_left_positions, persistent=False)
        self.register_buffer("spectral_plane_right_positions", plan.spectral_plane_right_positions, persistent=False)
        self.register_buffer("spectral_plane_output_positions", plan.spectral_plane_output_positions, persistent=False)
        self.register_buffer("spectral_plane_coefficients", plan.spectral_plane_coefficients, persistent=False)
        self.register_buffer("spectral_plane_to_local", plan.spectral_plane_to_local, persistent=False)
        self.register_buffer("spectral_nilpotent_to_local", plan.spectral_nilpotent_to_local, persistent=False)
        self.register_buffer("spectral_ideal_basis", plan.spectral_ideal_basis, persistent=False)
        self.register_buffer("spectral_lift_grades", plan.spectral_lift_grades, persistent=False)
        self.register_buffer("spectral_lift_local_positions", plan.spectral_lift_local_positions, persistent=False)
        self.register_buffer("spectral_lift_local_axes", plan.spectral_lift_local_axes, persistent=False)
        self.register_buffer("spectral_lift_local_mask", plan.spectral_lift_local_mask, persistent=False)
        self.register_buffer("spectral_lift_target_axes", plan.spectral_lift_target_axes, persistent=False)
        self.register_buffer("spectral_lift_target_map", plan.spectral_lift_target_map, persistent=False)
        self.register_buffer("spectral_tolerances", plan.spectral_tolerances, persistent=False)
        self.operator_scalar_position = int(plan.operator_scalar_position)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return ``exp(values)`` in ``output_layout`` lanes."""
        if values.shape[-1] != self.input_layout.dim:
            raise ValueError(f"bivector exp input dimension must be {self.input_layout.dim}, got {values.shape[-1]}")
        if self.executor_family == "closed_simple":
            return self._closed_simple(values)
        if self.executor_family == "closed_biquadratic":
            return self._closed_biquadratic(values)
        if self.executor_family == "spectral_local":
            return self._spectral_local(values)
        return self._left_matrix_exp(values)

    def _closed_simple(self, values: torch.Tensor) -> torch.Tensor:
        alpha = (values * values * self._signs_for(values)).sum(dim=-1, keepdim=True)
        scalar_part, coeff_part = self._real_cosh_sinhc_sqrt(alpha)
        return scalar_part * self.output_scalar_mask + (values * coeff_part) @ self.bivector_to_output

    def _closed_simple_operator(self, values: torch.Tensor) -> torch.Tensor:
        alpha = (values * values * self._signs_for(values)).sum(dim=-1, keepdim=True)
        scalar_part, coeff_part = self._real_cosh_sinhc_sqrt(alpha)
        return scalar_part * self.operator_scalar_mask + (values * coeff_part) @ self.bivector_to_operator

    def _closed_biquadratic(self, values: torch.Tensor) -> torch.Tensor:
        if self.bivector_wedge is None or self.grade4_square is None or self.bivector_grade4_product is None:
            raise RuntimeError("closed_biquadratic executor is missing its grade-4 product plans")

        # For n <= 5, B^2 = s + K with K grade-4 and K^2 scalar, so exp(B)
        # closes over {1, B, K, B K}.
        scalar_square = (values * values * self._signs_for(values)).sum(dim=-1, keepdim=True)
        grade4_part = self.bivector_wedge.forward_compact(values, values)
        grade4_square = self.grade4_square.forward_compact(grade4_part, grade4_part)
        scalar_part, bivector_coeff, grade4_coeff, bivector_grade4_coeff = self._closed_biquadratic_coefficients(
            scalar_square,
            grade4_square,
        )

        output = (
            scalar_part * self.output_scalar_mask
            + (values * bivector_coeff) @ self.bivector_to_output
            + (grade4_part * grade4_coeff) @ self.grade4_to_output
        )
        bivector_grade4 = self.bivector_grade4_product.forward_compact(values, grade4_part)
        return output + bivector_grade4 * bivector_grade4_coeff

    def _spectral_local(self, values: torch.Tensor) -> torch.Tensor:
        if self.left_product is None:
            raise RuntimeError("spectral_local executor is missing its matrix-exp backward product plan")
        return _spectral_local_bivector_exp_op(
            values,
            self.bivector_to_nondegenerate_generator,
            self.spectral_local_scalar_mask,
            self.spectral_local_plane_masks,
            self.spectral_local_product_table,
            self.spectral_local_sparse_left_positions,
            self.spectral_local_sparse_right_positions,
            self.spectral_local_sparse_output_positions,
            self.spectral_local_sparse_coefficients,
            self.spectral_plane_bivector_map,
            self.spectral_plane_eye,
            self.spectral_plane_left_positions,
            self.spectral_plane_right_positions,
            self.spectral_plane_output_positions,
            self.spectral_plane_coefficients,
            self.spectral_plane_to_local,
            self.spectral_nilpotent_to_local,
            self.spectral_ideal_basis,
            self.spectral_lift_grades,
            self.spectral_lift_local_positions,
            self.spectral_lift_local_axes,
            self.spectral_lift_local_mask,
            self.spectral_lift_target_axes,
            self.spectral_lift_target_map,
            self.spectral_tolerances,
            self.left_product.left_active_positions,
            self.left_product.right_active_positions,
            self.left_product.output_positions,
            self.left_product.coefficients,
            self.operator_eye,
            self.operator_to_output,
            self.bivector_to_mixed_generator,
        )

    def _closed_biquadratic_coefficients(
        self,
        scalar_square: torch.Tensor,
        grade4_square: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        split_mask = grade4_square > self.eps_sq
        complex_mask = grade4_square < -self.eps_sq
        base_mask = ~(split_mask | complex_mask)

        zeros = torch.zeros_like(scalar_square)
        ones = torch.ones_like(scalar_square)

        split_scalar = torch.where(split_mask, scalar_square, zeros)
        split_mu = torch.sqrt(torch.where(split_mask, grade4_square, ones))
        plus = split_scalar + split_mu
        minus = split_scalar - split_mu
        c_plus, s_plus = self._real_cosh_sinhc_sqrt(plus)
        c_minus, s_minus = self._real_cosh_sinhc_sqrt(minus)
        split_scalar_coeff = 0.5 * (c_plus + c_minus)
        split_grade4_coeff = (c_plus - c_minus) / (2.0 * split_mu)
        split_bivector_coeff = 0.5 * (s_plus + s_minus)
        split_bivector_grade4_coeff = (s_plus - s_minus) / (2.0 * split_mu)

        complex_scalar = torch.where(complex_mask, scalar_square, zeros)
        complex_nu = torch.sqrt(torch.where(complex_mask, -grade4_square, ones))
        (
            complex_scalar_coeff,
            complex_bivector_coeff,
            complex_grade4_coeff,
            complex_bivector_grade4_coeff,
        ) = self._complex_biquadratic_coefficients(
            complex_scalar,
            complex_nu,
        )

        base_scalar = torch.where(base_mask, scalar_square, zeros)
        base_scalar_coeff, base_bivector_coeff = self._real_cosh_sinhc_sqrt(base_scalar)
        base_grade4_coeff = 0.5 * base_bivector_coeff
        base_bivector_grade4_coeff = self._real_sinhc_sqrt_derivative(
            base_scalar,
            base_scalar_coeff,
            base_bivector_coeff,
        )

        scalar_coeff = torch.where(
            split_mask,
            split_scalar_coeff,
            torch.where(complex_mask, complex_scalar_coeff, base_scalar_coeff),
        )
        bivector_coeff = torch.where(
            split_mask,
            split_bivector_coeff,
            torch.where(complex_mask, complex_bivector_coeff, base_bivector_coeff),
        )
        grade4_coeff = torch.where(
            split_mask,
            split_grade4_coeff,
            torch.where(complex_mask, complex_grade4_coeff, base_grade4_coeff),
        )
        bivector_grade4_coeff = torch.where(
            split_mask,
            split_bivector_grade4_coeff,
            torch.where(complex_mask, complex_bivector_grade4_coeff, base_bivector_grade4_coeff),
        )
        return scalar_coeff, bivector_coeff, grade4_coeff, bivector_grade4_coeff

    def _real_cosh_sinhc_sqrt(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positive = values > self.eps_sq
        negative = values < -self.eps_sq
        active = positive | negative
        theta = torch.sqrt(torch.where(active, values.abs(), torch.ones_like(values)))
        values_sq = values * values
        cosh_series = 1.0 + 0.5 * values + values_sq / 24.0 + (values_sq * values) / 720.0
        sinhc_series = 1.0 + values / 6.0 + values_sq / 120.0 + (values_sq * values) / 5040.0
        cosh_sqrt = torch.where(positive, torch.cosh(theta), torch.where(negative, torch.cos(theta), cosh_series))
        sinhc_sqrt = torch.where(
            positive,
            torch.sinh(theta) / theta,
            torch.where(negative, torch.sin(theta) / theta, sinhc_series),
        )
        return cosh_sqrt, sinhc_sqrt

    def _real_sinhc_sqrt_derivative(
        self,
        values: torch.Tensor,
        cosh_sqrt: torch.Tensor,
        sinhc_sqrt: torch.Tensor,
    ) -> torch.Tensor:
        active = values.abs() > self.eps_sq
        safe_values = torch.where(active, values, torch.ones_like(values))
        raw = (cosh_sqrt - sinhc_sqrt) / (2.0 * safe_values)
        values_sq = values * values
        series = 1.0 / 6.0 + values / 60.0 + values_sq / 1680.0 + (values_sq * values) / 90720.0
        return torch.where(active, raw, series)

    def _complex_biquadratic_coefficients(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        u, v, radius = self._complex_sqrt_parts(real, imag)
        sinh_u = torch.sinh(u)
        cosh_u = torch.cosh(u)
        sin_v = torch.sin(v)
        cos_v = torch.cos(v)
        cosh_sqrt_real = cosh_u * cos_v
        cosh_sqrt_imag = sinh_u * sin_v
        real_numerator = sinh_u * cos_v
        imag_numerator = cosh_u * sin_v
        sinhc_sqrt_real = (real_numerator * u + imag_numerator * v) / radius
        sinhc_sqrt_imag = (imag_numerator * u - real_numerator * v) / radius
        return cosh_sqrt_real, sinhc_sqrt_real, cosh_sqrt_imag / imag, sinhc_sqrt_imag / imag

    def _complex_sqrt_parts(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        radius = torch.sqrt(real * real + imag * imag)
        u_sq = (radius + real).clamp_min(0.0) * 0.5
        v_sq = (radius - real).clamp_min(0.0) * 0.5
        u = torch.sqrt(u_sq.clamp_min(self.eps_sq))
        v = torch.sqrt(v_sq.clamp_min(self.eps_sq))
        return u, v, radius

    def _left_matrix_exp(self, values: torch.Tensor) -> torch.Tensor:
        if self.left_product is None:
            raise RuntimeError("left_matrix_exp executor is missing its left-product plan")
        basis = self._basis_for(values)
        columns = self.left_product.forward_compact(values.unsqueeze(-2), basis)
        operator = columns.transpose(-1, -2)
        exp_operator = torch.matrix_exp(operator)
        even_output = exp_operator[..., :, self.operator_scalar_position]
        return even_output @ self.operator_to_output

    def _operator_identity(self, values: torch.Tensor) -> torch.Tensor:
        ones = values.new_ones(*values.shape[:-1], 1)
        return ones * self.operator_scalar_mask

    def _operator_to_output(self, operator_values: torch.Tensor) -> torch.Tensor:
        return operator_values @ self.operator_to_output

    def _basis_for(self, values: torch.Tensor) -> torch.Tensor:
        return self.operator_eye

    def _signs_for(self, values: torch.Tensor) -> torch.Tensor:
        return self.bivector_squared_signs


__all__ = ["BivectorExpExecutor"]
