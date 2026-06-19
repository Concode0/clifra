# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Pure orthogonality formulas for multivector grade lanes.

The final axis is the Clifford lane axis. Full-lane values are ``[..., D]`` and
compact layout values are ``[..., L]``. Grade masks are ``[G, D]`` or
``[G, L]``; boolean lane masks are ``[D]`` or ``[L]``.
"""

from __future__ import annotations

import torch


def grade_masks(n_grades: int, dim: int, *, device=None) -> torch.Tensor:
    """Return ``[G, D]`` or ``[G, L]`` boolean masks keyed by basis-blade grade."""
    masks = torch.zeros(n_grades, dim, dtype=torch.bool, device=device)
    for idx in range(dim):
        grade = int(idx).bit_count()
        if grade < n_grades:
            masks[grade, idx] = True
    return masks


def target_mask_from_grades(masks: torch.Tensor, target_grades: list[int] | None) -> torch.Tensor:
    """Return a ``[D]`` or ``[L]`` boolean lane mask for the requested grades."""
    if target_grades is None:
        return torch.ones(masks.shape[-1], dtype=torch.bool, device=masks.device)
    return masks[target_grades].any(dim=0)


def parasitic_energy(values: torch.Tensor, parasitic_mask: torch.Tensor) -> torch.Tensor:
    """Return mean squared energy in non-target lanes of ``[..., D]`` or ``[..., L]`` values."""
    parasitic = values[..., parasitic_mask]
    if parasitic.numel() == 0:
        return values.new_zeros(())
    return (parasitic**2).mean()


def project_to_target_grades(values: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Return ``[..., D]`` or ``[..., L]`` values with non-target lanes zeroed."""
    return values * target_mask.to(device=values.device, dtype=values.dtype)


def grade_energies(values: torch.Tensor, masks: torch.Tensor) -> dict[int, float]:
    """Return mean squared energy per grade for ``[..., D]`` or ``[..., L]`` values."""
    energies = {}
    for grade in range(masks.shape[0]):
        components = values[..., masks[grade].to(device=values.device)]
        energies[grade] = (components**2).mean().item()
    return energies


def parasitic_ratio(values: torch.Tensor, masks: torch.Tensor, target_grades: list[int] | None) -> float:
    """Return the fraction of total grade energy outside ``target_grades``."""
    if target_grades is None:
        return 0.0
    energies = grade_energies(values, masks)
    total = sum(energies.values()) + 1e-12
    target = sum(energies.get(grade, 0.0) for grade in target_grades)
    return 1.0 - target / total


def cross_grade_coupling(values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Return the ``[G, G]`` correlation matrix of grade energies across batch axis 0."""
    batch = values.shape[0]
    energies = []
    for grade in range(masks.shape[0]):
        components = values[..., masks[grade].to(device=values.device)]
        energies.append((components**2).reshape(batch, -1).sum(dim=-1))
    stacked = torch.stack(energies, dim=0)
    normalized = (stacked - stacked.mean(dim=1, keepdim=True)) / (stacked.std(dim=1, keepdim=True) + 1e-8)
    return (normalized @ normalized.t()) / batch


def diagnostics(
    values: torch.Tensor,
    masks: torch.Tensor,
    *,
    target_grades: list[int] | None,
    tolerance: float,
) -> dict:
    """Return grade-energy, parasitic-ratio, and coupling diagnostics for multivectors."""
    energies = grade_energies(values, masks)
    ratio = parasitic_ratio(values, masks, target_grades)

    coupling = None
    max_off = 0.0
    if values.dim() >= 2 and values.shape[0] > 1:
        coupling = cross_grade_coupling(values, masks)
        n = coupling.shape[0]
        off_mask = ~torch.eye(n, dtype=torch.bool, device=coupling.device)
        if off_mask.any():
            max_off = coupling[off_mask].abs().max().item()

    return {
        "grade_energies": energies,
        "parasitic_ratio": ratio,
        "coupling_matrix": coupling,
        "orthogonality_satisfied": ratio < tolerance,
        "coupling_max_off_diag": max_off,
    }
