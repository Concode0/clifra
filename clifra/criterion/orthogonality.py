# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Orthogonality criteria for grade-constrained training."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import torch

from clifra.core.foundation.module import CliffordModule
from clifra.functional.orthogonality import (
    diagnostics,
    grade_masks,
    parasitic_energy,
    project_to_target_grades,
    target_mask_from_grades,
)


@dataclass
class OrthogonalitySettings:
    """Configuration for target-grade confinement."""

    enabled: bool = True
    mode: str = "loss"
    weight: float = 0.1
    target_grades: Optional[list[int]] = None
    tolerance: float = 1e-4
    monitor_interval: int = 10
    coupling_warn_threshold: float = 0.3


class StrictOrthogonality(CliffordModule):
    """Enforce grade confinement by loss penalty or hard projection."""

    def __init__(self, algebra, settings: Optional[OrthogonalitySettings] = None):
        """Initialize target-grade confinement."""
        super().__init__(algebra)
        self.settings = OrthogonalitySettings() if settings is None else settings
        masks = grade_masks(self.algebra.n + 1, self.algebra.dim)
        target_mask = target_mask_from_grades(masks, self.settings.target_grades)
        self.register_buffer("grade_masks_tensor", masks)
        self.register_buffer("target_mask", target_mask)
        self.register_buffer("parasitic_mask", ~target_mask)

    def parasitic_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Return mean squared energy in non-target grade components."""
        return parasitic_energy(x, self.parasitic_mask)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` with non-target grade lanes zeroed."""
        return project_to_target_grades(x, self.target_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return either a non-target-grade penalty or projected values."""
        if not self.settings.enabled:
            return x if self.settings.mode == "project" else x.new_zeros(())
        if self.settings.mode == "project":
            return self.project(x)
        return self.settings.weight * self.parasitic_energy(x)

    def anneal_weight(self, epoch: int, warmup_epochs: int, total_epochs: int) -> float:
        """Return the linearly warmed confinement weight."""
        if warmup_epochs <= 0:
            return self.settings.weight
        return self.settings.weight * min(epoch / warmup_epochs, 1.0)

    @torch.no_grad()
    def diagnostics(self, x: torch.Tensor) -> dict:
        """Return grade-energy and cross-grade coupling diagnostics."""
        report = diagnostics(
            x,
            self.grade_masks_tensor,
            target_grades=self.settings.target_grades,
            tolerance=self.settings.tolerance,
        )
        max_off = report["coupling_max_off_diag"]
        if max_off > self.settings.coupling_warn_threshold:
            warnings.warn(
                f"High cross-grade coupling detected: max_off_diag={max_off:.4f} "
                f"(threshold={self.settings.coupling_warn_threshold}). "
                "The network may be mixing grades.",
                RuntimeWarning,
                stacklevel=2,
            )
        return report

    def format_diagnostics(self, x: torch.Tensor) -> str:
        """Return a compact text report for grade-confinement diagnostics."""
        report = self.diagnostics(x)
        energies = report["grade_energies"]
        target_grades = set(self.settings.target_grades or range(self.algebra.n + 1))

        lines = ["  Grade energies (ASCII bar):"]
        max_energy = max(energies.values()) + 1e-12
        bar_width = 16
        for grade, energy in energies.items():
            filled = int(bar_width * energy / max_energy)
            bar = "#" * filled + " " * (bar_width - filled)
            tag = "  <- target" if grade in target_grades else ""
            lines.append(f"    G{grade} [{bar}]  {energy:.4f}{tag}")

        lines.append(f"  Parasitic ratio: {report['parasitic_ratio']:.4%}")
        status = "YES" if report["orthogonality_satisfied"] else "NO"
        lines.append(f"  Orthogonality satisfied: {status} (tol={self.settings.tolerance})")
        if report["coupling_matrix"] is not None:
            lines.append(f"  Coupling: max_off_diag={report['coupling_max_off_diag']:.4f}")
        return "\n".join(lines)
