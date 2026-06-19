# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Phase-based loss-weight curricula for continuum optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

import torch


class LossWeightSchedule(Protocol):
    """Return an effective loss weight for a named optimization term."""

    def weight(
        self,
        engine,
        names: str | Sequence[str],
        reference: torch.Tensor,
        *,
        base_weight: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """Return the scheduled weight as a tensor on ``reference``'s device."""
        ...


@dataclass(frozen=True)
class ConstantCurriculum:
    """Pass through each term's base weight."""

    def weight(
        self,
        engine,
        names: str | Sequence[str],
        reference: torch.Tensor,
        *,
        base_weight: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        del engine, names
        return torch.as_tensor(base_weight, device=reference.device, dtype=reference.dtype)


@dataclass(frozen=True)
class CurriculumKnot:
    """One phase knot in normalized optimization progress units."""

    phase: float
    weights: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PhaseCurriculum:
    """Piecewise-linear curriculum over normalized phase units.

    ``phase`` values are normalized optimization progress positions in ``[0, 1]``.
    A caller can address a term by any alias the engine provides, for example
    ``"target:name"``, ``"policy:name"``, ``"target"``, ``"policy"``, or ``"*"``.
    Omitted terms keep their caller-provided base weight.
    """

    knots: tuple[CurriculumKnot, ...]

    def __init__(self, knots: Sequence[CurriculumKnot | tuple[float, Mapping[str, float]]]):
        parsed = tuple(knot if isinstance(knot, CurriculumKnot) else CurriculumKnot(*knot) for knot in knots)
        object.__setattr__(self, "knots", _validate_knots(parsed))

    def weight(
        self,
        engine,
        names: str | Sequence[str],
        reference: torch.Tensor,
        *,
        base_weight: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        aliases = (names,) if isinstance(names, str) else tuple(names)
        base = torch.as_tensor(base_weight, device=reference.device, dtype=reference.dtype)
        if not self.knots:
            return base

        values = [float(_lookup_weight(knot.weights, aliases, base_weight)) for knot in self.knots]
        if len(values) == 1:
            return reference.new_tensor(values[0])

        phases = reference.new_tensor([float(knot.phase) for knot in self.knots])
        weights = reference.new_tensor(values)
        progress = engine.fit_progress_like(reference).clamp(phases[0], phases[-1])
        left_phase = phases[:-1]
        right_phase = phases[1:]
        left_weight = weights[:-1]
        right_weight = weights[1:]
        amount = ((progress - left_phase) / (right_phase - left_phase).clamp_min(1e-8)).clamp(0.0, 1.0)
        segment_weight = left_weight + amount * (right_weight - left_weight)
        active = ((progress >= left_phase) & (progress <= right_phase)).to(reference.dtype)
        return (segment_weight * active).sum() / active.sum().clamp_min(1.0)


def _lookup_weight(weights: Mapping[str, float], aliases: Sequence[str], base_weight: float | torch.Tensor) -> float:
    for alias in aliases:
        if alias in weights:
            return float(weights[alias])
    return float(base_weight)


def _validate_knots(knots: tuple[CurriculumKnot, ...]) -> tuple[CurriculumKnot, ...]:
    last = -float("inf")
    for knot in knots:
        phase = float(knot.phase)
        if phase < 0.0 or phase > 1.0:
            raise ValueError(f"curriculum phase must be in [0, 1], got {phase}")
        if phase <= last:
            raise ValueError("curriculum phases must be strictly increasing")
        last = phase
    return knots
