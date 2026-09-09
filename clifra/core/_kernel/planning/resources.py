# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Static allocation limits, independent of route-selection policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceLimits:
    """User-configurable static allocation boundaries."""

    warn_lanes: int = 2048
    max_lanes: int = 4096
    warn_pairs: int = 1_000_000
    max_pairs: int = 8_000_000

    def __post_init__(self) -> None:
        values = (self.warn_lanes, self.max_lanes, self.warn_pairs, self.max_pairs)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("resource limits must be non-negative integers")


DEFAULT_RESOURCE_LIMITS = ResourceLimits()


@dataclass(frozen=True)
class ResourceRequirements:
    """Conservative width and static pair/interaction footprint.

    ``pairs`` adds route-owned resident structures and child footprints that
    coexist. A route may also include its larger known fixed-shape temporary.
    Caller-controlled batch dimensions are excluded.
    """

    lanes: int = 0
    pairs: int = 0

    def __post_init__(self):
        for value in (self.lanes, self.pairs):
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError("resource requirements must be non-negative integers")

    def rejection_reason(self, limits: ResourceLimits) -> str | None:
        if self.lanes > limits.max_lanes:
            return f"intermediate lanes {self.lanes} exceed max_lanes={limits.max_lanes}"
        if self.pairs > limits.max_pairs:
            return f"static pair/interaction footprint {self.pairs} exceeds max_pairs={limits.max_pairs}"
        return None

    def warning_reason(self, limits: ResourceLimits) -> str | None:
        warnings = []
        if self.lanes >= limits.warn_lanes:
            warnings.append(f"intermediate lanes {self.lanes} are near max_lanes={limits.max_lanes}")
        if self.pairs >= limits.warn_pairs:
            warnings.append(f"static pair/interaction footprint {self.pairs} is near max_pairs={limits.max_pairs}")
        return "; ".join(warnings) or None
