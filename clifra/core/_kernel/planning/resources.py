# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Static allocation limits, independent of route-selection policy."""

from __future__ import annotations

from dataclasses import dataclass

from clifra.core.resources import ResourceLimits

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
