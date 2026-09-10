"""Static allocation boundaries for planned Clifford operations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceLimits:
    """Immutable coefficient-width and static interaction budgets.

    Counts cover execution structures and known fixed-shape temporaries, not
    caller-controlled batch dimensions or total device memory. Warnings start
    at the warning thresholds; requirements above a maximum are rejected.
    """

    warn_lanes: int = 2048
    max_lanes: int = 4096
    warn_pairs: int = 1_000_000
    max_pairs: int = 8_000_000

    def __post_init__(self) -> None:
        values = (self.warn_lanes, self.max_lanes, self.warn_pairs, self.max_pairs)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("resource limits must be non-negative integers")


__all__ = ["ResourceLimits"]
