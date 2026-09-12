"""Reusable request and adapter boundary for benchmark measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .cases import BenchmarkCase, ExecutionPlacement

TIMING_MODES = frozenset(
    (
        "construction_setup",
        "planning_preparation",
        "first_invocation",
        "steady_forward",
        "forward_backward",
    )
)


@dataclass(frozen=True)
class BenchmarkRequest:
    """One semantic workload with execution placement and timing stage."""

    case: BenchmarkCase
    placement: ExecutionPlacement
    mode: str

    def __post_init__(self) -> None:
        if self.mode not in TIMING_MODES:
            raise ValueError(f"unsupported timing mode {self.mode!r}")

    @property
    def row_id(self) -> str:
        selection = self.placement.selection
        return (
            f"{self.case.case_id}:{self.placement.dtype}:{self.placement.device}:"
            f"{self.mode}:{selection.mode}:{selection.route or 'normal'}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "case": self.case.to_dict(),
            "placement": self.placement.to_dict(),
            "timing_mode": self.mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> BenchmarkRequest:
        return cls(
            BenchmarkCase.from_dict(data["case"]),
            ExecutionPlacement.from_dict(data["placement"]),
            str(data["timing_mode"]),
        )


@dataclass(frozen=True)
class MeasurementIssue:
    status: str
    error_type: str
    message: str


@dataclass(frozen=True)
class ConstructedBenchmark:
    """Adapter-owned construction state plus common invocation arguments."""

    state: Any
    arguments: tuple[Any, ...]
    input_metadata: tuple[dict[str, object] | None, ...]
    output_metadata: dict[str, object] | None


@dataclass(frozen=True)
class PreparedBenchmark:
    """A fixed executable and its backend-specific descriptive metadata."""

    operation: Any
    constructed: ConstructedBenchmark
    metadata: dict[str, object]


class BenchmarkAdapter(Protocol):
    """Minimal backend operations needed by the common measurement engine."""

    name: str

    def preflight(self, request: BenchmarkRequest) -> MeasurementIssue | None: ...

    def construct(self, request: BenchmarkRequest, *, requires_grad: bool = False) -> ConstructedBenchmark: ...

    def prepare(self, request: BenchmarkRequest, constructed: ConstructedBenchmark) -> PreparedBenchmark: ...

    def invoke(self, prepared: PreparedBenchmark, *, backward: bool) -> None: ...

    def reset(self, prepared: PreparedBenchmark, *, backward: bool) -> None: ...

    def synchronize(self, request: BenchmarkRequest) -> None: ...

    def synchronization_description(self, request: BenchmarkRequest) -> str: ...
