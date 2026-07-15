"""Data models shared by the benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from clifra.core import GradeLayout


@dataclass(frozen=True)
class SignatureSpec:
    """One Clifford signature selected by the benchmark matrix."""

    p: int
    q: int = 0
    r: int = 0

    @property
    def n(self) -> int:
        return self.p + self.q + self.r

    @property
    def label(self) -> str:
        return f"Cl({self.p},{self.q},{self.r})"


@dataclass(frozen=True)
class SweepConfig:
    sweep_id: str
    layout_preset: str
    dimensions: tuple[int, ...]
    signature_families: tuple[str, ...]
    signatures: tuple[SignatureSpec, ...]
    devices: tuple[str, ...]
    dtypes: tuple[str, ...]
    batch_sizes: tuple[int, ...]
    compile_modes: tuple[str, ...]
    channels: int
    actions: int
    pairs: int
    cases: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TimingConfig:
    warmup_calls: int
    samples: int
    backward_warmup_calls: int
    backward_samples: int


@dataclass(frozen=True)
class ResourceConfig:
    max_estimated_bytes: int
    max_layout_lanes: int
    safety_factor: float


@dataclass(frozen=True)
class ProfilerConfig:
    enabled: bool
    case_ids: tuple[str, ...]
    record_shapes: bool
    profile_memory: bool


@dataclass(frozen=True)
class OutputConfig:
    root: Path
    publish: bool
    docs_root: Path
    baseline: Path | None


@dataclass(frozen=True)
class BenchmarkConfig:
    schema_version: int
    seed: int
    sweeps: tuple[SweepConfig, ...]
    timing: TimingConfig
    resources: ResourceConfig
    cumulative: tuple[dict[str, Any], ...]
    profiler: ProfilerConfig
    output: OutputConfig
    raw: dict[str, Any] = field(repr=False)


@dataclass
class PreparedCase:
    """One planned callable and its concrete benchmark tensors."""

    case_id: str
    kind: str
    operation: str
    module: Callable[..., torch.Tensor] | nn.Module
    args: tuple[torch.Tensor, ...]
    input_layout: GradeLayout | None
    output_layout: GradeLayout | None
    metadata: dict[str, Any]
    backward: bool
