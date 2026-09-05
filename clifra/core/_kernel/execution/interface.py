"""Small ordinary-Python contracts for a single executor route."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from clifra.core.tensors import TensorContract


@dataclass(frozen=True)
class ExecutorRequest:
    family: str
    operation: str
    inputs: tuple[TensorContract | None, ...]
    output: TensorContract
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self):
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "device", torch.device(self.device))


@dataclass(frozen=True, kw_only=True)
class Assessment:
    """An executable route. Exact means no deliberate approximation, not zero roundoff.

    Counts describe static coefficient lanes and interaction/table entries.
    Work and storage estimates are selection facts, not measured performance.
    Preparation belongs exclusively to the provider that returned this object.
    """

    lanes: int
    pairs: int
    forward_work: float
    exact: bool
    backward_work: float = 0.0
    peak_bytes: int = 0
    compile_work: float = 0.0
    truncated: bool = False
    value_dependent: bool = False
    preparation: object = None

    def __post_init__(self):
        for name in ("lanes", "pairs", "peak_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("forward_work", "backward_work", "compile_work"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.exact and (self.truncated or self.value_dependent):
            raise ValueError("exact routes cannot be truncated or value-dependent")


@dataclass(frozen=True)
class Rejected:
    """An unavailable route has no cost, quality guarantee, or preparation."""

    reason: str

    def __post_init__(self):
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("rejection requires a non-empty reason")


class ExecutorProvider(Protocol):
    @property
    def identity(self) -> tuple[str, str]: ...

    def assess(self, request: ExecutorRequest) -> Assessment | Rejected: ...

    def build(self, request: ExecutorRequest, assessment: Assessment) -> nn.Module: ...
