"""Explicit executor extensions, independent of built-in planning representations.

Providers implement one mathematical family/route and return an nn.Module whose
forward accepts the declared compact coefficient tensors and returns compact
output. Ordinary leading-dimension broadcasting applies to products; clifra
supplies pairwise expansion. None input contracts denote ordinary tensor operands.
Provider identity and behavior must remain fixed while a registry is in use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from clifra.core.tensors import TensorContract


@dataclass(frozen=True)
class ExecutorRequest:
    """Resolved operation, compact lane contracts, and requested tensor placement.

    No planner, algebra instance, tree, or built-in algorithm options are exposed.
    Providers must reject operations or contract combinations they do not implement.
    """

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
        if any(not isinstance(value, bool) for value in (self.exact, self.truncated, self.value_dependent)):
            raise ValueError("quality guarantees must be booleans")
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
    """One deterministic route; build must use its successful assessment unchanged."""

    @property
    def identity(self) -> tuple[str, str]: ...

    def assess(self, request: ExecutorRequest) -> Assessment | Rejected: ...

    def build(self, request: ExecutorRequest, assessment: Assessment) -> nn.Module: ...


@dataclass(frozen=True)
class ExecutorRegistry:
    """Immutable explicit provider collection, replacing the default registry.

    Use default().providers when retaining built-ins alongside supplied providers.
    Registration order breaks equal policy scores. New route identities use declared
    forward work as their default score; built-ins retain their regime rules.
    There is no global registry.
    """

    providers: tuple[ExecutorProvider, ...]

    def __post_init__(self):
        object.__setattr__(self, "providers", tuple(self.providers))
        identities = tuple(provider.identity for provider in self.providers)
        if any(
            not isinstance(key, tuple) or len(key) != 2 or any(not isinstance(value, str) or not value for value in key)
            for key in identities
        ):
            raise ValueError("provider identity must be a non-empty (family, route) pair")
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate executor family/route")

    @classmethod
    def default(cls):
        """Return a fresh immutable collection of the built-in route providers."""
        from ._kernel.execution.providers import builtin_providers

        return cls(builtin_providers())


__all__ = ["ExecutorRegistry", "ExecutorProvider", "ExecutorRequest", "Assessment", "Rejected"]
