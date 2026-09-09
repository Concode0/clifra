"""Explicit executor extensions, independent of built-in planning representations.

Providers implement one mathematical family/route and return an nn.Module whose
forward accepts the declared compact coefficient tensors and returns compact
output. Ordinary leading-dimension broadcasting applies to products; clifra
supplies pairwise expansion. None input contracts denote ordinary tensor operands.
Provider identity and behavior must remain fixed while a registry is in use.
"""

from __future__ import annotations

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
    """Acceptance of the declared mathematical operation and tensor contracts.

    No deliberate truncation or approximation of the operation is implied by
    acceptance. Ordinary numerical roundoff is permitted. Providers must report
    conservative maximum coefficient-lane and interaction/table-entry counts
    for their intermediates, excluding caller-controlled batch dimensions.
    Zero declares no additional requirement; input/output widths are guarded
    independently. Assessment must not allocate execution buffers.

    Preparation belongs exclusively to the provider. The exact assessment and
    request objects returned/received here are passed unchanged to build().
    """

    lanes: int = 0
    pairs: int = 0
    preparation: object = None

    def __post_init__(self):
        for name in ("lanes", "pairs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class Rejected:
    """An unavailable route has only a reason, with no preparation."""

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
    The first eligible provider establishes precedence. If it is external, it
    wins. If it is built-in, private policy chooses among eligible built-ins.
    Thus prepended external providers override built-ins and appended external
    providers are fallbacks when no built-in can execute the request. Rejected
    and over-budget providers do not establish precedence. External routes are
    never compared using built-in work scores.
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
        from ._kernel.providers import builtin_providers

        return cls(builtin_providers())


__all__ = ["ExecutorRegistry", "ExecutorProvider", "ExecutorRequest", "Assessment", "Rejected"]
