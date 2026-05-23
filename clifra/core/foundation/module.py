# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Base PyTorch module for components that share a Clifford algebra."""

from typing import Iterable, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn

from clifra.core.foundation.layout import GradeLayout


@runtime_checkable
class AlgebraLike(Protocol):
    """Protocol implemented by dense kernels and planned algebra contexts."""

    p: int
    q: int
    r: int
    n: int
    dim: int
    num_grades: int
    eps: float
    eps_sq: float
    planner: object
    planning_limits: object

    @property
    def device(self):
        """Return the device of algebra-owned buffers."""
        ...

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of algebra-owned floating-point buffers."""
        ...

    def _apply(self, fn):
        """Move/cast algebra-owned buffers."""
        ...

    def layout(self, grades: Optional[Iterable[int]] = None) -> GradeLayout:
        """Return a compact grade layout or the algebra's default layout."""
        ...

    def default_layout(self) -> GradeLayout:
        """Return the algebra default layout."""
        ...

    def resolve_layout(
        self,
        *,
        layout: Optional[GradeLayout] = None,
        grades: Optional[Iterable[int]] = None,
        mv=None,
    ) -> GradeLayout:
        """Resolve static layout metadata for tensors or multivectors."""
        ...

    def grade_indices(self, grades: Iterable[int], *, device=None) -> torch.Tensor:
        """Return canonical dense basis indices for ``grades``."""
        ...

    def hermitian_signs(
        self,
        layout: Optional[GradeLayout] = None,
        *,
        grades: Optional[Iterable[int]] = None,
        device=None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return Hermitian signs for a dense or compact layout."""
        ...

    def projected_product(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply a declared grade-restricted binary product."""
        ...

    def geometric_product(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply the geometric product."""
        ...

    def wedge(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply the exterior product."""
        ...

    def inner_product(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply the inner product."""
        ...

    def commutator(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply the commutator product."""
        ...

    def anti_commutator(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply the anti-commutator product."""
        ...

    def grade_projection(self, mv: torch.Tensor, grade: int, **kwargs) -> torch.Tensor:
        """Project to a single grade."""
        ...

    def reverse(self, mv: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply reversion."""
        ...

    def grade_involution(self, mv: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply grade involution."""
        ...

    def clifford_conjugation(self, mv: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply Clifford conjugation."""
        ...

    def exp(self, mv: torch.Tensor, **kwargs) -> torch.Tensor:
        """Exponentiate a declared bivector."""
        ...

    def planned_linear_action(self, values: torch.Tensor, matrix: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply a vector-space linear action to dense or compact grade lanes."""
        ...

    def paired_bivector_action(
        self,
        values: torch.Tensor,
        left_weights: torch.Tensor,
        right_weights: torch.Tensor,
        channel_to_pair: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Apply independent left/right bivector rotor pairs."""
        ...


class CliffordModule(nn.Module):
    """Base module for Clifford algebra-aware components.

    ``CliffordModule`` belongs to :mod:`core` because it is shared by layers,
    functional losses/activations, models, examples, and experiments. Keeping it
    out of :mod:`layers` prevents functional code from importing the eager layer
    package just to subclass this base type.

    The module stores a shared algebra reference without registering it as a
    PyTorch submodule. In Versor, one algebra instance often owns the
    precomputed geometric tensors used by many modules.
    """

    def __init__(self, algebra: AlgebraLike):
        """Set up the module with a shared algebra instance."""
        super().__init__()
        # Bypass nn.Module.__setattr__ to avoid registering algebra as a child.
        object.__setattr__(self, "_algebra", algebra)

    @property
    def algebra(self) -> AlgebraLike:
        """Return the shared algebra instance."""
        return self._algebra

    @property
    def p(self):
        return self._algebra.p

    @property
    def q(self):
        return self._algebra.q

    @property
    def r(self):
        return self._algebra.r

    def _apply(self, fn):
        """Apply device/dtype moves to this module and its shared algebra."""
        result = super()._apply(fn)
        if self._algebra is not None:
            self._algebra._apply(fn)
        return result

    def forward(self, x):
        """Perform the forward pass computation."""
        raise NotImplementedError


def is_dense_kernel_host(algebra: object) -> bool:
    """Return whether ``algebra`` owns dense Clifford kernel tables."""
    from clifra.core.runtime.algebra import CliffordAlgebra

    return isinstance(algebra, CliffordAlgebra)


def require_dense_kernel_host(algebra: object, feature: str) -> None:
    """Raise a consistent error for features that need dense kernel tables."""
    if not is_dense_kernel_host(algebra):
        raise ValueError(f"{feature} requires dense CliffordAlgebra kernels; declare compact layouts or use dense mode.")
