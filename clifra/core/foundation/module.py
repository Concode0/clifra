# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Base PyTorch module for components that share a Clifford algebra."""

from typing import Iterable, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn

from clifra.core.foundation.layout import GradeLayout


@runtime_checkable
class AlgebraLike(Protocol):
    """Protocol implemented by planner-capable algebra hosts and references."""

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
    product_execution_policy: object

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
        """Return canonical basis indices for ``grades``."""
        ...

    def conjugate_scalar_form_signs(
        self,
        layout: Optional[GradeLayout] = None,
        *,
        grades: Optional[Iterable[int]] = None,
        device=None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return signs for the signed Clifford-conjugation scalar form."""
        ...

    def projected_product(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply a declared grade-restricted binary product."""
        ...

    def plan_product(self, **kwargs):
        """Return a hot-path product handle for compact-lane values."""
        ...

    def plan_unary(self, **kwargs):
        """Return a hot-path unary handle for compact-lane values."""
        ...

    def plan_norm_sq(self, **kwargs):
        """Return a diagonal norm executor for compact-lane values."""
        ...

    def plan_dual(self, **kwargs):
        """Return a dual/pseudoscalar executor for compact-lane values."""
        ...

    def plan_exp(self, **kwargs):
        """Return a bivector exponential executor for compact-lane values."""
        ...

    def plan_sandwich_action(self, **kwargs):
        """Return a full-layout sandwich action handle."""
        ...

    def plan_versor_action(self, **kwargs):
        """Return a grade-1 or grade-2 versor action handle."""
        ...

    def plan_multi_versor_action(self, **kwargs):
        """Return a weighted multi-versor action handle."""
        ...

    def plan_paired_bivector_action(self, **kwargs):
        """Return an independent left/right bivector action handle."""
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

    def left_contraction(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply left contraction."""
        ...

    def right_contraction(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply right contraction."""
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

    def norm_sq(self, mv: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return the algebraic squared norm."""
        ...

    def dual(self, mv: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply the Hodge dual / right pseudoscalar product."""
        ...

    def pseudoscalar_product(self, mv: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply right multiplication by the pseudoscalar."""
        ...

    def blade_inverse(self, blade: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return the inverse of a non-null blade."""
        ...

    def blade_project(self, values: torch.Tensor, blade: torch.Tensor, **kwargs) -> torch.Tensor:
        """Project values onto a blade subspace."""
        ...

    def blade_reject(self, values: torch.Tensor, blade: torch.Tensor, **kwargs) -> torch.Tensor:
        """Reject values from a blade subspace."""
        ...

    def reflect(self, values: torch.Tensor, normal: torch.Tensor, **kwargs) -> torch.Tensor:
        """Reflect values through a vector normal."""
        ...

    def versor_product(self, versor: torch.Tensor, values: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply a general versor product."""
        ...

    def exp(self, mv: torch.Tensor, **kwargs) -> torch.Tensor:
        """Exponentiate a declared bivector."""
        ...

    def planned_linear_action(self, values: torch.Tensor, matrix: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply a vector-space linear action to full-lane or compact grade lanes."""
        ...

    def sandwich_action_matrices(self, left: torch.Tensor, right: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """Return full-layout sandwich action matrices."""
        ...

    def sandwich_product(self, left: torch.Tensor, values: torch.Tensor, right: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """Apply one full-layout sandwich action per leading batch item."""
        ...

    def per_channel_sandwich(
        self,
        left: torch.Tensor,
        values: torch.Tensor,
        right: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Apply one full-layout sandwich action per channel."""
        ...

    def multi_rotor_sandwich(
        self,
        left: torch.Tensor,
        values: torch.Tensor,
        right: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Apply every full-layout sandwich action to every input channel."""
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
    criteria, and functional modules. Keeping it out of :mod:`layers` prevents
    functional code from importing the eager layer package just to subclass this
    base type.

    The module stores a shared algebra reference without registering it as a
    PyTorch submodule. In clifra, one algebra instance often owns the
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
        """Return the algebra's positive metric dimension."""
        return self._algebra.p

    @property
    def q(self):
        """Return the algebra's negative metric dimension."""
        return self._algebra.q

    @property
    def r(self):
        """Return the algebra's null metric dimension."""
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
