# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Differentiable Clifford Algebra core.

Implements the geometric product, grade projections, and rotor operations
for arbitrary signatures Cl(p, q, r).
"""

from typing import Iterable, Optional

import torch

from clifra.core.foundation.basis import normalize_grades
from clifra.core.foundation.device import resolve_device, resolve_dtype
from clifra.core.foundation.host import AlgebraHostMixin
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout
from clifra.core.planning.exp import DEFAULT_BIVECTOR_EXP_EXECUTION_POLICY, BivectorExpExecutionPolicy
from clifra.core.planning.planner import GradePlanner
from clifra.core.planning.policy import (
    DEFAULT_PLANNING_LIMITS,
    DEFAULT_PRODUCT_EXECUTION_POLICY,
    PlanningLimits,
    ProductExecutionPolicy,
)


class AlgebraContext(AlgebraHostMixin):
    """Signature and planning host for layout-first Clifford Algebra."""

    def __init__(
        self,
        p: int,
        q: int = 0,
        r: int = 0,
        *,
        device="cuda",
        dtype: torch.dtype = torch.float32,
        default_grades: Optional[Iterable[int]] = None,
        planning_limits: Optional[PlanningLimits] = None,
        product_execution_policy: Optional[ProductExecutionPolicy] = None,
        bivector_exp_execution_policy: Optional[BivectorExpExecutionPolicy] = None,
    ):
        if p < 0 or q < 0 or r < 0:
            raise ValueError(f"signature counts must be non-negative, got Cl({p},{q},{r})")

        self.p = int(p)
        self.q = int(q)
        self.r = int(r)
        self.n = self.p + self.q + self.r
        self.dim = 1 << self.n
        self.num_grades = self.n + 1
        self.spec = AlgebraSpec(self.p, self.q, self.r)
        self._device = torch.device(resolve_device(device) if str(device) == "auto" else device)
        self._dtype = resolve_dtype(dtype)
        self.planning_limits = DEFAULT_PLANNING_LIMITS if planning_limits is None else planning_limits
        self.product_execution_policy = (
            DEFAULT_PRODUCT_EXECUTION_POLICY if product_execution_policy is None else product_execution_policy
        )
        self.bivector_exp_execution_policy = (
            DEFAULT_BIVECTOR_EXP_EXECUTION_POLICY
            if bivector_exp_execution_policy is None
            else bivector_exp_execution_policy
        )
        self._default_grades = None if default_grades is None else normalize_grades(default_grades, self.n)
        self._default_layout: Optional[GradeLayout] = None
        self._g1_indices_cache: dict[str, torch.Tensor] = {}
        self.planner = GradePlanner(self)
        self._sync_eps()

    @property
    def device(self):
        """Return the context device used for planned executor buffers."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Return the context floating-point dtype."""
        return self._dtype

    def bivector_squared_signs(self, *, device=None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Return ``(e_ab)^2`` signs in canonical grade-2 layout order."""
        return self.planner.bivector_squared_signs(
            device=self.device if device is None else device,
            dtype=self.dtype if dtype is None else dtype,
        )

    def _apply(self, fn):
        """Apply a PyTorch module-style device/dtype transform to cached executors."""
        probe = fn(torch.empty((), device=self.device, dtype=self.dtype))
        self._device = probe.device
        if probe.dtype.is_floating_point:
            self._dtype = probe.dtype
        self._sync_eps()
        self._g1_indices_cache.clear()
        self.planner._apply(fn)
        return self

    def to(self, device=None, dtype=None):
        """Move the context and cached executors."""
        if device is not None:
            self._device = torch.device(resolve_device(device) if str(device) == "auto" else device)
        if dtype is not None:
            self._dtype = resolve_dtype(dtype)
        self._sync_eps()
        self._g1_indices_cache.clear()
        self.planner.clear_cache()
        return self

    def geometric_product(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """Plan and execute a geometric product."""
        return self.projected_product(A, B, op="gp", **kwargs)

    def wedge(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """Plan and execute an exterior product."""
        return self.projected_product(A, B, op="wedge", **kwargs)

    def inner_product(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """Plan and execute the symmetric inner product route."""
        return self.projected_product(A, B, op="inner", **kwargs)

    def commutator(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """Plan and execute a commutator product."""
        return self.projected_product(A, B, op="commutator", **kwargs)

    def anti_commutator(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """Plan and execute an anti-commutator product."""
        return self.projected_product(A, B, op="anti_commutator", **kwargs)

    def grade_projection(self, mv: torch.Tensor, grade: int, **kwargs) -> torch.Tensor:
        """Project declared multivector coefficients to one grade."""
        kwargs.setdefault("output_grades", (int(grade),))
        return self.planned_unary(mv, op="grade_projection", **kwargs)

    def embed_vector(self, vectors: torch.Tensor) -> torch.Tensor:
        """Embed grade-1 vector coordinates into full-lane multivector coefficients."""
        if vectors.shape[-1] != self.n:
            raise ValueError(f"vectors last dimension must be {self.n}, got {vectors.shape[-1]}")
        output = vectors.new_zeros(*vectors.shape[:-1], self.dim)
        return output.index_copy(-1, self._basis_vector_indices(vectors.device), vectors)

    def reverse(self, mv: torch.Tensor, **kwargs) -> torch.Tensor:
        """Reverse full-lane or active multivector coefficients."""
        return self.planned_unary(mv, op="reverse", **kwargs)

    def grade_involution(self, mv: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply grade involution to full-lane or active multivector coefficients."""
        return self.planned_unary(mv, op="grade_involution", **kwargs)

    def clifford_conjugation(self, mv: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply Clifford conjugation to full-lane or active multivector coefficients."""
        return self.planned_unary(mv, op="clifford_conjugation", **kwargs)

    def exp(
        self,
        mv: torch.Tensor,
        *,
        input_grades=None,
        output_grades=None,
        input_layout: Optional[GradeLayout] = None,
        output_layout: Optional[GradeLayout] = None,
        return_layout: bool = False,
    ) -> torch.Tensor:
        """Exponentiate a declared bivector through the shared planner route."""
        return AlgebraHostMixin.exp(
            self,
            mv,
            input_grades=input_grades,
            output_grades=output_grades,
            input_layout=input_layout,
            output_layout=output_layout,
            return_layout=return_layout,
        )

    def _basis_vector_indices(self, device) -> torch.Tensor:
        resolved = torch.device(device)
        key = str(resolved)
        cached = self._g1_indices_cache.get(key)
        if cached is None:
            cached = torch.tensor([1 << bit for bit in range(self.n)], dtype=torch.long, device=resolved)
            self._g1_indices_cache[key] = cached
        return cached

    def _sync_eps(self) -> None:
        finfo = torch.finfo(self.dtype)
        self.eps = float(finfo.eps)
        self.eps_sq = float(finfo.eps**2)
