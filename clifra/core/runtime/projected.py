# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Shared projected-product facade for algebra hosts."""

from __future__ import annotations

from typing import Iterable, Optional

import torch

from clifra.core.foundation.layout import GradeLayout
from clifra.core.foundation.validation import check_multivector
from clifra.core.planning.action import apply_graded_linear_action
from clifra.core.runtime.accessors import default_layout as _default_layout
from clifra.core.runtime.accessors import grade_indices as _grade_indices
from clifra.core.runtime.accessors import hermitian_signs as _hermitian_signs
from clifra.core.runtime.accessors import materialize_dense
from clifra.core.runtime.accessors import resolve_layout as _resolve_layout
from clifra.core.runtime.actions import apply_multi_versor_action, apply_versor_action
from clifra.core.runtime.actions import grade_norms as _grade_norms
from clifra.core.storage import resolve_planned_dispatch


class AlgebraRuntimeMixin:
    """Shared runtime protocol for dense kernels and planned contexts."""

    def layout(self, grades: Optional[Iterable[int]] = None) -> GradeLayout:
        """Return a compact grade layout or the algebra's default layout."""
        if grades is None:
            return self.default_layout()
        return self.planner.layout(grades)

    def default_layout(self) -> GradeLayout:
        """Return the default layout using the central fallback policy."""
        return _default_layout(self)

    def product_executor(
        self,
        *,
        left_grades,
        right_grades,
        op: str = "gp",
        output_grades=None,
        dtype: Optional[torch.dtype] = None,
        device=None,
        cache: bool = True,
    ):
        """Return a preplanned product executor suitable for ``torch.compile``."""
        if dtype is None:
            dtype = getattr(self, "dtype", torch.float32)
        if device is None:
            device = getattr(self, "device", None)
        return self.planner.product_executor(
            op=op,
            left_grades=left_grades,
            right_grades=right_grades,
            output_grades=output_grades,
            dtype=dtype,
            device=device,
            cache=cache,
        )

    def resolve_layout(
        self,
        *,
        layout: Optional[GradeLayout] = None,
        grades: Optional[Iterable[int]] = None,
        mv=None,
        allow_full: bool = True,
        warn_full: bool = True,
    ) -> GradeLayout:
        """Resolve static layout metadata for tensors or multivectors."""
        return _resolve_layout(
            self,
            layout=layout,
            grades=grades,
            mv=mv,
            allow_full=allow_full,
            warn_full=warn_full,
        )

    def grade_indices(self, grades: Iterable[int], *, device=None) -> torch.Tensor:
        """Return canonical dense basis indices for ``grades``."""
        return _grade_indices(self, grades, device=self.device if device is None else device)

    def hermitian_signs(
        self,
        layout: Optional[GradeLayout] = None,
        *,
        grades: Optional[Iterable[int]] = None,
        device=None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return Hermitian signs for a dense or compact layout."""
        return _hermitian_signs(self, layout=layout, grades=grades, device=device, dtype=dtype)

    def versor_action(self, values: torch.Tensor, weights: torch.Tensor, **kwargs):
        """Apply a parameterized versor action through the host storage dispatcher."""
        return apply_versor_action(self, values, weights, **kwargs)

    def multi_versor_action(self, values: torch.Tensor, weights: torch.Tensor, mix: torch.Tensor, **kwargs):
        """Apply a weighted versor superposition through the host storage dispatcher."""
        return apply_multi_versor_action(self, values, weights, mix, **kwargs)

    def grade_norms(self, values: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return per-grade norms for dense or compact values."""
        return _grade_norms(self, values, **kwargs)

    def projected_product(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        *,
        left_grades=None,
        right_grades=None,
        output_grades=None,
        left_layout=None,
        right_layout=None,
        output_layout=None,
        op: str = "gp",
        left_compact: bool = False,
        right_compact: bool = False,
        compact_output: bool = False,
        return_layout: bool = False,
        pairwise: bool = False,
    ):
        """Compute a declared grade-restricted product through a static executor.

        By default, operands are multiplied elementwise over broadcastable
        prefix dimensions. Set ``pairwise=True`` when the dimension before each
        compact lane axis is an independent left/right item axis.
        """
        left_layout = self._declared_layout(left_grades, left_layout)
        right_layout = self._declared_layout(right_grades, right_layout)
        if not left_compact and left_layout is not None and A.shape[-1] == left_layout.dim:
            left_compact = left_layout.dim != self.dim
        if not right_compact and right_layout is not None and B.shape[-1] == right_layout.dim:
            right_compact = right_layout.dim != self.dim
        if not left_compact:
            check_multivector(A, self, "projected_product(A)")
        if not right_compact:
            check_multivector(B, self, "projected_product(B)")

        request = self.planner.product_request(
            A,
            B,
            left_grades=left_grades,
            right_grades=right_grades,
            output_grades=output_grades,
            left_layout=left_layout,
            right_layout=right_layout,
            output_layout=output_layout,
            op=op,
            left_compact=left_compact,
            right_compact=right_compact,
        )
        executor = self.planner.product_executor_for_request(request)

        if pairwise:
            values = self._execute_pairwise_product(A, B, request, executor)
        else:
            values = self._execute_elementwise_product(A, B, request, executor)
        dispatch = resolve_planned_dispatch(request, compact_output=compact_output)

        if return_layout:
            return values, dispatch.output_storage.layout
        if dispatch.output_storage.is_compact:
            return values
        return materialize_dense(self, values, layout=dispatch.output_storage.layout)

    def projected_geometric_product(self, A: torch.Tensor, B: torch.Tensor, **kwargs):
        """Projected geometric product convenience wrapper."""
        return self.projected_product(A, B, op="gp", **kwargs)

    def projected_wedge(self, A: torch.Tensor, B: torch.Tensor, **kwargs):
        """Projected wedge product convenience wrapper."""
        return self.projected_product(A, B, op="wedge", **kwargs)

    def projected_inner_product(self, A: torch.Tensor, B: torch.Tensor, **kwargs):
        """Projected inner product convenience wrapper."""
        return self.projected_product(A, B, op="inner", **kwargs)

    def projected_commutator(self, A: torch.Tensor, B: torch.Tensor, **kwargs):
        """Projected commutator convenience wrapper."""
        return self.projected_product(A, B, op="commutator", **kwargs)

    def projected_anti_commutator(self, A: torch.Tensor, B: torch.Tensor, **kwargs):
        """Projected anti-commutator convenience wrapper."""
        return self.projected_product(A, B, op="anti_commutator", **kwargs)

    def planned_unary(
        self,
        values: torch.Tensor,
        *,
        op: str,
        input_grades=None,
        output_grades=None,
        input_layout: Optional[GradeLayout] = None,
        output_layout: Optional[GradeLayout] = None,
        input_compact: bool = False,
        compact_output: bool = False,
        return_layout: bool = False,
    ):
        """Execute a unary operation through the shared static grade planner."""
        request = self.planner.unary_request(
            values,
            op=op,
            input_grades=input_grades,
            output_grades=output_grades,
            input_layout=input_layout,
            output_layout=output_layout,
            input_compact=input_compact,
        )
        executor = self.planner.unary_executor_for_request(request)
        output = executor.forward_compact(values) if request.input_compact else executor(values)
        dispatch = resolve_planned_dispatch(request, compact_output=compact_output)

        if return_layout:
            return output, dispatch.output_storage.layout
        if dispatch.output_storage.is_compact:
            return output
        return materialize_dense(self, output, layout=dispatch.output_storage.layout)

    def planned_linear_action(
        self,
        values: torch.Tensor,
        matrix: torch.Tensor,
        *,
        input_grades=None,
        output_grades=None,
        input_layout: Optional[GradeLayout] = None,
        output_layout: Optional[GradeLayout] = None,
        input_compact: bool = False,
        compact_output: bool = False,
        return_layout: bool = False,
    ):
        """Apply a vector-space linear action to compact grade lanes.

        The action is lifted to each active grade through exterior powers of
        ``matrix``. This is the compact execution path for grade-preserving
        versor actions such as rotations and reflections.
        """
        input_layout = self._declared_layout(input_grades, input_layout)
        if input_layout is None:
            if input_compact:
                raise ValueError("input_layout or input_grades is required for compact planned_linear_action")
            input_layout = self.default_layout()

        if output_layout is None:
            output_layout = self.layout(output_grades) if output_grades is not None else input_layout

        if input_compact:
            active_values = values
            if active_values.shape[-1] != input_layout.dim:
                raise ValueError(f"input compact dimension must be {input_layout.dim}, got {active_values.shape[-1]}")
        else:
            check_multivector(values, self, "planned_linear_action(values)")
            active_values = input_layout.compact(values)

        output = apply_graded_linear_action(
            active_values, matrix, input_layout=input_layout, output_layout=output_layout
        )

        if return_layout:
            return output, output_layout
        if compact_output:
            return output
        return materialize_dense(self, output, layout=output_layout)

    def _declared_layout(self, grades, layout):
        if layout is not None:
            return layout
        if grades is not None:
            return self.layout(grades)
        default_grades = getattr(self, "_default_grades", None)
        if default_grades is None:
            return None
        return self.layout(default_grades)

    def _execute_elementwise_product(self, left, right, request, executor):
        if request.left_compact or request.right_compact:
            left_values = left if request.left_compact else executor.left_layout.compact(left)
            right_values = right if request.right_compact else executor.right_layout.compact(right)
            self._check_elementwise_prefix(left_values, right_values)
            return executor.forward_compact(left_values, right_values)

        self._check_elementwise_prefix(left, right)
        return executor(left, right)

    def _execute_pairwise_product(self, left, right, request, executor):
        left_values = left if request.left_compact else executor.left_layout.compact(left)
        right_values = right if request.right_compact else executor.right_layout.compact(right)
        self._check_pairwise_prefix(left_values, right_values)
        return executor.forward_pairwise_compact(left_values, right_values)

    @staticmethod
    def _check_elementwise_prefix(left: torch.Tensor, right: torch.Tensor) -> None:
        try:
            torch.broadcast_shapes(left.shape[:-1], right.shape[:-1])
        except RuntimeError as exc:
            raise ValueError(
                "projected_product elementwise prefixes must be broadcastable; "
                f"got left prefix {tuple(left.shape[:-1])} and right prefix {tuple(right.shape[:-1])}. "
                "Use pairwise=True when left and right have distinct item axes."
            ) from exc

    @staticmethod
    def _check_pairwise_prefix(left: torch.Tensor, right: torch.Tensor) -> None:
        if left.ndim < 2 or right.ndim < 2:
            raise ValueError(
                "pairwise projected_product requires explicit item axes before the compact lane dimension; "
                f"got left shape {tuple(left.shape)} and right shape {tuple(right.shape)}"
            )
        try:
            torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
        except RuntimeError as exc:
            raise ValueError(
                "pairwise projected_product batch prefixes must be broadcastable; "
                f"got left prefix {tuple(left.shape[:-2])} and right prefix {tuple(right.shape[:-2])}"
            ) from exc
