# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Grade-aware translator from algebraic intent to static executors."""

from __future__ import annotations

import torch

from core.foundation.layout import AlgebraSpec, GradeLayout
from core.planning.grade_plan import GradeProductExecutor, build_grade_product_plan_from_request
from core.planning.request import ProductRequest, build_product_request, normalize_product_op
from core.planning.tree import build_grade_plan_tree
from core.planning.unary import (
    GradeUnaryExecutor,
    UnaryRequest,
    build_unary_plan_from_request,
    build_unary_request,
    normalize_unary_op,
)


class GradeTranslator:
    """Owns layout and product-plan lowering for one algebra instance.

    The translator is deliberately not an ``nn.Module``. It builds static
    executor modules keyed by signature, grades, dtype, and device, while the
    algebra remains the source of truth for buffers and dense reference paths.
    """

    def __init__(self, algebra):
        self.algebra = algebra
        self.spec = AlgebraSpec.from_algebra(algebra)
        self._product_executors = {}
        self._unary_executors = {}

    def layout(self, grades):
        """Return the compact layout for ``grades``."""
        return self.spec.layout(grades)

    def full_layout(self) -> GradeLayout:
        """Return the full dense basis layout."""
        return self.layout(range(self.spec.n + 1))

    def clear_cache(self) -> None:
        """Drop cached executor modules."""
        self._product_executors.clear()
        self._unary_executors.clear()

    def _apply(self, fn):
        """Apply a PyTorch module-style transform to cached executor buffers."""
        for executor in self._product_executors.values():
            executor._apply(fn)
        for executor in self._unary_executors.values():
            executor._apply(fn)
        return self

    def product_executor(
        self,
        *,
        op: str,
        left_grades,
        right_grades,
        output_grades,
        dtype,
        device,
        cache: bool = True,
    ):
        """Return a cached static executor for a projected bilinear product."""
        request = ProductRequest(
            spec=self.spec,
            op=normalize_product_op(op),
            left_layout=self.layout(left_grades),
            right_layout=self.layout(right_grades),
            output_layout=self.layout(output_grades),
            left_compact=False,
            right_compact=False,
            dtype=dtype,
            device=torch.device(device),
        )
        return self.executor_for_request(request, cache=cache)

    def executor_for_request(self, request: ProductRequest, *, cache: bool = True) -> GradeProductExecutor:
        """Return an executor for an already normalized product request."""
        key = request.cache_key
        executor = self._product_executors.get(key) if cache else None
        if executor is None:
            plan = build_grade_product_plan_from_request(request)
            executor = GradeProductExecutor(plan)
            if cache:
                self._product_executors[key] = executor
        return executor

    def product_tree(self, *, op: str, left_grades, right_grades, output_grades=None):
        """Return planner-only grade tree metadata for a product route."""
        return build_grade_plan_tree(
            self.spec,
            op=op,
            left_grades=left_grades,
            right_grades=right_grades,
            output_grades=output_grades,
        )

    def unary_executor(
        self,
        *,
        op: str,
        input_grades,
        output_grades=None,
        dtype,
        device,
        cache: bool = True,
    ) -> GradeUnaryExecutor:
        """Return a cached static executor for a unary operation."""
        op = normalize_unary_op(op)
        if op == "grade_projection" and output_grades is None:
            raise ValueError("output_grades is required for grade_projection")
        input_layout = self.layout(input_grades)
        output_layout = input_layout if output_grades is None else self.layout(output_grades)
        request = UnaryRequest(
            spec=self.spec,
            op=op,
            input_layout=input_layout,
            output_layout=output_layout,
            input_compact=False,
            dtype=dtype,
            device=torch.device(device),
        )
        return self.unary_executor_for_request(request, cache=cache)

    def unary_executor_for_request(self, request: UnaryRequest, *, cache: bool = True) -> GradeUnaryExecutor:
        """Return an executor for an already normalized unary request."""
        key = request.cache_key
        executor = self._unary_executors.get(key) if cache else None
        if executor is None:
            plan = build_unary_plan_from_request(request)
            executor = GradeUnaryExecutor(plan)
            if cache:
                self._unary_executors[key] = executor
        return executor

    def planned_unary(
        self,
        values: torch.Tensor,
        *,
        op: str,
        input_grades=None,
        output_grades=None,
        input_layout: GradeLayout = None,
        output_layout: GradeLayout = None,
        input_compact: bool = False,
        compact_output: bool = False,
        return_layout: bool = False,
    ):
        """Execute a unary operation using a static gather/sign plan."""
        request = build_unary_request(
            self.spec,
            values,
            op=op,
            input_grades=input_grades,
            output_grades=output_grades,
            input_layout=input_layout,
            output_layout=output_layout,
            input_compact=input_compact,
            full_layout_allowed=self._full_layout_allowed(),
        )
        executor = self.unary_executor_for_request(request)
        output = executor.forward_compact(values) if request.input_compact else executor(values)

        if return_layout:
            return output, executor.output_layout
        if compact_output:
            return output
        return executor.output_layout.dense(output)

    def projected_product(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        *,
        left_grades=None,
        right_grades=None,
        output_grades=None,
        left_layout: GradeLayout = None,
        right_layout: GradeLayout = None,
        output_layout: GradeLayout = None,
        op: str = "gp",
        left_compact: bool = False,
        right_compact: bool = False,
        compact_output: bool = False,
        return_layout: bool = False,
    ):
        """Execute a projected product using dense or compact input lanes."""
        request = build_product_request(
            self.spec,
            A,
            B,
            op=op,
            left_grades=left_grades,
            right_grades=right_grades,
            output_grades=output_grades,
            left_layout=left_layout,
            right_layout=right_layout,
            output_layout=output_layout,
            left_compact=left_compact,
            right_compact=right_compact,
            full_layout_allowed=self._full_layout_allowed(),
        )
        executor = self.executor_for_request(request)

        if request.left_compact or request.right_compact:
            A_values = A if request.left_compact else executor.left_layout.compact(A)
            B_values = B if request.right_compact else executor.right_layout.compact(B)
            values = executor.forward_compact(A_values, B_values)
        else:
            values = executor(A, B)

        if return_layout:
            return values, executor.output_layout
        if compact_output:
            return values
        return executor.output_layout.dense(values)

    def _full_layout_allowed(self) -> bool:
        return bool(getattr(self.algebra, "allow_full_layout_products", True))
