# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Shared projected-product facade for algebra hosts."""

from __future__ import annotations

import torch

from core.foundation.validation import check_multivector


class ProjectedProductMixin:
    """Route declared grade products through an algebra's grade translator."""

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
    ):
        """Compute a declared grade-restricted product through the translator."""
        if not left_compact and left_layout is not None and A.shape[-1] == left_layout.dim:
            left_compact = left_layout.dim != self.dim
        if not right_compact and right_layout is not None and B.shape[-1] == right_layout.dim:
            right_compact = right_layout.dim != self.dim
        if not left_compact:
            check_multivector(A, self, "projected_product(A)")
        if not right_compact:
            check_multivector(B, self, "projected_product(B)")
        return self.translator.projected_product(
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
            compact_output=compact_output,
            return_layout=return_layout,
        )

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
