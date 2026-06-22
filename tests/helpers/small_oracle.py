# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from clifra.core.foundation.basis import basis_index_tuple_for_grades, operation_coefficient, reverse_sign


@dataclass(frozen=True)
class SmallCliffordOracle:
    """Independent loop oracle for small Clifford-algebra test cases.

    The oracle depends only on the scalar basis rules in ``foundation.basis``.
    It deliberately avoids planner/executor code.
    """

    p: int
    q: int = 0
    r: int = 0

    @property
    def n(self) -> int:
        return self.p + self.q + self.r

    @property
    def dim(self) -> int:
        return 1 << self.n

    @property
    def full_indices(self) -> tuple[int, ...]:
        return tuple(range(self.dim))

    def indices_for_grades(self, grades: Iterable[int]) -> tuple[int, ...]:
        return basis_index_tuple_for_grades(self.n, grades)

    def product(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        op: str = "gp",
        left_indices: Iterable[int] | None = None,
        right_indices: Iterable[int] | None = None,
        output_indices: Iterable[int] | None = None,
    ) -> torch.Tensor:
        left_basis = self.full_indices if left_indices is None else tuple(int(index) for index in left_indices)
        right_basis = self.full_indices if right_indices is None else tuple(int(index) for index in right_indices)
        output_basis = self.full_indices if output_indices is None else tuple(int(index) for index in output_indices)
        output_positions = {index: position for position, index in enumerate(output_basis)}
        output_shape = torch.broadcast_shapes(left.shape[:-1], right.shape[:-1])
        output = torch.zeros(*output_shape, len(output_basis), dtype=torch.promote_types(left.dtype, right.dtype), device=left.device)

        for left_position, left_index in enumerate(left_basis):
            left_value = left[..., left_position]
            for right_position, right_index in enumerate(right_basis):
                coefficient = operation_coefficient(left_index, right_index, self.p, self.q, self.r, op)
                if coefficient == 0.0:
                    continue
                output_position = output_positions.get(left_index ^ right_index)
                if output_position is None:
                    continue
                output[..., output_position] = output[..., output_position] + left_value * right[..., right_position] * coefficient
        return output

    def project(self, values: torch.Tensor, grades: Iterable[int]) -> torch.Tensor:
        output = torch.zeros(*values.shape[:-1], self.dim, dtype=values.dtype, device=values.device)
        for index in self.indices_for_grades(grades):
            output[..., index] = values[..., index]
        return output

    def norm_sq(self, values: torch.Tensor, indices: Iterable[int] | None = None) -> torch.Tensor:
        basis = self.full_indices if indices is None else tuple(int(index) for index in indices)
        signs = [
            reverse_sign(index) * operation_coefficient(index, index, self.p, self.q, self.r, "gp")
            for index in basis
        ]
        sign_tensor = torch.tensor(signs, dtype=values.dtype, device=values.device)
        return (values * values * sign_tensor).sum(dim=-1, keepdim=True)

    def reverse(self, values: torch.Tensor, indices: Iterable[int] | None = None) -> torch.Tensor:
        basis = self.full_indices if indices is None else tuple(int(index) for index in indices)
        signs = torch.tensor([reverse_sign(index) for index in basis], dtype=values.dtype, device=values.device)
        return values * signs

    def grade_involution(self, values: torch.Tensor, indices: Iterable[int] | None = None) -> torch.Tensor:
        basis = self.full_indices if indices is None else tuple(int(index) for index in indices)
        signs = torch.tensor(
            [-1.0 if int(index).bit_count() % 2 else 1.0 for index in basis],
            dtype=values.dtype,
            device=values.device,
        )
        return values * signs

    def clifford_conjugation(self, values: torch.Tensor, indices: Iterable[int] | None = None) -> torch.Tensor:
        return self.grade_involution(self.reverse(values, indices), indices)

    def scalar_product(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        left_indices: Iterable[int] | None = None,
        right_indices: Iterable[int] | None = None,
    ) -> torch.Tensor:
        return self.product(
            left,
            right,
            op="gp",
            left_indices=left_indices,
            right_indices=right_indices,
            output_indices=(0,),
        )

    def conjugate_scalar_form(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        indices: Iterable[int] | None = None,
    ) -> torch.Tensor:
        basis = self.full_indices if indices is None else tuple(int(index) for index in indices)
        return self.scalar_product(self.clifford_conjugation(left, basis), right, left_indices=basis, right_indices=basis)

    def signature_trace_form(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        indices: Iterable[int] | None = None,
    ) -> torch.Tensor:
        basis = self.full_indices if indices is None else tuple(int(index) for index in indices)
        return self.scalar_product(self.reverse(left, basis), right, left_indices=basis, right_indices=basis)

    def dual(
        self,
        values: torch.Tensor,
        *,
        input_indices: Iterable[int] | None = None,
        output_indices: Iterable[int] | None = None,
    ) -> torch.Tensor:
        input_basis = self.full_indices if input_indices is None else tuple(int(index) for index in input_indices)
        output_basis = self.full_indices if output_indices is None else tuple(int(index) for index in output_indices)
        input_positions = {index: position for position, index in enumerate(input_basis)}
        pseudoscalar_index = self.dim - 1
        output = torch.zeros(*values.shape[:-1], len(output_basis), dtype=values.dtype, device=values.device)
        for output_position, output_index in enumerate(output_basis):
            source_index = output_index ^ pseudoscalar_index
            source_position = input_positions[source_index]
            coefficient = operation_coefficient(source_index, pseudoscalar_index, self.p, self.q, self.r, "gp")
            output[..., output_position] = values[..., source_position] * coefficient
        return output

    def blade_inverse(self, values: torch.Tensor, indices: Iterable[int] | None = None) -> torch.Tensor:
        denominator = _signed_clamp_min(self.norm_sq(values, indices), torch.finfo(values.dtype).eps**2)
        return self.reverse(values, indices) / denominator


def _signed_clamp_min(values: torch.Tensor, eps: float) -> torch.Tensor:
    sign = torch.where(values < 0, -values.new_ones(()), values.new_ones(()))
    return sign * values.abs().clamp_min(eps)
