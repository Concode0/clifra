# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import fsum

import torch

_PRODUCT_OPS = {
    "geometric_product",
    "wedge",
    "symmetric_product",
    "commutator_product",
    "anti_commutator_product",
    "left_contraction",
    "right_contraction",
}


@dataclass(frozen=True)
class SmallCliffordOracle:
    """Independent loop oracle for small Clifford-algebra test cases.

    Basis enumeration, blade reduction, and operation coefficients are defined
    here rather than imported from the implementation under test.
    """

    p: int
    q: int = 0
    r: int = 0

    def __post_init__(self) -> None:
        if min(self.p, self.q, self.r) < 0:
            raise ValueError("signature dimensions must be non-negative")

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
        normalized = tuple(sorted({int(grade) for grade in grades}))
        if not normalized or normalized[0] < 0 or normalized[-1] > self.n:
            raise ValueError(f"grades must be non-empty and lie in [0, {self.n}]")
        indices = []
        for grade in normalized:
            indices.extend(sum(1 << axis for axis in axes) for axes in combinations(range(self.n), grade))
        return tuple(sorted(indices))

    def basis_product(self, left_index: int, right_index: int) -> tuple[int, float]:
        return _basis_product(int(left_index), int(right_index), self.p, self.q, self.r)

    def operation_coefficient(self, left_index: int, right_index: int, op: str = "geometric_product") -> float:
        return _operation_coefficient(int(left_index), int(right_index), self.p, self.q, self.r, op)

    def reverse_sign(self, index: int) -> float:
        axes = _axes(int(index), self.n)
        sign = 1.0
        for left in range(len(axes)):
            for _ in axes[left + 1 :]:
                sign = -sign
        return sign

    def product(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        op: str = "geometric_product",
        left_indices: Iterable[int] | None = None,
        right_indices: Iterable[int] | None = None,
        output_indices: Iterable[int] | None = None,
    ) -> torch.Tensor:
        left_basis = self.full_indices if left_indices is None else tuple(int(index) for index in left_indices)
        right_basis = self.full_indices if right_indices is None else tuple(int(index) for index in right_indices)
        output_basis = self.full_indices if output_indices is None else tuple(int(index) for index in output_indices)
        output_positions = {index: position for position, index in enumerate(output_basis)}
        output_shape = torch.broadcast_shapes(left.shape[:-1], right.shape[:-1])
        dtype = torch.promote_types(left.dtype, right.dtype)
        output = torch.zeros(*output_shape, len(output_basis), dtype=dtype, device=left.device)

        for left_position, right_position, output_position, coefficient in _product_terms(
            self.p,
            self.q,
            self.r,
            str(op),
            left_basis,
            right_basis,
            tuple(output_positions),
        ):
            output[..., output_position] = (
                output[..., output_position] + left[..., left_position] * right[..., right_position] * coefficient
            )
        return output

    def product_fsum(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        op: str = "geometric_product",
        left_indices: Iterable[int] | None = None,
        right_indices: Iterable[int] | None = None,
        output_indices: Iterable[int] | None = None,
    ) -> torch.Tensor:
        """Return a CPU float64 reference using faithfully rounded scalar sums."""
        left_basis = self.full_indices if left_indices is None else tuple(int(index) for index in left_indices)
        right_basis = self.full_indices if right_indices is None else tuple(int(index) for index in right_indices)
        output_basis = self.full_indices if output_indices is None else tuple(int(index) for index in output_indices)
        output_shape = torch.broadcast_shapes(left.shape[:-1], right.shape[:-1])
        left_rows = (
            left.to(device="cpu", dtype=torch.float64)
            .expand(*output_shape, len(left_basis))
            .reshape(-1, len(left_basis))
        )
        right_rows = (
            right.to(device="cpu", dtype=torch.float64)
            .expand(*output_shape, len(right_basis))
            .reshape(-1, len(right_basis))
        )
        terms = _product_terms(self.p, self.q, self.r, str(op), left_basis, right_basis, output_basis)
        by_output = [[] for _ in output_basis]
        for left_position, right_position, output_position, coefficient in terms:
            by_output[output_position].append((left_position, right_position, coefficient))
        rows = [
            [
                fsum(float(left_row[i] * right_row[j]) * coefficient for i, j, coefficient in output_terms)
                for output_terms in by_output
            ]
            for left_row, right_row in zip(left_rows, right_rows)
        ]
        return torch.tensor(rows, dtype=torch.float64).reshape(*output_shape, len(output_basis))

    def project(self, values: torch.Tensor, grades: Iterable[int]) -> torch.Tensor:
        output = torch.zeros(*values.shape[:-1], self.dim, dtype=values.dtype, device=values.device)
        for index in self.indices_for_grades(grades):
            output[..., index] = values[..., index]
        return output

    def signature_norm_squared(self, values: torch.Tensor, indices: Iterable[int] | None = None) -> torch.Tensor:
        basis = self.full_indices if indices is None else tuple(int(index) for index in indices)
        signs = [self.reverse_sign(index) * self.operation_coefficient(index, index) for index in basis]
        sign_tensor = torch.tensor(signs, dtype=values.dtype, device=values.device)
        return (values * values * sign_tensor).sum(dim=-1, keepdim=True)

    def reverse(self, values: torch.Tensor, indices: Iterable[int] | None = None) -> torch.Tensor:
        basis = self.full_indices if indices is None else tuple(int(index) for index in indices)
        signs = torch.tensor([self.reverse_sign(index) for index in basis], dtype=values.dtype, device=values.device)
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
            op="geometric_product",
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
        return self.scalar_product(
            self.clifford_conjugation(left, basis), right, left_indices=basis, right_indices=basis
        )

    def signature_trace_form(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        indices: Iterable[int] | None = None,
    ) -> torch.Tensor:
        basis = self.full_indices if indices is None else tuple(int(index) for index in indices)
        return self.scalar_product(self.reverse(left, basis), right, left_indices=basis, right_indices=basis)

    def pseudoscalar_product(
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
            coefficient = self.operation_coefficient(source_index, pseudoscalar_index)
            output[..., output_position] = values[..., source_position] * coefficient
        return output

    def blade_inverse(self, values: torch.Tensor, indices: Iterable[int] | None = None) -> torch.Tensor:
        denominator = _signed_clamp_min(
            self.signature_norm_squared(values, indices), torch.finfo(values.dtype).eps ** 2
        )
        return self.reverse(values, indices) / denominator

    def blade_project(
        self,
        values: torch.Tensor,
        blade: torch.Tensor,
        *,
        input_indices: Iterable[int],
        blade_indices: Iterable[int],
        output_indices: Iterable[int],
    ) -> torch.Tensor:
        input_basis = tuple(input_indices)
        blade_basis = tuple(blade_indices)
        contracted = self.product(
            values,
            blade,
            op="left_contraction",
            left_indices=input_basis,
            right_indices=blade_basis,
        )
        inverse = self.blade_inverse(blade, blade_basis)
        return self.product(
            contracted,
            inverse,
            right_indices=blade_basis,
            output_indices=output_indices,
        )

    def versor_product(
        self,
        versor: torch.Tensor,
        values: torch.Tensor,
        *,
        versor_indices: Iterable[int],
        input_indices: Iterable[int],
        output_indices: Iterable[int],
    ) -> torch.Tensor:
        versor_basis = tuple(versor_indices)
        left = self.grade_involution(versor, versor_basis)
        inverse = self.blade_inverse(versor, versor_basis)
        middle = self.product(left, values, left_indices=versor_basis, right_indices=input_indices)
        return self.product(middle, inverse, right_indices=versor_basis, output_indices=output_indices)


def _signed_clamp_min(values: torch.Tensor, eps: float) -> torch.Tensor:
    sign = torch.where(values < 0, -values.new_ones(()), values.new_ones(()))
    return sign * values.abs().clamp_min(eps)


def _axes(index: int, n: int) -> list[int]:
    if index < 0 or index >= 1 << n:
        raise ValueError(f"basis index {index} lies outside an algebra of dimension {n}")
    return [axis for axis in range(n) if index & (1 << axis)]


def _basis_product(left_index: int, right_index: int, p: int, q: int, r: int) -> tuple[int, float]:
    n = p + q + r
    word = _axes(left_index, n) + _axes(right_index, n)
    sign = 1.0
    for upper in range(len(word) - 1, 0, -1):
        for position in range(upper):
            if word[position] > word[position + 1]:
                word[position], word[position + 1] = word[position + 1], word[position]
                sign = -sign

    output = 0
    position = 0
    while position < len(word):
        axis = word[position]
        if position + 1 < len(word) and word[position + 1] == axis:
            if axis >= p + q:
                return left_index ^ right_index, 0.0
            if axis >= p:
                sign = -sign
            position += 2
        else:
            output |= 1 << axis
            position += 1
    return output, sign


def _operation_coefficient(left_index: int, right_index: int, p: int, q: int, r: int, op: str) -> float:
    if op not in _PRODUCT_OPS:
        raise ValueError(f"unsupported oracle product {op!r}")
    output_index, left_right = _basis_product(left_index, right_index, p, q, r)
    if op == "geometric_product":
        return left_right
    overlap = left_index & right_index
    if op == "wedge":
        return left_right if overlap == 0 else 0.0

    left_grade = left_index.bit_count()
    right_grade = right_index.bit_count()
    output_grade = output_index.bit_count()
    if op == "left_contraction":
        return left_right if left_grade <= right_grade and output_grade == right_grade - left_grade else 0.0
    if op == "right_contraction":
        return left_right if right_grade <= left_grade and output_grade == left_grade - right_grade else 0.0

    _, right_left = _basis_product(right_index, left_index, p, q, r)
    if op == "symmetric_product":
        return 0.5 * (left_right + right_left)
    if op == "commutator_product":
        return left_right - right_left
    return left_right + right_left


@lru_cache(maxsize=None)
def _product_terms(
    p: int,
    q: int,
    r: int,
    op: str,
    left_basis: tuple[int, ...],
    right_basis: tuple[int, ...],
    output_basis: tuple[int, ...],
) -> tuple[tuple[int, int, int, float], ...]:
    output_positions = {index: position for position, index in enumerate(output_basis)}
    terms = []
    for left_position, left_index in enumerate(left_basis):
        for right_position, right_index in enumerate(right_basis):
            coefficient = _operation_coefficient(left_index, right_index, p, q, r, op)
            output_position = output_positions.get(left_index ^ right_index)
            if coefficient and output_position is not None:
                terms.append((left_position, right_position, output_position, coefficient))
    return tuple(terms)
