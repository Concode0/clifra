# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Canonical bitmask-basis utilities for Clifford algebra planning."""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Iterable, Literal, Optional, get_args

import torch

GradeProductOp = Literal[
    "gp",
    "wedge",
    "symmetric_product",
    "commutator_product",
    "anti_commutator_product",
    "left_contraction",
    "right_contraction",
]
_PUBLIC_GRADE_PRODUCT_OPS = {
    "geometric_product": "gp",
}
_GRADE_PRODUCT_OPS = set(get_args(GradeProductOp))

# NOTE: Torch-backed executors currently store canonical basis blades as signed
# int64 bitmasks. That makes n=63 the largest supported dimension: the highest
# usable basis bit is 1 << 62, while n=64 would require 1 << 63, which is outside
# torch.long's positive range. Supporting n>=64 requires kernel-level storage
# engineering, for example compact-position-only kernels, declared blade objects,
# multi-limb or variable-length bitsets, or another non-int64 basis identifier.
TORCH_LONG_BASIS_MAX_N = 63
_TORCH_LONG_MAX = (1 << TORCH_LONG_BASIS_MAX_N) - 1


def normalize_grade_product_op(op: str) -> GradeProductOp:
    """Return the internal grade-product key for a public product operation name."""
    normalized = str(op).lower()
    if normalized in _GRADE_PRODUCT_OPS:
        return normalized
    try:
        return _PUBLIC_GRADE_PRODUCT_OPS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_GRADE_PRODUCT_OPS | set(_PUBLIC_GRADE_PRODUCT_OPS)))
        raise ValueError(f"Unsupported grade product op {op!r}. Supported ops: {supported}") from exc


def normalize_grades(grades: Iterable[int], n: int, *, name: str = "grades") -> tuple[int, ...]:
    """Return sorted unique grades validated against ``0 <= grade <= n``."""
    normalized = tuple(sorted({int(grade) for grade in grades}))
    if not normalized:
        raise ValueError(f"{name} must contain at least one grade")
    invalid = [grade for grade in normalized if grade < 0 or grade > n]
    if invalid:
        raise ValueError(f"{name} contains invalid grades for n={n}: {invalid}")
    return normalized


def basis_index_tuple_for_grades(n: int, grades: Iterable[int]) -> tuple[int, ...]:
    """Return canonical bitmask basis indices whose popcount is in ``grades``."""
    indices: list[int] = []
    for grade in normalize_grades(grades, n):
        indices.extend(_basis_indices_for_grade(n, grade))
    return tuple(sorted(indices))


def basis_count_for_grades(n: int, grades: Iterable[int]) -> int:
    """Return the number of basis blades represented by ``grades``."""
    return sum(comb(n, grade) for grade in normalize_grades(grades, n))


def basis_indices_for_grades(n: int, grades: Iterable[int], *, device=None) -> torch.Tensor:
    """Return canonical bitmask basis indices as a tensor."""
    return basis_indices_tensor(basis_index_tuple_for_grades(n, grades), n=n, device=device)


def basis_indices_tensor(
    indices: Iterable[int],
    *,
    n: Optional[int] = None,
    role: str = "basis indices",
    device=None,
) -> torch.Tensor:
    """Tensorize canonical basis bitmasks with a clear signed-int64 boundary."""
    values = tuple(int(index) for index in indices)
    _validate_torch_long_basis_indices(values, n=n, role=role)
    return torch.tensor(values, dtype=torch.long, device=device)


def _basis_indices_for_grade(n: int, grade: int) -> tuple[int, ...]:
    if grade == 0:
        return (0,)
    if grade == n:
        return ((1 << n) - 1,)
    return tuple(sum(1 << bit for bit in bits) for bits in combinations(range(n), grade))


def _validate_torch_long_basis_indices(indices: tuple[int, ...], *, n: Optional[int], role: str) -> None:
    if not indices:
        return
    if max(indices) <= _TORCH_LONG_MAX:
        return
    dimension = "" if n is None else f" for n={n}"
    raise ValueError(
        f"{role}{dimension} cannot be represented as torch.long basis bitmasks. "
        f"Current Torch-backed executors support bitmask tensorization up to n={TORCH_LONG_BASIS_MAX_N}."
    )


def geometric_product_output_grades(left_grade: int, right_grade: int, n: int) -> tuple[int, ...]:
    """Return the possible output grades of a homogeneous geometric product."""
    low = abs(int(left_grade) - int(right_grade))
    high = min(int(left_grade) + int(right_grade), 2 * n - int(left_grade) - int(right_grade))
    return tuple(range(low, high + 1, 2))


def expand_output_grades(
    left_grades: Iterable[int],
    right_grades: Iterable[int],
    n: int,
    *,
    op: GradeProductOp = "gp",
    project_grades: Optional[Iterable[int]] = None,
) -> tuple[int, ...]:
    """Expand input grade sets into output grades required by ``op``."""
    op = normalize_grade_product_op(op)
    left = normalize_grades(left_grades, n, name="left_grades")
    right = normalize_grades(right_grades, n, name="right_grades")

    outputs: set[int] = set()
    for left_grade in left:
        for right_grade in right:
            outputs.update(product_output_grades(left_grade, right_grade, n, op=op))

    if project_grades is not None:
        outputs &= set(normalize_grades(project_grades, n, name="project_grades"))
    if not outputs:
        raise ValueError(
            f"Grade expansion is empty for op={op!r}, left_grades={left}, right_grades={right}, "
            f"project_grades={None if project_grades is None else tuple(project_grades)}"
        )
    return tuple(sorted(outputs))


def product_output_grades(
    left_grade: int,
    right_grade: int,
    n: int,
    *,
    op: GradeProductOp = "gp",
) -> tuple[int, ...]:
    """Return possible output grades for one homogeneous product route."""
    op = normalize_grade_product_op(op)
    left_grade = int(left_grade)
    right_grade = int(right_grade)
    if op == "left_contraction":
        return (right_grade - left_grade,) if left_grade <= right_grade else ()
    if op == "right_contraction":
        return (left_grade - right_grade,) if left_grade >= right_grade else ()
    if op == "wedge":
        grade = left_grade + right_grade
        return (grade,) if grade <= n else ()

    outputs = geometric_product_output_grades(left_grade, right_grade, n)
    if op == "gp":
        return outputs
    if op not in {"symmetric_product", "commutator_product", "anti_commutator_product"}:
        raise ValueError(f"Unsupported grade product op {op!r}")

    filtered = []
    for output_grade in outputs:
        overlap = (left_grade + right_grade - output_grade) // 2
        parity_odd = ((left_grade * right_grade - overlap) % 2) == 1
        if op == "commutator_product":
            keep = parity_odd
        else:
            keep = not parity_odd
        if keep:
            filtered.append(output_grade)
    return tuple(filtered)


def basis_product(index_a: int, index_b: int, p: int, q: int, r: int) -> tuple[int, float]:
    """Return ``(index, sign)`` for two canonical basis blade products."""
    n = p + q + r
    swap_count = 0
    for bit in range(n):
        if index_a & (1 << bit):
            swap_count += (index_b & ((1 << bit) - 1)).bit_count()

    sign = -1.0 if swap_count % 2 else 1.0

    negative_mask = ((1 << q) - 1) << p
    if ((index_a & index_b & negative_mask).bit_count() % 2) == 1:
        sign = -sign

    null_mask = ((1 << r) - 1) << (p + q)
    if (index_a & index_b & null_mask) != 0:
        sign = 0.0

    return index_a ^ index_b, sign


def reverse_sign(index: int) -> float:
    """Return the reversion sign for a canonical basis blade."""
    grade = int(index).bit_count()
    return -1.0 if ((grade * (grade - 1) // 2) % 2) else 1.0


def operation_coefficient(
    index_a: int,
    index_b: int,
    p: int,
    q: int,
    r: int,
    op: GradeProductOp,
) -> float:
    """Return the scalar coefficient multiplying ``A_i * B_j`` for ``op``."""
    op = normalize_grade_product_op(op)
    if not operation_may_be_nonzero(index_a, index_b, p, q, r, op):
        return 0.0

    _, sign_ab = basis_product(index_a, index_b, p, q, r)
    if op in {"gp", "wedge", "symmetric_product", "left_contraction", "right_contraction"}:
        return sign_ab

    if op == "commutator_product":
        return 2.0 * sign_ab
    if op == "anti_commutator_product":
        return 2.0 * sign_ab
    raise ValueError(f"Unsupported grade product op {op!r}")


def operation_may_be_nonzero(
    index_a: int,
    index_b: int,
    p: int,
    q: int,
    r: int,
    op: GradeProductOp,
) -> bool:
    """Return whether an operator can have a non-zero coefficient for a basis pair."""
    op = normalize_grade_product_op(op)
    overlap_mask = int(index_a) & int(index_b)
    if op == "wedge":
        return overlap_mask == 0

    if r > 0:
        null_mask = ((1 << r) - 1) << (p + q)
        if overlap_mask & null_mask:
            return False
    if op == "left_contraction":
        left_grade = int(index_a).bit_count()
        right_grade = int(index_b).bit_count()
        output_grade = (int(index_a) ^ int(index_b)).bit_count()
        return left_grade <= right_grade and output_grade == right_grade - left_grade
    if op == "right_contraction":
        left_grade = int(index_a).bit_count()
        right_grade = int(index_b).bit_count()
        output_grade = (int(index_a) ^ int(index_b)).bit_count()
        return left_grade >= right_grade and output_grade == left_grade - right_grade
    if op == "gp":
        return True

    parity_odd = _swap_parity_between_orders(index_a, index_b) == 1
    if op == "commutator_product":
        return parity_odd
    if op in {"symmetric_product", "anti_commutator_product"}:
        return not parity_odd
    raise ValueError(f"Unsupported grade product op {op!r}")


def _swap_parity_between_orders(index_a: int, index_b: int) -> int:
    left_grade = int(index_a).bit_count()
    right_grade = int(index_b).bit_count()
    overlap = (int(index_a) & int(index_b)).bit_count()
    return (left_grade * right_grade - overlap) % 2
