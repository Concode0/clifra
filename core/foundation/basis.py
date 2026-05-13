# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Canonical bitmask-basis utilities for Clifford algebra planning."""

from __future__ import annotations

from typing import Iterable, Literal, Optional

import torch

GradeProductOp = Literal["gp", "wedge", "inner", "commutator", "anti_commutator"]


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
    grade_set = set(normalize_grades(grades, n))
    return tuple(index for index in range(1 << n) if index.bit_count() in grade_set)


def basis_indices_for_grades(n: int, grades: Iterable[int], *, device=None) -> torch.Tensor:
    """Return canonical bitmask basis indices as a tensor."""
    return torch.tensor(basis_index_tuple_for_grades(n, grades), dtype=torch.long, device=device)


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
    left = normalize_grades(left_grades, n, name="left_grades")
    right = normalize_grades(right_grades, n, name="right_grades")
    if op not in {"gp", "wedge", "inner", "commutator", "anti_commutator"}:
        raise ValueError(f"Unsupported grade product op {op!r}")

    outputs: set[int] = set()
    for left_grade in left:
        for right_grade in right:
            if op == "wedge":
                grade = left_grade + right_grade
                if grade <= n:
                    outputs.add(grade)
            else:
                outputs.update(geometric_product_output_grades(left_grade, right_grade, n))

    if project_grades is not None:
        outputs &= set(normalize_grades(project_grades, n, name="project_grades"))
    if not outputs:
        raise ValueError(
            f"Grade expansion is empty for op={op!r}, left_grades={left}, right_grades={right}, "
            f"project_grades={None if project_grades is None else tuple(project_grades)}"
        )
    return tuple(sorted(outputs))


def basis_product(index_a: int, index_b: int, p: int, q: int, r: int) -> tuple[int, float]:
    """Return ``(index, sign)`` for two canonical basis blade products."""
    n = p + q + r
    swap_count = 0
    for bit in range(n):
        if index_a & (1 << bit):
            swap_count += (index_b & ((1 << bit) - 1)).bit_count()

    sign = -1.0 if swap_count % 2 else 1.0

    negative_mask = sum(1 << bit for bit in range(p, p + q))
    if ((index_a & index_b & negative_mask).bit_count() % 2) == 1:
        sign = -sign

    null_mask = sum(1 << bit for bit in range(p + q, n))
    if (index_a & index_b & null_mask) != 0:
        sign = 0.0

    return index_a ^ index_b, sign


def reverse_sign(index: int) -> float:
    """Return the reversion sign for a canonical basis blade."""
    grade = int(index).bit_count()
    return -1.0 if ((grade * (grade - 1) // 2) % 2) else 1.0


def operation_coefficient(index_a: int, index_b: int, p: int, q: int, r: int, op: GradeProductOp) -> float:
    """Return the scalar coefficient multiplying ``A_i * B_j`` for ``op``."""
    _, sign_ab = basis_product(index_a, index_b, p, q, r)
    if op == "gp":
        return sign_ab

    _, sign_ba = basis_product(index_b, index_a, p, q, r)
    if op == "wedge":
        return 0.5 * (sign_ab - sign_ba)
    if op == "inner":
        return 0.5 * (sign_ab + sign_ba)
    if op == "commutator":
        return sign_ab - sign_ba
    if op == "anti_commutator":
        return sign_ab + sign_ba
    raise ValueError(f"Unsupported grade product op {op!r}")
