# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from math import prod

import torch
from hypothesis import HealthCheck, assume, settings
from hypothesis import strategies as st

from clifra.core.foundation.basis import expand_output_grades

PROPERTY_SETTINGS = settings(
    max_examples=64,
    deadline=None,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
)

QUICK_PROPERTY_SETTINGS = settings(
    max_examples=32,
    deadline=None,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
)

COEFFICIENTS = st.floats(
    min_value=-2.0,
    max_value=2.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    width=32,
)

PRODUCT_OPS = (
    "gp",
    "wedge",
    "inner",
    "commutator",
    "anti_commutator",
    "left_contraction",
    "right_contraction",
)


@lru_cache(maxsize=None)
def small_signatures(*, min_n: int = 1, max_n: int = 6, include_degenerate: bool = True) -> tuple[tuple[int, int, int], ...]:
    signatures: list[tuple[int, int, int]] = []
    for n in range(min_n, max_n + 1):
        for p in range(n + 1):
            for q in range(n - p + 1):
                r = n - p - q
                if include_degenerate or r == 0:
                    signatures.append((p, q, r))
    return tuple(signatures)


@lru_cache(maxsize=None)
def grade_sets(n: int) -> tuple[tuple[int, ...], ...]:
    grades = tuple(range(n + 1))
    return tuple(
        tuple(selection)
        for size in range(1, len(grades) + 1)
        for selection in combinations(grades, size)
    )


def signature_strategy(*, min_n: int = 1, max_n: int = 6, include_degenerate: bool = True):
    return st.sampled_from(
        small_signatures(min_n=min_n, max_n=max_n, include_degenerate=include_degenerate)
    )


def tensor_with_shape(shape: tuple[int, ...], *, dtype: torch.dtype = torch.float64):
    size = prod(shape)
    return st.lists(COEFFICIENTS, min_size=size, max_size=size).map(
        lambda values: torch.tensor(values, dtype=dtype).reshape(shape)
    )


@st.composite
def full_product_cases(draw, *, include_degenerate: bool = True, max_n: int = 6):
    signature = draw(signature_strategy(max_n=max_n, include_degenerate=include_degenerate))
    op = draw(st.sampled_from(PRODUCT_OPS))
    batch = draw(st.integers(min_value=1, max_value=3))
    dim = 1 << sum(signature)
    left = draw(tensor_with_shape((batch, dim)))
    right = draw(tensor_with_shape((batch, dim)))
    return signature, op, left, right


@st.composite
def compact_product_cases(draw, *, include_degenerate: bool = True, max_n: int = 6):
    signature = draw(signature_strategy(max_n=max_n, include_degenerate=include_degenerate))
    p, q, r = signature
    n = p + q + r
    op = draw(st.sampled_from(PRODUCT_OPS))
    left_grades = draw(st.sampled_from(grade_sets(n)))
    right_grades = draw(st.sampled_from(grade_sets(n)))
    try:
        output_grades = expand_output_grades(left_grades, right_grades, n, op=op)
    except ValueError:
        assume(False)
    batch = draw(st.integers(min_value=1, max_value=3))
    left_dim = sum(1 for index in range(1 << n) if index.bit_count() in left_grades)
    right_dim = sum(1 for index in range(1 << n) if index.bit_count() in right_grades)
    left = draw(tensor_with_shape((batch, left_dim)))
    right = draw(tensor_with_shape((batch, right_dim)))
    return signature, op, left_grades, right_grades, output_grades, left, right


@st.composite
def full_multivector_cases(draw, *, min_n: int = 1, max_n: int = 6, include_degenerate: bool = True):
    signature = draw(signature_strategy(min_n=min_n, max_n=max_n, include_degenerate=include_degenerate))
    batch = draw(st.integers(min_value=1, max_value=3))
    dim = 1 << sum(signature)
    values = draw(tensor_with_shape((batch, dim)))
    return signature, values


@st.composite
def compact_multivector_cases(draw, *, min_n: int = 1, max_n: int = 6, include_degenerate: bool = True):
    signature = draw(signature_strategy(min_n=min_n, max_n=max_n, include_degenerate=include_degenerate))
    n = sum(signature)
    grades = draw(st.sampled_from(grade_sets(n)))
    batch = draw(st.integers(min_value=1, max_value=3))
    dim = sum(1 for index in range(1 << n) if index.bit_count() in grades)
    values = draw(tensor_with_shape((batch, dim)))
    return signature, grades, values
