# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from math import comb, prod

import torch
from hypothesis import assume, settings
from hypothesis import strategies as st

from clifra.core._kernel.basis import expand_output_grades

_PBT_SCALE = settings().max_examples / settings.get_profile("standard").max_examples


def _budget(standard_examples: int) -> settings:
    return settings(max_examples=max(1, round(standard_examples * _PBT_SCALE)))


PROPERTY_SETTINGS = _budget(64)
QUICK_PROPERTY_SETTINGS = _budget(32)
CORE_PROPERTY_SETTINGS = _budget(256)
CORE_NUMERIC_SETTINGS = _budget(96)
SIGNATURE_SWEEP_SETTINGS = _budget(8)
DEEP_PROPERTY_SETTINGS = _budget(1024)
DEEP_NUMERIC_SETTINGS = _budget(256)

COEFFICIENTS = st.one_of(
    st.sampled_from((0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0**-12, -(2.0**-12))),
    st.floats(
        min_value=-2.0,
        max_value=2.0,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
        width=32,
    ),
)

PRODUCT_OPS = (
    "geometric_product",
    "wedge",
    "symmetric_product",
    "commutator_product",
    "anti_commutator_product",
    "left_contraction",
    "right_contraction",
)


@lru_cache(maxsize=None)
def small_signatures(
    *, min_n: int = 1, max_n: int = 6, include_degenerate: bool = True
) -> tuple[tuple[int, int, int], ...]:
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
    return tuple(tuple(selection) for size in range(1, len(grades) + 1) for selection in combinations(grades, size))


@st.composite
def signature_strategy(draw, *, min_n: int = 1, max_n: int = 6, include_degenerate: bool = True):
    """Generate signatures compositionally rather than sampling a maintained table."""
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    r = draw(st.integers(min_value=0, max_value=n)) if include_degenerate else 0
    p = draw(st.integers(min_value=0, max_value=n - r))
    return p, n - r - p, r


@st.composite
def signature_for_null_count(draw, r: int, *, max_n: int = 63):
    """Generate every legal non-null split for one fixed null count."""
    r = int(r)
    if not 0 <= r <= max_n:
        raise ValueError(f"r must lie in [0, {max_n}]")
    nonnull = draw(st.integers(min_value=0, max_value=max_n - r))
    p = draw(st.integers(min_value=0, max_value=nonnull))
    return p, nonnull - p, r


def grade_set_strategy(n: int):
    """Generate non-empty grade sets without enumerating their power set."""
    return st.sets(st.integers(min_value=0, max_value=n), min_size=1, max_size=n + 1).map(
        lambda grades: tuple(sorted(grades))
    )


def blade_index_strategy(n: int):
    return st.integers(min_value=0, max_value=(1 << n) - 1)


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
    left_grades = draw(grade_set_strategy(n))
    right_grades = draw(grade_set_strategy(n))
    try:
        output_grades = expand_output_grades(left_grades, right_grades, n, op=op)
    except ValueError:
        assume(False)
    batch = draw(st.integers(min_value=1, max_value=3))
    left_dim = sum(comb(n, grade) for grade in left_grades)
    right_dim = sum(comb(n, grade) for grade in right_grades)
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
    grades = draw(grade_set_strategy(n))
    batch = draw(st.integers(min_value=1, max_value=3))
    dim = sum(comb(n, grade) for grade in grades)
    values = draw(tensor_with_shape((batch, dim)))
    return signature, grades, values
