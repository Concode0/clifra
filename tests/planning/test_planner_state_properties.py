# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from clifra.core.algebra import AlgebraContext
from tests.helpers.hypothesis_cases import QUICK_PROPERTY_SETTINGS, signature_strategy

pytestmark = [pytest.mark.unit, pytest.mark.property]

_CACHE_NAMES = (
    "_product_executors",
    "_unary_executors",
    "_signature_norm_squared_executors",
    "_pseudoscalar_product_executors",
    "_bivector_exp_executors",
    "_full_sandwich_action_executors",
    "_versor_action_plans",
    "_paired_bivector_action_plans",
    "_bivector_signs_cache",
)
_ACTIONS = (
    "product_geometric_product",
    "product_wedge",
    "unary_reverse",
    "unary_conjugate",
    "signature_norm",
    "pseudoscalar",
    "bivector_exp",
    "versor_action",
    "paired_action",
    "clear",
)


def _cache_snapshot(planner):
    return tuple(tuple(getattr(planner, name).items()) for name in _CACHE_NAMES)


def _plan(algebra, action: str):
    vector = algebra.layout((1,))
    bivector = algebra.layout((2,))
    if action == "product_geometric_product":
        return algebra.plan_product(
            op="geometric_product", left=vector, right=vector, output=algebra.layout((0, 2))
        )._kernel
    if action == "product_wedge":
        return algebra.plan_product(op="wedge", left=vector, right=vector, output=bivector)._kernel
    if action == "unary_reverse":
        return algebra.plan_unary(op="reverse", input=algebra.layout((0, 1, 2)))._kernel
    if action == "unary_conjugate":
        return algebra.plan_unary(op="clifford_conjugation", input=algebra.layout((0, 1, 2)))._kernel
    if action == "signature_norm":
        return algebra.plan_signature_norm_squared(input=algebra.layout((0, 1, 2)))._kernel
    if action == "pseudoscalar":
        return algebra.plan_pseudoscalar_product(input=vector, output=algebra.layout((algebra.n - 1,)))._kernel
    if action == "bivector_exp":
        return algebra.plan_bivector_exp(input=bivector, output=algebra.layout(range(0, algebra.n + 1, 2)))._kernel
    if action == "versor_action":
        return algebra._planner.versor_action_plan(
            grade=2,
            input_layout=vector,
            output_layout=vector,
            parameter_layout=bivector,
        )
    return algebra._planner.paired_bivector_action_plan(
        input_layout=vector,
        output_layout=vector,
        parameter_layout=bivector,
    )


@QUICK_PROPERTY_SETTINGS
@given(
    signature=signature_strategy(min_n=2, max_n=5),
    actions=st.lists(st.sampled_from(_ACTIONS), min_size=1, max_size=24),
)
def test_generated_planner_sequences_preserve_cache_identity_and_validation(signature, actions):
    algebra = AlgebraContext(*signature, device="cpu", dtype=torch.float32)
    planned = {}

    for action in actions:
        if action == "clear":
            algebra._planner.clear_cache()
            planned.clear()
            assert all(not getattr(algebra._planner, name) for name in _CACHE_NAMES)
            continue

        result = _plan(algebra, action)
        if action in planned:
            assert result is planned[action]
        else:
            planned[action] = result

    before = _cache_snapshot(algebra._planner)
    n = sum(signature)
    foreign_signature = (0, n, 0) if signature == (n, 0, 0) else (n, 0, 0)
    foreign = AlgebraContext(*foreign_signature, device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError, match="signature.*does not match"):
        algebra.plan_product(left=foreign.layout((1,)), right=foreign.layout((1,)), output=foreign.layout((0, 2)))
    assert _cache_snapshot(algebra._planner) == before
