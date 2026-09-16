"""Small helpers for exercising real planned action and product paths."""

import torch

from clifra.core import AlgebraContext


def _planned_graded_action(input_layout, output_layout, *, dtype=torch.float32, device="cpu"):
    spec = input_layout.spec
    algebra = AlgebraContext(spec.p, spec.q, spec.r, dtype=dtype, device=device)
    return algebra.plan_linear_action(input=input_layout, output=output_layout)._kernel


def _planned_full_sandwich(layout, *, dtype=torch.float32, device="cpu"):
    from clifra.core._kernel.providers import action_execution_request
    from tests.helpers.policy import PreferRoute

    spec = layout.spec
    algebra = AlgebraContext(spec.p, spec.q, spec.r, dtype=dtype, device=device)
    request = action_execution_request(
        algebra, "sandwich", left_layout=layout, input_layout=layout, right_layout=layout
    )
    selection = algebra._planner.router.select(
        request,
        PreferRoute("action", "full_action_matrix"),
        algebra._planner.limits,
    )
    return selection.build()


def select_product_route(algebra, *, op, left_layout, right_layout, output_layout, dtype, device, policy=None):
    from clifra.core._kernel.planning.layouts import ProductRequest
    from clifra.core._kernel.providers import product_execution_request

    request = ProductRequest.compact(
        algebra.spec,
        op=op,
        left_layout=left_layout,
        right_layout=right_layout,
        output_layout=output_layout,
        dtype=dtype,
        device=device,
    )
    return algebra._planner.router.select(
        product_execution_request(request),
        algebra._planner.policy if policy is None else policy,
        algebra._planner.limits,
    )
