"""Test-only compositions for historical action regression cases."""

from clifra.core._kernel.contracts import resolve_contract


def plan_versor_action(algebra, *, grade, input_layout=None, output_layout=None, parameter_layout=None):
    inputs = resolve_contract(algebra, layout=input_layout, name="input_layout").layout
    output = resolve_contract(algebra, layout=output_layout or inputs, name="output_layout").layout
    parameter = resolve_contract(
        algebra, layout=parameter_layout or algebra.layout((grade,)), name="parameter_layout"
    ).layout
    return algebra.plan_versor_action(grade=grade, input=inputs, output=output, parameter=parameter)._kernel


def plan_sandwich_action(algebra, *, layout=None, dtype=None, device=None, cache=True):
    layout = resolve_contract(algebra, layout=layout, name="layout").layout
    return algebra._planner.full_sandwich_action_executor(
        layout=layout,
        dtype=algebra.dtype if dtype is None else dtype,
        device=algebra.device if device is None else device,
        cache=cache,
    )


def per_channel_sandwich(algebra, left, values, right=None, *, layout=None):
    right = algebra.reverse(left, input=layout) if right is None else right
    device, dtype = algebra._placement(left, values, right)
    return plan_sandwich_action(algebra, layout=layout, device=device, dtype=dtype)(left, values, right)


def multi_rotor_sandwich(algebra, left, values, right=None, *, layout=None):
    right = algebra.reverse(left, input=layout) if right is None else right
    device, dtype = algebra._placement(left, values, right)
    return plan_sandwich_action(algebra, layout=layout, device=device, dtype=dtype)(left, values.unsqueeze(-2), right)


def sandwich_product(algebra, left, values, right=None, *, layout=None):
    right = algebra.reverse(left, input=layout) if right is None else right
    device, dtype = algebra._placement(left, values, right)
    return plan_sandwich_action(algebra, layout=layout, device=device, dtype=dtype)(
        left.unsqueeze(-2), values, right.unsqueeze(-2)
    )


def sandwich_action_matrices(algebra, left, right=None, *, layout=None):
    right = algebra.reverse(left, input=layout) if right is None else right
    device, dtype = algebra._placement(left, right)
    return plan_sandwich_action(algebra, layout=layout, device=device, dtype=dtype).action_matrices(left, right)
