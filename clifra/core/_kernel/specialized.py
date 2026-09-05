"""Private model-shaped kernels retained for numerical regression and benchmarks."""

from clifra.core._kernel.contracts import resolve_contract

from .execution.action import MultiVersorActionExecutor, PairedBivectorActionExecutor


def plan_versor_action(algebra, *, grade, input_layout=None, output_layout=None, parameter_layout=None):
    inputs = resolve_contract(algebra, layout=input_layout, name="input_layout").layout
    output = resolve_contract(algebra, layout=output_layout or inputs, name="output_layout").layout
    parameter = resolve_contract(
        algebra, layout=parameter_layout or algebra.layout((grade,)), name="parameter_layout"
    ).layout
    return algebra.plan_versor_action(grade=grade, input=inputs, output=output, parameter=parameter)._kernel


def plan_multi_versor_action(
    algebra,
    *,
    grade,
    input_layout=None,
    output_layout=None,
    parameter_layout=None,
    input_grades=None,
    output_grades=None,
):
    inputs = resolve_contract(algebra, layout=input_layout, grades=input_grades, name="input_layout").layout
    output = resolve_contract(
        algebra,
        layout=output_layout or (None if output_grades is not None else inputs),
        grades=output_grades,
        name="output_layout",
    ).layout
    parameter = resolve_contract(
        algebra, layout=parameter_layout or algebra.layout((grade,)), name="parameter_layout"
    ).layout
    plan = algebra._planner.versor_action_plan(
        grade=grade, input_layout=inputs, output_layout=output, parameter_layout=parameter
    )
    return MultiVersorActionExecutor(
        algebra,
        grade=grade,
        input_layout=inputs,
        output_layout=output,
        parameter_layout=parameter,
        execution_path=plan.execution_path,
    )


def plan_paired_bivector_action(
    algebra, *, input_layout=None, output_layout=None, parameter_layout=None, input_grades=None, output_grades=None
):
    inputs = resolve_contract(algebra, layout=input_layout, grades=input_grades, name="input_layout").layout
    output = resolve_contract(
        algebra,
        layout=output_layout or (None if output_grades is not None else inputs),
        grades=output_grades,
        name="output_layout",
    ).layout
    parameter = resolve_contract(
        algebra, layout=parameter_layout or algebra.layout((2,)), name="parameter_layout"
    ).layout
    plan = algebra._planner.paired_bivector_action_plan(
        input_layout=inputs, output_layout=output, parameter_layout=parameter
    )
    return PairedBivectorActionExecutor(
        algebra,
        input_layout=plan.input_layout,
        output_layout=plan.output_layout,
        parameter_layout=plan.parameter_layout,
        rotor_layout=plan.rotor_layout,
        middle_layout=plan.middle_layout,
        execution_path=plan.execution_path,
    )


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
    return plan_sandwich_action(algebra, layout=layout, device=device, dtype=dtype).per_channel(left, values, right)


def multi_rotor_sandwich(algebra, left, values, right=None, *, layout=None):
    right = algebra.reverse(left, input=layout) if right is None else right
    device, dtype = algebra._placement(left, values, right)
    return plan_sandwich_action(algebra, layout=layout, device=device, dtype=dtype).multi(left, values, right)


def sandwich_product(algebra, left, values, right=None, *, layout=None):
    right = algebra.reverse(left, input=layout) if right is None else right
    device, dtype = algebra._placement(left, values, right)
    return plan_sandwich_action(algebra, layout=layout, device=device, dtype=dtype).batched(left, values, right)


def sandwich_action_matrices(algebra, left, right=None, *, layout=None):
    right = algebra.reverse(left, input=layout) if right is None else right
    device, dtype = algebra._placement(left, right)
    return plan_sandwich_action(algebra, layout=layout, device=device, dtype=dtype).action_matrices(left, right)


def multi_versor_action(algebra, values, weights, mix, **kwargs):
    return plan_multi_versor_action(algebra, **kwargs)(values, weights, mix)


def paired_bivector_action(algebra, values, left_weights, right_weights, channel_to_pair, **kwargs):
    return plan_paired_bivector_action(algebra, **kwargs)(values, left_weights, right_weights, channel_to_pair)
