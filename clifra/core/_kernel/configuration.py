"""Private tuning entry point for kernel tests and benchmark harnesses."""


def configured_algebra(
    p,
    q=0,
    r=0,
    *,
    device="cpu",
    dtype=None,
    planning_policy=None,
    resource_limits=None,
    executor_registry=None,
):
    from clifra.core.algebra import AlgebraContext

    kwargs = {} if dtype is None else {"dtype": dtype}
    algebra = AlgebraContext(
        p, q, r, device=device, registry=executor_registry, resource_limits=resource_limits, **kwargs
    )
    if planning_policy is not None:
        algebra._planning_policy = planning_policy
    algebra._planner.policy = algebra._planning_policy
    algebra._planner.limits = algebra._resource_limits
    return algebra
