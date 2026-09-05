"""Private tuning entry point for kernel tests and benchmark harnesses."""


def configured_algebra(
    p, q=0, r=0, *, device="cpu", dtype=None, planning_policy=None, resource_limits=None, bivector_exp_options=None
):
    from clifra.core.algebra import AlgebraContext

    kwargs = {} if dtype is None else {"dtype": dtype}
    algebra = AlgebraContext(p, q, r, device=device, **kwargs)
    if planning_policy is not None:
        algebra._planning_policy = planning_policy
    if resource_limits is not None:
        algebra._resource_limits = resource_limits
    if bivector_exp_options is not None:
        algebra._bivector_exp_options = bivector_exp_options
    return algebra
