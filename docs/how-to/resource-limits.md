# Set resource limits

Planning checks static coefficient widths and pair/interaction footprints
before constructing execution buffers. An unavailable or over-budget route is
rejected; another eligible route may still serve the same operation. If none
can, planning raises an error rather than silently truncating the algebra.

Pass a `ResourceLimits` value when constructing the algebra. The defaults
warn at 2,048 lanes or 1,000,000 interactions and reject requirements above
4,096 lanes or 8,000,000 interactions. These bounds govern feasibility only; they do not select a preferred algorithm or promise a particular runtime.

Reduce the declared coefficient spaces when the mathematics permits it. A
vector dot product needs vector inputs and scalar output:

```python
import torch
from clifra import AlgebraConfig, ResourceLimits, make_algebra, make_algebra_from_config

limits = ResourceLimits()
algebra = make_algebra(16, resource_limits=limits)
assert algebra.resource_limits is limits
vector = algebra.layout((1,))
scalar = algebra.layout((0,))
dot = algebra.plan_product(left=vector, right=vector, output=scalar)
x = torch.arange(16, dtype=torch.float32)
torch.testing.assert_close(dot(x, x), x.square().sum().reshape(1))
assert vector.dim == 16
assert algebra.dim == 65536

config = AlgebraConfig(p=16, resource_limits=limits)
assert make_algebra_from_config(config).resource_limits is limits

small = ResourceLimits(warn_lanes=2, max_lanes=2)
restricted = make_algebra(3, resource_limits=small)
try:
    restricted.plan_product(
        left=restricted.layout((1,)), right=restricted.layout((1,)),
        output=restricted.layout((0,)),
    )
except (RuntimeError, ValueError) as error:
    assert "max_lanes" in str(error)
else:
    raise AssertionError("the vector input exceeds the configured lane budget")
```

Here canonical full storage would have 65,536 lanes, while the operation only
needs 16 input coefficients and one output coefficient. A compact declaration
can change feasibility, not just allocation size at the call site.

`algebra.resource_limits` exposes the exact injected frozen value through a
read-only property. To use another budget, construct another algebra and
build its plans. Moving an algebra or `CliffordModule` preserves the budget,
and moving a previously built plan does not trigger reassessment against another budget.

The static budget includes resident route structures, coexisting child plans,
and larger known fixed-shape temporaries. It excludes caller-controlled batch
dimensions and is not a byte-level device memory limit. Large batches and
pairwise item axes can still exhaust memory after a plan succeeds. Chunk those
axes in ordinary PyTorch code. For bivector-generated transformations, request
an induced action when you need transformed values rather than a materialized
rotor. Custom providers are subject to the same guards and must report
conservative resource requirements through their assessments.
