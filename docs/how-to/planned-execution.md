# Understand and control planned execution

Use `plan_*` when the same mathematical operation recurs with new coefficients.
The returned module fixes the signature, operation, input and output layouts,
storage contracts, and any pairwise item-axis convention. It does not fix the
coefficient values or ordinary leading batch dimensions.

```python
import torch
from clifra import ResourceLimits, TensorContract, make_algebra

limits = ResourceLimits()
algebra = make_algebra(3, dtype=torch.float64, resource_limits=limits)
vector = algebra.layout((1,))
bivector = algebra.layout((2,))
wedge = algebra.plan_product(
    op="wedge", left=vector, right=vector, output=bivector
)
assert wedge.inputs == (TensorContract.compact(vector),) * 2
assert wedge.output == TensorContract.compact(bivector)

for batch in (2, 5):
    x = torch.randn(batch, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn(3, dtype=torch.float64)
    result = wedge(x, y)
    assert result.shape == (batch, 3)
    result.square().sum().backward()
    assert torch.isfinite(x.grad).all()
```

Each call executes the resolved operation without consulting the algebra or
planner. Gradients belong to the input tensor graph; reusing a plan does not
retain a previous call's graph. Register plans on an `nn.Module` to move their
buffers and compose them with other operations.

Construction checks the algebra's `resource_limits` against the static
requirements of eligible execution routes. These limits bound coefficient
widths and fixed interaction footprints, excluding caller-controlled batch
sizes. A successful plan therefore does not guarantee that every batch fits
device memory. The limits determine feasibility, not a performance policy;
see [planning resource limits](resource-limits.md) to configure them.

Direct calls such as `algebra.wedge(...)` remain useful for exploratory work.
They resolve declarations and use cached execution machinery. An explicit plan
makes that boundary visible and gives a stable callable for a loop or compiled
function.

Changing an input layout, requested output grades, or `pairwise` requires a new
plan. `.to()` changes execution placement, not the mathematical contracts. Build
plans in the intended precision: movement converts buffers but does not rerun
algorithm selection. Request only the output grades needed by the next operation;
the requested result is a grade projection, not a requirement to construct every
intermediate full multivector.
