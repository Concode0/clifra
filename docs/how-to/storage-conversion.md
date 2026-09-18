# Convert between compact and canonical storage

Keep tensors compact when only selected grades are needed. Use canonical
storage when exchanging coefficients with code that expects all $2^n$ basis
lanes. The semantic layout and physical storage are separate declarations.

`layout.full` scatters compact coefficients into the canonical basis, filling
absent lanes with zeros. `layout.compact` gathers the declared lanes.
`target.convert(values, source)` transfers shared lanes directly between
compact layouts, dropping absent target grades and zero-filling new ones.

```python
import torch
from clifra import make_algebra
from clifra.core import TensorContract

algebra = make_algebra(3, dtype=torch.float64)
vectors = algebra.layout((1,))
scalar_vectors = algebra.layout((0, 1))
x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
canonical = vectors.full(x)
assert canonical.shape == (8,)
torch.testing.assert_close(vectors.compact(canonical), x)
extended = scalar_vectors.convert(x, vectors)
torch.testing.assert_close(extended, torch.tensor([0., 1., 2., 3.], dtype=x.dtype))

contract = TensorContract.canonical(vectors)
norm_squared = algebra.plan_signature_norm_squared(input=contract)
result = norm_squared(canonical)
torch.testing.assert_close(result, torch.tensor([14.0], dtype=x.dtype))
result.sum().backward()
torch.testing.assert_close(x.grad, 2 * x.detach())
```

A canonical contract for vectors requires full-width storage but still
semantically represents only grade 1. Operations gather only the declared lanes,
ignoring coefficients outside the layout. Canonical outputs place zeros
outside the declared layout. `TensorContract.validate` checks the lane axis,
not whether undeclared canonical lanes contain zeros.

Passing a layout directly is shorthand for a compact contract. A full-basis
layout in compact storage already has $2^n$ lanes; “full” is a grade selection,
not a third storage mode. Conversions preserve leading axes, dtype, device,
and the PyTorch gradient path.
