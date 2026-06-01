# Algebra and Layouts

An algebra is declared by signature. The dense backend materializes every basis
blade. The context backend keeps grade declarations compact and plans products
from layout contracts.

## Dense Algebra

```python
import torch

from clifra.core.config import make_algebra
from clifra.functional import geometric_product, wedge

algebra = make_algebra(3, 0, kernel="dense", device="cpu")

a = algebra.embed_vector(torch.tensor([[1.0, 0.0, 0.0]]))
b = algebra.embed_vector(torch.tensor([[0.0, 1.0, 0.0]]))

ab = geometric_product(algebra, a, b)
area = wedge(algebra, a, b)

assert ab.shape[-1] == algebra.dim
assert area.shape[-1] == algebra.dim
```

## Compact Grade Layouts

Use compact layouts when the operation only needs selected grades.

```python
import torch

from clifra.core.config import make_algebra
from clifra.layers import WedgeLayer

algebra = make_algebra(6, 0, kernel="context", default_grades=(1,), device="cpu")
vector_layout = algebra.layout((1,))
bivector_layout = algebra.layout((2,))

left = torch.randn(4, vector_layout.dim)
right = torch.randn(4, vector_layout.dim)

wedge = WedgeLayer(
    algebra,
    left_layout=vector_layout,
    right_layout=vector_layout,
    output_layout=bivector_layout,
)
out = wedge(left, right)

assert out.shape == (4, bivector_layout.dim)
```

The layout is part of the contract. If a tensor has the wrong lane width for
its declared grades, clifra raises before the product executes.
