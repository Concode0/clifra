# Use pairwise operations

Use `pairwise=True` when every item on the left should combine with every
item on the right. Inputs have shapes `(..., N, left_lanes)` and
`(..., M, right_lanes)`; the result has shape `(..., N, M, output_lanes)`.
Earlier dimensions broadcast. The item axes are not reduced.

```python
import torch
from clifra import make_algebra

algebra = make_algebra(3, dtype=torch.float64)
vectors = algebra.layout((1,))
scalar = algebra.layout((0,))
pairwise = algebra.plan_product(
    left=vectors, right=vectors, output=scalar, pairwise=True,
)
x = torch.arange(18, dtype=torch.float64).reshape(2, 3, 3)
y = torch.arange(15, dtype=torch.float64).reshape(5, 3)
scores = pairwise(x, y)
assert scores.shape == (2, 3, 5, 1)
reference = algebra.geometric_product(
    x.unsqueeze(-2), y.unsqueeze(-3),
    left=vectors, right=vectors, output=scalar,
)
torch.testing.assert_close(scores, reference)
torch.testing.assert_close(scores.squeeze(-1), x @ y.T)
```

The last equality uses Euclidean vectors. In a mixed signature the scalar
geometric product is the signed bilinear form, so it is not an ordinary
coefficient dot product. Select output grades explicitly when only scalar
scores or bivector relations are needed.

The pairwise axis convention is fixed when planning. For a one-item input,
insert an item axis with `unsqueeze(-2)`. For large item sets, split an item
axis into chunks before execution, as the pairwise output still grows as $NM$
regardless of coefficient compactness. Reduce results with ordinary PyTorch
operations after deciding which item axis the reduction represents.
