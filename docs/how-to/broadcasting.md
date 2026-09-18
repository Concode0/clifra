# Work with leading dimensions and broadcasting

The final tensor axis contains coefficients. All earlier axes follow ordinary PyTorch broadcasting rules; their batch, channel, or spatial roles are application-defined. Insert singleton axes to state which values share a parameter.

Here each of two batches contains four vectors. One vector per batch is
multiplied with every item in that batch.

```python
import torch
from clifra import make_algebra

algebra = make_algebra(3, dtype=torch.float64)
vectors = algebra.layout((1,))
output = algebra.layout((0, 2))
product = algebra.plan_product(left=vectors, right=vectors, output=output)
x = torch.arange(24, dtype=torch.float64).reshape(2, 4, 3)
w = torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=x.dtype)
y = product(x, w[:, None, :])
assert y.shape == (2, 4, 4)
for batch in range(2):
    for item in range(4):
        torch.testing.assert_close(y[batch, item], product(x[batch, item], w[batch]))
```

The alignment must be explicit: without `[:, None, :]`, the leading axes
`(2, 4)` and `(2,)` from shapes `(2, 4, 3)` and `(2, 3)` are incompatible.
A scalar-valued Clifford result keeps a final lane of width one. Use
`squeeze(-1)` only when the consumer needs an ordinary scalar tensor.

The plan fixes the coefficient contracts, not these leading sizes. Reuse it
with different broadcast-compatible leading dimensions. Tensor arithmetic,
`stack`, `cat`, indexing, and reductions remain PyTorch operations; ensure
coefficient layouts agree before adding tensors or concatenating item axes.
