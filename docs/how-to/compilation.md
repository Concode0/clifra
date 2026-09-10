# Compile Clifford computations

Plan before calling `torch.compile`. Planning resolves the algebraic operation
and storage contracts; compilation then sees its tensor execution. Keep Python
configuration and plan construction outside the compiled function.

```python
import torch
from clifra import make_algebra

algebra = make_algebra(3, dtype=torch.float64)
vector = algebra.layout((1,))
even = algebra.layout((0, 2))
product = algebra.plan_product(left=vector, right=vector, output=even)
compiled = torch.compile(product, backend="aot_eager", fullgraph=True)

x = torch.tensor([[1., 2., 0.]], dtype=torch.float64, requires_grad=True)
y = torch.tensor([[0., 1., 3.]], dtype=torch.float64, requires_grad=True)
expected, actual = product(x, y), compiled(x, y)
torch.testing.assert_close(actual, expected)
expected_grad = torch.autograd.grad(expected.square().sum(), (x, y))
actual_grad = torch.autograd.grad(actual.square().sum(), (x, y))
for got, want in zip(actual_grad, expected_grad):
    torch.testing.assert_close(got, want)
```

`aot_eager` checks graph capture and differentiation without generating an
optimized device kernel. This is also the backend used by the repository's
representative compilation tests; it does not establish a speedup or support
for every optimizing backend. Select and test an optimizing backend for your
device, dtype, operation, and workload separately.

Batch shapes remain ordinary tensor shapes. PyTorch may specialize or recompile
for new shapes; a reusable clifra plan does not promise one compiled graph for
every batch size. Move the module to its intended device and dtype before
compiling. Data-dependent strict geometry checks may raise from compiled
execution when their mathematical preconditions fail.
