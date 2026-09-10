# Apply a bivector-generated action without materializing the rotor

Plan a grade-2 versor action when the output you need is a transformed
tensor. For parameter $B$, the convention is

\[
X'=R X\widetilde R,\qquad R=\exp(-B/2).
\]

Pass $B$ itself to the action. The minus sign and factor of one half belong
to this API's action convention; applying them again changes the result.

```python
import torch
from clifra import make_algebra

algebra = make_algebra(3, dtype=torch.float64)
vectors = algebra.layout((1,))
bivectors = algebra.layout((2,))
action = algebra.plan_versor_action(grade=2, input=vectors, parameter=bivectors)
theta = torch.tensor(torch.pi / 2, dtype=torch.float64, requires_grad=True)
B = torch.stack((theta, theta * 0, theta * 0))
x = torch.tensor([[1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
y = action(x, B)
torch.testing.assert_close(y, torch.tensor([[0., 1., 0.], [0., 0., 1.]], dtype=x.dtype))
gradient, = torch.autograd.grad(y[0, 0], (theta,))
torch.testing.assert_close(gradient, torch.tensor(-1., dtype=x.dtype))
```

The vector generator is $G(B)x=-\tfrac12(Bx-xB)$, and the vector action is
$\exp(G(B))x$. Applying its exterior powers extends the transformation to
other grades. The planner can use this $n\times n$ representation without
materializing rotor coefficients; the selected execution route depends on
the declared layouts and planning configuration. The action interface does
not require a rotor tensor from the caller.

Actions preserve grades. An explicit output layout selects which grades
to retain; it does not cause an input vector to become a bivector. Parameter
and value leading axes broadcast normally, so `B[:, None, :]` shares one
generator across an item axis and `B` shaped like the values' leading axes
defines a pointwise transformation field.

For a separately constructed vector-space matrix $M$, use
`plan_linear_action(input=..., output=...)` and call the plan as
`operation(values, matrix)`, with matrix shape `(..., n, n)`. It acts as
$x'_i=\sum_jM_{ij}x_j$ on vectors and extends by exterior powers. A general
linear matrix need not preserve the signature form; a bivector-generated
action does so mathematically, subject to floating-point error.
