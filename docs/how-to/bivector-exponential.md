# Materialize a bivector exponential

Use `bivector_exp` when you need coefficients of $\exp(B)$ for storage,
composition, or another Clifford product. The argument is the actual
exponent: no minus sign or half-angle is inserted. A compact grade-2 input
defaults to all even grades in the result.

```python
import torch
from clifra import make_algebra

algebra = make_algebra(3, dtype=torch.float64)
bivectors = algebra.layout((2,))
even = algebra.layout((0, 2))
exponential = algebra.plan_bivector_exp(input=bivectors, output=even)
theta = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
B = torch.stack((theta, theta * 0, theta * 0))
value = exponential(B)
expected = torch.stack((theta.cos(), theta.sin(), theta * 0, theta * 0))
torch.testing.assert_close(value, expected)
derivative, = torch.autograd.grad(value[0], (theta,))
torch.testing.assert_close(derivative, -theta.detach().sin())
```

Here $e_{12}^2=-1$, giving the circular exponential. A bivector whose square
is a positive scalar gives hyperbolic functions; a square-zero bivector
gives $1+B$. General bivectors need not have scalar squares, so applying
this two-term formula to their coefficient norm is incorrect.

Use an explicit output layout to select coefficients of the exponential.
This selection does not project every intermediate power in the series.
Full-basis input is accepted but projected to grade 2 first; this API is not
a general mixed-grade multivector exponential.

Materialization may require larger intermediate coefficient spaces than
the requested output. Built-in materialized methods cover dimensions 2
through 12 subject to resource limits. Their numerical routes include
finite closures, scaling and squaring, and matrix exponentiation; the
Taylor route requires coefficient L1 norm at most 65,536. Large hyperbolic
arguments can overflow regardless of the exact algebraic formula.

If only transformed values are needed, use an
[induced action](induced-actions.md). Its vector-space representation can
avoid materializing the exponentially growing even coefficient space.
