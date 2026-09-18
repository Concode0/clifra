# Project, reject, and invert safely

For a non-null blade $A$, use

\[
A^{-1}=\frac{\widetilde A}{\langle A\widetilde A\rangle_0},\qquad
\operatorname{proj}_A(x)=(x\mathbin{\lrcorner}A)A^{-1},\qquad
\operatorname{rej}_A(x)=x-\operatorname{proj}_A(x).
\]

For vector $x$, projection gives the component in the subspace represented
by $A$. Supply a decomposable blade: a homogeneous tensor is not necessarily
a blade in higher dimensions. The inverse helper uses this formula; it does
not solve a general multivector inverse or verify decomposability.

```python
import torch
from clifra import make_algebra

algebra = make_algebra(3, dtype=torch.float64)
vectors = algebra.layout((1,))
bivectors = algebra.layout((2,))

plane = torch.tensor([2., 0., 0.], dtype=torch.float64)  # 2 e12
x = torch.tensor([1., 2., 3.], dtype=torch.float64)

project = algebra.plan_strict_blade_project(input=vectors, blade=bivectors)
projection = project(x, plane)
rejection = algebra.strict_blade_reject(x, plane, input=vectors, blade=bivectors)

torch.testing.assert_close(projection, torch.tensor([1., 2., 0.], dtype=x.dtype))
torch.testing.assert_close(projection + rejection, x)

inverse = algebra.strict_blade_inverse(plane, input=bivectors)
identity = algebra.geometric_product(
    plane, inverse, left=bivectors, right=bivectors, output=algebra.layout((0,)),
)

torch.testing.assert_close(identity, torch.ones(1, dtype=x.dtype))

try:
    algebra.strict_blade_inverse(torch.zeros_like(plane), input=bivectors)
except ValueError as error:
    assert "zero denominator" in str(error)
else:
    raise AssertionError("a zero blade has no inverse")
```

The `strict_` variants divide by the computed signed square and reject any
exactly zero denominator in the input batch. Non-null does not mean
positive: negative denominators are valid and keep their sign. A nonzero
blade with zero signed square is singular, including null blades in mixed
or degenerate signatures.

The corresponding methods without `strict_` clamp the denominator's
magnitude to `algebra.eps_sq`, preserving negative signs and assigning a
positive sign at zero. They can remain finite at singular inputs, but do
not produce a mathematical inverse there. Near-null inverses can still be
very large. For the zero blade, the regularized inverse and projection are
zero and rejection returns the input. Rejection requires an output layout
matching its input layout.

For a supplied versor $V$, `versor_product(V, x, input=..., versor=...)`
computes $\widehat V x V^{-1}$; `strict_versor_product` uses the exact
denominator. This twisted convention includes a vector's reflection sign.
For an explicit even rotor, it reduces to the usual sandwich. Arbitrary
mixed-parity tensors are not guaranteed to be versors.
