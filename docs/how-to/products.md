# Plan and reuse products

Use the geometric product for $AB$, the wedge for its highest-grade part,
and contractions when one homogeneous factor should reduce the grade of the
other. For homogeneous $A_r$ and $B_s$, the conventions are

\[
A_r\wedge B_s=\langle A_rB_s\rangle_{r+s},\qquad
A_r\mathbin{\lrcorner}B_s=\langle A_rB_s\rangle_{s-r}\quad(r\leq s),\qquad
A_r\mathbin{\llcorner}B_s=\langle A_rB_s\rangle_{r-s}\quad(r\geq s).
\]

The contractions are zero outside the indicated grade conditions and extend
bilinearly across mixed grades. No reversal of either argument is implicit.

```python
import torch
from clifra import make_algebra

algebra = make_algebra(2, dtype=torch.float64)
vectors = algebra.layout((1,))
bivectors = algebra.layout((2,))
even = algebra.layout((0, 2))

e1 = torch.tensor([1., 0.], dtype=torch.float64)
e2 = torch.tensor([0., 1.], dtype=torch.float64)

product = algebra.geometric_product(e1, e2, left=vectors, right=vectors)
torch.testing.assert_close(product, torch.tensor([0., 1.], dtype=e1.dtype))
wedge = algebra.wedge(e1, e2, left=vectors, right=vectors)
torch.testing.assert_close(wedge, torch.tensor([1.], dtype=e1.dtype))
contracted = algebra.left_contraction(e1, wedge, left=vectors, right=bivectors)
torch.testing.assert_close(contracted, e2)
planned = algebra.plan_product(op="geometric_product", left=vectors, right=vectors, output=even)
torch.testing.assert_close(planned(e1, e2), product)
```

`symmetric_product` means $(AB+BA)/2$. `commutator_product` means $AB-BA$,
and `anti_commutator_product` means $AB+BA$; the latter two omit the factor of
one half. `scalar_product` selects $\langle AB\rangle_0$, without reversing
either factor. These distinctions matter for mixed-grade coefficients.

An explicit `output` selects grades of the requested product. For a sequence
of products, retain every intermediate grade needed by later factors:
projecting $AB$ before multiplying by $C$ can change the selected part of
$ABC$. See [rotor composition](rotor-composition.md) for a three-factor use.
