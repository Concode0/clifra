# Compute norms, energies, and signed forms correctly

Use coefficient energy for a nonnegative penalty on stored coordinates.
Use a signature-sensitive form when the algebraic quadratic form is the
quantity of interest. They coincide on Euclidean vectors but not in general.

| Operation | Quantity |
| --- | --- |
| `lane_energy(A)` | $\sum_i a_i^2$ |
| `lane_norm(A)` | $\sqrt{\sum_i a_i^2}$ |
| `lane_dot_product(A, B)` | $\sum_i a_i b_i$ |
| `lane_distance(A, B)` | $\sqrt{\sum_i(a_i-b_i)^2}$ |
| `signature_norm_squared(A)` | $\langle A\widetilde A\rangle_0$ |
| `scalar_product(A, B)` | $\langle AB\rangle_0$ |
| `conjugate_scalar_form(A, B)` | $\langle\overline A B\rangle_0$ |

```python
import torch
from clifra import make_algebra

algebra = make_algebra(1, 1, dtype=torch.float64)
vectors = algebra.layout((1,))
x = torch.tensor([[1., 1.], [0., 2.]], dtype=torch.float64)
energy = algebra.lane_energy(x, input=vectors)
signed = algebra.signature_norm_squared(x, input=vectors)
torch.testing.assert_close(energy, torch.tensor([[2.], [4.]], dtype=x.dtype))
torch.testing.assert_close(signed, torch.tensor([[0.], [-4.]], dtype=x.dtype))
conjugate = algebra.conjugate_scalar_form(x, x, left=vectors, right=vectors)
torch.testing.assert_close(conjugate, -signed)
by_grade = algebra.lane_grade_energy(x, input=vectors)
assert by_grade.shape == (2, 3)
torch.testing.assert_close(by_grade[:, 1:2], energy)
```

Scalar quantities retain a final width-one axis and do not reduce batches.
Grade energies and norms instead have a final axis of length $n+1$, including
zeros for absent grades. `lane_energy(..., grades=(...))` restricts the
coefficient penalty to selected grades. `lane_grade_distribution` normalizes
grade energies with an additive denominator epsilon; its sum need not be
exactly one.

The coefficient dot, distance, and conjugate form align differing declared
layouts by canonical blade identity, treating absent lanes as zero. Their
leading axes still broadcast normally.

A zero signed square need not mean zero coefficients; the first row above
is a nonzero null vector. Do not take its square root as a positive norm or
use it as a nonnegative loss. If division by this form is required, choose
the domain and singular-input policy explicitly. Squared coefficient energy
also avoids the square-root derivative singularity of a norm at zero.
