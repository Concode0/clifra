# Use unary and grade operations

Reversal changes the order of basis-vector factors. Grade involution changes
the sign of odd grades, and Clifford conjugation combines the two. On a
grade-$k$ component their signs are respectively
$(-1)^{k(k-1)/2}$, $(-1)^k$, and $(-1)^{k(k+1)/2}$.
They preserve the coefficient space unless an output layout selects grades.

```python
import torch
from clifra import make_algebra

algebra = make_algebra(3, dtype=torch.float64)
mixed = algebra.layout((0, 1, 2))
bivectors = algebra.layout((2,))
x = torch.arange(1, mixed.dim + 1, dtype=torch.float64)
grades = mixed.grade_indices_tensor()
reverse = algebra.plan_unary(op="reverse", input=mixed)
torch.testing.assert_close(reverse(reverse(x)), x)
conjugated = algebra.clifford_conjugation(x, input=mixed)
torch.testing.assert_close(
    conjugated, algebra.grade_involution(reverse(x), input=mixed),
)
selected = algebra.grade_projection(x, input=mixed, output=bivectors)
torch.testing.assert_close(selected, x[grades == 2])
assert selected.shape == (3,)
```

`grade_projection` requires `output`; it returns that layout's compact lanes
unless a canonical output contract is supplied. This differs from zeroing
lanes with `x * mixed.grade_mask((2,))`, which keeps the original layout and
width. Use projection when downstream operations should declare a smaller
coefficient space.

`pseudoscalar_product(x, input=layout)` computes right multiplication by
$I=e_1e_2\cdots e_n$ and defaults to complementary output grades. It is
not multiplication by $I^{-1}$. In a degenerate signature $I$ is not
invertible, so this operation should not be assumed to define an invertible
duality. The planning counterpart is `plan_pseudoscalar_product`.
