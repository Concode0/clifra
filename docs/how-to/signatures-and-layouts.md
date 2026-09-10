# Define signatures and layouts

Choose the signature from the quadratic form your computation needs. In
`make_algebra(p, q, r)`, the first `p` basis vectors square to $+1$, the next
`q` to $-1$, and the remaining `r` to zero. Distinct basis vectors anticommute.
Changing a signature changes products, not just metadata attached to tensors.

A layout defines the coefficient space. `algebra.layout((1,))` contains every
vector lane; `algebra.layout((0, 2))` contains the scalar and every bivector
lane. Grades are normalized to a sorted set, but coefficient lanes follow
increasing canonical bitmask index, **not** a concatenation of grade blocks.
Bit $i-1$ denotes basis vector $e_i$, for $i=1,\ldots,n$.

```python
import torch
from clifra import make_algebra

algebra = make_algebra(2, 1, dtype=torch.float64)
vectors = algebra.layout((1,))
mixed = algebra.layout((0, 1, 2))
assert vectors.basis_indices == (1, 2, 4)
assert mixed.basis_indices == (0, 1, 2, 3, 4, 5, 6)
# Mixed lanes: 1, e1, e2, e12, e3, e13, e23.
e3 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
square = algebra.scalar_product(e3, e3, left=vectors, right=vectors)
torch.testing.assert_close(square, torch.tensor([-1.0], dtype=torch.float64))
assert mixed.positions_for_grades((1,)).tolist() == [1, 2, 4]
```

Use `positions_for_grades` or `positions_for_basis` when selecting lanes;
contiguous slices are not generally grade selections. A grade-$k$ layout in
dimension $n$ has $\binom nk$ lanes, while `algebra.layout()` has $2^n$.
Whole-grade layouts are the currently supported layout implementation.

Pass the layout at operation boundaries. The same width can represent
different grades, so clifra does not infer a layout from a tensor's shape.
Omitting a declaration means the full basis.
