# Layouts, Storage, and Tensor Composition

A layout defines the coefficient space. Storage determines how its coefficients
occupy a tensor's final axis. Leading axes remain ordinary PyTorch axes: they
can describe samples, channels, grids, or any other application structure.

## Choose the coefficient space

```python
import torch
from clifra import TensorContract, make_algebra

algebra = make_algebra(3, 0)
vectors = algebra.layout((1,))
products = algebra.layout((0, 2))
assert vectors.dim == 3
assert products.dim == 4
```

Basis blades follow increasing bitmask order. Bit zero represents $e_1$, bit
one $e_2$, and bit two $e_3$. The canonical basis is therefore
$1,e_1,e_2,e_{12},e_3,e_{13},e_{23},e_{123}$, not grade order.
A compact layout filters this ordering to its selected grades. The vector lanes
are $e_1,e_2,e_3$; the product lanes are $1,e_{12},e_{13},e_{23}$.

Two arrays of the same width can mean different things. Width three could
describe vectors or bivectors in this algebra, so operations need declarations
rather than guessing a layout from tensor shape.

```python
product = algebra.plan_product(
    op="geometric_product", left=vectors, right=vectors, output=products,
)
a = torch.tensor([1.0, 0.0, 0.0])
b = torch.tensor([0.0, 1.0, 0.0])
result = product(a, b)
assert torch.equal(result, torch.tensor([0.0, 1.0, 0.0, 0.0]))
```

## Convert storage explicitly

Compact storage holds only the selected lanes. Canonical storage has $2^n$
lanes, placing those same coefficients at their full-basis positions and zeros
elsewhere. Conversion does not change the mathematical value.

```python
canonical = products.full(result)
assert canonical.shape == (8,)
assert canonical[3] == 1  # e12 has bitmask 3
assert torch.equal(products.compact(canonical), result)

canonical_product = algebra.plan_product(
    op="geometric_product",
    left=TensorContract(vectors, storage="canonical"),
    right=TensorContract(vectors, storage="canonical"),
    output=TensorContract(products, storage="canonical"),
)
assert torch.equal(canonical_product(vectors.full(a), vectors.full(b)), canonical)
```

Passing a layout directly declares compact storage. `TensorContract` makes the
storage choice explicit while retaining the same semantic layout. Prefer
compact storage when the computation needs only a few grades; a canonical
array allocates all basis lanes even when most are zero.

## Compose leading dimensions

Broadcasting follows PyTorch's usual rules on every axis before the coefficient
axis. A singleton axis can create all point-to-direction products with the
same operation.

```python
points = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
directions = torch.eye(3)
all_products = product(points[:, None, :], directions[None, :, :])
assert all_products.shape == (2, 3, 4)
assert torch.equal(all_products[0, 1], result)

# Pool over directions, leaving one multivector per point.
pooled = all_products.mean(dim=1)
assert pooled.shape == (2, 4)
```

Stacking, indexing, concatenating along sample axes, and reducing those axes
are ordinary tensor operations. Changing the final axis can change the
coefficient meaning, so it requires a matching layout rather than just a new
shape. The next tutorial examines what the reusable `product` has fixed and
what remains dynamic in [Planned Differentiation](planned-differentiation.md).
