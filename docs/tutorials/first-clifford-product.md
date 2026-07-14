# First Clifford Product

The Euclidean algebra $Cl(2, 0)$ is small enough to expose every part of a
Clifford calculation. The example below constructs the algebra, embeds its two
basis vectors, and evaluates their geometric product. It assumes Python and
PyTorch but no prior clifra API knowledge.

## Construct an Algebra

```python
import torch

from clifra import format_multivector, make_algebra

algebra = make_algebra(2, 0, device="cpu")
```

The signature says that the algebra has two basis vectors whose squares are
positive. It therefore has $2^2 = 4$ canonical basis lanes: scalar, $e_1$,
$e_2$, and $e_{12}$.

## Embed Two Vectors

`embed_vector` maps ordinary coordinates into the grade-1 lanes of a canonical
multivector tensor.

```python
e1 = algebra.embed_vector(torch.tensor([[1.0, 0.0]]))
e2 = algebra.embed_vector(torch.tensor([[0.0, 1.0]]))

assert e1.shape == (1, algebra.dim)
assert algebra.dim == 4
```

The final tensor axis always holds Clifford coefficients. Batch and channel
axes, when present, come before it.

## Multiply Them

```python
e1_e2 = algebra.geometric_product(e1, e2)
e1_squared = algebra.geometric_product(e1, e1)

print(format_multivector(algebra, e1_e2))
print(format_multivector(algebra, e1_squared))
```

The meaningful parts of the output are:

```text
e12
1
```

Orthogonal vectors produce the oriented plane $e_{12}$, while the Euclidean
basis vector satisfies $e_1^2 = 1$.

## Compare the Exterior Product

```python
oriented_area = algebra.wedge(e1, e2)

assert torch.equal(oriented_area, e1_e2)
print(format_multivector(algebra, oriented_area))
```

For these orthogonal basis vectors the geometric and exterior products agree.
For general vectors, the geometric product can also contain a scalar inner-product
part.

The example establishes the canonical representation and its product semantics.
[Layouts and Planned Execution](layouts-and-plans.md) expresses the same class
of calculation with compact tensor contracts.
