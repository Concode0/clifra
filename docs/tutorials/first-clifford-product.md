# First Clifford Product

In the Euclidean algebra $Cl(2,0)$, the basis vectors satisfy
$e_1^2=e_2^2=1$ and $e_1e_2=-e_2e_1$. Represent them with a compact
grade-1 layout:

```python
import torch
from clifra import format_multivector, make_algebra

algebra = make_algebra(2, 0)
vectors = algebra.layout((1,))
e1 = torch.tensor([1.0, 0.0])
e2 = torch.tensor([0.0, 1.0])
```

The final axis stores coefficients of $e_1,e_2$. The layout supplied to
the operation explicitly assigns the tensor's geometric meaning.

```python
products = algebra.layout((0, 2))
e1_e2 = algebra.geometric_product(
    e1, e2, left=vectors, right=vectors, output=products,
)
e1_squared = algebra.geometric_product(
    e1, e1, left=vectors, right=vectors, output=products,
)
assert torch.equal(e1_e2, torch.tensor([0.0, 1.0]))
assert torch.equal(e1_squared, torch.tensor([1.0, 0.0]))
print(format_multivector(algebra, e1_e2, layout=products))  # e12
```

The result lanes are scalar and $e_{12}$. For general vectors,
$ab=\langle ab\rangle_0+a\wedge b$.

The scalar part is the Euclidean inner product in this signature. The bivector
part measures oriented area: reversing the order reverses its sign. This
yields a mixed-grade multivector, distinct from elementwise multiplication of the arrays.

```python
a = torch.tensor([2.0, 1.0])
b = torch.tensor([1.0, 3.0])

ab = algebra.geometric_product(a, b, left=vectors, right=vectors, output=products)
ba = algebra.geometric_product(b, a, left=vectors, right=vectors, output=products)

assert torch.equal(ab, torch.tensor([5.0, 5.0]))
assert torch.equal(ba, torch.tensor([5.0, -5.0]))
```

Here $\langle ab\rangle_0=2\cdot1+1\cdot3=5$ and the $e_{12}$
coefficient is $2\cdot3-1\cdot1=5$. In a mixed signature, the scalar
part changes with the basis squares. The exterior coefficient does not.

```python
bivectors = algebra.layout((2,))
area = algebra.wedge(e1, e2, left=vectors, right=vectors, output=bivectors)
assert torch.equal(area, torch.tensor([1.0]))
```

The exterior product retains only the oriented-area coefficient. It has a
different output layout from the geometric product, even when both represent
$e_{12}$.

Nothing wraps these tensors or takes ownership of their gradients. For example,
the derivative of the scalar part of $ab$ with respect to $a$ is $b$ in this
Euclidean signature:

```python
a = a.clone().requires_grad_()
ab = algebra.geometric_product(a, b, left=vectors, right=vectors, output=products)
ab[0].backward()
assert torch.equal(a.grad, b)
```

The layout defines which coefficient space the final axis represents; PyTorch
owns the values and the differentiation graph. Continue with
[Layouts, Storage, and Tensor Composition](layouts-and-plans.md) to make this
distinction explicit for larger algebras and batches.
