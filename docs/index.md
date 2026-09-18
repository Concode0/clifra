# clifra

clifra is a differentiable Clifford algebra computation layer for PyTorch.
Signatures and layouts give tensor coefficients their mathematical meaning.
Operations act on those tensors directly or can be planned for reuse.

A vector in $Cl(3,0)$ has three coefficients. Its geometric product with another
vector has a scalar part and a bivector part:

```python
import torch
from clifra import make_algebra

algebra = make_algebra(3, 0)
vectors = algebra.layout((1,))

a = torch.tensor([1.0, 0.0, 0.0])
b = torch.tensor([0.0, 1.0, 0.0])

ab = algebra.geometric_product(a, b, left=vectors, right=vectors)
# [0, 1, 0, 0]: coefficients of 1, e12, e13, e23
```

The final axis holds coefficients in the declared layout. Leading dimensions
broadcast as ordinary PyTorch dimensions; autograd differentiates the executed
tensor operations.

## Computation model

clifra keeps coefficient data in ordinary PyTorch tensors.

Layouts describe the algebraic meaning of the final coefficient axis without
changing PyTorch's ownership of leading dimensions, devices, dtypes, or
autograd.

Operations can be executed directly or planned from algebraic structure and
reused across changing tensor values and batch shapes.

The same model applies across Euclidean, projective, conformal, and indefinite
Clifford algebras.

---


The [tutorials](tutorials/index.md) introduce coefficient tensors, layouts,
broadcasting, and planned differentiation, followed by geometric applications.
The [how-to guides](how-to/index.md) cover individual tasks, including strict
blade geometry, resource budgets, compilation, and executor extensions.

The [explanations](explanations/index.md) develop representation, products,
signed forms, numerical behavior, and execution semantics.

The [research](research/index.md) notes show how the same computational model
is used in geometric optimization and transformation-field experiments.

The [API reference](reference/index.md) documents the stable public interfaces.
