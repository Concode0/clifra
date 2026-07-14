# Layouts and Planned Execution

Compact grade layouts retain only the basis blades required by an operation.
The following $Cl(3, 0)$ example includes the algebra construction, layout
declarations, product plan, execution, and storage conversion. It does not
depend on the canonical-storage tutorial.

## Declare the Layouts

```python
import torch

from clifra import format_multivector, make_algebra

algebra = make_algebra(3, 0, device="cpu")
vectors = algebra.layout((1,))
scalar_and_bivector = algebra.layout((0, 2))

assert vectors.dim == 3
assert scalar_and_bivector.dim == 4
```

`vectors` stores only $e_1$, $e_2$, and $e_3$. The output layout stores the
scalar plus the three bivector lanes.

## Plan Once

```python
vector_product = algebra.plan_product(
    op="gp",
    left_layout=vectors,
    right_layout=vectors,
    output_layout=scalar_and_bivector,
)
```

Planning resolves basis interactions, output positions, signs, and executor
selection before the data arrives. The returned handle is a normal callable
PyTorch module.

## Execute on Compact Tensors

```python
left = torch.tensor([[1.0, 0.0, 0.0]])
right = torch.tensor([[0.0, 1.0, 0.0]])

result = vector_product(left, right)

assert result.shape == (1, scalar_and_bivector.dim)
print(format_multivector(algebra, result, layout=scalar_and_bivector))
```

The output is `e12`. Reuse `vector_product` for every tensor that follows the
same layouts, dtype, and device.

## Convert Between Storage Forms

```python
canonical = scalar_and_bivector.full(result)
compact_again = scalar_and_bivector.compact(canonical)

assert canonical.shape == (1, algebra.dim)
assert torch.equal(compact_again, result)
```

The layout gives both representations the same semantic meaning. Only their
physical last-axis widths differ.

The same layout contract can be attached to PyTorch layers. A minimal trained
case appears in [Learn a Geometric Transformation](learn-geometric-transform.md).
