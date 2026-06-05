# Core Design

::: clifra

## Build An Algebra

```python
from clifra import make_algebra

algebra = make_algebra(3, 0, device="cpu")
```

## Declare Layouts

```python
vector_layout = algebra.layout((1,))
bivector_layout = algebra.layout((2,))
full_layout = algebra.layout()
```

## Plan A Product

```python
gp = algebra.plan_product(
    op="gp",
    left_layout=vector_layout,
    right_layout=vector_layout,
    output_layout=algebra.layout((0, 2)),
)
```

## Execute A Planned Graph

```python
import torch

left = torch.randn(8, vector_layout.dim)
right = torch.randn(8, vector_layout.dim)
result = gp(left, right)
```

## Use Full Lanes Universally

```python
import torch

vectors = torch.randn(8, algebra.n)
full = algebra.embed_vector(vectors)
rotor = algebra.exp(torch.randn(8, bivector_layout.dim), input_layout=bivector_layout)
```

## Use Layers With Layout Contracts

```python
from clifra.layers import CliffordLinear

layer = CliffordLinear(algebra, in_channels=2, out_channels=4, layout=vector_layout)
```
