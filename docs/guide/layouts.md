# Layouts

Layouts describe which basis blades occupy the last tensor dimension. They are
the first design decision for every clifra model because they determine tensor
width, product cost, and which grades a layer may read or write.

## Dense Multivectors

For small algebras, dense tensors are straightforward:

```python
from clifra.core.config import make_algebra

algebra = make_algebra(3, 0, kernel="dense", device="cpu")
x = algebra.embed_vector(points)  # [..., 2 ** algebra.n]
y = algebra.geometric_product(x, x)
```

Dense mode is useful when `2 ** n` is small enough that full blade materializing
is cheaper than managing active lanes.

## Compact Grade Layouts

For larger signatures, choose a context algebra and keep tensors on selected
grades:

```python
algebra = make_algebra(8, 0, kernel="context", default_grades=(1,))
vectors = algebra.layout((1,))
bivectors = algebra.layout((2,))
```

A `GradeLayout` records the algebra signature, selected grades, dense basis
indices, and compact lane count. Product planning uses that information to
avoid touching unrelated blades.

## Layer Contracts

Layers accept a layout directly when they should operate on compact lanes:

```python
from clifra.layers import RotorLayer

layout = algebra.layout((1,))
rotor = RotorLayer(algebra, channels=4, layout=layout)
```

If a layer receives dense input but its output is compact, the layer validates
that conversion explicitly. This avoids hidden reshapes and makes model
interfaces easier to inspect.

## Choosing Dense or Context

Use dense mode when:

- the algebra has a small `n`
- full-grade operations are common
- direct debugging is more important than memory width

Use context mode when:

- only a few grades matter
- `2 ** n` is too wide for routine tensor operations
- product plans should be cached and reused across model calls
