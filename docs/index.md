---
layout: home

hero:
  name: clifra
  text: Layout-first Clifford algebra for PyTorch.
  tagline: Build geometric neural layers from explicit algebra signatures, grade layouts, and tensor contracts.
  actions:
    - theme: brand
      text: Read the Framework
      link: /framework/
    - theme: alt
      text: Open the Demo
      link: /demo

features:
  - title: Algebra is explicit
    details: Start with a signature `(p, q, r)` and choose dense execution or a compact planning context.
  - title: Layouts come first
    details: Grade layouts describe tensor lanes before layers or products execute, which keeps high-dimensional use predictable.
  - title: Modules stay thin
    details: Layers hold parameters and validation; pure formulas stay in `clifra.functional`; trainable criteria stay in `clifra.criterion`.
---

## What clifra Provides

clifra is a PyTorch framework for neural networks whose tensors carry Clifford
algebra structure. A multivector is represented by lanes for basis blades:
scalars, vectors, bivectors, and higher grades. The package makes those lanes
explicit through algebra specifications and layouts instead of hiding them
behind adapter-specific tensor conventions.

The core workflow is:

```python
from clifra.core.config import make_algebra
from clifra.layers import BladeSelector, RotorLayer

algebra = make_algebra(3, 0, kernel="dense", device="cpu")
data = algebra.embed_vector(points)

rotor = RotorLayer(algebra, channels=1)
selector = BladeSelector(algebra, channels=1, grades=(1,))
output = selector(rotor(data))
```

Use dense algebras for small dimensions where full multivectors are cheap. Use
the context backend and grade layouts when the algebra is too large to
materialize every blade.

## Package Shape

- `clifra.core` builds algebras, layouts, plans, and runtime execution tables.
- `clifra.functional` exposes stateless mathematical formulas.
- `clifra.criterion` wraps loss and orthogonality formulas as PyTorch modules.
- `clifra.layers.primitives` contains trainable neural layers.
- `clifra.layers.blocks` combines primitives into reusable model blocks.
- `clifra.layers.adapters` contains example projective and conformal embeddings.

The docs are intentionally framework-facing. They explain how the package is
organized and how to compose it, while source docstrings remain the reference
for individual arguments and return values.
