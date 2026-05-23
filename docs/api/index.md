# API Map

The API documentation is curated around framework entry points. Source
docstrings describe individual parameters, while this page shows where each
kind of work belongs.

## Core

```python
from clifra.core.config import AlgebraConfig, make_algebra, make_algebra_from_config
```

Use `make_algebra()` for direct construction and `make_algebra_from_config()`
for Hydra or mapping-based configuration. Runtime algebras expose dense
operations; context algebras expose layout and planning contracts.

Important namespaces:

- `clifra.core.foundation`: basis utilities, layouts, device and dtype helpers
- `clifra.core.planning`: static grade plans for products, unary operations,
  actions, and decomposition
- `clifra.core.runtime`: dense algebra execution and metric-aware operations
- `clifra.core.analysis`: optional analyzers for dimension, signature,
  geodesic flow, and symmetry

## Functional

```python
from clifra.functional import geometric_product, grade_projection, geometric_mse
```

Use this namespace for stateless tensor formulas. Functional modules should not
own trainable parameters or presentation logic.

## Layers

```python
from clifra.layers import RotorLayer, CliffordLinear, GeometricTransformerBlock
```

Use layers for trainable neural modules. Primitives live under
`clifra.layers.primitives`; larger compositions live under
`clifra.layers.blocks`; projective and conformal embedding examples live under
`clifra.layers.adapters`.

## Criteria

```python
from clifra.criterion import GeometricMSELoss, IsometryLoss, StrictOrthogonality
```

Use criteria when a loss or regularizer needs module state, cached masks, or a
standard PyTorch training-loop interface.

## Optimizers

```python
from clifra.optimizers import RiemannianSGD
```

Optimizers remain separate from algebra and layer definitions. They can be
adopted selectively when a training loop needs manifold-aware updates.
