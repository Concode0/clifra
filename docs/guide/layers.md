# Layers

clifra layers are PyTorch modules that keep Clifford algebra structure visible.
They accept an algebra object, optional grade layouts, and ordinary PyTorch
tensors.

## Primitives

The primitive layer namespace is the main building surface:

```python
from clifra.layers import (
    BladeSelector,
    CliffordLayerNorm,
    CliffordLinear,
    GeometricGELU,
    RotorLayer,
)
```

- `RotorLayer` and `MultiRotorLayer` learn geometric actions from bivectors.
- `ReflectionLayer` models Pin-style reflection actions.
- `CliffordLinear` mixes channels while preserving declared blade lanes.
- `BladeSelector` projects or gates selected basis blades.
- `GeometricGELU`, `GeometricSquare`, and `GradeSwish` wrap pure activation
  formulas from `clifra.functional.activation`.
- `ProductLayer` and its specializations expose planned geometric products as
  modules.

## Blocks

Blocks compose primitives into reusable neural components:

```python
from clifra.layers import GeometricProductAttention, GeometricTransformerBlock
```

Use blocks when the model shape is already known and the repeated composition is
more important than direct control over every product call.

## Example Adapters

Adapters are examples, not framework infrastructure. The package keeps only the
layout-first projective and conformal embeddings:

```python
from clifra.layers.adapters import ConformalEmbedding, ProjectiveEmbedding
```

These classes demonstrate how domain embeddings can map ordinary coordinates
into multivector lanes and extract coordinates back out. New application
embeddings should follow that pattern without adding global adapter state.

## A Minimal Model

```python
import torch
from torch import nn

from clifra.core.config import make_algebra
from clifra.layers import BladeSelector, CliffordLayerNorm, RotorLayer


class VectorRotorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.algebra = make_algebra(3, 0, kernel="dense", device="cpu")
        self.rotor = RotorLayer(self.algebra, channels=1)
        self.norm = CliffordLayerNorm(self.algebra, channels=1)
        self.readout = BladeSelector(self.algebra, channels=1, grades=(1,))

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        x = self.algebra.embed_vector(vectors).unsqueeze(1)
        return self.readout(self.norm(self.rotor(x)))
```
