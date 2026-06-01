# clifra

clifra is a PyTorch package for tensors with Clifford algebra structure. It
keeps the algebra signature, grade layout, product route, and trainable module
boundaries visible so models can be built from small parts.

## Install

```bash
uv sync
uv sync --extra dev
```

For documentation work:

```bash
uv sync --group docs
uv run --group docs mkdocs serve
```

## Package Map

- `clifra.core`: algebra construction, layouts, planning, dense execution, and
  analysis helpers.
- `clifra.functional`: stateless products, activations, losses, and
  orthogonality helpers.
- `clifra.layers`: trainable primitives and reusable blocks.
- `clifra.criterion`: loss and regularizer modules.
- `clifra.optimizers`: Riemannian optimizers for tagged geometric parameters.

The repository no longer includes downstream task runners, datasets,
task-specific models, or old experiment scripts. New examples should import
from `clifra` and stay small enough to run on CPU.

## First Check

```python
import torch

from clifra.core.config import make_algebra
from clifra.layers import BladeSelector, CliffordLinear, GeometricGELU, RotorLayer

algebra = make_algebra(3, 0, kernel="dense", device="cpu")
x = torch.randn(8, 2, algebra.dim)

rotor = RotorLayer(algebra, channels=2)
linear = CliffordLinear(algebra, in_channels=2, out_channels=4)
activation = GeometricGELU(algebra, channels=4)
selector = BladeSelector(algebra, channels=4)

y = selector(activation(linear(rotor(x))))
assert y.shape == (8, 4, algebra.dim)
```

Run the checked examples in this documentation:

```bash
uv run python docs/examples/quickstart.py
uv run python docs/examples/products_and_layouts.py
uv run python docs/examples/training_step.py
```
