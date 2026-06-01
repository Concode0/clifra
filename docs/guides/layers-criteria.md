# Layers and Criteria

Layers own parameters. Functional modules stay stateless. Criteria wrap loss
formulas when PyTorch module state or a training-loop interface is useful.

## Common Layers

```python
import torch

from clifra.core.config import make_algebra
from clifra.layers import (
    BladeSelector,
    CliffordLayerNorm,
    CliffordLinear,
    GeometricGELU,
    RotorLayer,
)

algebra = make_algebra(3, 0, kernel="dense", device="cpu")
x = torch.randn(8, 2, algebra.dim)

rotor = RotorLayer(algebra, channels=2)
linear = CliffordLinear(algebra, in_channels=2, out_channels=4)
norm = CliffordLayerNorm(algebra, channels=4)
act = GeometricGELU(algebra, channels=4)
selector = BladeSelector(algebra, channels=4)

y = selector(act(norm(linear(rotor(x)))))
assert y.shape == (8, 4, algebra.dim)
```

## Criteria

```python
import torch

from clifra.core.config import make_algebra
from clifra.criterion import GeometricMSELoss

algebra = make_algebra(3, 0, kernel="dense", device="cpu")
criterion = GeometricMSELoss(algebra)

pred = torch.randn(8, 2, algebra.dim)
target = torch.zeros_like(pred)
loss = criterion(pred, target)

assert loss.ndim == 0
```

Use `clifra.functional` directly when no module state is needed.
