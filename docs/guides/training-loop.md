# Training Loop

clifra layers are ordinary PyTorch modules. Use `make_riemannian_optimizer`
when a model has parameters tagged with manifold metadata, such as rotor
bivectors.

```python
import torch

from clifra.core.config import make_algebra
from clifra.criterion import GeometricMSELoss
from clifra.layers import CliffordLinear, RotorLayer
from clifra.optimizers import make_riemannian_optimizer

algebra = make_algebra(3, 0, kernel="dense", device="cpu")
model = torch.nn.Sequential(
    RotorLayer(algebra, channels=2),
    CliffordLinear(algebra, in_channels=2, out_channels=2),
)
criterion = GeometricMSELoss(algebra)
optimizer = make_riemannian_optimizer(model, algebra, lr=1e-2)

x = torch.randn(8, 2, algebra.dim)
target = torch.zeros_like(x)

optimizer.zero_grad()
loss = criterion(model(x), target)
loss.backward()
optimizer.step()

assert torch.isfinite(loss)
```

For parameters without manifold tags, the optimizer uses the Euclidean update
path. For rotor bivectors, it clips in the Lie algebra and leaves the exp map
to the forward pass.
