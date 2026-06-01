# Quickstart

This page builds one small CPU model. It uses a dense algebra because `Cl(3,0)`
has only eight basis lanes.

```python
import torch

from clifra.core.config import make_algebra
from clifra.layers import BladeSelector, CliffordLinear, GeometricGELU, RotorLayer

algebra = make_algebra(3, 0, kernel="dense", device="cpu")

rotor = RotorLayer(algebra, channels=2)
mix = CliffordLinear(algebra, in_channels=2, out_channels=4)
act = GeometricGELU(algebra, channels=4)
select = BladeSelector(algebra, channels=4)

points = torch.randn(16, 3)
x = algebra.embed_vector(points).unsqueeze(1).repeat(1, 2, 1)
y = select(act(mix(rotor(x))))

assert y.shape == (16, 4, algebra.dim)
```

Use `make_algebra(p, q, r)` for direct construction:

- `p`: basis vectors with square `+1`
- `q`: basis vectors with square `-1`
- `r`: null basis vectors with square `0`

Use `kernel="dense"` when full multivectors are small. Use
`kernel="context"` when you want compact declared grade layouts instead of a
full `2 ** n` lane tensor.

## Next Steps

- Read [Algebra and Layouts](guides/algebra-layouts.md) for dense vs compact
  execution.
- Read [Layers and Criteria](guides/layers-criteria.md) for model components.
- Read [Training Loop](guides/training-loop.md) for optimizer setup.
