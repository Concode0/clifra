# Put planned operations inside nn.Module

A `PlannedOperation` is an `nn.Module`. Assign it as an attribute to register
its execution buffers alongside ordinary parameters. Construct plans in
`__init__`; `forward` then composes tensor operations.

```python
import torch
from torch import nn
from clifra import CliffordModule, make_algebra

class ScalarPairing(nn.Module):
    def __init__(self, algebra):
        super().__init__()
        vector = algebra.layout((1,))
        self.weight = nn.Parameter(torch.ones(
            vector.dim, dtype=algebra.dtype, device=algebra.device
        ))
        self.product = algebra.plan_product(
            left=vector, right=vector, output=algebra.layout((0,))
        )

    def forward(self, values):
        return self.product(values, self.weight)

algebra = make_algebra(3)
pairing = ScalarPairing(algebra).to(dtype=torch.float64)
values = torch.tensor([[1., 2., 3.]], dtype=torch.float64)
torch.testing.assert_close(pairing(values), values.sum(-1, keepdim=True))
pairing(values).sum().backward()
torch.testing.assert_close(pairing.weight.grad, values.squeeze(0))
assert algebra.dtype == torch.float32
```

This module selects the scalar part of a vector product. In the Euclidean
signature above it equals a coefficient dot product; with a mixed signature
the same declaration computes the signed pairing. The parameter and loss
remain ordinary PyTorch tensors in either case.

Moving the module moves its parameters and registered plans without affecting
caller inputs or the original algebra. Plans execute and move independently, even when they originally shared cached execution buffers.

Use `CliffordModule` when the module also needs an algebra reference for future
construction or direct algebra calls:

```python
class PairingWithAlgebra(CliffordModule):
    def __init__(self, algebra):
        super().__init__(algebra)
        self.pairing = ScalarPairing(algebra)

    def forward(self, values):
        return self.pairing(values)

module = PairingWithAlgebra(algebra).to(dtype=torch.float64)
assert module.algebra.dtype == torch.float64
assert algebra.dtype == torch.float32
assert module.algebra.resource_limits is algebra.resource_limits
torch.testing.assert_close(module(values), values.sum(-1, keepdim=True))
```

`CliffordModule.to()` replaces its owned algebra reference with one using the
new placement defaults. It preserves the signature, registry, and resource
limits. An ordinary `nn.Module` is sufficient when all required operations are
already planned.

Save application parameters with normal PyTorch state dictionaries. Reconstruct
plans from signatures and contracts before loading; private execution buffers
are not a portable plan format across clifra versions.
