# Optimize Clifford parameters

Clifford coefficients are ordinary PyTorch parameters. Use an ordinary
optimizer for unconstrained coefficients in any declared layout. No clifra
optimizer is required for differentiation.

When a coefficient-space bound is part of the chosen parameterization, apply it
explicitly after the optimizer step:

```python
import torch
from clifra import make_algebra
from clifra.optimizers import clip_coefficients_, normalize_coefficients_, post_update

algebra = make_algebra(2)
mixed = algebra.layout((0, 1, 2))
coefficients = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3, 0.4]))
assert coefficients.shape[-1] == mixed.dim
optimizer = torch.optim.Adam([coefficients], lr=0.01)
optimizer.zero_grad()
algebra.lane_energy(coefficients, input=mixed).sum().backward()
optimizer.step()
post_update(lambda: clip_coefficients_(coefficients, max_norm=0.25))
assert torch.linalg.vector_norm(coefficients) <= 0.25 + 1e-6

directions = torch.tensor([[3., 4.], [0., 0.]])
normalize_coefficients_(directions)
torch.testing.assert_close(directions, torch.tensor([[0.6, 0.8], [0., 0.]]))
```

These in-place helpers run without recording gradients and preserve tensor
identity. Their norm is the positive Euclidean coefficient norm; normalization
scales the coefficient vector to length one without establishing a signed unit
norm or rotor membership. Zero vectors remain zero. Inputs must be finite real
floating-point tensors. The helpers scale coefficients before taking a norm to
avoid overflow for large finite values.

`post_update` executes zero-argument callbacks in order after `optimizer.step()`
and outside any loss closure. It runs only the specified callbacks and does not discover parameters or modify optimizer moments itself. An exception leaves earlier adjustments applied.
`PostUpdateSGD(..., post_update=callback)` provides the same explicit callback
after each successful SGD step, including steps with no gradients. Reattach
the callback when restoring optimizer state; `differentiable=True` is unsupported.

## Update a stored rotor

For a unit rotor stored directly, the separate tangent helpers use canonical
full-lane tensors. They require matching shapes, dtype, and device, and an
algebra with that placement. The caller must establish
$\widetilde R R=1$; neither helper checks or repairs rotor membership.

```python
from clifra import TensorContract, make_algebra
from clifra.optimizers import exponential_update, project_to_rotor_tangent_space

algebra = make_algebra(3, dtype=torch.float64)
bivector = algebra.layout((2,))
full = TensorContract.canonical(algebra.layout())
rotor = torch.zeros(algebra.dim, dtype=torch.float64)
rotor[0] = 1
ambient = bivector.full(torch.tensor([0.02, 0., 0.], dtype=torch.float64))
tangent = project_to_rotor_tangent_space(rotor, ambient, algebra)
updated = exponential_update(rotor, tangent, algebra)
reverse = algebra.reverse(updated, input=full, output=full)
unit = algebra.geometric_product(reverse, updated, left=full, right=full, output=full)
torch.testing.assert_close(unit, rotor)
```

The projection is $R\langle\widetilde R V\rangle_2$, an idempotent map onto
tangents $RB$ under the unit-rotor assumption. It is not generally a Euclidean
orthogonal projection in mixed or degenerate signatures. The update is
$R\exp(\langle\widetilde R T\rangle_2)$. Supply the complete signed step in
$T$, including the learning rate and any half-angle convention, as the function
uses $T$ exactly as provided. Both functions are
differentiable, but they leave optimizer moments unchanged and do not define a general
Riemannian exponential for an independently chosen metric.
