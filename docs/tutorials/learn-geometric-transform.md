# Fit a Rotation with PyTorch

Fit a rotation in $Cl(3,0)$ from point correspondences. A bivector supplies
three generator coefficients, in the order $e_{12},e_{13},e_{23}$. The planned
action turns those coefficients into a length-preserving transformation;
PyTorch supplies the parameter, coordinate loss, and optimizer. This is one
use of differentiation through a planned operation: the training loop acts on
ordinary tensors and does not change the operation's contracts.

## Construct a correspondence problem

```python
import torch
from clifra import make_algebra

torch.manual_seed(7)
algebra = make_algebra(3, 0, dtype=torch.float64)
vectors = algebra.layout((1,))
bivectors = algebra.layout((2,))
action = algebra.plan_versor_action(
    grade=2, input=vectors, parameter=bivectors, output=vectors,
)
points = torch.randn(32, 3, dtype=torch.float64)
true_generator = torch.tensor([0.6, -0.35, 0.25], dtype=torch.float64)
with torch.no_grad():
    targets = action(points, true_generator)
```

The single generator broadcasts across all 32 points. This is a global
rotation about the origin: it has no translation, scaling, or deformation
parameters. The points span three dimensions, so the correspondences constrain
the rotation. A single vector would leave rotation about its own axis
undetermined.

## Fit the generator

```python
generator = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
optimizer = torch.optim.Adam([generator], lr=0.06)
initial_loss = (action(points, generator) - targets).square().mean().detach()

for step in range(240):
    optimizer.zero_grad()
    prediction = action(points, generator)
    loss = (prediction - targets).square().mean()
    loss.backward()
    optimizer.step()
```

Each optimizer step updates three ordinary real coefficients. Autograd
differentiates the induced action with respect to them; there is no need to
optimize a separate matrix and repair its orthogonality after each update.
The parameterization restricts the fit to the action's geometric family.

The loss is squared Euclidean distance between vector coordinates. That is a
suitable residual here because the signature is Euclidean and the point
correspondences are known. It is not a general prescription to minimize a
signature-sensitive quadratic form, which may be negative or vanish for a
nonzero residual.

## Validate the transformation

Check held-out points and a geometric invariant, rather than accepting only a
falling training loss.

```python
with torch.no_grad():
    fitted = action(points, generator)
    final_loss = (fitted - targets).square().mean()
    held_out = torch.randn(16, 3, dtype=torch.float64)
    expected = action(held_out, true_generator)
    actual = action(held_out, generator)
    assert final_loss < initial_loss * 1e-6
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual.square().sum(-1), held_out.square().sum(-1))
    torch.testing.assert_close(action(actual, -generator), held_out)
```

The recovered action matters more than exact recovery of the generator.
Exponential coordinates are not globally unique, and large rotations can
produce equivalent transformations from different coefficients. Starting near
zero and fitting a moderate rotation keeps this example in a simple local
regime; arbitrary initializations need not have the same optimization behavior.

A module can register both the generator and the planned action as attributes,
then use ordinary `model.parameters()` and `.to()`. No additional abstraction
is required for this three-parameter fit.
