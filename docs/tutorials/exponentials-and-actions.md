# Bivector Exponentials and Induced Actions

A bivector exponential and a bivector-generated action are different outputs.
The first returns coefficients of $\exp(B)$. The second applies
$\exp(-B/2)x\exp(B/2)$ to an input without requiring the caller to construct
those multivectors. The sign and half-angle connect the two APIs. This example
uses compact coefficient tensors and reusable plans for both calculations.

## Materialize a planar exponential

In $Cl(2,0)$, $e_{12}^2=-1$, so
$\exp(t e_{12})=\cos t+e_{12}\sin t$. The result occupies scalar and
bivector lanes, even though the input has only one bivector lane.

```python
import torch
from clifra import make_algebra

algebra = make_algebra(2, 0, dtype=torch.float64)
vectors = algebra.layout((1,))
bivectors = algebra.layout((2,))
even = algebra.layout((0, 2))
exponential = algebra.plan_bivector_exp(input=bivectors, output=even)
theta = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
exp_b = exponential(theta)
torch.testing.assert_close(exp_b, torch.cat((theta.cos(), theta.sin())))
```

This trigonometric formula follows from the generator's square in this
particular signature. Other signatures can give hyperbolic or nilpotent
behavior. In larger algebras a general bivector exponential can contain higher
even grades; scalar plus bivector storage is not a universal rotor layout.

## Apply the corresponding rotation

For the action convention used here, a positive coefficient of $e_{12}$
rotates $e_1$ toward $e_2$. Form the rotor $R=\exp(-B/2)$ and its inverse
$\exp(B/2)$ to verify this explicitly.

```python
action = algebra.plan_versor_action(
    grade=2, input=vectors, parameter=bivectors, output=vectors,
)
sandwich = algebra.plan_sandwich_action(
    left=even, input=vectors, right=even, output=vectors,
)
x = torch.tensor([1.0, 0.0], dtype=torch.float64)
rotor = exponential(-theta / 2)
inverse_rotor = exponential(theta / 2)
explicit = sandwich(rotor, x, inverse_rotor)
induced = action(x, theta)
expected = torch.cat((theta.cos(), theta.sin()))
torch.testing.assert_close(induced, expected)
torch.testing.assert_close(explicit, induced)
```

The direct action returns compact vector coefficients. Its `grade=2` declares
the generator grade, not the output grade. Bivector conjugation preserves the
grade of the input, so a vector result does not require an intervening full
multivector tensor in user code.

Use the exponential when the rotor itself is needed, for example to compose or
inspect explicit versors. Use the induced action when the desired result is
the transformed geometry. Neither numerical route should be assumed to be
bitwise identical to the other; the comparison uses floating-point tolerances.

## Check geometry and gradients

```python
points = torch.tensor([[1.0, 2.0], [-0.5, 0.3]], dtype=torch.float64)
rotated = action(points, theta)
torch.testing.assert_close(rotated.square().sum(-1), points.square().sum(-1))
torch.testing.assert_close(action(rotated, -theta), points)

derivative, = torch.autograd.grad(induced[1], theta)
torch.testing.assert_close(derivative, theta.cos())
```

The length check is appropriate to this Euclidean signature. In an indefinite
signature the preserved quadratic form has metric signs and need not be a
positive loss. The inverse check uses the negative of the same generator;
adding arbitrary generators is not in general equivalent to composing their
actions, because bivectors need not commute.

[Fit a Rotation with PyTorch](learn-geometric-transform.md) uses these
generator coefficients as ordinary trainable parameters.
