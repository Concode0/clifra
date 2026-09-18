# Differentiate a Spatial Deformation

A global rotor moves every point by the same rotation. A field of bivectors
can rotate different parts of a surface by different amounts. Here we sample
an elliptical cylinder, twist each cross-section around its axis, and learn
the angle field from sparse point correspondences. The example composes a
clifra action with a polynomial evaluated in PyTorch; both computations remain
in the same autograd graph.

The field is evaluated on persistent material coordinates: each point keeps
its original height label throughout the calculation. This distinction matters
when inverting a learned deformation. Negating the generator inverts the local
action only if it is the same generator attached to the same point.

## Sample and twist a surface

```python
import math
import torch
from clifra import make_algebra

torch.manual_seed(7)
algebra = make_algebra(3, 0, dtype=torch.float64)
vectors = algebra.layout((1,))
bivectors = algebra.layout((2,))
action = algebra.plan_versor_action(
    grade=2, input=vectors, parameter=bivectors, output=vectors,
)
heights = torch.linspace(-1, 1, 25, dtype=torch.float64)
angles = torch.arange(40, dtype=torch.float64) * (2 * math.pi / 40)
z, phi = torch.meshgrid(heights, angles, indexing="ij")
surface = torch.stack((phi.cos(), 0.5 * phi.sin(), z), dim=-1)
features = torch.stack((torch.ones_like(z), z, z.square(), z.pow(3)), dim=-1)
true_coefficients = torch.tensor([0.15, 0.8, -0.25, 0.35], dtype=torch.float64)

def generator_field(coefficients):
    angle = features @ coefficients
    zeros = torch.zeros_like(angle)
    return torch.stack((angle, zeros, zeros), dim=-1)

with torch.no_grad():
    target = action(surface, generator_field(true_coefficients))
assert surface.shape == target.shape == (25, 40, 3)
```

Only the $e_{12}$ lane is nonzero, so every action rotates the horizontal plane
and leaves $e_3$ unchanged. The ellipse makes its orientation visible, unlike
an unmarked circular cross-section. The angle varies cubically with height;
four real coefficients describe the entire field.

The two leading tensor axes index height and circumference. The generator
field has matching leading axes, giving one bivector per surface point. All
points use the same planned operation, with no per-point planning.

## Learn from sparse correspondences

Observe five height rings and eight points on each ring. These measurements
constrain a cubic field, while the unobserved points test its interpolation.
This is a correspondence problem: target points have known material labels.
An unordered surface-matching objective would require a different loss and
would introduce additional ambiguities.

```python
observed = torch.zeros(z.shape, dtype=torch.bool)
observed[::6, ::5] = True
coefficients = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
optimizer = torch.optim.Adam([coefficients], lr=0.05)

for step in range(500):
    optimizer.zero_grad()
    prediction = action(surface[observed], generator_field(coefficients)[observed])
    loss = (prediction - target[observed]).square().mean()
    loss.backward()
    optimizer.step()
```

The matrix multiplication that evaluates the polynomial is ordinary PyTorch.
clifra is responsible for the meaning of the resulting bivectors and their
action on vectors. The fixed low-degree basis provides enough smoothness for this example, so no separate regularization term is used. A more flexible field
would need enough observations or a suitable prior to determine behavior
between samples.

## Check interpolation and inversion

```python
with torch.no_grad():
    learned_field = generator_field(coefficients)
    deformed = action(surface, learned_field)
    unseen_error = (deformed[~observed] - target[~observed]).square().mean()
    assert unseen_error < 1e-8
    torch.testing.assert_close(deformed[..., 2], surface[..., 2])
    torch.testing.assert_close(
        deformed[..., :2].square().sum(-1),
        surface[..., :2].square().sum(-1),
    )
    restored = action(deformed, -learned_field)
    torch.testing.assert_close(restored, surface)
```

Each local action preserves radius and height. It does not follow that a
spatially varying field preserves the distance between arbitrary pairs of
points: neighboring rings rotate by different angles. This deformation twists
the surface while each individual vector undergoes an orthogonal action.

The inverse above reuses `learned_field` with its sign reversed. For this field,
height is unchanged, so reevaluating the polynomial at the deformed height
would also work. A field depending on coordinates changed by its own action
would require more care; pointwise invertibility alone does not establish a
globally invertible deformation.

## Inspect the geometry

The following optional block requires Matplotlib. Run it after the preceding
blocks to compare the original surface, sparse observations, and the fitted
field. The assertions above are independent of plotting.

```python title="Optional plotting"
import matplotlib.pyplot as plt

figure = plt.figure(figsize=(12, 4))
for column, (title, values) in enumerate(
    (("Original", surface), ("Target", target), ("Learned field", deformed)), start=1,
):
    axes = figure.add_subplot(1, 3, column, projection="3d")
    xyz = values.detach().cpu().numpy()
    axes.plot_wireframe(xyz[..., 0], xyz[..., 1], xyz[..., 2], rstride=2, cstride=4)
    if column == 2:
        samples = target[observed].cpu().numpy()
        axes.scatter(samples[:, 0], samples[:, 1], samples[:, 2], color="black", s=8)
    axes.set(title=title, xlabel="e1", ylabel="e2", zlabel="e3")
    axes.set_box_aspect((2, 2, 2))
figure.tight_layout()
plt.show()
```
