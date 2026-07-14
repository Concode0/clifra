# Train a Surface Projection

The following $Cl(3, 0)$ experiment trains a rotor action followed by a learned
lane gate to project a sampled surface toward the $e_1$–$e_2$ plane. The data
formula, model, loss, optimizer, robustness measurement, and surface plotting
helper are included. Familiarity with PyTorch training loops is assumed; every
clifra-specific object is introduced locally.

The surface is a grid patch:

```text
z = 0.5 * x * y
```

The grade-2 `VersorLayer` learns a bivector, exponentiates it to a rotor, and
applies the planned isometric action. `BladeSelector` then learns per-lane
gates. The selector makes this model a projection rather than an invertible
change of coordinates; a single global rotor cannot flatten a curved surface.

![Sampled surface before projection](../assets/first-guide/manifold_original.png)

![Projected surface representation](../assets/first-guide/manifold_latent.png)

## Imports

```python
import torch

from clifra import make_algebra
from clifra.core.foundation import CliffordModule
from clifra.layers import BladeSelector, VersorLayer
from clifra.optimizers import make_riemannian_optimizer
```

## Sample the Manifold

$Cl(3, 0)$ has eight canonical lanes. `embed_vector` places Cartesian coordinates
in the grade-1 subspace; `GradeLayout.compact` retrieves them without encoding
canonical lane positions in the example.

```python
def sample_patch(algebra, n: int = 24) -> tuple[torch.Tensor, tuple[int, int]]:
    x = torch.linspace(-1.0, 1.0, n, device=algebra.device, dtype=algebra.dtype)
    y = torch.linspace(-1.0, 1.0, n, device=algebra.device, dtype=algebra.dtype)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    Z = 0.5 * X * Y

    xyz = torch.stack((X.reshape(-1), Y.reshape(-1), Z.reshape(-1)), dim=-1)
    return algebra.embed_vector(xyz).unsqueeze(1), (n, n)  # [N, C=1, 8]


def vector_coordinates(algebra, values: torch.Tensor) -> torch.Tensor:
    """Gather Cartesian vector coordinates from canonical multivectors."""
    return algebra.layout((1,)).compact(values)


def coordinate_component(algebra, values: torch.Tensor, axis: int) -> torch.Tensor:
    """Return one Cartesian coordinate without using a basis-lane index."""
    return vector_coordinates(algebra, values).select(-1, int(axis))
```

## Build the Projection

A grade-2 rotor action is followed by a blade selector. The selector starts as
pass-through and learns which lanes to attenuate.

```python
class SurfaceProjection(CliffordModule):
    """Apply a rotor action and filter residual lane energy."""

    def __init__(self, algebra):
        super().__init__(algebra)
        full_layout = algebra.layout(range(algebra.n + 1))
        self.rotor = VersorLayer(
            algebra,
            channels=1,
            grade=2,
            input_layout=full_layout,
            output_layout=full_layout,
        )
        self.selector = BladeSelector(algebra, channels=1, layout=full_layout)

    def forward(self, x):
        return self.selector(self.rotor(x))

    def selector_penalty(self):
        return self.selector.weights.abs().mean()
```

## Define the Loss

The loss has two terms:

| Term | Purpose |
| --- | --- |
| `z_energy` | pushes the `e3` coordinate toward zero |
| `selector_deviation` | penalizes selector logits away from the pass-through value |

```python
def loss_terms(model, noisy):
    output = model(noisy)

    z = coordinate_component(model.algebra, output, axis=2)
    z_energy = z.square().mean()
    selector_deviation = model.selector_penalty()
    weighted_selector_deviation = 1.0e-3 * selector_deviation

    loss = z_energy + weighted_selector_deviation
    metrics = {
        "loss": loss.detach(),
        "z": z_energy.detach(),
        "selector_deviation": selector_deviation.detach(),
        "weighted_selector_deviation": weighted_selector_deviation.detach(),
    }
    return loss, output, metrics
```

## Train

```python
torch.manual_seed(7)

algebra = make_algebra(3, 0, device="cpu", dtype=torch.float32)
data, grid_shape = sample_patch(algebra)
model = SurfaceProjection(algebra)
optimizer = make_riemannian_optimizer(
    model,
    algebra,
    optimizer="adam",
    lr=0.03,
    max_bivector_norm=1.2,
)

history = []
for step in range(240):
    vector_noise = torch.randn_like(vector_coordinates(algebra, data)) * 0.01
    noisy = algebra.layout((1,)).full(vector_coordinates(algebra, data) + vector_noise)
    optimizer.zero_grad()
    loss, output, metrics = loss_terms(model, noisy)
    loss.backward()
    optimizer.step()
    history.append({"step": step, **{key: float(value) for key, value in metrics.items()}})

with torch.no_grad():
    projected = model(data)
```

## Inspect

```python
def measure(algebra, values, model=None):
    metrics = {
        "z": float(coordinate_component(algebra, values, axis=2).square().mean()),
    }
    if model is not None:
        metrics["selector_deviation"] = float(model.selector_penalty().detach())
    return {
        key: round(value, 6)
        for key, value in metrics.items()
    }


raw_metrics = measure(algebra, data)
projected_metrics = measure(algebra, projected, model)
print("raw", raw_metrics)
print("projected", projected_metrics)
```

Representative run:

```text
raw {'z': 0.032819}
projected {'z': 0.000601, 'selector_deviation': 0.327933}
```

Inspect the learned gate in vector coordinates rather than indexing canonical
lanes:

```python
with torch.no_grad():
    gates = 2.0 * torch.sigmoid(model.selector.weights)
    vector_gates = algebra.layout((1,)).compact(gates)

print("vector gates", vector_gates.squeeze())
```

The $e_3$ gate is smaller than the $e_1$ and $e_2$ gates. This is the
non-invertible step that removes most of the height variation.

## Noise Check

```python
def noise_test(model, data):
    rows = []
    for noise_std in [0.0, 0.01, 0.05, 0.1, 0.2]:
        coordinates = vector_coordinates(model.algebra, data)
        noisy_coordinates = coordinates + torch.randn_like(coordinates) * noise_std
        noisy = model.algebra.layout((1,)).full(noisy_coordinates)
        with torch.no_grad():
            output = model(noisy)
        rows.append({"noise": noise_std, **measure(model.algebra, output, model)})
    return rows


noise_rows = noise_test(model, data)
```

![Training metrics](../assets/first-guide/training_metrics.png)

![Noise robustness](../assets/first-guide/noise_robustness.png)

## Plot

The plotting code first gathers grade-1 coordinates through the layout. It does
not depend on the canonical positions of $e_1$, $e_2$, or $e_3$. Both surfaces
use the same coordinate bounds. Without shared bounds, Matplotlib expands the
small residual $e_3$ range in the projected result and makes it appear much
larger than it is.

```python
def xyz(algebra, values):
    coordinates = vector_coordinates(algebra, values).detach().cpu()
    if coordinates.ndim == 3:
        coordinates = coordinates.squeeze(-2)
    return coordinates.unbind(dim=-1)


def shared_bounds(algebra, *values):
    coordinates = torch.cat(
        [vector_coordinates(algebra, value).reshape(-1, 3) for value in values],
        dim=0,
    )
    lower = coordinates.amin(dim=0)
    upper = coordinates.amax(dim=0)
    padding = (upper - lower).clamp_min(1.0e-6) * 0.05
    return tuple(
        (float(lo - pad), float(hi + pad))
        for lo, hi, pad in zip(lower, upper, padding)
    )


def plot_patch(algebra, values, grid_shape, title, path, bounds):
    import matplotlib.pyplot as plt

    x, y, z = xyz(algebra, values)
    n0, n1 = grid_shape
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        x.reshape(n0, n1),
        y.reshape(n0, n1),
        z.reshape(n0, n1),
        cmap="viridis",
        norm=plt.Normalize(vmin=bounds[2][0], vmax=bounds[2][1]),
        linewidth=0,
        alpha=0.88,
    )
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_zlim(*bounds[2])
    ax.set_box_aspect(tuple(hi - lo for lo, hi in bounds))
    ax.view_init(elev=28, azim=-60)
    ax.set_title(title)
    ax.set_xlabel("e1 / x")
    ax.set_ylabel("e2 / y")
    ax.set_zlabel("e3 / z")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


bounds = shared_bounds(algebra, data, projected)
plot_patch(
    algebra,
    data,
    grid_shape,
    "Sampled surface",
    "surface-original.png",
    bounds,
)
plot_patch(
    algebra,
    projected,
    grid_shape,
    "Projected representation",
    "surface-projected.png",
    bounds,
)
```

## Experiment components

| Piece | Role |
| --- | --- |
| `make_algebra(3, 0)` | 3-D Euclidean Clifford algebra |
| `algebra.embed_vector(xyz)` | Formula samples into full-lane multivectors |
| `VersorLayer(..., grade=2)` | learned bivector and isometric rotor action |
| `BladeSelector` | learned lane gate that supplies the projection step |
| `coordinate_component(..., axis=2)` | z-energy pressure without a basis-lane literal |
| `model.selector_penalty()` | selector-logit regularization toward pass-through |

The selector can discard information, so the projected result is not an
invertible flattening or evidence of a change in the surface's intrinsic
geometry.
