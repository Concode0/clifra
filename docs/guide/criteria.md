# Criteria and Functional Formulas

The training surface is split into pure formulas and module wrappers.

## Pure Formulas

`clifra.functional` contains stateless functions. They should be safe to call
from modules, tests, or research scripts without carrying parameters or hidden
state.

```python
from clifra.functional import geometric_mse, grade_projection, wedge

loss_value = geometric_mse(prediction, target)
vectors = grade_projection(algebra, multivectors, grade=1)
area = wedge(algebra, u, v)
```

Activation formulas, product formulas, loss formulas, and orthogonality
diagnostics live here when they are pure tensor transformations.

## Criterion Modules

`clifra.criterion` wraps training losses and regularizers as PyTorch modules:

```python
from clifra.criterion import GeometricMSELoss, IsometryLoss, StrictOrthogonality

criterion = GeometricMSELoss(algebra)
regularizer = StrictOrthogonality(algebra)
```

Use criterion modules when the loss needs an algebra instance, cached masks,
settings, or a standard `nn.Module` interface.

## Orthogonality

Orthogonality tools are split by responsibility:

- `clifra.functional.orthogonality` contains masks, projections, grade energy,
  and numeric diagnostics.
- `clifra.criterion.orthogonality` contains `StrictOrthogonality` and its
  settings object for training loops.

Visualization and report formatting do not belong in these modules. Keep those
concerns in examples, notebooks, or docs components.
