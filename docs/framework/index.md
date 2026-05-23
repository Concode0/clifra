# Framework

clifra is organized around a small set of boundaries: algebra construction,
layout planning, pure formulas, trainable modules, and example embeddings.
Keeping those boundaries visible is what makes the framework usable beyond
low-dimensional dense Clifford algebra.

## Algebra and Layouts

An algebra is declared by its signature:

- `p`: basis vectors with square `+1`
- `q`: basis vectors with square `-1`
- `r`: null basis vectors with square `0`

The dense backend materializes full multivectors with `2 ** (p + q + r)` lanes.
The context backend keeps grade layouts and product plans compact so large
signatures do not force dense tensor allocation.

```python
from clifra.core.config import make_algebra

dense = make_algebra(3, 0, kernel="dense", device="cpu")
context = make_algebra(8, 0, kernel="context", default_grades=(1,))
```

Layouts are the contract between the core and layers. A layer can say it accepts
only vectors, only bivectors, or a selected grade range before it performs any
tensor operation.

## Execution Path

The framework separates planning from execution.

1. A caller declares operand grades or passes tensors with known layouts.
2. The planner resolves input, output, and active-lane contracts.
3. Executors gather the required basis lanes and reduce the planned products.
4. Layers wrap those operations with trainable parameters and PyTorch module
   state.

The same product functions are available through both object methods and the
stateless `clifra.functional.products` namespace.

## Layer Boundaries

`clifra.layers.primitives` is the trainable layer surface: rotors, reflections,
product layers, normalization, activation modules, and projections. Blocks such
as geometric attention and transformer layers compose these primitives.

Adapters are deliberately narrow. The framework does not depend on a large
adapter registry; `clifra.layers.adapters` now holds projective and conformal
embeddings as examples of how a domain-specific layout can be written on top of
the core.

## Mathematical Boundaries

`clifra.functional` is for pure formulas and stateless operations. It should not
own parameters, plotting, persistence, or framework state.

`clifra.criterion` wraps formulas as `nn.Module` losses or regularizers. This
keeps training-time objects separate from reusable math.

Orthogonality follows the same rule: pure masks, projections, and diagnostics
live in `clifra.functional.orthogonality`; module settings and trainable loss
integration live in `clifra.criterion.orthogonality`.
