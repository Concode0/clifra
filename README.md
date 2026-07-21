# clifra

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Docs](https://img.shields.io/badge/docs-MkDocs-brightgreen)](https://concode0.github.io/clifra/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18939518.svg)](https://doi.org/10.5281/zenodo.18939518)

Layout-first Clifford algebra tools for PyTorch.

A clifra algebra host owns its layouts, policies, and operation plans. Full-lane
tensors and compact grade layouts share that algebra, while planning builds
static executors for products, metrics, exponentials, and actions. Layers and
other library components reuse the same operations.

## Install

```bash
uv sync
uv sync --extra dev
```

Docs:

```bash
uv sync --group docs
uv run --group docs mkdocs serve
```

## Minimal Use

```python
import torch

from clifra import make_algebra

algebra = make_algebra(3, 0, device="cpu")
vectors = algebra.layout((1,))
products = algebra.plan_product(
    op="gp",
    left_layout=vectors,
    right_layout=vectors,
    output_layout=algebra.layout((0, 2)),
)

left = torch.randn(8, vectors.dim)
right = torch.randn(8, vectors.dim)
out = products(left, right)
```

## Checks

```bash
uv run pytest tests/ -m unit -q --tb=short
uv run pytest tests/ -m "not slow" -q --tb=short
uv run ruff check .
uv run --group docs mkdocs build
```

See the [documentation](https://concode0.github.io/clifra/) for tutorials,
explanations, benchmarks, and the generated API reference.

## Research Showcase: Clifford Transformation Fields

`research/continuum_solver` is Clifra's clearest end-to-end demonstration of
layout-directed geometric learning. It turns sampled bivector generators into
differentiable fields of local Clifford actions. Coordinate values and
persistent sample labels remain distinct, while samplers, action paths, and
differentiable objectives are independently configurable.

[Bivector field basics](research/continuum_solver/examples/bivector_field_basics.py)
is the compact introduction. It learns a coordinate-dependent action on
unordered points and demonstrates metric preservation, labeled inversion, and
permutation equivariance.

```bash
uv run research/continuum_solver/examples/bivector_field_basics.py
```

[Physics-informed deformation design](research/continuum_solver/examples/physics_informed_deformation_design.py)
drives the same transformation-field mechanism with material mechanics,
boundary conditions, guarded optimization, strict validation, and
visualization. It demonstrates a complete scientific system built from the
general field construction.

```bash
uv run --group viz research/continuum_solver/examples/physics_informed_deformation_design.py
```

Clifra compiles declared geometric coordinates into differentiable actions. The
continuum solver assembles those actions into trainable fields, and applications
decide what those fields learn through their objectives and constraints.

See [Why bivector coordinate fields work](https://concode0.github.io/clifra/explanations/transformation-fields/)
for the derivation, field semantics, and inversion model.

## Contribution

Found a problem or want to propose a change? Please open an Issue first,
especially before a PR, so the scope is clear.

For direct contact, email: nemonanconcode@gmail.com

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Citation

```bibtex
@software{kim2026clifra,
  author  = {Kim, Eunkyum},
  title   = {clifra: Clifford Algebra Layers for PyTorch},
  url     = {https://github.com/Concode0/clifra},
  version = {1.2.0},
  year    = {2026},
  doi     = {10.5281/zenodo.18939518},
  license = {Apache-2.0}
}
```

## References

These works provide conceptual background for `RotorGadget` and the tag-aware
optimizers. Clifra's behavior is defined by its public API, source, and tests.

### RotorGadget Background

- Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers from Irreducibles." *arXiv:2507.11688*.

### Optimization Background

The tag-aware optimizers in `clifra/optimizers/` dispatch `spin`, `sphere`, and
`euclidean` post-update handling.

- Absil, P.-A., Mahony, R., & Sepulchre, R. (2008). *Optimization Algorithms on Matrix Manifolds*. Princeton University Press.
- Boumal, N. (2023). *An Introduction to Optimization on Smooth Manifolds*. Cambridge University Press.
