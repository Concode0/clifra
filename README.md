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

## Research

`research/continuum_solver` is one repository-local system built on clifra. It
uses clifra charts, bivector fields, and planned algebra operations to study
continuum deformations. It is an example of the library's use as research
infrastructure, not a definition of clifra's scope.

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

### RotorGadget Background

The following paper is background reference material for `RotorGadget`. It is
not maintained as a normative implementation specification, and its inclusion
does not claim strict reproduction or conformance:

- Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers from Irreducibles." *arXiv:2507.11688*.

### Optimization Background

The tag-aware optimizers in `clifra/optimizers/` dispatch `spin`, `sphere`, and
`euclidean` post-update handling. The following books are background references,
not normative specifications for those implementations:

- Absil, P.-A., Mahony, R., & Sepulchre, R. (2008). *Optimization Algorithms on Matrix Manifolds*. Princeton University Press.
- Boumal, N. (2023). *An Introduction to Optimization on Smooth Manifolds*. Cambridge University Press.
