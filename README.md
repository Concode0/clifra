# clifra

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Docs](https://img.shields.io/badge/docs-MkDocs-brightgreen)](https://concode0.github.io/clifra/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18939518.svg)](https://doi.org/10.5281/zenodo.18939518)

Layout-first Clifford algebra tools for PyTorch.

Clifra exposes one planner-owned algebra host. Full-lane tensors and compact
grade layouts share the same algebra, while planning builds static executors for
products, metrics, exponentials, actions, layers, and analysis utilities.

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

## Docs

> Please check [documentation milestone and status](https://github.com/Concode0/clifra/issues/21)

The docs are intentionally small: API pages come from live docstrings, and
`docs/core-design.md` keeps the process snippets.

See `docs/first-guide.md`, If you are a little confused or think you need
an intuitive understanding, please refer to the first guide and move on to the `docs/core-design.md` or `docs/api/`.

## Research

`research/continuum_solver` is a research engine built on clifra for fitting
stable, fast continuum deformations. It uses diffeomorphism-style coordinate
morphing over clifra charts and invertible bivector fields, inheriting clifra's
planned algebra execution.

The strict PGA metamaterial example in
`research/continuum_solver/examples/metamaterial_design.py` demonstrates a
difficult auxetic-to-positive phase target with reversible deformation paths.
The recorded validation run reached the target loss (`0.001632 < 0.01`) with
no validation failures, zero folded cells, and extremely small inverse-path
error (`inverse_rmse=4.10e-16`, `inverse_max_abs=2.00e-15`), producing VTK/GIF
artifacts under `outputs/metamaterial_design/`.

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
  version = {1.1.1},
  year    = {2026},
  doi     = {10.5281/zenodo.18939518},
  license = {Apache-2.0}
}
```

## References

### RotorGadget Layer Architectures
This project implements the equivariant layer architectures derived from the irreducible decomposition of Clifford multivectors. 
* *RotorGadget* implementations found in `clifra/layers/primitives/rotor_gadget.py` are based on:
  - Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers from Irreducibles." *arXiv:2507.11688*.

### Optimization on Manifolds
The core solvers and adaptive optimizers in `clifra/optimizers/` leverage Riemannian optimization techniques to handle the non-linear constraints of multivector rotors and versor transformations:
  - Absil, P.-A., Mahony, R., & Sepulchre, R. (2008). *Optimization Algorithms on Matrix Manifolds*. Princeton University Press.
  - Boumal, N. (2023). *An Introduction to Optimization on Smooth Manifolds*. Cambridge University Press.
