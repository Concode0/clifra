# clifra

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Docs](https://img.shields.io/badge/docs-MkDocs-brightgreen)](https://concode0.github.io/clifra/)

clifra is a PyTorch framework for Clifford geometric algebra neural networks.
It provides explicit algebra signatures, grade layouts, tensor products,
geometric layers, criteria, and Riemannian optimizers.

The repository is framework-only. Legacy task runners, datasets,
task-specific models, experiments, and synthetic demos have been removed so new
examples can be rebuilt around the current `clifra` API.

## Install

```bash
uv sync
uv sync --extra dev
```

Documentation dependencies are separate:

```bash
uv sync --group docs
uv run --group docs mkdocs serve
```

## Quickstart

```python
import torch

from clifra.core.config import make_algebra
from clifra.layers import CliffordLinear, GeometricGELU, RotorLayer

algebra = make_algebra(3, 0, kernel="dense", device="cpu")
x = torch.randn(8, 2, algebra.dim)

rotor = RotorLayer(algebra, channels=2)
linear = CliffordLinear(algebra, in_channels=2, out_channels=4)
activation = GeometricGELU(algebra, channels=4)

y = activation(linear(rotor(x)))
assert y.shape == (8, 4, algebra.dim)
```

## Verified Examples

```bash
uv run python docs/examples/quickstart.py
uv run python docs/examples/products_and_layouts.py
uv run python docs/examples/training_step.py
```

## Development

```bash
uv run pytest tests/ -m unit -q --tb=short
uv run pytest tests/ -m "not slow" -q --tb=short
uv run ruff check .
uv run ruff format .
uv run --group docs mkdocs build --strict
```

## Package Map

```text
clifra/
├── core/               # Algebra construction, layouts, planning, execution
├── criterion/          # Loss and regularizer modules
├── functional/         # Stateless products, activations, losses, helpers
├── layers/             # Neural primitives, blocks, and adapters
├── optimizers/         # Riemannian optimizers
└── utils/              # Compatibility helpers
benchmarks/             # Framework benchmark scripts
docs/                   # MkDocs documentation and runnable examples
tests/                  # Pytest coverage
```

## Documentation

The documentation is at
[concode0.github.io/clifra](https://concode0.github.io/clifra/). It is built
with MkDocs and mkdocstrings from the live API docstrings plus the checked
example scripts in `docs/examples/`.

## License

clifra is licensed under Apache License 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).

## Citation

```bibtex
@software{kim2026clifra,
  author  = {Kim, Eunkyum},
  title   = {clifra: Clifford Algebra Layers for PyTorch},
  url     = {https://github.com/Concode0/clifra},
  version = {1.0.0},
  year    = {2026},
  doi     = {10.5281/zenodo.18939519},
  license = {Apache-2.0}
}
```
