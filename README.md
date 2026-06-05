# clifra

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Docs](https://img.shields.io/badge/docs-MkDocs-brightgreen)](https://concode0.github.io/clifra/)

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

The docs are intentionally small: API pages come from live docstrings, and
`docs/core-design.md` keeps the process snippets.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

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
