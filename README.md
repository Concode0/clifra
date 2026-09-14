# clifra

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-MkDocs-brightgreen)](https://concode0.github.io/clifra/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18939518.svg)](https://doi.org/10.5281/zenodo.18939518)

clifra is a differentiable Clifford algebra computation layer for PyTorch.
Signatures and grade layouts describe ordinary coefficient tensors; operations
can be planned once and reused.

```bash
uv add clifra
```

```python
import torch
from clifra import make_algebra

algebra = make_algebra(3, 0)
vectors = algebra.layout((1,))
product = algebra.plan_product(
    left=vectors,
    right=vectors,
    output=algebra.layout((0, 2)),
)

left = torch.randn(8, vectors.dim, requires_grad=True)
right = torch.randn(vectors.dim)
out = product(left, right)  # [8, 4]: scalar, e12, e13, e23
out.square().mean().backward()
```

The inputs store three vector coefficients each. The product broadcasts over
the leading dimension and returns four scalar-and-bivector coefficients.

A signature defines basis products. A layout selects coefficient lanes, and a
tensor contract declares compact or canonical storage. Planned operations fix
these declarations while coefficient values and leading tensor dimensions
remain ordinary PyTorch inputs.

[Tutorials](https://concode0.github.io/clifra/tutorials/) · [How-to guides](https://concode0.github.io/clifra/how-to/) · [Explanations](https://concode0.github.io/clifra/explanations/) · [Research](https://concode0.github.io/clifra/research/) · [API reference](https://concode0.github.io/clifra/reference/)

## Development

```bash
uv sync --group dev
uv run --group dev ruff check .
uv run --group dev pytest tests/ -n12 -q --tb=short
uv run --group docs mkdocs build
```

Please open an issue before proposing a substantial change.
Contact: nemonanconcode@gmail.com.

## Citation

```bibtex
@software{kim2026clifra,
  author  = {Kim, Eunkyum},
  title   = {clifra: Layout-first Clifford algebra tools for PyTorch},
  url     = {https://github.com/Concode0/clifra},
  version = {2.0.1},
  year    = {2026},
  doi     = {10.5281/zenodo.18939518},
  license = {Apache-2.0}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
