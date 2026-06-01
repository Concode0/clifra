import torch

from clifra.core.config import make_algebra
from clifra.functional import geometric_product, wedge
from clifra.layers import WedgeLayer


def dense_products() -> None:
    algebra = make_algebra(3, 0, kernel="dense", device="cpu")
    a = algebra.embed_vector(torch.tensor([[1.0, 0.0, 0.0]]))
    b = algebra.embed_vector(torch.tensor([[0.0, 1.0, 0.0]]))

    ab = geometric_product(algebra, a, b)
    area = wedge(algebra, a, b)

    assert ab.shape == (1, algebra.dim)
    assert area.shape == (1, algebra.dim)
    assert torch.isfinite(ab).all()
    assert torch.isfinite(area).all()


def compact_layout_product() -> None:
    algebra = make_algebra(6, 0, kernel="context", default_grades=(1,), device="cpu")
    vector_layout = algebra.layout((1,))
    bivector_layout = algebra.layout((2,))

    left = torch.randn(4, vector_layout.dim)
    right = torch.randn(4, vector_layout.dim)
    layer = WedgeLayer(
        algebra,
        left_layout=vector_layout,
        right_layout=vector_layout,
        output_layout=bivector_layout,
    )
    out = layer(left, right)

    assert out.shape == (4, bivector_layout.dim)
    assert torch.isfinite(out).all()


def main() -> None:
    torch.manual_seed(0)
    dense_products()
    compact_layout_product()
    print("products and layouts ok")


if __name__ == "__main__":
    main()
