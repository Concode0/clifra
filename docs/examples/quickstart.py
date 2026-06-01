import torch

from clifra.core.config import make_algebra
from clifra.layers import BladeSelector, CliffordLinear, GeometricGELU, RotorLayer


def main() -> None:
    torch.manual_seed(0)
    algebra = make_algebra(3, 0, kernel="dense", device="cpu")

    rotor = RotorLayer(algebra, channels=2)
    mix = CliffordLinear(algebra, in_channels=2, out_channels=4)
    act = GeometricGELU(algebra, channels=4)
    select = BladeSelector(algebra, channels=4)

    points = torch.randn(16, 3)
    x = algebra.embed_vector(points).unsqueeze(1).repeat(1, 2, 1)
    y = select(act(mix(rotor(x))))

    assert y.shape == (16, 4, algebra.dim)
    assert torch.isfinite(y).all()
    print("quickstart ok", tuple(y.shape))


if __name__ == "__main__":
    main()
