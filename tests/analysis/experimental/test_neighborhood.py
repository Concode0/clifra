# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.analysis.experimental import NeighborhoodBivectorAnalyzer, compare_coordinate_lifts
from clifra.core.config import make_algebra

pytestmark = pytest.mark.unit

_DEVICES = (
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) + (["mps"] if torch.backends.mps.is_available() else [])
)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("n,count", [(3, 8), (3, 1), (1, 5)])
def test_neighborhood_normalizes_to_algebra_including_small_inputs(device, n, count):
    algebra = make_algebra(n, device=device, dtype=torch.float32)
    data = torch.randn(count, algebra.dim, dtype=torch.float64)
    analyzer = NeighborhoodBivectorAnalyzer(algebra, k=3)
    normalized = data.to(device=algebra.device, dtype=algebra.dtype)
    for name in (
        "mean_neighbor_bivectors",
        "per_point_bivector_alignment",
        "_bivector_alignment_tensor",
        "_bivector_dissimilarity_tensor",
    ):
        actual = getattr(analyzer, name)(data)
        assert actual.device == normalized.device
        assert actual.dtype == algebra.dtype
        torch.testing.assert_close(actual, getattr(analyzer, name)(normalized))


def test_neighborhood_normalization_preserves_gradients():
    torch.manual_seed(18)
    algebra = make_algebra(3, dtype=torch.float64)
    data = torch.randn(8, algebra.dim, requires_grad=True)
    score = NeighborhoodBivectorAnalyzer(algebra, k=3)._bivector_alignment_tensor(data)
    assert score.dtype == algebra.dtype
    score.backward()
    assert data.grad is not None and torch.isfinite(data.grad).all()
    assert data.grad.abs().sum() > 0


@pytest.mark.parametrize("data", [torch.zeros(2, 1, 8), torch.zeros(0, 8), torch.ones(3, 8, dtype=torch.int64)])
def test_neighborhood_normalization_does_not_relax_input_contract(data):
    with pytest.raises((ValueError, TypeError)):
        NeighborhoodBivectorAnalyzer(make_algebra(3)).mean_neighbor_bivectors(data)


def circle():
    angles = torch.arange(32, dtype=torch.float64) * (2 * torch.pi / 32)
    return torch.stack((angles.cos(), angles.sin(), torch.zeros_like(angles)), dim=-1)


def test_planar_alignment_and_grade_two_connections():
    algebra = make_algebra(3, dtype=torch.float64)
    mv = algebra.layout((1,)).full(circle())
    flow = NeighborhoodBivectorAnalyzer(algebra, k=4)
    connections = flow._normalized_neighbor_bivectors(mv)
    assert connections.shape == (32, 4, algebra.dim)
    non_bivectors = [i for i in range(algebra.dim) if i.bit_count() != 2]
    assert connections[..., non_bivectors].count_nonzero() == 0
    assert flow.mean_neighbor_bivectors(mv).shape == mv.shape
    assert flow.neighbor_bivector_alignment(mv) == pytest.approx(1)
    torch.testing.assert_close(flow.per_point_bivector_alignment(mv), torch.ones(32, dtype=mv.dtype))
    assert flow.neighbor_bivector_dissimilarity(mv) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("n,count", [(3, 1), (1, 5), (3, 2)])
def test_small_neighborhoods_remain_finite(n, count):
    algebra = make_algebra(n, dtype=torch.float64)
    mv = algebra.layout((1,)).full(torch.randn(count, n, dtype=torch.float64))
    flow = NeighborhoodBivectorAnalyzer(algebra, k=8)
    for value in (
        flow._bivector_alignment_tensor(mv),
        flow._bivector_dissimilarity_tensor(mv),
        flow.per_point_bivector_alignment(mv),
        flow.mean_neighbor_bivectors(mv),
    ):
        assert torch.isfinite(value).all()
        assert value.dtype == mv.dtype


@pytest.mark.parametrize("method", ["_bivector_alignment_tensor", "_bivector_dissimilarity_tensor"])
def test_scores_backpropagate_to_coordinates(method):
    torch.manual_seed(16)
    algebra = make_algebra(3, dtype=torch.float64)
    data = torch.randn(12, 3, dtype=torch.float64, requires_grad=True)
    mv = algebra.layout((1,)).full(data)
    score = getattr(NeighborhoodBivectorAnalyzer(algebra, k=4), method)(mv)
    score.backward()
    assert data.grad is not None and torch.isfinite(data.grad).all()
    assert data.grad.abs().sum() > 0


def test_scalar_alignment_reduces_per_point_scores():
    algebra = make_algebra(3)
    mv = algebra.layout((1,)).full(torch.randn(12, 3))
    flow = NeighborhoodBivectorAnalyzer(algebra, k=5)
    assert flow.neighbor_bivector_alignment(mv) == pytest.approx(flow.per_point_bivector_alignment(mv).mean().item())


def test_dissimilarity_uses_first_neighbor_neighborhood():
    torch.manual_seed(17)
    algebra = make_algebra(3)
    mv = algebra.layout((1,)).full(torch.randn(9, 3))
    flow = NeighborhoodBivectorAnalyzer(algebra, k=3)
    neighbors = flow._knn(mv)
    connections = flow._normalized_neighbor_bivectors(mv, compact_output=True)
    total = []
    for i in range(len(mv)):
        j = neighbors[i, 0]
        total.append((connections[i] @ connections[j].T).abs().mean())
    expected = 1 - torch.stack(total).mean()
    assert flow.neighbor_bivector_dissimilarity(mv) == pytest.approx(expected.item(), abs=1e-6)


@pytest.mark.parametrize("positive,negative", [(3, 0), (2, 1)])
def test_lift_comparison_matches_explicit_embeddings(positive, negative):
    data = circle()
    results = compare_coordinate_lifts(data, positive, negative, k=4)
    assert set(results) == {"original", "positive_count_extension", "negative_count_extension"}
    for key, p, q, fill in [
        ("original", positive, negative, None),
        ("positive_count_extension", positive + 1, negative, 1.0),
        ("negative_count_extension", positive, negative + 1, 1.0),
    ]:
        algebra = make_algebra(p, q, dtype=data.dtype, device=data.device)
        if fill is None:
            coordinates = data
        elif key == "positive_count_extension":
            coordinates = torch.cat([data[:, :positive], data.new_ones(len(data), 1), data[:, positive:]], dim=1)
        else:
            coordinates = torch.cat([data, data.new_ones(len(data), 1)], dim=1)
        flow = NeighborhoodBivectorAnalyzer(algebra, k=4)
        mv = algebra.layout((1,)).full(coordinates)
        assert results[key] == {
            "algebra_signature": (p, q),
            "neighbor_bivector_alignment": flow.neighbor_bivector_alignment(mv),
            "neighbor_bivector_dissimilarity": flow.neighbor_bivector_dissimilarity(mv),
        }


@pytest.mark.parametrize(
    "data", [torch.zeros(0, 3), torch.zeros(2, 4), torch.zeros(3), torch.ones(2, 3, dtype=torch.int64)]
)
def test_invalid_lift_input(data):
    with pytest.raises((ValueError, TypeError)):
        compare_coordinate_lifts(data, 3, 0)


@pytest.mark.parametrize("appended_value", [1.0, 0.0, -2.5])
def test_coordinate_lifts_preserve_original_quadratic_form(monkeypatch, appended_value):
    data = torch.tensor([[2.0, 3.0, 5.0], [1.0, 4.0, 6.0]], dtype=torch.float64)
    squares = []

    def record_square(self, mv):
        layout = self.algebra.layout()
        squares.append(self.algebra.geometric_product(mv, mv, left=layout, right=layout, output=layout)[:, 0])
        return 0.0

    monkeypatch.setattr(NeighborhoodBivectorAnalyzer, "neighbor_bivector_alignment", record_square)
    monkeypatch.setattr(NeighborhoodBivectorAnalyzer, "neighbor_bivector_dissimilarity", lambda self, mv: 0.0)
    compare_coordinate_lifts(data, 1, 2, appended_value=appended_value)
    expected = data[:, 0].square() - data[:, 1:].square().sum(dim=1)
    torch.testing.assert_close(squares[0], expected)
    torch.testing.assert_close(squares[1], expected + appended_value**2)
    torch.testing.assert_close(squares[2], expected - appended_value**2)


def test_removed_neighborhood_conveniences():
    flow = NeighborhoodBivectorAnalyzer(make_algebra(3))
    for name in (
        "_embed",
        "alignment_threshold_report",
        "_random_connection_alignment_baseline",
        "approximate_bivector_interpolation",
    ):
        assert not hasattr(flow, name)
