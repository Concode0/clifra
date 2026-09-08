# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.analysis.experimental import SignatureProbeAnalyzer, SignatureProbeResult, signature
from clifra.analysis.experimental.signature import _initialize_bivector_parameters, _pca_reduce, _SignatureProbe
from clifra.core.config import make_algebra

pytestmark = pytest.mark.unit


def estimate_for_width(data):
    return {
        "candidate_pq": (data.shape[1], 0),
        "neighbor_bivector_alignment": 0.7,
        "neighbor_bivector_dissimilarity": 0.2,
        "bivector_parameter_summary": {},
        "per_probe_results": [
            {"best_training_loss": -0.5, "neighbor_bivector_alignment": 0.7, "neighbor_bivector_dissimilarity": 0.2}
        ],
    }


@pytest.mark.parametrize("num_probes", [1, 3])
def test_probe_training_smoke_and_public_output(num_probes):
    torch.manual_seed(31)
    analyzer = SignatureProbeAnalyzer(num_probes=num_probes, probe_epochs=2, k=3)
    result = analyzer.analyze(torch.randn(9, 2))
    assert isinstance(result, SignatureProbeResult)
    p, q = result.candidate_pq
    assert all(isinstance(value, int) and value >= 0 for value in (p, q))
    assert p + q <= 2
    assert result.pca_output_width is None
    assert len(result.per_probe_results) == num_probes
    assert torch.isfinite(torch.tensor([entry["best_training_loss"] for entry in result.per_probe_results])).all()
    assert result.bivector_parameter_summary["bivector_square_signs"]


def test_double_precision_probe_training():
    result = SignatureProbeAnalyzer(dtype=torch.float64, num_probes=1, probe_epochs=1, k=3).analyze(torch.randn(8, 2))
    assert torch.isfinite(torch.tensor(result.neighbor_bivector_alignment))


@pytest.mark.parametrize("epochs", [1, 3])
def test_best_training_loss_matches_restored_parameters(epochs):
    torch.manual_seed(27)
    analyzer = SignatureProbeAnalyzer(dtype=torch.float64, probe_epochs=epochs, probe_lr=0.1, k=3)
    data, algebra = analyzer._quadratic_lift(torch.randn(8, 2, dtype=torch.float64))
    result = analyzer._train_probe(data, algebra)
    probe = result["probe"]
    neighborhood = signature.NeighborhoodBivectorAnalyzer(algebra, k=3)
    with torch.no_grad():
        output = probe(data).squeeze(1)
        alignment = neighborhood.neighbor_bivector_alignment(output)
        dissimilarity = neighborhood.neighbor_bivector_dissimilarity(output)
        loss = (
            -alignment
            + signature._DISSIMILARITY_WEIGHT * dissimilarity
            + signature._L1_WEIGHT * probe.rotor.parameter_l1_penalty().item()
        )
    assert result["best_training_loss"] == pytest.approx(loss, abs=1e-12)
    assert result["neighbor_bivector_alignment"] == alignment
    assert result["neighbor_bivector_dissimilarity"] == dissimilarity


@pytest.mark.parametrize("shape,cap,expected_width", [((12, 5), 2, 2), ((3, 10), 5, 3), ((8, 2), 5, None)])
def test_reduction_is_local_and_records_actual_width(monkeypatch, shape, cap, expected_width):
    analyzer = SignatureProbeAnalyzer(dtype=torch.float64, max_probe_features=cap)
    captured = []

    def fake(data):
        captured.append(data)
        return estimate_for_width(data)

    monkeypatch.setattr(analyzer, "_run_probes", fake)
    result = analyzer.analyze(torch.randn(shape))
    assert result.pca_output_width == expected_width
    assert captured[0].dtype == torch.float64
    assert captured[0].shape == (shape[0], expected_width or shape[1])
    if expected_width is not None:
        torch.testing.assert_close(
            captured[0].mean(dim=0), torch.zeros(expected_width, dtype=torch.float64), atol=1e-12, rtol=0
        )


def test_local_svd_preserves_centered_geometry_when_retained_rank_is_sufficient():
    t = torch.linspace(-2, 2, 12, dtype=torch.float64)
    data = torch.stack((t, t.square(), 2 * t, -t.square()), dim=1)
    projected = _pca_reduce(data, 2)
    centered = data - data.mean(dim=0)
    torch.testing.assert_close(projected @ projected.T, centered @ centered.T)
    assert _pca_reduce(torch.zeros(3, 8), 5).shape == (3, 3)
    assert _pca_reduce(torch.zeros(3, 8), 5).count_nonzero() == 0


def test_bootstrap_votes_representative_and_seeded_sampling(monkeypatch):
    analyzer = SignatureProbeAnalyzer()
    data = torch.arange(18, dtype=torch.float64).reshape(9, 2)
    tuples = [(2, 0), (1, 1), (1, 1)]
    seen, produced = [], []

    def fake(sample):
        seen.append(sample.clone())
        result = SignatureProbeResult(tuples[(len(seen) - 1) % 3], 0.5, 0.4, {})
        produced.append(result)
        return result

    monkeypatch.setattr(analyzer, "analyze", fake)
    winner, votes = analyzer.analyze_bootstrap(data, n_bootstrap=3, max_samples=5, seed=7)
    assert winner is produced[1]
    assert votes == {"candidate_pq_counts": {(2, 0): 1, (1, 1): 2}, "modal_fraction": 2 / 3, "n_bootstrap": 3}
    analyzer.analyze_bootstrap(data, n_bootstrap=3, max_samples=5, seed=7)
    for first, second in zip(seen[:3], seen[3:]):
        assert first.shape == (5, 2)
        assert first.dtype == data.dtype
        assert torch.equal(first, second)


def test_probe_budget_is_checked_before_constructing_host(monkeypatch):
    def no_host(*args, **kwargs):
        pytest.fail("oversized probe must be rejected before constructing the algebra")

    monkeypatch.setattr(signature, "make_algebra", no_host)
    with pytest.raises(ValueError, match="action matrix"):
        SignatureProbeAnalyzer(max_probe_features=11)._quadratic_lift(torch.randn(2, 11))


def test_bootstrap_mode_need_not_be_a_majority(monkeypatch):
    analyzer = SignatureProbeAnalyzer()
    candidates = [(2, 0), (1, 1), (1, 0)]
    results = [SignatureProbeResult(candidate, 0.5, 0.4, {}) for candidate in candidates]
    remaining = iter(results)
    monkeypatch.setattr(analyzer, "analyze", lambda data: next(remaining))
    representative, votes = analyzer.analyze_bootstrap(torch.ones(4, 2), n_bootstrap=3)
    assert representative is results[0]
    assert votes["candidate_pq_counts"] == dict.fromkeys(candidates, 1)
    assert votes["modal_fraction"] == 1 / 3


def test_quadratic_lift_uses_declared_layout_and_dtype():
    data = torch.tensor([[2.0, 3.0], [0.0, 1.0]], dtype=torch.float64)
    mv, algebra = SignatureProbeAnalyzer(dtype=torch.float64)._quadratic_lift(data)
    assert (algebra.p, algebra.q, algebra.r) == (3, 1, 0)
    assert mv.shape == (2, 1, 16) and mv.dtype == data.dtype
    expected = torch.cat((data, data.square().sum(dim=1, keepdim=True) / 2, torch.ones(2, 1, dtype=data.dtype)), dim=1)
    torch.testing.assert_close(algebra.layout((1,)).compact(mv[:, 0]), expected)


def test_private_probe_forward_and_biased_initialization():
    torch.manual_seed(12)
    algebra = make_algebra(3, 1)
    probe = _SignatureProbe(algebra, channels=2)
    assert all(not hasattr(parameter, "_manifold") for parameter in probe.parameters())
    data = torch.randn(5, 1, algebra.dim)
    assert probe(data).shape == data.shape
    _initialize_bivector_parameters(probe, algebra, "elliptic_weighted")
    squares = algebra._planner.bivector_squared_signs(device=algebra.device, dtype=algebra.dtype)
    energies = probe.rotor.bivector_parameters.detach().abs().mean(dim=0)
    assert energies[squares < 0].mean() > energies[squares > 0].mean()
    for bias in ("non_null_weighted", "uniform", "normal"):
        _initialize_bivector_parameters(probe, algebra, bias)
        assert torch.isfinite(probe.rotor.bivector_parameters).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_probes": 0},
        {"probe_epochs": 0},
        {"max_probe_features": 0},
        {"k": 0},
        {"probe_lr": float("nan")},
        {"bivector_parameter_energy_threshold": 2},
    ],
)
def test_invalid_probe_controls(kwargs):
    with pytest.raises(ValueError):
        SignatureProbeAnalyzer(**kwargs)


@pytest.mark.parametrize("kwargs", [{"n_bootstrap": 0}, {"max_samples": 0}])
def test_invalid_bootstrap_counts(kwargs):
    with pytest.raises(ValueError):
        SignatureProbeAnalyzer().analyze_bootstrap(torch.ones(3, 2), **kwargs)


def test_no_lower_level_estimator_or_generic_dimension_api():
    assert not hasattr(signature, "RotorProbeSignatureEstimator")
    analyzer = SignatureProbeAnalyzer()
    for name in ("estimate", "estimate_detailed", "analyze_with_confidence"):
        assert not hasattr(analyzer, name)
    with pytest.raises(TypeError):
        analyzer.analyze(torch.ones(4, 2), dim_result=None)


def test_probe_training_does_not_backpropagate_into_caller_representation():
    source = torch.randn(9, 5, requires_grad=True)
    representation = source.square()
    analyzer = SignatureProbeAnalyzer(max_probe_features=2, num_probes=1, probe_epochs=2, k=3)
    result = analyzer.analyze(representation)
    assert result.pca_output_width == 2
    assert source.grad is None
    representation.sum().backward()
    torch.testing.assert_close(source.grad, 2 * source.detach())
