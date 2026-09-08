# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields

import pytest
import torch

from clifra.analysis import CommutatorAnalyzer, SpectralAnalyzer, TransformationDiagnosticsAnalyzer
from clifra.core.config import make_algebra

pytestmark = pytest.mark.unit

_DEVICES = (
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) + (["mps"] if torch.backends.mps.is_available() else [])
)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("analyzer_type", [SpectralAnalyzer, CommutatorAnalyzer, TransformationDiagnosticsAnalyzer])
def test_observations_use_algebra_dtype_and_device(device, analyzer_type):
    algebra = make_algebra(3, device=device, dtype=torch.float32)
    data = torch.randn(7, algebra.dim, dtype=torch.float64)
    analyzer = analyzer_type(algebra)
    result = analyzer.analyze(data)
    normalized = data.to(device=algebra.device, dtype=algebra.dtype)
    expected = analyzer.analyze(normalized)
    for item in fields(result):
        actual_value = getattr(result, item.name)
        expected_value = getattr(expected, item.name)
        if isinstance(actual_value, torch.Tensor):
            assert actual_value.dtype == algebra.dtype
            assert actual_value.device == normalized.device
            torch.testing.assert_close(actual_value, expected_value)
        else:
            assert actual_value == expected_value
        if hasattr(analyzer, item.name):
            direct = getattr(analyzer, item.name)(data)
            if isinstance(direct, torch.Tensor):
                torch.testing.assert_close(direct, expected_value)
            else:
                assert direct == expected_value


def test_observation_dtype_conversion_preserves_gradients():
    algebra = make_algebra(3, dtype=torch.float64)
    data = torch.randn(5, algebra.dim, requires_grad=True)
    SpectralAnalyzer(algebra).grade_coefficient_energy(data).sum().backward()
    torch.testing.assert_close(data.grad, 2 * data.detach() / len(data))


@pytest.fixture(params=[(3, 0, 0), (2, 1, 0), (2, 0, 1)])
def algebra(request):
    return make_algebra(*request.param, dtype=torch.float64)


@pytest.mark.parametrize("analyzer_type", [SpectralAnalyzer, CommutatorAnalyzer, TransformationDiagnosticsAnalyzer])
def test_channels_require_explicit_caller_aggregation(algebra, analyzer_type):
    data = torch.randn(12, 3, algebra.dim, dtype=torch.float64)
    analyzer = analyzer_type(algebra)
    with pytest.raises(ValueError, match="canonical"):
        analyzer.analyze(data)
    analyzer.analyze(data.mean(dim=1))


@pytest.mark.parametrize(
    "analyzer_type,method",
    [
        (SpectralAnalyzer, "grade_coefficient_energy"),
        (SpectralAnalyzer, "mean_bivector"),
        (SpectralAnalyzer, "left_multiplication_eigenvalue_magnitudes"),
        (CommutatorAnalyzer, "vector_pair_commutator_norms"),
        (CommutatorAnalyzer, "adjoint_eigenvalue_magnitudes"),
        (CommutatorAnalyzer, "mean_commutator_norm"),
        (CommutatorAnalyzer, "basis_bivector_commutator_norm_ratios"),
        (TransformationDiagnosticsAnalyzer, "vector_coefficient_energy"),
        (TransformationDiagnosticsAnalyzer, "odd_grade_energy_fraction"),
        (TransformationDiagnosticsAnalyzer, "basis_reflection_marginal_scores"),
    ],
)
def test_direct_measurements_reject_channel_input(analyzer_type, method):
    analyzer = analyzer_type(make_algebra(3))
    with pytest.raises(ValueError, match="canonical"):
        getattr(analyzer, method)(torch.ones(4, 1, 8))


@pytest.mark.parametrize("analyzer_type", [SpectralAnalyzer, CommutatorAnalyzer, TransformationDiagnosticsAnalyzer])
@pytest.mark.parametrize("shape", [(0, 8), (2, 0, 8), (3, 3), (8,), (2, 2, 2, 8)])
def test_rejects_noncanonical_or_empty_input(analyzer_type, shape):
    with pytest.raises(ValueError):
        analyzer_type(make_algebra(3)).analyze(torch.zeros(shape))


@pytest.mark.parametrize("analyzer_type", [SpectralAnalyzer, CommutatorAnalyzer, TransformationDiagnosticsAnalyzer])
def test_requires_declared_algebra_and_floating_data(analyzer_type):
    with pytest.raises(TypeError, match="AlgebraContext"):
        analyzer_type(None).analyze(torch.zeros(3, 8))
    with pytest.raises(TypeError, match="floating"):
        analyzer_type(make_algebra(3)).analyze(torch.zeros(3, 8, dtype=torch.int64))


def test_grade_energy_uses_observation_energy_not_energy_of_population_mean(algebra):
    data = torch.zeros(2, algebra.dim, dtype=torch.float64)
    data[:, 1] = torch.tensor([2.0, -2.0])
    data[:, 3] = 3
    torch.testing.assert_close(SpectralAnalyzer(algebra).grade_coefficient_energy(data), data.new_tensor([0, 4, 9, 0]))
    opposite_channels = torch.stack([data, -data], dim=1)
    assert (
        SpectralAnalyzer(algebra).analyze(opposite_channels.mean(dim=1)).grade_coefficient_energy.count_nonzero() == 0
    )


def test_mean_bivector_preserves_tiny_coefficients(algebra):
    data = torch.zeros(4, algebra.dim, dtype=torch.float64)
    data[:, 0] = 5
    data[:, 3] = 1e-20
    result = SpectralAnalyzer(algebra).analyze(data)
    expected = torch.zeros_like(data[0])
    expected[3] = 1e-20
    assert torch.equal(result.mean_bivector, expected)
    assert result.mean_bivector.shape == (algebra.dim,)


def test_spectrum_uses_complete_mean_and_does_not_consume_rng(algebra):
    data = torch.zeros(100, algebra.dim, dtype=torch.float64)
    data[:70, 0], data[70:, 0] = 1, 10
    analyzer = SpectralAnalyzer(algebra)
    state = torch.random.get_rng_state().clone()
    result = analyzer.analyze(data)
    assert torch.equal(state, torch.random.get_rng_state())
    expected = torch.full((algebra.dim,), 3.7, dtype=data.dtype)
    torch.testing.assert_close(result.left_multiplication_eigenvalue_magnitudes, expected)
    torch.testing.assert_close(analyzer.left_multiplication_eigenvalue_magnitudes(data), expected)


def test_commutator_measurements_have_distinct_fixed_meanings():
    algebra = make_algebra(3, dtype=torch.float64)
    analyzer = CommutatorAnalyzer(algebra)
    data = algebra.layout((1,)).full(torch.tensor([[1.0, 0, 0], [0, 1.0, 0]], dtype=torch.float64))
    assert analyzer.mean_commutator_norm(data) == pytest.approx(1)
    assert analyzer.vector_pair_commutator_norms(data).count_nonzero() == 0
    combined = algebra.layout((1,)).full(data.new_tensor([[1, 2, 0]]))
    pairs = analyzer.vector_pair_commutator_norms(combined)
    assert pairs[0, 1] == pairs[1, 0] == 4
    assert pairs.count_nonzero() == 2
    e1 = algebra.layout((1,)).full(data.new_tensor([[1, 0, 0]]))
    torch.testing.assert_close(analyzer.basis_bivector_commutator_norm_ratios(e1), data.new_tensor([2, 2, 0]))
    torch.testing.assert_close(analyzer.adjoint_eigenvalue_magnitudes(e1), data.new_tensor([2, 2, 2, 2, 0, 0, 0, 0]))


def test_mean_commutator_includes_nonvector_grades(algebra):
    data = torch.zeros(2, algebra.dim, dtype=torch.float64)
    data[0, 3], data[1, 5] = 1, 1
    analyzer = CommutatorAnalyzer(algebra)
    assert analyzer.vector_pair_commutator_norms(data).count_nonzero() == 0
    assert analyzer.mean_commutator_norm(data) == pytest.approx(1)


def test_explicit_closure_of_closed_and_nonclosed_spans():
    analyzer = CommutatorAnalyzer(make_algebra(3, dtype=torch.float64))
    closed = analyzer.bivector_bracket_closure([6, 3, 5])
    assert closed["blade_indices"] == [6, 3, 5]
    assert closed["mean_relative_closure_residual"] == 0
    structure = closed["projected_bracket_coefficients"]
    torch.testing.assert_close(structure, -structure.transpose(0, 1))
    assert structure.abs().max() == 2
    assert analyzer.bivector_bracket_closure([3, 5])["mean_relative_closure_residual"] == 1
    assert analyzer.bivector_bracket_closure([])["projected_bracket_coefficients"].shape == (0, 0, 0)
    assert analyzer.bivector_bracket_closure([3])["projected_bracket_coefficients"].shape == (1, 1, 1)


@pytest.mark.parametrize("indices", [[3, 3], [0], [1], [8], [True], [3.0]])
def test_closure_rejects_invalid_indices(indices):
    with pytest.raises(ValueError):
        CommutatorAnalyzer(make_algebra(3)).bivector_bracket_closure(indices)


def test_closure_rejects_large_span_without_selection():
    algebra = make_algebra(7)
    with pytest.raises(ValueError, match="at most 15"):
        CommutatorAnalyzer(algebra).bivector_bracket_closure(algebra.layout((2,)).basis_indices)


def test_vector_energy_and_odd_fraction_have_no_threshold_classification(algebra):
    data = torch.zeros(2, algebra.dim, dtype=torch.float64)
    data[0, 0] = 3
    data[0, 1] = 4
    result = TransformationDiagnosticsAnalyzer(algebra).analyze(data)
    torch.testing.assert_close(result.vector_coefficient_energy, data.new_tensor([8, 0, 0]))
    assert result.odd_grade_energy_fraction == pytest.approx((16 / 25) / 2)
    assert not hasattr(result, "near_commuting_mode_count")
    assert not hasattr(result, "low_energy_vector_directions")


def test_reflection_extends_to_each_grade_and_preserves_scalars(algebra):
    analyzer = TransformationDiagnosticsAnalyzer(algebra)
    basis = torch.eye(algebra.dim, dtype=torch.float64)
    reflected, valid = analyzer._planned_basis_reflections(basis)
    for direction in range(algebra.p + algebra.q):
        signs = basis.new_tensor([-1 if blade & (1 << direction) else 1 for blade in range(algebra.dim)])
        torch.testing.assert_close(reflected[direction], basis * signs)
        torch.testing.assert_close(reflected[direction, 0], basis[0])
        assert valid[direction]


def test_reflection_scores_are_defined_distances_and_nulls_are_unavailable(algebra):
    data = torch.zeros(1, algebra.dim, dtype=torch.float64)
    data[0, 0], data[0, 1] = 3, 4
    result = TransformationDiagnosticsAnalyzer(algebra).analyze(data)
    by_direction = {item["direction"]: item["score"] for item in result.basis_reflection_marginal_scores}
    assert by_direction[0] == pytest.approx(64 / 25)
    assert by_direction[1] == 0
    if algebra.r:
        assert by_direction[2] is None
        assert result.skipped["basis_reflection_marginal_scores"]["details"]["directions"] == [2]
        assert result.basis_reflection_marginal_scores[-1]["direction"] == 2


@pytest.mark.parametrize("n", [0, 1])
def test_absent_bivectors_and_zero_observations(n):
    algebra = make_algebra(n, dtype=torch.float64)
    data = torch.zeros(3, algebra.dim, dtype=torch.float64)
    spectral = SpectralAnalyzer(algebra).analyze(data)
    commutator = CommutatorAnalyzer(algebra).analyze(data)
    transformation = TransformationDiagnosticsAnalyzer(algebra).analyze(data)
    assert spectral.mean_bivector.count_nonzero() == 0
    assert commutator.basis_bivector_commutator_norm_ratios.numel() == 0
    assert commutator.mean_commutator_norm == 0
    assert transformation.vector_coefficient_energy.shape == (n,)
    assert transformation.odd_grade_energy_fraction == 0


def test_ordinary_composition_does_not_change_transformation_results(algebra):
    data = torch.randn(6, algebra.dim, dtype=torch.float64)
    analyzer = TransformationDiagnosticsAnalyzer(algebra)
    before = analyzer.analyze(data)
    CommutatorAnalyzer(algebra).analyze(data)
    SpectralAnalyzer(algebra).analyze(data)
    after = analyzer.analyze(data)
    assert before.basis_reflection_marginal_scores == after.basis_reflection_marginal_scores
    assert before.odd_grade_energy_fraction == after.odd_grade_energy_fraction
    torch.testing.assert_close(before.vector_coefficient_energy, after.vector_coefficient_energy)
