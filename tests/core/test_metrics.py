# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.core.config import make_algebra
from clifra.core.runtime.metric import (
    clifford_conjugate,
    conjugate_form_distance_like,
    conjugate_form_magnitude,
    conjugate_grade_magnitude_spectrum,
    conjugate_scalar_form,
    conjugate_scalar_form_signs,
    grade_purity,
    lane_distance,
    lane_grade_distribution,
    lane_grade_energy,
    lane_inner_product,
    lane_norm,
    mean_grade,
    scalar_product,
    signature_magnitude,
    signature_norm_squared,
    signature_trace_form,
)

pytestmark = pytest.mark.unit


class TestLaneGeometry:
    def test_lane_norm_is_positive_definite_for_mixed_signature(self, algebra_minkowski):
        values = torch.randn(8, algebra_minkowski.dim)
        norms = lane_norm(algebra_minkowski, values)

        assert torch.all(norms >= 0)
        assert torch.allclose(lane_norm(algebra_minkowski, torch.zeros_like(values)), torch.zeros(8, 1))

    def test_lane_distance_is_a_euclidean_coefficient_distance(self, algebra_minkowski):
        a = torch.randn(algebra_minkowski.dim)
        b = torch.randn(algebra_minkowski.dim)
        c = torch.randn(algebra_minkowski.dim)

        assert torch.allclose(lane_distance(algebra_minkowski, a, a), torch.zeros(1))
        assert torch.allclose(lane_distance(algebra_minkowski, a, b), lane_distance(algebra_minkowski, b, a))
        assert lane_distance(algebra_minkowski, a, c) <= lane_distance(algebra_minkowski, a, b) + lane_distance(
            algebra_minkowski, b, c
        ) + 1e-6

    def test_lane_inner_product_matches_coefficient_dot_product(self, algebra_3d):
        a = torch.randn(4, algebra_3d.dim)
        b = torch.randn(4, algebra_3d.dim)

        assert torch.allclose(lane_inner_product(algebra_3d, a, b), (a * b).sum(dim=-1, keepdim=True))

    def test_compact_layout_uses_declared_lanes_without_materializing_full_values(self):
        algebra = make_algebra(9, 0, 0, device="cpu", dtype=torch.float32)
        layout = algebra.layout((1,))
        values = torch.ones(3, layout.dim)

        assert torch.allclose(lane_norm(algebra, values, layout=layout), torch.full((3, 1), algebra.n**0.5))

    def test_lane_grade_energy_and_distribution_fill_absent_grades(self, algebra_3d):
        values = torch.zeros(algebra_3d.dim)
        values[0] = 2.0
        values[algebra_3d.layout((1,)).indices_tensor()] = 1.0

        energy = lane_grade_energy(algebra_3d, values)
        distribution = lane_grade_distribution(algebra_3d, values)

        assert torch.allclose(energy, torch.tensor([4.0, 3.0, 0.0, 0.0]))
        assert torch.allclose(distribution.sum(), torch.tensor(1.0))

    def test_grade_purity_and_mean_grade_use_lane_energy(self, algebra_3d):
        values = torch.zeros(algebra_3d.dim)
        values[0] = 1.0
        values[algebra_3d.layout((1,)).indices_tensor()] = 1.0

        assert torch.allclose(grade_purity(algebra_3d, values, 1), torch.tensor(0.75))
        assert torch.allclose(mean_grade(algebra_3d, values), torch.tensor(0.75))


class TestConjugateScalarForm:
    def test_signs_match_conjugate_geometric_product_scalar(self, algebra_minkowski):
        torch.manual_seed(42)
        a = torch.randn(algebra_minkowski.dim)
        b = torch.randn(algebra_minkowski.dim)

        signs = conjugate_scalar_form_signs(algebra_minkowski)
        signed = (signs * a * b).sum()
        prod = algebra_minkowski.geometric_product(clifford_conjugate(algebra_minkowski, a).unsqueeze(0), b.unsqueeze(0))

        assert torch.allclose(signed, prod[0, 0], atol=1e-5)

    def test_conjugate_scalar_form_matches_full_product(self, algebra_3d):
        a = torch.randn(algebra_3d.dim)
        b = torch.randn(algebra_3d.dim)

        actual = conjugate_scalar_form(algebra_3d, a, b)
        prod = algebra_3d.geometric_product(clifford_conjugate(algebra_3d, a).unsqueeze(0), b.unsqueeze(0))

        assert torch.allclose(actual.squeeze(), prod[0, 0], atol=1e-5)

    def test_compact_layout_matches_canonical_values(self):
        algebra = make_algebra(3, 1, 0, device="cpu", dtype=torch.float64)
        layout = algebra.layout((1, 2))
        generator = torch.Generator(device="cpu").manual_seed(719)
        a = torch.randn(5, layout.dim, dtype=torch.float64, generator=generator)
        b = torch.randn(5, layout.dim, dtype=torch.float64, generator=generator)

        compact = conjugate_scalar_form(algebra, a, b, layout=layout)
        canonical = conjugate_scalar_form(algebra, layout.full(a), layout.full(b))

        assert torch.allclose(compact, canonical, atol=1e-12, rtol=1e-12)

    def test_magnitude_and_distance_like_are_non_negative_but_signed_form_based(self, algebra_minkowski):
        a = torch.randn(10, algebra_minkowski.dim)
        b = torch.randn(10, algebra_minkowski.dim)

        assert torch.all(conjugate_form_magnitude(algebra_minkowski, a) >= 0)
        assert torch.all(conjugate_form_distance_like(algebra_minkowski, a, b) >= 0)
        assert torch.allclose(conjugate_form_distance_like(algebra_minkowski, a, a), torch.zeros(10, 1))

    def test_conjugate_grade_magnitude_spectrum_has_all_grades(self, algebra_3d):
        values = torch.randn(algebra_3d.dim)
        spectrum = conjugate_grade_magnitude_spectrum(algebra_3d, values)

        assert spectrum.shape == (algebra_3d.n + 1,)
        assert torch.all(spectrum >= 0)


class TestSignatureTraceForm:
    def test_scalar_product_projects_geometric_product_to_grade_zero(self, algebra_3d):
        a = torch.randn(algebra_3d.dim)
        b = torch.randn(algebra_3d.dim)
        product = algebra_3d.geometric_product(a.unsqueeze(0), b.unsqueeze(0))

        assert torch.allclose(scalar_product(algebra_3d, a, b).squeeze(), product[0, 0], atol=1e-5)

    def test_trace_form_matches_reverse_product_scalar(self, algebra_3d):
        a = torch.randn(algebra_3d.dim)
        product = algebra_3d.geometric_product(algebra_3d.reverse(a).unsqueeze(0), a.unsqueeze(0))

        assert torch.allclose(signature_trace_form(algebra_3d, a, a).squeeze(), product[0, 0], atol=1e-5)

    def test_trace_form_can_be_negative_in_mixed_signature(self, algebra_minkowski):
        found_negative = False
        for _ in range(100):
            values = torch.randn(algebra_minkowski.dim)
            if signature_trace_form(algebra_minkowski, values, values) < -1e-6:
                found_negative = True
                break

        assert found_negative

    def test_signature_norm_squared_and_magnitude_are_distinct_contracts(self, algebra_minkowski):
        values = torch.randn(algebra_minkowski.dim)

        assert torch.allclose(
            signature_norm_squared(algebra_minkowski, values),
            signature_trace_form(algebra_minkowski, values, values),
        )
        assert torch.all(signature_magnitude(algebra_minkowski, values) >= 0)
