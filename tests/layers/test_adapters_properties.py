# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from clifra.core.runtime.algebra import AlgebraContext
from clifra.layers.adapters.conformal import ConformalEmbedding
from clifra.layers.adapters.projective import ProjectiveEmbedding
from tests.helpers.hypothesis_cases import QUICK_PROPERTY_SETTINGS, tensor_with_shape
from tests.helpers.small_oracle import SmallCliffordOracle

pytestmark = pytest.mark.unit


@QUICK_PROPERTY_SETTINGS
@given(d=st.integers(min_value=1, max_value=3), data=st.data())
def test_conformal_embedding_round_trips_and_lands_on_null_cone(d, data):
    algebra = AlgebraContext(d + 1, 1, 0, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(algebra.p, algebra.q, algebra.r)
    embed = ConformalEmbedding(algebra, euclidean_dim=d)
    batch = data.draw(st.integers(min_value=1, max_value=4))
    points = data.draw(tensor_with_shape((batch, d)))

    conformal = embed.embed(points)
    null_square = oracle.product(conformal, conformal)

    assert torch.allclose(null_square, torch.zeros_like(null_square), atol=1e-10, rtol=1e-10)
    assert torch.allclose(embed.extract(conformal), points, atol=1e-10, rtol=1e-10)


@QUICK_PROPERTY_SETTINGS
@given(d=st.integers(min_value=1, max_value=3), data=st.data())
def test_compact_conformal_embedding_round_trips_and_lands_on_null_cone(d, data):
    algebra = AlgebraContext(d + 1, 1, 0, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(algebra.p, algebra.q, algebra.r)
    layout = algebra.layout((1,))
    embed = ConformalEmbedding(algebra, euclidean_dim=d, layout=layout)
    batch = data.draw(st.integers(min_value=1, max_value=4))
    points = data.draw(tensor_with_shape((batch, d)))

    conformal = embed.embed(points)
    null_square = oracle.product(
        conformal,
        conformal,
        left_indices=layout.basis_indices,
        right_indices=layout.basis_indices,
    )

    assert conformal.shape == (batch, layout.dim)
    assert torch.allclose(null_square, torch.zeros_like(null_square), atol=1e-10, rtol=1e-10)
    assert torch.allclose(embed.extract(conformal), points, atol=1e-10, rtol=1e-10)


@QUICK_PROPERTY_SETTINGS
@given(d=st.integers(min_value=1, max_value=4), data=st.data())
def test_projective_embedding_round_trips_and_uses_grade_one_lanes(d, data):
    algebra = AlgebraContext(d, 0, 1, device="cpu", dtype=torch.float64)
    embed = ProjectiveEmbedding(algebra, euclidean_dim=d)
    batch = data.draw(st.integers(min_value=1, max_value=4))
    points = data.draw(tensor_with_shape((batch, d)))

    projective = embed.embed(points)
    grade_one = torch.zeros(algebra.dim, dtype=torch.bool)
    grade_one[algebra.layout((1,)).indices_tensor()] = True

    assert torch.allclose(projective[:, ~grade_one], torch.zeros_like(projective[:, ~grade_one]))
    assert torch.allclose(projective[:, embed._idx_e0], torch.ones(batch, dtype=torch.float64))
    assert torch.allclose(embed.extract(projective), points, atol=1e-10, rtol=1e-10)


@QUICK_PROPERTY_SETTINGS
@given(d=st.integers(min_value=1, max_value=4), data=st.data())
def test_compact_projective_embedding_round_trips_and_preserves_homogeneous_coordinate(d, data):
    algebra = AlgebraContext(d, 0, 1, device="cpu", dtype=torch.float64)
    layout = algebra.layout((1,))
    embed = ProjectiveEmbedding(algebra, euclidean_dim=d, layout=layout)
    batch = data.draw(st.integers(min_value=1, max_value=4))
    points = data.draw(tensor_with_shape((batch, d)))
    directions = data.draw(tensor_with_shape((batch, d)))

    projective = embed.embed(points)
    ideal = embed.embed_direction(directions)

    assert projective.shape == (batch, layout.dim)
    assert torch.allclose(projective[:, embed._idx_e0], torch.ones(batch, dtype=torch.float64))
    assert torch.allclose(ideal[:, embed._idx_e0], torch.zeros(batch, dtype=torch.float64))
    assert torch.allclose(embed.extract(projective), points, atol=1e-10, rtol=1e-10)


@QUICK_PROPERTY_SETTINGS
@given(d=st.integers(min_value=2, max_value=4), angle=st.floats(-math.pi, math.pi), data=st.data())
def test_projective_rotor_rotation_matches_euclidean_rotation(d, angle, data):
    algebra = AlgebraContext(d, 0, 1, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(algebra.p, algebra.q, algebra.r)
    embed = ProjectiveEmbedding(algebra, euclidean_dim=d)
    batch = data.draw(st.integers(min_value=1, max_value=3))
    points = data.draw(tensor_with_shape((batch, d)))
    bivector = torch.zeros(1, algebra.dim, dtype=torch.float64)
    bivector[0, 3] = -0.5 * float(angle)
    rotor = algebra.bivector_exp(bivector)
    rotor_reverse = oracle.reverse(rotor)
    projective = embed.embed(points)

    rotated = oracle.product(oracle.product(rotor, projective), rotor_reverse)
    expected = points.clone()
    cos_angle = math.cos(float(angle))
    sin_angle = math.sin(float(angle))
    expected[:, 0] = cos_angle * points[:, 0] - sin_angle * points[:, 1]
    expected[:, 1] = sin_angle * points[:, 0] + cos_angle * points[:, 1]

    assert torch.allclose(embed.extract(rotated), expected, atol=1e-9, rtol=1e-9)
    assert torch.allclose(rotated[:, embed._idx_e0], torch.ones(batch, dtype=torch.float64), atol=1e-9, rtol=1e-9)
