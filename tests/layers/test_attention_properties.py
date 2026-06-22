# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from clifra.core.foundation.numerics import eps_like
from clifra.core.runtime.algebra import AlgebraContext
from clifra.layers.blocks.attention import EntropyGatedAttention, GeometricProductAttention
from tests.helpers.hypothesis_cases import QUICK_PROPERTY_SETTINGS, tensor_with_shape
from tests.helpers.small_oracle import SmallCliffordOracle

pytestmark = pytest.mark.unit


def _oracle_attention_score(
    oracle: SmallCliffordOracle,
    q_head: torch.Tensor,
    k_head: torch.Tensor,
    *,
    bivector_weight: float,
    scale_dim: int,
) -> torch.Tensor:
    product = oracle.product(q_head.unsqueeze(3), oracle.reverse(k_head).unsqueeze(2))
    score_g0 = product[..., 0].sum(-1)
    g2_indices = oracle.indices_for_grades((2,))
    if g2_indices:
        g2 = product[..., g2_indices]
        score_g2 = g2.pow(2).sum(dim=(-1, -2)).clamp_min(eps_like(g2)).sqrt()
    else:
        score_g2 = torch.zeros_like(score_g0)
    return (score_g0 + bivector_weight * score_g2) / math.sqrt(q_head.shape[3] * scale_dim)


@QUICK_PROPERTY_SETTINGS
@given(n=st.integers(min_value=2, max_value=4), data=st.data())
def test_geometric_attention_score_matches_small_oracle_full_lanes(n, data):
    algebra = AlgebraContext(n, 0, 0, device="cpu", dtype=torch.float64)
    oracle = SmallCliffordOracle(n, 0, 0)
    batch = data.draw(st.integers(min_value=1, max_value=2))
    heads = data.draw(st.integers(min_value=1, max_value=3))
    query_len = data.draw(st.integers(min_value=1, max_value=4))
    key_len = data.draw(st.integers(min_value=1, max_value=4))
    head_channels = data.draw(st.integers(min_value=1, max_value=3))
    bivector_weight = data.draw(st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False))
    q_head = data.draw(tensor_with_shape((batch, heads, query_len, head_channels, algebra.dim)))
    k_head = data.draw(tensor_with_shape((batch, heads, key_len, head_channels, algebra.dim)))
    attention = GeometricProductAttention(
        algebra,
        channels=heads * head_channels,
        num_heads=heads,
        causal=False,
        bivector_weight=float(bivector_weight),
    ).to(dtype=torch.float64)

    actual = attention._compute_score(q_head, k_head)
    expected = _oracle_attention_score(
        oracle,
        q_head,
        k_head,
        bivector_weight=float(bivector_weight),
        scale_dim=algebra.dim,
    )

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


@QUICK_PROPERTY_SETTINGS
@given(n=st.integers(min_value=2, max_value=5), data=st.data())
def test_geometric_attention_score_matches_full_oracle_for_compact_vectors(n, data):
    algebra = AlgebraContext(n, 0, 0, device="cpu", dtype=torch.float64, default_grades=(1,))
    oracle = SmallCliffordOracle(n, 0, 0)
    layout = algebra.layout((1,))
    batch = data.draw(st.integers(min_value=1, max_value=2))
    heads = data.draw(st.integers(min_value=1, max_value=2))
    query_len = data.draw(st.integers(min_value=1, max_value=3))
    key_len = data.draw(st.integers(min_value=1, max_value=3))
    head_channels = data.draw(st.integers(min_value=1, max_value=3))
    q_head = data.draw(tensor_with_shape((batch, heads, query_len, head_channels, layout.dim)))
    k_head = data.draw(tensor_with_shape((batch, heads, key_len, head_channels, layout.dim)))
    attention = GeometricProductAttention(
        algebra,
        channels=heads * head_channels,
        num_heads=heads,
        causal=False,
        bivector_weight=0.25,
        layout=layout,
    ).to(dtype=torch.float64)

    actual = attention._compute_score(q_head, k_head)
    expected = _oracle_attention_score(
        oracle,
        layout.full(q_head),
        layout.full(k_head),
        bivector_weight=0.25,
        scale_dim=layout.dim,
    )

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


@QUICK_PROPERTY_SETTINGS
@given(data=st.data())
def test_entropy_gated_attention_reports_entropy_from_bivector_lane_energy(data):
    algebra = AlgebraContext(3, 0, 0, device="cpu", dtype=torch.float64)
    layout = algebra.layout((2,))
    batch = data.draw(st.integers(min_value=1, max_value=2))
    length = data.draw(st.integers(min_value=1, max_value=4))
    channels = data.draw(st.sampled_from((2, 4)))
    values = data.draw(tensor_with_shape((batch, length, channels, layout.dim)))
    attention = EntropyGatedAttention(algebra, channels=channels, num_heads=2, layout=layout).to(dtype=torch.float64)

    output, entropy, gate = attention(values, return_gating=True)
    energy = values.square().sum(dim=(-1, -2))
    total = energy.sum(dim=1, keepdim=True)
    eps = eps_like(energy, min_value=torch.finfo(values.dtype).tiny)
    probability = torch.where(total > 0, energy / total.clamp_min(eps), torch.zeros_like(energy))
    expected_entropy = -(probability * torch.log(probability.clamp_min(eps))).sum(dim=1)
    expected_gate = attention.eta * torch.sigmoid(expected_entropy - attention.H_base)

    assert output.shape == values.shape
    assert torch.allclose(entropy, expected_entropy, atol=1e-12, rtol=1e-12)
    assert torch.allclose(gate, expected_gate, atol=1e-12, rtol=1e-12)
