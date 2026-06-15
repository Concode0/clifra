# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from clifra.core import GeometricAttentionScoreExecutor
from clifra.core.runtime.algebra import AlgebraContext

pytestmark = pytest.mark.unit


def _reference_attention_score(algebra, q_head, k_head, bivector_weight, *, scale_dim: int):
    product = algebra.geometric_product(q_head.unsqueeze(3), algebra.reverse(k_head).unsqueeze(2))
    score_g0 = product[..., 0].sum(-1)

    g2_idx = algebra.layout((2,)).indices_tensor(device=product.device)
    if g2_idx.numel() > 0:
        g2 = torch.index_select(product, -1, g2_idx)
        score_g2 = g2.pow(2).sum(dim=(-1, -2)).sqrt()
    else:
        score_g2 = torch.zeros_like(score_g0)

    scale = math.sqrt(q_head.shape[3] * scale_dim)
    return (score_g0 + bivector_weight * score_g2) / scale


def test_full_lane_attention_scorer_matches_direct_product():
    algebra = AlgebraContext(3, 0, 0, device="cpu", dtype=torch.float64)
    layout = algebra.default_layout()
    scorer = GeometricAttentionScoreExecutor(
        algebra,
        head_channels=2,
        bivector_weight=0.25,
        layout=layout,
    )
    q_head = torch.randn(2, 2, 3, 2, algebra.dim, dtype=torch.float64)
    k_head = torch.randn(2, 2, 4, 2, algebra.dim, dtype=torch.float64)

    actual = scorer(q_head, k_head)
    expected = _reference_attention_score(algebra, q_head, k_head, scorer.bivector_weight, scale_dim=layout.dim)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_compact_attention_scorer_matches_full_lane_reference():
    context = AlgebraContext(4, 0, device="cpu", default_grades=(1,), dtype=torch.float64)
    full_context = AlgebraContext(4, 0, 0, device="cpu", dtype=torch.float64)
    layout = context.layout((1,))
    scorer = GeometricAttentionScoreExecutor(context, head_channels=2, bivector_weight=0.25, layout=layout)
    q_head = torch.randn(2, 2, 3, 2, layout.dim, dtype=torch.float64)
    k_head = torch.randn(2, 2, 4, 2, layout.dim, dtype=torch.float64)

    actual = scorer(q_head, k_head)
    expected = _reference_attention_score(
        full_context,
        layout.full(q_head),
        layout.full(k_head),
        scorer.bivector_weight,
        scale_dim=layout.dim,
    )

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
