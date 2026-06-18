# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

from clifra.core.planning import PlanningLimits
from clifra.core.runtime.algebra import AlgebraContext
from clifra.layers import ProductLayer, WedgeLayer
from clifra.layers.blocks.multi_rotor_ffn import MultiRotorFFN
from clifra.optimizers import make_riemannian_optimizer

pytestmark = pytest.mark.unit


def test_product_layer_full_lane_matches_algebra(algebra_3d):
    left = torch.randn(4, 5, algebra_3d.dim)
    right = torch.randn(4, 5, algebra_3d.dim)
    layer = ProductLayer(algebra_3d)

    actual = layer(left, right)
    expected = algebra_3d.geometric_product(left, right)

    assert torch.allclose(actual, expected)


def test_wedge_layer_declared_grades_return_output_layout_lanes(algebra_3d):
    left = algebra_3d.embed_vector(torch.randn(4, 5, algebra_3d.n))
    right = algebra_3d.embed_vector(torch.randn(4, 5, algebra_3d.n))
    output_layout = algebra_3d.layout((2,))
    layer = WedgeLayer(
        algebra_3d,
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(2,),
    )

    actual = layer(left, right)
    expected = algebra_3d.wedge(left, right, left_grades=(1,), right_grades=(1,), output_grades=(2,))

    assert actual.shape[-1] == output_layout.dim
    assert torch.allclose(actual, expected)


def test_product_layer_pairwise_compact_widths_match_full_lane_reference():
    context = AlgebraContext(p=5, q=0, device="cpu")
    full_context = AlgebraContext(p=5, q=0, device="cpu")
    left_layout = context.layout((2,))
    right_layout = context.layout((1,))
    output_layout = context.layout((3,))
    generator = torch.Generator(device="cpu").manual_seed(43)

    left = torch.randn(2, 3, left_layout.dim, generator=generator)
    right = torch.randn(2, 4, right_layout.dim, generator=generator)
    layer = WedgeLayer(
        context,
        left_grades=(2,),
        right_grades=(1,),
        output_grades=(3,),
        pairwise=True,
    )

    actual = layer(left, right)
    cache_size = len(context.planner._product_executors)
    repeated = layer(left, right)
    expected_full = full_context.wedge(
        left_layout.full(left).unsqueeze(2),
        right_layout.full(right).unsqueeze(1),
        left_grades=(2,),
        right_grades=(1,),
        output_grades=(3,),
    )
    expected = expected_full

    assert actual.shape == (2, 3, 4, output_layout.dim)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(repeated, actual)
    assert cache_size == 1
    assert len(context.planner._product_executors) == cache_size


def test_product_layer_validates_declared_layouts_against_grades():
    context = AlgebraContext(p=5, q=0, device="cpu")
    vector_layout = context.layout((1,))

    with pytest.raises(ValueError, match="left_layout and left_grades disagree"):
        ProductLayer(context, left_grades=(2,), left_layout=vector_layout)


def test_compact_layer_pipeline_trains_with_riemannian_optimizer_factory():
    context = AlgebraContext(p=6, q=0, device="cpu")
    vector_layout = context.layout((1,))
    trivector_layout = context.layout((3,))

    class CompactPipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.wedge_vectors = WedgeLayer(
                context,
                left_grades=(1,),
                right_grades=(1,),
                output_grades=(2,),
            )
            self.wedge_trivector = WedgeLayer(
                context,
                left_grades=(2,),
                right_grades=(1,),
                output_grades=(3,),
            )
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, left_vector, right_vector, third_vector):
            bivector = self.wedge_vectors(left_vector, right_vector)
            return self.scale * self.wedge_trivector(bivector, third_vector)

    full_left = context.embed_vector(torch.randn(8, context.n))
    full_right = context.embed_vector(torch.randn(8, context.n))
    compact_third = vector_layout.compact(context.embed_vector(torch.randn(8, context.n)))
    model = CompactPipeline()
    optimizer = make_riemannian_optimizer(model, context, optimizer="adam", lr=0.01)

    output = model(full_left, full_right, compact_third)
    assert output.shape == (8, trivector_layout.dim)
    assert len(context.planner._product_executors) == 2

    loss = output.square().mean()
    loss.backward()
    optimizer.step()

    assert model.scale.grad is not None
    assert torch.isfinite(model.scale).all()


def test_rotor_backend_block_trains_with_riemannian_optimizer_factory():
    algebra = AlgebraContext(p=3, q=0, device="cpu")
    model = MultiRotorFFN(
        algebra,
        channels=2,
        ffn_mult=2,
        num_rotors=2,
        use_rotor_backend=True,
    )
    optimizer = make_riemannian_optimizer(model, algebra, optimizer="adam", lr=0.01)
    x = torch.randn(4, 2, algebra.dim)

    output = model(x)
    loss = output.square().mean()
    loss.backward()
    optimizer.step()

    assert output.shape == x.shape
    assert any(group.get("manifold") == "spin" for group in optimizer.param_groups)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_product_layer_uses_context_planning_limits():
    limits = PlanningLimits(warn_lanes=32, max_lanes=512, warn_pairs=32, max_pairs=64)
    context = AlgebraContext(p=16, q=0, device="cpu", planning_limits=limits)
    vector_layout = context.layout((1,))
    left = torch.zeros(1, vector_layout.dim)
    right = torch.zeros(1, vector_layout.dim)
    layer = ProductLayer(
        context,
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
    )

    with pytest.raises(ValueError, match="basis interactions"):
        layer(left, right)
