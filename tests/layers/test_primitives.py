# clifra: Clifford algebra layers for PyTorch
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import pytest
import torch

from clifra.core.execution.handles import MultiVersorActionHandle, PairedBivectorActionHandle, VersorActionHandle
from clifra.core.runtime.algebra import AlgebraContext
from clifra.layers import (
    BladeSelector,
    CliffordLayerNorm,
    CliffordLinear,
    MultiRotorLayer,
    ReflectionLayer,
    RotorGadget,
    RotorLayer,
)

pytestmark = pytest.mark.unit


def _planned_grade2_full_factors(algebra, weights: torch.Tensor, parameter_layout):
    rotor_layout = algebra.layout(range(0, algebra.n + 1, 2))
    exp = algebra.plan_exp(input_layout=parameter_layout, output_layout=rotor_layout)
    reverse = algebra.plan_unary(op="reverse", input_layout=rotor_layout, output_layout=rotor_layout)
    rotor = exp(-0.5 * weights)
    return rotor_layout.full(rotor), rotor_layout.full(reverse(rotor))


class TestLayers:
    def test_linear_shape(self, algebra_3d):
        # Batch=4, In=2 channels, Out=3 channels
        # x: [4, 2, 8]
        x = torch.randn(4, 2, 8)
        layer = CliffordLinear(algebra_3d, 2, 3)
        y = layer(x)
        assert y.shape == (4, 3, 8)

    def test_linear_accepts_extra_leading_dimensions(self, algebra_3d):
        x = torch.randn(2, 3, 4, algebra_3d.dim)
        layer = CliffordLinear(algebra_3d, 4, 5)

        y = layer(x)
        expected = torch.einsum("oi,...id->...od", layer.weight, x) + layer.bias.view(1, 1, 5, algebra_3d.dim)

        assert y.shape == (2, 3, 5, algebra_3d.dim)
        assert torch.allclose(y, expected)

    def test_layer_norm_accepts_extra_leading_dimensions(self, algebra_3d):
        x = torch.randn(2, 3, 4, algebra_3d.dim)
        layer = CliffordLayerNorm(algebra_3d, 4)

        y = layer(x)

        assert y.shape == x.shape

    def test_blade_selector_starts_as_pass_through(self, algebra_3d):
        x = torch.randn(2, 3, 4, algebra_3d.dim)
        layer = BladeSelector(algebra_3d, 4)

        y = layer(x)

        assert torch.allclose(y, x)

    def test_compact_layout_preserving_primitives_with_context(self):
        context = AlgebraContext(5, 0, device="cpu", default_grades=(1,))
        layout = context.layout((1,))
        x = torch.randn(2, 3, 4, layout.dim)

        linear = CliffordLinear(context, 4, 2, layout=layout)
        norm = CliffordLayerNorm(context, 2, layout=layout)
        selector = BladeSelector(context, 2, layout=layout)

        y = selector(norm(linear(x)))

        assert y.shape == (2, 3, 2, layout.dim)

    def test_rotor_shape(self, algebra_3d):
        # Batch=4, Channels=5
        x = torch.randn(4, 5, 8)
        layer = RotorLayer(algebra_3d, 5)
        y = layer(x)
        assert y.shape == (4, 5, 8)

        # Test equivariance (norm preservation for vector part)
        # Vector part is indices 1,2,4 (for 3D basis 1, e1, e2, e3... indices are bitmasks)
        # 1=001, 2=010, 4=100
        vec_indices = [1, 2, 4]

        # Create pure vector input
        x_vec = torch.zeros(4, 5, 8)
        x_vec[..., vec_indices] = torch.randn(4, 5, 3)

        y_vec = layer(x_vec)

        # Norm should be preserved
        x_norm = x_vec.norm(dim=-1)
        y_norm = y_vec.norm(dim=-1)

        # Note: Rotor preserves magnitude of the multivector,
        # and specifically rotates k-vectors to k-vectors.
        # So the norm of the whole multivector should be preserved exactly.

        assert torch.allclose(x_norm, y_norm, atol=1e-5)

    def test_rotor_accepts_extra_leading_dimensions(self, algebra_3d):
        x = torch.randn(2, 3, 4, algebra_3d.dim)
        layer = RotorLayer(algebra_3d, 4)

        y = layer(x)

        assert y.shape == x.shape

    def test_versor_layers_store_planned_action_handles(self, algebra_3d):
        rotor = RotorLayer(algebra_3d, 4)
        reflection = ReflectionLayer(algebra_3d, 4)
        multi = MultiRotorLayer(algebra_3d, 4, num_rotors=2)
        gadget = RotorGadget(algebra_3d, in_channels=4, out_channels=3, num_rotor_pairs=2)

        assert isinstance(rotor.action, VersorActionHandle)
        assert isinstance(reflection.action, VersorActionHandle)
        assert isinstance(multi.action, MultiVersorActionHandle)
        assert isinstance(gadget.action, PairedBivectorActionHandle)

    def test_compact_rotor_matches_full_lane_reference(self):
        context = AlgebraContext(3, 0, device="cpu", default_grades=(1,))
        full_context = AlgebraContext(3, 0, device="cpu")
        layout = context.layout((1,))
        x = torch.randn(2, 4, layout.dim)

        compact_layer = RotorLayer(context, 4, input_layout=layout)
        full_layer = RotorLayer(full_context, 4)
        full_layer.grade_weights.data.copy_(compact_layer.grade_weights.data)

        actual = compact_layer(x)
        expected = layout.compact(full_layer(layout.full(x)))

        assert actual.shape == x.shape
        assert torch.allclose(actual, expected, atol=1e-4)

    def test_compact_rotor_multigrade_matches_full_lane_reference(self):
        context = AlgebraContext(3, 0, device="cpu")
        full_context = AlgebraContext(3, 0, device="cpu")
        layout = context.layout((0, 1, 2))
        full_x = torch.randn(2, 3, full_context.dim)
        x = layout.compact(full_x)

        compact_layer = RotorLayer(context, 3, input_layout=layout)
        full_layer = RotorLayer(full_context, 3)
        full_layer.grade_weights.data.copy_(compact_layer.grade_weights.data)

        actual = compact_layer(x)
        expected = layout.compact(full_layer(full_x))

        assert actual.shape == x.shape
        assert torch.allclose(actual, expected, atol=1e-4)

    def test_declared_rotor_layout_accepts_full_lane_input_and_returns_compact(self):
        full_context = AlgebraContext(3, 0, device="cpu")
        layout = full_context.layout((1,))
        x = full_context.embed_vector(torch.randn(2, 4, full_context.n))

        declared_layer = RotorLayer(full_context, 4, input_layout=layout)
        reference_layer = RotorLayer(full_context, 4)
        reference_layer.grade_weights.data.copy_(declared_layer.grade_weights.data)

        actual = declared_layer(x)
        expected = layout.compact(reference_layer(x))

        assert actual.shape == (2, 4, layout.dim)
        assert torch.allclose(actual, expected, atol=1e-4)

    def test_multi_rotor_shape(self, algebra_3d):
        x = torch.randn(4, 5, 8)
        layer = MultiRotorLayer(algebra_3d, 5, num_rotors=4)
        y = layer(x)
        assert y.shape == (4, 5, 8)

    def test_multi_rotor_accepts_extra_leading_dimensions(self, algebra_3d):
        x = torch.randn(2, 3, 4, algebra_3d.dim)
        layer = MultiRotorLayer(algebra_3d, 4, num_rotors=3)

        y = layer(x)
        inv = layer(x, return_invariants=True)

        assert y.shape == x.shape
        assert inv.shape == (2, 3, 4, algebra_3d.num_grades)

    def test_compact_multi_rotor_matches_full_lane_reference(self):
        context = AlgebraContext(3, 0, device="cpu", default_grades=(1,))
        full_context = AlgebraContext(3, 0, device="cpu")
        layout = context.layout((1,))
        x = torch.randn(2, 3, layout.dim)

        compact_layer = MultiRotorLayer(context, 3, num_rotors=2, input_layout=layout)
        full_layer = MultiRotorLayer(full_context, 3, num_rotors=2)
        full_layer.rotor_grade_weights.data.copy_(compact_layer.rotor_grade_weights.data)
        full_layer.weights.data.copy_(compact_layer.weights.data)

        actual = compact_layer(x)
        expected = layout.compact(full_layer(layout.full(x)))

        assert actual.shape == x.shape
        assert torch.allclose(actual, expected, atol=1e-4)

    def test_multi_rotor_invariants(self, algebra_3d):
        x = torch.randn(4, 5, 8)
        layer = MultiRotorLayer(algebra_3d, 5, num_rotors=4)
        inv = layer(x, return_invariants=True)
        # 3D algebra has 4 grades (0, 1, 2, 3)
        assert inv.shape == (4, 5, 4)

    def test_rotor_layer_preserves_norm(self, algebra_3d):
        x = torch.randn(4, 5, 8)
        layer = RotorLayer(algebra_3d, 5)
        y = layer(x)

        # Check output shape
        assert y.shape == (4, 5, 8)

        # Check norm preservation (rotor property)
        x_norm = x.norm(dim=-1)
        y_norm = y.norm(dim=-1)
        assert torch.allclose(x_norm, y_norm, atol=1e-4)

    def test_rotor_layer_repeated_evaluation_matches(self, algebra_3d):
        layer_a = RotorLayer(algebra_3d, 3)
        layer_b = RotorLayer(algebra_3d, 3)
        layer_b.grade_weights.data = layer_a.grade_weights.data.clone()

        x = torch.randn(2, 3, 8)

        y_fast = layer_a(x)
        y_exact = layer_b(x)

        # For n=3, all bivectors are simple so results should match
        assert torch.allclose(y_fast, y_exact, atol=1e-3)

    def test_rotor_layer_backward(self, algebra_3d):
        x = torch.randn(2, 3, 8, requires_grad=True)
        layer = RotorLayer(algebra_3d, 3)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert layer.grade_weights.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        assert not torch.isnan(layer.grade_weights.grad).any()
        assert not torch.isinf(layer.grade_weights.grad).any()

    def test_multi_rotor_layer_shape(self, algebra_3d):
        x = torch.randn(4, 5, 8)
        layer = MultiRotorLayer(algebra_3d, 5, num_rotors=4)
        y = layer(x)

        assert y.shape == (4, 5, 8)

    def test_multi_rotor_layer_backward(self, algebra_3d):
        x = torch.randn(2, 3, 8, requires_grad=True)
        layer = MultiRotorLayer(algebra_3d, 3, num_rotors=4)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert layer.rotor_grade_weights.grad is not None
        assert layer.weights.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        assert not torch.isnan(layer.rotor_grade_weights.grad).any()
        assert not torch.isinf(layer.rotor_grade_weights.grad).any()

    def test_rotor_layer_rotor_property(self, algebra_3d):
        """Verify that exp-produced rotors satisfy R * ~R = 1."""
        layer = RotorLayer(algebra_3d, 2)

        B = torch.zeros(layer.channels, algebra_3d.dim)
        indices = layer.grade_indices.unsqueeze(0).expand(layer.channels, -1)
        B.scatter_(1, indices, layer.grade_weights)

        R = algebra_3d.exp(-0.5 * B)
        R_rev = algebra_3d.reverse(R)

        # R * ~R should be identity
        for i in range(layer.channels):
            identity = algebra_3d.geometric_product(R[i : i + 1], R_rev[i : i + 1])

            expected_identity = torch.zeros_like(identity)
            expected_identity[..., 0] = 1.0

            assert torch.allclose(identity, expected_identity, atol=1e-4)

    def test_reflection_shape(self, algebra_3d):
        B, C = 4, 5
        layer = ReflectionLayer(algebra_3d, channels=C)
        x = torch.randn(B, C, 8)
        y = layer(x)
        assert y.shape == (B, C, 8)

    def test_reflection_accepts_extra_leading_dimensions(self, algebra_3d):
        x = torch.randn(2, 3, 4, algebra_3d.dim)
        layer = ReflectionLayer(algebra_3d, channels=4)

        y = layer(x)

        assert y.shape == x.shape

    def test_compact_reflection_matches_full_lane_reference(self):
        context = AlgebraContext(3, 0, device="cpu", default_grades=(1,))
        full_context = AlgebraContext(3, 0, device="cpu")
        layout = context.layout((1,))
        x = torch.randn(2, 4, layout.dim)

        compact_layer = ReflectionLayer(context, channels=4, input_layout=layout)
        full_layer = ReflectionLayer(full_context, channels=4)
        full_layer.vector_weights.data.copy_(compact_layer.vector_weights.data)

        actual = compact_layer(x)
        expected = layout.compact(full_layer(layout.full(x)))

        assert actual.shape == x.shape
        assert torch.allclose(actual, expected, atol=1e-4)

    def test_declared_reflection_layout_accepts_full_lane_input_and_returns_compact(self):
        full_context = AlgebraContext(3, 0, device="cpu")
        layout = full_context.layout((1,))
        x = full_context.embed_vector(torch.randn(2, 4, full_context.n))

        declared_layer = ReflectionLayer(full_context, channels=4, input_layout=layout)
        reference_layer = ReflectionLayer(full_context, channels=4)
        reference_layer.vector_weights.data.copy_(declared_layer.vector_weights.data)

        actual = declared_layer(x)
        expected = layout.compact(reference_layer(x))

        assert actual.shape == (2, 4, layout.dim)
        assert torch.allclose(actual, expected, atol=1e-4)

    def test_reflection_preserves_norm(self, algebra_3d):
        C = 3
        layer = ReflectionLayer(algebra_3d, channels=C)
        x = torch.randn(2, C, 8)
        y = layer(x)
        x_norms = algebra_3d.norm_sq(x.reshape(-1, 8))
        y_norms = algebra_3d.norm_sq(y.reshape(-1, 8))
        assert torch.allclose(x_norms, y_norms, atol=1e-4)

    def test_reflection_gradient_flow(self, algebra_3d):
        C = 4
        layer = ReflectionLayer(algebra_3d, channels=C)
        x = torch.randn(2, C, 8)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert layer.vector_weights.grad is not None
        assert not torch.all(layer.vector_weights.grad == 0)

    def test_reflection_eval_has_no_full_lane_cache(self, algebra_3d):
        C = 3
        layer = ReflectionLayer(algebra_3d, channels=C)
        layer.eval()
        x = torch.randn(2, C, 8)
        y = layer(x)
        assert y.shape == x.shape
        assert not hasattr(layer, "_cached_n")
        assert not hasattr(layer, "_cached_n_inv")
        layer.train()

    def test_reflection_different_signatures(self):
        for p, q in [(2, 0), (3, 0), (2, 1), (3, 1)]:
            alg = AlgebraContext(p, q, device="cpu")
            C = 2
            layer = ReflectionLayer(alg, channels=C)
            x = torch.randn(3, C, alg.dim)
            y = layer(x)
            assert y.shape == x.shape

    def test_reflection_sparsity_loss(self, algebra_3d):
        layer = ReflectionLayer(algebra_3d, channels=4)
        loss = layer.sparsity_loss()
        assert loss.dim() == 0
        assert loss.item() > 0

    # --- Multi-rotor action-matrix equivalence ---

    def test_multi_rotor_action_matrix_equivalence(self, algebra_3d):
        """Verify action-matrix sandwich matches two-GP sandwich numerically."""
        K, C, B = 4, 5, 3
        layer = MultiRotorLayer(algebra_3d, C, num_rotors=K)
        x = torch.randn(B, C, algebra_3d.dim)

        V_left, V_right = _planned_grade2_full_factors(algebra_3d, layer.rotor_grade_weights, layer.parameter_layout)

        # Action-matrix path (current implementation)
        y_action = algebra_3d.multi_rotor_sandwich(V_left, x, V_right)

        # Two-GP reference path
        x_expanded = x.unsqueeze(2)  # [B, C, 1, D]
        VL = V_left.view(1, 1, K, -1)  # [1, 1, K, D]
        VR = V_right.view(1, 1, K, -1)  # [1, 1, K, D]
        Vx = algebra_3d.geometric_product(VL, x_expanded)  # [B, C, K, D]
        y_gp = algebra_3d.geometric_product(Vx, VR)  # [B, C, K, D]

        assert torch.allclose(y_action, y_gp, atol=1e-5), f"Max diff: {(y_action - y_gp).abs().max().item():.2e}"

    def test_rotor_gadget_mean_routing_keeps_remainder_channels(self, algebra_3d):
        layer = RotorGadget(algebra_3d, in_channels=5, out_channels=3, num_rotor_pairs=2, aggregation="mean")
        with torch.no_grad():
            layer.bivector_left.zero_()
            layer.bivector_right.zero_()

        x = torch.zeros(1, 5, algebra_3d.dim)
        x[0, :, 0] = torch.arange(1, 6, dtype=x.dtype)

        y = layer(x)

        assert torch.allclose(y[0, :, 0], torch.tensor([1.5, 3.5, 5.0]))

    def test_rotor_gadget_accepts_extra_leading_dimensions(self, algebra_3d):
        x = torch.randn(2, 3, 4, algebra_3d.dim)
        layer = RotorGadget(algebra_3d, in_channels=4, out_channels=6, num_rotor_pairs=2)

        y = layer(x)

        assert y.shape == (2, 3, 6, algebra_3d.dim)


# --- torch.compile smoke tests ---


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
class TestCompile:
    def test_compile_rotor_layer(self, algebra_3d):
        """RotorLayer compiles with aot_eager fullgraph."""
        layer = RotorLayer(algebra_3d, channels=4)
        compiled = torch.compile(layer, backend="aot_eager", fullgraph=True)
        x = torch.randn(2, 4, 8)
        y = compiled(x)
        assert y.shape == (2, 4, 8)

    def test_compile_multi_rotor_layer(self, algebra_3d):
        """MultiRotorLayer compiles with aot_eager fullgraph."""
        layer = MultiRotorLayer(algebra_3d, channels=4, num_rotors=3)
        compiled = torch.compile(layer, backend="aot_eager", fullgraph=True)
        x = torch.randn(2, 4, 8)
        y = compiled(x)
        assert y.shape == (2, 4, 8)

    def test_compile_compact_versor_layers(self):
        """Compact versor layers compile through the polymorphic core dispatcher."""
        context = AlgebraContext(3, 0, device="cpu")
        layout = context.layout((1,))
        x = torch.randn(2, 4, layout.dim)

        layers = (
            RotorLayer(context, channels=4, input_layout=layout),
            ReflectionLayer(context, channels=4, input_layout=layout),
            MultiRotorLayer(context, channels=4, num_rotors=2, input_layout=layout),
        )
        for layer in layers:
            compiled = torch.compile(layer, backend="aot_eager", fullgraph=True)
            y = compiled(x)
            assert y.shape == x.shape

    def test_compile_backward(self, algebra_3d):
        """Gradients flow through compiled RotorLayer."""
        layer = RotorLayer(algebra_3d, channels=4)
        compiled = torch.compile(layer, backend="aot_eager")
        x = torch.randn(2, 4, 8, requires_grad=True)
        y = compiled(x)
        y.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    @pytest.mark.skipif(
        not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        reason="MPS not available",
    )
    def test_mps_compile_smoke(self):
        """RotorLayer compiles and runs on MPS."""
        alg = AlgebraContext(3, 0, device="mps")
        layer = RotorLayer(alg, channels=4).to("mps")
        compiled = torch.compile(layer, backend="aot_eager")
        x = torch.randn(2, 4, 8, device="mps")
        y = compiled(x)
        torch.mps.synchronize()
        assert y.shape == (2, 4, 8)
