# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from clifra.core.config import make_algebra
from clifra.core.execution.action import BivectorVectorGeneratorExecutor
from clifra.core.execution.exp import BivectorExpExecutor
from clifra.core.foundation.layout import AlgebraSpec
from clifra.core.planning.exp import (
    SPECTRAL_LOCAL_TRUNCATION_NOTICE,
    BivectorExpExecutionPolicy,
    format_spectral_exp_uniform_tail_stress,
    select_bivector_exp_executor_family,
    spectral_exp_angle_diagnostics,
    spectral_exp_preselection,
    spectral_exp_uniform_tail_stress,
)
from clifra.core.runtime.algebra import AlgebraContext
from tests.helpers.bivector_exp_oracle import bivector_exp_cpu_reference

pytestmark = pytest.mark.unit

DEVICE = "cpu"


def _mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


@pytest.mark.parametrize("signature", [(4, 0, 0), (3, 1, 2), (0, 3, 2), (0, 0, 3)])
def test_bivector_exp_plan_partitions_bivector_lanes_by_metric_signature(signature):
    algebra = AlgebraContext(*signature, device=DEVICE, dtype=torch.float64)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout(range(0, algebra.n + 1, 2))
    executor = algebra.planner.bivector_exp_executor_for_layouts(
        input_layout=bivector_layout,
        output_layout=even_layout,
        dtype=torch.float64,
        device=DEVICE,
    )
    split = algebra.p + algebra.q
    expected_nondegenerate = []
    expected_mixed = []
    expected_nilpotent = []

    for position, index in enumerate(bivector_layout.basis_indices):
        bits = [bit for bit in range(algebra.n) if index & (1 << bit)]
        if bits[0] >= split and bits[1] >= split:
            expected_nilpotent.append(position)
        elif bits[0] >= split or bits[1] >= split:
            expected_mixed.append(position)
        else:
            expected_nondegenerate.append(position)

    metric_signs = [1.0] * algebra.p + [-1.0] * algebra.q + [0.0] * algebra.r
    assert executor.metric_signs.tolist() == metric_signs
    assert executor.nondegenerate_bivector_positions.tolist() == expected_nondegenerate
    assert executor.mixed_degenerate_bivector_positions.tolist() == expected_mixed
    assert executor.nilpotent_bivector_positions.tolist() == expected_nilpotent


def test_bivector_exp_plan_generator_maps_match_vector_generator_blocks():
    algebra = AlgebraContext(3, 1, 2, device=DEVICE, dtype=torch.float64)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout(range(0, algebra.n + 1, 2))
    executor = algebra.planner.bivector_exp_executor_for_layouts(
        input_layout=bivector_layout,
        output_layout=even_layout,
        dtype=torch.float64,
        device=DEVICE,
    )
    generator = BivectorVectorGeneratorExecutor(
        bivector_layout=bivector_layout,
        dtype=torch.float64,
        device=DEVICE,
    )
    values = torch.randn(
        7,
        bivector_layout.dim,
        dtype=torch.float64,
        generator=torch.Generator(device=DEVICE).manual_seed(293),
    )
    full_generator = generator.execute(values)
    nondegenerate_dim = algebra.p + algebra.q

    nondegenerate = torch.matmul(
        values,
        executor.bivector_to_nondegenerate_generator.reshape(bivector_layout.dim, -1),
    ).reshape(values.shape[0], nondegenerate_dim, nondegenerate_dim)
    mixed = torch.matmul(
        values,
        executor.bivector_to_mixed_generator.reshape(bivector_layout.dim, -1),
    ).reshape(values.shape[0], algebra.r, nondegenerate_dim)

    assert torch.allclose(nondegenerate, full_generator[:, :nondegenerate_dim, :nondegenerate_dim])
    assert torch.allclose(mixed, full_generator[:, nondegenerate_dim:, :nondegenerate_dim])


def test_planner_bivector_exp_executor_outputs_match_cpu_oracle_for_layouts():
    algebra = AlgebraContext(4, 0, 0, device=DEVICE, dtype=torch.float64)
    bivector_layout = algebra.layout((2,))
    even_layout = algebra.layout((0, 2, 4))
    full_layout = algebra.layout()
    generator = torch.Generator(device=DEVICE).manual_seed(281)
    compact = torch.randn(3, bivector_layout.dim, dtype=torch.float64, generator=generator) * 0.1

    compact_executor = algebra.planner.bivector_exp_executor_for_layouts(
        input_layout=bivector_layout,
        output_layout=even_layout,
        dtype=torch.float64,
        device=DEVICE,
    )
    full_executor = algebra.planner.bivector_exp_executor_for_layouts(
        input_layout=bivector_layout,
        output_layout=full_layout,
        dtype=torch.float64,
        device=DEVICE,
    )

    actual_compact = compact_executor(compact)
    actual_full = full_executor(compact)
    expected_compact = bivector_exp_cpu_reference(
        algebra,
        compact,
        input_layout=bivector_layout,
        output_layout=even_layout,
    )
    expected_full = bivector_exp_cpu_reference(
        algebra,
        compact,
        input_layout=bivector_layout,
        output_layout=full_layout,
    )

    assert isinstance(compact_executor, BivectorExpExecutor)
    assert isinstance(full_executor, BivectorExpExecutor)
    assert compact_executor.executor_family == "closed_biquadratic"
    assert full_executor.output_layout == full_layout
    assert torch.allclose(actual_compact, expected_compact, atol=1e-10, rtol=1e-10)
    assert torch.allclose(actual_full, expected_full, atol=1e-10, rtol=1e-10)


def test_bivector_exp_executor_policy_selects_remaining_families():
    low_dim = AlgebraSpec(3, 0, 0)
    closed_dim = AlgebraSpec(5, 0, 0)
    meso_dim = AlgebraSpec(6, 0, 0)
    spectral_dim = AlgebraSpec(10, 0, 0)
    matrix_dim = AlgebraSpec(3, 3, 0)

    assert select_bivector_exp_executor_family(low_dim, torch.device("mps")) == "closed_simple"
    assert select_bivector_exp_executor_family(closed_dim, torch.device("mps")) == "closed_biquadratic"
    assert select_bivector_exp_executor_family(closed_dim, torch.device("cpu")) == "closed_biquadratic"
    assert select_bivector_exp_executor_family(meso_dim, torch.device("cpu")) == "left_matrix_exp"
    assert select_bivector_exp_executor_family(spectral_dim, torch.device("cpu")) == "spectral_local"
    assert select_bivector_exp_executor_family(meso_dim, torch.device("mps")) == "spectral_local"
    assert select_bivector_exp_executor_family(matrix_dim, torch.device("cpu")) == "left_matrix_exp"
    assert select_bivector_exp_executor_family(matrix_dim, torch.device("mps")) == "cpu_matrix_exp"


def test_bivector_exp_spectral_preselection_selects_default_and_cap():
    spec = AlgebraSpec(10, 0, 0)

    enabled = spectral_exp_preselection(spec, torch.device("cpu"), dtype=torch.float32)
    capped = spectral_exp_preselection(spec, torch.device("cpu"), dtype=torch.float32, max_planes=2)
    dominant = spectral_exp_preselection(spec, torch.device("cpu"), dtype=torch.float32, dominant_rel=0.05)

    assert enabled.eligible
    assert enabled.reason == "eligible"
    assert enabled.max_planes == 4
    assert capped.max_planes == 2
    assert dominant.dominant_rel == 0.05
    assert enabled.dominant_rel == max(torch.finfo(torch.float32).eps**0.5, torch.finfo(torch.float32).eps * 32.0)
    assert enabled.solver_family == "symmetric"
    assert spectral_exp_preselection(spec, torch.device("cpu"), dtype=torch.float32, max_planes=8).max_planes == 4
    assert (
        select_bivector_exp_executor_family(
            spec,
            torch.device("cpu"),
            dtype=torch.float32,
            spectral_max_planes=4,
        )
        == "spectral_local"
    )


def test_bivector_exp_spectral_angle_diagnostics_report_tail_bound_and_gvc():
    angles = torch.tensor([[3.0, -1.0, 2.0, 0.5, -0.25]], dtype=torch.float64)

    diagnostics = spectral_exp_angle_diagnostics(angles, max_planes=2)

    expected_angles = torch.tensor([[3.0, 2.0, 1.0, 0.5, 0.25]], dtype=torch.float64)
    expected_gvc = torch.tensor([(3.0**2 + 2.0**2) / (3.0**2 + 2.0**2 + 1.0**2 + 0.5**2 + 0.25**2)])
    assert diagnostics.selected_planes == 2
    assert diagnostics.total_planes == 5
    assert diagnostics.truncates
    assert diagnostics.notice == SPECTRAL_LOCAL_TRUNCATION_NOTICE
    assert torch.allclose(diagnostics.sorted_abs_angles, expected_angles)
    assert torch.allclose(diagnostics.tail_angle_sum_bound, torch.tensor([1.75], dtype=torch.float64))
    assert torch.allclose(diagnostics.geometric_variance_captured, expected_gvc.to(torch.float64))


def test_bivector_exp_spectral_uniform_tail_stress_reports_static_worst_case():
    rows = spectral_exp_uniform_tail_stress(
        [AlgebraSpec(10, 0, 0), (60, 0, 3), (3, 0, 1)],
        max_planes=4,
        bivector_norm=2.0,
    )

    first, high_dim, covered = rows
    assert first.total_planes == 5
    assert first.selected_planes == 4
    assert first.clipped_planes == 1
    assert first.uniform_angle == pytest.approx(2.0 / (5.0**0.5))
    assert first.tail_angle_sum_bound == pytest.approx(2.0 / (5.0**0.5))
    assert first.geometric_variance_captured == pytest.approx(4.0 / 5.0)
    assert first.truncates
    assert first.notice == SPECTRAL_LOCAL_TRUNCATION_NOTICE

    assert high_dim.total_planes == 31
    assert high_dim.clipped_planes == 27
    assert high_dim.tail_angle_sum_bound == pytest.approx(27.0 * 2.0 / (31.0**0.5))
    assert high_dim.geometric_variance_captured == pytest.approx(4.0 / 31.0)
    assert not covered.truncates
    assert covered.geometric_variance_captured == 1.0

    table = format_spectral_exp_uniform_tail_stress(rows)
    assert "signature | planes | kept | clipped" in table
    assert "Cl(60,0,3)" in table


@pytest.mark.parametrize(
    ("spec", "dtype", "device", "reason"),
    [
        (AlgebraSpec(5, 0, 0), torch.float32, torch.device("cpu"), "closed_formula_preferred"),
        (AlgebraSpec(6, 0, 0), torch.float32, torch.device("cpu"), "matrix_exp_below_spectral_transition"),
        (AlgebraSpec(4, 0, 5), torch.float32, torch.device("cpu"), "ideal_dim_exceeds_block_cap"),
        (AlgebraSpec(3, 3, 0), torch.float32, torch.device("cpu"), "pseudo_euclidean_matrix_exp"),
        (AlgebraSpec(6, 0, 0), torch.bfloat16, torch.device("cpu"), "dtype_error_floor_too_high"),
    ],
)
def test_bivector_exp_spectral_preselection_rejection_reasons(spec, dtype, device, reason):
    decision = spectral_exp_preselection(spec, device, dtype=dtype)

    assert not decision.eligible
    assert decision.reason == reason


def test_bivector_exp_spectral_preselection_uses_mps_routes_without_mps_matrix_exp():
    euclidean = spectral_exp_preselection(AlgebraSpec(6, 0, 0), torch.device("mps"), dtype=torch.float32)
    mixed = spectral_exp_preselection(AlgebraSpec(3, 3, 0), torch.device("mps"), dtype=torch.float32)

    assert euclidean.eligible
    assert euclidean.reason == "eligible"
    assert euclidean.solver_family == "symmetric"
    assert not mixed.eligible
    assert mixed.reason == "pseudo_euclidean_mps_cpu_matrix_exp"
    assert mixed.solver_family == "general_complex"


def test_bivector_exp_spectral_preselection_truncates_degenerate_kernels_by_default():
    odd = spectral_exp_preselection(AlgebraSpec(5, 0, 1), torch.device("cpu"), dtype=torch.float32, transition_n=6)
    uncovered = spectral_exp_preselection(AlgebraSpec(10, 0, 2), torch.device("cpu"), dtype=torch.float32)
    conservative_odd = spectral_exp_preselection(
        AlgebraSpec(5, 0, 1),
        torch.device("cpu"),
        dtype=torch.float32,
        transition_n=6,
        allow_truncated_degenerate=False,
    )
    conservative_uncovered = spectral_exp_preselection(
        AlgebraSpec(10, 0, 2),
        torch.device("cpu"),
        dtype=torch.float32,
        allow_truncated_degenerate=False,
    )

    assert odd.eligible
    assert odd.max_planes == 2
    assert uncovered.eligible
    assert uncovered.max_planes == 4
    assert conservative_odd.reason == "odd_nondegenerate_kernel_deferred"
    assert conservative_uncovered.reason == "degenerate_block_requires_full_plane_cap"


def test_bivector_exp_policy_knobs_connect_through_algebra_context():
    policy = BivectorExpExecutionPolicy(
        spectral_max_planes=2,
        spectral_tol_abs=0.0,
        spectral_tol_rel=0.0,
        spectral_dominant_rel=0.05,
        spectral_allow_truncated_degenerate=False,
    )
    algebra = make_algebra(10, 0, 2, device=DEVICE, dtype=torch.float64, bivector_exp_execution_policy=policy)
    bivector_layout = algebra.layout((2,))
    output_layout = algebra.layout((0, 2))
    conservative = algebra.plan_exp(input_layout=bivector_layout, output_layout=output_layout)
    override = algebra.plan_exp(
        input_layout=bivector_layout,
        output_layout=output_layout,
        spectral_allow_truncated_degenerate=True,
        spectral_max_planes=8,
        cache=False,
    )

    assert conservative.executor_family == "left_matrix_exp"
    assert override.executor_family == "spectral_local"
    assert override.spectral_max_planes == 4
    assert override.spectral_dominant_rel == 0.05


def test_bivector_exp_spectral_local_uses_compute_cap_beyond_cl8():
    spec = AlgebraSpec(10, 0, 0)
    default = spectral_exp_preselection(spec, torch.device("cpu"), dtype=torch.float64)
    capped = spectral_exp_preselection(spec, torch.device("cpu"), dtype=torch.float64, max_planes=4)

    assert default.eligible
    assert default.max_planes == 4
    assert capped.eligible
    assert select_bivector_exp_executor_family(spec, torch.device("cpu"), dtype=torch.float64) == "spectral_local"
    assert (
        select_bivector_exp_executor_family(
            spec,
            torch.device("cpu"),
            dtype=torch.float64,
            spectral_max_planes=4,
        )
        == "spectral_local"
    )


def test_bivector_exp_spectral_local_plans_high_dimension_without_full_even_operator_layout():
    algebra = AlgebraContext(63, 0, 0, device=DEVICE, dtype=torch.float32)
    bivector_layout = algebra.layout((2,))
    output_layout = algebra.layout((0, 2))

    executor = algebra.plan_exp(
        input_layout=bivector_layout,
        output_layout=output_layout,
        dtype=torch.float32,
        device=DEVICE,
        spectral_max_planes=4,
        cache=False,
    )

    assert executor.executor_family == "spectral_local"
    assert executor.operator_layout.grades == (0,)
    assert executor.operator_eye.shape == (1, 1)
    assert executor.spectral_local_axis_count == 8


@pytest.mark.skipif(not _mps_available(), reason="MPS not available")
def test_mps_high_dim_bivector_exp_plans_spectral_local_family():
    algebra = AlgebraContext(6, 0, device="mps", dtype=torch.float32)
    input_layout = algebra.layout((2,))
    output_layout = algebra.layout((0, 2, 4, 6))
    executor = algebra.plan_exp(input_layout=input_layout, output_layout=output_layout, dtype=torch.float32, device="mps")

    assert executor.executor_family == "spectral_local"
    assert executor.left_product is None


@pytest.mark.skipif(not _mps_available(), reason="MPS not available")
def test_mps_mixed_bivector_exp_plans_cpu_matrix_exp_family():
    algebra = AlgebraContext(3, 3, device="mps", dtype=torch.float32)
    input_layout = algebra.layout((2,))
    output_layout = algebra.layout((0, 2, 4, 6))
    executor = algebra.plan_exp(input_layout=input_layout, output_layout=output_layout, dtype=torch.float32, device="mps")

    assert executor.executor_family == "cpu_matrix_exp"
    assert executor.left_product is not None
    assert executor.operator_eye.device.type == "cpu"
    assert executor.left_product.output_positions.device.type == "cpu"


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_bivector_exp_executor_compiles_fullgraph_with_aot_eager():
    algebra = AlgebraContext(4, 0, device=DEVICE, dtype=torch.float32)
    input_layout = algebra.layout((2,))
    output_layout = algebra.layout((0, 2, 4))
    executor = algebra.planner.bivector_exp_executor_for_layouts(
        input_layout=input_layout,
        output_layout=output_layout,
        dtype=torch.float32,
        device=DEVICE,
    )
    values = torch.randn(
        4,
        input_layout.dim,
        dtype=torch.float32,
        generator=torch.Generator(device=DEVICE).manual_seed(283),
    ) * 0.1

    compiled = torch.compile(executor, backend="aot_eager", fullgraph=True)

    expected = bivector_exp_cpu_reference(
        algebra,
        values,
        input_layout=input_layout,
        output_layout=output_layout,
    )
    actual = compiled(values)

    assert isinstance(executor, BivectorExpExecutor)
    assert executor.executor_family == "closed_biquadratic"
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_planned_bivector_exp_public_call_compiles_fullgraph_after_cache_warm():
    algebra = AlgebraContext(4, 0, device=DEVICE, dtype=torch.float32)
    input_layout = algebra.layout((2,))
    output_layout = algebra.layout((0, 2, 4))
    values = torch.randn(
        4,
        input_layout.dim,
        dtype=torch.float32,
        generator=torch.Generator(device=DEVICE).manual_seed(287),
    ) * 0.1

    def exp_public(x):
        return algebra.exp(x, input_layout=input_layout, output_layout=output_layout)

    algebra.plan_exp(input_layout=input_layout, output_layout=output_layout, dtype=torch.float32, device=DEVICE)
    expected = bivector_exp_cpu_reference(
        algebra,
        values,
        input_layout=input_layout,
        output_layout=output_layout,
    )
    compiled = torch.compile(exp_public, backend="aot_eager", fullgraph=True)
    actual = compiled(values)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
