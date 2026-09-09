# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from tests.planning._grade_plan_helpers import (
    DEVICE,
    ResourceLimits,
    pytest,
    torch,
)

pytestmark = pytest.mark.unit


def test_context_static_product_cost_limits_raise_before_executor_build():
    limits = ResourceLimits(warn_lanes=512, max_lanes=512, warn_pairs=512, max_pairs=64)
    algebra = configured_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32, resource_limits=limits)
    layout = algebra.layout((1,))
    left = torch.zeros(1, layout.dim)
    right = torch.zeros(1, layout.dim)

    with pytest.raises(ValueError, match="pair/interaction footprint"):
        algebra.geometric_product(left, right, left=layout, right=layout)


def test_context_static_layout_cost_limit_raises_before_basis_materialization():
    limits = ResourceLimits(warn_lanes=32, max_lanes=64, warn_pairs=512, max_pairs=1024)
    algebra = configured_algebra(32, 0, 0, device=DEVICE, dtype=torch.float32, resource_limits=limits)

    with pytest.raises(ValueError, match="compact lanes"):
        algebra.layout((1, 2))


def test_high_dimensional_vector_product_plan_avoids_full_basis_enumeration():
    algebra = configured_algebra(
        32, 0, 0, device=DEVICE, dtype=torch.float32, resource_limits=ResourceLimits(max_lanes=4096, max_pairs=100000)
    )
    vector_layout = algebra.layout((1,))
    executor = algebra.plan_product(
        op="geometric_product", left=algebra.layout((1,)), right=algebra.layout((1,)), output=algebra.layout((0, 2))
    )._kernel

    assert vector_layout.dim == 32
    assert executor.output_dim == 1 + 32 * 31 // 2
    assert executor.pair_count == 32 * 32


def test_high_dimensional_vector_product_plan_avoids_dense_lookup_at_int64_limit():
    algebra = configured_algebra(
        63, 0, 0, device=DEVICE, dtype=torch.float32, resource_limits=ResourceLimits(max_lanes=4096, max_pairs=200000)
    )
    executor = algebra.plan_product(
        op="geometric_product", left=algebra.layout((1,)), right=algebra.layout((1,)), output=algebra.layout((0, 2))
    )._kernel

    assert executor.output_dim == 1 + 63 * 62 // 2
    assert executor.pair_count == 63 * 63


def test_high_dimensional_product_plan_reports_int64_bitmask_boundary():
    algebra = configured_algebra(
        64, 0, 0, device=DEVICE, dtype=torch.float32, resource_limits=ResourceLimits(max_lanes=4096, max_pairs=200000)
    )

    with pytest.raises(ValueError, match="Current Torch-backed executors support bitmask tensorization up to n=63"):
        algebra.plan_product(
            op="geometric_product", left=algebra.layout((1,)), right=algebra.layout((1,)), output=algebra.layout((0, 2))
        )


def test_context_static_product_cost_warns_near_configured_limits():
    limits = ResourceLimits(warn_lanes=512, max_lanes=512, warn_pairs=128, max_pairs=4096)
    algebra = configured_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32, resource_limits=limits)
    layout = algebra.layout((1,))
    left = torch.zeros(1, layout.dim)
    right = torch.zeros(1, layout.dim)

    with pytest.warns(RuntimeWarning, match="pair/interaction footprint"):
        values = algebra.geometric_product(left, right, left=layout, right=layout)

    assert values.shape[-1] == algebra.layout((0, 2)).dim


from clifra.core._kernel.configuration import configured_algebra
