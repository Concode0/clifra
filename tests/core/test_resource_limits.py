"""Public budgets reach planning unchanged and survive module ownership."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from clifra import AlgebraConfig, AlgebraContext, CliffordModule, ResourceLimits, make_algebra, make_algebra_from_config
from clifra.core import ResourceLimits as CoreResourceLimits
from clifra.core._kernel.planning.resources import ResourceLimits as PrivateResourceLimits


def test_resource_limits_exports_and_defaults():
    assert ResourceLimits is CoreResourceLimits is PrivateResourceLimits
    limits = make_algebra(2).resource_limits
    assert limits == ResourceLimits()
    assert (limits.warn_lanes, limits.max_lanes, limits.warn_pairs, limits.max_pairs) == (
        2048,
        4096,
        1_000_000,
        8_000_000,
    )


@pytest.mark.parametrize("field", ["warn_lanes", "max_lanes", "warn_pairs", "max_pairs"])
@pytest.mark.parametrize("value", [-1, True, 1.5, "2", None])
def test_resource_limits_reject_invalid_counts(field, value):
    with pytest.raises(ValueError, match="non-negative integers"):
        ResourceLimits(**{field: value})


def test_limits_are_frozen_and_algebra_reference_is_read_only():
    limits = ResourceLimits(max_pairs=0)
    algebra = make_algebra(2, resource_limits=limits)
    with pytest.raises(FrozenInstanceError):
        limits.max_pairs = 100
    with pytest.raises(AttributeError):
        algebra.resource_limits = ResourceLimits()
    with pytest.raises(ValueError, match="max_pairs=0"):
        algebra.plan_product()


@pytest.mark.parametrize("value", [{"max_pairs": 3}, 1, False, object()])
def test_construction_rejects_non_resource_limits(value):
    for construct in (
        lambda: AlgebraContext(2, resource_limits=value),
        lambda: make_algebra(2, resource_limits=value),
        lambda: AlgebraConfig(2, resource_limits=value),
        lambda: make_algebra_from_config({"p": 2, "resource_limits": value}),
    ):
        with pytest.raises(TypeError, match="resource_limits must be a ResourceLimits"):
            construct()


def test_config_and_placement_preserve_budget_identity():
    limits = ResourceLimits(max_lanes=128, max_pairs=1024)
    config = AlgebraConfig.from_mapping({"p": 3, "resource_limits": limits})
    for algebra in (
        AlgebraContext(3, resource_limits=limits),
        make_algebra(3, resource_limits=limits),
        make_algebra_from_config(config),
        make_algebra_from_config({"p": 3}, resource_limits=limits),
    ):
        assert algebra.resource_limits is limits
        assert algebra._planner.limits is limits
        owned = CliffordModule(algebra).to(dtype=torch.float64)
        assert owned.algebra is not algebra
        assert owned.algebra.resource_limits is limits
        assert owned.algebra._planner.limits is limits
        assert algebra.to(dtype=torch.float64).resource_limits is limits


def test_distinct_budgets_control_feasibility_without_mutating_existing_plans():
    allowed = make_algebra(3, resource_limits=ResourceLimits(max_lanes=8))
    denied = make_algebra(3, resource_limits=ResourceLimits(max_lanes=7))
    product = allowed.plan_product()
    with pytest.raises(ValueError, match="max_lanes=7"):
        denied.plan_product()
    value = torch.zeros(8)
    value[0] = 2
    torch.testing.assert_close(product(value, value), 2 * value)
