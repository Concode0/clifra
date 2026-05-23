# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Static grade planning and Torch executor lowering."""

from .action import (
    LinearActionPlan,
    PairedBivectorActionPlan,
    VersorActionPlan,
    build_linear_action_plan,
    build_paired_bivector_action_plan,
    build_versor_action_plan,
)
from .decomposition import BivectorDecompositionPlan, build_bivector_decomposition_plan
from .flow import GradeFlow
from .layouts import ProductRequest, build_product_request
from .planner import GradePlanner
from .policy import DEFAULT_PLANNING_LIMITS, DENSE_AUTO_MAX_N, DENSE_EXPLICIT_MAX_N, PlanCost, PlanningLimits
from .product import GradeProductExecutor, GradeProductPlan, build_grade_product_plan
from .tree import GradePathNode, GradePlanTree, build_grade_plan_tree
from .unary import GradeUnaryExecutor, GradeUnaryOp, GradeUnaryPlan, UnaryRequest, build_unary_request

__all__ = [
    "GradeFlow",
    "GradePathNode",
    "GradeProductExecutor",
    "GradeProductPlan",
    "GradePlanTree",
    "GradePlanner",
    "PlanningLimits",
    "PlanCost",
    "DEFAULT_PLANNING_LIMITS",
    "BivectorDecompositionPlan",
    "DENSE_AUTO_MAX_N",
    "DENSE_EXPLICIT_MAX_N",
    "LinearActionPlan",
    "PairedBivectorActionPlan",
    "VersorActionPlan",
    "build_linear_action_plan",
    "build_paired_bivector_action_plan",
    "build_versor_action_plan",
    "GradeUnaryExecutor",
    "GradeUnaryOp",
    "GradeUnaryPlan",
    "ProductRequest",
    "UnaryRequest",
    "build_grade_product_plan",
    "build_grade_plan_tree",
    "build_bivector_decomposition_plan",
    "build_product_request",
    "build_unary_request",
]
