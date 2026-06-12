# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Static grade planning and executor family selection."""

from .action import (
    LinearActionPlan,
    PairedBivectorActionPlan,
    VersorActionPlan,
    build_linear_action_plan,
    build_paired_bivector_action_plan,
    build_versor_action_plan,
)
from .exp import (
    DEFAULT_BIVECTOR_EXP_EXECUTION_POLICY,
    SPECTRAL_LOCAL_TRUNCATION_NOTICE,
    BivectorExpExecutionPolicy,
    BivectorExpPlan,
    SpectralExpAngleDiagnostics,
    SpectralExpPreselection,
    SpectralExpUniformTailStress,
    build_bivector_exp_plan,
    format_spectral_exp_uniform_tail_stress,
    spectral_exp_angle_diagnostics,
    spectral_exp_preselection,
    spectral_exp_uniform_tail_stress,
)
from .flow import GradeFlow
from .layouts import ProductRequest, build_product_request
from .metric import NormSquaredPlan, build_norm_squared_plan
from .permutation import DualPlan, build_dual_plan
from .planner import GradePlanner
from .policy import (
    DEFAULT_PLANNING_LIMITS,
    DEFAULT_PRODUCT_EXECUTION_POLICY,
    FULL_TABLE_AUTO_MAX_N,
    FULL_TABLE_EXPLICIT_MAX_N,
    PlanCost,
    PlanningLimits,
    ProductExecutionPolicy,
    ProductExecutorCost,
    estimate_product_executor_cost,
)
from .product import (
    FullTableProductPlan,
    GradeProductPlan,
    build_full_table_product_plan,
    build_grade_product_plan,
)
from .tree import GradePathNode, GradePlanTree, build_grade_plan_tree
from .unary import GradeUnaryOp, GradeUnaryPlan, UnaryRequest, build_unary_request

__all__ = [
    "GradeFlow",
    "GradePathNode",
    "GradeProductPlan",
    "FullTableProductPlan",
    "GradePlanTree",
    "GradePlanner",
    "PlanningLimits",
    "PlanCost",
    "NormSquaredPlan",
    "DualPlan",
    "DEFAULT_PLANNING_LIMITS",
    "DEFAULT_PRODUCT_EXECUTION_POLICY",
    "ProductExecutionPolicy",
    "ProductExecutorCost",
    "BivectorExpPlan",
    "BivectorExpExecutionPolicy",
    "DEFAULT_BIVECTOR_EXP_EXECUTION_POLICY",
    "SPECTRAL_LOCAL_TRUNCATION_NOTICE",
    "SpectralExpAngleDiagnostics",
    "SpectralExpPreselection",
    "SpectralExpUniformTailStress",
    "FULL_TABLE_AUTO_MAX_N",
    "FULL_TABLE_EXPLICIT_MAX_N",
    "LinearActionPlan",
    "PairedBivectorActionPlan",
    "VersorActionPlan",
    "build_linear_action_plan",
    "build_paired_bivector_action_plan",
    "build_versor_action_plan",
    "GradeUnaryOp",
    "GradeUnaryPlan",
    "ProductRequest",
    "UnaryRequest",
    "build_grade_product_plan",
    "build_full_table_product_plan",
    "build_grade_plan_tree",
    "build_bivector_exp_plan",
    "format_spectral_exp_uniform_tail_stress",
    "spectral_exp_angle_diagnostics",
    "spectral_exp_preselection",
    "spectral_exp_uniform_tail_stress",
    "build_norm_squared_plan",
    "build_dual_plan",
    "build_product_request",
    "build_unary_request",
    "estimate_product_executor_cost",
]
