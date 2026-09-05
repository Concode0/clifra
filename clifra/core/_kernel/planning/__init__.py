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
    DEFAULT_BIVECTOR_EXP_OPTIONS,
    SPECTRAL_LOCAL_TRUNCATION_NOTICE,
    BivectorExpOptions,
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
from .layouts import ProductRequest, build_product_request
from .metric import SignatureNormSquaredPlan, build_signature_norm_squared_plan
from .permutation import PseudoscalarProductPlan, build_pseudoscalar_product_plan
from .planner import GradePlanner
from .policy import (
    DEFAULT_PLANNING_POLICY,
    BoundaryRegion,
    FormulaConstraint,
    FormulaPolicy,
    PlanCandidate,
    PlanFacts,
    PlanningPolicy,
    PolicyCoverageError,
    PolicyEvaluation,
    Polynomial,
    PolynomialTerm,
    RouteRule,
)
from .product import (
    FullTableProductPlan,
    GradeProductPlan,
    build_full_table_product_plan,
    build_grade_product_plan,
    select_product_route,
)
from .resources import ResourceLimits
from .tree import GradePathNode, GradePlanTree, build_grade_plan_tree
from .unary import GradeUnaryOp, GradeUnaryPlan, UnaryRequest, build_unary_request

__all__ = [
    "GradePathNode",
    "GradeProductPlan",
    "FullTableProductPlan",
    "GradePlanTree",
    "GradePlanner",
    "ResourceLimits",
    "PlanningPolicy",
    "FormulaPolicy",
    "PlanCandidate",
    "PlanFacts",
    "PolicyEvaluation",
    "SignatureNormSquaredPlan",
    "PseudoscalarProductPlan",
    "DEFAULT_PLANNING_POLICY",
    "PolynomialTerm",
    "Polynomial",
    "FormulaConstraint",
    "BoundaryRegion",
    "RouteRule",
    "PolicyCoverageError",
    "BivectorExpPlan",
    "BivectorExpOptions",
    "DEFAULT_BIVECTOR_EXP_OPTIONS",
    "SPECTRAL_LOCAL_TRUNCATION_NOTICE",
    "SpectralExpAngleDiagnostics",
    "SpectralExpPreselection",
    "SpectralExpUniformTailStress",
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
    "build_signature_norm_squared_plan",
    "build_pseudoscalar_product_plan",
    "build_product_request",
    "build_unary_request",
    "select_product_route",
]
