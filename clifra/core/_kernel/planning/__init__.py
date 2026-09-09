# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Static grade planning and executor family selection."""

from .action import (
    LinearActionPlan,
    VersorActionPlan,
    build_linear_action_plan,
    build_versor_action_plan,
)
from .exp import (
    BivectorExpPlan,
    build_bivector_exp_plan,
)
from .layouts import ProductRequest
from .metric import SignatureNormSquaredPlan, build_signature_norm_squared_plan
from .permutation import PseudoscalarProductPlan, build_pseudoscalar_product_plan
from .planner import GradePlanner
from .policy import (
    DEFAULT_PLANNING_POLICY,
    ActionFacts,
    BivectorExpFacts,
    PlanCandidate,
    PlanningPolicy,
    PolicyCoverageError,
    PolicyEvaluation,
    ProductFacts,
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
from .unary import GradeUnaryOp, GradeUnaryPlan, UnaryRequest

__all__ = [
    "GradePathNode",
    "GradeProductPlan",
    "FullTableProductPlan",
    "GradePlanTree",
    "GradePlanner",
    "ResourceLimits",
    "PlanningPolicy",
    "PlanCandidate",
    "ProductFacts",
    "ActionFacts",
    "BivectorExpFacts",
    "PolicyEvaluation",
    "SignatureNormSquaredPlan",
    "PseudoscalarProductPlan",
    "DEFAULT_PLANNING_POLICY",
    "PolicyCoverageError",
    "BivectorExpPlan",
    "LinearActionPlan",
    "VersorActionPlan",
    "build_linear_action_plan",
    "build_versor_action_plan",
    "GradeUnaryOp",
    "GradeUnaryPlan",
    "ProductRequest",
    "UnaryRequest",
    "build_grade_product_plan",
    "build_full_table_product_plan",
    "build_grade_plan_tree",
    "build_bivector_exp_plan",
    "build_signature_norm_squared_plan",
    "build_pseudoscalar_product_plan",
    "select_product_route",
]
