# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Core mathematical package for Geometric Algebra.

Provides Clifford algebra hosts, layout contracts, planner/executor utilities,
metric functions, and signature search utilities.

Analysis tools live under ``clifra.core.analysis`` and should be imported from
their owning modules.
"""

from .config import AlgebraConfig, make_algebra, make_algebra_from_config
from .execution import (
    BivectorExpExecutor,
    FullSandwichActionExecutor,
    FullSandwichActionHandle,
    FullTableProductExecutor,
    GeometricAttentionScoreExecutor,
    GradedLinearActionExecutor,
    GradeProductExecutor,
    GradeUnaryExecutor,
    MultiVersorActionExecutor,
    MultiVersorActionHandle,
    PairedBivectorActionExecutor,
    PairedBivectorActionHandle,
    ProductPlanHandle,
    PseudoscalarProductExecutor,
    SignatureNormSquaredExecutor,
    UnaryPlanHandle,
    VersorActionExecutor,
    VersorActionHandle,
)
from .formatting import Multivector, basis_blade_label, format_multivector
from .foundation.basis import (
    GradeProductOp,
    basis_indices_for_grades,
    basis_product,
    expand_output_grades,
    geometric_product_output_grades,
    normalize_grade_product_op,
    normalize_grades,
    operation_coefficient,
    operation_may_be_nonzero,
    product_output_grades,
    reverse_sign,
)
from .foundation.device import dtype_name, optional_dtype, resolve_device, resolve_dtype
from .foundation.layout import AlgebraSpec, GradeLayout
from .foundation.module import AlgebraLike, CliffordModule
from .foundation.numerics import covariance_regularizer, eps_for, eps_like, signed_clamp_min
from .foundation.validation import check_channels, check_multivector
from .planning.exp import (
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
from .planning.flow import GradeFlow
from .planning.layouts import ProductRequest, build_product_request
from .planning.metric import (
    SignatureNormSquaredPlan,
    build_signature_norm_squared_plan,
)
from .planning.permutation import PseudoscalarProductPlan, build_pseudoscalar_product_plan
from .planning.planner import GradePlanner
from .planning.policy import DEFAULT_PLANNING_LIMITS, PlanCost, PlanningLimits
from .planning.product import (
    FullTableProductPlan,
    GradeProductPlan,
    build_full_table_product_plan,
    build_grade_product_plan,
)
from .planning.tree import GradePathNode, GradePlanTree, build_grade_plan_tree
from .planning.unary import GradeUnaryOp, GradeUnaryPlan, UnaryRequest, build_unary_request
from .runtime.algebra import AlgebraContext
from .runtime.metric import (
    clifford_conjugate,
    conjugate_form_distance_like,
    conjugate_form_magnitude,
    conjugate_grade_magnitude_spectrum,
    conjugate_scalar_form,
    conjugate_scalar_form_signs,
    grade_purity,
    lane_distance,
    lane_dot_product,
    lane_energy,
    lane_grade_distribution,
    lane_grade_energy,
    lane_grade_norms,
    lane_norm,
    mean_grade,
    scalar_product,
    signature_distance_like,
    signature_magnitude,
    signature_norm_squared,
    signature_trace_form,
)
from .runtime.tensors import (
    LaneStorage,
    TensorContract,
    canonical_values,
    check_layout_spec,
    compact_values,
    infer_contract,
    metric_self_signs,
    normalize_lane_storage,
    resolve_contract,
    resolve_layout,
)

__all__ = [
    # algebra
    "AlgebraContext",
    "AlgebraConfig",
    "AlgebraLike",
    "CliffordModule",
    "GeometricAttentionScoreExecutor",
    "FullSandwichActionExecutor",
    "FullSandwichActionHandle",
    "GradedLinearActionExecutor",
    "VersorActionExecutor",
    "MultiVersorActionExecutor",
    "PairedBivectorActionExecutor",
    "ProductPlanHandle",
    "UnaryPlanHandle",
    "VersorActionHandle",
    "MultiVersorActionHandle",
    "PairedBivectorActionHandle",
    "Multivector",
    "AlgebraSpec",
    "GradeLayout",
    "GradePlanner",
    "LaneStorage",
    "TensorContract",
    "BivectorExpExecutor",
    "BivectorExpPlan",
    "BivectorExpExecutionPolicy",
    "DEFAULT_BIVECTOR_EXP_EXECUTION_POLICY",
    "SPECTRAL_LOCAL_TRUNCATION_NOTICE",
    "SpectralExpAngleDiagnostics",
    "SpectralExpPreselection",
    "SpectralExpUniformTailStress",
    "PlanningLimits",
    "PlanCost",
    "DEFAULT_PLANNING_LIMITS",
    "make_algebra",
    "make_algebra_from_config",
    # device / validation
    "dtype_name",
    "optional_dtype",
    "resolve_device",
    "resolve_dtype",
    "eps_for",
    "eps_like",
    "signed_clamp_min",
    "covariance_regularizer",
    "check_multivector",
    "check_channels",
    "basis_blade_label",
    "format_multivector",
    "normalize_lane_storage",
    "check_layout_spec",
    "resolve_layout",
    "resolve_contract",
    "infer_contract",
    # metric
    "scalar_product",
    "signature_magnitude",
    "signature_distance_like",
    "grade_purity",
    "mean_grade",
    "clifford_conjugate",
    "lane_dot_product",
    "lane_energy",
    "lane_norm",
    "lane_distance",
    "lane_grade_energy",
    "lane_grade_norms",
    "lane_grade_distribution",
    "conjugate_scalar_form_signs",
    "conjugate_scalar_form",
    "conjugate_form_magnitude",
    "conjugate_form_distance_like",
    "conjugate_grade_magnitude_spectrum",
    "signature_trace_form",
    "signature_norm_squared",
    "compact_values",
    "canonical_values",
    "metric_self_signs",
    # static executor planning
    "GradeProductOp",
    "FullTableProductExecutor",
    "FullTableProductPlan",
    "GradeProductExecutor",
    "GradeProductPlan",
    "SignatureNormSquaredExecutor",
    "SignatureNormSquaredPlan",
    "PseudoscalarProductExecutor",
    "PseudoscalarProductPlan",
    "GradePathNode",
    "GradePlanTree",
    "GradeFlow",
    "ProductRequest",
    "GradeUnaryExecutor",
    "GradeUnaryOp",
    "GradeUnaryPlan",
    "UnaryRequest",
    "basis_indices_for_grades",
    "basis_product",
    "build_full_table_product_plan",
    "build_grade_product_plan",
    "build_signature_norm_squared_plan",
    "build_pseudoscalar_product_plan",
    "build_bivector_exp_plan",
    "format_spectral_exp_uniform_tail_stress",
    "spectral_exp_angle_diagnostics",
    "spectral_exp_preselection",
    "spectral_exp_uniform_tail_stress",
    "build_grade_plan_tree",
    "build_product_request",
    "build_unary_request",
    "expand_output_grades",
    "geometric_product_output_grades",
    "normalize_grade_product_op",
    "normalize_grades",
    "operation_coefficient",
    "operation_may_be_nonzero",
    "product_output_grades",
    "reverse_sign",
]
