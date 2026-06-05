# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Core mathematical package for Geometric Algebra.

Provides Clifford algebra hosts, layout contracts, planner/executor utilities,
metric functions, bivector decomposition, and signature search utilities.

The ``clifra.core.analysis`` sub-package (``MetricSearch``, ``GeodesicFlow``,
``GeometricAnalyzer``, etc.) is **lazily imported** - it is not loaded
until first access, keeping ``import clifra.core`` lightweight.
"""

from .config import AlgebraConfig, make_algebra, make_algebra_from_config
from .execution import (
    BivectorExpExecutor,
    DualExecutor,
    FullSandwichActionHandle,
    FullTableProductExecutor,
    GeometricAttentionScoreExecutor,
    GradedLinearActionExecutor,
    GradeProductExecutor,
    GradeUnaryExecutor,
    MultiVersorActionExecutor,
    MultiVersorActionHandle,
    NormSquaredExecutor,
    PairedBivectorActionExecutor,
    PairedBivectorActionHandle,
    ProductPlanHandle,
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
    normalize_grades,
    operation_coefficient,
    operation_may_be_nonzero,
    product_output_grades,
    reverse_sign,
)
from .foundation.device import DeviceConfig, dtype_name, optional_dtype, resolve_device, resolve_dtype
from .foundation.layout import AlgebraSpec, GradeLayout
from .foundation.module import AlgebraLike, CliffordModule
from .foundation.numerics import covariance_regularizer, eps_for, eps_like, signed_clamp_min
from .foundation.validation import check_channels, check_multivector
from .planning.exp import BivectorExpPlan, build_bivector_exp_plan
from .planning.flow import GradeFlow
from .planning.layouts import ProductRequest, build_product_request
from .planning.metric import NormSquaredPlan, build_norm_squared_plan
from .planning.permutation import DualPlan, build_dual_plan
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
from .runtime.decomposition import (
    ExpPolicy,
    differentiable_invariant_decomposition,
    ga_power_iteration,
)
from .runtime.metric import (
    clifford_conjugate,
    geometric_distance,
    grade_hermitian_norm,
    grade_purity,
    hermitian_angle,
    hermitian_distance,
    hermitian_grade_spectrum,
    hermitian_inner_product,
    hermitian_norm,
    induced_norm,
    inner_product,
    mean_active_grade,
    signature_norm_squared,
    signature_trace_form,
)
from .storage import (
    ExecutionBoundary,
    ExecutorPath,
    LaneFormat,
    LayerLayout,
    ValueLayout,
    active_values,
    compact_grade_norms,
    hermitian_signs,
    layout_for_values,
    materialize_full,
    metric_self_signs,
    resolve_layer_layout,
    resolve_layer_layout_contract,
    resolve_operand_layout,
    resolve_output_boundary,
    resolve_value_layout,
)

__all__ = [
    # algebra
    "AlgebraContext",
    "AlgebraConfig",
    "AlgebraLike",
    "CliffordModule",
    "GeometricAttentionScoreExecutor",
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
    "LaneFormat",
    "ExecutorPath",
    "ValueLayout",
    "LayerLayout",
    "ExecutionBoundary",
    "BivectorExpExecutor",
    "BivectorExpPlan",
    "PlanningLimits",
    "PlanCost",
    "DEFAULT_PLANNING_LIMITS",
    "make_algebra",
    "make_algebra_from_config",
    # device / validation
    "DeviceConfig",
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
    "resolve_value_layout",
    "resolve_operand_layout",
    "resolve_output_boundary",
    "layout_for_values",
    "resolve_layer_layout",
    "resolve_layer_layout_contract",
    # metric
    "inner_product",
    "induced_norm",
    "geometric_distance",
    "grade_purity",
    "mean_active_grade",
    "clifford_conjugate",
    "hermitian_inner_product",
    "hermitian_norm",
    "hermitian_distance",
    "hermitian_angle",
    "grade_hermitian_norm",
    "hermitian_grade_spectrum",
    "signature_trace_form",
    "signature_norm_squared",
    "active_values",
    "compact_grade_norms",
    "hermitian_signs",
    "materialize_full",
    "metric_self_signs",
    # decomposition
    "ExpPolicy",
    "ga_power_iteration",
    "differentiable_invariant_decomposition",
    # static executor planning
    "GradeProductOp",
    "FullTableProductExecutor",
    "FullTableProductPlan",
    "GradeProductExecutor",
    "GradeProductPlan",
    "NormSquaredExecutor",
    "NormSquaredPlan",
    "DualExecutor",
    "DualPlan",
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
    "build_norm_squared_plan",
    "build_dual_plan",
    "build_bivector_exp_plan",
    "build_grade_plan_tree",
    "build_product_request",
    "build_unary_request",
    "expand_output_grades",
    "geometric_product_output_grades",
    "normalize_grades",
    "operation_coefficient",
    "operation_may_be_nonzero",
    "product_output_grades",
    "reverse_sign",
    # analysis (lazy)
    "MetricSearch",
    "GeodesicFlow",
    "DimensionLifter",
    "GeometricAnalyzer",
    "AnalysisReport",
]

# Lazy imports for the analysis sub-package.
# These names are only resolved when first accessed, keeping
# ``import clifra.core`` fast and avoiding circular-import issues.
_ANALYSIS_NAMES = {
    "MetricSearch",
    "GeodesicFlow",
    "DimensionLifter",
    "GeometricAnalyzer",
    "AnalysisReport",
    "compute_uncertainty_and_alignment",
    "SignatureSearchAnalyzer",
    "EffectiveDimensionAnalyzer",
    "SpectralAnalyzer",
    "SymmetryDetector",
    "CommutatorAnalyzer",
    "StatisticalSampler",
    "SamplingConfig",
    "AnalysisConfig",
}


def __getattr__(name: str):
    if name in _ANALYSIS_NAMES:
        from . import analysis as _analysis  # noqa: F811

        obj = getattr(_analysis, name)
        # Cache on the module to avoid repeated __getattr__ calls
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
