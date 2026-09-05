"""Concrete preparation for the built-in mathematical execution routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from clifra.core.layout import AlgebraSpec, GradeLayout

if TYPE_CHECKING:
    from clifra.core._kernel.planning.exp import BivectorExpOptions, SpectralExpPreselection
    from clifra.core._kernel.planning.layouts import ProductRequest
    from clifra.core._kernel.planning.planner import GradePlanner
    from clifra.core._kernel.planning.tree import GradePlanTree
    from clifra.core._kernel.planning.unary import UnaryRequest
    from clifra.core.algebra import AlgebraContext

    from .registry import Selection

import torch

from clifra.core._kernel.planning.policy import (
    NoAvailableRouteError,
    PlanFacts,
    PolicyCoverageError,
    RouteDecision,
    compose_plan_facts,
)
from clifra.core._kernel.planning.resources import ResourceLimitError, ResourceRequirements
from clifra.core.executors import Assessment, ExecutorRequest, Rejected
from clifra.core.tensors import TensorContract


@dataclass(frozen=True)
class ProductExecutionRequest(ExecutorRequest):
    declaration: ProductRequest
    algebra: AlgebraContext


@dataclass(frozen=True)
class ExpExecutionRequest(ExecutorRequest):
    spec: AlgebraSpec
    preselection: SpectralExpPreselection
    options: BivectorExpOptions
    planner: GradePlanner | None


@dataclass(frozen=True)
class ActionExecutionRequest(ExecutorRequest):
    algebra: AlgebraContext
    grade: int | None = None
    rotor_layout: GradeLayout | None = None
    middle_layout: GradeLayout | None = None


@dataclass(frozen=True)
class UnaryExecutionRequest(ExecutorRequest):
    declaration: UnaryRequest


@dataclass(frozen=True)
class BuiltinPreparation:
    facts: PlanFacts


@dataclass(frozen=True)
class ProductPreparation(BuiltinPreparation):
    declaration: ProductRequest
    tree: GradePlanTree | None


@dataclass(frozen=True)
class ExpPreparation(BuiltinPreparation):
    preselection: SpectralExpPreselection
    left_product: Selection | None = None
    bivector_wedge: Selection | None = None
    grade4_square: Selection | None = None
    bivector_grade4_product: Selection | None = None


@dataclass(frozen=True)
class ActionPreparation(BuiltinPreparation):
    rotor_layout: object = None
    middle_layout: object = None
    exponential: Selection | None = None
    reverse: Selection | None = None
    left_product: object = None
    right_product: Selection | None = None
    norm: Selection | None = None
    involution: Selection | None = None


def product_execution_request(algebra, request):
    return ProductExecutionRequest(
        "product",
        request.op,
        (TensorContract.compact(request.left_layout), TensorContract.compact(request.right_layout)),
        TensorContract.compact(request.output_layout),
        request.dtype,
        request.device,
        request,
        algebra,
    )


def exp_execution_request(spec, device, dtype, output_layout, preselection, *, planner=None, options=None, cache=True):
    from clifra.core._kernel.planning.exp import BivectorExpOptions

    output_layout = spec.layout((0,)) if output_layout is None else output_layout
    inputs = spec.layout((2,)) if spec.n >= 2 else spec.layout(())
    return ExpExecutionRequest(
        "bivector_exp",
        "bivector_exp",
        (TensorContract.compact(inputs),),
        TensorContract.compact(output_layout),
        dtype,
        device,
        spec,
        preselection,
        BivectorExpOptions() if options is None else options,
        planner,
    )


def action_execution_request(
    algebra,
    operation,
    *,
    input_layout,
    output_layout=None,
    parameter_layout=None,
    grade=None,
    rotor_layout=None,
    middle_layout=None,
):
    inputs = TensorContract.compact(input_layout)
    output = TensorContract.compact(output_layout or input_layout)
    contracts = (inputs, TensorContract.compact(parameter_layout)) if parameter_layout is not None else (inputs, None)
    if operation == "multi":
        contracts = (*contracts, None)
    if operation == "paired":
        contracts = (*contracts, contracts[1], None)
    if operation == "sandwich":
        contracts = (inputs, inputs, inputs)
    return ActionExecutionRequest(
        "action",
        operation,
        contracts,
        output,
        algebra.dtype,
        algebra.device,
        algebra,
        grade,
        rotor_layout,
        middle_layout,
    )


def _accepted(facts, preparation):
    return Assessment(
        lanes=facts.resources.lanes,
        pairs=facts.resources.pairs,
        forward_work=facts.forward_work,
        backward_work=facts.backward_work,
        peak_bytes=facts.peak_bytes,
        compile_work=facts.compile_work,
        exact=facts.exact,
        truncated=facts.truncated,
        value_dependent=facts.value_dependent,
        preparation=preparation,
    )


def _product_child(planner, left, right, output, dtype, device, op="geometric_product"):
    from clifra.core._kernel.planning.layouts import ProductRequest
    from clifra.core._kernel.planning.resources import validate_product_request

    declaration = ProductRequest.compact(
        planner.spec,
        op=op,
        left_layout=left,
        right_layout=right,
        output_layout=output,
        dtype=dtype,
        device=device,
    )
    validate_product_request(planner.algebra, declaration)
    request = product_execution_request(planner.algebra, declaration)
    return planner.registry.select(request, planner.policy, planner.limits)


def _unary_child(planner, layout, op, dtype, device):
    from clifra.core._kernel.planning.resources import validate_unary_request
    from clifra.core._kernel.planning.unary import UnaryRequest

    declaration = UnaryRequest.compact(
        planner.spec, op=op, input_layout=layout, output_layout=layout, dtype=dtype, device=device
    )
    validate_unary_request(planner.algebra, declaration)
    request = UnaryExecutionRequest(
        "unary",
        op,
        (TensorContract.compact(layout),),
        TensorContract.compact(layout),
        dtype,
        device,
        declaration,
    )
    return planner.registry.select(request, planner.policy, planner.limits)


def _exp_child(planner, inputs, output, dtype, device):
    from clifra.core._kernel.planning.exp import spectral_exp_preselection

    options = planner.algebra._bivector_exp_options
    preselection = spectral_exp_preselection(
        planner.spec,
        device,
        dtype=dtype,
        max_planes=options.spectral_max_planes,
        tol_abs=options.spectral_tol_abs,
        tol_rel=options.spectral_tol_rel,
        dominant_rel=options.spectral_dominant_rel,
        allow_degenerate=options.spectral_allow_degenerate,
        allow_truncated_degenerate=options.spectral_allow_truncated_degenerate,
    )
    request = exp_execution_request(planner.spec, device, dtype, output, preselection, planner=planner, options=options)
    return planner.registry.select(request, planner.policy, planner.limits)


def _simple_facts(request, pairs=None):
    lanes = max(request.output.layout.dim, *(item.layout.dim for item in request.inputs if item is not None))
    pairs = lanes if pairs is None else pairs
    return PlanFacts(
        pairs, 2 * pairs, pairs * request.dtype.itemsize, pairs, resources=ResourceRequirements(lanes, pairs)
    )


@dataclass(frozen=True)
class BuiltinProvider:
    identity: tuple[str, str]

    def assess(self, request):
        try:
            return self._assess(request)
        except (NoAvailableRouteError, PolicyCoverageError, ResourceLimitError) as error:
            return Rejected(f"required_child_unavailable: {error}")

    def _assess(self, request):
        family, route = self.identity
        if family == "product":
            if request.output.spec.n > 63:
                return Rejected("Current Torch-backed executors support bitmask tensorization up to n=63")
            from .product import assess_product_routes

            candidates = assess_product_routes(
                request.algebra,
                op=request.operation,
                left_layout=request.inputs[0].layout,
                right_layout=request.inputs[1].layout,
                output_layout=request.output.layout,
                dtype=request.dtype,
                device=request.device,
            )
            candidate = next(item for item in candidates if item.route == route)
            if candidate.unavailable_reason:
                return Rejected(candidate.unavailable_reason)
            from clifra.core._kernel.planning.tree import build_grade_plan_tree

            declaration = request.declaration
            tree = (
                build_grade_plan_tree(
                    declaration.spec,
                    op=declaration.op,
                    left_grades=declaration.left_grades,
                    right_grades=declaration.right_grades,
                    output_grades=declaration.output_grades,
                )
                if route == "sparse"
                else None
            )
            return _accepted(candidate.facts, ProductPreparation(candidate.facts, declaration, tree))
        if family == "bivector_exp":
            return _assess_exp(request, route)
        if family == "action":
            return _assess_action(request, route)
        facts = _simple_facts(request)
        return _accepted(facts, BuiltinPreparation(facts))

    def build(self, request, assessment):
        family, route = self.identity
        preparation = assessment.preparation
        if family == "product":
            from clifra.core._kernel.planning.product import (
                build_full_table_product_plan_from_request,
                build_grade_product_plan_from_tree,
            )

            from .product import FullTableProductExecutor, GradeProductExecutor

            if route == "full_table":
                return FullTableProductExecutor(build_full_table_product_plan_from_request(preparation.declaration))
            return GradeProductExecutor(
                build_grade_product_plan_from_tree(preparation.tree, dtype=request.dtype, device=request.device)
            )
        if family == "bivector_exp":
            return _build_exp(request, route, preparation)
        if family == "action":
            return _build_action(request, route, preparation)
        if family == "unary":
            from clifra.core._kernel.planning.unary import build_unary_plan_from_request

            from .unary import GradeUnaryExecutor

            return GradeUnaryExecutor(build_unary_plan_from_request(request.declaration))
        if family == "metric":
            from clifra.core._kernel.planning.metric import build_signature_norm_squared_plan

            from .metric import SignatureNormSquaredExecutor

            return SignatureNormSquaredExecutor(
                build_signature_norm_squared_plan(
                    request.inputs[0].spec,
                    input_layout=request.inputs[0].layout,
                    dtype=request.dtype,
                    device=request.device,
                )
            )
        from clifra.core._kernel.planning.permutation import build_pseudoscalar_product_plan

        from .permutation import PseudoscalarProductExecutor

        return PseudoscalarProductExecutor(
            build_pseudoscalar_product_plan(
                request.inputs[0].spec,
                input_layout=request.inputs[0].layout,
                output_layout=request.output.layout,
                dtype=request.dtype,
                device=request.device,
            )
        )


def _assess_exp(request, route):
    from .exp import assess_bivector_exp_routes

    candidate = next(
        item
        for item in assess_bivector_exp_routes(
            request.spec,
            request.device,
            dtype=request.dtype,
            output_layout=request.output.layout,
            preselection=request.preselection,
        )
        if item.route == route
    )
    if candidate.unavailable_reason:
        return Rejected(candidate.unavailable_reason)
    facts = candidate.facts
    if request.planner is not None:
        reason = facts.resources.rejection_reason(request.planner.limits)
        if reason:
            return Rejected(reason)
    left_product = wedge = square = mixed = None
    # Standalone route diagnostics do not construct executors.
    if request.planner is not None:
        spec, planner = request.spec, request.planner
        inputs, output = request.inputs[0].layout, request.output.layout
        if route in {"left_matrix_exp", "cpu_matrix_exp"}:
            even = spec.layout(range(0, spec.n + 1, 2))
            device = torch.device("cpu") if route == "cpu_matrix_exp" else request.device
            left_product = _product_child(planner, inputs, even, even, request.dtype, device)
        elif route == "closed_biquadratic":
            grade4 = spec.layout((4,))
            wedge = _product_child(planner, inputs, inputs, grade4, request.dtype, request.device, "wedge")
            square = _product_child(planner, grade4, grade4, spec.layout((0,)), request.dtype, request.device)
            mixed = _product_child(planner, inputs, grade4, output, request.dtype, request.device)
        children = tuple(child.facts for child in (left_product, wedge, square, mixed) if child is not None)
        if children:
            facts = compose_plan_facts(facts, *children, peak_bytes=facts.peak_bytes, extensions=facts.extensions)
    return _accepted(facts, ExpPreparation(facts, request.preselection, left_product, wedge, square, mixed))


def _build_exp(request, route, preparation):
    from clifra.core._kernel.planning.exp import build_bivector_exp_plan

    from .exp import BivectorExpExecutor

    options = request.options
    plan = build_bivector_exp_plan(
        request.spec,
        input_layout=request.inputs[0].layout,
        output_layout=request.output.layout,
        dtype=request.dtype,
        device=request.device,
        spectral_max_planes=options.spectral_max_planes,
        spectral_tol_abs=options.spectral_tol_abs,
        spectral_tol_rel=options.spectral_tol_rel,
        spectral_dominant_rel=options.spectral_dominant_rel,
        spectral_allow_degenerate=options.spectral_allow_degenerate,
        spectral_allow_truncated_degenerate=options.spectral_allow_truncated_degenerate,
        route_decision=RouteDecision(route, preparation.facts, "bivector_exp"),
        preselection=preparation.preselection,
    )

    def build(child):
        return None if child is None else child.build()

    return BivectorExpExecutor(
        plan,
        build(preparation.left_product),
        bivector_wedge=build(preparation.bivector_wedge),
        grade4_square=build(preparation.grade4_square),
        bivector_grade4_product=build(preparation.bivector_grade4_product),
    )


def _assess_action(request, route):
    from clifra.core._kernel.basis import expand_output_grades

    from .action import _action_extensions

    algebra, spec = request.algebra, request.inputs[0].spec
    planner = algebra._planner
    inputs, output = request.inputs[0].layout, request.output.layout
    operation, grade = request.operation, request.grade
    parameter = request.inputs[1].layout if request.inputs[1] is not None else None
    full = inputs.dim == spec.dim and output.dim == spec.dim
    if operation == "linear":
        if route != "graded_linear":
            return Rejected("requires_linear_action")
        facts = _simple_facts(request, inputs.dim * output.dim)
        return _accepted(facts, ActionPreparation(facts))
    if operation == "sandwich":
        if route != "full_action_matrix" or not full:
            return Rejected("requires_full_sandwich")
        facts = _simple_facts(request, spec.dim**2)
        return _accepted(facts, ActionPreparation(facts))
    paired = operation == "paired"
    grade = 2 if paired else grade
    if grade not in (1, 2) or parameter.grades != (grade,):
        return Rejected("invalid_action_parameter_grade")
    allowed = (
        route == "vector_matrix"
        and not paired
        and (grade == 1 or inputs.grades == output.grades == (1,))
        or route == "full_action_matrix"
        and full
        or route == "rotor_product"
        and not paired
        and grade == 2
        or route == "paired_rotor_product"
        and paired
    )
    if not allowed:
        return Rejected("unsupported_action_domain")
    pairs = spec.n**2 if route == "vector_matrix" else (spec.dim**2 if route == "full_action_matrix" else 0)
    lanes = spec.dim if route == "full_action_matrix" else max(inputs.dim, output.dim, parameter.dim)
    reason = ResourceRequirements(lanes, pairs).rejection_reason(planner.limits)
    if reason:
        return Rejected(reason)
    rotor = middle = exponential = reverse = left = right = norm = involution = None
    if route != "vector_matrix":
        if grade == 2:
            rotor = spec.layout(range(0, spec.n + 1, 2))
            exponential = _exp_child(planner, parameter, rotor, request.dtype, request.device)
            reverse = _unary_child(planner, rotor, "reverse", request.dtype, request.device)
            if route != "full_action_matrix":
                middle = spec.layout(expand_output_grades(rotor.grades, inputs.grades, spec.n, op="geometric_product"))
                left = _product_child(planner, rotor, inputs, middle, request.dtype, request.device)
                right = _product_child(planner, middle, rotor, output, request.dtype, request.device)
        else:
            norm_request = ExecutorRequest(
                "metric",
                "signature_norm_squared",
                (request.inputs[1],),
                TensorContract.compact(spec.layout((0,))),
                request.dtype,
                request.device,
            )
            norm = planner.registry.select(norm_request, planner.policy, planner.limits)
            involution = _unary_child(planner, parameter, "grade_involution", request.dtype, request.device)
            reverse = _unary_child(planner, parameter, "reverse", request.dtype, request.device)
    work = spec.n**3 + inputs.dim * output.dim if route == "vector_matrix" else pairs
    facts = PlanFacts(
        work, 2 * work, pairs * request.dtype.itemsize, pairs, resources=ResourceRequirements(lanes, pairs)
    )
    children = [child.facts for child in (exponential, reverse, left, right, norm, involution) if child is not None]
    if paired and exponential is not None:
        children.append(exponential.facts)
    intermediate = (2 * rotor.dim if rotor is not None else 0) + (middle.dim if middle is not None else 0) + output.dim
    facts = compose_plan_facts(
        facts,
        *children,
        peak_bytes=(pairs + intermediate) * request.dtype.itemsize,
        extensions=_action_extensions(
            algebra,
            input_layout=inputs,
            output_layout=output,
            parameter_layout=parameter,
            intermediate_lanes=middle.dim if middle is not None else lanes,
        ),
    )
    return _accepted(
        facts, ActionPreparation(facts, rotor, middle, exponential, reverse, left, right, norm, involution)
    )


def _operation(child):
    if child is None:
        return None
    from clifra.core.operation import PlannedOperation

    method = "forward_compact" if child.family in {"product", "unary"} else "forward"
    return PlannedOperation(child.build(), child.request.inputs, child.request.output, method=method)


def _build_action(request, route, preparation):
    from .action import (
        ActionComponents,
        FullSandwichActionExecutor,
        GradedLinearActionExecutor,
        MultiVersorActionExecutor,
        PairedBivectorActionExecutor,
        VersorActionExecutor,
    )

    inputs, output = request.inputs[0].layout, request.output.layout
    if request.operation == "linear":
        return GradedLinearActionExecutor(input_layout=inputs, output_layout=output)
    if request.operation == "sandwich":
        return FullSandwichActionExecutor.from_layout(inputs, device=request.device, dtype=request.dtype)
    components = ActionComponents(
        preparation.rotor_layout,
        preparation.middle_layout,
        _operation(preparation.exponential),
        _operation(preparation.reverse),
        _operation(preparation.left_product),
        _operation(preparation.right_product),
        _operation(preparation.norm),
        _operation(preparation.involution),
    )
    if request.operation == "paired":
        return PairedBivectorActionExecutor(
            request.algebra,
            input_layout=inputs,
            output_layout=output,
            parameter_layout=request.inputs[1].layout,
            rotor_layout=preparation.rotor_layout,
            middle_layout=preparation.middle_layout or request.middle_layout,
            route=route,
            components=components,
        )
    executor = MultiVersorActionExecutor if request.operation == "multi" else VersorActionExecutor
    return executor(
        request.algebra,
        grade=request.grade,
        input_layout=inputs,
        output_layout=output,
        parameter_layout=request.inputs[1].layout,
        route=route,
        components=components,
    )


def builtin_providers():
    return tuple(
        BuiltinProvider((family, route))
        for family, routes in (
            ("product", ("full_table", "sparse")),
            ("unary", ("grade_map",)),
            ("metric", ("diagonal",)),
            ("permutation", ("pseudoscalar",)),
            (
                "bivector_exp",
                ("closed_simple", "closed_biquadratic", "spectral_local", "left_matrix_exp", "cpu_matrix_exp"),
            ),
            (
                "action",
                ("vector_matrix", "rotor_product", "full_action_matrix", "paired_rotor_product", "graded_linear"),
            ),
        )
        for route in routes
    )
