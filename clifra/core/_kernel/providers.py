"""Bridge planning assessments and prepared children to built-in executors.

Assessment may select required child routes but allocates no execution buffers.
Construction consumes the retained preparation without selecting routes again.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from clifra.core.layout import AlgebraSpec

if TYPE_CHECKING:
    from clifra.core._kernel.planning.layouts import ProductRequest
    from clifra.core._kernel.planning.planner import GradePlanner
    from clifra.core._kernel.planning.tree import GradePlanTree
    from clifra.core._kernel.planning.unary import UnaryRequest
    from clifra.core.algebra import AlgebraContext

    from .routing import Selection


from clifra.core._kernel.planning.policy import (
    ActionFacts,
    BivectorExpFacts,
    NoAvailableRouteError,
    PolicyCoverageError,
    ProductFacts,
)
from clifra.core._kernel.planning.resources import ResourceRequirements
from clifra.core._kernel.planning.work import ActionWorkProfile, BivectorExpWorkProfile, action_lift_profile
from clifra.core.executors import Assessment, ExecutorRequest, Rejected
from clifra.core.tensors import TensorContract


@dataclass(frozen=True)
class ProductExecutionRequest(ExecutorRequest):
    declaration: ProductRequest


@dataclass(frozen=True)
class ExpExecutionRequest(ExecutorRequest):
    spec: AlgebraSpec
    planner: GradePlanner | None


@dataclass(frozen=True)
class ActionExecutionRequest(ExecutorRequest):
    algebra: AlgebraContext
    grade: int | None = None


@dataclass(frozen=True)
class UnaryExecutionRequest(ExecutorRequest):
    declaration: UnaryRequest


@dataclass(frozen=True)
class BuiltinPreparation:
    facts: ProductFacts | ActionFacts | BivectorExpFacts | None


@dataclass(frozen=True)
class ProductPreparation(BuiltinPreparation):
    declaration: ProductRequest
    tree: GradePlanTree | None


@dataclass(frozen=True)
class ExpPreparation(BuiltinPreparation):
    left_product: Selection | None = None
    bivector_wedge: Selection | None = None
    grade4_square: Selection | None = None
    bivector_grade4_product: Selection | None = None
    polynomial_12: tuple = ()
    polynomial_18: tuple = ()
    full_polynomial_12: tuple = ()
    full_polynomial_18: tuple = ()
    square: Selection | None = None


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


def product_execution_request(request):
    return ProductExecutionRequest(
        "product",
        request.op,
        (TensorContract.compact(request.left_layout), TensorContract.compact(request.right_layout)),
        TensorContract.compact(request.output_layout),
        request.dtype,
        request.device,
        request,
    )


def exp_execution_request(spec, device, dtype, output_layout, *, planner=None):
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
):
    inputs = TensorContract.compact(input_layout)
    output = TensorContract.compact(output_layout or input_layout)
    contracts = (inputs, TensorContract.compact(parameter_layout)) if parameter_layout is not None else (inputs, None)
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
    )


def _accepted(preparation, resources):
    return Assessment(
        lanes=resources.lanes,
        pairs=resources.pairs,
        preparation=preparation,
    )


def _product_child(planner, left, right, output, dtype, device, op="geometric_product"):
    from clifra.core._kernel.planning.layouts import ProductRequest

    declaration = ProductRequest.compact(
        planner.spec,
        op=op,
        left_layout=left,
        right_layout=right,
        output_layout=output,
        dtype=dtype,
        device=device,
    )
    request = product_execution_request(declaration)
    return planner.router.select(request, planner.policy, planner.limits, warn_selected=False)


def _unary_child(planner, layout, op, dtype, device):
    from clifra.core._kernel.planning.unary import UnaryRequest

    declaration = UnaryRequest.compact(
        planner.spec, op=op, input_layout=layout, output_layout=layout, dtype=dtype, device=device
    )
    request = UnaryExecutionRequest(
        "unary",
        op,
        (TensorContract.compact(layout),),
        TensorContract.compact(layout),
        dtype,
        device,
        declaration,
    )
    return planner.router.select(request, planner.policy, planner.limits, warn_selected=False)


def _exp_child(planner, inputs, output, dtype, device):
    request = exp_execution_request(planner.spec, device, dtype, output, planner=planner)
    return planner.router.select(request, planner.policy, planner.limits, warn_selected=False)


def _simple_requirements(request, pairs=None):
    lanes = max(request.output.layout.dim, *(item.layout.dim for item in request.inputs if item is not None))
    pairs = lanes if pairs is None else pairs
    return ResourceRequirements(lanes, pairs)


def _combined_requirements(base, children):
    # Child plan buffers coexist in the constructed parent. Match build-time
    # module reuse so repeated Horner contracts contribute resident storage once.
    unique = {}
    for child in children:
        key = (child.family, child.route, child.request.operation, child.request.inputs, child.request.output)
        unique.setdefault(key, child)
    return ResourceRequirements(
        max((base.lanes, *(child.assessment.lanes for child in unique.values()))),
        base.pairs + sum(child.assessment.pairs for child in unique.values()),
    )


@dataclass(frozen=True)
class BuiltinProvider:
    identity: tuple[str, str]

    def assess(self, request):
        try:
            return self._assess(request)
        except (NoAvailableRouteError, PolicyCoverageError) as error:
            return Rejected(f"required_child_unavailable: {error}")

    def _assess(self, request):
        family, route = self.identity
        if family == "product":
            if request.output.spec.n > 63:
                return Rejected("Current Torch-backed executors support bitmask tensorization up to n=63")
            from .planning.product import assess_product_routes

            candidates = assess_product_routes(
                op=request.operation,
                left_layout=request.inputs[0].layout,
                right_layout=request.inputs[1].layout,
                output_layout=request.output.layout,
            )
            candidate = next(item for item in candidates if item.route == route)
            if candidate.unavailable_reason:
                return Rejected(candidate.unavailable_reason)
            declaration = request.declaration
            preparation = ProductPreparation(candidate.facts, declaration, candidate.tree)
            return _accepted(preparation, candidate.resources)
        if family == "bivector_exp":
            return _assess_exp(request, route)
        if family == "action":
            return _assess_action(request, route)
        preparation = BuiltinPreparation(None)
        return _accepted(preparation, _simple_requirements(request))

    def build(self, request, assessment):
        family, route = self.identity
        preparation = assessment.preparation
        if family == "product":
            from clifra.core._kernel.planning.product import (
                build_full_table_product_plan_from_request,
                build_grade_product_plan_from_tree,
            )

            from .execution.product import FullTableProductExecutor, GradeProductExecutor

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

            from .execution.unary import GradeUnaryExecutor

            return GradeUnaryExecutor(build_unary_plan_from_request(request.declaration))
        if family == "metric":
            from clifra.core._kernel.planning.metric import build_signature_norm_squared_plan

            from .execution.metric import SignatureNormSquaredExecutor

            return SignatureNormSquaredExecutor(
                build_signature_norm_squared_plan(
                    request.inputs[0].spec,
                    input_layout=request.inputs[0].layout,
                    dtype=request.dtype,
                    device=request.device,
                )
            )
        from clifra.core._kernel.planning.permutation import build_pseudoscalar_product_plan

        from .execution.permutation import PseudoscalarProductExecutor

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
    from clifra.core._kernel.planning.exp import taylor_layouts

    from .planning.exp import assess_bivector_exp_route

    candidate = assess_bivector_exp_route(
        request.spec,
        request.device,
        dtype=request.dtype,
        output_layout=request.output.layout,
        route=route,
    )
    if candidate.unavailable_reason:
        return Rejected(candidate.unavailable_reason)
    facts = candidate.facts
    left_product = wedge = square4 = mixed = square = None
    polynomial_12 = polynomial_18 = full_polynomial_12 = full_polynomial_18 = ()
    if request.planner is not None:
        spec, planner = request.spec, request.planner
        inputs, output = request.inputs[0].layout, request.output.layout
        even = spec.layout(range(0, spec.n + 1, 2))
        route_device = "cpu" if route == "closed" and spec.n >= 4 and request.device.type == "mps" else request.device

        def child(left, right, out, op="geometric_product"):
            return _product_child(planner, left, right, out, request.dtype, route_device, op)

        if route == "left_matrix_exp":
            # The measured matrix regime uses CPU, including small MPS inputs.
            left_product = _product_child(planner, inputs, even, even, request.dtype, "cpu")
        elif route == "closed" and spec.n >= 4:
            grade4 = spec.layout((4,))
            wedge = child(inputs, inputs, grade4, "wedge")
            square4 = child(grade4, grade4, spec.layout((0,)))
            mixed = child(inputs, grade4, output)
        elif route == "taylor":

            def polynomial_children(out, degree):
                layouts = taylor_layouts(spec, out, degree)
                return tuple(child(inputs, left, right) for left, right in zip(layouts, layouts[1:]))

            polynomial_12 = polynomial_children(output, 12)
            polynomial_18 = polynomial_children(output, 18)
            full_polynomial_12 = polynomial_12 if output == even else polynomial_children(even, 12)
            full_polynomial_18 = polynomial_18 if output == even else polynomial_children(even, 18)
            # x*x equals the symmetric Clifford product. The shared planner
            # removes anticommuting basis pairs before allocating intermediates.
            square = child(even, even, even, "symmetric_product")
        children = tuple(
            c
            for c in (
                left_product,
                wedge,
                square4,
                mixed,
                square,
                *polynomial_12,
                *polynomial_18,
                *full_polynomial_12,
                *full_polynomial_18,
            )
            if c is not None
        )
    else:
        children = ()
    if request.planner is not None:
        spec, output = request.spec, request.output.layout
        even = spec.layout(range(0, spec.n + 1, 2))
        degree = 18 if request.dtype == torch.float64 else 12
        plain = polynomial_18 if degree == 18 else polynomial_12
        scaled = full_polynomial_18 if degree == 18 else full_polynomial_12
        fixed = {
            "closed": (wedge, square4, mixed),
            "left_matrix_exp": (left_product,),
            "taylor": (),
        }[route]
        facts = replace(
            facts,
            work_profile=BivectorExpWorkProfile(
                route=route,
                bivector_width=request.inputs[0].layout.dim,
                grade4_width=spec.layout((4,)).dim if route == "closed" and spec.n >= 4 else 0,
                even_width=even.dim,
                output_width=output.dim,
                fixed_products=tuple(item.facts.work_profile for item in fixed if item is not None),
                plain_products=tuple(item.facts.work_profile for item in plain),
                scaled_products=tuple(item.facts.work_profile for item in scaled),
                square_product=None if square is None else square.facts.work_profile,
                plain_stage_widths=tuple(layout.dim for layout in taylor_layouts(spec, output, degree)[1:])
                if route == "taylor"
                else (),
                scaled_stage_widths=tuple(layout.dim for layout in taylor_layouts(spec, even, degree)[1:])
                if route == "taylor"
                else (),
            ),
        )
    preparation = ExpPreparation(
        facts,
        left_product,
        wedge,
        square4,
        mixed,
        polynomial_12,
        polynomial_18,
        full_polynomial_12,
        full_polynomial_18,
        square,
    )
    return _accepted(preparation, _combined_requirements(candidate.resources, children))


def _build_exp(request, route, preparation):
    from clifra.core._kernel.planning.exp import build_bivector_exp_plan

    from .execution.exp import BivectorExpExecutor, TaylorPolynomial

    plan = build_bivector_exp_plan(
        request.spec,
        input_layout=request.inputs[0].layout,
        output_layout=request.output.layout,
        dtype=request.dtype,
        device=request.device,
        route=route,
    )
    cache = {}

    def build(child):
        if child is None:
            return None
        # Repeated Horner layouts share the same optimized product executor.
        key = (child.route, child.request.operation, child.request.inputs, child.request.output)
        if key not in cache:
            cache[key] = child.build()
        return cache[key]

    taylor = None
    taylor_work = (0, 0, 0)
    schedules = {
        "output_12": preparation.polynomial_12,
        "output_18": preparation.polynomial_18,
        "full_12": preparation.full_polynomial_12,
        "full_18": preparation.full_polynomial_18,
    }
    if any(schedules.values()):
        unique_children = []
        child_positions = {}
        schedule_positions = {}
        schedule_layouts = {}
        for name, children in schedules.items():
            positions = []
            layouts = []
            for child in children:
                key = (child.route, child.request.operation, child.request.inputs, child.request.output)
                if key not in child_positions:
                    child_positions[key] = len(unique_children)
                    unique_children.append(child)
                positions.append(child_positions[key])
                layouts.append(child.request.output.layout)
            schedule_positions[name] = tuple(positions)
            schedule_layouts[name] = tuple(layouts)
        taylor = TaylorPolynomial(
            [build(child) for child in unique_children],
            schedule_positions,
            schedule_layouts,
            dtype=request.dtype,
            device=request.device,
        )
        degree = 18 if request.dtype == torch.float64 else 12

        def product_proxy(child):
            # Preserve the eager-CPU partition heuristic's old P + indexed-P proxy exactly.
            profile = child.facts.work_profile
            return profile.bulk + (
                profile.bulk if profile.route == "sparse" and profile.output > 1 and profile.bulk > 0 else 0
            )

        def schedule_work(children):
            return sum(product_proxy(child) for child in children)

        plain = preparation.polynomial_18 if degree == 18 else preparation.polynomial_12
        full = preparation.full_polynomial_18 if degree == 18 else preparation.full_polynomial_12
        taylor_work = (
            schedule_work(plain),
            schedule_work(full),
            product_proxy(preparation.square),
        )

    return BivectorExpExecutor(
        plan,
        build(preparation.left_product),
        bivector_wedge=build(preparation.bivector_wedge),
        grade4_square=build(preparation.grade4_square),
        bivector_grade4_product=build(preparation.bivector_grade4_product),
        polynomial=taylor,
        square=build(preparation.square),
        taylor_work=taylor_work,
    )


def _assess_action(request, route):
    from clifra.core._kernel.basis import expand_output_grades

    from .planning.action import _linear_action_structure

    algebra, spec = request.algebra, request.inputs[0].spec
    planner = algebra._planner
    inputs, output = request.inputs[0].layout, request.output.layout
    operation, grade = request.operation, request.grade
    parameter = request.inputs[1].layout if request.inputs[1] is not None else None
    full = inputs.dim == spec.dim and output.dim == spec.dim
    if request.device.type == "mps" and request.dtype == torch.float64:
        return Rejected("mps_does_not_support_float64_output")
    if route in {"graded_linear", "vector_matrix"}:
        needs_det = any(g >= 4 for g in set(inputs.grades) & set(output.grades))
        if (needs_det or grade == 2) and request.dtype not in (torch.float32, torch.float64):
            return Rejected("matrix_action_requires_float32_or_float64")
    if operation == "linear":
        if route != "graded_linear":
            return Rejected("requires_linear_action")
        _, resources = _linear_action_structure(inputs, output, device=request.device)
        preparation = ActionPreparation(None)
        return _accepted(preparation, resources)
    if operation == "sandwich":
        if route != "full_action_matrix" or not full:
            return Rejected("requires_full_sandwich")
        preparation = ActionPreparation(None)
        return _accepted(preparation, _simple_requirements(request, 6 * spec.dim**2))
    if operation != "versor":
        return Rejected("unsupported_action_operation")
    if grade not in (1, 2) or parameter is None or parameter.grades != (grade,):
        return Rejected("invalid_action_parameter_grade")
    allowed = (
        route == "vector_matrix" or route == "full_action_matrix" and full or route == "rotor_product" and grade == 2
    )
    if not allowed:
        return Rejected("unsupported_action_domain")
    pairs = 2 * spec.n**2 + spec.n if route == "vector_matrix" and grade == 1 else 0
    if route == "full_action_matrix":
        pairs = 6 * spec.dim**2
    lanes = spec.dim if route == "full_action_matrix" else max(inputs.dim, output.dim, parameter.dim)
    resources = ResourceRequirements(lanes, pairs)
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
            norm = planner.router.select(norm_request, planner.policy, planner.limits, warn_selected=False)
            involution = _unary_child(planner, inputs, "grade_involution", request.dtype, request.device)
            reverse = _unary_child(planner, parameter, "reverse", request.dtype, request.device)
    product_facts = ()
    if route == "vector_matrix":
        generator_terms, resources = _linear_action_structure(
            inputs,
            output,
            generator_layout=parameter if grade == 2 else None,
            device=request.device,
        )
        if grade == 1:
            resources = ResourceRequirements(resources.lanes, resources.pairs + 2 * spec.n**2 + spec.n)
    else:
        product_facts = tuple(
            child.facts for child in (left, right) if child is not None and isinstance(child.facts, ProductFacts)
        )
        generator_terms = 0
    facts = ActionFacts(
        ActionWorkProfile(
            route=route,
            grade=grade,
            generator_terms=generator_terms,
            reflection_cells=spec.n**2 if route == "vector_matrix" and grade == 1 else 0,
            vector_matrix_exp_order=spec.n**3 if route == "vector_matrix" and grade == 2 else 0,
            lift=action_lift_profile(spec.n, inputs.grades, output.grades) if route == "vector_matrix" else None,
            product_children=tuple(item.work_profile for item in product_facts),
            exponential_child=None if exponential is None else exponential.facts.work_profile,
            full_action_matrix_order=spec.dim**3 if route == "full_action_matrix" else 0,
            full_action_cells=spec.dim**2 if route == "full_action_matrix" else 0,
        )
    )
    children = tuple(child for child in (exponential, reverse, left, right, norm, involution) if child is not None)
    resources = ResourceRequirements(
        max(resources.lanes, 0 if rotor is None else rotor.dim, 0 if middle is None else middle.dim),
        resources.pairs,
    )
    resources = _combined_requirements(resources, children)
    preparation = ActionPreparation(facts, rotor, middle, exponential, reverse, left, right, norm, involution)
    return _accepted(preparation, resources)


def _operation(child):
    if child is None:
        return None
    from clifra.core.operation import PlannedOperation

    method = "forward_compact" if child.family in {"product", "unary"} else "forward"
    return PlannedOperation(child.build(), child.request.inputs, child.request.output, method=method)


def _build_action(request, route, preparation):
    from .execution.action import (
        ActionComponents,
        BivectorVectorGeneratorExecutor,
        FullSandwichActionExecutor,
        GradedLinearActionExecutor,
        VersorActionExecutor,
        VersorVectorMatrixExecutor,
    )
    from .planning.action import (
        _direct_action_grade,
        _direct_action_plan_tensors,
        _graded_action_plan_tensors,
        _scalar_action_positions,
        build_full_sandwich_action_buffers,
        build_versor_vector_buffers,
    )

    inputs, output = request.inputs[0].layout, request.output.layout

    def graded_action():
        grades = tuple(grade for grade in inputs.grades if grade > 0 and grade in output.grades)
        direct_grades = tuple(grade for grade in grades if _direct_action_grade(inputs, output, grade, request.device))
        grade_buffers = tuple(
            (grade, *_graded_action_plan_tensors(inputs, output, grade=grade))
            for grade in (() if inputs.grades == output.grades == (1,) else grades)
            if grade not in direct_grades
        )
        direct_buffers = tuple(
            (
                grade,
                _direct_action_plan_tensors(inputs.spec, grade, dtype=request.dtype, device=request.device),
            )
            for grade in direct_grades
        )
        return GradedLinearActionExecutor(
            input_layout=inputs,
            output_layout=output,
            scalar_flat_positions=_scalar_action_positions(inputs, output).to(request.device),
            grade_buffers=tuple(
                (grade, *(buffer.to(request.device) for buffer in buffers)) for grade, *buffers in grade_buffers
            ),
            direct_buffers=direct_buffers,
        )

    def full_action():
        buffers = build_full_sandwich_action_buffers(inputs, device=request.device, dtype=request.dtype)
        return FullSandwichActionExecutor(
            layout=inputs,
            cayley_indices=buffers[0],
            left_sign_t=buffers[1],
            geometric_product_sign_t=buffers[2],
        )

    if request.operation == "linear":
        return graded_action()
    if request.operation == "sandwich":
        return full_action()
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
    action = vector_matrix = selected_full_action = None
    if route == "vector_matrix":
        action = graded_action()
        generator_buffers, metric_signs, eye = build_versor_vector_buffers(
            request.inputs[1].layout,
            grade=request.grade,
            dtype=request.dtype,
            device=request.device,
        )
        generator = None
        if generator_buffers is not None:
            generator = BivectorVectorGeneratorExecutor(
                bivector_layout=request.inputs[1].layout,
                lane_positions=generator_buffers[0],
                flat_positions=generator_buffers[1],
                coefficients=generator_buffers[2],
            )
        vector_matrix = VersorVectorMatrixExecutor(
            grade=request.grade,
            parameter_layout=request.inputs[1].layout,
            eps=request.algebra.eps_sq,
            generator=generator,
            metric_signs=metric_signs,
            eye=eye,
        )
    elif route == "full_action_matrix":
        selected_full_action = full_action()

    rotor_indices = (
        preparation.rotor_layout.indices_tensor(device=request.device)
        if preparation.rotor_layout is not None
        else torch.empty(0, dtype=torch.long, device=request.device)
    )
    return VersorActionExecutor(
        grade=request.grade,
        input_layout=inputs,
        output_layout=output,
        parameter_layout=request.inputs[1].layout,
        route=route,
        components=components,
        action=action,
        vector_matrix=vector_matrix,
        full_action=selected_full_action,
        rotor_full_indices=rotor_indices,
        parameter_full_indices=request.inputs[1].layout.indices_tensor(device=request.device),
        eps_sq=request.algebra.eps_sq,
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
                ("closed", "taylor", "left_matrix_exp"),
            ),
            (
                "action",
                ("vector_matrix", "rotor_product", "full_action_matrix", "graded_linear"),
            ),
        )
        for route in routes
    )
