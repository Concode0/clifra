# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Intent and layout plans for linear and versor-style actions."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch

from clifra.core._kernel.basis import expand_output_grades
from clifra.core._kernel.contracts import _check_contract_spec
from clifra.core._kernel.planning.policy import (
    PlanCandidate,
    PlanFacts,
    compose_plan_facts,
    environment_extensions,
    select_policy_route,
)
from clifra.core._kernel.planning.product import select_product_route
from clifra.core.layout import AlgebraSpec, GradeLayout
from clifra.core.tensors import TensorContract


def _contract(spec, layout, role: str) -> TensorContract:
    return _check_contract_spec(spec, TensorContract.compact(layout), f"{role}_layout")


def _bind_contracts(plan, *roles: str) -> None:
    spec = plan.input_layout.spec
    for role in roles:
        object.__setattr__(plan, f"{role}_contract", _contract(spec, getattr(plan, f"{role}_layout"), role))


@dataclass(frozen=True)
class LinearActionPlan:
    """Resolved contract for a vector-space action lifted to multivector grades."""

    input_layout: GradeLayout
    output_layout: GradeLayout
    input_contract: TensorContract = field(init=False, repr=False)
    output_contract: TensorContract = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _bind_contracts(self, "input", "output")

    @property
    def input_grades(self) -> tuple[int, ...]:
        """Return the grades accepted by the action input layout."""
        return self.input_layout.grades

    @property
    def output_grades(self) -> tuple[int, ...]:
        """Return the grades emitted by the action output layout."""
        return self.output_layout.grades


@dataclass(frozen=True)
class VersorActionPlan:
    """Resolved contract for grade-1 or grade-2 versor actions."""

    grade: int
    input_layout: GradeLayout
    output_layout: GradeLayout
    parameter_layout: GradeLayout
    execution_path: str
    input_contract: TensorContract = field(init=False, repr=False)
    output_contract: TensorContract = field(init=False, repr=False)
    parameter_contract: TensorContract = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _bind_contracts(self, "input", "output", "parameter")

    @property
    def linear_action(self) -> LinearActionPlan:
        """Return the equivalent linear action over the same input/output layouts."""
        return LinearActionPlan(input_layout=self.input_layout, output_layout=self.output_layout)


@dataclass(frozen=True)
class PairedBivectorActionPlan:
    """Resolved contract for independent left/right bivector rotor actions."""

    input_layout: GradeLayout
    output_layout: GradeLayout
    parameter_layout: GradeLayout
    rotor_layout: GradeLayout
    middle_layout: GradeLayout
    execution_path: str
    input_contract: TensorContract = field(init=False, repr=False)
    output_contract: TensorContract = field(init=False, repr=False)
    parameter_contract: TensorContract = field(init=False, repr=False)
    rotor_contract: TensorContract = field(init=False, repr=False)
    middle_contract: TensorContract = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _bind_contracts(self, "input", "output", "parameter", "rotor", "middle")

    @property
    def input_grades(self) -> tuple[int, ...]:
        """Return the grades accepted before the paired bivector action."""
        return self.input_layout.grades

    @property
    def output_grades(self) -> tuple[int, ...]:
        """Return the grades retained after the paired bivector action."""
        return self.output_layout.grades


def build_linear_action_plan(
    *,
    input_layout: GradeLayout,
    output_layout: GradeLayout | None = None,
) -> LinearActionPlan:
    """Build a plan-only linear action contract."""
    spec = input_layout.spec
    output_layout = input_layout if output_layout is None else output_layout
    _contract(spec, output_layout, "output")
    return LinearActionPlan(input_layout=input_layout, output_layout=output_layout)


def build_versor_action_plan(
    algebra,
    *,
    grade: int,
    input_layout: GradeLayout,
    output_layout: GradeLayout | None = None,
    parameter_layout: GradeLayout | None = None,
) -> VersorActionPlan:
    """Build a plan-only versor action contract."""
    spec = AlgebraSpec.from_algebra(algebra)
    grade = int(grade)
    if grade not in {1, 2}:
        raise ValueError("planned versor actions currently support grade=1 and grade=2")
    output_layout = input_layout if output_layout is None else output_layout
    parameter_layout = algebra.layout((grade,)) if parameter_layout is None else parameter_layout
    input_layout = _contract(spec, input_layout, "input").layout
    output_layout = _contract(spec, output_layout, "output").layout
    parameter_layout = _contract(spec, parameter_layout, "parameter").layout
    if parameter_layout.grades != (grade,):
        raise ValueError(f"parameter_layout must contain grade {grade}, got {parameter_layout.grades}")
    decision = _select_versor_action_route(
        algebra,
        grade=grade,
        input_layout=input_layout,
        output_layout=output_layout,
        parameter_layout=parameter_layout,
    )
    return VersorActionPlan(
        grade=grade,
        input_layout=input_layout,
        output_layout=output_layout,
        parameter_layout=parameter_layout,
        execution_path=decision.route,
    )


def build_paired_bivector_action_plan(
    algebra,
    *,
    input_layout: GradeLayout,
    output_layout: GradeLayout | None = None,
    parameter_layout: GradeLayout | None = None,
) -> PairedBivectorActionPlan:
    """Build a plan for ``R_left x R_right_reverse`` with independent rotors.

    Unlike a true versor sandwich ``R x R~``, independent left/right rotors are
    not generally grade-preserving. The planner therefore expands the default
    output layout through both geometric products and lets callers explicitly
    project with ``output_layout`` when they want a narrower result.
    """
    spec = AlgebraSpec.from_algebra(algebra)
    parameter_layout = algebra.layout((2,)) if parameter_layout is None else parameter_layout
    input_layout = _contract(spec, input_layout, "input").layout
    parameter_layout = _contract(spec, parameter_layout, "parameter").layout
    if parameter_layout.grades != (2,):
        raise ValueError(f"parameter_layout must contain grade 2, got {parameter_layout.grades}")

    rotor_layout = spec.layout(range(0, spec.n + 1, 2))
    middle_grades = expand_output_grades(rotor_layout.grades, input_layout.grades, spec.n, op="geometric_product")
    middle_layout = spec.layout(middle_grades)
    inferred_output = spec.layout(
        expand_output_grades(middle_layout.grades, rotor_layout.grades, spec.n, op="geometric_product")
    )
    output_layout = inferred_output if output_layout is None else output_layout
    output_layout = _contract(spec, output_layout, "output").layout
    decision = _select_paired_action_route(
        algebra,
        input_layout=input_layout,
        output_layout=output_layout,
        parameter_layout=parameter_layout,
        rotor_layout=rotor_layout,
        middle_layout=middle_layout,
    )
    return PairedBivectorActionPlan(
        input_layout=input_layout,
        output_layout=output_layout,
        parameter_layout=parameter_layout,
        rotor_layout=rotor_layout,
        middle_layout=middle_layout,
        execution_path=decision.route,
    )


def _action_extensions(algebra, *, input_layout, output_layout, parameter_layout, intermediate_lanes: int = 0):
    device_type = getattr(getattr(algebra, "device", None), "type", str(getattr(algebra, "device", "cpu")))
    dtype = getattr(algebra, "dtype", None)
    dtype_bytes = 4 if dtype is None else torch.finfo(dtype).bits // 8
    return {
        **environment_extensions(algebra, device_type, dtype_bytes),
        "layout.input_lanes": input_layout.dim,
        "layout.output_lanes": output_layout.dim,
        "action.parameter_lanes": parameter_layout.dim,
        "action.intermediate_lanes": intermediate_lanes,
        "action.full_lanes": algebra.dim,
    }


def _select_versor_action_route(algebra, *, grade, input_layout, output_layout, parameter_layout):
    full = input_layout.dim == algebra.dim and output_layout.dim == algebra.dim
    vector_to_vector = input_layout.grades == (1,) and output_layout.grades == (1,)
    dtype_bytes = torch.finfo(algebra.dtype).bits // 8
    matrix_work = float(algebra.n**3 + input_layout.dim * output_layout.dim)
    rotor_facts = PlanFacts()
    exp_facts = PlanFacts()
    rotor_intermediate_lanes = max(input_layout.dim, output_layout.dim)
    if grade == 2:
        spec = AlgebraSpec.from_algebra(algebra)
        rotor_layout = spec.layout(range(0, algebra.n + 1, 2))
        middle_layout = spec.layout(
            expand_output_grades(rotor_layout.grades, input_layout.grades, algebra.n, op="geometric_product")
        )
        exp_facts = _bivector_exp_facts(algebra, rotor_layout)
        left_facts = _product_facts(algebra, rotor_layout, input_layout, middle_layout)
        right_facts = _product_facts(algebra, middle_layout, rotor_layout, output_layout)
        rotor_facts = compose_plan_facts(
            exp_facts,
            left_facts,
            right_facts,
            peak_bytes=(rotor_layout.dim + middle_layout.dim + output_layout.dim) * dtype_bytes,
        )
        rotor_intermediate_lanes = middle_layout.dim
    full_action_facts = _full_action_facts(algebra, exp_facts, 1, dtype_bytes)
    candidates = (
        _action_candidate(
            algebra,
            "vector_matrix",
            PlanFacts(matrix_work, matrix_work * 2, algebra.n**2 * dtype_bytes, algebra.n**2),
            input_layout,
            output_layout,
            parameter_layout,
            algebra.n * algebra.n,
            None if grade == 1 or vector_to_vector else "vector_matrix_domain",
        ),
        _action_candidate(
            algebra,
            "rotor_product",
            rotor_facts,
            input_layout,
            output_layout,
            parameter_layout,
            rotor_intermediate_lanes,
            None if grade == 2 else "rotor_product_requires_bivector_parameter",
        ),
        _action_candidate(
            algebra,
            "full_action_matrix",
            full_action_facts,
            input_layout,
            output_layout,
            parameter_layout,
            algebra.dim * algebra.dim,
            None if full else "requires_full_input_and_output",
        ),
    )
    return select_policy_route(algebra._planning_policy, candidates)


def _select_paired_action_route(
    algebra,
    *,
    input_layout,
    output_layout,
    parameter_layout,
    rotor_layout,
    middle_layout,
):
    full = input_layout.dim == algebra.dim and output_layout.dim == algebra.dim
    dtype_bytes = torch.finfo(algebra.dtype).bits // 8
    exp_facts = _bivector_exp_facts(algebra, rotor_layout)
    left_facts = _product_facts(algebra, rotor_layout, input_layout, middle_layout)
    right_facts = _product_facts(algebra, middle_layout, rotor_layout, output_layout)
    paired_facts = compose_plan_facts(
        exp_facts,
        exp_facts,
        left_facts,
        right_facts,
        peak_bytes=(2 * rotor_layout.dim + middle_layout.dim + output_layout.dim) * dtype_bytes,
    )
    full_action_facts = _full_action_facts(algebra, exp_facts, 2, dtype_bytes)
    candidates = (
        _action_candidate(
            algebra,
            "full_action_matrix",
            full_action_facts,
            input_layout,
            output_layout,
            parameter_layout,
            algebra.dim * algebra.dim,
            None if full else "requires_full_input_and_output",
        ),
        _action_candidate(
            algebra,
            "paired_rotor_product",
            paired_facts,
            input_layout,
            output_layout,
            parameter_layout,
            middle_layout.dim,
            None,
        ),
    )
    return select_policy_route(algebra._planning_policy, candidates)


def _full_action_facts(algebra, exp_facts: PlanFacts, exp_count: int, dtype_bytes: int) -> PlanFacts:
    matrix_elements = algebra.dim**2
    matrix = PlanFacts(
        float(matrix_elements), float(2 * matrix_elements), matrix_elements * dtype_bytes, matrix_elements
    )
    return compose_plan_facts(
        *((exp_facts,) * exp_count),
        matrix,
        peak_bytes=(matrix_elements + 2 * exp_count * algebra.dim) * dtype_bytes,
    )


def _product_facts(algebra, left_layout, right_layout, output_layout) -> PlanFacts:
    decision = select_product_route(
        algebra,
        op="geometric_product",
        left_layout=left_layout,
        right_layout=right_layout,
        output_layout=output_layout,
        dtype=algebra.dtype,
        device=algebra.device,
    )
    return decision.facts


def _bivector_exp_facts(algebra, output_layout) -> PlanFacts:
    from clifra.core._kernel.planning.exp import select_bivector_exp_route, spectral_exp_preselection

    options = algebra._bivector_exp_options
    preselection = spectral_exp_preselection(
        AlgebraSpec.from_algebra(algebra),
        algebra.device,
        dtype=algebra.dtype,
        max_planes=options.spectral_max_planes,
        tol_abs=options.spectral_tol_abs,
        tol_rel=options.spectral_tol_rel,
        dominant_rel=options.spectral_dominant_rel,
        allow_degenerate=options.spectral_allow_degenerate,
        allow_truncated_degenerate=options.spectral_allow_truncated_degenerate,
    )
    decision = select_bivector_exp_route(
        AlgebraSpec.from_algebra(algebra),
        algebra.device,
        dtype=algebra.dtype,
        output_layout=output_layout,
        preselection=preselection,
        policy=algebra._planning_policy,
    )
    return decision.facts


def _action_candidate(
    algebra,
    route: str,
    facts: PlanFacts,
    input_layout,
    output_layout,
    parameter_layout,
    intermediate_lanes: int,
    unavailable_reason: str | None,
) -> PlanCandidate:
    extensions = {
        **dict(facts.extensions),
        **_action_extensions(
            algebra,
            input_layout=input_layout,
            output_layout=output_layout,
            parameter_layout=parameter_layout,
            intermediate_lanes=intermediate_lanes,
        ),
    }
    return PlanCandidate(
        "action",
        route,
        replace(facts, extensions=extensions),
        unavailable_reason,
    )
