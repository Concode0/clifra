# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Intent and layout plans for linear and versor-style actions."""

from __future__ import annotations

from dataclasses import dataclass, field

from clifra.core._kernel.basis import expand_output_grades
from clifra.core._kernel.contracts import _check_contract_spec
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
    route: str
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
    route: str
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
        route=decision.route,
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
        route=decision.route,
    )


def _select_versor_action_route(algebra, **parameters):
    return _select_action_route(algebra, "versor", parameters)


def _select_paired_action_route(algebra, **parameters):
    return _select_action_route(algebra, "paired", parameters)


def _select_action_route(algebra, operation, parameters):
    from clifra.core._kernel.execution.providers import action_execution_request

    request = action_execution_request(algebra, operation, **parameters)
    return algebra._planner.router.select(request, algebra._planner.policy, algebra._planner.limits)
