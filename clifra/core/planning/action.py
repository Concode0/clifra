# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Intent and layout plans for linear and versor-style actions."""

from __future__ import annotations

from dataclasses import dataclass

from clifra.core.foundation.basis import expand_output_grades
from clifra.core.foundation.layout import GradeLayout


@dataclass(frozen=True)
class LinearActionPlan:
    """Resolved contract for a vector-space action lifted to multivector grades."""

    input_layout: GradeLayout
    output_layout: GradeLayout

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
    output_layout = input_layout if output_layout is None else output_layout
    if input_layout.spec != output_layout.spec:
        raise ValueError(f"layout mismatch: {input_layout.spec} vs {output_layout.spec}")
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
    grade = int(grade)
    if grade not in {1, 2}:
        raise ValueError("planned versor actions currently support grade=1 and grade=2")
    output_layout = input_layout if output_layout is None else output_layout
    parameter_layout = algebra.layout((grade,)) if parameter_layout is None else parameter_layout
    if input_layout.spec != output_layout.spec or input_layout.spec != parameter_layout.spec:
        raise ValueError("input, output, and parameter layouts must share one algebra spec")
    if parameter_layout.grades != (grade,):
        raise ValueError(f"parameter_layout must contain grade {grade}, got {parameter_layout.grades}")
    return VersorActionPlan(
        grade=grade,
        input_layout=input_layout,
        output_layout=output_layout,
        parameter_layout=parameter_layout,
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
    spec = input_layout.spec
    parameter_layout = algebra.layout((2,)) if parameter_layout is None else parameter_layout
    if input_layout.spec != parameter_layout.spec:
        raise ValueError("input and parameter layouts must share one algebra spec")
    if parameter_layout.grades != (2,):
        raise ValueError(f"parameter_layout must contain grade 2, got {parameter_layout.grades}")

    rotor_layout = spec.layout(range(0, spec.n + 1, 2))
    middle_grades = expand_output_grades(rotor_layout.grades, input_layout.grades, spec.n, op="gp")
    middle_layout = spec.layout(middle_grades)
    inferred_output = spec.layout(expand_output_grades(middle_layout.grades, rotor_layout.grades, spec.n, op="gp"))
    output_layout = inferred_output if output_layout is None else output_layout
    if output_layout.spec != spec:
        raise ValueError(f"output layout signature {output_layout.spec} does not match input signature {spec}")
    return PairedBivectorActionPlan(
        input_layout=input_layout,
        output_layout=output_layout,
        parameter_layout=parameter_layout,
        rotor_layout=rotor_layout,
        middle_layout=middle_layout,
    )
