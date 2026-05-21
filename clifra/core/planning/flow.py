# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Grade-flow metadata for AoT layout propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from clifra.core.foundation.basis import GradeProductOp, expand_output_grades, normalize_grades
from clifra.core.foundation.layout import AlgebraSpec, GradeLayout


@dataclass(frozen=True)
class GradeFlow:
    """Static grade/layout metadata passed between planned operations."""

    spec: AlgebraSpec
    layout: GradeLayout

    @classmethod
    def from_grades(cls, spec: AlgebraSpec, grades: Iterable[int]) -> "GradeFlow":
        """Create a flow from active grades."""
        return cls(spec=spec, layout=spec.layout(grades))

    @classmethod
    def full(cls, spec: AlgebraSpec) -> "GradeFlow":
        """Create a full-layout flow."""
        return cls.from_grades(spec, range(spec.n + 1))

    @classmethod
    def scalar(cls, spec: AlgebraSpec) -> "GradeFlow":
        """Create a scalar-only flow."""
        return cls.from_grades(spec, (0,))

    @classmethod
    def vector(cls, spec: AlgebraSpec) -> "GradeFlow":
        """Create a vector-only flow."""
        return cls.from_grades(spec, (1,))

    @property
    def grades(self) -> tuple[int, ...]:
        """Active grades represented by this flow."""
        return self.layout.grades

    @property
    def dim(self) -> int:
        """Compact lane count represented by this flow."""
        return self.layout.dim

    def project(self, grades: Iterable[int]) -> "GradeFlow":
        """Narrow the flow to a subset of active grades."""
        projected = normalize_grades(grades, self.spec.n, name="grades")
        missing = tuple(grade for grade in projected if grade not in self.grades)
        if missing:
            raise ValueError(f"Cannot project missing grades {missing} from active grades {self.grades}")
        return GradeFlow.from_grades(self.spec, projected)

    def unary(self, op: str, output_grades: Optional[Iterable[int]] = None) -> "GradeFlow":
        """Propagate flow through a unary operation."""
        if output_grades is not None:
            return self.project(output_grades)
        if op in {"identity", "reverse", "grade_involution", "clifford_conjugation"}:
            return self
        if op == "grade_projection":
            raise ValueError("grade_projection requires output_grades")
        raise ValueError(f"Unsupported unary flow op {op!r}")

    def product(
        self,
        other: "GradeFlow",
        *,
        op: GradeProductOp = "gp",
        output_grades: Optional[Iterable[int]] = None,
    ) -> "GradeFlow":
        """Propagate flow through a bilinear grade product."""
        self._check_spec(other)
        grades = (
            expand_output_grades(self.grades, other.grades, self.spec.n, op=op)
            if output_grades is None
            else normalize_grades(output_grades, self.spec.n, name="output_grades")
        )
        return GradeFlow.from_grades(self.spec, grades)

    def merge(self, other: "GradeFlow") -> "GradeFlow":
        """Union two flows, e.g. for addition or concatenation."""
        self._check_spec(other)
        return GradeFlow.from_grades(self.spec, (*self.grades, *other.grades))

    def _check_spec(self, other: "GradeFlow") -> None:
        if self.spec != other.spec:
            raise ValueError(f"GradeFlow spec mismatch: {self.spec} vs {other.spec}")
