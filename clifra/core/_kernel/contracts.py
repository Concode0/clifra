"""Private request normalization and numerical lane adaptation."""

from __future__ import annotations

from typing import Iterable, Optional

import torch

from clifra.core._kernel.basis import normalize_grades, operation_coefficient
from clifra.core.layout import AlgebraSpec, GradeLayout
from clifra.core.tensors import LaneStorage, TensorContract


def normalize_lane_storage(storage: LaneStorage | str) -> LaneStorage:
    """Return a validated lane storage enum."""
    if isinstance(storage, LaneStorage):
        return storage
    try:
        return LaneStorage(str(storage))
    except ValueError as exc:
        raise ValueError("storage must be 'compact' or 'canonical'") from exc


def check_layout_spec(spec: AlgebraSpec, layout: GradeLayout, name: str) -> None:
    """Validate that ``layout`` belongs to ``spec``."""
    if layout.spec != spec:
        raise ValueError(f"{name} signature {layout.spec} does not match algebra signature {spec}")


def _check_contract_spec(spec: AlgebraSpec, contract: TensorContract, name: str) -> TensorContract:
    """Validate that ``contract`` belongs to the receiving ``spec``."""
    if contract.spec != spec:
        raise ValueError(f"{name} signature {contract.spec} does not match algebra signature {spec}")
    return contract


def resolve_layout(algebra_or_spec, *, layout: Optional[GradeLayout] = None, grades=None) -> GradeLayout:
    """Resolve explicit layout metadata through a compact tensor contract."""
    return resolve_contract(algebra_or_spec, layout=layout, grades=grades).layout


def resolve_contract(
    algebra_or_spec,
    *,
    layout: Optional[GradeLayout] = None,
    grades=None,
    storage: LaneStorage | str = LaneStorage.COMPACT,
    name: str = "layout",
) -> TensorContract:
    """Resolve a tensor contract from explicit semantic layout and storage."""
    spec = algebra_or_spec if isinstance(algebra_or_spec, AlgebraSpec) else AlgebraSpec.from_algebra(algebra_or_spec)
    if layout is not None:
        contract = TensorContract(layout=layout, storage=storage)
        _check_contract_spec(spec, contract, name)
        if grades is not None and layout.grades != normalize_grades(grades, spec.n, name="grades"):
            grades_name = f"{name[:-7]}_grades" if name.endswith("_layout") else "grades"
            raise ValueError(f"{name} and {grades_name} disagree")
        return contract
    if grades is not None:
        resolved = spec.layout(grades)
    else:
        default_grades = getattr(algebra_or_spec, "_default_grades", None)
        if default_grades is not None:
            resolved = spec.layout(default_grades)
        elif hasattr(algebra_or_spec, "default_layout"):
            resolved = algebra_or_spec.layout()
        else:
            resolved = spec.full_layout()
    contract = TensorContract(layout=resolved, storage=storage)
    return _check_contract_spec(spec, contract, name)


def infer_contract(
    spec: AlgebraSpec,
    tensor: torch.Tensor,
    *,
    layout: Optional[GradeLayout] = None,
    grades=None,
    storage: LaneStorage | str | None = None,
    side: str = "value",
) -> TensorContract:
    """Infer storage at a public boundary, then return an explicit contract."""
    if tensor.ndim < 1:
        raise ValueError(f"{side} must include a coefficient lane dimension, got shape {tuple(tensor.shape)}")
    resolved_contract = resolve_contract(spec, layout=layout, grades=grades, name=f"{side}_layout")
    resolved = resolved_contract.layout
    if storage is None:
        if tensor.shape[-1] == resolved.dim:
            storage = LaneStorage.COMPACT
        elif tensor.shape[-1] == spec.dim:
            storage = LaneStorage.CANONICAL
        else:
            raise ValueError(
                f"{side} last dimension must be {resolved.dim} for compact grades {resolved.grades} "
                f"or {spec.dim} canonical lanes, got {tensor.shape[-1]}"
            )
    contract = TensorContract(layout=resolved, storage=normalize_lane_storage(storage))
    contract.validate(tensor, name=side)
    return contract


def compact_values(
    algebra, value: torch.Tensor, *, layout: Optional[GradeLayout] = None, grades=None
) -> tuple[torch.Tensor, GradeLayout]:
    """Return compact values and resolved semantic layout."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected Tensor value, got {type(value)!r}")
    contract = infer_contract(
        AlgebraSpec.from_algebra(algebra),
        value,
        layout=layout,
        grades=grades,
        storage=None,
    )
    return contract.to_compact(value), contract.layout


def canonical_values(
    algebra, value: torch.Tensor, *, layout: Optional[GradeLayout] = None, grades=None
) -> torch.Tensor:
    """Return canonical full-basis values."""
    values, resolved = compact_values(algebra, value, layout=layout, grades=grades)
    if resolved.dim == resolved.spec.dim and resolved.grades == tuple(range(resolved.spec.n + 1)):
        return values
    return resolved.full(values)


def union_layout(algebra_or_spec, left: GradeLayout, right: GradeLayout) -> GradeLayout:
    """Return a compact layout containing every grade from ``left`` and ``right``."""
    spec = algebra_or_spec if isinstance(algebra_or_spec, AlgebraSpec) else AlgebraSpec.from_algebra(algebra_or_spec)
    left = resolve_contract(spec, layout=left, name="left layout").layout
    right = resolve_contract(spec, layout=right, name="right layout").layout
    if left == right:
        return left
    grades = _grade_union(left.grades, right.grades)
    return algebra_or_spec.layout(grades) if hasattr(algebra_or_spec, "layout") else spec.layout(grades)


def compact_pair_values(
    algebra_or_spec,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    layout: Optional[GradeLayout] = None,
    left_layout: Optional[GradeLayout] = None,
    right_layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
    left_grades: Optional[Iterable[int]] = None,
    right_grades: Optional[Iterable[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, GradeLayout]:
    """Return compact values aligned to one runtime layout."""
    shared_left_layout = left_layout if left_layout is not None else layout
    shared_right_layout = right_layout if right_layout is not None else layout
    shared_left_grades = left_grades if left_grades is not None else grades
    shared_right_grades = right_grades if right_grades is not None else grades
    left_values, resolved_left = compact_values(
        algebra_or_spec,
        left,
        layout=shared_left_layout,
        grades=shared_left_grades,
    )
    right_values, resolved_right = compact_values(
        algebra_or_spec,
        right,
        layout=shared_right_layout,
        grades=shared_right_grades,
    )
    resolved = union_layout(algebra_or_spec, resolved_left, resolved_right)
    if resolved_left != resolved:
        left_values = resolved.convert(left_values, resolved_left)
    if resolved_right != resolved:
        right_values = resolved.convert(right_values, resolved_right)
    return left_values, right_values, resolved


def metric_self_signs(layout: GradeLayout, *, device=None, dtype=None) -> torch.Tensor:
    """Return basis self-product signs for a layout."""
    signs = [
        operation_coefficient(index, index, layout.spec.p, layout.spec.q, layout.spec.r, "geometric_product")
        for index in layout.basis_indices
    ]
    return torch.tensor(signs, device=device, dtype=torch.float32 if dtype is None else dtype)


def _grade_union(left: Iterable[int], right: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(set(left).union(right)))
