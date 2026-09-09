"""Private validation of explicit semantic layouts and tensor contracts."""

from __future__ import annotations

from typing import Optional

from clifra.core._kernel.basis import normalize_grades
from clifra.core.layout import AlgebraSpec, GradeLayout
from clifra.core.tensors import LaneStorage, TensorContract


def _check_contract_spec(spec: AlgebraSpec, contract: TensorContract, name: str) -> TensorContract:
    """Validate that ``contract`` belongs to the receiving ``spec``."""
    if contract.spec != spec:
        raise ValueError(f"{name} signature {contract.spec} does not match algebra signature {spec}")
    return contract


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
        resolved = spec.full_layout()
    contract = TensorContract(layout=resolved, storage=storage)
    return _check_contract_spec(spec, contract, name)
