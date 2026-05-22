"""Debug formatting for multivector coefficient tensors.

This module intentionally avoids algebra operations. ``Multivector`` is a
display proxy for development and logging, not a runtime tensor wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch

from clifra.core.foundation.layout import AlgebraSpec, GradeLayout


@dataclass(frozen=True)
class Multivector:
    """Lightweight display proxy for multivector coefficient tensors.

    The proxy stores only formatting options and never overloads arithmetic.
    Compact tensors must be declared with ``layout`` or ``grades`` so the
    formatter can map lanes to blade labels without guessing.
    """

    algebra: object
    values: torch.Tensor
    layout: Optional[GradeLayout] = None
    grades: Optional[Iterable[int]] = None
    name: Optional[str] = None
    max_terms: int = 16
    precision: int = 4
    atol: Optional[float] = None

    def format(
        self,
        *,
        sample: Optional[Sequence[int]] = None,
        include_shape: bool = True,
    ) -> str:
        """Return a blade-term string for the tensor or one tensor sample."""
        return format_multivector(
            self.algebra,
            self.values,
            layout=self.layout,
            grades=self.grades,
            name=self.name,
            max_terms=self.max_terms,
            precision=self.precision,
            atol=self.atol,
            sample=sample,
            include_shape=include_shape,
        )

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        return f"Multivector({self.format()})"


def format_multivector(
    algebra,
    values: torch.Tensor,
    *,
    layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
    name: Optional[str] = None,
    max_terms: int = 16,
    precision: int = 4,
    atol: Optional[float] = None,
    sample: Optional[Sequence[int]] = None,
    include_shape: bool = True,
) -> str:
    """Format a coefficient tensor as a sum of basis-blade terms.

    Full tensors are recognized by ``values.shape[-1] == algebra.dim``.
    Compact tensors require ``layout`` or ``grades``. Batched tensors format a
    single sample by default to keep debug output bounded.
    """
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values)
    if values.ndim < 1:
        raise ValueError(f"multivector values must include a lane dimension, got shape {tuple(values.shape)}")

    basis_indices = _basis_indices_for_values(algebra, values, layout=layout, grades=grades)
    if values.shape[-1] != len(basis_indices):
        raise ValueError(f"values last dimension must be {len(basis_indices)}, got {values.shape[-1]}")

    vector, sample_label = _sample_vector(values, sample=sample)
    body = _format_vector(
        vector,
        basis_indices=basis_indices,
        n=int(algebra.n),
        max_terms=max_terms,
        precision=precision,
        atol=_default_atol(vector, atol),
    )

    prefix = "" if name is None else f"{name} = "
    if include_shape and values.ndim > 1:
        return f"{prefix}shape={tuple(values.shape)}, sample{sample_label} = {body}"
    return f"{prefix}{body}"


def basis_blade_label(index: int, *, n: int, scalar: str = "1", prefix: str = "e") -> str:
    """Return the canonical display label for a basis blade bitmask."""
    index = int(index)
    if index == 0:
        return scalar
    bits = [bit + 1 for bit in range(int(n)) if index & (1 << bit)]
    if not bits:
        return scalar
    if int(n) <= 9:
        return prefix + "".join(str(bit) for bit in bits)
    return f"{prefix}[{','.join(str(bit) for bit in bits)}]"


def _basis_indices_for_values(
    algebra,
    values: torch.Tensor,
    *,
    layout: Optional[GradeLayout],
    grades: Optional[Iterable[int]],
) -> tuple[int, ...]:
    spec = AlgebraSpec.from_algebra(algebra)
    if layout is not None:
        if layout.spec != spec:
            raise ValueError(f"layout signature {layout.spec} does not match algebra signature {spec}")
        return layout.basis_indices
    if grades is not None:
        return algebra.layout(grades).basis_indices
    if values.shape[-1] == spec.dim:
        return tuple(range(spec.dim))
    raise ValueError("compact multivector formatting requires layout or grades")


def _sample_vector(values: torch.Tensor, *, sample: Optional[Sequence[int]]) -> tuple[torch.Tensor, str]:
    if values.ndim == 1:
        if sample is not None and tuple(sample):
            raise ValueError("sample indices are only valid for batched multivector tensors")
        return values, ""

    prefix_ndim = values.ndim - 1
    if sample is None:
        index = (0,) * prefix_ndim
    else:
        index = tuple(int(part) for part in sample)
        if len(index) != prefix_ndim:
            raise ValueError(f"sample must have {prefix_ndim} indices for shape {tuple(values.shape)}")
    return values[index], f"[{','.join(str(part) for part in index)}]"


def _format_vector(
    vector: torch.Tensor,
    *,
    basis_indices: tuple[int, ...],
    n: int,
    max_terms: int,
    precision: int,
    atol: float,
) -> str:
    flat = vector.detach()
    if flat.device.type != "cpu":
        flat = flat.cpu()
    nonzero_terms: list[tuple[float, str]] = []
    for position, coefficient in enumerate(flat):
        value = float(coefficient.item())
        if abs(value) <= atol:
            continue
        nonzero_terms.append((value, basis_blade_label(basis_indices[position], n=n)))

    if not nonzero_terms:
        return "0"

    shown = nonzero_terms[: max(1, int(max_terms))]
    chunks = [
        _format_term(value, label, first=position == 0, precision=precision, atol=atol)
        for position, (value, label) in enumerate(shown)
    ]
    omitted = len(nonzero_terms) - len(shown)
    if omitted > 0:
        chunks.append(f" + ... ({omitted} more)")
    return "".join(chunks)


def _format_term(value: float, label: str, *, first: bool, precision: int, atol: float) -> str:
    sign = "-" if value < 0 else "+"
    magnitude = abs(value)
    if first:
        prefix = "-" if value < 0 else ""
    else:
        prefix = f" {sign} "
    if label != "1" and abs(magnitude - 1.0) <= atol:
        body = label
    elif label == "1":
        body = _format_number(magnitude, precision)
    else:
        body = f"{_format_number(magnitude, precision)}{label}"
    return f"{prefix}{body}"


def _format_number(value: float, precision: int) -> str:
    return f"{value:.{max(1, int(precision))}g}"


def _default_atol(values: torch.Tensor, atol: Optional[float]) -> float:
    if atol is not None:
        return float(atol)
    if values.dtype.is_floating_point:
        return float(torch.finfo(values.dtype).eps * 64)
    return 0.0
