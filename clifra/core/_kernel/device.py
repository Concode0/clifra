# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Low-level device and dtype normalization helpers."""

from __future__ import annotations

from typing import Any, Optional

import torch

FLOAT_DTYPES: dict[str, torch.dtype] = {
    "float64": torch.float64,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

_DTYPE_ALIASES: dict[str, torch.dtype] = {
    **FLOAT_DTYPES,
    "fp64": torch.float64,
    "double": torch.float64,
    "fp32": torch.float32,
    "float": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "half": torch.float16,
}


def optional_dtype(value: Any) -> Optional[torch.dtype]:
    """Parse a torch dtype declaration, preserving ``None`` as unset."""
    if value is None or isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized in _DTYPE_ALIASES:
            return _DTYPE_ALIASES[normalized]
    raise ValueError(f"Unsupported torch dtype declaration: {value!r}")


def resolve_dtype(value: Any, default: torch.dtype = torch.float32) -> torch.dtype:
    """Parse a torch dtype declaration and fall back to ``default`` when unset."""
    return optional_dtype(value) or default


def dtype_name(dtype: torch.dtype) -> str:
    """Return the canonical short name for a torch dtype."""
    for name, candidate in FLOAT_DTYPES.items():
        if candidate == dtype:
            return name
    return str(dtype).replace("torch.", "")


def resolve_device(device: str = "auto") -> str:
    """Resolve ``'auto'`` to the best available accelerator.

    Priority: cuda > mps > cpu.
    """
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
