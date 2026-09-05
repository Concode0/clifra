# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0
"""Stable algebra construction, independent of kernel tuning."""

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from ._kernel.device import resolve_dtype
from .algebra import AlgebraContext


@dataclass(frozen=True)
class AlgebraConfig:
    p: int
    q: int = 0
    r: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any], **overrides):
        values = dict(config)
        values.update({key: value for key, value in overrides.items() if value is not None})
        unknown = values.keys() - cls.__dataclass_fields__.keys()
        if unknown:
            raise TypeError(f"unknown algebra configuration fields {sorted(unknown)!r}")
        if "dtype" in values:
            values["dtype"] = resolve_dtype(values["dtype"])
        return cls(**values)


def make_algebra(p: int, q: int = 0, r: int = 0, *, device="cpu", dtype=torch.float32) -> AlgebraContext:
    return AlgebraContext(p, q, r, device=device, dtype=dtype)


def make_algebra_from_config(config: Mapping[str, Any] | AlgebraConfig, **overrides) -> AlgebraContext:
    values = vars(config) if isinstance(config, AlgebraConfig) else config
    return make_algebra(**vars(AlgebraConfig.from_mapping(values, **overrides)))
