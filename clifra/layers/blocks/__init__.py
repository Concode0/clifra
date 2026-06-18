# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Reusable neural network blocks built from clifra primitives."""

from .attention import EntropyGatedAttention, GeometricProductAttention

__all__ = [
    "EntropyGatedAttention",
    "GeometricProductAttention",
]
